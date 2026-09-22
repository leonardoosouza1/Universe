//! Bounded, versioned snapshots of resident GPU state.
//!
//! Capture copies every named buffer into one staging allocation and maps it
//! once. Restore validates every name, dimension, version and hash before the
//! first queue write. A snapshot is a runtime-state checkpoint; it is not a DNA
//! export and does not claim that a life can resume from genes alone.

use std::path::Path;

use sha2::{Digest, Sha256};

const MAGIC: &[u8; 16] = b"NMDGPU-SNAPSHOT\0";
pub const FORMAT_VERSION: u32 = 1;
const MAX_BUFFERS: usize = 64;
const MAX_NAME: usize = 128;
const MAX_BUFFER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 1024 * 1024 * 1024;

pub type NamedBuffer<'a> = (&'static str, &'a wgpu::Buffer);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotBuffer {
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotBundle {
    pub version: u32,
    pub buffers: Vec<SnapshotBuffer>,
}

pub struct PendingCapture {
    staging: wgpu::Buffer,
    entries: Vec<(&'static str, u64, u64)>,
}

impl PendingCapture {
    /// Map the one staging range after the command buffer containing the copy
    /// has completed. The caller must submit/poll before calling this method.
    pub fn read(self, device: &wgpu::Device) -> Result<SnapshotBundle, String> {
        let total = self.entries.last().map(|(_, _, end)| *end).unwrap_or(0);
        let slice = self.staging.slice(0..total);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        if !matches!(receiver.recv(), Ok(Ok(()))) {
            return Err("GPU snapshot mapping failed".into());
        }
        let mapped = slice.get_mapped_range();
        let mut buffers = Vec::with_capacity(self.entries.len());
        for (name, start, end) in &self.entries {
            buffers.push(SnapshotBuffer {
                name: (*name).into(),
                bytes: mapped[*start as usize..*end as usize].to_vec(),
            });
        }
        drop(mapped);
        self.staging.unmap();
        let bundle = SnapshotBundle {
            version: FORMAT_VERSION,
            buffers,
        };
        validate_bundle(&bundle)?;
        Ok(bundle)
    }
}

/// Add all named copies to one command encoder and return a single readback.
pub fn encode_capture(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    buffers: &[NamedBuffer<'_>],
) -> Result<PendingCapture, String> {
    validate_names_and_sizes(
        buffers
            .iter()
            .map(|(name, buffer)| (*name, buffer.size()))
            .collect(),
    )?;
    let total: u64 = buffers.iter().map(|(_, buffer)| buffer.size()).sum();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-state-snapshot-staging"),
        size: total,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut offset = 0;
    let mut entries = Vec::with_capacity(buffers.len());
    for (name, buffer) in buffers {
        let end = offset + buffer.size();
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, offset, buffer.size());
        entries.push((*name, offset, end));
        offset = end;
    }
    Ok(PendingCapture { staging, entries })
}

pub fn restore(
    queue: &wgpu::Queue,
    buffers: &[NamedBuffer<'_>],
    bundle: &SnapshotBundle,
) -> Result<(), String> {
    validate_names_and_sizes(
        buffers
            .iter()
            .map(|(name, buffer)| (*name, buffer.size()))
            .collect(),
    )?;
    validate_bundle(bundle)?;
    if buffers.len() != bundle.buffers.len() {
        return Err("snapshot buffer count does not match runtime".into());
    }
    for ((name, buffer), saved) in buffers.iter().zip(&bundle.buffers) {
        if *name != saved.name || buffer.size() != saved.bytes.len() as u64 {
            return Err(format!("snapshot dimension/name mismatch for {name}"));
        }
    }
    // All checks, including every hash, have completed before any write.
    for ((_, buffer), saved) in buffers.iter().zip(&bundle.buffers) {
        queue.write_buffer(buffer, 0, &saved.bytes);
    }
    Ok(())
}

pub fn write_atomic(path: &Path, bundle: &SnapshotBundle) -> Result<(), String> {
    validate_bundle(bundle)?;
    let bytes = encode(bundle)?;
    let parent = path
        .parent()
        .ok_or_else(|| "snapshot path has no parent".to_string())?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let temp_path = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("snapshot"),
        std::process::id()
    ));
    let mut temp = std::fs::File::create(&temp_path).map_err(|e| e.to_string())?;
    use std::io::Write;
    temp.write_all(&bytes).map_err(|e| e.to_string())?;
    temp.sync_all().map_err(|e| e.to_string())?;
    std::fs::rename(&temp_path, path).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn read_file(path: &Path) -> Result<SnapshotBundle, String> {
    const MAX_ENCODED_SNAPSHOT: u64 = 1024 * 1024 * 1024 + 64 * 1024;
    let size = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    if size > MAX_ENCODED_SNAPSHOT {
        return Err("snapshot file exceeds bounded size".into());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    decode(&bytes)
}

fn validate_names_and_sizes(items: Vec<(&str, u64)>) -> Result<(), String> {
    if items.is_empty() {
        return Err("snapshot buffer bundle is empty".into());
    }
    if items.len() > MAX_BUFFERS {
        return Err("too many snapshot buffers".into());
    }
    let mut total: u64 = 0;
    for (name, size) in items {
        if name.is_empty() || name.len() > MAX_NAME || !name.is_ascii() {
            return Err("invalid snapshot buffer name".into());
        }
        if size == 0 || size > MAX_BUFFER_BYTES {
            return Err(format!("invalid snapshot size for {name}"));
        }
        total = total
            .checked_add(size)
            .ok_or_else(|| "snapshot size overflow".to_string())?;
    }
    if total > MAX_TOTAL_BYTES {
        return Err("snapshot exceeds total size bound".into());
    }
    Ok(())
}

fn validate_bundle(bundle: &SnapshotBundle) -> Result<(), String> {
    if bundle.version != FORMAT_VERSION {
        return Err("unsupported snapshot version".into());
    }
    validate_names_and_sizes(
        bundle
            .buffers
            .iter()
            .map(|b| (b.name.as_str(), b.bytes.len() as u64))
            .collect(),
    )?;
    for buffer in &bundle.buffers {
        if sha256(&buffer.bytes).is_empty() {
            return Err("unreachable hash failure".into());
        }
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest.finalize().into()
}

fn encode(bundle: &SnapshotBundle) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&bundle.version.to_le_bytes());
    out.extend_from_slice(&(bundle.buffers.len() as u32).to_le_bytes());
    for buffer in &bundle.buffers {
        let name = buffer.name.as_bytes();
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&(buffer.bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&sha256(&buffer.bytes));
        out.extend_from_slice(&buffer.bytes);
    }
    Ok(out)
}

fn decode(bytes: &[u8]) -> Result<SnapshotBundle, String> {
    let mut cursor = 0usize;
    let take = |cursor: &mut usize, count: usize| -> Result<&[u8], String> {
        let end = cursor
            .checked_add(count)
            .ok_or_else(|| "snapshot offset overflow".to_string())?;
        if end > bytes.len() {
            return Err("truncated snapshot".into());
        }
        let part = &bytes[*cursor..end];
        *cursor = end;
        Ok(part)
    };
    if take(&mut cursor, 16)? != MAGIC {
        return Err("invalid snapshot magic".into());
    }
    let version = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap());
    let count = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap()) as usize;
    if count > MAX_BUFFERS {
        return Err("too many snapshot buffers".into());
    }
    let mut buffers = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = u16::from_le_bytes(take(&mut cursor, 2)?.try_into().unwrap()) as usize;
        let name = String::from_utf8(take(&mut cursor, name_len)?.to_vec())
            .map_err(|_| "invalid snapshot name".to_string())?;
        let size = u64::from_le_bytes(take(&mut cursor, 8)?.try_into().unwrap());
        if size == 0 || size > MAX_BUFFER_BYTES || size > usize::MAX as u64 {
            return Err("invalid snapshot payload size".into());
        }
        let expected: [u8; 32] = take(&mut cursor, 32)?.try_into().unwrap();
        let payload = take(&mut cursor, size as usize)?.to_vec();
        if sha256(&payload) != expected {
            return Err(format!("snapshot hash mismatch for {name}"));
        }
        buffers.push(SnapshotBuffer {
            name,
            bytes: payload,
        });
    }
    if cursor != bytes.len() {
        return Err("trailing snapshot bytes".into());
    }
    let bundle = SnapshotBundle { version, buffers };
    validate_bundle(&bundle)?;
    Ok(bundle)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn format_roundtrip_and_hash_rejection() {
        let bundle = SnapshotBundle {
            version: FORMAT_VERSION,
            buffers: vec![SnapshotBuffer {
                name: "state".into(),
                bytes: vec![1, 2, 3, 4],
            }],
        };
        let encoded = encode(&bundle).unwrap();
        assert_eq!(decode(&encoded).unwrap(), bundle);
        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(decode(&corrupt).is_err());
    }
    #[test]
    fn bounds_reject_empty_and_oversized_buffers() {
        assert!(validate_bundle(&SnapshotBundle {
            version: FORMAT_VERSION,
            buffers: vec![]
        })
        .is_err());
        assert!(validate_names_and_sizes(vec![("x", MAX_BUFFER_BYTES + 1)]).is_err());
    }
}
