//! Shared two-phase GPU readback support.
//!
//! A map callback is only serviced by a `Device::poll` that happens after the
//! `map_async` request. Keeping the request and collection separate lets a caller
//! request several independent staging buffers, poll the device once, and then
//! decode every mapped buffer without additional GPU/CPU synchronization points.

use std::sync::mpsc::{self, Receiver};

/// Completion handle for one `MAP_READ` request.
///
/// The handle does not borrow the buffer, so several subsystems can issue requests
/// before the shared device poll. The subsystem that owns the buffer remains
/// responsible for reading and unmapping it after [`ReadbackTicket::is_ready`].
pub struct ReadbackTicket {
    completion: Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl ReadbackTicket {
    /// Request a mapping for `bytes` from the beginning of `buffer`.
    pub fn request(buffer: &wgpu::Buffer, bytes: u64) -> Self {
        let slice = buffer.slice(0..bytes);
        let (sender, completion) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        Self { completion }
    }

    /// Consume the ticket after the caller has polled the device.
    pub fn is_ready(self) -> bool {
        matches!(self.completion.recv(), Ok(Ok(())))
    }
}
