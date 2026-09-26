//! Records the exact source fingerprint and git state embedded in every binary.

use sha2::{Digest, Sha256};
use std::{
    env,
    error::Error,
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
};

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    Some(text.trim().to_string())
}

fn collect_files(
    directory: &Path,
    files: &mut Vec<PathBuf>,
    directories: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    directories.push(directory.to_path_buf());
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files(&entry.path(), files, directories)?;
        } else if file_type.is_file() {
            files.push(entry.path());
        } else {
            return Err(format!(
                "source fingerprint does not support {}",
                entry.path().display()
            )
            .into());
        }
    }
    Ok(())
}

fn relative_utf8(root: &Path, path: &Path) -> Result<String, Box<dyn Error>> {
    let relative = path.strip_prefix(root)?;
    let mut components = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(value) => components.push(
                value
                    .to_str()
                    .ok_or_else(|| format!("source path is not UTF-8: {}", path.display()))?,
            ),
            _ => return Err(format!("source path is not canonical: {}", path.display()).into()),
        }
    }
    Ok(components.join("/"))
}

fn source_sha256(root: &Path) -> Result<String, Box<dyn Error>> {
    let mut files = vec![
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
        root.join("build.rs"),
    ];
    let mut directories = Vec::new();
    collect_files(&root.join("src"), &mut files, &mut directories)?;

    let mut files = files
        .into_iter()
        .map(|path| -> Result<_, Box<dyn Error>> { Ok((relative_utf8(root, &path)?, path)) })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    files.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));

    for directory in directories {
        println!(
            "cargo:rerun-if-changed={}",
            relative_utf8(root, &directory)?
        );
    }

    // Canonical source fingerprint v1: for every regular file, sorted by its
    // slash-separated relative UTF-8 path bytes, hash u64-LE(path length), path
    // bytes, then the 32 raw bytes of SHA-256(file contents). Only src/**,
    // Cargo.toml, Cargo.lock and build.rs participate.
    let mut source = Sha256::new();
    for (relative, path) in files {
        println!("cargo:rerun-if-changed={relative}");
        let contents = fs::read(&path)?;
        source.update((relative.len() as u64).to_le_bytes());
        source.update(relative.as_bytes());
        source.update(Sha256::digest(contents));
    }
    Ok(format!("{:x}", source.finalize()))
}

fn main() -> Result<(), Box<dyn Error>> {
    let root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("missing CARGO_MANIFEST_DIR")?);
    let source_sha256 = source_sha256(&root)?;
    println!("cargo:rustc-env=UNIVERSE_SOURCE_SHA256={source_sha256}");

    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| if s.is_empty() { "0" } else { "1" })
        .unwrap_or("unknown");
    println!("cargo:rustc-env=UNIVERSE_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=UNIVERSE_GIT_DIRTY={dirty}");
    // Re-run when HEAD moves or the index changes; a dirty tree is only re-detected on
    // the next build that cargo triggers anyway.
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Some(head) = git(&["symbolic-ref", "-q", "HEAD"]) {
        println!("cargo:rerun-if-changed=.git/{head}");
    }
    println!("cargo:rerun-if-changed=.git/index");
    Ok(())
}
