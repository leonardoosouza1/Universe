//! Records the git commit the binary was built from, so every run manifest can name it
//! (`provenance.git_commit`). Seven of eleven school batteries of 2026-09-22/23 had no
//! `commit.txt` and cannot be rebuilt (docs/GAP-SWEEP-20260924.md P1-10). Falls back to
//! "unknown" outside a git checkout; never fails the build.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    Some(text.trim().to_string())
}

fn main() {
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
}
