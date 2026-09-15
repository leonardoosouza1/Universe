//! Shared reporting and process-isolation support for the GPU validation binary.
//!
//! Validation must communicate its aggregate verdict to callers. Printing a FAIL is
//! insufficient when the process still exits successfully, and a forked child must
//! return its measured verdict rather than unconditionally calling `_exit(0)`.

use std::sync::atomic::{AtomicBool, Ordering};

static FAILED: AtomicBool = AtomicBool::new(false);

/// Clears the process-wide validation verdict before a run.
pub fn reset() {
    FAILED.store(false, Ordering::Relaxed);
}

/// Returns whether any requested validation row failed or was unavailable.
pub fn failed() -> bool {
    FAILED.load(Ordering::Relaxed)
}

/// Prints one validation row and records a failure in the aggregate verdict.
///
/// Rows marked as an optional drift measurement may be unavailable without failing
/// the run; every other unavailable requested row is a failure.
pub fn print_table_row(name: &str, total: usize, identical: usize, max_diff: i32, ok: bool) {
    if max_diff < 0 {
        println!(
            "{:<38} {:>11} {:>11} {:>10}  NOT MEASURED",
            name, "—", "—", "—"
        );
        if !name.contains("drift") {
            FAILED.store(true, Ordering::Relaxed);
        }
        return;
    }

    let pct = identical as f64 / total.max(1) as f64 * 100.0;
    let verdict = if ok {
        format!("PASS  ({pct:.4}% matched by section criterion, max_dif={max_diff})")
    } else {
        FAILED.store(true, Ordering::Relaxed);
        format!("FAIL  (max_dif={max_diff}, {pct:.4}% matched by section criterion)")
    };
    println!(
        "{:<38} {:>11} {:>11} {:>10}  {}",
        name, total, identical, max_diff, verdict
    );
}

/// Runs a validation in a child process and returns its measured verdict.
#[cfg(unix)]
pub fn run_in_child<F: FnOnce() -> bool>(f: F) -> bool {
    use std::io::Write;

    unsafe {
        let pid = libc::fork();
        if pid < 0 {
            let _ = std::io::stderr()
                .write_all(b"  WARNING: fork() failed; running without crash isolation.\n");
            return f();
        }
        if pid == 0 {
            libc::_exit(if f() { 0 } else { 1 });
        }

        let mut status: libc::c_int = 0;
        if libc::waitpid(pid, &mut status, 0) < 0 {
            return false;
        }
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0
    }
}

/// Runs a validation directly on platforms without `fork`.
#[cfg(not(unix))]
pub fn run_in_child<F: FnOnce() -> bool>(f: F) -> bool {
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn failed_measured_row_sets_aggregate_failure() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        print_table_row("synthetic", 10, 9, 2, false);
        assert!(failed());
    }

    #[test]
    fn optional_unmeasured_drift_does_not_fail_run() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        print_table_row("brain (drift optional)", 0, 0, -1, false);
        assert!(!failed());
    }

    #[test]
    fn child_failed_verdict_is_propagated() {
        let _guard = TEST_LOCK.lock().unwrap();
        assert!(!run_in_child(|| false));
        assert!(run_in_child(|| true));
    }
}
