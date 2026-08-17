use std::fs::File;

// --- Log rotation for race_reports.md ---
// The file grows indefinitely if we do not rotate it. Policy: on open,
// if >5MB, rename to race_reports.1.md (overwriting the previous backup)
// and start a new one. A single backup is enough for recent debugging.
const LOG_MAX_BYTES: u64 = 5 * 1024 * 1024;
const LOG_PATH: &str = "race_reports.md";
const LOG_BACKUP: &str = "race_reports.1.md";

pub fn open_report_log() -> Option<File> {
    if let Ok(meta) = std::fs::metadata(LOG_PATH) {
        if meta.len() > LOG_MAX_BYTES {
            let _ = std::fs::rename(LOG_PATH, LOG_BACKUP);
        }
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_PATH)
        .ok()
}
