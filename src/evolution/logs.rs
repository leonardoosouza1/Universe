use std::fs::File;

// --- Log rotation para race_reports.md ---
// Arquivo cresce indefinidamente se não rotacionarmos. Política: ao abrir,
// se >5MB, renomeia para race_reports.1.md (sobrescrevendo backup anterior)
// e começa um novo. Um único backup é suficiente para debug recente.
const LOG_MAX_BYTES: u64 = 5 * 1024 * 1024;
const LOG_PATH: &str = "race_reports.md";
const LOG_BACKUP: &str = "race_reports.1.md";

pub fn open_report_log() -> Option<File> {
    if let Ok(meta) = std::fs::metadata(LOG_PATH) {
        if meta.len() > LOG_MAX_BYTES {
            let _ = std::fs::rename(LOG_PATH, LOG_BACKUP);
        }
    }
    std::fs::OpenOptions::new().create(true).append(true).open(LOG_PATH).ok()
}
