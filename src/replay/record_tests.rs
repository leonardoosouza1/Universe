use super::*;

#[test]
fn frame_offsets_preserve_the_existing_binary_contract() {
    assert_eq!(std::mem::size_of::<ReplayFrame>(), 128);
    assert_eq!(std::mem::offset_of!(ReplayFrame, fuel_norm), 36);
    assert_eq!(std::mem::offset_of!(ReplayFrame, _alignment), 37);
    assert_eq!(std::mem::offset_of!(ReplayFrame, tick), 40);
    assert_eq!(std::mem::offset_of!(ReplayFrame, final_thr), 88);
}

#[test]
fn truncated_or_out_of_population_records_are_rejected() {
    let path = std::env::temp_dir().join(format!(
        "universe-replay-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut writer = ReplayWriter::create(&path, SOURCE_CORTEX, 1, 64.0, 8005).unwrap();
    writer
        .write_frame(
            0,
            &ReplayFrame {
                tick: 4,
                ..Default::default()
            },
        )
        .unwrap();
    writer.flush().unwrap();
    drop(writer);
    assert!(ReplayReader::open(&path).is_ok());
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[65] = 1;
    std::fs::write(&path, &bytes).unwrap();
    assert!(ReplayReader::open(&path).is_err());
    bytes[65] = 0;
    bytes.pop();
    std::fs::write(&path, &bytes).unwrap();
    assert!(ReplayReader::open(&path).is_err());
    std::fs::remove_file(path).unwrap();
}
