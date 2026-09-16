use super::{
    CFC_SLOTS_PER_BEING, FAST_CFC_STATE_BYTES, L1_NEURONS, brain_control_word,
    fast_state_copy_range,
};

#[test]
fn packed_control_keeps_ablation_and_observation_independent() {
    assert_eq!(brain_control_word(false, None), 0);
    assert_eq!(brain_control_word(true, None), 1);
    assert_eq!(brain_control_word(false, Some(0)), 2);
    assert_eq!(brain_control_word(true, Some(7)), 17);
}

#[test]
fn fast_state_layout_excludes_counter_and_uses_97_slot_stride() {
    assert_eq!(L1_NEURONS, 96);
    assert_eq!(CFC_SLOTS_PER_BEING, 97);
    assert_eq!(FAST_CFC_STATE_BYTES, 96 * 4);
    assert_eq!(
        fast_state_copy_range(3, 0, FAST_CFC_STATE_BYTES, 0).unwrap(),
        (0, 384)
    );
    assert_eq!(
        fast_state_copy_range(3, 1, FAST_CFC_STATE_BYTES, 0).unwrap(),
        (97 * 4, 384)
    );
    assert_eq!(
        fast_state_copy_range(3, 2, 768, 384).unwrap(),
        (2 * 97 * 4, 384)
    );
}

#[test]
fn fast_state_copy_rejects_out_of_bounds_requests() {
    assert!(fast_state_copy_range(2, 2, 384, 0).is_err());
    assert!(fast_state_copy_range(2, 0, 383, 0).is_err());
    assert!(fast_state_copy_range(2, 0, 384, 2).is_err());
    assert!(fast_state_copy_range(2, 0, 384, u64::MAX - 3).is_err());
}
