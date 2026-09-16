use universe::gpu::brain::{ABLATION_WORDS_PER_BEING, AblationPlan};

#[test]
fn plan_encodes_each_stage_for_only_the_selected_being() {
    let plan = AblationPlan::parse("cfc:95,l1:17,l1:80,l2:47", 1, 6000, 3).unwrap();
    let words = plan.mask_words(3);

    assert_eq!(words.len(), 3 * ABLATION_WORDS_PER_BEING);
    assert!(
        words[..ABLATION_WORDS_PER_BEING]
            .iter()
            .all(|word| *word == 0)
    );
    assert!(
        words[2 * ABLATION_WORDS_PER_BEING..]
            .iter()
            .all(|word| *word == 0)
    );

    let selected = &words[ABLATION_WORDS_PER_BEING..2 * ABLATION_WORDS_PER_BEING];
    assert_eq!(selected[2], 1 << 31); // cfc:95
    assert_eq!(selected[3], 1 << 17); // l1:17
    assert_eq!(selected[5], 1 << 16); // l1:80
    assert_eq!(selected[7], 1 << 15); // l2:47
    assert_eq!(plan.from_tick(), 6000);
    assert_eq!(plan.being(), 1);
}

#[test]
fn plan_rejects_invalid_stage_index_and_being() {
    assert!(AblationPlan::parse("hidden:0", 0, 1, 1).is_err());
    assert!(AblationPlan::parse("cfc:96", 0, 1, 1).is_err());
    assert!(AblationPlan::parse("l1:96", 0, 1, 1).is_err());
    assert!(AblationPlan::parse("l2:48", 0, 1, 1).is_err());
    assert!(AblationPlan::parse("l2:0", 1, 1, 1).is_err());
    assert!(AblationPlan::parse("", 0, 1, 1).is_err());
}

#[test]
fn plan_reports_the_first_dispatch_at_or_after_the_requested_tick() {
    let plan = AblationPlan::parse("l2:2", 0, 6000, 1).unwrap();

    assert!(!plan.active_at(5999));
    assert!(plan.active_at(6000));
    assert!(plan.active_at(6001));
}
