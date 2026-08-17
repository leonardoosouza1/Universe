//! # iara-gate
//!
//! Generic consensus / gating framework extracted from two empirically validated patterns:
//!
//! - **saci-trader** `backtest_v9.py`: consensus gate turned −65% OOS into +82% net-win
//! - **iara-brain** `iara_brain_v5.rs`: anti-poison filter turned 65% into 95% intelligence
//!
//! The crate is pure Rust with no external dependencies.

// ────────────────────────────────────────────────────────────────────────────
// Core trait
// ────────────────────────────────────────────────────────────────────────────

/// A single opinion with an attached confidence score.
#[derive(Debug, Clone)]
pub struct Vote<D> {
    pub decision: D,
    /// Normalised confidence in [0.0, 1.0].
    pub confidence: f32,
}

/// Any component that can cast a typed vote.
pub trait Voter {
    type Decision: PartialEq + Clone;

    fn vote(&self) -> Vote<Self::Decision>;
}

// ────────────────────────────────────────────────────────────────────────────
// Consensus gate
// ────────────────────────────────────────────────────────────────────────────

/// Parameters for the consensus gate.
#[derive(Debug, Clone)]
pub struct ConsensusGate<D: PartialEq + Clone> {
    /// Minimum fraction of voters that must agree on the same decision (e.g. 0.6).
    pub min_agreement: f32,
    /// Minimum average confidence of the agreeing voters (e.g. 0.55).
    pub min_confidence: f32,
    /// A coordinator whose confidence exceeds this threshold can override the gate (e.g. 0.85).
    pub override_threshold: f32,

    _phantom: std::marker::PhantomData<D>,
}

/// Outcome returned by the gate after evaluating a round of votes.
#[derive(Debug, Clone, PartialEq)]
pub enum GateResult<D> {
    /// Sufficient consensus; carries the winning decision and its agreement score.
    Pass(D, f32),
    /// Consensus not reached — caller should do nothing (hold / abstain).
    Wait,
    /// A coordinator with very high (or very low) confidence overrides the vote;
    /// carries its decision and raw confidence.
    Override(D, f32),
}

impl<D: PartialEq + Clone> ConsensusGate<D> {
    pub fn new(min_agreement: f32, min_confidence: f32, override_threshold: f32) -> Self {
        Self {
            min_agreement,
            min_confidence,
            override_threshold,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Evaluate a slice of votes from independent voters.
    ///
    /// Algorithm (mirrors saci-trader `backtest_v9.py`):
    /// 1. Count votes per decision variant.
    /// 2. `agreement = max_count / total`.
    /// 3. If `agreement < min_agreement` → `Wait`.
    /// 4. Compute average confidence of the majority voters.
    /// 5. If `avg_confidence < min_confidence` → `Wait`.
    /// 6. Otherwise → `Pass(majority_decision, agreement)`.
    pub fn evaluate(&self, votes: &[Vote<D>]) -> GateResult<D> {
        if votes.is_empty() {
            return GateResult::Wait;
        }

        let total = votes.len() as f32;

        // Find the decision with the most votes.
        // We iterate through each unique decision linearly (no HashMap needed).
        let majority = majority_decision(votes);
        let (best_decision, best_count) = match majority {
            Some(v) => v,
            None => return GateResult::Wait,
        };

        let agreement = best_count as f32 / total;
        if agreement < self.min_agreement {
            return GateResult::Wait;
        }

        // Average confidence of the voters that chose the majority decision.
        let avg_confidence: f32 = votes
            .iter()
            .filter(|v| v.decision == best_decision)
            .map(|v| v.confidence)
            .sum::<f32>()
            / best_count as f32;

        if avg_confidence < self.min_confidence {
            return GateResult::Wait;
        }

        GateResult::Pass(best_decision, agreement)
    }

    /// Evaluate votes plus a coordinator that may override the gate.
    ///
    /// Override logic:
    /// - `coordinator.confidence > override_threshold` → `Override(coordinator.decision, confidence)`
    /// - Otherwise → fall through to `evaluate(votes)`.
    ///
    /// Note: the original iara-brain also had a "low-confidence anti-override" branch for binary
    /// decisions; that is domain-specific and intentionally left out of this generic API — callers
    /// can implement it on top of `Override` / `Wait` as needed.
    pub fn evaluate_with_coordinator(
        &self,
        votes: &[Vote<D>],
        coordinator: &Vote<D>,
    ) -> GateResult<D> {
        if coordinator.confidence > self.override_threshold {
            return GateResult::Override(coordinator.decision.clone(), coordinator.confidence);
        }
        self.evaluate(votes)
    }
}

/// Returns `Some((decision, count))` for the most common decision, or `None` when votes is empty.
fn majority_decision<D: PartialEq + Clone>(votes: &[Vote<D>]) -> Option<(D, usize)> {
    let mut best: Option<(D, usize)> = None;

    for vote in votes {
        // Find existing tally for this decision.
        let count = votes
            .iter()
            .filter(|v| v.decision == vote.decision)
            .count();

        match &best {
            None => best = Some((vote.decision.clone(), count)),
            Some((_, best_count)) if count > *best_count => {
                best = Some((vote.decision.clone(), count));
            }
            _ => {}
        }
    }

    best
}

// ────────────────────────────────────────────────────────────────────────────
// Anti-poison filter
// ────────────────────────────────────────────────────────────────────────────

/// Detects leaked / poisoned content — specifically English reasoning that leaked into what
/// should be Portuguese responses, unicode-escape fragments, and evasion prefixes.
///
/// Extracted verbatim from `iara_brain_v5.rs`.
#[derive(Debug, Clone)]
pub struct PoisonFilter {
    /// Full phrases that betray leaked English reasoning chains.
    pub markers: Vec<String>,
    /// Unicode escape fragments (e.g. `"u00e7"` for ç).
    pub unicode_patterns: Vec<String>,
    /// Portuguese meta-prefixes that signal the model is narrating its own task.
    pub meta_prefixes: Vec<String>,
    /// Common English function / reasoning words used for density check.
    pub en_reasoning_words: Vec<String>,
    /// How many EN density words must appear before we flag the text (default 2).
    pub min_en_hits: usize,
}

impl PoisonFilter {
    /// Default markers tuned for PT-BR / EN detection (from `iara_brain_v5.rs`).
    pub fn default_ptbr() -> Self {
        Self {
            markers: vec![
                "okay, the user".into(),
                "let me think".into(),
                "i need to explain".into(),
                "so the answer".into(),
                "the question asks".into(),
                "let me break".into(),
                "step by step".into(),
                "in this case".into(),
            ],
            unicode_patterns: vec![
                "u00e7".into(), // ç
                "u00e3".into(), // ã
                "u00ed".into(), // í
                "u00f3".into(), // ó
                "u00fa".into(), // ú
            ],
            meta_prefixes: vec![
                "a pergunta ".into(),
                "explicação:".into(),
                "explicacao:".into(),
                "parece ter um erro".into(),
                "não está clara".into(),
                "nao esta clara".into(),
            ],
            en_reasoning_words: vec![
                "the".into(),
                "this".into(),
                "that".into(),
                "which".into(),
                "would".into(),
                "should".into(),
                "could".into(),
                "because".into(),
            ],
            min_en_hits: 2,
        }
    }

    /// Returns `true` if `text` looks like leaked or poisoned content.
    ///
    /// A text is flagged if **any** of the following is true:
    /// - It contains one of the full `markers` (case-insensitive).
    /// - It contains one of the `unicode_patterns`.
    /// - It starts with one of the `meta_prefixes` (case-insensitive).
    /// - It contains at least `min_en_hits` distinct EN density words.
    pub fn is_poisoned(&self, text: &str) -> bool {
        let lower = text.to_lowercase();

        // 1. Full reasoning markers.
        for marker in &self.markers {
            if lower.contains(marker.as_str()) {
                return true;
            }
        }

        // 2. Unicode escape fragments.
        for pattern in &self.unicode_patterns {
            if lower.contains(pattern.as_str()) {
                return true;
            }
        }

        // 3. Meta-prefixes (checked against the start of the lowercased text).
        for prefix in &self.meta_prefixes {
            if lower.starts_with(prefix.as_str()) || lower.contains(prefix.as_str()) {
                return true;
            }
        }

        // 4. EN density check — count how many distinct EN words appear.
        let hits = self
            .en_reasoning_words
            .iter()
            .filter(|word| {
                // Match whole words with simple boundary check.
                word_boundary_contains(&lower, word)
            })
            .count();

        hits >= self.min_en_hits
    }

    /// Filter a collection, keeping only items whose associated text is **not** poisoned.
    pub fn filter<'a, T>(&self, items: &'a [T], text_fn: impl Fn(&T) -> &str) -> Vec<&'a T> {
        items
            .iter()
            .filter(|item| !self.is_poisoned(text_fn(item)))
            .collect()
    }
}

/// Returns `true` if `haystack` contains `word` as a whole word (ASCII word boundary).
fn word_boundary_contains(haystack: &str, word: &str) -> bool {
    let bytes = haystack.as_bytes();
    let word_bytes = word.as_bytes();
    let wlen = word_bytes.len();

    if wlen == 0 || wlen > bytes.len() {
        return false;
    }

    'outer: for start in 0..=(bytes.len() - wlen) {
        if &bytes[start..start + wlen] != word_bytes {
            continue;
        }
        // Check left boundary.
        if start > 0 && bytes[start - 1].is_ascii_alphanumeric() {
            continue 'outer;
        }
        // Check right boundary.
        let end = start + wlen;
        if end < bytes.len() && bytes[end].is_ascii_alphanumeric() {
            continue 'outer;
        }
        return true;
    }
    false
}

// ────────────────────────────────────────────────────────────────────────────
// Swarm consensus functions
// ────────────────────────────────────────────────────────────────────────────

/// Weighted geometric mean: `exp(Σ wᵢ·ln(sᵢ) / Σ wᵢ)`.
///
/// Scores must be positive. Zero or negative scores are skipped.
/// Returns 0.0 if no valid pairs are present.
pub fn geometric_consensus(scores: &[(f32, f32)]) -> f32 {
    let mut weighted_log_sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for &(score, weight) in scores {
        if score > 0.0 && weight > 0.0 {
            weighted_log_sum += weight * score.ln();
            weight_sum += weight;
        }
    }

    if weight_sum == 0.0 {
        return 0.0;
    }

    (weighted_log_sum / weight_sum).exp()
}

/// Tropical variant: `0.85 · geometric + 0.15 · max`.
///
/// Blends the geometric mean with the maximum score, giving a slight boost to
/// the highest-confidence voter without letting it dominate.
pub fn tropical_consensus(scores: &[(f32, f32)]) -> f32 {
    let geo = geometric_consensus(scores);

    let max_score = scores
        .iter()
        .filter(|&&(s, w)| s > 0.0 && w > 0.0)
        .map(|&(s, _)| s)
        .fold(f32::NEG_INFINITY, f32::max);

    if max_score == f32::NEG_INFINITY {
        return 0.0;
    }

    0.85 * geo + 0.15 * max_score
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────────

    fn vote(decision: &str, confidence: f32) -> Vote<String> {
        Vote {
            decision: decision.to_string(),
            confidence,
        }
    }

    fn gate(min_agreement: f32, min_confidence: f32) -> ConsensusGate<String> {
        ConsensusGate::new(min_agreement, min_confidence, 0.85)
    }

    // ── ConsensusGate ─────────────────────────────────────────────────────

    #[test]
    fn all_five_voters_agree_gives_pass() {
        let g = gate(0.6, 0.5);
        let votes = vec![
            vote("buy", 0.8),
            vote("buy", 0.7),
            vote("buy", 0.9),
            vote("buy", 0.75),
            vote("buy", 0.85),
        ];
        let result = g.evaluate(&votes);
        assert!(
            matches!(result, GateResult::Pass(ref d, score) if d == "buy" && (score - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn split_3_2_passes_when_threshold_is_low() {
        let g = gate(0.5, 0.5);
        let votes = vec![
            vote("buy", 0.8),
            vote("buy", 0.7),
            vote("buy", 0.9),
            vote("sell", 0.6),
            vote("sell", 0.65),
        ];
        // agreement = 3/5 = 0.6 ≥ 0.5 → should pass
        assert!(matches!(g.evaluate(&votes), GateResult::Pass(ref d, _) if d == "buy"));
    }

    #[test]
    fn split_3_2_waits_when_threshold_is_high() {
        let g = gate(0.7, 0.5);
        let votes = vec![
            vote("buy", 0.8),
            vote("buy", 0.7),
            vote("buy", 0.9),
            vote("sell", 0.6),
            vote("sell", 0.65),
        ];
        // agreement = 3/5 = 0.6 < 0.7 → Wait
        assert_eq!(g.evaluate(&votes), GateResult::Wait);
    }

    #[test]
    fn low_majority_confidence_gives_wait() {
        // 5/5 agree but confidence is very low
        let g = gate(0.6, 0.75);
        let votes = vec![
            vote("hold", 0.3),
            vote("hold", 0.4),
            vote("hold", 0.35),
            vote("hold", 0.45),
            vote("hold", 0.2),
        ];
        assert_eq!(g.evaluate(&votes), GateResult::Wait);
    }

    #[test]
    fn empty_votes_gives_wait() {
        let g = gate(0.6, 0.55);
        assert_eq!(g.evaluate(&[]), GateResult::Wait);
    }

    #[test]
    fn coordinator_override_when_confidence_high() {
        let g = gate(0.6, 0.55);
        let votes = vec![
            vote("sell", 0.7),
            vote("sell", 0.8),
            vote("buy", 0.6),
            vote("buy", 0.65),
            vote("buy", 0.62),
        ];
        let coordinator = Vote {
            decision: "sell".to_string(),
            confidence: 0.92, // > 0.85 override_threshold
        };
        let result = g.evaluate_with_coordinator(&votes, &coordinator);
        assert!(
            matches!(result, GateResult::Override(ref d, c) if d == "sell" && (c - 0.92).abs() < 1e-6)
        );
    }

    #[test]
    fn coordinator_below_threshold_falls_through_to_evaluate() {
        let g = gate(0.6, 0.55);
        let votes = vec![
            vote("buy", 0.8),
            vote("buy", 0.75),
            vote("buy", 0.85),
            vote("sell", 0.6),
            vote("sell", 0.7),
        ];
        let coordinator = Vote {
            decision: "sell".to_string(),
            confidence: 0.70, // ≤ 0.85 → no override
        };
        // Majority is "buy" (3/5 = 0.6 ≥ threshold), so evaluate returns Pass
        let result = g.evaluate_with_coordinator(&votes, &coordinator);
        assert!(matches!(result, GateResult::Pass(ref d, _) if d == "buy"));
    }

    // ── PoisonFilter ──────────────────────────────────────────────────────

    #[test]
    fn detects_english_reasoning_leak() {
        let f = PoisonFilter::default_ptbr();
        assert!(f.is_poisoned("Okay, the user wants to know about this topic."));
        assert!(f.is_poisoned("Let me think about this carefully step by step."));
        assert!(f.is_poisoned("So the answer would be 42."));
    }

    #[test]
    fn detects_unicode_escape_fragments() {
        let f = PoisonFilter::default_ptbr();
        // Escaped ç and ã
        assert!(f.is_poisoned("A palavra Fran\\u00e7a tem cedilha"));
        assert!(f.is_poisoned("Ele n\\u00e3o foi"));
    }

    #[test]
    fn detects_meta_prefixes() {
        let f = PoisonFilter::default_ptbr();
        assert!(f.is_poisoned("a pergunta não está clara"));
        assert!(f.is_poisoned("explicação: o modelo responde assim"));
    }

    #[test]
    fn detects_evasion_patterns() {
        let f = PoisonFilter::default_ptbr();
        assert!(f.is_poisoned("Parece ter um erro na formulação"));
        assert!(f.is_poisoned("Não está clara a questão colocada"));
    }

    #[test]
    fn clean_ptbr_text_passes() {
        let f = PoisonFilter::default_ptbr();
        let clean = "A capital do Brasil é Brasília, fundada em 1960.";
        assert!(!f.is_poisoned(clean));
    }

    #[test]
    fn filter_keeps_only_clean_items() {
        let f = PoisonFilter::default_ptbr();
        let items = vec![
            ("Brasília foi inaugurada em 1960.", true),   // clean
            ("Let me think about this question.", false), // poisoned
            ("O Rio Amazonas é o maior do mundo.", true), // clean
            ("Step by step the answer is 42.", false),   // poisoned
        ];
        let kept: Vec<_> = f.filter(&items, |(text, _)| text);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().all(|(_, expected)| *expected));
    }

    // ── Swarm consensus ───────────────────────────────────────────────────

    #[test]
    fn geometric_consensus_equal_scores_returns_that_score() {
        let scores = vec![(0.8, 1.0), (0.8, 1.0), (0.8, 1.0)];
        let result = geometric_consensus(&scores);
        assert!((result - 0.8).abs() < 1e-5, "expected ~0.8, got {result}");
    }

    #[test]
    fn geometric_consensus_weighted() {
        // exp(2*ln(0.9) + 1*ln(0.6)) / 3 ≈ 0.795
        let scores = vec![(0.9, 2.0), (0.6, 1.0)];
        let result = geometric_consensus(&scores);
        let expected = ((2.0f32 * 0.9f32.ln() + 1.0f32 * 0.6f32.ln()) / 3.0f32).exp();
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn geometric_consensus_empty_returns_zero() {
        assert_eq!(geometric_consensus(&[]), 0.0);
    }

    #[test]
    fn tropical_consensus_greater_than_geometric_when_max_differs() {
        // Mix of scores with a high outlier
        let scores = vec![(0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.95, 1.0)];
        let geo = geometric_consensus(&scores);
        let trop = tropical_consensus(&scores);
        // tropical blends in 0.15 * max (0.95) on top of geo, so it must be larger
        assert!(
            trop > geo,
            "expected tropical ({trop}) > geometric ({geo})"
        );
    }

    #[test]
    fn tropical_consensus_equal_scores_same_as_geometric() {
        let scores = vec![(0.7, 1.0), (0.7, 1.0), (0.7, 1.0)];
        let geo = geometric_consensus(&scores);
        let trop = tropical_consensus(&scores);
        // When all scores equal, geo = max, so tropical = 0.85*geo + 0.15*geo = geo
        assert!((trop - geo).abs() < 1e-5);
    }
}
