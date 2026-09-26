//! Bounded forensic checkpoints, independent of selection and skill promotion.
//! Preserve a promising non-champion so its actual task can be replayed later.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io, path::Path};
use universe::gpu::{evolution::Genome, genome_io::save_genome, school::*};

pub fn save(
    out: &Path,
    generation: usize,
    genomes: &[Genome],
    records: &[[f32; SCHOOL_STRIDE_F32]],
    rows: &[Value],
) -> io::Result<()> {
    // At most three role checkpoints plus sixteen successful start checkpoints.
    // A saved partial skill is evidence, never a success or a promoted parent.
    if genomes.len() != records.len()
        || rows.len() != records.len()
        || rows.iter().enumerate().any(|(slot, row)| {
            row["slot"].as_u64() != Some(slot as u64) || row["start_index"].as_u64().is_none()
        })
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "forensic genomes, outcomes and start identities must align",
        ));
    }
    let mut selected: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (role, eligible) in [
        ("best_observed_score", 0),
        ("accepted_pickup", 1),
        ("accepted_delivery", 2),
    ] {
        let candidate = records
            .iter()
            .enumerate()
            .filter(|(_, r)| match eligible {
                1 => r[R_ACCEPTED_PICKUP_TICK] >= 0.0,
                2 => r[R_ACCEPTED_DROPOFF_TICK] >= 0.0 && r[R_DELIVERY_DELTA] > 0.0,
                _ => true,
            })
            .filter(|(_, r)| r[R_SCORE].is_finite())
            .max_by(|(a, x), (b, y)| x[R_SCORE].total_cmp(&y[R_SCORE]).then_with(|| b.cmp(a)));
        if let Some((slot, _)) = candidate {
            selected.entry(slot).or_default().push(role.to_owned());
        }
    }
    // Aggregate selection can discard a rare skill in a difficult direction.
    // Keep one actual clean full trip per world start for forensic replay, without
    // modifying selection, RNG state, competence promotion or the recorded outcome.
    let mut by_start: BTreeMap<u64, usize> = BTreeMap::new();
    for (slot, record) in records.iter().enumerate() {
        if record[R_STATUS] != STATUS_SUCCESS as f32
            || !record[R_SCORE].is_finite()
            || !record[R_ACCEPTED_PICKUP_TICK].is_finite()
            || !record[R_ACCEPTED_DROPOFF_TICK].is_finite()
            || record[R_ACCEPTED_PICKUP_TICK] < 0.0
            || !(record[R_ACCEPTED_DROPOFF_TICK] > record[R_ACCEPTED_PICKUP_TICK])
            || !(record[R_DELIVERY_DELTA] > 0.0)
            || record[R_CONTACT_TICKS] != 0.0
            || record[R_COLLISION_J] != 0.0
        {
            continue;
        }
        let start = rows[slot]["start_index"].as_u64().unwrap();
        let best = by_start.entry(start).or_insert(slot);
        if record[R_SCORE] > records[*best][R_SCORE] {
            *best = slot;
        }
    }
    let omitted_clean_start_indices: Vec<_> = by_start.keys().skip(16).copied().collect();
    for (start, slot) in by_start.iter().take(16) {
        selected
            .entry(*slot)
            .or_default()
            .push(format!("clean_delivery_start:{start}"));
    }
    let directory = out.join("diagnostic_candidates");
    std::fs::create_dir_all(&directory)?;
    let mut candidates = Vec::new();
    for (slot, roles) in selected {
        let file = format!("gen{generation:04}-slot{slot:04}.bin");
        let path = directory.join(&file);
        save_genome(
            &genomes[slot],
            path.to_str()
                .ok_or_else(|| io::Error::other("non-UTF8 diagnostic path"))?,
        )?;
        candidates.push(json!({"slot":slot,"roles":roles,"file":file,
            "genome_sha256":format!("{:x}",Sha256::digest(std::fs::read(&path)?)),
            "observed_outcome":rows[slot]}));
    }
    std::fs::write(
        directory.join(format!("gen{generation:04}.json")),
        serde_json::to_vec_pretty(&json!({
            "schema_version":2,"generation":generation,"candidates":candidates,
            "maximum_checkpoints":19,"omitted_clean_start_indices":omitted_clean_start_indices,
            "scope":"genotype-only forensic checkpoint; rerun with original start and configuration; not acquired state, selection or proof of competence",
            "configuration":"retain original run command, driving.meta.json and generation manifest",
        }))?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // File/identity contract only: no neural or physics evaluation on the host.
    #[test]
    fn preserves_clean_delivery_from_a_lower_scoring_start() {
        let out =
            std::env::temp_dir().join(format!("universe-forensic-starts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let genomes = vec![Genome::random(31), Genome::random(32)];
        let mut records = vec![[0.0; SCHOOL_STRIDE_F32]; 2];
        for (slot, row) in records.iter_mut().enumerate() {
            row[R_STATUS] = STATUS_SUCCESS as f32;
            row[R_SCORE] = 2100.0 - slot as f32 * 100.0;
            row[R_ACCEPTED_PICKUP_TICK] = 10.0;
            row[R_ACCEPTED_DROPOFF_TICK] = 100.0;
            row[R_DELIVERY_DELTA] = 1.0;
        }
        let rows = vec![
            json!({"slot":0,"start_index":0}),
            json!({"slot":1,"start_index":2}),
        ];
        save(&out, 12, &genomes, &records, &rows).unwrap();
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(out.join("diagnostic_candidates/gen0012.json")).unwrap(),
        )
        .unwrap();
        let candidates = manifest["candidates"].as_array().unwrap();
        assert!(
            candidates.iter().any(|c| c["slot"] == 1),
            "the west specialist was lost behind the aggregate champion"
        );
        for candidate in candidates {
            let path = out
                .join("diagnostic_candidates")
                .join(candidate["file"].as_str().unwrap());
            assert_eq!(
                candidate["genome_sha256"],
                format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
            );
        }
        std::fs::remove_dir_all(out).unwrap();
    }
    #[test]
    fn rejects_misaligned_outcomes_before_writing_artifacts() {
        let out =
            std::env::temp_dir().join(format!("universe-forensic-identity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let genomes = vec![Genome::random(41)];
        let records = vec![[0.0; SCHOOL_STRIDE_F32]];
        let rows = vec![json!({"slot":7,"start_index":0})];
        assert!(save(&out, 0, &genomes, &records, &rows).is_err());
        assert!(!out.exists());
    }

    #[test]
    fn reports_start_coverage_truncated_by_checkpoint_budget() {
        let out =
            std::env::temp_dir().join(format!("universe-forensic-budget-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let genomes = vec![Genome::random(51); 24];
        let mut records = vec![[0.0; SCHOOL_STRIDE_F32]; 24];
        for row in &mut records {
            row[R_STATUS] = STATUS_SUCCESS as f32;
            row[R_SCORE] = 2100.0;
            row[R_ACCEPTED_PICKUP_TICK] = 10.0;
            row[R_ACCEPTED_DROPOFF_TICK] = 100.0;
            row[R_DELIVERY_DELTA] = 1.0;
        }
        let rows: Vec<_> = (0..24)
            .map(|slot| json!({"slot":slot,"start_index":slot}))
            .collect();
        save(&out, 0, &genomes, &records, &rows).unwrap();
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(out.join("diagnostic_candidates/gen0000.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            manifest["omitted_clean_start_indices"],
            json!([16, 17, 18, 19, 20, 21, 22, 23])
        );
        assert!(manifest["candidates"].as_array().unwrap().len() <= 19);
        std::fs::remove_dir_all(out).unwrap();
    }
}
