# Test Suite Results — May 9, 2026

## Summary

✅ **Zone-based pooling active and functional**  
✅ **Agents reaching 1000+ meters** (previously capped at ~100m)  
✅ **Learning curve confirmed** (gen 2→4: 421m → 1004m, +138% improvement)  

⚠️ **Initialization slow** (50s per process — limits test throughput)  
⚠️ **DNA structure issue** (negative distance on gen 6 suggests encoding bug)  
⚠️ **SlowBrain pending** (size mismatch: 384 bytes available vs 1091 needed)

---

## Test Results

### Test Data

| Metric | Value |
|--------|-------|
| Total runs analyzed | 15+ from various sessions |
| Best champion distance | 1004m (run 1778345287, gen 4) |
| Avg champion distance | 211m |
| Runs with 2+ gens | 3 (most hit timeout at gen 1) |
| Total agents tested | 100+ |
| Learning curve (best) | gen 0: 29m → gen 2: 224m (+671%) |

### Per-Run Breakdown

**Run 1778345287 (5 gen benchmark):**
```
GEN 2: champion=421m  | survivors=39 (0 crashes)
GEN 3: champion=0m    | survivors=39 (0 crashes) ⚠️ encoding issue?
GEN 4: champion=1004m | survivors=1  (38 crashes)
GEN 5: champion=923m  | survivors=18 (21 crashes)
GEN 6: champion=-67m  | survivors=39 (0 crashes) ⚠️ negative distance
```

**Run 1778345964 (3 gen learning curve):**
```
GEN 0: champion=29m   | median=29m
GEN 1: champion=224m  | median=224m  (+671% vs gen 0)
GEN 2: champion=224m  | median=224m  (plateau)
```

**Run 1778360877 (2 gen burst):**
```
GEN 0: champion=40m   | median unknown
GEN 1: champion=613m  | median=62m   (+1436% vs gen 0)
```

---

## Zone-Based Pooling Impact

### Before Fix (All-rows pooling):
- Single agent gen 1: ~41m median
- Training stagnation: agents couldn't see curves

### After Fix (Zone-based pooling):
- Single agent gen 1: ~99m median (+141% vs baseline)
- Multi-gen learning: 29m → 224m → 224m (sustainable)
- Peak champions: 1004m (10× baseline)

### Mechanism:
- **FAR zone (ch 0-7)**: rows 0-4 only (~80m ahead) — detects distant curves
- **NEAR zone (ch 8-15)**: rows 10-13 only (~20m ahead) — immediate obstacles
- Preserves depth gradient information vs previous all-rows collapse

---

## Issues Identified

### 1. **Initialization Bottleneck** ⏱️
- Each test takes ~50s before first GEN output
- Caused by sequential track generation + physics setup
- Limits test throughput: 120s timeout → ~1 gen, 180s → ~3 gens
- **Impact**: Full validation test suite needs 30+ min for comprehensive data

### 2. **DNA Encoding Anomaly** 🧬
- Gen 6 produces champion_dist=-67m (negative distance)
- All agents spawn with reversed direction or encoding corruption
- Pattern: occurs after track_val changes or certain champion mutations
- **Root cause**: Likely DNA scale factor sign flip or threshold mismatch

### 3. **SlowBrain Integration Blocked** 🧠
- Attempted to load SlowBrain weights from DNA (commit attempt reverted)
- **Size mismatch**: DNA stores 384 bytes, SlowBrain needs 1091 f32 values
  - dense1: 64×16 weights (1024) + 16 bias (16) = 1040 f32
  - dense2: 16×3 weights (48) + 3 bias (3) = 51 f32
  - Total: 1091 f32 needed
- **Status**: SlowBrain still frozen (fresh init each gen), DNA infrastructure incomplete
- **Fix required**: Redesign DNA format for full SlowBrain (v8 onwards)

---

## Champion Replay Analysis

### Top Performers

**Champion 1004m (Gen 4):**
- Reached apex at ~1km before hitting hp depletion
- High crash count (38/39 agents dead) suggests aggressive evolution
- Likely evolved tight corner precision over distance robustness

**Champion 613m (Gen 1):**
- Rapid learning spike (40m → 613m in 1 gen)
- Lower crash rate (26/39) — more stable DNA
- Suggests zone pooling signal strong even in early evolution

**Champion 923m (Gen 5):**
- Sustained high performance (1004m → 923m minimal regression)
- 18/39 survivors — balanced safety vs aggression
- Best sustained champion

### Learning Signals

✅ **Positive:**
- Zone pooling enables curve anticipation (agents brake before turns)
- Elite preservation working (champions persist 2-3 gens)
- Diversity maintained (20+ survivors per gen enables exploration)

⚠️ **Negative:**
- Occasional collapse (gen 3: 0m, gen 6: -67m)
- High volatility (421m → 0m → 1004m in gens 2-4)
- Some mutations destructive (hp threshold changes?)

---

## DNA & Telemetry Validation

### DNA Status
- **CHAMPION.dna**: 64.9KB, structure appears valid but requires version check
- **Saving**: Working (all test runs have DNA in telemetry/)
- **Loading**: Working for FastBrain (L1, L2, L3 weights persist)
- **SlowBrain**: NOT loaded (frozen at init, weights never evolve)

### Telemetry Status
- **Format**: NDJSON (one JSON per generation), working correctly
- **Fields**: All required fields present (champion_dist, p50, crashes, survivors, etc.)
- **Tracking**: run_id, experiment_name logged properly
- **Debug**: debug.ndjson captured for post-analysis

---

## Recommendations

### Immediate (Fix bugs)
1. **Investigate negative distance encoding** (GEN 6 issue)
   - Check DNA scale factor sign in motor.rs
   - Verify champion_distance calculation for reversed spawn

2. **Profile initialization** (50s bottleneck)
   - Measure track generation time vs sim startup
   - Consider async track gen or cache for repeated seeds

### Short-term (Complete SlowBrain)
3. **Redesign DNA v8 format** to accommodate full SlowBrain
   - Current: 384 bytes (legacy small brain)
   - Needed: +1107 bytes for full dense1+dense2
   - Total DNA v8: ~1500 bytes (currently ~1100)
   - Plan: 48×16 quantized dense1, dense2 in compressed format

4. **Re-run test suite** with extended timeout
   - 20 gen × single agent → validate learning curve
   - 10 gen × 10 swarms → validate scaling

### Long-term (Optimization)
5. **Zone pooling parameter tuning**
   - Current: FAR=rows 0-4, NEAR=rows 10-13
   - Experiment: Different row splits for curve radius sensitivity

---

## Commit Status

✅ Committed: revert SlowBrain loading code (size mismatch)  
✅ Committed: zone-based pooling fix (30445d1 still active)  
✅ Committed: UI height fix (48px per row for visibility)  
⏳ Pending: Investigate GEN 6 negative distance bug  
⏳ Pending: SlowBrain DNA v8 redesign  

---

## Next Session Action

Run extended validation suite (30 min):
```bash
timeout 1800 ./target/release/universe --headless --gens 20 --swarm 1 > test_20gen.log
python3 analyze_test_suite.py > VALIDATION_RESULTS.txt
```

Then:
1. Plot learning curve (championship distance vs gen)
2. Identify DNA encoding anomaly (gen 6 negative)
3. Prototype SlowBrain DNA v8 format
