#!/usr/bin/env python3
"""Fail-closed real-GPU contract gate. No brain/physics CPU simulation.

Build only the library test target and gpu_evolve, execute exact ignored GPU
fixtures sequentially, then prove disconnected L2 and legacy steering are caught.
The resulting receipt binds this gate, the current source and the binary.
It certifies these mechanisms, never learned navigation or delivery.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_research_battery import ROOT, ENV, sha, source_hash, read_build_provenance

FLOW = "gpu::brain::flow_contract_tests::native_brain_action_body_flow_contract"
ROTATION = "gpu::brain::flow_contract_tests::native_motor_physics_is_equivalent_in_four_headings"
CENTERED_RETINA = "gpu::brain::flow_contract_tests::vision::centered_legacy_retina_preserves_sampling_and_matches_wall_faces"
POWERTRAIN = "gpu::brain::flow_contract_tests::powertrain::"
SCHOOL = "gpu::school_tests::"
TESTS = [POWERTRAIN + "brake_priority_preserves_requested_authority",
         POWERTRAIN + "load_stall_and_no_ghost_drive",
         POWERTRAIN + "paid_start_refusal_reset_and_reverse_contract",
         POWERTRAIN + "gearbox_shift_reverse_lockout_and_birth_contract",
         POWERTRAIN + "engine_sensors_reach_native_start_head",
         POWERTRAIN + "city_judge_allows_turns_and_charges_engine_events",
         FLOW,
         FLOW + "_stable_wheels",
         ROTATION,
         "gpu::brain::flow_contract_tests::stable_wheels_timestep_refinement_contract",
         "gpu::brain::flow_contract_tests::stable_tcs_does_not_filter_a_non_slipping_pedal_request",
         "gpu::brain::flow_contract_tests::stable_tcs_state_survives_checkpoint_and_resets_at_birth",
         "gpu::brain::flow_contract_tests::resident_birth_retains_physics_profile_and_child_body",
         CENTERED_RETINA,
         "gpu::brain::flow_contract_tests::vision::retina_perception_native_action_contract",
         "gpu::organism_tests::gpu_body_paid_repair_growth_cooling_age_death_and_birth_contracts",
         "gpu::organism_tests::gpu_body_two_temperature_exchange_and_energy_budget",
         "gpu::organism_tests::gpu_body_fatigue_time_constants_and_unclipped_temperature",
         "gpu::organism_tests::gpu_body_extreme_genes_reserve_basal_before_motion",
         "gpu::organism_tests::gpu_body_resident_water_heat_and_fuel_stress",
         "gpu::taxi::tests::taxi_gpu_scripted_stock_payment_once_and_moving_refusal",
         "gpu::taxi::tests::taxi_gpu_navigation_v4_follows_legal_route_gradient",
         "gpu::traffic::gpu_tests::gpu_contact_mask_is_numeric_across_traffic_feedback_and_telemetry",
         SCHOOL + "service_v2_accepts_canonical_pickup_without_hidden_hold",
         SCHOOL + "service_v2_accepts_canonical_delivery_without_hidden_hold",
         SCHOOL + "service_v2_invalid_delivery_cannot_earn_terminal_credit",
         SCHOOL + "service_v2_blocked_timeout_is_seconds_and_does_not_kill_body",
         SCHOOL + "service_v2_idle_at_station_times_out_without_a_service_event",
         SCHOOL + "service_v2_banked_progress_counterexample"]
FAULTS = [("no-navigation-preview", "gpu::taxi::tests::taxi_gpu_navigation_v4_follows_legal_route_gradient", "preview must expose the turn before reaching the junction"),
          ("cancel-brake-priority", POWERTRAIN + "brake_priority_preserves_requested_authority", "brake priority must preserve requested authority"),
          ("unreachable-stall", POWERTRAIN + "load_stall_and_no_ghost_drive", "high gear at zero coupled RPM must stall"),
          ("disconnect-l2", FLOW, "disconnected or misindexed L2"),
          ("legacy-tire-frame", FLOW, "passive steering must not accelerate"),
          ("explicit-wheel-step", ROTATION, "heading-dependent dynamics"),
          ("instant-tcs-recovery", ROTATION, "heading-dependent dynamics"),
          ("corner-retina-grid", CENTERED_RETINA, "retina distance must use metres and the wall face"),
          ("lost-birth-profile", "gpu::brain::flow_contract_tests::resident_birth_retains_physics_profile_and_child_body", "newborn lost the resident physics profile")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(Path("/media/llm-data/universe-runs")) or out.exists():
        parser.error("--out must be a new directory below /media/llm-data/universe-runs")
    out.mkdir(parents=True)
    env = dict(os.environ, **ENV)
    env.pop("DISPLAY", None)
    env.pop("UNIVERSE_CONTRACT_FAULT", None)
    start_hash = source_hash()
    gate_hash = sha(Path(__file__))
    report = dict(schema_version=1, status="running", source_sha256=start_hash,
                  gate_sha256=gate_hash, scope="mechanisms_only_not_learned_competence", checks=[])

    def run(label, command, *, extra_env=None, timeout=180):
        (out / f"{label}.command.json").write_text(json.dumps(command, indent=2))
        with (out / f"{label}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=dict(env, **(extra_env or {})),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        return result.returncode, (out / f"{label}.log").read_text()

    try:
        code, log = run("build-tests", ["cargo", "test", "--lib", "--no-run", "--message-format=json"])
        if code:
            raise ValueError("library test build failed; see build-tests.log")
        executables = []
        for line in log.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("reason") == "compiler-artifact" and record.get("executable") and record.get("profile", {}).get("test"):
                executables.append(record["executable"])
        if len(executables) != 1:
            raise ValueError(f"expected exactly one test executable, found {executables}")
        test_binary = executables[0]
        report["test_binary_sha256"] = sha(Path(test_binary))
        code, listing = run("list-tests", [test_binary, "--list", "--ignored"])
        available = {line.removesuffix(": test") for line in listing.splitlines() if line.endswith(": test")}
        if code or not set(TESTS).issubset(available):
            raise ValueError(f"required GPU tests absent from executable: {sorted(set(TESTS)-available)}")
        for index, test in enumerate(TESTS):
            print(f"GPU contract {index+1}/{len(TESTS)}: {test}", flush=True)
            code, log = run(f"test-{index}", [test_binary, "--ignored", "--exact", test, "--nocapture", "--test-threads=1"], timeout=60)
            if code or "1 passed; 0 failed; 0 ignored;" not in log:
                raise ValueError(f"contract failed or was not executed: {test}; see test-{index}.log")
            if index == 0 and "(DiscreteGpu)" not in log and "(IntegratedGpu)" not in log:
                raise ValueError("physical GPU adapter not confirmed")
            report["checks"].append(dict(test=test, status="passed", log=f"test-{index}.log"))
        for fault, test, expected in FAULTS:
            code, log = run(f"negative-{fault}", [test_binary, "--ignored", "--exact", test, "--nocapture", "--test-threads=1"],
                            extra_env={"UNIVERSE_CONTRACT_FAULT": fault}, timeout=60)
            if code == 0 or "0 passed; 1 failed;" not in log or expected not in log:
                raise ValueError(f"negative control did not fail at its intended boundary: {fault}")
            report["checks"].append(dict(fault=fault, status="detected", log=f"negative-{fault}.log"))
        code, _ = run("build-runtime", ["cargo", "build", "--bin", "gpu_evolve"])
        if code:
            raise ValueError("runtime build failed")
        binary = ROOT / "target/debug/gpu_evolve"
        provenance = read_build_provenance(binary)
        if source_hash() != start_hash or provenance["source_sha256"] != start_hash or sha(Path(__file__)) != gate_hash:
            raise ValueError("source/gate changed during verification or runtime build is stale")
        report.update(status="passed", binary_sha256=sha(binary), binary=str(binary), provenance=provenance)
    except Exception as error:
        report.update(status="failed", error=str(error))
    (out / "receipt.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "receipt": str(out / "receipt.json"), "error": report.get("error")}), flush=True)
    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
