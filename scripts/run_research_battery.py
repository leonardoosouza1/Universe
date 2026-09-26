#!/usr/bin/env python3
"""Bounded GPU-only research runner. Plans by default; --execute runs sequentially."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
# Every executed run, gate receipt and publication lives below this partition.
RUNS_ROOT = Path("/media/llm-data/universe-runs")
ENV = dict(WGPU_BACKEND="vulkan", VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/radeon_icd.json",
           MESA_VK_VERSION_OVERRIDE="1.3", RAYON_NUM_THREADS="1", CARGO_BUILD_JOBS="1")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(root=ROOT):
    digest = hashlib.sha256()
    paths = [root / "Cargo.toml", root / "Cargo.lock", root / "build.rs"]
    paths += sorted(p for p in (root / "src").rglob("*") if p.is_file())
    paths = sorted((p.relative_to(root).as_posix(), p) for p in paths)
    for relative, path in paths:
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "little"))
        digest.update(relative_bytes)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def report_run(directory, generations):
    files = sorted(directory.glob("school_result_gen*.json"))
    if len(files) != generations:
        raise ValueError(f"{directory}: expected {generations} results, got {len(files)}")
    reports = []
    for path in files:
        r = json.loads(path.read_text())
        beings = r["beings"]
        if len(beings) != r["attempts"]:
            raise ValueError("attempt/record mismatch")
        if sum(r["failures_by_code"].values()) != r["failures"]:
            raise ValueError("incomplete failure histogram")
        if sum(bool(b["success"]) for b in beings) != r["successes"]:
            raise ValueError("success count differs from individual outcomes")
        if r.get("event_span") != "judged_attempt_frozen_at_outcome_v1":
            raise ValueError("service evidence does not declare a frozen judged span")
        for b in beings:
            required = {"pickup_requests", "dropoff_requests", "accepted_pickup_tick", "accepted_dropoff_tick"}
            if not required.issubset(b):
                raise ValueError("missing service evidence, cannot substitute zero")
            for key in ("accepted_pickup_tick", "accepted_dropoff_tick"):
                if b[key] is not None and b[key] > b["outcome_tick"]:
                    raise ValueError("accepted event lies after the judged outcome")
        pickup_requests = sum(b["pickup_requests"] for b in beings)
        dropoff_requests = sum(b["dropoff_requests"] for b in beings)
        accepted_pickups = sum(b["accepted_pickup_tick"] is not None for b in beings)
        accepted_dropoffs = sum(b["accepted_dropoff_tick"] is not None for b in beings)
        full = [b for b in beings if b["success"] and b["accepted_pickup_tick"] is not None
                and b["accepted_dropoff_tick"] is not None
                and b["accepted_pickup_tick"] < b["accepted_dropoff_tick"] and b["distance_m"] > 0]
        reports.append(dict(generation=r["generation"], attempts=r["attempts"], successes=r["successes"],
                            deaths=r["deaths"], truncated=r["truncated"], failures=r["failures_by_code"],
                            requests=dict(pickup=pickup_requests, dropoff=dropoff_requests),
                            accepted=dict(pickup=accepted_pickups, dropoff=accepted_dropoffs),
                            pickup_requests=pickup_requests, dropoff_requests=dropoff_requests,
                            accepted_pickups=accepted_pickups, accepted_dropoffs=accepted_dropoffs,
                            contact_free_successes=sum(b["success"] and b["contact_ticks"] == 0 and b["collision_energy_j"] == 0 for b in beings),
                            complete_deliveries=len(full),
                            complete_contact_free_deliveries=sum(b["contact_ticks"] == 0 and b["collision_energy_j"] == 0 for b in full),
                            complete_contact_free_alive_deliveries=sum(b["contact_ticks"] == 0 and b["collision_energy_j"] == 0 and b["alive"] and b["integrity"] >= .99 for b in full),
                            median_score=r["score_median"], research=r.get("research"),
                            start_fold=r.get("start_fold"), per_start_outcomes=r.get("per_start_outcomes")))
    return dict(generations=reports, first=reports[0], middle=reports[len(reports)//2], last=reports[-1],
                independent_unit="evolution seed; replicas and ticks are not independent")


def parse_u64_list(value, label):
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{label} must contain only u64 values") from error
    if not values or len(set(values)) != len(values) or any(item < 0 or item >= 2**64 for item in values):
        raise ValueError(f"{label} must be a non-empty comma-separated list of distinct u64 values")
    return values


def evaluation_jobs(args, controller):
    """Plan frozen audits; every job receives the same exact genome file."""
    common = ["--scenario", "organism-taxi-v1", "--school", args.lesson, "--school-mode", "bench",
              "--beings", "1", "--school-eval-starts", "3", "--school-start-pool", "3",
              "--gens", "1", "--ticks", str(args.ticks), "--school-horizon", str(args.ticks),
              "--seed", "0", "--resume", "--resume-mix", "1.0", "--replay-every", "1",
              "--no-slow", "--research-controller", controller, "--navigation-observation", args.navigation_observation]
    common += ["--evolving-body" if args.evaluate_body == "evolved" else "--fixed-body"]
    if controller == "linear":
        common += ["--no-region-mask"]
    input_hash = sha(args.evaluate_genome)
    if args.service_course_m is not None:
        common[common.index("--school-eval-starts")+1] = "4"
        common[common.index("--school-start-pool")+1] = "4"
        return [dict(name=f"course-{args.service_course_m}m", kind="course", input_genome_sha256=input_hash,
                     args=common+["--school-service-course",str(args.service_course_m)])]
    jobs = []
    for fold in ("validation", "test"):
        jobs.append(dict(name=f"starts-{fold}", kind=f"{fold}_starts", input_genome_sha256=input_hash,
                         args=common + ["--school-start-fold", fold]))
    for family in args.city_families:
        for city_seed in args.city_seeds:
            jobs.append(dict(name=f"city-{family}-s{city_seed}", kind="city_environment",
                         city_family=family, city_seed=city_seed,
                         input_genome_sha256=input_hash,
                         args=common + ["--school-start-fold", "test",
                                             "--city-family", family, "--city-seed", str(city_seed)]))
    return jobs


def argv_configuration(argv):
    """Extract only configuration stated by an argv; absent flags stay unknown."""
    values = {}
    argv = list(argv)
    for key, flag in (("navigation_observation", "--navigation-observation"),
                      ("controller", "--research-controller"),
                      ("body", "--research-body")):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} has no value in manifest job argv")
            values[key] = argv[index + 1]
    if "body" not in values:
        if "--evolving-body" in argv:
            values["body"] = "evolved"
        elif "--fixed-body" in argv:
            values["body"] = "fixed"
    return values


def _known_configuration(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"training manifest {label} must be an object")
    allowed = {"navigation_observation", "controller", "body"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"training manifest {label} has unknown configuration fields: {sorted(unknown)}")
    if any(not isinstance(item, str) for item in value.values()):
        raise ValueError(f"training manifest {label} configuration values must be strings")
    return {key: value[key] for key in allowed if key in value}


def _hash_value(value):
    return value if isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    ) else None


def _manifest_provenance(payload, contract):
    """Find explicit DNA and build provenance without inferring missing links."""
    locations = [payload, contract]
    if isinstance(payload.get("provenance"), dict):
        locations.append(payload["provenance"])
    for run in payload.get("runs", []):
        if isinstance(run, dict):
            locations.append(run)
    genome_hashes = set()
    binary_hashes = set()
    source_hashes = set()
    for location in locations:
        for key in ("evaluated_genome_sha256", "genome_sha256", "checkpoint_sha256",
                    "checkpoint_genome_sha256"):
            value = location.get(key)
            values = value if isinstance(value, list) else [value]
            genome_hashes.update(found for item in values if (found := _hash_value(item)))
        for key, destination in (("binary_sha256", binary_hashes), ("source_sha256", source_hashes)):
            found = _hash_value(location.get(key))
            if found:
                destination.add(found)
        for catalog_key in ("checkpoint_catalog", "genome_catalog"):
            catalog = location.get(catalog_key)
            if isinstance(catalog, list):
                for item in catalog:
                    if isinstance(item, dict):
                        found = _hash_value(item.get("genome_sha256") or item.get("checkpoint_genome_sha256"))
                        if found:
                            genome_hashes.add(found)
    return {"genome_sha256": sorted(genome_hashes),
            "binary_sha256": sorted(binary_hashes),
            "source_sha256": sorted(source_hashes),
            "linked": bool(genome_hashes and binary_hashes and source_hashes)}


def load_training_contract(path):
    """Read only an explicit training contract; absence remains unknown."""
    payload = json.loads(path.read_text())
    contract = payload.get("training_contract")
    if not isinstance(contract, dict):
        return None
    folds = contract.get("start_folds")
    cities = contract.get("city_pairs")
    if not isinstance(folds, list) or not all(isinstance(value, str) for value in folds):
        raise ValueError("training manifest training_contract.start_folds must be an explicit string list")
    if not isinstance(cities, list) or not all(
        isinstance(value, dict) and isinstance(value.get("family"), str)
        and isinstance(value.get("seed"), int) for value in cities
    ):
        raise ValueError("training manifest training_contract.city_pairs must be explicit family/seed objects")
    configurations = []
    explicit = contract.get("configurations")
    if explicit is not None:
        if not isinstance(explicit, list) or not all(isinstance(value, dict) for value in explicit):
            raise ValueError("training manifest training_contract.configurations must be a list of objects")
        configurations.extend(_known_configuration(value, "training_contract.configurations entry")
                             for value in explicit)
    scalar_configuration = {
        key: contract[key]
        for key in ("navigation_observation", "controller", "body")
        if key in contract
    }
    if scalar_configuration:
        configurations.append(_known_configuration(scalar_configuration, "training_contract"))
    for entry in payload.get("jobs", []):
        if not isinstance(entry, dict):
            continue
        argv = next((entry[key] for key in ("args", "argv", "command") if key in entry), None)
        if argv is None:
            continue
        if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
            raise ValueError("training manifest job argv must be a string list")
        configuration = argv_configuration(argv)
        if configuration:
            configurations.append(configuration)
    unique_configurations = []
    for configuration in configurations:
        if configuration not in unique_configurations:
            unique_configurations.append(configuration)
    provenance = _manifest_provenance(payload, contract)
    return {"start_folds": sorted(set(folds)),
            "city_pairs": sorted({(value["family"], value["seed"]) for value in cities}),
            "configurations": unique_configurations,
            "provenance": provenance}


def classify_evaluation_jobs(jobs, training_contract):
    """Mark transfer only when the manifest proves the target was excluded."""
    for job in jobs:
        audit_configuration = argv_configuration(job["args"])
        job["audit_configuration"] = audit_configuration
        if job["kind"] == "course":
            job.update(scenario_role="audit", scenario_role_reason="explicit_training_course_reevaluation")
            job["name"] += "-audit"
            continue
        if training_contract is None:
            job["scenario_role"] = "audit"
            job["training_configuration_status"] = "unknown"
            job["scenario_role_reason"] = "no_training_manifest_contract"
        else:
            training_configurations = training_contract.get("configurations", [])
            statuses = []
            for training_configuration in training_configurations:
                overlap = set(audit_configuration).intersection(training_configuration)
                if not overlap:
                    continue
                statuses.append(all(audit_configuration[key] == training_configuration[key]
                                    for key in overlap))
            if statuses and not any(statuses):
                raise ValueError(f"{job['name']}: audit configuration conflicts with training manifest")
            configuration_status = ("matched" if any(statuses) else "unknown")
            job["training_configuration_status"] = configuration_status
            provenance = training_contract.get("provenance", {})
            genome_linked = provenance.get("linked") is True and job.get("input_genome_sha256") in provenance.get("genome_sha256", [])
            if job["kind"].endswith("_starts"):
                fold = job["kind"].removesuffix("_starts")
                known_disjoint = "all" not in training_contract["start_folds"] and fold not in training_contract["start_folds"]
            else:
                known_disjoint = (job["city_family"], job["city_seed"]) not in training_contract["city_pairs"]
            if not genome_linked:
                job["scenario_role"] = "audit"
                job["scenario_role_reason"] = "training_manifest_lacks_matching_genome_provenance"
            elif known_disjoint and configuration_status == "matched":
                job["scenario_role"] = "transfer"
                job["scenario_role_reason"] = "explicit_fold_or_city_exclusion_and_matching_genome_provenance"
            else:
                job["scenario_role"] = "audit"
                job["scenario_role_reason"] = ("training_manifest_does_not_exclude_target"
                                                 if not known_disjoint else "training_configuration_unknown")
        suffix = job["scenario_role"]
        if not job["name"].endswith(f"-{suffix}"):
            job["name"] = f"{job['name']}-{suffix}"
    return jobs


def argv_has_value(argv, flag, expected):
    return flag in argv and argv[argv.index(flag) + 1:argv.index(flag) + 2] == [expected]


def verify_mutation_proofs(directory, argv, generations):
    """Verify the ancestry evidence emitted by the head-only mutation profile."""
    if not argv_has_value(argv, "--mutation-profile", "head-only"):
        return {"status": "not_requested", "files": []}
    proof_paths = sorted(directory.glob("mutation_proof_gen*.json"))
    expected = max(generations - 1, 0)
    if expected == 0:
        return {"status": "not_applicable", "reason": "no_evolution_step", "files": []}
    if len(proof_paths) != expected:
        raise ValueError(f"{directory}: head-only expected {expected} mutation proofs, got {len(proof_paths)}")
    verified = []
    for path in proof_paths:
        payload = json.loads(path.read_text())
        if payload.get("all_match") is not True or not isinstance(payload.get("offspring"), list):
            raise ValueError(f"{path}: mutation proof does not establish all_match")
        for offspring in payload["offspring"]:
            if not isinstance(offspring, dict) or not isinstance(offspring.get("matching_parent"), int):
                raise ValueError(f"{path}: mutation proof has incomplete parent evidence")
            if not isinstance(offspring.get("immutable_sha256"), str) or len(offspring["immutable_sha256"]) != 64:
                raise ValueError(f"{path}: mutation proof has incomplete immutable hash evidence")
            if not isinstance(offspring.get("service_weights_different_from_matching_parent"), int):
                raise ValueError(f"{path}: mutation proof has incomplete service mutation evidence")
        verified.append({"file": path.name, "sha256": sha(path), "offspring": len(payload["offspring"])})
    return {"status": "verified", "expected_files": expected, "files": verified}


def _resolve_artifact(directory, value):
    path = Path(value)
    if path.is_file():
        return path
    relative = directory / path
    return relative if relative.is_file() else None


def verify_reinjections(directory):
    """Verify every reinjected genome file named by a reinjection manifest."""
    manifests = sorted(directory.glob("skill_reinjection_gen*.json"))
    files = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text())
        sources = payload.get("sources")
        if not isinstance(sources, list):
            raise ValueError(f"{manifest}: reinjection manifest has no sources list")
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("file"), str):
                raise ValueError(f"{manifest}: reinjection source has no file")
            path = _resolve_artifact(directory, source["file"])
            if path is None:
                raise ValueError(f"{manifest}: reinjected file is missing: {source['file']}")
            actual = sha(path)
            expected = source.get("genome_sha256")
            if not isinstance(expected, str) or actual != expected:
                raise ValueError(f"{manifest}: reinjected file hash mismatch: {path}")
            files.append({"manifest": manifest.name, "file": str(path), "sha256": actual})
    standalone = sorted((directory / "skills").glob("reinjected-*.bin")) if (directory / "skills").is_dir() else []
    named = {Path(entry["file"]).resolve() for entry in files}
    for path in standalone:
        if path.resolve() not in named:
            files.append({"manifest": None, "file": str(path), "sha256": sha(path), "unreferenced": True})
    if not manifests and not standalone:
        return {"status": "not_present", "files": []}
    if any(entry.get("unreferenced") for entry in files):
        return {"status": "unreferenced_files", "files": files}
    return {"status": "verified", "files": files}


def verify_run_artifacts(directory, argv, generations):
    return {"mutation": verify_mutation_proofs(directory, argv, generations),
            "reinjection": verify_reinjections(directory)}


def read_build_provenance(binary):
    """Query the binary's read-only provenance path before any GPU invocation."""
    raw = subprocess.check_output([str(binary), "--build-provenance"], cwd=ROOT, text=True)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("source_sha256"), str):
        raise ValueError("binary --build-provenance returned no source_sha256")
    return payload


def main():
    runner_hash_at_entry = sha(Path(__file__).resolve())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT/"target/debug/gpu_evolve")
    p.add_argument("--suite", choices=["services", "canary", "body", "curriculum", "full-service"], default="services")
    p.add_argument("--service-course-m", type=int, help="full pickup+delivery on an explicit short straight course; 12..96 metres in 3m cells")
    p.add_argument("--navigation-observation", choices=["v2", "v3", "v4"], default="v2")
    p.add_argument("--lesson", choices=["pickup", "delivery"], default="pickup")
    p.add_argument("--seeds", default="11,12,13")
    p.add_argument("--gens", type=int, default=6)
    p.add_argument("--beings", type=int, default=8)
    p.add_argument("--curriculum-body", choices=["fixed", "evolved"], default="fixed")
    p.add_argument("--curriculum-controller", choices=["native", "linear"], default="native",
                   help="controller contract for the curriculum distance-learning A/B")
    p.add_argument("--start-fold", choices=["all", "train"], default="all")
    p.add_argument("--ticks", type=int, default=256)
    p.add_argument("--timeout-seconds", type=int, default=300)
    p.add_argument("--evaluate-genome", type=Path,
                   help="run the frozen validation/test/city suite for one exact genome snapshot")
    p.add_argument("--diagnostics", action="store_true", help="capture native neural and motor traces during frozen evaluation only")
    p.add_argument("--evaluate-body", choices=["fixed", "evolved"], help="required frozen morphology expression contract")
    p.add_argument("--evaluate-controller", choices=["native", "linear"],
                   help="controller configuration used to produce --evaluate-genome; required with it")
    p.add_argument("--city-seeds", default="1001,1002,1003",
                   help="city environment seeds for --evaluate-genome")
    p.add_argument("--city-families", default="grid,compact,wide",
                   help="city spacing families for --evaluate-genome")
    p.add_argument("--training-manifest", type=Path,
                   help="optional manifest with explicit training_contract exclusions")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--gate-receipt", type=Path, help="required with --execute: current source/binary receipt from check_native_flow_gpu.py")
    a = p.parse_args()
    if a.diagnostics and (a.evaluate_genome is None or a.evaluate_controller != "native"):
        p.error("--diagnostics requires a frozen native controller evaluation")
    if a.suite == "full-service" and a.service_course_m is None:
        p.error("--suite full-service requires --service-course-m")
    if a.service_course_m is not None and (a.service_course_m not in range(12,97,3)
            or a.lesson != "delivery" or a.start_fold != "all"
            or (a.evaluate_genome is None and a.suite != "full-service")):
        p.error("short service course requires delivery, all starts and full-service suite or frozen evaluation")
    try:
        seeds = parse_u64_list(a.seeds, "--seeds")
        city_seeds = parse_u64_list(a.city_seeds, "--city-seeds")
    except ValueError as error:
        p.error(str(error))
    city_families = [family.strip() for family in a.city_families.split(",") if family.strip()]
    if not city_families or any(family not in {"grid", "compact", "wide"} for family in city_families):
        p.error("--city-families must contain grid, compact and/or wide")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**64 for s in seeds):
        p.error("seeds must be distinct u64 values")
    if not 6 <= a.beings <= 64 or not 1 <= a.gens <= 100 or not 32 <= a.ticks <= 32768 or not 1 <= a.timeout_seconds <= 3600:
        p.error("invalid bounded generation/tick/time budget")
    if a.evaluate_genome is not None:
        if a.evaluate_body is None:
            p.error("--evaluate-genome requires --evaluate-body fixed|evolved to preserve expressed morphology")
        if a.evaluate_controller is None:
            p.error("--evaluate-genome requires --evaluate-controller native|linear")
        if not a.evaluate_genome.is_file():
            p.error(f"--evaluate-genome is not a file: {a.evaluate_genome}")
        # A frozen suite has its own one-generation contract; callers do not
        # need to remember to override the evolution-suite default.
        a.gens = 1
    training_contract = None
    if a.training_manifest is not None:
        if not a.training_manifest.is_file():
            p.error(f"--training-manifest is not a file: {a.training_manifest}")
        try:
            training_contract = load_training_contract(a.training_manifest)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            p.error(f"invalid --training-manifest: {error}")
    arms = {
        "services": [("native", []), ("head-only", ["--mutation-profile", "head-only"]),
                     ("linear", ["--research-controller", "linear", "--no-region-mask"])],
        "body": [(m, ["--research-body", m]) for m in ("fixed", "random", "evolved")],
        "curriculum": [("near-to-native", ["--school-start-curriculum", "0,6,18,36,native", "--school-retain-skills", "2"])],
        "canary": [(f, ["--city-family", f, "--city-seed", "1001"]) for f in ("grid", "compact", "wide")],
        "full-service": [("full-service", ["--school-service-course", str(a.service_course_m)])],
    }[a.suite]
    if a.evaluate_genome is not None:
        a.city_seeds = city_seeds
        a.city_families = city_families
        jobs = evaluation_jobs(a, a.evaluate_controller)
        try:
            classify_evaluation_jobs(jobs, training_contract)
        except ValueError as error:
            p.error(str(error))
    else:
        jobs = []
        for seed in seeds:
            for label, extra in arms:
                args = ["--scenario", "organism-taxi-v1", "--navigation-observation", a.navigation_observation, "--school", a.lesson, "--school-mode", "bench", "--no-slow",
                        "--beings", str(a.beings), "--school-start-fold", a.start_fold, "--school-eval-starts", "3", "--school-start-pool", "9",
                        "--gens", str(a.gens), "--ticks", str(a.ticks), "--school-horizon", str(a.ticks),
                        "--school-deadline-frac", "1", "--seed", str(seed), "--replay-every", str(a.gens)]
                if a.suite != "body":
                    args += ["--research-body", a.curriculum_body if a.suite in ("curriculum", "full-service") else "fixed"]
                if a.suite == "full-service":
                    args[args.index("--school-eval-starts")+1] = "4"
                    args[args.index("--school-start-pool")+1] = "4"
                if a.suite in ("curriculum", "full-service"):
                    args += ["--research-controller", a.curriculum_controller]
                    if a.curriculum_controller == "linear":
                        args += ["--no-region-mask"]
                if a.suite not in ("curriculum", "full-service"):
                    args += ["--school-start-distance-m", "0"]
                jobs.append(dict(name=f"{label}-s{seed}", kind="evolution", evolution_seed=seed, args=args+extra))
    plan = dict(schema_version=2, suite=a.suite, jobs=jobs, gpu_environment=ENV,
                lesson=a.lesson,
                ticks_ceiling=len(jobs)*(4 if a.service_course_m is not None else 3)*a.ticks*(1 if a.evaluate_genome else a.beings*a.gens),
                claim="measurement only; audit by default; transfer only when the training manifest proves exclusion",
                frozen_evaluation=a.evaluate_genome is not None,
                scenario_role=("transfer" if jobs and all(job.get("scenario_role") == "transfer" for job in jobs)
                                else "audit" if jobs and all(job.get("scenario_role", "audit") == "audit" for job in jobs)
                                else "mixed"),
                controller=(a.evaluate_controller if a.evaluate_genome is not None
                            else a.curriculum_controller if a.suite in ("curriculum", "full-service") else None),
                stop="first process/contract failure or timeout", timeout_seconds_per_run=a.timeout_seconds)
    if a.evaluate_genome is None:
        plan["training_contract"] = {"start_folds":[a.start_fold], "fresh_founders":True,
            "navigation_observation":a.navigation_observation,
            "controller":a.curriculum_controller if a.suite in ("curriculum", "full-service") else None,
            "body":a.curriculum_body if a.suite in ("curriculum", "full-service") else "fixed",
            "city_pairs":[{"family":f,"seed":1001} for f in ("grid","compact","wide")] if a.suite=="canary" else [{"family":"legacy-grid","seed":0}]}
        if a.service_course_m is not None:
            plan["training_contract"]["service_course_m"] = a.service_course_m
            plan["training_contract"]["city_pairs"] = [{"family":f"short-service-{a.service_course_m}m","seed":0}]
        if a.suite not in ("curriculum", "full-service"):
            plan["training_contract"].pop("controller")
        if a.suite == "body":
            plan["training_contract"].pop("body")
    if a.evaluate_genome is not None:
        plan["evaluation_body"] = a.evaluate_body
        plan.update(evaluate_genome=str(a.evaluate_genome.resolve()),
                    input_genome_sha256=sha(a.evaluate_genome),
                    evaluation_contract={"gens": 1, "beings": 1, "eval_starts": 3,
                                         "resume": True, "resume_mix": 1.0,
                                         "controller_flags": (["--no-slow", "--research-controller", a.evaluate_controller]
                                                              + (["--no-region-mask"] if a.evaluate_controller == "linear" else [])),
                                         "start_folds": ["validation", "test"],
                                         "city_families": city_families, "city_seeds": city_seeds,
                                         "online_updates": False,
                                         "scenario_role": plan["scenario_role"]})
        plan["training_contract"] = training_contract
        if a.service_course_m is not None:
            plan["evaluation_contract"].update(start_folds=["all"], city_families=[], city_seeds=[],
                                               eval_starts=4, service_course_m=a.service_course_m)
    if a.training_manifest is not None:
        plan["training_manifest"] = str(a.training_manifest.resolve())
        plan["training_manifest_sha256"] = sha(a.training_manifest)
    if not a.execute:
        print(json.dumps(plan, indent=2))
        return
    try:
        if a.gate_receipt is None:
            raise ValueError("--execute requires --gate-receipt from scripts/check_native_flow_gpu.py")
        receipt = json.loads(a.gate_receipt.read_text())
        if (receipt.get("status") != "passed" or receipt.get("schema_version") != 1
                or receipt.get("source_sha256") != source_hash()
                or receipt.get("binary_sha256") != sha(a.binary)
                or receipt.get("gate_sha256") != sha(ROOT / "scripts/check_native_flow_gpu.py")):
            raise ValueError("GPU gate receipt failed, is stale, or belongs to another source/binary")
        from check_native_flow_gpu import TESTS, FAULTS
        passed = {item.get("test") for item in receipt.get("checks", []) if item.get("status") == "passed"}
        detected = {item.get("fault") for item in receipt.get("checks", []) if item.get("status") == "detected"}
        if passed != set(TESTS) or detected != {fault for fault, _, _ in FAULTS}:
            raise ValueError("GPU gate receipt does not cover all required contracts and negative controls")
        plan["gpu_contract_gate"] = {"path": str(a.gate_receipt.resolve()), "sha256": sha(a.gate_receipt), "scope": receipt["scope"]}
    except (OSError, ValueError, KeyError) as error:
        p.error(str(error))
    out = a.out.resolve()
    if not out.is_relative_to(RUNS_ROOT):
        p.error(f"executed runs must be on {RUNS_ROOT}")
    if out.exists():
        p.error("output must be a new directory; evidence is never overwritten")
    out.mkdir(parents=True)
    binary = out/"gpu_evolve"
    shutil.copy2(a.binary, binary)
    # The receipt was checked against the path above; the copy is what runs, so
    # a rebuild between the two checks must not slip through.
    if sha(binary) != receipt["binary_sha256"]:
        raise ValueError("binary changed between the gate-receipt check and the evidence copy")
    plan.update(binary_sha256=sha(binary), source_sha256=source_hash(),
                runner_sha256=runner_hash_at_entry,
                base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                dirty_files=subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                                    cwd=ROOT, text=True).splitlines())
    build_provenance = read_build_provenance(binary)
    if build_provenance["source_sha256"] != plan["source_sha256"]:
        raise ValueError("binary build provenance does not match the runner source snapshot")
    plan["build_provenance"] = build_provenance
    if a.evaluate_genome is not None:
        plan["input_genome_sha256"] = sha(a.evaluate_genome)
    (out/"manifest.json").write_text(json.dumps(plan, indent=2))
    (out/"source.patch").write_bytes(subprocess.check_output(["git", "diff", "--", "src", "scripts"], cwd=ROOT))
    # Include untracked modules too; the tracked diff alone is not a source snapshot.
    snapshot=out/"source"
    shutil.copytree(ROOT/"src", snapshot/"src")
    for name in ("Cargo.toml", "Cargo.lock", "build.rs"):
        shutil.copy2(ROOT/name, snapshot/name)
    (snapshot/"scripts").mkdir()
    shutil.copy2(Path(__file__).resolve(), snapshot/"scripts"/"run_research_battery.py")
    if source_hash(snapshot) != plan["source_sha256"] or sha(snapshot/"scripts"/"run_research_battery.py") != runner_hash_at_entry:
        raise ValueError("source or runner changed during evidence snapshot; refusing mixed provenance")
    if a.training_manifest is not None:
        shutil.copy2(a.training_manifest, snapshot/"training_manifest.json")
    env=os.environ.copy(); env.update(ENV); env.pop("DISPLAY", None)
    reports=[]
    try:
        for job in jobs:
            directory=out/job["name"]; directory.mkdir()
            if a.evaluate_genome is not None:
                # The binary's population resume contract looks for this exact
                # filename. Keep a second immutable copy so the post-run hash
                # proves which bytes were supplied to the frozen evaluation.
                input_genome = directory/"genome_input.bin"
                checkpoint = directory/"evolve_best_gen0000.bin"
                shutil.copy2(a.evaluate_genome, input_genome)
                shutil.copy2(a.evaluate_genome, checkpoint)
                job["input_genome_sha256"] = sha(input_genome)
                if job["input_genome_sha256"] != plan["input_genome_sha256"]:
                    raise ValueError(f"{job['name']}: genome snapshot changed while planning")
            command=[str(binary), *job["args"], "--out", str(directory)]
            if a.diagnostics:
                # DrivingDiagnostics owns observation.jsonl. The legacy --observe
                # writer is a separate stream and must never open that same path.
                command += ["--driving-diagnostics", "--kernel-profile", "1024"]
            (directory/"command.json").write_text(json.dumps(command, indent=2))
            (directory/"provenance.json").write_text(json.dumps({
                "binary_sha256": plan["binary_sha256"],
                "source_sha256": plan["source_sha256"],
                "runner_sha256": plan["runner_sha256"],
                "input_copy_sha256": job.get("input_genome_sha256"),
                "frozen": a.evaluate_genome is not None,
                "claim": "measurement only; no update claim",
            }, indent=2))
            start=time.monotonic()
            with (directory/"stdout.log").open("w") as log:
                subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=a.timeout_seconds, check=True)
            result=report_run(directory, 1 if a.evaluate_genome is not None else a.gens)
            verification = verify_run_artifacts(directory, job["args"],
                                                 1 if a.evaluate_genome is not None else a.gens)
            if a.evaluate_genome is not None:
                saved_checkpoint = directory/"evolve_best_gen0000.bin"
                if not saved_checkpoint.is_file() or sha(saved_checkpoint) != plan["input_genome_sha256"]:
                    raise ValueError(f"{job['name']}: exported genome does not match the supplied input")
                metadata_path = directory/"gen0000.meta.json"
                if not metadata_path.is_file():
                    raise ValueError(f"{job['name']}: missing replay metadata with checkpoint_loaded evidence")
                replay_metadata = json.loads(metadata_path.read_text())
                if replay_metadata.get("checkpoint_loaded") is not True:
                    raise ValueError(f"{job['name']}: replay metadata does not prove checkpoint loading")
                if replay_metadata.get("checkpoint_mix") != 1.0:
                    raise ValueError(f"{job['name']}: replay metadata reports checkpoint mixing")
                (directory/"provenance.json").write_text(json.dumps({
                    "binary_sha256": plan["binary_sha256"],
                    "source_sha256": plan["source_sha256"],
                    "runner_sha256": plan["runner_sha256"],
                    "input_copy_sha256": job["input_genome_sha256"],
                    "exported_genome_sha256": sha(saved_checkpoint),
                    "checkpoint_loaded": replay_metadata["checkpoint_loaded"],
                    "checkpoint_mix": replay_metadata["checkpoint_mix"],
                    "scenario_role": job.get("scenario_role", "audit"),
                    "verification": verification,
                    "claim": "measurement only; no update claim",
                }, indent=2))
            provenance = json.loads((directory/"provenance.json").read_text())
            provenance["verification"] = verification
            (directory/"provenance.json").write_text(json.dumps(provenance, indent=2))
            checkpoint = directory / f"evolve_best_gen{(1 if a.evaluate_genome is not None else a.gens) - 1:04}.bin"
            checkpoint_catalog_entry = ({"name": job["name"], "file": str(checkpoint),
                                         "genome_sha256": sha(checkpoint)}
                                        if checkpoint.is_file() else None)
            result.update(name=job["name"], kind=job.get("kind", "evolution"),
                          evolution_seed=job.get("evolution_seed"),
                          city_family=job.get("city_family"), city_seed=job.get("city_seed"),
                          scenario_role=job.get("scenario_role"),
                          input_copy_sha256=job.get("input_genome_sha256"),
                          exported_genome_sha256=(sha(directory/"evolve_best_gen0000.bin")
                                                  if a.evaluate_genome is not None else None),
                          checkpoint_loaded=(replay_metadata.get("checkpoint_loaded")
                                            if a.evaluate_genome is not None else None),
                          checkpoint_mix=(replay_metadata.get("checkpoint_mix")
                                         if a.evaluate_genome is not None else None),
                          verification=verification,
                          checkpoint_file=(str(checkpoint) if checkpoint_catalog_entry else None),
                          checkpoint_genome_sha256=(checkpoint_catalog_entry["genome_sha256"]
                                                    if checkpoint_catalog_entry else None),
                          binary_sha256=plan["binary_sha256"], source_sha256=plan["source_sha256"],
                          frozen=a.evaluate_genome is not None,
                          online_updates=False if a.evaluate_genome is not None else None,
                          wall_seconds=time.monotonic()-start)
            reports.append(result)
            (out/"report.json").write_text(json.dumps(dict(status="running", runs=reports), indent=2))
            print(job["name"], result["last"], flush=True)
    except (Exception, KeyboardInterrupt) as error:
        (out/"report.json").write_text(json.dumps(dict(status="failed", error=str(error), runs=reports), indent=2))
        raise
    catalog = [dict(name=run["name"], file=run["checkpoint_file"],
                    genome_sha256=run["checkpoint_genome_sha256"])
               for run in reports if run.get("checkpoint_genome_sha256")]
    plan["checkpoint_catalog"] = catalog
    (out/"manifest.json").write_text(json.dumps(plan, indent=2))
    (out/"report.json").write_text(json.dumps(dict(status="completed", runs=reports,
                                                    checkpoint_catalog=catalog,
                                                    binary_sha256=plan["binary_sha256"],
                                                    source_sha256=plan["source_sha256"]), indent=2))


if __name__ == "__main__":
    main()
