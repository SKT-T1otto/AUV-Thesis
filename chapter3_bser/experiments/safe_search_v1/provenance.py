"""Exact opt-in SafeSearch inventory; never relaxes the historical entry gate.

The manifest is sealed only after source review. Runtime checks hash raw bytes,
not normalized text, and require one complete reviewed checkout profile.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "docs/provenance/safe_search_v1_evolution.json"
REVIEW = "docs/chapter3/search_diagnostics/safe_search_v1/source_review.md"
PLAN = "docs/chapter3/search_diagnostics/3090_20260928/experiment_plan.json"
PIN = "docs/chapter3/baselines/final_production_source.json"
PROTECTED = "docs/chapter3/baselines/framework_protected_baseline.json"
BASELINE_EVOLUTION = "docs/provenance/baseline_maddpg_evolution.json"
MIGRATION = "docs/provenance/ch3_to_core_migration_manifest.json"
COLLISION = "docs/provenance/collision_terminal_evolution.json"
HGR = "docs/provenance/hgr_evolution.json"
SCHEMA = "ch3.safe_search.reviewed_evolution.v1"
PATTERNS = (("core", "*.py"), ("chapter3_bser", "*.py"),
            ("tools/ch3_baselines", "*.py"), ("configs", "*.json"),
            ("scripts", "*.py"), ("scripts", "*.sh"))
FROZEN_RAW = {
    PIN: ("8a8a92c708924699a3638341a91857da9a7b330d5f17282aba1a7bcf0e5cdb40",),
    PROTECTED: ("b1daeadfad570d782e8df1cfc86f84f51900253156bf8f626367f1cb02fb2204",),
    BASELINE_EVOLUTION: ("0ba9da52edc45ffb7106cb6e87545e252b8ff4fa43fb7028ef236418bc7086ac",),
    MIGRATION: ("df1b34ac73d23606078c10866717fac27986e0c24d3f8d4a4d45a2e40dbc4c69",),
    COLLISION: ("9cf26c6f5c187410789f281d115cb62ea6ce7b9fe4d43084b7863500c9628388",
                "a3f8367b378f81f4bd1751928e3a2c5a4cce582854bcaf714570dced5c651b91"),
    HGR: ("14272e954f50bea1a5ff05a6810268df17e990a66b0b866a13588dc8357a4dc9",
          "d31e772cf4db7ff8406e7bc7ec2a9c432760baeafa1e0b315f15e74b601e68fb"),
}
# The nine pre-existing production differences were reviewed independently of
# this change. These hashes name that intermediate state, including runtime.py
# before the new default-equivalent construction hooks.
REVIEWED_HGR = {
    "chapter3_bser/experiments/hgr/build_phase2_source.py": "4a7378d3c9eb0235e59279b795174f64477efa83af6c3379dea68aaac9929e9c",
    "chapter3_bser/experiments/hgr/phase1_acceptance.py": "88f8d3d4889a607d6952c46d9374e46e57247e7a321af6f7938f09b8d45fc7e4",
    "chapter3_bser/experiments/hgr/phase2_gradient_efficiency.py": "ae8669eea0be61a583a9830213eeac39543cc4f8095ebc6eeb05e8554555dc97",
    "chapter3_bser/experiments/hgr/runtime.py": "b3dfa615b438b101cc47fc74715dcd9a1ab9fc04ab4bb23bc38022ba4a5ab19d",
    "chapter3_bser/experiments/hgr/train.py": "f43c7dd91d0adc51bdcd687a04cbfe2328fd8998d5dae38f465c2c5ff78f5edf",
    "chapter3_bser/models/hgr/estimator.py": "e5a63ffe1de3aec9d320da6b270e16c7c287121f43b57b03dfb70f936f5ec1b4",
    "chapter3_bser/models/hgr/phase1.py": "06d711e1e04a68b458ebdcca6c8299823b2ed3968bba390dec754615d8d09bbc",
    "chapter3_bser/models/hgr/policy.py": "f572bbed8ea9d0b5e755761f56ef42d5bf0a1fba59f202f62d1bcc1fbb10a7d8",
    "chapter3_bser/models/hgr/stable_predictor.py": "0a3d64833218a1955cbcb45aae42eee2af0e24ee12206b117155053ee08233a8",
}
REVIEWED_LAUNCHER = {
    "scripts/linux/run_hgr_phase1_acceptance.sh": "533e7203a3716fc160cd81a3a624330b71eeaee0c8fb637e58039c32d036ac6e",
}
HOOK_PATHS = {"chapter3_bser/experiments/hgr/runtime.py",
              "tools/ch3_baselines/basic_search_prior.py", "chapter3_bser/online/allocator.py"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_files(files):
    if not isinstance(files, dict) or not files:
        raise ValueError("source inventory must be nonempty")
    for name, sha in files.items():
        if not isinstance(name, str):
            raise ValueError("invalid source filename")
        path = PurePosixPath(name)
        if (not isinstance(name, str) or name != path.as_posix() or path.is_absolute()
                or ".." in path.parts or "\\" in name
                or not isinstance(sha, str) or re.fullmatch("[0-9a-f]{64}", sha) is None):
            raise ValueError("invalid source record: " + str(name))


def inventory(root=None):
    root = ROOT if root is None else Path(root)
    paths = {p for directory, pattern in PATTERNS for p in (root / directory).rglob(pattern)}
    paths.update(root / name for name in (*FROZEN_RAW, PLAN, REVIEW, ".gitattributes",
                                        "tools/benchmark_d2_performance.py"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("source inventory cannot contain symlinks")
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)}
    return dict(files=files, sha256=digest(files))


def _historical_profiles(root):
    # Verify original bytes before interpreting their records. LF and CRLF
    # alternatives here name exact historical files, never arbitrary endings.
    for name, allowed in FROZEN_RAW.items():
        if file_sha256(root / name) not in allowed:
            raise ValueError("frozen historical manifest changed: " + name)
    pin, protected, evolution = (read_json(root / n) for n in (PIN, PROTECTED, BASELINE_EVOLUTION))
    if (pin["schema"] != "ch3.final_experiment.production.v1"
            or protected["schema"] != "ch3.final_experiment.protected_baseline.v1"
            or evolution["schema"] != "ch3.baseline.maddpg_evolution.v1"
            or protected["sha256"] != digest(protected["files"])
            or evolution["sha256"] != digest({k: v for k, v in evolution.items() if k != "sha256"})
            or evolution["historical_protected_sha256"] != protected["sha256"]
            or evolution["historical_production_sha256"] != pin["production"]["sha256"]
            or protected["production_source_sha256"] != pin["production"]["sha256"]):
        raise ValueError("historical baseline bindings changed")
    migration = read_json(root / MIGRATION)
    if migration["authority_record_count"] != 27 or len(migration["records"]) != 27:
        raise ValueError("all 27 migration records are required")
    sources = {"git_lf": pin["production"], **pin["reviewed_worktree_profiles"]}
    if set(sources) != {"git_lf", "windows_existing"}:
        raise ValueError("unexpected original source profiles")
    result = {}
    for profile, source in sources.items():
        _validate_files(source["files"])
        if source["sha256"] != hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest():
            raise ValueError("historical source aggregate mismatch")
        files = dict(source["files"])
        if files.keys() & evolution["added_production"].keys():
            raise ValueError("baseline evolution replaced historical production")
        files.update(evolution["added_production"])
        baseline = dict(protected["files"])
        for name, record in evolution["protected_changes"].items():
            if baseline.get(name) != record["before"]:
                raise ValueError("historical protected before hash mismatch")
            baseline[name] = record["after"]
        files.update(baseline)
        result[profile] = files
    # Git attributes preserve six production files while JSON metadata remains
    # LF. This is a third complete exact profile, not a per-file acceptance mix.
    result["git_clone_preserved"] = dict(result["windows_existing"])
    return result


def _prior_changes(root, profiles):
    declared = read_json(root / PLAN)["source_gate"]["existing_differences"]
    if {r["path"]: r["actual_sha256"] for r in declared} != REVIEWED_HGR or len(declared) != 9:
        raise ValueError("the nine reviewed HGR differences changed")
    for files in profiles.values():
        for record in declared:
            if files.get(record["path"]) != record["expected_sha256"]:
                raise ValueError("reviewed HGR before hash mismatch: " + record["path"])
        files.update(REVIEWED_HGR)
        if files.keys() & REVIEWED_LAUNCHER.keys():
            raise ValueError("reviewed launcher must be an addition")
        files.update(REVIEWED_LAUNCHER)
    return profiles


def framework_sources(root=None):
    """Fail closed unless every byte/name matches a complete sealed profile."""
    root = ROOT if root is None else Path(root)
    prior = _prior_changes(root, _historical_profiles(root))
    manifest = read_json(root / MANIFEST)
    if (manifest.get("schema") != SCHEMA or manifest.get("review_status") != "reviewed"
            or manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"})
            or manifest.get("source_review_sha256") != file_sha256(root / REVIEW)
            or manifest.get("reviewed_hgr_changes") != REVIEWED_HGR
            or manifest.get("reviewed_existing_launcher") != REVIEWED_LAUNCHER
            or set(manifest.get("profiles", {})) != set(prior)):
        raise ValueError("invalid SafeSearch reviewed evolution manifest")
    changed = manifest.get("changed_paths")
    added = manifest.get("added_paths")
    if (not isinstance(changed, list) or set(changed) != HOOK_PATHS or len(changed) != len(HOOK_PATHS)
            or not isinstance(added, list) or len(added) != len(set(added))
            or any(not (n.startswith("chapter3_bser/experiments/safe_search_v1/") and n.endswith(".py")
                        or n.startswith("configs/chapter3/safe_search_v1/") and n.endswith(".json")
                        or n.startswith("scripts/linux/") and "safe_search" in n and n.endswith(".sh"))
                   for n in added)):
        raise ValueError("unexpected SafeSearch change scope")
    current = inventory(root)
    matched = []
    for profile, files in prior.items():
        reviewed = manifest["profiles"][profile]
        expected = reviewed["files"]
        _validate_files(expected)
        if reviewed.get("sha256") != digest(expected):
            raise ValueError("SafeSearch profile aggregate mismatch")
        changes = reviewed["safe_changes"]
        if set(changes) != set(changed) | set(added):
            raise ValueError("SafeSearch exact change list mismatch")
        for name, record in changes.items():
            if record["before"] != files.get(name) or record["after"] != expected.get(name):
                raise ValueError("SafeSearch before/after hash mismatch: " + name)
            if (name in added) != (record["before"] is None):
                raise ValueError("SafeSearch addition replaced historical source")
        for name, sha in files.items():
            if expected.get(name) != changes.get(name, {}).get("after", sha):
                raise ValueError("unreviewed historical source evolution: " + name)
        if set(expected) - set(files) - set(added) != set(reviewed["additional_inputs"]):
            raise ValueError("additional input inventory mismatch")
        if any(expected[n] != sha for n, sha in reviewed["additional_inputs"].items()):
            raise ValueError("additional input hashes mismatch")
        # A separately reviewed successor may add chapter-local modules and
        # this dispatch hook. Historical manifests and every old profile above
        # are still checked in full; no directory or hash is exempted.
        successor = root / "docs/provenance/safe_search_v2_evolution.json"
        if successor.exists():
            from chapter3_bser.experiments.safe_search_v2.provenance import extend_profile
            expected = extend_profile(root, profile, expected)
        effect = root / "docs/provenance/bser_effect_v1_evolution.json"
        if effect.exists():
            from chapter3_bser.experiments.bser_effect_v1.provenance import extend_profile as extend_effect
            expected = extend_effect(root, profile, expected)
        final = root / "docs/provenance/bser_final_v1_evolution.json"
        if final.exists():
            from chapter3_bser.experiments.bser_final_v1.provenance import extend_profile as extend_final
            expected = extend_final(root, profile, expected)
        d2 = root / "docs/provenance/d2_v1_evolution.json"
        if d2.exists():
            from chapter3_bser.experiments.d2_v1.provenance import extend_profile as extend_d2
            expected = extend_d2(root, profile, expected)
        suite = root / "docs/provenance/d2_suite_v1_evolution.json"
        if suite.exists():
            from chapter3_bser.experiments.d2_suite_v1.provenance import extend_profile as extend_suite
            expected = extend_suite(root, profile, expected)
        linux = root / "docs/provenance/d2_suite_linux_v1_evolution.json"
        if linux.exists():
            from chapter3_bser.experiments.d2_suite_v1.linux_provenance import extend_profile as extend_linux
            expected = extend_linux(root, profile, expected)
        linux_evaluation = root / "docs/provenance/d2_suite_linux_evaluation_v1_evolution.json"
        if linux_evaluation.exists():
            from chapter3_bser.experiments.d2_suite_v1.evaluation_provenance import extend_profile as extend_evaluation
            expected = extend_evaluation(root, profile, expected)
        pipeline = root / "docs/provenance/d2_suite_pipeline_v1_evolution.json"
        if pipeline.exists():
            from chapter3_bser.experiments.d2_suite_v1.pipeline_provenance import extend_profile as extend_pipeline
            expected = extend_pipeline(root, profile, expected)
        performance = root / "docs/provenance/d2_performance_v1_evolution.json"
        if performance.exists():
            from chapter3_bser.experiments.d2_performance.provenance import extend_profile as extend_performance
            expected = extend_performance(root, profile, expected)
        if current == dict(files=expected, sha256=digest(expected)):
            matched.append(profile)
    if matched:
        result = dict(experiment="ch3.safe_search.v1", checkout_profile=matched[0],
                    evolution_sha256=manifest["sha256"], inventory=current,
                    historical_record_count=27, historical_production_sha256=
                    read_json(root / PIN)["production"]["sha256"])
        if successor.exists():
            result["successor_evolution_sha256"] = file_sha256(successor)
        if effect.exists():
            result["bser_effect_evolution_sha256"] = file_sha256(effect)
        if final.exists():
            result["bser_final_evolution_sha256"] = file_sha256(final)
        if d2.exists():
            result["d2_evolution_sha256"] = file_sha256(d2)
        if suite.exists():
            result["d2_suite_evolution_sha256"] = file_sha256(suite)
        if linux.exists():
            result["d2_suite_linux_evolution_sha256"] = file_sha256(linux)
        if linux_evaluation.exists():
            result["d2_suite_linux_evaluation_sha256"] = file_sha256(linux_evaluation)
        if pipeline.exists():
            result["d2_suite_pipeline_evolution_sha256"] = file_sha256(pipeline)
        if performance.exists():
            result["d2_performance_evolution_sha256"] = file_sha256(performance)
        return result
    expected = manifest["profiles"]["windows_existing"]["files"]
    differences = sorted(n for n in current["files"].keys() | expected.keys()
                         if current["files"].get(n) != expected.get(n))
    raise ValueError("SafeSearch source inventory mismatch: " + ", ".join(differences))


def verify_sources(before, root=None):
    if before != framework_sources(root):
        raise ValueError("SafeSearch source changed during evaluation")


def build_reviewed_manifest(*, reviewed_new_paths, root=None):
    """Build a REVIEW CANDIDATE in memory; caller must review and seal it.

    This does not write a file or accept arbitrary production drift. It is a
    release operation after code review, never called by evaluation or tests.
    Explicit new paths plus the three declared hook paths are the only allowed
    changes to the original production and protected-baseline inventory.
    """
    root = ROOT if root is None else Path(root)
    prior = _prior_changes(root, _historical_profiles(root))
    current = inventory(root)["files"]
    new = set(reviewed_new_paths)
    if len(new) != len(reviewed_new_paths):
        raise ValueError("duplicate review paths")
    profiles = {}
    # Only the six original production representations and the two original
    # metadata representations differ. Runtime verification never normalizes.
    original = _historical_profiles(root)
    for profile, before in prior.items():
        expected = dict(current)
        if profile != "windows_existing":
            # Record the exact Git checkout representation of newly protected
            # JSON/Python/shell/plan inputs. This construction-time conversion
            # is reviewed in the sealed inventory; runtime never performs it.
            for name in expected.keys() - before.keys():
                data = (root / name).read_bytes()
                expected[name] = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        for name in original[profile]:
            if (name not in REVIEWED_HGR and name not in HOOK_PATHS
                    and original["git_lf"].get(name) != original["windows_existing"].get(name)):
                if current.get(name) not in {original[p].get(name) for p in original}:
                    raise ValueError("unreviewed source bytes: " + name)
                expected[name] = original[profile][name]
        # A Windows checkout uses retained historical JSON bytes; a Git/LF
        # checkout uses their exact LF file representations.
        for name, allowed in FROZEN_RAW.items():
            expected[name] = allowed[0] if profile == "windows_existing" else allowed[-1]
        changed = {n for n in before if expected.get(n) != before[n]}
        if changed != HOOK_PATHS:
            raise ValueError("review must contain exactly the default-equivalent hooks: " + str(sorted(changed)))
        if any(n in before or n not in expected for n in new):
            raise ValueError("reviewed new path is missing or already historical")
        production_additions = {n for n in expected if n.startswith(("core/", "chapter3_bser/", "tools/ch3_baselines/"))} - set(before)
        protected_additions = {n for n in expected if n.endswith(".sh") or n.startswith("configs/chapter3/baselines/")} - set(before)
        if (production_additions | protected_additions) - new:
            raise ValueError("new source/launcher missing explicit review")
        changes = {n: dict(before=before.get(n), after=expected[n]) for n in sorted(new | HOOK_PATHS)}
        extras = {n: sha for n, sha in expected.items() if n not in before and n not in new}
        profiles[profile] = dict(files=expected, sha256=digest(expected), safe_changes=changes,
                                 additional_inputs=extras)
    result = dict(schema=SCHEMA, review_status="candidate_requires_review",
                  authorization="2026-09-28 user request to implement safe-search repairs and bounded verification",
                  source_review_sha256=file_sha256(root / REVIEW), reviewed_hgr_changes=REVIEWED_HGR,
                  reviewed_existing_launcher=REVIEWED_LAUNCHER, changed_paths=sorted(HOOK_PATHS),
                  added_paths=sorted(new), profiles=profiles)
    result["sha256"] = digest(result)
    return result
