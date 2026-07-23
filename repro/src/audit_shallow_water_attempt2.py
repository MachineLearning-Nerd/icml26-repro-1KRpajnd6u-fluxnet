"""Independent evidence audit for shallow-water Attempt 2.

This module deliberately does not import the experiment driver's metric or
verdict code.  It can audit the fixed protocol while training is incomplete,
then bind the completed checkpoints, raw rollouts, metrics, and verdict into a
single hash-addressed evidence report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


FIELD_NAMES = ("h", "mx", "my")
MODEL_NAMES = ("FluxNet_SW_LAP_pf", "FNO_SW_Proj_box_mass_pf")
OFFICIAL_COMMIT = "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c"
V1_PDF_SHA256 = "40b79be23b629ec42d41eae9bdd578120e9a401d9e2dbef54011b5ba6ec2a285"
DATASET_SHA256 = "ee1d684d479f989f5588afbd40f493a58079ac9ed10dd97d03b16ad6725473cf"
SOURCE_SHA256 = {
    "generator": "04f632977d95d32e6b8f62a852c7d23ab6f0a49bb0cc463b113310fd51c2daca",
    "fluxnet_model": "49daf1e8be28f67b5aa983954aafa2cc434b420f1490329d869ed73694ad15d4",
    "projection_baseline_model": "0b8b285a064a530bbcb0420cb32e548c319eb835be2b236f4febf9478fd44745",
    "released_dataloader": "f9cd7d646eb3ad987441432b54df0572b5d02f6d0aecdf14e0afc2b0d2d98c9f",
    "released_trainer": "b47a7e0263226ad99e349b5103093410990555162df10e69ac4fed9d27b9cd90",
    "released_single_seed_config": "30051056b7534b438f52bcfd0120f19cc363257c6224f89b6ff0555ff1c5cda3",
}
SOURCE_RELATIVE_PATHS = {
    "generator": "dataset/shallow_water/dataset.py",
    "fluxnet_model": "src/models/fluxnet_sw_lap.py",
    "projection_baseline_model": "src/models/sw_baselines.py",
    "released_dataloader": "src/training/dataloader.py",
    "released_trainer": "src/training/trainer_unified.py",
    "released_single_seed_config": "experiments/shallow_water/run_single_seed.py",
}
EXPECTED_CONFIG = {
    "seed": 42,
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "unroll_steps": 5,
    "base_channels": 64,
    "num_blocks": 4,
    "kernel_size": 3,
    "neighborhood_size": 3,
    "fno_modes": 16,
    "fno_width": 64,
    "fno_layers": 4,
    "scheduler_patience": 15,
    "scheduler_factor": 0.5,
    "prediction_loss_weight": 0.5,
    "stability_loss_weight": 0.5,
    "train_horizon": 2.4,
    "test_horizon": 4.8,
    "divergence_threshold": 100.0,
    "mass_drift_threshold": 1e-5,
}
EXPECTED_SPLIT_COUNTS = {"train": 50, "val": 20, "test": 50}
EXPECTED_CATEGORY_COUNTS = {
    "train": {"CaseA1": 10, "CaseA2": 10, "CaseB1": 15, "CaseB2": 15},
    "val": {"CaseA1": 4, "CaseA2": 4, "CaseB1": 6, "CaseB2": 6},
    "test": {"CaseA1": 10, "CaseA2": 10, "CaseB1": 15, "CaseB2": 15},
}
RAW_KEYS = {
    "model",
    "sample_id",
    "category",
    "times",
    "truth",
    "prediction",
    "per_step_field_mae",
    "absolute_integral_drift",
    "relative_l1_normalized_integral_drift",
    "divergent",
}


def file_sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), size


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _assert_equal(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"expected mapping at {path}")
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(f"missing {path}.{key}")
            _assert_equal(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, float):
        if not math.isclose(expected, float(actual), rel_tol=1e-12, abs_tol=1e-14):
            raise AssertionError(f"value mismatch at {path}: expected={expected}, actual={actual}")
    elif expected != actual:
        raise AssertionError(f"value mismatch at {path}: expected={expected!r}, actual={actual!r}")


def _assert_close(expected: Any, actual: Any, path: str = "summary") -> None:
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise AssertionError(
                f"key mismatch at {path}: expected={sorted(expected)}, actual={sorted(actual)}"
            )
        for key, value in expected.items():
            _assert_close(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, float):
        if not np.isclose(expected, actual, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise AssertionError(f"metric mismatch at {path}: recorded={expected}, raw={actual}")
    elif expected != actual:
        raise AssertionError(f"metric mismatch at {path}: recorded={expected}, raw={actual}")


def _assert_array_close(recorded: np.ndarray, recomputed: np.ndarray, path: str) -> None:
    if recorded.shape != recomputed.shape:
        raise AssertionError(f"shape mismatch at {path}: {recorded.shape} != {recomputed.shape}")
    if not np.allclose(recorded, recomputed, rtol=1e-6, atol=1e-8, equal_nan=True):
        maximum = float(np.nanmax(np.abs(recorded.astype(np.float64) - recomputed.astype(np.float64))))
        raise AssertionError(f"array mismatch at {path}; maximum absolute difference={maximum}")


def _validate_time_grid(times: np.ndarray) -> None:
    expected = np.arange(121, dtype=np.float64) * 0.04
    if times.shape != (121,):
        raise AssertionError(f"expected 121 rollout times, got {times.shape}")
    if not np.all(np.diff(times) > 0.0):
        raise AssertionError("rollout times are not strictly increasing")
    if not np.allclose(times, expected, rtol=0.0, atol=1e-6):
        raise AssertionError(f"unexpected rollout time grid: endpoints={times[[0, -1]]}")


def _validate_initial_condition(prediction: np.ndarray, truth: np.ndarray) -> None:
    if not np.array_equal(prediction[:, 0], truth[:, 0]):
        raise AssertionError("prediction does not start from the exact reference initial condition")


def _validate_order(actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        mismatch = next(
            (index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
            min(len(actual), len(expected)),
        )
        raise AssertionError(f"trajectory identity/order mismatch at index {mismatch}")


def recompute(
    prediction: np.ndarray,
    truth: np.ndarray,
    times: np.ndarray,
    require_complete: bool = True,
    train_horizon: float = 2.4,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if prediction.shape != truth.shape or prediction.ndim != 5 or prediction.shape[2] != 3:
        raise AssertionError(f"unexpected raw shapes: prediction={prediction.shape}, truth={truth.shape}")
    if require_complete and prediction.shape != (50, 121, 3, 64, 64):
        raise AssertionError(f"expected 50 complete 120-step rollouts, got {prediction.shape}")
    if times.shape != (prediction.shape[1],):
        raise AssertionError(f"time length does not match rollout: {times.shape} vs {prediction.shape[1]}")
    if not np.isfinite(truth).all():
        raise AssertionError("non-finite reference data")
    absolute = np.abs(prediction - truth)
    per_step = absolute.mean(axis=(-2, -1))
    late = times > train_horizon + 1e-8
    if not late.any():
        raise AssertionError("late-horizon selection is empty")
    totals = prediction.sum(axis=(-2, -1), dtype=np.float64)
    drift = np.abs(totals - totals[:, :1])
    scale = np.abs(prediction[:, 0]).sum(axis=(-2, -1), dtype=np.float64)[:, None, :] + 1e-12
    relative = drift / scale
    h = prediction[:, 1:, 0]
    violations = h < 0.0
    count = int(violations.sum())
    divergent = (~np.isfinite(prediction).all(axis=(1, 2, 3, 4))) | (
        np.nanmax(np.abs(prediction), axis=(1, 2, 3, 4)) > 100.0
    )
    summary = {
        "trajectory_count": int(prediction.shape[0]),
        "rollout_steps": int(prediction.shape[1] - 1),
        "final_mae": {name: float(per_step[:, -1, index].mean()) for index, name in enumerate(FIELD_NAMES)},
        "final_mae_trajectory_std": {
            name: float(per_step[:, -1, index].std(ddof=1)) for index, name in enumerate(FIELD_NAMES)
        },
        "late_mae": {name: float(per_step[:, late, index].mean()) for index, name in enumerate(FIELD_NAMES)},
        "divergent_trajectories": int(divergent.sum()),
        "h_violation_rate_percent": float(violations.mean() * 100.0),
        "h_conditional_violation_magnitude": float(np.maximum(-h, 0.0).sum() / count) if count else 0.0,
        "prediction_h_min": float(np.nanmin(prediction[:, :, 0])),
        "max_absolute_integral_drift": {
            name: float(drift[:, :, index].max()) for index, name in enumerate(FIELD_NAMES)
        },
        "max_relative_l1_normalized_integral_drift": {
            name: float(relative[:, :, index].max()) for index, name in enumerate(FIELD_NAMES)
        },
    }
    arrays = {
        "per_step_field_mae": per_step.astype(np.float32),
        "absolute_integral_drift": drift.astype(np.float32),
        "relative_l1_normalized_integral_drift": relative.astype(np.float32),
        "divergent": divergent,
    }
    return summary, arrays


def _dataset_digest(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(files, key=lambda item: item["path"]):
        digest.update(f"{entry['path']}\0{entry['sha256']}\0{entry['bytes']}\n".encode())
    return digest.hexdigest()


def _expected_test_identity(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    entries = sorted(
        (entry for entry in manifest["files"] if entry["split"] == "test"),
        key=lambda item: (item["category"], int(item["sample_index"])),
    )
    sample_ids = [
        f"{entry['category']}:{int(entry['sample_index']):02d}:seed{int(entry['sample_seed'])}"
        for entry in entries
    ]
    categories = [entry["category"] for entry in entries]
    return sample_ids, categories


def audit_protocol(root: Path, output_dir: Path, manifest_path: Path, rehash_data: bool) -> dict[str, Any]:
    run_config_path = output_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    _assert_equal(EXPECTED_CONFIG, run_config["config"], "run_config.config")
    _assert_equal(list(MODEL_NAMES), run_config["models"], "run_config.models")
    _assert_equal(OFFICIAL_COMMIT, run_config["official_commit"], "run_config.official_commit")
    _assert_equal(V1_PDF_SHA256, run_config["v1_pdf_sha256"], "run_config.v1_pdf_sha256")
    _assert_equal(SOURCE_SHA256, run_config["source_sha256"], "run_config.source_sha256")

    official = root / "official"
    commit = subprocess.run(
        ["git", "-C", str(official), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_COMMIT:
        raise AssertionError(f"official checkout commit drift: {commit}")
    status = subprocess.run(
        ["git", "-C", str(official), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise AssertionError("official checkout is not clean")
    source_evidence = {}
    for name, relative in SOURCE_RELATIVE_PATHS.items():
        path = official / relative
        digest, size = file_sha256_and_size(path)
        if digest != SOURCE_SHA256[name]:
            raise AssertionError(f"pinned source hash drift for {name}: {digest}")
        source_evidence[name] = {"path": relative, "sha256": digest, "bytes": size}

    manifest = json.loads(manifest_path.read_text())
    _assert_equal(OFFICIAL_COMMIT, manifest["official_commit"], "manifest.official_commit")
    _assert_equal(V1_PDF_SHA256, manifest["v1_pdf_sha256"], "manifest.v1_pdf_sha256")
    _assert_equal(SOURCE_SHA256, manifest["source_sha256"], "manifest.source_sha256")
    _assert_equal(DATASET_SHA256, manifest["dataset_sha256"], "manifest.dataset_sha256")
    _assert_equal(EXPECTED_SPLIT_COUNTS, manifest["split_counts"], "manifest.split_counts")
    _assert_equal(EXPECTED_CATEGORY_COUNTS, manifest["category_counts"], "manifest.category_counts")
    if len(manifest["files"]) != 120:
        raise AssertionError(f"expected 120 manifest files, got {len(manifest['files'])}")
    identities: set[tuple[str, int, int]] = set()
    paths: set[str] = set()
    rehashed = 0
    for entry in manifest["files"]:
        identity = (entry["category"], int(entry["sample_index"]), int(entry["sample_seed"]))
        if identity in identities:
            raise AssertionError(f"duplicate trajectory identity: {identity}")
        identities.add(identity)
        if entry["path"] in paths:
            raise AssertionError(f"duplicate trajectory path: {entry['path']}")
        paths.add(entry["path"])
        expected_shape = [121, 64, 64] if entry["split"] == "test" else [61, 64, 64]
        expected_horizon = 4.8 if entry["split"] == "test" else 2.4
        _assert_equal(expected_shape, entry["shape"], f"manifest.files.{entry['path']}.shape")
        _assert_equal(expected_horizon, entry["horizon"], f"manifest.files.{entry['path']}.horizon")
        data_path = manifest_path.parent / entry["path"]
        if not data_path.is_file():
            raise AssertionError(f"missing trajectory file: {data_path}")
        if data_path.stat().st_size != int(entry["bytes"]):
            raise AssertionError(f"trajectory byte-size drift: {data_path}")
        if rehash_data:
            digest, size = file_sha256_and_size(data_path)
            if digest != entry["sha256"] or size != int(entry["bytes"]):
                raise AssertionError(f"trajectory hash/size drift: {data_path}")
            rehashed += 1
    if _dataset_digest(manifest["files"]) != DATASET_SHA256:
        raise AssertionError("aggregate dataset digest does not match the preregistered fingerprint")

    events = []
    for line_number, line in enumerate((output_dir / "events.jsonl").read_text().splitlines(), start=1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise AssertionError(f"invalid JSON event at line {line_number}") from error

    run_plan_environments = []
    explicit_training_environments = []
    for line_number, event in enumerate(events, start=1):
        if event.get("event") != "run_plan":
            continue
        metadata = event.get("metadata", {})
        environment = metadata.get("execution_environment")
        record = {
            "event_line": line_number,
            "unix_time": float(event["unix_time"]),
            "stage": event.get("stage"),
            "models": event.get("models"),
            "device": metadata.get("device"),
            "execution_environment": environment,
        }
        run_plan_environments.append(record)
        if event.get("stage") == "train" and environment is not None:
            _assert_equal("1", environment.get("OMP_NUM_THREADS"), f"events.{line_number}.OMP_NUM_THREADS")
            _assert_equal("1", environment.get("MKL_NUM_THREADS"), f"events.{line_number}.MKL_NUM_THREADS")
            _assert_equal(
                "0",
                environment.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                f"events.{line_number}.PYTORCH_ENABLE_MPS_FALLBACK",
            )
            explicit_training_environments.append(record)
    if not explicit_training_environments:
        raise AssertionError("no immutable train-stage run_plan records an explicit execution environment")
    authoritative_training_environment = explicit_training_environments[-1]
    mutable_environment = run_config.get("execution_environment")
    run_config_environment_clobbered = (
        mutable_environment != authoritative_training_environment["execution_environment"]
    )
    progress = {}
    histories = {}
    for model in MODEL_NAMES:
        history_path = output_dir / "models" / model / "training_history.json"
        history = json.loads(history_path.read_text()) if history_path.is_file() else []
        epochs = [int(record["epoch"]) for record in history]
        if epochs != list(range(1, len(history) + 1)):
            raise AssertionError(f"non-contiguous training history for {model}: {epochs[:3]}...{epochs[-3:]}")
        for record in history:
            for section in ("train", "validation"):
                if not all(math.isfinite(float(value)) for value in record[section].values()):
                    raise AssertionError(f"non-finite {section} loss in {model} epoch {record['epoch']}")
        event_epochs = [
            int(event["epoch"])
            for event in events
            if event.get("event") == "training_epoch" and event.get("model") == model
        ]
        if event_epochs != epochs:
            raise AssertionError(f"event/history epoch mismatch for {model}")
        progress[model] = {
            "completed_epochs": len(history),
            "target_epochs": 100,
            "state": "complete" if len(history) == 100 else "in_progress",
            "best_validation_total": min(
                (float(record["validation"]["total"]) for record in history), default=None
            ),
            "latest_epoch": history[-1] if history else None,
        }
        histories[model] = {
            "path": str(history_path.relative_to(root)),
            "sha256": file_sha256_and_size(history_path)[0] if history_path.is_file() else None,
        }

    completion_times = {
        model: [
            float(event["unix_time"])
            for event in events
            if event.get("event") == "training_complete" and event.get("model") == model
        ]
        for model in MODEL_NAMES
    }
    premature_evaluation_plans = []
    for line_number, event in enumerate(events, start=1):
        if event.get("event") != "run_plan" or event.get("stage") != "evaluate":
            continue
        planned_at = float(event["unix_time"])
        incomplete = [
            model for model in MODEL_NAMES if not any(completed <= planned_at for completed in completion_times[model])
        ]
        if incomplete:
            premature_evaluation_plans.append(
                {"event_line": line_number, "unix_time": planned_at, "models_incomplete": incomplete}
            )

    training_complete = all(item["state"] == "complete" for item in progress.values())
    final_blockers = []
    if not training_complete:
        final_blockers.append("both preregistered models do not yet have 100 contiguous epochs")

    return {
        "status": "passed",
        "audit_scope": "fixed protocol and current training ledger",
        "training_state": "complete" if training_complete else "in_progress",
        "final_evidence_eligible": training_complete,
        "final_evidence_blockers": final_blockers,
        "rehash_data_requested": rehash_data,
        "rehashed_trajectory_count": rehashed,
        "official_commit": commit,
        "source_evidence": source_evidence,
        "dataset": {
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": file_sha256_and_size(manifest_path)[0],
            "dataset_sha256": manifest["dataset_sha256"],
            "trajectory_count": len(manifest["files"]),
            "split_counts": manifest["split_counts"],
            "category_counts": manifest["category_counts"],
        },
        "run_config": {
            "path": str(run_config_path.relative_to(root)),
            "sha256": file_sha256_and_size(run_config_path)[0],
            "mutable_execution_environment_snapshot": mutable_environment,
            "authoritative_for_execution_environment": False,
        },
        "events": {
            "path": str((output_dir / "events.jsonl").relative_to(root)),
            "line_count_at_snapshot": len(events),
            "execution_environment_authority": "immutable run_plan records",
            "authoritative_training_environment": authoritative_training_environment,
            "run_plan_environments": run_plan_environments,
        },
        "training_progress": progress,
        "training_histories": histories,
        "ledger_anomalies": {
            "premature_evaluation_plans": premature_evaluation_plans,
            "premature_evaluation_plan_count": len(premature_evaluation_plans),
            "mutable_run_config_environment_clobbered_by_later_stage": run_config_environment_clobbered,
            "mutable_run_config_environment": mutable_environment,
            "recovered_training_environment": authoritative_training_environment["execution_environment"],
            "interpretation": (
                "recorded incomplete attempts are excluded from final evidence; final mode requires complete training"
            ),
        },
    }


def recompute_comparison(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flux = metrics[MODEL_NAMES[0]]["summary"]
    projection = metrics[MODEL_NAMES[1]]["summary"]
    lower_final = all(flux["final_mae"][field] < projection["final_mae"][field] for field in FIELD_NAMES)
    lower_late = all(flux["late_mae"][field] < projection["late_mae"][field] for field in FIELD_NAMES)
    mass_ok = all(
        flux["max_relative_l1_normalized_integral_drift"][field] <= 1e-5 for field in FIELD_NAMES
    )
    direct_support = (
        lower_final
        and lower_late
        and flux["divergent_trajectories"] == 0
        and flux["h_violation_rate_percent"] == 0.0
        and mass_ok
    )
    projection_wins = all(
        projection["final_mae"][field] <= flux["final_mae"][field]
        and projection["late_mae"][field] <= flux["late_mae"][field]
        for field in FIELD_NAMES
    )
    paper_flux = {"h": 3.12e-3, "mx": 4.41e-3, "my": 4.64e-3}
    paper_projection = {"h": 6.74e-3, "mx": 12.9e-3, "my": 11.5e-3}
    ratios = {field: flux["final_mae"][field] / projection["final_mae"][field] for field in FIELD_NAMES}
    anchor = all(flux["final_mae"][field] <= 2.0 * paper_flux[field] for field in FIELD_NAMES) and all(
        ratios[field] <= 2.0 * (paper_flux[field] / paper_projection[field]) for field in FIELD_NAMES
    )
    return {
        "paper_target": {
            "version": "arXiv:2602.01941v1",
            "table": 3,
            "fluxnet_lap_final_mae": paper_flux,
            "fno_box_mass_projection_final_mae": paper_projection,
        },
        "models": list(MODEL_NAMES),
        "observed_fluxnet_to_baseline_final_mae_ratio": ratios,
        "direct_support_criteria": {
            "lower_final_mae_all_fields": lower_final,
            "lower_late_mae_all_fields": lower_late,
            "zero_fluxnet_divergence": flux["divergent_trajectories"] == 0,
            "zero_fluxnet_h_violations": flux["h_violation_rate_percent"] == 0.0,
            "fluxnet_mass_drift_at_most_1e-5_all_fields": mass_ok,
        },
        "paper_anchor_consistent": anchor,
        "controlled_result": "supports" if direct_support else ("contradicts" if projection_wins else "inconclusive"),
        "challenge_assessment_ceiling": "partial C3 evidence; spinodal is reserved for a separate protocol",
    }


def audit_completed(output_dir: Path, root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected_ids, expected_categories = _expected_test_identity(manifest)
    reference_truth_sha256 = None
    reference_times = None
    metrics_by_model = {}
    model_evidence = {}
    for name in MODEL_NAMES:
        raw_path = output_dir / "raw" / f"{name}_rollouts.npz"
        metrics_path = output_dir / "metrics" / f"{name}.json"
        metrics = json.loads(metrics_path.read_text())
        _assert_equal(name, metrics["model"], f"metrics.{name}.model")
        _assert_equal(EXPECTED_CONFIG, metrics["config"], f"metrics.{name}.config")
        _assert_equal(DATASET_SHA256, metrics["dataset_sha256"], f"metrics.{name}.dataset_sha256")
        _assert_equal(OFFICIAL_COMMIT, metrics["official_commit"], f"metrics.{name}.official_commit")
        _assert_equal(SOURCE_SHA256, metrics["source_sha256"], f"metrics.{name}.source_sha256")
        if int(metrics["best_checkpoint_epoch"]) not in range(1, 101):
            raise AssertionError(f"invalid best checkpoint epoch for {name}")
        raw_hash, raw_bytes = file_sha256_and_size(raw_path)
        _assert_equal(raw_hash, metrics["raw_artifact_sha256"], f"metrics.{name}.raw_artifact_sha256")
        _assert_equal(raw_bytes, metrics["raw_artifact_bytes"], f"metrics.{name}.raw_artifact_bytes")
        best_path = output_dir / "models" / name / "best_checkpoint.pt"
        latest_path = output_dir / "models" / name / "latest_checkpoint.pt"
        _assert_equal(
            file_sha256_and_size(best_path)[0],
            metrics["checkpoint_sha256"],
            f"metrics.{name}.checkpoint_sha256",
        )
        _assert_equal(
            file_sha256_and_size(latest_path)[0],
            metrics["training_completion_checkpoint_sha256"],
            f"metrics.{name}.training_completion_checkpoint_sha256",
        )
        history = json.loads((output_dir / "models" / name / "training_history.json").read_text())
        if len(history) != 100 or [int(record["epoch"]) for record in history] != list(range(1, 101)):
            raise AssertionError(f"{name} does not have exactly 100 contiguous training epochs")
        expected_best_epoch = min(range(100), key=lambda index: float(history[index]["validation"]["total"])) + 1
        _assert_equal(expected_best_epoch, metrics["best_checkpoint_epoch"], f"metrics.{name}.best_checkpoint_epoch")

        with np.load(raw_path, allow_pickle=False) as raw:
            if set(raw.files) != RAW_KEYS:
                raise AssertionError(f"unexpected raw keys for {name}: {sorted(raw.files)}")
            if str(raw["model"].item()) != name:
                raise AssertionError(f"raw model identity mismatch for {name}")
            sample_ids = [str(value) for value in raw["sample_id"].tolist()]
            categories = [str(value) for value in raw["category"].tolist()]
            _validate_order(sample_ids, expected_ids)
            _validate_order(categories, expected_categories)
            times = raw["times"]
            _validate_time_grid(times)
            truth = raw["truth"]
            prediction = raw["prediction"]
            _validate_initial_condition(prediction, truth)
            summary, arrays = recompute(prediction, truth, times)
            _assert_close(metrics["summary"], summary)
            for key, recomputed in arrays.items():
                _assert_array_close(raw[key], recomputed, f"raw.{name}.{key}")
            truth_sha256 = array_sha256(truth)
            if reference_truth_sha256 is None:
                reference_truth_sha256 = truth_sha256
                reference_times = times.copy()
            else:
                if truth_sha256 != reference_truth_sha256:
                    raise AssertionError("models were evaluated against different truth arrays")
                if not np.array_equal(times, reference_times):
                    raise AssertionError("models were evaluated on different time grids")
            category_array = np.asarray(categories)
            for category, expected_count in EXPECTED_CATEGORY_COUNTS["test"].items():
                selected = category_array == category
                if int(selected.sum()) != expected_count:
                    raise AssertionError(f"unexpected raw count for {name}/{category}")
                case_summary, _ = recompute(
                    prediction[selected], truth[selected], times, require_complete=False
                )
                _assert_close(metrics["per_case"][category], case_summary, f"per_case.{name}.{category}")
        metrics_by_model[name] = metrics
        model_evidence[name] = {
            "raw_path": str(raw_path.relative_to(root)),
            "raw_sha256": raw_hash,
            "raw_bytes": raw_bytes,
            "metrics_path": str(metrics_path.relative_to(root)),
            "metrics_sha256": file_sha256_and_size(metrics_path)[0],
            "best_checkpoint_sha256": metrics["checkpoint_sha256"],
            "completion_checkpoint_sha256": metrics["training_completion_checkpoint_sha256"],
            "truth_array_sha256": truth_sha256,
            "recomputed_summary": summary,
        }

    recorded_comparison = json.loads((output_dir / "comparison.json").read_text())
    independent_comparison = recompute_comparison(metrics_by_model)
    _assert_close(recorded_comparison, independent_comparison, "comparison")
    return {
        "status": "passed",
        "audit_scope": "completed raw rollout, checkpoint, metric, ordering, and verdict evidence",
        "models": model_evidence,
        "shared_truth_array_sha256": reference_truth_sha256,
        "independently_recomputed_comparison": independent_comparison,
        "comparison_sha256": file_sha256_and_size(output_dir / "comparison.json")[0],
    }


def calibrate_negative_controls() -> dict[str, Any]:
    rng = np.random.default_rng(20260720)
    truth = rng.normal(size=(3, 6, 3, 2, 2)).astype(np.float32)
    prediction = truth.copy()
    prediction[:, 1:] += 0.01
    times = np.asarray([0.0, 1.0, 2.0, 2.41, 3.2, 4.8])
    summary, arrays = recompute(prediction, truth, times, require_complete=False)
    detected: dict[str, bool] = {}

    tampered = copy.deepcopy(summary)
    tampered["final_mae"]["h"] += 1e-3
    try:
        _assert_close(tampered, summary)
        detected["tampered_headline_metric"] = False
    except AssertionError:
        detected["tampered_headline_metric"] = True

    altered_array = arrays["per_step_field_mae"].copy()
    altered_array[0, 0, 0] += 1e-3
    try:
        _assert_array_close(altered_array, arrays["per_step_field_mae"], "negative_control")
        detected["tampered_raw_metric_array"] = False
    except AssertionError:
        detected["tampered_raw_metric_array"] = True

    try:
        _validate_initial_condition(prediction + 1e-4, truth)
        detected["altered_initial_condition"] = False
    except AssertionError:
        detected["altered_initial_condition"] = True

    try:
        _validate_order(["b", "a"], ["a", "b"])
        detected["swapped_trajectory_order"] = False
    except AssertionError:
        detected["swapped_trajectory_order"] = True

    bad_times = np.arange(121, dtype=np.float64) * 0.04
    bad_times[60] = bad_times[59]
    try:
        _validate_time_grid(bad_times)
        detected["nonmonotonic_time_grid"] = False
    except AssertionError:
        detected["nonmonotonic_time_grid"] = True

    if not all(detected.values()):
        raise AssertionError(f"negative-control calibration failed: {detected}")
    return {
        "status": "passed",
        "purpose": "validator self-calibration; these are synthetic research-integrity controls, not model tests",
        "seed": 20260720,
        "controls": detected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--mode", choices=("protocol", "final", "calibrate"), default="final")
    parser.add_argument(
        "--rehash-data",
        action="store_true",
        help="rehash all 120 HDF5 files; omitted in live protocol snapshots to avoid competing disk I/O",
    )
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    root = Path(__file__).resolve().parents[2]
    manifest_path = (args.data_manifest or root / "data" / "shallow_water_attempt2" / "manifest.json").resolve()
    started = time.time()

    if args.mode == "calibrate":
        result = {"negative_control_calibration": calibrate_negative_controls()}
    else:
        protocol = audit_protocol(root, output_dir, manifest_path, args.rehash_data)
        result = {
            "schema_version": 2,
            "status": "passed",
            "audit_mode": args.mode,
            "protocol": protocol,
            "negative_control_calibration": calibrate_negative_controls(),
        }
        if args.mode == "final":
            if protocol["training_state"] != "complete":
                raise AssertionError("final audit requires both preregistered models to complete 100 epochs")
            manifest = json.loads(manifest_path.read_text())
            result["completed_evidence"] = audit_completed(output_dir, root, manifest)
        result["elapsed_seconds"] = time.time() - started

    write_path = args.write
    if write_path is None and args.mode != "calibrate":
        write_path = output_dir / ("audit.json" if args.mode == "final" else "protocol_audit.json")
    if write_path is not None:
        atomic_json(write_path.resolve(), result)
        result["written_to"] = str(write_path.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
