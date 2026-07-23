"""Import and independently verify the terminal HF spinodal Attempt 2b return.

This command is intentionally terminal-only and fail-closed.  It inspects the
one authorized Job once, snapshots the persisted bucket inventory before and
after download, verifies every content hash and provenance link, reruns the
independent raw-HDF5 audit, and only then atomically retains a hash-named local
copy.  It never starts, stops, resumes, or publishes anything.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PAPER_ID = "1KRpajnd6u"
JOB_ID = "6a5dd080bee6ee1cf4ed215e"
JOB_NAMESPACE = "DineshAI"
BUCKET_ID = "DineshAI/1KRpajnd6u-artifacts"
BUCKET_PREFIX = "hf-jobs/spinodal-attempt2b/full-v1"
REMOTE_RUN_ROOT = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}"
OFFICIAL_COMMIT = "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c"
EXISTING_SPACE = "DineshAI/1KRpajnd6u"
EXISTING_SPACE_SHA = "e367aa22ce98c5926ee41ac5a74a7a2fbf78f364"
FULL_SOURCE_MANIFEST_SHA256 = "f40896a9d8871f649b37d1318aaf7de76a89f2e2c4e31128cbc798ffffc6eff5"
RUNNER_SHA256 = "6e574948dcf9d50a8d21ecb62924d3d6ede82ca2d609490c2700abb44a1f9762"
AUDITOR_SHA256 = "ead541330615c12005252f551e95f840a4abf79f04e19c6311ee2e3b709ae44f"
DATASET_SHA256 = "d437b96f6c7f8b0b5ff4b530af0d1fd8b74f2e57a9422f59c5e008b0fb193b06"
IMAGE = "pytorch/pytorch@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9"
MODEL = "FluxNet_D_pf_100dt"
EXPECTED_COMMAND = "python /workspace/hf_jobs/bootstrap_spinodal_attempt2b_h5py.py"
EXPECTED_LABELS = {
    "paper": PAPER_ID,
    "purpose": "spinodal-attempt2b-full-v1",
    "attempt": "3",
    "name": "fluxnet-spinodal-attempt2b-full-v1",
}
EXPECTED_METRIC_KEYS = ("mean_final_mae", "maximum_relative_mass_drift")
EXPECTED_RATIO_KEYS = ("point", "lower_95", "upper_95")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    require(actual == expected, f"{label} mismatch: {actual!r} != {expected!r}")


def require_close(actual: Any, expected: Any, label: str, tolerance: float = 2.0e-7) -> None:
    require(
        math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance),
        f"{label} mismatch: {actual!r} != {expected!r}",
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def run_json(command: list[str]) -> Any:
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"command did not return JSON: {command!r}") from error


def unwrap_single_job(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        require(len(payload) == 1 and isinstance(payload[0], dict), "expected exactly one Job")
        return payload[0]
    require(isinstance(payload, dict), "Job inspection is not a JSON mapping")
    for key in ("job", "item"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def normalized_job_status(job: dict[str, Any]) -> str:
    status = job.get("status")
    if isinstance(status, dict):
        status = status.get("stage") or status.get("status")
    return str(status).upper()


def verify_terminal_job() -> dict[str, Any]:
    job = unwrap_single_job(
        run_json(["hf", "jobs", "inspect", JOB_ID, "--namespace", JOB_NAMESPACE, "--format", "json"])
    )
    observed_id = str(job.get("id") or job.get("job_id") or "").split("/")[-1]
    require_equal(observed_id, JOB_ID, "Job ID")
    require_equal(normalized_job_status(job), "COMPLETED", "Job terminal status")
    require_equal(job.get("flavor"), "t4-small", "Job flavor")
    docker_image = job.get("docker_image") or job.get("image")
    require_equal(docker_image, IMAGE, "Job image")
    command = job.get("command")
    command_text = " ".join(str(item) for item in command) if isinstance(command, list) else str(command)
    require_equal(command_text, EXPECTED_COMMAND, "Job command")
    labels = job.get("labels")
    require(isinstance(labels, dict), "Job labels are missing")
    for key, expected in EXPECTED_LABELS.items():
        require_equal(str(labels.get(key)), expected, f"Job label {key}")
    serialized = json.dumps(job, sort_keys=True)
    for token in (
        BUCKET_ID,
        "DineshAI/jobs-artifacts",
        "/artifacts",
        "/workspace/official",
        "/workspace/repro/src",
        "/workspace/docs",
        "/workspace/hf_jobs",
    ):
        require(token in serialized, f"Job inspection lacks expected mount token {token}")
    volumes = job.get("volumes")
    require(isinstance(volumes, list), "Job inspection lacks structured volume evidence")
    artifact_mounts = [
        volume
        for volume in volumes
        if isinstance(volume, dict)
        and volume.get("source") == BUCKET_ID
        and volume.get("mount_path") == "/artifacts"
    ]
    require(len(artifact_mounts) == 1, "expected exactly one persisted artifact mount")
    require(artifact_mounts[0].get("read_only") is False, "persisted artifact mount is not writable")
    return {
        "id": observed_id,
        "status": "COMPLETED",
        "flavor": job["flavor"],
        "image": docker_image,
        "command": command_text,
        "labels": {key: str(labels[key]) for key in sorted(EXPECTED_LABELS)},
    }


def inventory_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (payload[key] for key in ("items", "files", "results") if isinstance(payload.get(key), list)),
            None,
        )
        require(rows is not None, "bucket inventory has no recognized file list")
    else:
        raise RuntimeError("bucket inventory is not JSON list/mapping")
    require(all(isinstance(row, dict) for row in rows), "bucket inventory contains a non-mapping row")
    return rows


def relative_remote_path(value: str) -> str:
    normalized = value.removeprefix("hf://buckets/").lstrip("/")
    for prefix in (f"{BUCKET_ID}/{BUCKET_PREFIX}/", f"{BUCKET_PREFIX}/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    path = PurePosixPath(normalized)
    require(normalized not in {"", "."}, "empty bucket file path")
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe bucket path: {value}")
    return str(path)


def normalize_remote_inventory(payload: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in inventory_rows(payload):
        kind = str(row.get("type") or row.get("kind") or "").lower()
        if kind in {"directory", "dir", "folder"}:
            continue
        raw_path = row.get("path") or row.get("name") or row.get("key")
        size = row.get("size")
        if raw_path is None or size is None:
            continue
        item = {"path": relative_remote_path(str(raw_path)), "bytes": int(size)}
        for key in ("sha256", "xet_hash", "xetHash", "etag", "oid"):
            if row.get(key) is not None:
                item["remote_identity"] = str(row[key])
                break
        normalized.append(item)
    normalized.sort(key=lambda item: item["path"])
    require(normalized, "bucket inventory contains no files")
    paths = [item["path"] for item in normalized]
    require(len(paths) == len(set(paths)), "bucket inventory contains duplicate paths")
    return normalized


def bucket_inventory() -> list[dict[str, Any]]:
    return normalize_remote_inventory(
        run_json(["hf", "buckets", "list", f"{BUCKET_ID}/{BUCKET_PREFIX}", "--recursive", "--format", "json"])
    )


def safe_local_path(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    resolved = path.resolve(strict=False)
    require(resolved == root.resolve() or root.resolve() in resolved.parents, f"unsafe local path: {relative}")
    return path


def locate_synced_root(staging: Path) -> Path:
    candidates = [path.parent for path in staging.rglob("completion.json")]
    require(len(candidates) == 1, f"download has {len(candidates)} completion records")
    root = candidates[0]
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"download contains symlink: {path}")
        if path.is_file():
            require(root == path.parent or root in path.parents, f"download contains file outside run root: {path}")
    return root


def local_inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"import contains symlink: {path}")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    require(records, "downloaded run root is empty")
    return records


def compare_remote_and_local(remote: list[dict[str, Any]], local: list[dict[str, Any]]) -> None:
    remote_sizes = {row["path"]: row["bytes"] for row in remote}
    local_sizes = {row["path"]: row["bytes"] for row in local}
    require_equal(local_sizes, remote_sizes, "downloaded path/size inventory")


def expected_plan() -> list[dict[str, Any]]:
    plans = [
        {"split": "train", "seed": 12345, "steps": 52000, "save_start": 2000, "save_interval": 10},
        {"split": "val", "seed": 67890, "steps": 52000, "save_start": 2000, "save_interval": 10},
    ]
    plans.extend(
        {
            "split": "test",
            "seed": 22345 + 12345 * index,
            "steps": 102000,
            "save_start": 2000,
            "save_interval": 100,
        }
        for index in range(20)
    )
    for plan in plans:
        plan["frame_count"] = (plan["steps"] - plan["save_start"]) // plan["save_interval"] + 1
        plan["path"] = f"{plan['split']}/seed_{plan['seed']}.h5"
    return plans


def verify_local_source_pins() -> dict[str, str]:
    paths = {
        "full_source_manifest": ROOT / "hf_jobs/spinodal_attempt2b_full_sources.sha256",
        "frozen_runner": ROOT / "repro/src/hf_spinodal_attempt2b_full.py",
        "independent_auditor": ROOT / "repro/src/audit_spinodal_attempt2b.py",
    }
    expected = {
        "full_source_manifest": FULL_SOURCE_MANIFEST_SHA256,
        "frozen_runner": RUNNER_SHA256,
        "independent_auditor": AUDITOR_SHA256,
    }
    observed = {key: sha256_file(path) for key, path in paths.items()}
    require_equal(observed, expected, "local frozen source pins")
    return observed


def dataset_digest(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        digest.update(f"{entry['path']}\0{entry['sha256']}\0{entry['bytes']}\n".encode())
    return digest.hexdigest()


def require_finite_json(value: Any, label: str = "root") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite numeric value at {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json(item, f"{label}[{index}]")


def compare_scientific_summary(evaluation: dict[str, Any], audit: dict[str, Any]) -> None:
    for key in ("trajectory_count", "finite_trajectories", "verdict", "supports", "falsified"):
        require_equal(audit.get(key), evaluation.get(key), f"audit/evaluation {key}")
    for key in EXPECTED_METRIC_KEYS:
        require_close(audit[key], evaluation[key], f"audit/evaluation {key}")
    for key in EXPECTED_RATIO_KEYS:
        require_close(
            audit["radial_auc_ratio"][key],
            evaluation["radial_auc_ratio"][key],
            f"audit/evaluation radial ratio {key}",
        )
    for key in ("replicates", "seed"):
        require_equal(
            audit["radial_auc_ratio"][key],
            evaluation["radial_auc_ratio"][key],
            f"audit/evaluation radial ratio {key}",
        )


def verify_terminal_links(
    root: Path,
    completion: dict[str, Any],
    state: dict[str, Any],
    evaluation: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    require_equal(completion.get("status"), "complete", "completion status")
    require_equal(state.get("status"), "complete", "state status")
    require_equal(state.get("stage"), "complete", "state stage")
    require_equal(state.get("completion_sha256"), sha256_file(root / "completion.json"), "state completion hash")
    require_equal(state.get("audit_sha256"), sha256_file(root / "outputs/audit_certificate.json"), "state audit hash")
    require_equal(completion.get("dataset", {}).get("dataset_sha256"), DATASET_SHA256, "completion dataset digest")
    prediction_hashes = {
        str(entry["seed"]): entry["prediction_sha256"] for entry in evaluation.get("entries", [])
    }
    require_equal(completion.get("evaluation", {}).get("prediction_sha256"), prediction_hashes, "completion prediction hashes")
    require_equal(completion.get("evaluation", {}).get("verdict"), evaluation.get("verdict"), "completion evaluation verdict")
    require_equal(completion.get("audit", {}).get("verdict"), audit.get("verdict"), "completion audit verdict")
    require_equal(state.get("verdict"), evaluation.get("verdict"), "state verdict")
    compare_scientific_summary(evaluation, audit)


def verify_manifest(root: Path, completion: dict[str, Any]) -> dict[str, Any]:
    record = completion["dataset"]
    require_equal(record["manifest_path"], "data/manifest.json", "manifest path")
    path = safe_local_path(root, record["manifest_path"])
    require_equal(sha256_file(path), record["manifest_sha256"], "manifest SHA-256")
    manifest = load_json(path)
    plans = expected_plan()
    require_equal(manifest.get("dataset_sha256"), DATASET_SHA256, "fixed dataset SHA-256")
    require_equal(record["dataset_sha256"], DATASET_SHA256, "completion dataset SHA-256")
    entries = manifest.get("files")
    require(isinstance(entries, list), "manifest files is not a list")
    require_equal(len(entries), 22, "manifest trajectory count")
    require_equal(record["trajectory_count"], 22, "completion trajectory count")
    require_equal([entry.get("path") for entry in entries], [plan["path"] for plan in plans], "trajectory order")
    for entry, plan in zip(entries, plans):
        # The frozen schema-v1 producer writes the five persisted plan fields
        # plus the inspected HDF5 shape. ``frame_count`` is a derived property
        # of those fields, not a serialized manifest field.  Verify it from
        # the stored shape instead of requiring a field the producer never
        # writes; this remains stricter than accepting an unbound count.
        for key in ("split", "seed", "steps", "save_start", "save_interval", "path"):
            expected = plan[key]
            require_equal(entry.get(key), expected, f"manifest {plan['path']} {key}")
        expected_shape = [plan["frame_count"], 128, 128]
        require_equal(entry.get("shape"), expected_shape, f"manifest {plan['path']} shape")
        artifact = safe_local_path(root / "data", entry["path"])
        require(artifact.is_file(), f"missing trajectory {entry['path']}")
        require_equal(artifact.stat().st_size, int(entry["bytes"]), f"trajectory bytes {entry['path']}")
        require_equal(sha256_file(artifact), entry["sha256"], f"trajectory hash {entry['path']}")
        with h5py.File(artifact, "r") as handle:
            require_equal(
                list(handle["phi_data"].shape), expected_shape,
                f"trajectory HDF5 shape {entry['path']}",
            )
            require_equal(
                np.dtype(handle["phi_data"].dtype), np.dtype(np.float32),
                f"trajectory HDF5 dtype {entry['path']}",
            )
            expected_steps = np.arange(
                plan["save_start"], plan["steps"] + 1, plan["save_interval"], dtype=np.int64,
            )
            require(
                np.array_equal(handle["base_steps"][:], expected_steps),
                f"trajectory HDF5 time grid mismatch {entry['path']}",
            )
    require_equal(dataset_digest(entries), DATASET_SHA256, "aggregate dataset digest")
    require_equal(record["trajectory_sha256"], {entry["path"]: entry["sha256"] for entry in entries}, "completion trajectory hashes")
    return manifest


def safe_load_checkpoint(path: Path) -> dict[str, Any]:
    safe_numpy_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(safe_numpy_globals):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(checkpoint, dict), f"checkpoint is not a mapping: {path}")
    return checkpoint


def verify_training(root: Path, completion: dict[str, Any], provenance: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    record = completion["training"]
    expected_paths = {
        "latest_checkpoint": f"outputs/models/{MODEL}/latest_checkpoint.pt",
        "best_checkpoint": f"outputs/models/{MODEL}/best_checkpoint.pt",
        "history": f"outputs/models/{MODEL}/training_history.json",
    }
    for key, expected in expected_paths.items():
        require_equal(record[key], expected, f"training {key} path")
        digest_key = f"{key}_sha256" if key != "history" else "history_sha256"
        require_equal(sha256_file(safe_local_path(root, expected)), record[digest_key], f"training {key} hash")
    require_equal(record["completed_epochs"], 100, "completion epochs")
    history = load_json(safe_local_path(root, expected_paths["history"]))
    require(isinstance(history, list), "training history is not a list")
    require_equal([row.get("epoch") for row in history], list(range(1, 101)), "training epoch sequence")
    require_finite_json(history, "training_history")
    scientific_source_sha = {
        "released_generator": provenance["source_verification"]["scientific_source_sha256"]["official/dataset/spinodal_decomposition/phase_field_generator.cu"],
        "released_test_generator": provenance["source_verification"]["scientific_source_sha256"]["official/dataset/spinodal_decomposition/phase_field_generator_test.cu"],
        "released_model": provenance["source_verification"]["scientific_source_sha256"]["official/src/models/fluxnet_d_2d.py"],
        "released_dataloader": provenance["source_verification"]["scientific_source_sha256"]["official/src/training/dataloader.py"],
        "released_trainer": provenance["source_verification"]["scientific_source_sha256"]["official/src/training/trainer_unified.py"],
        "released_100dt_config": provenance["source_verification"]["scientific_source_sha256"]["official/experiments/spinodal_decomposition/run_single_seed_100dt.py"],
        "solver_port": provenance["source_verification"]["scientific_source_sha256"]["repro/src/spinodal_solver.cpp"],
        "attempt2b_wrapper": provenance["source_verification"]["scientific_source_sha256"]["repro/src/spinodal_attempt2b.py"],
        "preregistration": provenance["source_verification"]["scientific_source_sha256"]["docs/claim3_spinodal_attempt2b_preregistration.md"],
    }
    fingerprint = canonical_sha256(
        {
            "model": MODEL,
            "config": provenance["config"],
            "dataset_sha256": DATASET_SHA256,
            "source_sha256": scientific_source_sha,
            "official_commit": OFFICIAL_COMMIT,
        }
    )
    latest = safe_load_checkpoint(safe_local_path(root, expected_paths["latest_checkpoint"]))
    best = safe_load_checkpoint(safe_local_path(root, expected_paths["best_checkpoint"]))
    for label, checkpoint in (("latest", latest), ("best", best)):
        require_equal(checkpoint.get("schema_version"), 1, f"{label} checkpoint schema")
        require_equal(checkpoint.get("model_name"), MODEL, f"{label} checkpoint model")
        require_equal(checkpoint.get("fingerprint"), fingerprint, f"{label} checkpoint fingerprint")
        require_equal(checkpoint.get("config"), provenance["config"], f"{label} checkpoint config")
        require_equal(checkpoint.get("dataset_sha256"), DATASET_SHA256, f"{label} checkpoint dataset")
        require_equal(checkpoint.get("official_commit"), OFFICIAL_COMMIT, f"{label} checkpoint commit")
        require_equal(checkpoint.get("source_sha256"), scientific_source_sha, f"{label} checkpoint sources")
        require(checkpoint.get("mps_rng_state") is None, f"{label} CUDA checkpoint has MPS state")
    require_equal(latest.get("completed_epochs"), 100, "latest checkpoint epochs")
    require_equal(latest.get("history"), history, "latest checkpoint history")
    best_epoch = min(range(100), key=lambda index: float(history[index]["validation"]["total"])) + 1
    require_equal(best.get("completed_epochs"), best_epoch, "best checkpoint epoch")
    require_equal(best.get("history"), history[:best_epoch], "best checkpoint history")
    require_close(latest["best_validation_loss"], history[best_epoch - 1]["validation"]["total"], "best validation loss", 0.0)
    return {"fingerprint": fingerprint, "best_epoch": best_epoch, "completed_epochs": 100}


def verify_evaluation(root: Path, completion: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    record = completion["evaluation"]
    require_equal(record["path"], "outputs/evaluation.json", "evaluation path")
    path = safe_local_path(root, record["path"])
    require_equal(sha256_file(path), record["sha256"], "evaluation SHA-256")
    evaluation = load_json(path)
    require_finite_json(evaluation, "evaluation")
    require_equal(evaluation.get("dataset_sha256"), DATASET_SHA256, "evaluation dataset")
    require_equal(evaluation.get("trajectory_count"), 20, "evaluation trajectory count")
    require_equal(record["trajectory_count"], 20, "completion evaluation count")
    require_equal(evaluation.get("finite_trajectories"), 20, "finite trajectory count")
    require_equal(evaluation.get("claim_boundary"), "128x128 released-training-scale 100dt mechanism study; not literal v1 Table 5", "evaluation claim boundary")
    require_equal(evaluation.get("thresholds"), {"mass_drift": 1.0e-5, "mean_final_mae": 4.32e-2, "radial_ratio_upper_95_support": 1.25, "radial_ratio_lower_95_falsify": 1.25}, "evaluation thresholds")
    require(evaluation.get("verdict") in {"supports", "inconclusive", "falsified"}, "invalid evaluation verdict")
    require_equal(record["verdict"], evaluation["verdict"], "completion evaluation verdict")
    require_equal(evaluation.get("checkpoint"), f"models/{MODEL}/best_checkpoint.pt", "evaluation checkpoint path")
    best_path = safe_local_path(root / "outputs", evaluation["checkpoint"])
    latest_path = root / "outputs/models" / MODEL / "latest_checkpoint.pt"
    require_equal(sha256_file(best_path), evaluation["checkpoint_sha256"], "evaluation best checkpoint hash")
    require_equal(sha256_file(latest_path), evaluation["latest_checkpoint_sha256"], "evaluation latest checkpoint hash")
    plans = [plan for plan in expected_plan() if plan["split"] == "test"]
    entries = evaluation.get("entries")
    require(isinstance(entries, list) and len(entries) == 20, "evaluation entries are incomplete")
    prediction_hashes: dict[str, str] = {}
    manifest_by_path = {entry["path"]: entry for entry in manifest["files"]}
    for entry, plan in zip(entries, plans):
        seed = plan["seed"]
        require_equal(entry.get("seed"), seed, f"evaluation seed {seed}")
        require_equal(entry.get("source_path"), plan["path"], f"evaluation source {seed}")
        prediction_relative = f"predictions/seed_{seed}.h5"
        require_equal(entry.get("prediction_path"), prediction_relative, f"prediction path {seed}")
        require_equal(sha256_file(safe_local_path(root / "data", plan["path"])), manifest_by_path[plan["path"]]["sha256"], f"evaluation source hash {seed}")
        prediction_path = safe_local_path(root / "outputs", prediction_relative)
        require_equal(sha256_file(prediction_path), entry["prediction_sha256"], f"prediction hash {seed}")
        sidecar = root / "outputs/evaluation_entries" / f"seed_{seed}.json"
        require_equal(load_json(sidecar), entry, f"evaluation sidecar {seed}")
        prediction_hashes[str(seed)] = entry["prediction_sha256"]
    require_equal(record["prediction_sha256"], prediction_hashes, "completion prediction hashes")
    audit_record = completion["audit"]
    require_equal(audit_record["path"], "outputs/audit_certificate.json", "audit path")
    audit_path = safe_local_path(root, audit_record["path"])
    require_equal(sha256_file(audit_path), audit_record["sha256"], "audit SHA-256")
    audit = load_json(audit_path)
    require_finite_json(audit, "audit")
    require_equal(audit.get("audit_source_sha256"), AUDITOR_SHA256, "audit source SHA-256")
    require_equal(audit.get("official_commit"), OFFICIAL_COMMIT, "audit official commit")
    require_equal(audit.get("manifest_sha256"), completion["dataset"]["manifest_sha256"], "audit manifest hash")
    require_equal(audit.get("evaluation_sha256"), record["sha256"], "audit evaluation hash")
    require_equal(audit.get("checkpoint_sha256"), evaluation["checkpoint_sha256"], "audit checkpoint hash")
    require_equal(audit.get("prediction_sha256"), prediction_hashes, "audit prediction hashes")
    compare_scientific_summary(evaluation, audit)
    require_equal(audit_record["verdict"], evaluation["verdict"], "completion audit verdict")
    return evaluation, audit


def verify_chain(root: Path, completion: dict[str, Any], state: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    require_equal(completion.get("schema_version"), 1, "completion schema")
    require_equal(completion.get("paper_id"), PAPER_ID, "completion paper")
    require_equal(completion.get("official_commit"), OFFICIAL_COMMIT, "completion commit")
    contract = dict(provenance)
    contract_sha = contract.pop("contract_sha256", None)
    require_equal(canonical_sha256(contract), contract_sha, "provenance contract digest")
    require_equal(provenance.get("paper_id"), PAPER_ID, "provenance paper")
    require_equal(provenance.get("official_commit"), OFFICIAL_COMMIT, "provenance commit")
    require_equal(provenance.get("trajectory_plan"), expected_plan(), "provenance trajectory plan")
    require_equal(provenance.get("source_verification", {}).get("full_source_manifest_sha256"), FULL_SOURCE_MANIFEST_SHA256, "provenance full source manifest")
    require_equal(provenance.get("job", {}).get("flavor"), "t4-small", "provenance flavor")
    require_equal(provenance.get("job", {}).get("image"), IMAGE, "provenance image")
    require_equal(completion.get("contract_sha256"), contract_sha, "completion contract")
    require_equal(state.get("contract_sha256"), contract_sha, "state contract")
    require_equal(state.get("paper_id"), PAPER_ID, "state paper")
    recorded_evaluation = load_json(root / "outputs/evaluation.json")
    recorded_audit = load_json(root / "outputs/audit_certificate.json")
    verify_terminal_links(root, completion, state, recorded_evaluation, recorded_audit)
    manifest = verify_manifest(root, completion)
    training = verify_training(root, completion, provenance, manifest)
    evaluation, audit = verify_evaluation(root, completion, manifest)
    return {"training": training, "evaluation": evaluation, "audit": audit}


def rerun_independent_audit(root: Path) -> dict[str, Any]:
    auditor = ROOT / "repro/src/audit_spinodal_attempt2b.py"
    with tempfile.TemporaryDirectory(prefix="spinodal-reaudit-") as temporary:
        certificate = Path(temporary) / "certificate.json"
        subprocess.run(
            [
                sys.executable,
                str(auditor),
                "--data-dir",
                str(root / "data"),
                "--output-dir",
                str(root / "outputs"),
                "--certificate",
                str(certificate),
            ],
            check=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return load_json(certificate)


def compare_reaudit(recorded: dict[str, Any], recomputed: dict[str, Any]) -> None:
    for key in (
        "schema_version",
        "audit",
        "audit_source_sha256",
        "official_commit",
        "manifest_sha256",
        "evaluation_sha256",
        "checkpoint_sha256",
        "prediction_sha256",
        "trajectory_count",
        "finite_trajectories",
        "verdict",
        "supports",
        "falsified",
    ):
        require_equal(recomputed.get(key), recorded.get(key), f"local re-audit {key}")
    for key in EXPECTED_METRIC_KEYS:
        require_close(recomputed[key], recorded[key], f"local re-audit {key}")
    for key in (*EXPECTED_RATIO_KEYS, "replicates", "seed"):
        if key in ("replicates", "seed"):
            require_equal(recomputed["radial_auc_ratio"][key], recorded["radial_auc_ratio"][key], f"local re-audit radial ratio {key}")
        else:
            require_close(recomputed["radial_auc_ratio"][key], recorded["radial_auc_ratio"][key], f"local re-audit radial ratio {key}")


def exercise_negative_controls(root: Path, completion: dict[str, Any], state: dict[str, Any]) -> list[str]:
    controls = []
    evaluation = load_json(root / "outputs/evaluation.json")
    audit = load_json(root / "outputs/audit_certificate.json")
    mutations = {
        "nonterminal_completion": lambda c, s: c.__setitem__("status", "running"),
        "state_completion_hash": lambda c, s: s.__setitem__("completion_sha256", "0" * 64),
        "manifest_digest": lambda c, s: c["dataset"].__setitem__("dataset_sha256", "0" * 64),
        "prediction_hash": lambda c, s: c["evaluation"]["prediction_sha256"].__setitem__(next(iter(c["evaluation"]["prediction_sha256"])), "0" * 64),
        "verdict_flip": lambda c, s: c["evaluation"].__setitem__("verdict", "falsified" if c["evaluation"]["verdict"] != "falsified" else "supports"),
    }
    for name, mutate in mutations.items():
        changed_completion, changed_state = copy.deepcopy(completion), copy.deepcopy(state)
        mutate(changed_completion, changed_state)
        try:
            verify_terminal_links(root, changed_completion, changed_state, evaluation, audit)
        except (KeyError, RuntimeError, TypeError, ValueError):
            controls.append(name)
        else:
            raise RuntimeError(f"negative control was not detected: {name}")
    tampered_evaluation = copy.deepcopy(evaluation)
    tampered_evaluation["mean_final_mae"] = float(tampered_evaluation["mean_final_mae"]) + 1.0e-3
    try:
        compare_scientific_summary(tampered_evaluation, audit)
    except RuntimeError:
        controls.append("recorded_metric")
    else:
        raise RuntimeError("negative control was not detected: recorded_metric")
    return controls


def atomic_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite report: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        require(not path.exists(), f"report appeared concurrently: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--import-root",
        type=Path,
        default=ROOT / "outputs/hf-jobs/spinodal-attempt2b/verified-returns",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/hf-jobs/spinodal-attempt2b/terminal-import-verification.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_pins = verify_local_source_pins()
    job = verify_terminal_job()
    before = bucket_inventory()
    required = {"completion.json", "state.json", "provenance.json", "data/manifest.json", "outputs/evaluation.json", "outputs/audit_certificate.json"}
    require(required <= {row["path"] for row in before}, f"terminal bucket files missing: {sorted(required - {row['path'] for row in before})}")
    args.import_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    require(not args.report.exists(), f"refusing to overwrite report: {args.report}")
    with tempfile.TemporaryDirectory(prefix=".spinodal-import-", dir=args.import_root.parent) as temporary:
        staging = Path(temporary)
        subprocess.run(["hf", "buckets", "sync", REMOTE_RUN_ROOT, str(staging), "--format", "json"], check=True)
        after = bucket_inventory()
        require_equal(after, before, "bucket inventory changed during download")
        root = locate_synced_root(staging)
        inventory = local_inventory(root)
        compare_remote_and_local(before, inventory)
        completion = load_json(root / "completion.json")
        state = load_json(root / "state.json")
        provenance = load_json(root / "provenance.json")
        verified = verify_chain(root, completion, state, provenance)
        reaudited = rerun_independent_audit(root)
        compare_reaudit(verified["audit"], reaudited)
        controls = exercise_negative_controls(root, completion, state)
        snapshot_sha256 = canonical_sha256(inventory)
        destination = args.import_root / snapshot_sha256
        require(not destination.exists(), f"refusing to overwrite existing import: {destination}")
        os.replace(root, destination)
    evaluation = verified["evaluation"]
    report = {
        "schema_version": 1,
        "status": "verified",
        "paper_id": PAPER_ID,
        "job": job,
        "remote_run_root": REMOTE_RUN_ROOT,
        "remote_inventory_sha256": canonical_sha256(before),
        "local_snapshot_sha256": snapshot_sha256,
        "local_snapshot_path": str(destination),
        "file_count": len(inventory),
        "total_bytes": sum(row["bytes"] for row in inventory),
        "completion_sha256": sha256_file(destination / "completion.json"),
        "contract_sha256": provenance["contract_sha256"],
        "dataset_sha256": DATASET_SHA256,
        "source_pins": source_pins,
        "training": verified["training"],
        "result": {
            key: evaluation[key]
            for key in (
                "verdict",
                "supports",
                "falsified",
                "trajectory_count",
                "finite_trajectories",
                "mean_final_mae",
                "maximum_relative_mass_drift",
                "radial_auc_ratio",
                "thresholds",
                "claim_boundary",
            )
        },
        "independent_raw_hdf5_reaudit": "passed",
        "negative_controls_detected": controls,
        "publication_gate": {
            "publication_performed": False,
            "existing_space_only": EXISTING_SPACE,
            "expected_pre_update_space_sha": EXISTING_SPACE_SHA,
            "eligible_for_root_review": evaluation["verdict"] == "supports",
            "requires_official_validator": True,
            "requires_root_approval": True,
            "new_space_forbidden": True,
        },
    }
    atomic_json_new(args.report, report)
    print(json.dumps({"event": "spinodal_terminal_import_verified", "report": str(args.report), "report_sha256": sha256_file(args.report), "snapshot": str(destination), "snapshot_sha256": snapshot_sha256, "verdict": evaluation["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
