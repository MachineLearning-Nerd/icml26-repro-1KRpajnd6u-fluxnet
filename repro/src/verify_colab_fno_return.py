"""Fail-closed verifier for the user-returned Colab FNO result archive.

The verifier never imports artifacts into the local result tree by default. It
checks the transport hash, archive paths, fixed protocol, checkpoint metadata,
immutable event ledger, and independently recomputed raw metrics first. An
optional import copies only the already-verified tree to a new hash-named path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from audit_shallow_water_attempt2 import (
    DATASET_SHA256,
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_CONFIG,
    FIELD_NAMES,
    OFFICIAL_COMMIT,
    RAW_KEYS,
    SOURCE_SHA256,
    V1_PDF_SHA256,
    _assert_array_close,
    _assert_close,
    _assert_equal,
    _expected_test_identity,
    _validate_initial_condition,
    _validate_order,
    _validate_time_grid,
    array_sha256,
    atomic_json,
    file_sha256_and_size,
    recompute,
)


MODEL = "FNO_SW_Proj_box_mass_pf"
ARCHIVE_ROOT = "shallow_water_attempt2_cuda"
COLAB_BUNDLE_SHA256 = "f717ebee4fa3ce5ce3783c5e6883f39a79199b6d60d0a3c9a5bf38960aa4e050"
BUNDLED_WRAPPER_SHA256 = "be256f981820568d6671e8e963c9c6eba36c785ba33895ff137d05cb6bbdbe43"
BUNDLED_MANIFEST_SHA256 = "ba9d0f6e4336b21ab00bcd29b3259ebdb860f1765b119286c5a4006c80ef9aab"
REQUIRED_FILES = {
    "run_config.json",
    "spinodal_boundary.json",
    "events.jsonl",
    f"models/{MODEL}/latest_checkpoint.pt",
    f"models/{MODEL}/best_checkpoint.pt",
    f"models/{MODEL}/training_history.json",
    f"raw/{MODEL}_rollouts.npz",
    f"metrics/{MODEL}.json",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_hex_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return normalized


def verify_local_colab_bundle(root: Path) -> dict[str, Any]:
    bundle_path = root / "outputs" / "colab" / "fluxnet_colab_fno_cuda_bundle.tar"
    bundle_hash, bundle_bytes = file_sha256_and_size(bundle_path)
    if bundle_hash != COLAB_BUNDLE_SHA256:
        raise AssertionError(f"local Colab bundle hash drift: {bundle_hash}")
    member_evidence = {}
    with tarfile.open(bundle_path, "r") as archive:
        for member_name, expected_hash in (
            ("repro/src/shallow_water_attempt2.py", BUNDLED_WRAPPER_SHA256),
            ("data/shallow_water_attempt2/manifest.json", BUNDLED_MANIFEST_SHA256),
        ):
            member = archive.getmember(member_name)
            stream = archive.extractfile(member)
            if stream is None:
                raise AssertionError(f"cannot read required bundle member {member_name}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            actual_hash = digest.hexdigest()
            if actual_hash != expected_hash:
                raise AssertionError(f"bundle member hash drift for {member_name}: {actual_hash}")
            member_evidence[member_name] = {"sha256": actual_hash, "bytes": size}
    return {
        "path": str(bundle_path.relative_to(root)),
        "sha256": bundle_hash,
        "bytes": bundle_bytes,
        "members": member_evidence,
    }


def safe_extract_result(archive_path: Path, destination: Path) -> tuple[Path, list[dict[str, Any]]]:
    regular_files = set()
    member_evidence = []
    declared_regular_bytes = 0
    allowed_directories = {""}
    for required in REQUIRED_FILES:
        parent = PurePosixPath(required).parent
        while parent.as_posix() != ".":
            allowed_directories.add(parent.as_posix())
            parent = parent.parent
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise AssertionError("result archive is empty")
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise AssertionError(f"unsafe archive path: {member.name!r}")
            if pure.parts[0] != ARCHIVE_ROOT:
                raise AssertionError(f"unexpected archive root: {member.name!r}")
            if not (member.isdir() or member.isreg()):
                raise AssertionError(f"links and special archive members are forbidden: {member.name!r}")
            relative = PurePosixPath(*pure.parts[1:])
            if member.isdir() and relative.as_posix() not in (".", *allowed_directories):
                raise AssertionError(f"unexpected result directory: {relative.as_posix()}")
            if member.isreg():
                relative_text = relative.as_posix()
                if relative_text in regular_files:
                    raise AssertionError(f"duplicate result file in archive: {relative_text}")
                regular_files.add(relative_text)
                if relative_text not in REQUIRED_FILES:
                    raise AssertionError(f"unexpected result file: {relative_text}")
                declared_regular_bytes += int(member.size)
                if int(member.size) > 2 * 1024**3 or declared_regular_bytes > 3 * 1024**3:
                    raise AssertionError("result archive exceeds the fixed 3 GiB extraction budget")
                source = archive.extractfile(member)
                if source is None:
                    raise AssertionError(f"cannot read archive member: {member.name}")
                target = destination / ARCHIVE_ROOT / Path(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with target.open("wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != int(member.size):
                    raise AssertionError(f"archive member size mismatch: {member.name}")
                member_evidence.append(
                    {"path": relative_text, "sha256": digest.hexdigest(), "bytes": size}
                )
        missing = REQUIRED_FILES - regular_files
        if missing:
            raise AssertionError(f"result archive is missing required files: {sorted(missing)}")
    return destination / ARCHIVE_ROOT, sorted(member_evidence, key=lambda item: item["path"])


def safe_load_checkpoint(path: Path) -> dict[str, Any]:
    safe_numpy_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(safe_numpy_globals):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise AssertionError(f"checkpoint is not a mapping: {path}")
    return checkpoint


def parse_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError(f"invalid JSON event at line {line_number}") from error
        if not isinstance(event, dict):
            raise AssertionError(f"event line {line_number} is not a mapping")
        events.append(event)
    return events


def verify_event_ledger(events: list[dict[str, Any]]) -> dict[str, Any]:
    training_epochs = [
        int(event["epoch"])
        for event in events
        if event.get("event") == "training_epoch" and event.get("model") == MODEL
    ]
    if training_epochs != list(range(1, 101)):
        raise AssertionError(f"expected exactly epochs 1..100 in event ledger, got {training_epochs}")
    completion_times = [
        float(event["unix_time"])
        for event in events
        if event.get("event") == "training_complete"
        and event.get("model") == MODEL
        and int(event.get("completed_epochs", -1)) == 100
    ]
    evaluation_events = [
        event for event in events if event.get("event") == "evaluation_complete" and event.get("model") == MODEL
    ]
    if not completion_times:
        raise AssertionError("no immutable FNO 100-epoch completion event")
    if not evaluation_events or float(evaluation_events[-1]["unix_time"]) <= min(completion_times):
        raise AssertionError("no FNO evaluation completion after 100-epoch training completion")
    resume_events = [
        event for event in events if event.get("event") == "checkpoint_resumed" and event.get("model") == MODEL
    ]
    if not resume_events:
        raise AssertionError("no checkpoint-resume evidence in Colab event ledger")

    run_plans = []
    for line_number, event in enumerate(events, start=1):
        if event.get("event") != "run_plan" or event.get("stage") not in ("train", "evaluate"):
            continue
        metadata = event.get("metadata", {})
        _assert_equal("cuda", metadata.get("device"), f"events.{line_number}.metadata.device")
        _assert_equal(EXPECTED_CONFIG, metadata.get("config"), f"events.{line_number}.metadata.config")
        _assert_equal(OFFICIAL_COMMIT, metadata.get("official_commit"), f"events.{line_number}.official_commit")
        _assert_equal(SOURCE_SHA256, metadata.get("source_sha256"), f"events.{line_number}.source_sha256")
        if event.get("models") != [MODEL]:
            raise AssertionError(f"unexpected selected models at event line {line_number}: {event.get('models')}")
        environment = metadata.get("execution_environment", {})
        _assert_equal("1", environment.get("OMP_NUM_THREADS"), f"events.{line_number}.OMP_NUM_THREADS")
        _assert_equal("1", environment.get("MKL_NUM_THREADS"), f"events.{line_number}.MKL_NUM_THREADS")
        run_plans.append(
            {
                "event_line": line_number,
                "stage": event["stage"],
                "unix_time": float(event["unix_time"]),
                "device": metadata["device"],
                "platform": metadata.get("platform"),
                "torch": metadata.get("torch"),
                "execution_environment": environment,
            }
        )
    if not any(plan["stage"] == "train" for plan in run_plans):
        raise AssertionError("no CUDA train run_plan")
    if not any(plan["stage"] == "evaluate" for plan in run_plans):
        raise AssertionError("no CUDA evaluation run_plan")
    return {
        "training_epochs": 100,
        "checkpoint_resume_event_count": len(resume_events),
        "training_complete_unix_time": max(completion_times),
        "evaluation_complete_unix_time": float(evaluation_events[-1]["unix_time"]),
        "run_plans": run_plans,
        "environment_authority": "immutable run_plan metadata; mutable run_config environment is not used",
        "gpu_model_limitation": "the wrapper records CUDA device selection but not torch.cuda.get_device_name",
    }


def verify_checkpoints(result_root: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    model_dir = result_root / "models" / MODEL
    latest_path = model_dir / "latest_checkpoint.pt"
    best_path = model_dir / "best_checkpoint.pt"
    history_path = model_dir / "training_history.json"
    latest = safe_load_checkpoint(latest_path)
    best = safe_load_checkpoint(best_path)
    history = json.loads(history_path.read_text())
    if [int(record["epoch"]) for record in history] != list(range(1, 101)):
        raise AssertionError("returned training history is not exactly 100 contiguous epochs")
    _assert_close(history, latest["history"], "checkpoint.latest.history")
    expected_fingerprint = canonical_sha256(
        {
            "model": MODEL,
            "config": EXPECTED_CONFIG,
            "dataset_sha256": DATASET_SHA256,
            "source_sha256": SOURCE_SHA256,
            "official_commit": OFFICIAL_COMMIT,
        }
    )
    for label, checkpoint in (("latest", latest), ("best", best)):
        _assert_equal(1, checkpoint["schema_version"], f"checkpoint.{label}.schema_version")
        _assert_equal(MODEL, checkpoint["model_name"], f"checkpoint.{label}.model_name")
        _assert_equal(EXPECTED_CONFIG, checkpoint["config"], f"checkpoint.{label}.config")
        _assert_equal(DATASET_SHA256, checkpoint["dataset_sha256"], f"checkpoint.{label}.dataset_sha256")
        _assert_equal(OFFICIAL_COMMIT, checkpoint["official_commit"], f"checkpoint.{label}.official_commit")
        _assert_equal(SOURCE_SHA256, checkpoint["source_sha256"], f"checkpoint.{label}.source_sha256")
        _assert_equal(expected_fingerprint, checkpoint["fingerprint"], f"checkpoint.{label}.fingerprint")
    _assert_equal(100, latest["completed_epochs"], "checkpoint.latest.completed_epochs")
    if not latest.get("cuda_rng_state_all"):
        raise AssertionError("completion checkpoint lacks CUDA RNG state")
    if latest.get("mps_rng_state") is not None:
        raise AssertionError("CUDA checkpoint unexpectedly contains MPS RNG state")
    expected_best_epoch = min(range(100), key=lambda index: float(history[index]["validation"]["total"])) + 1
    _assert_equal(expected_best_epoch, best["completed_epochs"], "checkpoint.best.completed_epochs")
    _assert_equal(expected_best_epoch, metrics["best_checkpoint_epoch"], "metrics.best_checkpoint_epoch")
    expected_best_loss = float(history[expected_best_epoch - 1]["validation"]["total"])
    if not np.isclose(float(latest["best_validation_loss"]), expected_best_loss, rtol=1e-12, atol=1e-14):
        raise AssertionError("completion checkpoint best loss does not match returned history")
    latest_hash, latest_bytes = file_sha256_and_size(latest_path)
    best_hash, best_bytes = file_sha256_and_size(best_path)
    _assert_equal(latest_hash, metrics["training_completion_checkpoint_sha256"], "metrics.latest_checkpoint_sha256")
    _assert_equal(best_hash, metrics["checkpoint_sha256"], "metrics.best_checkpoint_sha256")
    return {
        "fingerprint": expected_fingerprint,
        "completed_epochs": 100,
        "best_epoch": expected_best_epoch,
        "best_validation_total": expected_best_loss,
        "cuda_rng_state_count": len(latest["cuda_rng_state_all"]),
        "safe_load_mode": "torch weights_only with narrowly allowlisted NumPy RNG-state types",
        "latest": {"sha256": latest_hash, "bytes": latest_bytes},
        "best": {"sha256": best_hash, "bytes": best_bytes},
        "history": {
            "sha256": file_sha256_and_size(history_path)[0],
            "epochs": len(history),
        },
    }


def verify_raw_result(result_root: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_path = result_root / "raw" / f"{MODEL}_rollouts.npz"
    metrics_path = result_root / "metrics" / f"{MODEL}.json"
    metrics = json.loads(metrics_path.read_text())
    _assert_equal(MODEL, metrics["model"], "metrics.model")
    _assert_equal(EXPECTED_CONFIG, metrics["config"], "metrics.config")
    _assert_equal(DATASET_SHA256, metrics["dataset_sha256"], "metrics.dataset_sha256")
    _assert_equal(OFFICIAL_COMMIT, metrics["official_commit"], "metrics.official_commit")
    _assert_equal(SOURCE_SHA256, metrics["source_sha256"], "metrics.source_sha256")
    raw_hash, raw_bytes = file_sha256_and_size(raw_path)
    _assert_equal(raw_hash, metrics["raw_artifact_sha256"], "metrics.raw_artifact_sha256")
    _assert_equal(raw_bytes, metrics["raw_artifact_bytes"], "metrics.raw_artifact_bytes")
    expected_ids, expected_categories = _expected_test_identity(manifest)
    with np.load(raw_path, allow_pickle=False) as raw:
        if set(raw.files) != RAW_KEYS:
            raise AssertionError(f"unexpected raw keys: {sorted(raw.files)}")
        if str(raw["model"].item()) != MODEL:
            raise AssertionError("raw model identity mismatch")
        sample_ids = [str(value) for value in raw["sample_id"].tolist()]
        categories = [str(value) for value in raw["category"].tolist()]
        _validate_order(sample_ids, expected_ids)
        _validate_order(categories, expected_categories)
        times = raw["times"]
        truth = raw["truth"]
        prediction = raw["prediction"]
        _validate_time_grid(times)
        _validate_initial_condition(prediction, truth)
        summary, arrays = recompute(prediction, truth, times)
        _assert_close(metrics["summary"], summary)
        for key, recomputed in arrays.items():
            _assert_array_close(raw[key], recomputed, f"raw.{key}")
        category_array = np.asarray(categories)
        for category, expected_count in EXPECTED_CATEGORY_COUNTS["test"].items():
            selected = category_array == category
            if int(selected.sum()) != expected_count:
                raise AssertionError(f"unexpected category count for {category}")
            case_summary, _ = recompute(prediction[selected], truth[selected], times, require_complete=False)
            _assert_close(metrics["per_case"][category], case_summary, f"metrics.per_case.{category}")
        truth_hash = array_sha256(truth)
    return metrics, {
        "raw": {"sha256": raw_hash, "bytes": raw_bytes},
        "metrics": {"sha256": file_sha256_and_size(metrics_path)[0]},
        "truth_array_sha256": truth_hash,
        "independently_recomputed_summary": summary,
        "sample_order_verified": True,
        "category_counts": EXPECTED_CATEGORY_COUNTS["test"],
    }


def verify_return(archive_path: Path, expected_archive_sha256: str, root: Path) -> tuple[dict[str, Any], Path]:
    archive_hash, archive_bytes = file_sha256_and_size(archive_path)
    if archive_hash != expected_archive_sha256:
        raise AssertionError(
            f"downloaded archive hash mismatch: expected={expected_archive_sha256}, actual={archive_hash}"
        )
    bundle_evidence = verify_local_colab_bundle(root)
    manifest_path = root / "data" / "shallow_water_attempt2" / "manifest.json"
    manifest_hash, _ = file_sha256_and_size(manifest_path)
    if manifest_hash != BUNDLED_MANIFEST_SHA256:
        raise AssertionError(f"local dataset manifest hash drift: {manifest_hash}")
    manifest = json.loads(manifest_path.read_text())
    _assert_equal(DATASET_SHA256, manifest["dataset_sha256"], "manifest.dataset_sha256")

    temporary_context = tempfile.TemporaryDirectory(prefix="fluxnet-colab-verify-")
    temporary = Path(temporary_context.name)
    result_root, member_evidence = safe_extract_result(archive_path, temporary)
    run_config_path = result_root / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    _assert_equal(EXPECTED_CONFIG, run_config["config"], "run_config.config")
    _assert_equal("cuda", run_config["device"], "run_config.device")
    _assert_equal(OFFICIAL_COMMIT, run_config["official_commit"], "run_config.official_commit")
    _assert_equal(V1_PDF_SHA256, run_config["v1_pdf_sha256"], "run_config.v1_pdf_sha256")
    _assert_equal(SOURCE_SHA256, run_config["source_sha256"], "run_config.source_sha256")
    events_path = result_root / "events.jsonl"
    events = parse_events(events_path)
    event_evidence = verify_event_ledger(events)
    metrics, raw_evidence = verify_raw_result(result_root, manifest)
    checkpoint_evidence = verify_checkpoints(result_root, metrics)
    report = {
        "schema_version": 1,
        "status": "passed",
        "verification_boundary": (
            "artifact provenance and numerical evidence verified; GPU model name requires executed-notebook output"
        ),
        "transport": {
            "archive_path": str(archive_path),
            "expected_sha256_from_colab_stdout": expected_archive_sha256,
            "verified_sha256": archive_hash,
            "bytes": archive_bytes,
        },
        "colab_input_bundle": bundle_evidence,
        "archive_members": member_evidence,
        "protocol": {
            "official_commit": OFFICIAL_COMMIT,
            "source_sha256": SOURCE_SHA256,
            "dataset_sha256": DATASET_SHA256,
            "dataset_manifest_sha256": manifest_hash,
            "bundled_wrapper_sha256": BUNDLED_WRAPPER_SHA256,
            "config": EXPECTED_CONFIG,
        },
        "run_config": {
            "sha256": file_sha256_and_size(run_config_path)[0],
            "device": run_config["device"],
            "mutable_execution_environment_not_used_as_authority": run_config.get("execution_environment"),
        },
        "event_ledger": {
            "sha256": file_sha256_and_size(events_path)[0],
            "line_count": len(events),
            **event_evidence,
        },
        "checkpoints": checkpoint_evidence,
        "raw_evidence": raw_evidence,
    }
    report_path = temporary / "verification_report.json"
    atomic_json(report_path, report)
    report["_temporary_context"] = temporary_context
    return report, result_root


def import_verified(result_root: Path, import_root: Path, archive_sha256: str) -> Path:
    import_root.mkdir(parents=True, exist_ok=True)
    destination = import_root / archive_sha256
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing verified import: {destination}")
    temporary = import_root / f".{archive_sha256}.{os.getpid()}.tmp"
    if temporary.exists():
        raise FileExistsError(f"temporary import path already exists: {temporary}")
    shutil.copytree(result_root, temporary)
    os.replace(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--expected-archive-sha256",
        required=True,
        help="SHA-256 printed by the final Colab packaging cell; binds the download to the Colab output",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--import-root",
        type=Path,
        help="optional destination parent; verified files are copied to a new SHA-named child and never overwrite",
    )
    args = parser.parse_args()
    archive_path = args.archive.resolve()
    if not archive_path.is_file():
        parser.error(f"archive does not exist: {archive_path}")
    expected_hash = validate_hex_sha256(args.expected_archive_sha256, "--expected-archive-sha256")
    root = Path(__file__).resolve().parents[2]
    report, result_root = verify_return(archive_path, expected_hash, root)
    temporary_context = report.pop("_temporary_context")
    try:
        if args.import_root is not None:
            imported = import_verified(result_root, args.import_root.resolve(), expected_hash)
            report["verified_import"] = str(imported)
            atomic_json(imported / "verification_report.json", report)
        if args.report is not None:
            atomic_json(args.report.resolve(), report)
            report["report_written_to"] = str(args.report.resolve())
        print(json.dumps(report, sort_keys=True))
    finally:
        temporary_context.cleanup()


if __name__ == "__main__":
    main()
