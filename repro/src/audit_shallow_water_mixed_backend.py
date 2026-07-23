"""Fail-closed audit of the mixed-backend shallow-water comparison.

This audit deliberately keeps the completed local FluxNet MPS evidence and the
verified returned FNO CUDA evidence in separate roots.  It rejects the partial
local FNO as comparison evidence, binds the Colab stdout digest through the
transport-verification report, rehashes the shared dataset locally and inside
the Colab input bundle, and independently recomputes both models' raw metrics
and the preregistered comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

from audit_shallow_water_attempt2 import (
    DATASET_SHA256,
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_CONFIG,
    EXPECTED_SPLIT_COUNTS,
    FIELD_NAMES,
    MODEL_NAMES,
    OFFICIAL_COMMIT,
    RAW_KEYS,
    SOURCE_RELATIVE_PATHS,
    SOURCE_SHA256,
    V1_PDF_SHA256,
    _assert_array_close,
    _assert_close,
    _assert_equal,
    _dataset_digest,
    _expected_test_identity,
    _validate_initial_condition,
    _validate_order,
    _validate_time_grid,
    array_sha256,
    atomic_json,
    calibrate_negative_controls,
    file_sha256_and_size,
    recompute,
    recompute_comparison,
)
from verify_colab_fno_return import (
    BUNDLED_MANIFEST_SHA256,
    BUNDLED_WRAPPER_SHA256,
    COLAB_BUNDLE_SHA256,
    canonical_sha256,
    parse_events,
    safe_load_checkpoint,
    validate_hex_sha256,
    verify_event_ledger,
)


FLUXNET = MODEL_NAMES[0]
FNO = MODEL_NAMES[1]


def hash_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def assert_separate_roots(local_output: Path, returned_fno: Path) -> None:
    if local_output == returned_fno:
        raise AssertionError("local MPS and returned CUDA evidence roots must be different")
    if local_output in returned_fno.parents or returned_fno in local_output.parents:
        raise AssertionError("one backend evidence root must not contain the other")


def verify_source_checkout(root: Path) -> dict[str, Any]:
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
    evidence = {}
    for name, relative in SOURCE_RELATIVE_PATHS.items():
        digest, size = file_sha256_and_size(official / relative)
        _assert_equal(SOURCE_SHA256[name], digest, f"official_source.{name}.sha256")
        evidence[name] = {"path": f"official/{relative}", "sha256": digest, "bytes": size}
    return {"official_commit": commit, "files": evidence}


def verify_bundle_and_local_dataset(
    root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    bundle_path: Path,
) -> dict[str, Any]:
    manifest_hash, manifest_bytes = file_sha256_and_size(manifest_path)
    _assert_equal(BUNDLED_MANIFEST_SHA256, manifest_hash, "local_manifest.sha256")
    _assert_equal(DATASET_SHA256, manifest["dataset_sha256"], "manifest.dataset_sha256")
    _assert_equal(EXPECTED_SPLIT_COUNTS, manifest["split_counts"], "manifest.split_counts")
    _assert_equal(EXPECTED_CATEGORY_COUNTS, manifest["category_counts"], "manifest.category_counts")
    if len(manifest["files"]) != 120:
        raise AssertionError(f"expected 120 dataset files, got {len(manifest['files'])}")
    _assert_equal(DATASET_SHA256, _dataset_digest(manifest["files"]), "manifest.aggregate_digest")

    local_files = {}
    for entry in manifest["files"]:
        path = manifest_path.parent / entry["path"]
        digest, size = file_sha256_and_size(path)
        _assert_equal(entry["sha256"], digest, f"local_dataset.{entry['path']}.sha256")
        _assert_equal(int(entry["bytes"]), size, f"local_dataset.{entry['path']}.bytes")
        local_files[entry["path"]] = {"sha256": digest, "bytes": size}

    bundle_hash, bundle_bytes = file_sha256_and_size(bundle_path)
    _assert_equal(COLAB_BUNDLE_SHA256, bundle_hash, "colab_input_bundle.sha256")
    expected_members = {
        "repro/src/shallow_water_attempt2.py": {
            "sha256": BUNDLED_WRAPPER_SHA256,
            "bytes": (root / "repro" / "src" / "shallow_water_attempt2.py").stat().st_size,
        },
        "data/shallow_water_attempt2/manifest.json": {
            "sha256": manifest_hash,
            "bytes": manifest_bytes,
        },
        **{
            f"data/shallow_water_attempt2/{entry['path']}": {
                "sha256": entry["sha256"],
                "bytes": int(entry["bytes"]),
            }
            for entry in manifest["files"]
        },
    }
    bundled_files = {}
    apple_double_files = []
    with tarfile.open(bundle_path, "r") as archive:
        regular_names = set()
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise AssertionError(f"unsafe Colab input bundle path: {member.name!r}")
            if not (member.isdir() or member.isreg()):
                raise AssertionError(f"links/special files forbidden in Colab input bundle: {member.name!r}")
            if member.isreg():
                if member.name in regular_names:
                    raise AssertionError(f"duplicate Colab input bundle member: {member.name}")
                regular_names.add(member.name)
        missing = set(expected_members) - regular_names
        if missing:
            raise AssertionError(
                f"Colab input bundle is missing required scientific files: {sorted(missing)}"
            )
        logical_targets = set(expected_members)
        for name in expected_members:
            parent = PurePosixPath(name).parent
            while parent.as_posix() != ".":
                logical_targets.add(parent.as_posix())
                parent = parent.parent
        extras = regular_names - set(expected_members)
        declared_apple_double_bytes = 0
        for name in extras:
            pure = PurePosixPath(name)
            if not pure.name.startswith("._"):
                raise AssertionError(f"unexpected non-AppleDouble Colab input bundle member: {name}")
            logical = pure.parent / pure.name[2:]
            if logical.as_posix() not in logical_targets:
                raise AssertionError(f"AppleDouble member has no expected logical target: {name}")
            member = archive.getmember(name)
            if int(member.size) > 1024 * 1024:
                raise AssertionError(f"AppleDouble member exceeds 1 MiB bound: {name}")
            declared_apple_double_bytes += int(member.size)
        if declared_apple_double_bytes > 16 * 1024 * 1024:
            raise AssertionError("AppleDouble metadata exceeds the fixed 16 MiB aggregate bound")
        for name, expected in expected_members.items():
            member = archive.getmember(name)
            stream = archive.extractfile(member)
            if stream is None:
                raise AssertionError(f"cannot read Colab input bundle member: {name}")
            digest, size = hash_stream(stream)
            _assert_equal(expected["sha256"], digest, f"colab_bundle.{name}.sha256")
            _assert_equal(expected["bytes"], size, f"colab_bundle.{name}.bytes")
            bundled_files[name] = {"sha256": digest, "bytes": size}
        for name in sorted(extras):
            member = archive.getmember(name)
            stream = archive.extractfile(member)
            if stream is None:
                raise AssertionError(f"cannot read AppleDouble bundle member: {name}")
            digest, size = hash_stream(stream)
            apple_double_files.append({"path": name, "sha256": digest, "bytes": size})

    wrapper_hash, wrapper_bytes = file_sha256_and_size(
        root / "repro" / "src" / "shallow_water_attempt2.py"
    )
    _assert_equal(BUNDLED_WRAPPER_SHA256, wrapper_hash, "local_wrapper.sha256")
    return {
        "dataset_sha256": DATASET_SHA256,
        "manifest": {
            "path": relative_to_root(manifest_path, root),
            "sha256": manifest_hash,
            "bytes": manifest_bytes,
        },
        "local_dataset_rehashed_files": len(local_files),
        "bundled_dataset_rehashed_files": len(manifest["files"]),
        "local_and_bundled_file_hashes_identical": all(
            local_files[entry["path"]]
            == bundled_files[f"data/shallow_water_attempt2/{entry['path']}"]
            for entry in manifest["files"]
        ),
        "colab_input_bundle": {
            "path": relative_to_root(bundle_path, root),
            "sha256": bundle_hash,
            "bytes": bundle_bytes,
            "scientific_regular_file_count": len(bundled_files),
            "appledouble_metadata": {
                "policy": (
                    "ignored as inert metadata only when the ._ companion maps to an expected file/directory; "
                    "all contents remain bound by the bundle hash"
                ),
                "regular_file_count": len(apple_double_files),
                "bytes": sum(item["bytes"] for item in apple_double_files),
                "inventory_sha256": canonical_sha256(apple_double_files),
            },
        },
        "wrapper": {
            "path": "repro/src/shallow_water_attempt2.py",
            "sha256": wrapper_hash,
            "bytes": wrapper_bytes,
        },
    }


def verify_transport_binding(
    root: Path,
    archive_path: Path,
    report_path: Path,
    returned_fno: Path,
    expected_archive_sha256: str,
    expected_archive_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(report_path.read_text())
    _assert_equal("passed", report.get("status"), "transport_report.status")
    _assert_equal(1, report.get("schema_version"), "transport_report.schema_version")
    transport = report["transport"]
    _assert_equal(
        expected_archive_sha256,
        transport["expected_sha256_from_colab_stdout"],
        "transport_report.expected_sha256_from_colab_stdout",
    )
    _assert_equal(expected_archive_sha256, transport["verified_sha256"], "transport_report.verified_sha256")
    _assert_equal(expected_archive_bytes, int(transport["bytes"]), "transport_report.bytes")
    archive_hash, archive_bytes = file_sha256_and_size(archive_path)
    _assert_equal(expected_archive_sha256, archive_hash, "returned_archive.sha256")
    _assert_equal(expected_archive_bytes, archive_bytes, "returned_archive.bytes")
    if returned_fno.name != expected_archive_sha256:
        raise AssertionError("verified FNO import directory is not named by the accepted archive SHA-256")
    if Path(report["verified_import"]).resolve() != returned_fno:
        raise AssertionError("transport report points to a different verified import")

    imported_report_path = returned_fno / "verification_report.json"
    report_hash, report_bytes = file_sha256_and_size(report_path)
    imported_report_hash, imported_report_bytes = file_sha256_and_size(imported_report_path)
    _assert_equal(report_hash, imported_report_hash, "imported_verification_report.sha256")
    _assert_equal(report_bytes, imported_report_bytes, "imported_verification_report.bytes")

    expected_members = {entry["path"]: entry for entry in report["archive_members"]}
    actual_files = {
        str(path.relative_to(returned_fno))
        for path in returned_fno.rglob("*")
        if path.is_file() and path.name != "verification_report.json"
    }
    if actual_files != set(expected_members):
        raise AssertionError(
            "verified import inventory differs from the transport report: "
            f"missing={sorted(set(expected_members) - actual_files)}, "
            f"extra={sorted(actual_files - set(expected_members))}"
        )
    for relative, expected in expected_members.items():
        digest, size = file_sha256_and_size(returned_fno / relative)
        _assert_equal(expected["sha256"], digest, f"verified_import.{relative}.sha256")
        _assert_equal(int(expected["bytes"]), size, f"verified_import.{relative}.bytes")

    _assert_equal(OFFICIAL_COMMIT, report["protocol"]["official_commit"], "transport_report.official_commit")
    _assert_equal(SOURCE_SHA256, report["protocol"]["source_sha256"], "transport_report.source_sha256")
    _assert_equal(DATASET_SHA256, report["protocol"]["dataset_sha256"], "transport_report.dataset_sha256")
    _assert_equal(EXPECTED_CONFIG, report["protocol"]["config"], "transport_report.config")
    return report, {
        "status": "passed",
        "stdout_binding": {
            "expected_archive_sha256": expected_archive_sha256,
            "expected_archive_bytes": expected_archive_bytes,
            "verified_archive_sha256": archive_hash,
            "verified_archive_bytes": archive_bytes,
        },
        "archive": {"path": relative_to_root(archive_path, root), "sha256": archive_hash, "bytes": archive_bytes},
        "transport_report": {
            "path": relative_to_root(report_path, root),
            "sha256": report_hash,
            "bytes": report_bytes,
        },
        "imported_transport_report": {
            "path": relative_to_root(imported_report_path, root),
            "sha256": imported_report_hash,
            "bytes": imported_report_bytes,
        },
        "verified_import": relative_to_root(returned_fno, root),
        "verified_member_count": len(expected_members),
    }


def verify_run_config(path: Path, expected_device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run_config = json.loads(path.read_text())
    _assert_equal(EXPECTED_CONFIG, run_config["config"], "run_config.config")
    _assert_equal(list(MODEL_NAMES), run_config["models"], "run_config.models")
    _assert_equal(OFFICIAL_COMMIT, run_config["official_commit"], "run_config.official_commit")
    _assert_equal(V1_PDF_SHA256, run_config["v1_pdf_sha256"], "run_config.v1_pdf_sha256")
    _assert_equal(SOURCE_SHA256, run_config["source_sha256"], "run_config.source_sha256")
    _assert_equal(expected_device, run_config["device"], "run_config.device")
    digest, size = file_sha256_and_size(path)
    return run_config, {"sha256": digest, "bytes": size, "device": expected_device}


def verify_local_fluxnet_events(path: Path) -> dict[str, Any]:
    events = parse_events(path)
    epochs = [
        int(event["epoch"])
        for event in events
        if event.get("event") == "training_epoch" and event.get("model") == FLUXNET
    ]
    if epochs != list(range(1, 101)):
        raise AssertionError("local FluxNet event ledger does not contain exactly epochs 1..100")
    completion_times = [
        float(event["unix_time"])
        for event in events
        if event.get("event") == "training_complete"
        and event.get("model") == FLUXNET
        and int(event.get("completed_epochs", -1)) == 100
    ]
    evaluation_times = [
        float(event["unix_time"])
        for event in events
        if event.get("event") == "evaluation_complete" and event.get("model") == FLUXNET
    ]
    if not completion_times:
        raise AssertionError("local FluxNet ledger lacks a 100-epoch completion event")
    if not evaluation_times or max(evaluation_times) <= max(completion_times):
        raise AssertionError("local FluxNet ledger lacks evaluation after completion")

    run_plans = []
    explicit_train_environment = False
    for line_number, event in enumerate(events, start=1):
        if event.get("event") != "run_plan" or FLUXNET not in event.get("models", []):
            continue
        if event.get("stage") not in ("train", "evaluate"):
            continue
        metadata = event.get("metadata", {})
        _assert_equal("mps", metadata.get("device"), f"local_events.{line_number}.device")
        _assert_equal(EXPECTED_CONFIG, metadata.get("config"), f"local_events.{line_number}.config")
        _assert_equal(OFFICIAL_COMMIT, metadata.get("official_commit"), f"local_events.{line_number}.official_commit")
        _assert_equal(SOURCE_SHA256, metadata.get("source_sha256"), f"local_events.{line_number}.source_sha256")
        environment = metadata.get("execution_environment")
        if event.get("stage") == "train" and environment is not None:
            _assert_equal("1", environment.get("OMP_NUM_THREADS"), f"local_events.{line_number}.OMP_NUM_THREADS")
            _assert_equal("1", environment.get("MKL_NUM_THREADS"), f"local_events.{line_number}.MKL_NUM_THREADS")
            _assert_equal(
                "0",
                environment.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                f"local_events.{line_number}.PYTORCH_ENABLE_MPS_FALLBACK",
            )
            explicit_train_environment = True
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
        raise AssertionError("local FluxNet ledger lacks an MPS train run_plan")
    if not any(plan["stage"] == "evaluate" for plan in run_plans):
        raise AssertionError("local FluxNet ledger lacks an MPS evaluation run_plan")
    if not explicit_train_environment:
        raise AssertionError("local FluxNet ledger lacks explicit single-thread, no-fallback MPS training metadata")
    digest, size = file_sha256_and_size(path)
    return {
        "sha256": digest,
        "bytes": size,
        "line_count": len(events),
        "training_epochs": 100,
        "training_complete_unix_time": max(completion_times),
        "evaluation_complete_unix_time": max(evaluation_times),
        "run_plans": run_plans,
        "environment_authority": "immutable run_plan metadata; mutable run_config environment is not used",
    }


def verify_checkpoint_evidence(
    model_root: Path,
    model: str,
    backend: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    model_dir = model_root / "models" / model
    latest_path = model_dir / "latest_checkpoint.pt"
    best_path = model_dir / "best_checkpoint.pt"
    history_path = model_dir / "training_history.json"
    latest = safe_load_checkpoint(latest_path)
    best = safe_load_checkpoint(best_path)
    history = json.loads(history_path.read_text())
    if [int(record["epoch"]) for record in history] != list(range(1, 101)):
        raise AssertionError(f"{model} training history is not exactly 100 contiguous epochs")
    _assert_close(history, latest["history"], f"checkpoint.{model}.latest.history")
    fingerprint = canonical_sha256(
        {
            "model": model,
            "config": EXPECTED_CONFIG,
            "dataset_sha256": DATASET_SHA256,
            "source_sha256": SOURCE_SHA256,
            "official_commit": OFFICIAL_COMMIT,
        }
    )
    for label, checkpoint in (("latest", latest), ("best", best)):
        _assert_equal(1, checkpoint["schema_version"], f"checkpoint.{model}.{label}.schema_version")
        _assert_equal(model, checkpoint["model_name"], f"checkpoint.{model}.{label}.model_name")
        _assert_equal(fingerprint, checkpoint["fingerprint"], f"checkpoint.{model}.{label}.fingerprint")
        _assert_equal(EXPECTED_CONFIG, checkpoint["config"], f"checkpoint.{model}.{label}.config")
        _assert_equal(DATASET_SHA256, checkpoint["dataset_sha256"], f"checkpoint.{model}.{label}.dataset_sha256")
        _assert_equal(OFFICIAL_COMMIT, checkpoint["official_commit"], f"checkpoint.{model}.{label}.official_commit")
        _assert_equal(SOURCE_SHA256, checkpoint["source_sha256"], f"checkpoint.{model}.{label}.source_sha256")
    _assert_equal(100, latest["completed_epochs"], f"checkpoint.{model}.latest.completed_epochs")
    best_epoch = min(range(100), key=lambda index: float(history[index]["validation"]["total"])) + 1
    _assert_equal(best_epoch, best["completed_epochs"], f"checkpoint.{model}.best.completed_epochs")
    _assert_equal(best_epoch, metrics["best_checkpoint_epoch"], f"metrics.{model}.best_checkpoint_epoch")
    best_loss = float(history[best_epoch - 1]["validation"]["total"])
    if not np.isclose(float(latest["best_validation_loss"]), best_loss, rtol=1e-12, atol=1e-14):
        raise AssertionError(f"{model} completion checkpoint best loss disagrees with history")

    if backend == "mps":
        if latest.get("mps_rng_state") is None or latest.get("cuda_rng_state_all") is not None:
            raise AssertionError("local FluxNet checkpoint does not carry exclusive MPS RNG provenance")
        rng_evidence = {"mps_rng_state_present": True, "cuda_rng_state_count": 0}
    elif backend == "cuda":
        cuda_states = latest.get("cuda_rng_state_all")
        if not cuda_states or latest.get("mps_rng_state") is not None:
            raise AssertionError("returned FNO checkpoint does not carry exclusive CUDA RNG provenance")
        rng_evidence = {"mps_rng_state_present": False, "cuda_rng_state_count": len(cuda_states)}
    else:
        raise AssertionError(f"unsupported backend: {backend}")

    latest_hash, latest_bytes = file_sha256_and_size(latest_path)
    best_hash, best_bytes = file_sha256_and_size(best_path)
    history_hash, history_bytes = file_sha256_and_size(history_path)
    _assert_equal(
        metrics["training_completion_checkpoint_sha256"],
        latest_hash,
        f"metrics.{model}.training_completion_checkpoint_sha256",
    )
    _assert_equal(metrics["checkpoint_sha256"], best_hash, f"metrics.{model}.checkpoint_sha256")
    return {
        "safe_load_mode": "torch weights_only with narrowly allowlisted NumPy RNG-state types",
        "fingerprint": fingerprint,
        "completed_epochs": 100,
        "best_epoch": best_epoch,
        "best_validation_total": best_loss,
        "rng_provenance": rng_evidence,
        "latest": {"sha256": latest_hash, "bytes": latest_bytes},
        "best": {"sha256": best_hash, "bytes": best_bytes},
        "history": {"sha256": history_hash, "bytes": history_bytes, "epochs": len(history)},
    }


def audit_raw_model(
    root: Path,
    model_root: Path,
    model: str,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_path = model_root / "raw" / f"{model}_rollouts.npz"
    metrics_path = model_root / "metrics" / f"{model}.json"
    metrics = json.loads(metrics_path.read_text())
    _assert_equal(model, metrics["model"], f"metrics.{model}.model")
    _assert_equal(EXPECTED_CONFIG, metrics["config"], f"metrics.{model}.config")
    _assert_equal(DATASET_SHA256, metrics["dataset_sha256"], f"metrics.{model}.dataset_sha256")
    _assert_equal(OFFICIAL_COMMIT, metrics["official_commit"], f"metrics.{model}.official_commit")
    _assert_equal(SOURCE_SHA256, metrics["source_sha256"], f"metrics.{model}.source_sha256")
    raw_hash, raw_bytes = file_sha256_and_size(raw_path)
    _assert_equal(raw_hash, metrics["raw_artifact_sha256"], f"metrics.{model}.raw_artifact_sha256")
    _assert_equal(raw_bytes, metrics["raw_artifact_bytes"], f"metrics.{model}.raw_artifact_bytes")

    expected_ids, expected_categories = _expected_test_identity(manifest)
    expected_identity_hash = canonical_sha256(
        {"sample_id": expected_ids, "category": expected_categories}
    )
    with np.load(raw_path, allow_pickle=False) as raw:
        if set(raw.files) != RAW_KEYS:
            raise AssertionError(f"unexpected raw keys for {model}: {sorted(raw.files)}")
        _assert_equal(model, str(raw["model"].item()), f"raw.{model}.model")
        sample_ids = [str(value) for value in raw["sample_id"].tolist()]
        categories = [str(value) for value in raw["category"].tolist()]
        _validate_order(sample_ids, expected_ids)
        _validate_order(categories, expected_categories)
        identity_hash = canonical_sha256({"sample_id": sample_ids, "category": categories})
        _assert_equal(expected_identity_hash, identity_hash, f"raw.{model}.test_identity_sha256")
        times = raw["times"]
        truth = raw["truth"]
        prediction = raw["prediction"]
        _validate_time_grid(times)
        _validate_initial_condition(prediction, truth)
        summary, arrays = recompute(prediction, truth, times)
        _assert_close(metrics["summary"], summary, f"metrics.{model}.summary")
        for key, recomputed_array in arrays.items():
            _assert_array_close(raw[key], recomputed_array, f"raw.{model}.{key}")
        category_array = np.asarray(categories)
        for category, expected_count in EXPECTED_CATEGORY_COUNTS["test"].items():
            selected = category_array == category
            _assert_equal(expected_count, int(selected.sum()), f"raw.{model}.category_count.{category}")
            case_summary, _ = recompute(
                prediction[selected], truth[selected], times, require_complete=False
            )
            _assert_close(metrics["per_case"][category], case_summary, f"metrics.{model}.per_case.{category}")
        truth_hash = array_sha256(truth)
        times_hash = array_sha256(times)

    metrics_hash, metrics_bytes = file_sha256_and_size(metrics_path)
    return metrics, {
        "raw": {
            "path": relative_to_root(raw_path, root),
            "sha256": raw_hash,
            "bytes": raw_bytes,
        },
        "metrics": {
            "path": relative_to_root(metrics_path, root),
            "sha256": metrics_hash,
            "bytes": metrics_bytes,
        },
        "test_identity_sha256": identity_hash,
        "sample_order_verified": True,
        "category_counts": EXPECTED_CATEGORY_COUNTS["test"],
        "times_array_sha256": times_hash,
        "truth_array_sha256": truth_hash,
        "independently_recomputed_summary": summary,
    }


def verify_partial_local_fno_excluded(root: Path, local_output: Path) -> dict[str, Any]:
    model_dir = local_output / "models" / FNO
    history_path = model_dir / "training_history.json"
    history = json.loads(history_path.read_text())
    epochs = [int(record["epoch"]) for record in history]
    if epochs != list(range(1, len(history) + 1)) or len(history) >= 100:
        raise AssertionError("local FNO exclusion requires a contiguous but incomplete local history")
    forbidden = [
        local_output / "raw" / f"{FNO}_rollouts.npz",
        local_output / "metrics" / f"{FNO}.json",
    ]
    present = [relative_to_root(path, root) for path in forbidden if path.exists()]
    if present:
        raise AssertionError(f"partial local FNO unexpectedly has result evidence: {present}")
    history_hash, history_bytes = file_sha256_and_size(history_path)
    latest_hash, latest_bytes = file_sha256_and_size(model_dir / "latest_checkpoint.pt")
    best_hash, best_bytes = file_sha256_and_size(model_dir / "best_checkpoint.pt")
    return {
        "status": "excluded",
        "reason": "local FNO training is incomplete and has no raw/metric result; only hash-named verified CUDA evidence is used",
        "completed_epochs": len(history),
        "target_epochs": 100,
        "history": {"path": relative_to_root(history_path, root), "sha256": history_hash, "bytes": history_bytes},
        "latest_checkpoint": {"sha256": latest_hash, "bytes": latest_bytes},
        "best_checkpoint": {"sha256": best_hash, "bytes": best_bytes},
        "raw_and_metrics_absent": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--verified-fno-dir", type=Path, required=True)
    parser.add_argument("--transport-report", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--colab-input-bundle", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-archive-bytes", type=int, required=True)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    started = time.time()
    root = Path(__file__).resolve().parents[2]
    local_output = args.local_output_dir.resolve()
    returned_fno = args.verified_fno_dir.resolve()
    report_path = args.transport_report.resolve()
    archive_path = args.archive.resolve()
    bundle_path = args.colab_input_bundle.resolve()
    manifest_path = (
        args.data_manifest or root / "data" / "shallow_water_attempt2" / "manifest.json"
    ).resolve()
    write_path = (args.write or local_output / "mixed_backend_audit.json").resolve()
    expected_archive_sha256 = validate_hex_sha256(
        args.expected_archive_sha256, "--expected-archive-sha256"
    )
    if args.expected_archive_bytes <= 0:
        parser.error("--expected-archive-bytes must be positive")
    for path, label in (
        (local_output, "--local-output-dir"),
        (returned_fno, "--verified-fno-dir"),
    ):
        if not path.is_dir():
            parser.error(f"{label} is not a directory: {path}")
    for path, label in (
        (report_path, "--transport-report"),
        (archive_path, "--archive"),
        (bundle_path, "--colab-input-bundle"),
        (manifest_path, "--data-manifest"),
    ):
        if not path.is_file():
            parser.error(f"{label} is not a file: {path}")

    assert_separate_roots(local_output, returned_fno)
    manifest = json.loads(manifest_path.read_text())
    source_evidence = verify_source_checkout(root)
    dataset_evidence = verify_bundle_and_local_dataset(root, manifest, manifest_path, bundle_path)
    transport_report, transport_evidence = verify_transport_binding(
        root,
        archive_path,
        report_path,
        returned_fno,
        expected_archive_sha256,
        args.expected_archive_bytes,
    )
    local_run_config, local_config_evidence = verify_run_config(
        local_output / "run_config.json", "mps"
    )
    returned_run_config, returned_config_evidence = verify_run_config(
        returned_fno / "run_config.json", "cuda"
    )
    _assert_equal(local_run_config["config"], returned_run_config["config"], "shared.config")
    _assert_equal(local_run_config["source_sha256"], returned_run_config["source_sha256"], "shared.source")
    _assert_equal(local_run_config["official_commit"], returned_run_config["official_commit"], "shared.commit")
    _assert_equal(local_run_config["v1_pdf_sha256"], returned_run_config["v1_pdf_sha256"], "shared.v1_pdf")

    local_event_evidence = verify_local_fluxnet_events(local_output / "events.jsonl")
    returned_events_path = returned_fno / "events.jsonl"
    returned_events = parse_events(returned_events_path)
    returned_event_evidence = verify_event_ledger(returned_events)
    returned_event_hash, returned_event_bytes = file_sha256_and_size(returned_events_path)
    _assert_equal(
        transport_report["event_ledger"]["sha256"],
        returned_event_hash,
        "transport_report.event_ledger.sha256",
    )
    _assert_equal(
        transport_report["event_ledger"]["line_count"],
        len(returned_events),
        "transport_report.event_ledger.line_count",
    )
    returned_event_evidence.update(
        {"sha256": returned_event_hash, "bytes": returned_event_bytes, "line_count": len(returned_events)}
    )

    local_metrics, local_raw_evidence = audit_raw_model(root, local_output, FLUXNET, manifest)
    returned_metrics, returned_raw_evidence = audit_raw_model(root, returned_fno, FNO, manifest)
    local_checkpoint_evidence = verify_checkpoint_evidence(
        local_output, FLUXNET, "mps", local_metrics
    )
    returned_checkpoint_evidence = verify_checkpoint_evidence(
        returned_fno, FNO, "cuda", returned_metrics
    )
    _assert_equal(
        transport_report["raw_evidence"]["truth_array_sha256"],
        returned_raw_evidence["truth_array_sha256"],
        "transport_report.raw_evidence.truth_array_sha256",
    )
    _assert_equal(
        transport_report["raw_evidence"]["independently_recomputed_summary"],
        returned_raw_evidence["independently_recomputed_summary"],
        "transport_report.raw_evidence.independently_recomputed_summary",
    )
    for identity_key in ("test_identity_sha256", "times_array_sha256", "truth_array_sha256"):
        _assert_equal(
            local_raw_evidence[identity_key],
            returned_raw_evidence[identity_key],
            f"shared_test_identity.{identity_key}",
        )

    comparison = recompute_comparison({FLUXNET: local_metrics, FNO: returned_metrics})
    partial_local_fno = verify_partial_local_fno_excluded(root, local_output)
    result = {
        "schema_version": 1,
        "status": "passed",
        "audit_scope": (
            "completed local FluxNet MPS and hash-named verified FNO CUDA raw/checkpoint evidence; "
            "mixed-backend provenance remains separate"
        ),
        "evidence_selection": {
            FLUXNET: {
                "role": "local completed FluxNet evidence only",
                "root": relative_to_root(local_output, root),
                "backend": "mps",
            },
            FNO: {
                "role": "verified returned FNO evidence only",
                "root": relative_to_root(returned_fno, root),
                "backend": "cuda",
            },
            "roots_are_separate": True,
            "partial_local_fno": partial_local_fno,
        },
        "transport_binding": transport_evidence,
        "shared_protocol_identity": {
            "status": "passed",
            "config": EXPECTED_CONFIG,
            "official_commit": OFFICIAL_COMMIT,
            "v1_pdf_sha256": V1_PDF_SHA256,
            "source_sha256": SOURCE_SHA256,
            "source_checkout": source_evidence,
            "dataset": dataset_evidence,
            "test_identity": {
                "test_identity_sha256": local_raw_evidence["test_identity_sha256"],
                "times_array_sha256": local_raw_evidence["times_array_sha256"],
                "truth_array_sha256": local_raw_evidence["truth_array_sha256"],
                "trajectory_count": 50,
                "rollout_steps": 120,
                "category_counts": EXPECTED_CATEGORY_COUNTS["test"],
                "identical_across_backends": True,
            },
        },
        "backend_evidence": {
            FLUXNET: {
                "backend": "mps",
                "run_config": local_config_evidence,
                "event_ledger": local_event_evidence,
                "checkpoint": local_checkpoint_evidence,
                "raw": local_raw_evidence,
            },
            FNO: {
                "backend": "cuda",
                "run_config": returned_config_evidence,
                "event_ledger": returned_event_evidence,
                "checkpoint": returned_checkpoint_evidence,
                "raw": returned_raw_evidence,
            },
        },
        "independently_recomputed_comparison": comparison,
        "negative_control_calibration": calibrate_negative_controls(),
        "interpretation": {
            "accuracy_result": (
                "FluxNet has lower final and late MAE than the FNO projection baseline in every field"
            ),
            "fixed_controlled_result": comparison["controlled_result"],
            "reason_if_inconclusive": (
                "the preregistered momentum relative-drift criterion uses a 1e-12 floor for trajectories "
                "whose initial momentum L1 norm is zero, so it fails despite small absolute drift"
            ),
            "paper_anchor_consistent": comparison["paper_anchor_consistent"],
        },
        "limitations": [
            "The two models were trained/evaluated on different hardware backends, so this is not a same-device timing or determinism comparison.",
            "The Colab wrapper proves CUDA selection but did not record the exact GPU model name.",
            "This is a single-seed controlled attempt and does not reproduce paper uncertainty estimates.",
            "The fixed momentum relative-drift statistic is undefined in scale when initial momentum is exactly zero; its preregistered 1e-12 floor is retained.",
            "Spinodal decomposition and paper runtime claims remain outside this evidence.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(write_path, result)
    result["written_to"] = str(write_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
