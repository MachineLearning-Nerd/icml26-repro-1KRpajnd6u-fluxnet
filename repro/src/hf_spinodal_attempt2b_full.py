"""Resumable T4 execution of the preregistered spinodal Attempt 2b protocol.

The mounted Hugging Face bucket is the source of truth. Generation files,
epoch checkpoints, evaluation trajectories, and the independent audit are
committed atomically. Relaunching the same command verifies and reuses every
complete artifact; incompatible or damaged artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

import audit_spinodal_attempt2b as independent_audit
import hf_spinodal_attempt2b_preflight as preflight
import spinodal_attempt2b as experiment


PAPER_ID = "1KRpajnd6u"
OFFICIAL_COMMIT = "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c"
PREFLIGHT_SHA256 = "ebc2bf33922783a4a06b1a48d02716d15be3740be67e4e0e6403146eb5f1df3e"
PREFLIGHT_SOURCE_MANIFEST_SHA256 = (
    "1288b2e67a76d2dfa09b5526c2e9ae91d765d2e38bb1a4980afda0a64bf1b029"
)
T4_RATE_USD_PER_HOUR = 0.40
JOB_TIMEOUT_HOURS = 12.0
INTERNAL_DEADLINE_HOURS = 11.5
JOB_WORST_CASE_COST_USD = JOB_TIMEOUT_HOURS * T4_RATE_USD_PER_HOUR
AUTHORIZED_CAMPAIGN_CAP_USD = 40.00
CUDA_MEMORY_GATE_BYTES = 12 * 1024**3
EXPECTED_BUFFERED_HOURS = 7.608320569313277
EXPECTED_BUFFERED_COST_USD = 3.0433282277253113
EXPECTED_PEAK_RESERVED_BYTES = 3_135_242_240
GRID = 128
H5PY_VERSION = "3.14.0"
H5PY_WHEEL_SHA256 = "723a40ee6505bd354bfd26385f2dae7bbfa87655f4e61bab175a49d72ebfc06b"
H5PY_WHEEL_BYTES = 4_516_618
H5PY_WHEEL_PATH = (
    "hf_jobs/wheels/"
    "h5py-3.14.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)

FULL_SOURCE_PATHS = (
    "repro/src/spinodal_attempt2b.py",
    "repro/src/spinodal_solver.cpp",
    "repro/src/audit_spinodal_attempt2b.py",
    "repro/src/hf_spinodal_attempt2b_preflight.py",
    "repro/src/hf_spinodal_attempt2b_full.py",
    "docs/claim3_spinodal_attempt2b_preregistration.md",
    "official/dataset/spinodal_decomposition/phase_field_generator.cu",
    "official/dataset/spinodal_decomposition/phase_field_generator_test.cu",
    "official/src/models/fluxnet_d_2d.py",
    "official/src/training/dataloader.py",
    "official/src/training/trainer_unified.py",
    "official/experiments/spinodal_decomposition/run_single_seed_100dt.py",
    "hf_jobs/bootstrap_spinodal_attempt2b_h5py.py",
    "hf_jobs/launch_spinodal_attempt2b_preflight.sh",
    "hf_jobs/launch_spinodal_attempt2b_full.sh",
    "hf_jobs/spinodal_attempt2b_h5py_requirements.txt",
    "hf_jobs/wheels/h5py-3.14.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        value /= 1024.0
    return value / 1024.0


def atomic_json_new(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    payload = json.dumps(value, indent=2, sort_keys=True, default=str).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"immutable artifact appeared concurrently: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def atomic_json_replace(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_hash_manifest(path: Path) -> tuple[dict[str, str], str]:
    records: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise RuntimeError(f"invalid full source manifest line {number}")
        digest, relative = fields[0].lower(), fields[1].lstrip("*")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in records:
            raise RuntimeError(f"unsafe or duplicate source path: {relative}")
        int(digest, 16)
        records[relative] = digest
    if set(records) != set(FULL_SOURCE_PATHS):
        raise RuntimeError(
            "full source inventory mismatch: "
            f"missing={sorted(set(FULL_SOURCE_PATHS) - set(records))}, "
            f"extra={sorted(set(records) - set(FULL_SOURCE_PATHS))}"
        )
    return records, sha256_file(path)


def verify_full_sources(
    source_root: Path,
    full_manifest_path: Path,
    preflight_manifest_path: Path,
    preflight_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scientific = preflight.verify_sources(source_root, preflight_manifest_path)
    if scientific["manifest_sha256"] != PREFLIGHT_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("preflight source manifest hash mismatch")
    expected, full_manifest_sha256 = load_hash_manifest(full_manifest_path)
    observed = {}
    for relative, digest in sorted(expected.items()):
        path = source_root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"full mounted source hash mismatch: {relative}")
        observed[relative] = digest
    if sha256_file(preflight_report_path) != PREFLIGHT_SHA256:
        raise RuntimeError("returned preflight report hash mismatch")
    report = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or not report.get("gates", {}).get(
        "eligible_for_separately_authorized_full_campaign"
    ):
        raise RuntimeError("returned preflight did not pass every gate")
    cuda = report.get("cuda_runtime", {})
    if "T4" not in str(cuda.get("name", "")).upper():
        raise RuntimeError("preflight was not measured on an NVIDIA T4")
    peak = int(report.get("cuda_benchmark", {}).get("peak_cuda_reserved_bytes", -1))
    if peak != EXPECTED_PEAK_RESERVED_BYTES or peak > CUDA_MEMORY_GATE_BYTES:
        raise RuntimeError("preflight CUDA memory result mismatch")
    projection = report.get("projection", {})
    if not math.isclose(
        float(projection.get("total_buffered_hours", math.nan)),
        EXPECTED_BUFFERED_HOURS,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        float(projection.get("estimated_full_campaign_cost_usd", math.nan)),
        EXPECTED_BUFFERED_COST_USD,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("preflight projection mismatch")
    if report.get("source_verification", {}).get("source_sha256") != scientific["source_sha256"]:
        raise RuntimeError("preflight report source inventory mismatch")
    return {
        "official_commit": OFFICIAL_COMMIT,
        "preflight_sha256": PREFLIGHT_SHA256,
        "preflight_source_manifest_sha256": PREFLIGHT_SOURCE_MANIFEST_SHA256,
        "full_source_manifest_sha256": full_manifest_sha256,
        "source_sha256": observed,
        "scientific_source_sha256": scientific["source_sha256"],
        "preflight_gpu": cuda,
        "preflight_peak_reserved_bytes": peak,
        "preflight_projection": projection,
    }, report


def scientific_provenance(verification: dict[str, Any]) -> dict[str, Any]:
    hashes = verification["scientific_source_sha256"]
    return {
        "official_commit": OFFICIAL_COMMIT,
        "source_sha256": {
            "released_generator": hashes[
                "official/dataset/spinodal_decomposition/phase_field_generator.cu"
            ],
            "released_test_generator": hashes[
                "official/dataset/spinodal_decomposition/phase_field_generator_test.cu"
            ],
            "released_model": hashes["official/src/models/fluxnet_d_2d.py"],
            "released_dataloader": hashes["official/src/training/dataloader.py"],
            "released_trainer": hashes["official/src/training/trainer_unified.py"],
            "released_100dt_config": hashes[
                "official/experiments/spinodal_decomposition/run_single_seed_100dt.py"
            ],
            "solver_port": hashes["repro/src/spinodal_solver.cpp"],
            "attempt2b_wrapper": hashes["repro/src/spinodal_attempt2b.py"],
            "preregistration": hashes["docs/claim3_spinodal_attempt2b_preregistration.md"],
        },
    }


def compile_solver_linux(output_path: Path) -> dict[str, Any]:
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required for the preregistered CPU generator")
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-ffp-contract=off",
        "-fno-fast-math",
        str(experiment.SOLVER_SOURCE),
        "-o",
        str(output_path),
    ]
    command_sha256 = experiment.canonical_sha256(command)
    metadata_path = output_path.with_suffix(".build.json")
    if output_path.is_file() and metadata_path.is_file():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            recorded.get("command_sha256") == command_sha256
            and recorded.get("solver_source_sha256") == sha256_file(experiment.SOLVER_SOURCE)
            and recorded.get("binary_sha256") == sha256_file(output_path)
        ):
            return recorded
        raise RuntimeError("refusing incompatible persisted solver build")
    orphaned_output = output_path.is_file() and not metadata_path.exists()
    if (output_path.exists() and not orphaned_output) or metadata_path.exists():
        raise RuntimeError("incomplete persisted solver build state")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    compiler_version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    compile_command = [*command[:-1], str(temporary)]
    try:
        subprocess.run(compile_command, check=True, capture_output=True, text=True, timeout=300)
        os.chmod(temporary, 0o755)
        temporary_sha256 = sha256_file(temporary)
        if orphaned_output:
            if sha256_file(output_path) != temporary_sha256:
                raise RuntimeError("orphaned solver binary does not match a fresh pinned build")
        elif output_path.exists():
            raise RuntimeError("solver binary appeared concurrently")
        else:
            os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    record = {
        "command": command,
        "command_sha256": command_sha256,
        "compiler_version": compiler_version,
        "solver_source_sha256": sha256_file(experiment.SOLVER_SOURCE),
        "binary_sha256": sha256_file(output_path),
    }
    atomic_json_new(metadata_path, record)
    return record


def set_cuda_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def require_t4() -> dict[str, Any]:
    if h5py.__version__ != H5PY_VERSION:
        raise RuntimeError(f"pinned h5py {H5PY_VERSION} required, found {h5py.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("full campaign requires exactly one CUDA device")
    properties = torch.cuda.get_device_properties(0)
    if "T4" not in properties.name.upper():
        raise RuntimeError(f"T4-only authorization violated: {properties.name}")
    if properties.total_memory < 14 * 1024**3 or properties.total_memory > 16 * 1024**3:
        raise RuntimeError(f"unexpected T4 memory size: {properties.total_memory}")
    query = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    smi = subprocess.run(query, check=True, capture_output=True, text=True, timeout=30)
    return {
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "h5py": h5py.__version__,
        "numpy": np.__version__,
        "nvidia_smi_csv": smi.stdout.strip(),
    }


def update_state(run_root: Path, contract_sha256: str, **values: Any) -> None:
    path = run_root / "state.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("contract_sha256") != contract_sha256:
            raise RuntimeError("persisted state belongs to another execution contract")
    else:
        state = {
            "schema_version": 1,
            "paper_id": PAPER_ID,
            "contract_sha256": contract_sha256,
            "created_at": utc_now(),
        }
    state.update(values)
    state["updated_at"] = utc_now()
    atomic_json_replace(path, state)


def create_or_verify_provenance(
    run_root: Path,
    verification: dict[str, Any],
    cuda: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = {
        "schema_version": 1,
        "paper_id": PAPER_ID,
        "attempt": "claim3_spinodal_attempt2b",
        "claim_boundary": "128x128 released-training-scale 100dt mechanism study",
        "official_commit": OFFICIAL_COMMIT,
        "config": asdict(experiment.RunConfig()),
        "trajectory_plan": [
            {**asdict(item), "frame_count": item.frame_count, "path": str(item.relative_path)}
            for item in experiment.build_plan()
        ],
        "source_verification": verification,
        "preflight_expected_buffered_hours": EXPECTED_BUFFERED_HOURS,
        "preflight_expected_buffered_cost_usd": EXPECTED_BUFFERED_COST_USD,
        "job": {
            "flavor": "t4-small",
            "timeout_hours": JOB_TIMEOUT_HOURS,
            "internal_deadline_hours": INTERNAL_DEADLINE_HOURS,
            "rate_usd_per_hour": T4_RATE_USD_PER_HOUR,
            "worst_case_cost_usd": JOB_WORST_CASE_COST_USD,
            "authorized_campaign_cap_usd": AUTHORIZED_CAMPAIGN_CAP_USD,
            "image": "pytorch/pytorch@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9",
        },
        "paths": {
            "run_root": str(run_root),
            "data": str(run_root / "data"),
            "outputs": str(run_root / "outputs"),
        },
        "runtime_packages": {key: cuda[key] for key in ("torch", "torch_cuda", "cudnn", "h5py", "numpy")},
        "dependency_bootstrap": {
            "isolation": "pip --no-index --no-deps --require-hashes --target /tmp/fluxnet-h5py-3.14.0",
            "h5py_version": H5PY_VERSION,
            "requirements": "hf_jobs/spinodal_attempt2b_h5py_requirements.txt",
            "wheel": H5PY_WHEEL_PATH,
            "wheel_bytes": H5PY_WHEEL_BYTES,
            "wheel_sha256": H5PY_WHEEL_SHA256,
        },
    }
    contract_sha256 = canonical_sha256(contract)
    document = {**contract, "contract_sha256": contract_sha256}
    path = run_root / "provenance.json"
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != document or sha256_file(path) != sha256_bytes(document):
            raise RuntimeError("persisted immutable provenance mismatch")
    else:
        atomic_json_new(path, document)
    return document, contract_sha256


def sha256_bytes(value: Any) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, default=str).encode() + b"\n"
    return hashlib.sha256(payload).hexdigest()


def write_once_or_verify(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"persisted immutable JSON mismatch: {path}")
    else:
        atomic_json_new(path, value)


def reconcile_training_artifacts(
    config: Any,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    model_dir = output_dir / "models/FluxNet_D_pf_100dt"
    latest_path = model_dir / "latest_checkpoint.pt"
    if not latest_path.exists():
        return None
    latest = experiment.load_training_checkpoint(latest_path)
    fingerprint = experiment.checkpoint_fingerprint(config, manifest, provenance)
    if latest.get("fingerprint") != fingerprint:
        raise RuntimeError("persisted latest checkpoint fingerprint mismatch")
    completed_epochs = int(latest.get("completed_epochs", -1))
    history = latest.get("history")
    if not isinstance(history, list) or len(history) != completed_epochs or completed_epochs < 1:
        raise RuntimeError("latest checkpoint has an inconsistent training history")
    if [int(record.get("epoch", -1)) for record in history] != list(
        range(1, completed_epochs + 1)
    ):
        raise RuntimeError("latest checkpoint epoch sequence is invalid")
    best_record = min(history, key=lambda record: float(record["validation"]["total"]))
    expected_best_epoch = int(best_record["epoch"])
    expected_best_loss = float(best_record["validation"]["total"])
    if not math.isclose(
        float(latest["best_validation_loss"]),
        expected_best_loss,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("latest checkpoint best-validation loss is inconsistent")
    best_path = model_dir / "best_checkpoint.pt"
    best_matches = False
    if best_path.exists():
        best = experiment.load_training_checkpoint(best_path)
        best_matches = bool(
            best.get("fingerprint") == fingerprint
            and int(best.get("completed_epochs", -1)) == expected_best_epoch
            and math.isclose(
                float(best.get("best_validation_loss", math.nan)),
                expected_best_loss,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
    if not best_matches:
        if expected_best_epoch != completed_epochs or not bool(history[-1].get("is_best")):
            raise RuntimeError("best checkpoint is incompatible and cannot be safely reconstructed")
        experiment.atomic_torch_save(best_path, latest)
    history_path = model_dir / "training_history.json"
    history_matches = False
    if history_path.exists():
        try:
            history_matches = json.loads(history_path.read_text(encoding="utf-8")) == history
        except (OSError, json.JSONDecodeError):
            history_matches = False
    if not history_matches:
        experiment.atomic_json(history_path, history)
    return latest


def summary_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise RuntimeError(f"evaluation entry missing {key}")
        actual_value = actual[key]
        if isinstance(expected_value, bool):
            if bool(actual_value) != expected_value:
                raise RuntimeError(f"evaluation entry {key} mismatch")
        elif isinstance(expected_value, (float, int)):
            if not math.isclose(float(actual_value), float(expected_value), rel_tol=2.0e-7, abs_tol=2.0e-7):
                raise RuntimeError(f"evaluation entry {key} mismatch")
        elif actual_value != expected_value:
            raise RuntimeError(f"evaluation entry {key} mismatch")


def verify_manifest_trajectory_hashes(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    plans = experiment.build_plan()
    expected_paths = {str(plan.relative_path) for plan in plans}
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(plans):
        raise RuntimeError("manifest trajectory inventory is incomplete")
    entries = {entry.get("path"): entry for entry in files}
    if len(entries) != len(files) or set(entries) != expected_paths:
        raise RuntimeError("manifest trajectory paths do not match the fixed plan")
    for relative, entry in entries.items():
        path = manifest_path.parent / relative
        if not path.is_file():
            raise RuntimeError(f"manifested trajectory is missing: {relative}")
        if path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"manifested trajectory size mismatch: {relative}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"manifested trajectory hash mismatch: {relative}")
    if experiment.dataset_digest(files) != manifest.get("dataset_sha256"):
        raise RuntimeError("manifest aggregate dataset hash mismatch")
    return entries


def verify_evaluation_artifact_hashes(
    manifest_path: Path,
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    output_dir: Path,
    *,
    require_sidecars: bool,
) -> None:
    entries_by_path = verify_manifest_trajectory_hashes(manifest_path, manifest)
    if evaluation.get("dataset_sha256") != manifest.get("dataset_sha256"):
        raise RuntimeError("evaluation dataset fingerprint mismatch")
    expected_checkpoint = Path("models/FluxNet_D_pf_100dt/best_checkpoint.pt")
    expected_latest = Path("models/FluxNet_D_pf_100dt/latest_checkpoint.pt")
    if evaluation.get("checkpoint") != str(expected_checkpoint):
        raise RuntimeError("evaluation checkpoint path mismatch")
    checkpoint_path = output_dir / expected_checkpoint
    latest_path = output_dir / expected_latest
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != evaluation.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError("evaluation checkpoint hash mismatch")
    if not latest_path.is_file() or sha256_file(latest_path) != evaluation.get(
        "latest_checkpoint_sha256"
    ):
        raise RuntimeError("evaluation latest-checkpoint hash mismatch")
    plans = [item for item in experiment.build_plan() if item.split == "test"]
    recorded_entries = evaluation.get("entries")
    if (
        not isinstance(recorded_entries, list)
        or len(recorded_entries) != len(plans)
        or evaluation.get("trajectory_count") != len(plans)
    ):
        raise RuntimeError("evaluation trajectory inventory is incomplete")
    for plan, recorded in zip(plans, recorded_entries):
        source_relative = str(plan.relative_path)
        prediction_relative = f"predictions/seed_{plan.seed}.h5"
        expected_identity = {
            "seed": plan.seed,
            "source_path": source_relative,
            "prediction_path": prediction_relative,
        }
        for key, expected in expected_identity.items():
            if recorded.get(key) != expected:
                raise RuntimeError(f"evaluation entry {key} mismatch for seed {plan.seed}")
        source_path = manifest_path.parent / source_relative
        if sha256_file(source_path) != entries_by_path[source_relative]["sha256"]:
            raise RuntimeError(f"evaluation source hash mismatch for seed {plan.seed}")
        prediction_path = output_dir / prediction_relative
        if not prediction_path.is_file() or sha256_file(prediction_path) != recorded.get(
            "prediction_sha256"
        ):
            raise RuntimeError(f"evaluation prediction hash mismatch for seed {plan.seed}")
        if require_sidecars:
            entry_path = output_dir / "evaluation_entries" / f"seed_{plan.seed}.json"
            if not entry_path.is_file() or json.loads(
                entry_path.read_text(encoding="utf-8")
            ) != recorded:
                raise RuntimeError(f"evaluation sidecar mismatch for seed {plan.seed}")


def validate_prediction_artifact(
    source_path: Path,
    artifact_path: Path,
    entry_path: Path,
    plan: Any,
    dataset_sha256: str,
    checkpoint_sha256: str,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    with h5py.File(source_path, "r") as source, h5py.File(artifact_path, "r") as artifact:
        truth = source["phi_data"][:].astype(np.float32, copy=False)
        prediction = artifact["prediction"][:].astype(np.float32, copy=False)
        source_steps = source["base_steps"][:]
        artifact_steps = artifact["base_steps"][:]
        metadata = dict(artifact["metadata"].attrs.items())
        recorded_raw = {name: artifact[f"metrics/{name}"][:] for name in artifact["metrics"]}
    if not np.array_equal(source_steps, artifact_steps):
        raise RuntimeError(f"prediction step grid mismatch for seed {plan.seed}")
    expected_metadata = {
        "seed": plan.seed,
        "source_path": str(plan.relative_path),
        "source_sha256": sha256_file(source_path),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "official_commit": OFFICIAL_COMMIT,
    }
    for key, expected in expected_metadata.items():
        actual = metadata[key]
        actual = int(actual) if isinstance(expected, int) else str(actual)
        if actual != expected:
            raise RuntimeError(f"prediction metadata {key} mismatch for seed {plan.seed}")
    summary, raw = experiment.trajectory_metrics(prediction, truth, source_steps)
    if set(recorded_raw) != set(raw):
        raise RuntimeError(f"prediction metric inventory mismatch for seed {plan.seed}")
    for name, values in raw.items():
        np.testing.assert_allclose(recorded_raw[name], values, rtol=2.0e-6, atol=2.0e-7)
    artifact_sha256 = sha256_file(artifact_path)
    computed_entry = {
        "seed": plan.seed,
        "source_path": str(plan.relative_path),
        "prediction_path": str(artifact_path.relative_to(output_dir)),
        "prediction_sha256": artifact_sha256,
        **summary,
    }
    if entry_path.exists():
        recorded_entry = json.loads(entry_path.read_text(encoding="utf-8"))
        summary_matches(recorded_entry, computed_entry)
        entry = recorded_entry
    else:
        entry = {**computed_entry, "trajectory_seconds": None, "recovered_existing_artifact": True}
        atomic_json_new(entry_path, entry)
    return entry, raw["truth_radial"]


def write_prediction_artifact(
    path: Path,
    prediction: np.ndarray,
    base_steps: np.ndarray,
    raw: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing prediction overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(temporary, "x") as handle:
            handle.create_dataset(
                "prediction", data=prediction, chunks=(1, GRID, GRID), compression="gzip", shuffle=True
            )
            handle.create_dataset("base_steps", data=base_steps)
            metrics = handle.create_group("metrics")
            for name, values in raw.items():
                metrics.create_dataset(name, data=values, compression="gzip")
            group = handle.create_group("metadata")
            group.attrs.update(metadata)
            handle.flush()
        if path.exists():
            raise FileExistsError(f"prediction appeared concurrently: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_resumably(
    manifest_path: Path,
    manifest: dict[str, Any],
    config: Any,
    device: torch.device,
    provenance: dict[str, Any],
    output_dir: Path,
    log_path: Path,
    state_callback: Any,
) -> Path:
    evaluation_path = output_dir / "evaluation.json"
    recorded_evaluation = None
    if evaluation_path.exists():
        recorded_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        verify_evaluation_artifact_hashes(
            manifest_path,
            manifest,
            recorded_evaluation,
            output_dir,
            require_sidecars=True,
        )
    checkpoint, checkpoint_path, latest_path = experiment.load_evaluation_checkpoint(
        config, manifest, provenance, output_dir
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    model = None
    test_plans = [item for item in experiment.build_plan() if item.split == "test"]
    entries = []
    truth_radial: dict[int, np.ndarray] = {}
    for index, plan in enumerate(test_plans, start=1):
        source_path = manifest_path.parent / plan.relative_path
        artifact_path = output_dir / "predictions" / f"seed_{plan.seed}.h5"
        entry_path = output_dir / "evaluation_entries" / f"seed_{plan.seed}.json"
        if recorded_evaluation is not None:
            recorded_entry = recorded_evaluation["entries"][index - 1]
            if not artifact_path.is_file() or not entry_path.is_file():
                raise RuntimeError(f"completed evaluation artifact missing for seed {plan.seed}")
        if entry_path.exists() and not artifact_path.exists():
            raise RuntimeError(f"evaluation entry exists without raw artifact: {entry_path}")
        if artifact_path.exists():
            entry, radial = validate_prediction_artifact(
                source_path,
                artifact_path,
                entry_path,
                plan,
                manifest["dataset_sha256"],
                checkpoint_sha256,
                output_dir,
            )
            if recorded_evaluation is not None and entry != recorded_entry:
                raise RuntimeError(f"persisted evaluation entry mismatch for seed {plan.seed}")
            disposition = "verified_existing"
        else:
            if recorded_evaluation is not None:
                raise RuntimeError(f"completed prediction artifact missing for seed {plan.seed}")
            if model is None:
                set_cuda_determinism(config.seed)
                model = experiment.build_model(config).to(device)
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
            with h5py.File(source_path, "r") as handle:
                truth = handle["phi_data"][:].astype(np.float32, copy=False)
                base_steps = handle["base_steps"][:]
            prediction = np.empty_like(truth)
            prediction[0] = truth[0]
            started = time.monotonic()
            with torch.no_grad():
                current = torch.from_numpy(truth[0:1, None]).to(device)
                for step in range(1, len(truth)):
                    current, _ = experiment.predict_with_dcl(model, current)
                    prediction[step] = current[0, 0].detach().cpu().numpy()
            synchronize_cuda(device)
            summary, raw = experiment.trajectory_metrics(prediction, truth, base_steps)
            write_prediction_artifact(
                artifact_path,
                prediction,
                base_steps,
                raw,
                {
                    "seed": plan.seed,
                    "source_path": str(plan.relative_path),
                    "source_sha256": sha256_file(source_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "dataset_sha256": manifest["dataset_sha256"],
                    "official_commit": OFFICIAL_COMMIT,
                },
            )
            entry = {
                "seed": plan.seed,
                "source_path": str(plan.relative_path),
                "prediction_path": str(artifact_path.relative_to(output_dir)),
                "prediction_sha256": sha256_file(artifact_path),
                "trajectory_seconds": time.monotonic() - started,
                **summary,
            }
            atomic_json_new(entry_path, entry)
            radial = raw["truth_radial"]
            disposition = "generated"
        entries.append(entry)
        truth_radial[plan.seed] = radial
        atomic_json_replace(
            output_dir / "evaluation_progress.json",
            {
                "schema_version": 1,
                "dataset_sha256": manifest["dataset_sha256"],
                "checkpoint_sha256": checkpoint_sha256,
                "completed": len(entries),
                "total": len(test_plans),
                "entries": entries,
                "updated_at": utc_now(),
            },
        )
        state_callback(
            stage="evaluate",
            status="running",
            evaluation_completed=len(entries),
            evaluation_total=len(test_plans),
        )
        experiment.emit(
            "spinodal_evaluation_trajectory",
            log_path,
            index=index,
            total=len(test_plans),
            disposition=disposition,
            **entry,
        )
    matched = np.asarray([entry["radial_error_auc_T1_T2"] for entry in entries], dtype=np.float64)
    intrinsic = []
    normalized_time = np.arange(501, dtype=np.float64) / 500.0
    for first, second in zip(test_plans[::2], test_plans[1::2]):
        error = np.abs(truth_radial[first.seed][500:] - truth_radial[second.seed][500:]).mean(axis=1)
        intrinsic.append(float(np.trapezoid(error, x=normalized_time)))
    intrinsic_array = np.asarray(intrinsic, dtype=np.float64)
    ratio = experiment.bootstrap_ratio_interval(matched, intrinsic_array, config.seed)
    finite_all = all(entry["finite"] for entry in entries)
    maximum_drift = max(entry["maximum_relative_mass_drift"] for entry in entries)
    mean_final_mae = float(np.mean([entry["final_mae"] for entry in entries]))
    supports = bool(
        finite_all
        and maximum_drift <= config.mass_drift_threshold
        and mean_final_mae <= config.final_mae_threshold
        and ratio["upper_95"] <= config.radial_ratio_threshold
    )
    falsified = bool(ratio["lower_95"] > config.radial_ratio_threshold)
    verdict = "supports" if supports else "falsified" if falsified else "inconclusive"
    result = {
        "schema_version": 1,
        "claim_boundary": "128x128 released-training-scale 100dt mechanism study; not literal v1 Table 5",
        "verdict": verdict,
        "supports": supports,
        "falsified": falsified,
        "trajectory_count": len(entries),
        "finite_trajectories": sum(entry["finite"] for entry in entries),
        "mean_final_mae": mean_final_mae,
        "maximum_relative_mass_drift": maximum_drift,
        "radial_auc_ratio": ratio,
        "matched_radial_aucs": matched.tolist(),
        "intrinsic_pair_radial_aucs": intrinsic_array.tolist(),
        "thresholds": {
            "mass_drift": config.mass_drift_threshold,
            "mean_final_mae": config.final_mae_threshold,
            "radial_ratio_upper_95_support": config.radial_ratio_threshold,
            "radial_ratio_lower_95_falsify": config.radial_ratio_threshold,
        },
        "checkpoint": str(checkpoint_path.relative_to(output_dir)),
        "checkpoint_sha256": checkpoint_sha256,
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "evaluation_seconds": float(
            sum(entry.get("trajectory_seconds") or 0.0 for entry in entries)
        ),
        "peak_rss_mib": (
            recorded_evaluation["peak_rss_mib"]
            if recorded_evaluation is not None
            else peak_rss_mib()
        ),
        "entries": entries,
        **provenance,
    }
    if recorded_evaluation is not None:
        if result != recorded_evaluation:
            raise RuntimeError("persisted evaluation summary does not match recomputed artifacts")
    else:
        atomic_json_new(evaluation_path, result)
    return evaluation_path


def assert_close(label: str, actual: float, expected: float, tolerance: float = 2.0e-7) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")


def run_or_verify_independent_audit(
    data_dir: Path,
    output_dir: Path,
    audit_source: Path,
) -> Path:
    certificate_path = output_dir / "audit_certificate.json"
    manifest_path = data_dir / "manifest.json"
    evaluation_path = output_dir / "evaluation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    checkpoint_path = output_dir / evaluation["checkpoint"]
    verify_evaluation_artifact_hashes(
        manifest_path,
        manifest,
        evaluation,
        output_dir,
        require_sidecars=True,
    )
    if certificate_path.exists():
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        expected_exact = {
            "official_commit": OFFICIAL_COMMIT,
            "audit_source_sha256": sha256_file(audit_source),
            "manifest_sha256": sha256_file(manifest_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "checkpoint_sha256": evaluation["checkpoint_sha256"],
            "trajectory_count": 20,
            "finite_trajectories": evaluation["finite_trajectories"],
            "verdict": evaluation["verdict"],
            "supports": evaluation["supports"],
            "falsified": evaluation["falsified"],
            "prediction_sha256": {
                str(entry["seed"]): entry["prediction_sha256"]
                for entry in evaluation["entries"]
            },
        }
        for key, value in expected_exact.items():
            if certificate.get(key) != value:
                raise RuntimeError(f"persisted audit certificate {key} mismatch")
        for key in ("mean_final_mae", "maximum_relative_mass_drift"):
            assert_close(key, float(certificate[key]), float(evaluation[key]))
        for key in ("point", "lower_95", "upper_95"):
            assert_close(
                f"radial_auc_ratio.{key}",
                float(certificate["radial_auc_ratio"][key]),
                float(evaluation["radial_auc_ratio"][key]),
            )
        for key in ("replicates", "seed"):
            if certificate["radial_auc_ratio"][key] != evaluation["radial_auc_ratio"][key]:
                raise RuntimeError(f"persisted audit certificate radial_auc_ratio.{key} mismatch")
        return certificate_path
    if manifest["dataset_sha256"] != evaluation["dataset_sha256"]:
        raise RuntimeError("evaluation dataset fingerprint mismatch")
    if sha256_file(checkpoint_path) != evaluation["checkpoint_sha256"]:
        raise RuntimeError("evaluation checkpoint hash mismatch")
    entries_by_path = {entry["path"]: entry for entry in manifest["files"]}
    rows, columns = np.indices((GRID, GRID), dtype=np.float64)
    radius_map = np.sqrt((rows - GRID // 2) ** 2 + (columns - GRID // 2) ** 2)
    audited_entries = []
    truth_radial = {}
    started = time.monotonic()
    for recorded in evaluation["entries"]:
        source_relative = recorded["source_path"]
        source_path = data_dir / source_relative
        prediction_path = output_dir / recorded["prediction_path"]
        if source_relative not in entries_by_path:
            raise RuntimeError(f"unmanifested audit source: {source_relative}")
        if sha256_file(source_path) != entries_by_path[source_relative]["sha256"]:
            raise RuntimeError(f"audit source hash mismatch: {source_relative}")
        if sha256_file(prediction_path) != recorded["prediction_sha256"]:
            raise RuntimeError(f"audit prediction hash mismatch: {prediction_path}")
        audited, radial = independent_audit.audit_trajectory(
            source_path,
            prediction_path,
            recorded,
            radius_map,
            manifest["dataset_sha256"],
            evaluation["checkpoint_sha256"],
        )
        audited_entries.append({"seed": int(recorded["seed"]), **audited})
        truth_radial[int(recorded["seed"])] = radial
    if len(audited_entries) != 20 or len(truth_radial) != 20:
        raise RuntimeError("independent audit requires 20 unique trajectories")
    matched = np.asarray([entry["radial_error_auc_T1_T2"] for entry in audited_entries])
    intrinsic = []
    seeds = [entry["seed"] for entry in audited_entries]
    normalized_time = np.arange(501, dtype=np.float64) / 500.0
    for first, second in zip(seeds[::2], seeds[1::2]):
        error = np.mean(np.abs(truth_radial[first][500:] - truth_radial[second][500:]), axis=1)
        intrinsic.append(float(np.trapezoid(error, x=normalized_time)))
    ratio = independent_audit.independent_bootstrap(matched, np.asarray(intrinsic))
    for name in ("point", "lower_95", "upper_95"):
        assert_close(name, ratio[name], float(evaluation["radial_auc_ratio"][name]))
    mean_final_mae = float(np.mean([entry["final_mae"] for entry in audited_entries]))
    maximum_drift = float(np.max([entry["maximum_relative_mass_drift"] for entry in audited_entries]))
    finite_all = all(entry["finite"] for entry in audited_entries)
    supports = bool(
        finite_all
        and maximum_drift <= 1.0e-5
        and mean_final_mae <= 4.32e-2
        and ratio["upper_95"] <= 1.25
    )
    falsified = bool(ratio["lower_95"] > 1.25)
    verdict = "supports" if supports else "falsified" if falsified else "inconclusive"
    if (
        verdict != evaluation["verdict"]
        or supports != evaluation["supports"]
        or falsified != evaluation["falsified"]
    ):
        raise RuntimeError("independent verdict mismatch")
    certificate = {
        "schema_version": 1,
        "audit": "independent raw HDF5 recomputation without experiment metric imports",
        "audit_source_sha256": sha256_file(audit_source),
        "official_commit": OFFICIAL_COMMIT,
        "manifest_sha256": sha256_file(manifest_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "prediction_sha256": {
            str(entry["seed"]): entry["prediction_sha256"] for entry in evaluation["entries"]
        },
        "trajectory_count": len(audited_entries),
        "finite_trajectories": sum(entry["finite"] for entry in audited_entries),
        "mean_final_mae": mean_final_mae,
        "maximum_relative_mass_drift": maximum_drift,
        "radial_auc_ratio": ratio,
        "verdict": verdict,
        "supports": supports,
        "falsified": falsified,
        "audit_seconds": time.monotonic() - started,
    }
    atomic_json_new(certificate_path, certificate)
    return certificate_path


def completion_record(
    run_root: Path,
    contract_sha256: str,
    manifest_path: Path,
    output_dir: Path,
) -> Path:
    completion_path = run_root / "completion.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation_path = output_dir / "evaluation.json"
    audit_path = output_dir / "audit_certificate.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    latest = output_dir / "models/FluxNet_D_pf_100dt/latest_checkpoint.pt"
    best = output_dir / "models/FluxNet_D_pf_100dt/best_checkpoint.pt"
    history = output_dir / "models/FluxNet_D_pf_100dt/training_history.json"
    verify_evaluation_artifact_hashes(
        manifest_path,
        manifest,
        evaluation,
        output_dir,
        require_sidecars=True,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_audit_links = {
        "official_commit": OFFICIAL_COMMIT,
        "manifest_sha256": sha256_file(manifest_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "prediction_sha256": {
            str(entry["seed"]): entry["prediction_sha256"] for entry in evaluation["entries"]
        },
        "verdict": evaluation["verdict"],
        "supports": evaluation["supports"],
        "falsified": evaluation["falsified"],
    }
    for key, expected in expected_audit_links.items():
        if audit.get(key) != expected:
            raise RuntimeError(f"audit certificate {key} mismatch before completion")
    record = {
        "schema_version": 1,
        "status": "complete",
        "paper_id": PAPER_ID,
        "contract_sha256": contract_sha256,
        "official_commit": OFFICIAL_COMMIT,
        "dataset": {
            "manifest_path": str(manifest_path.relative_to(run_root)),
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_sha256": manifest["dataset_sha256"],
            "trajectory_count": len(manifest["files"]),
            "trajectory_sha256": {entry["path"]: entry["sha256"] for entry in manifest["files"]},
        },
        "training": {
            "completed_epochs": 100,
            "latest_checkpoint": str(latest.relative_to(run_root)),
            "latest_checkpoint_sha256": sha256_file(latest),
            "best_checkpoint": str(best.relative_to(run_root)),
            "best_checkpoint_sha256": sha256_file(best),
            "history": str(history.relative_to(run_root)),
            "history_sha256": sha256_file(history),
        },
        "evaluation": {
            "path": str(evaluation_path.relative_to(run_root)),
            "sha256": sha256_file(evaluation_path),
            "trajectory_count": evaluation["trajectory_count"],
            "prediction_sha256": {
                str(entry["seed"]): entry["prediction_sha256"] for entry in evaluation["entries"]
            },
            "verdict": evaluation["verdict"],
        },
        "audit": {
            "path": str(audit_path.relative_to(run_root)),
            "sha256": sha256_file(audit_path),
            "verdict": audit["verdict"],
        },
        "completed_at": utc_now(),
    }
    if completion_path.exists():
        existing = json.loads(completion_path.read_text(encoding="utf-8"))
        existing_without_time = {key: value for key, value in existing.items() if key != "completed_at"}
        record_without_time = {key: value for key, value in record.items() if key != "completed_at"}
        if existing_without_time != record_without_time:
            raise RuntimeError("persisted completion record mismatch")
    else:
        atomic_json_new(completion_path, record)
    return completion_path


def validate_run_root(path: Path) -> None:
    resolved = path.resolve(strict=False)
    root = Path("/artifacts").resolve(strict=False)
    if root not in resolved.parents:
        raise RuntimeError(f"run root must be within mounted bucket: {resolved}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--full-source-manifest",
        type=Path,
        default=Path("/workspace/hf_jobs/spinodal_attempt2b_full_sources.sha256"),
    )
    parser.add_argument(
        "--preflight-source-manifest",
        type=Path,
        default=Path("/workspace/hf_jobs/spinodal_attempt2b_preflight_sources.sha256"),
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=Path("/artifacts/hf-jobs/spinodal-attempt2b/preflight-v1.json"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/artifacts/hf-jobs/spinodal-attempt2b/full-v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_run_root(args.run_root)
    if JOB_WORST_CASE_COST_USD > AUTHORIZED_CAMPAIGN_CAP_USD:
        raise RuntimeError("job timeout commitment exceeds the aggregate authorized campaign cap")
    args.run_root.mkdir(parents=True, exist_ok=True)
    attempt_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{platform.node()}-{os.getpid()}"
    attempt_report_path = args.run_root / "job_attempts" / f"{attempt_id}.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "paper_id": PAPER_ID,
        "started_at": utc_now(),
        "status": "running",
        "hostname": platform.node(),
        "pid": os.getpid(),
        "errors": [],
    }
    started = time.monotonic()
    contract_sha256 = "not_verified"

    def deadline(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"internal {INTERNAL_DEADLINE_HOURS} hour deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(round(INTERNAL_DEADLINE_HOURS * 3600))
    exit_code = 0
    try:
        verification, _ = verify_full_sources(
            args.source_root,
            args.full_source_manifest,
            args.preflight_source_manifest,
            args.preflight_report,
        )
        cuda = require_t4()
        report["source_verification"] = verification
        report["cuda_runtime"] = cuda
        _, contract_sha256 = create_or_verify_provenance(args.run_root, verification, cuda)
        report["contract_sha256"] = contract_sha256
        update_state(
            args.run_root,
            contract_sha256,
            status="running",
            stage="verify",
            current_attempt=attempt_id,
        )
        experiment.compile_solver = compile_solver_linux
        experiment.set_determinism = set_cuda_determinism
        experiment.synchronize = synchronize_cuda
        provenance = scientific_provenance(verification)
        data_dir = args.run_root / "data"
        output_dir = args.run_root / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "events.jsonl"
        manifest_path = data_dir / "manifest.json"
        update_state(args.run_root, contract_sha256, status="running", stage="generate")
        if manifest_path.exists():
            manifest = experiment.verify_dataset(manifest_path, provenance)
        else:
            experiment.prepare_dataset(data_dir, 4, provenance, log_path)
            manifest = experiment.verify_dataset(manifest_path, provenance)
        update_state(
            args.run_root,
            contract_sha256,
            status="running",
            stage="train",
            dataset_sha256=manifest["dataset_sha256"],
            generated_trajectories=len(manifest["files"]),
        )
        device = torch.device("cuda:0")
        config = experiment.RunConfig()
        run_metadata = {
            "schema_version": 1,
            "config": asdict(config),
            **provenance,
            "full_source_manifest_sha256": verification["full_source_manifest_sha256"],
            "preflight_sha256": PREFLIGHT_SHA256,
            "device": "cuda:0",
            "cuda": {
                key: cuda[key]
                for key in (
                    "name",
                    "total_memory_bytes",
                    "capability",
                    "torch",
                    "torch_cuda",
                    "cudnn",
                    "h5py",
                    "numpy",
                )
            },
            "environment": {
                key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "loss_formula": "0.25*p + 0.25*stability + 0.5*dcl + 0.5*dcl_n",
        }
        write_once_or_verify(output_dir / "run_metadata.json", run_metadata)
        latest_path = output_dir / "models/FluxNet_D_pf_100dt/latest_checkpoint.pt"
        latest_checkpoint = reconcile_training_artifacts(
            config, manifest, provenance, output_dir
        )
        training_complete = bool(
            latest_checkpoint is not None
            and int(latest_checkpoint.get("completed_epochs", -1)) == config.epochs
        )
        if not training_complete:
            train, validation = experiment.load_training_arrays(manifest_path)
            experiment.train_model(
                train,
                validation,
                config,
                device,
                manifest,
                provenance,
                output_dir,
                log_path,
                True,
                None,
            )
        latest_checkpoint = reconcile_training_artifacts(
            config, manifest, provenance, output_dir
        )
        if latest_checkpoint is None:
            raise RuntimeError("training produced no latest checkpoint")
        if int(latest_checkpoint["completed_epochs"]) != 100:
            raise RuntimeError("training did not complete all 100 preregistered epochs")
        update_state(
            args.run_root,
            contract_sha256,
            status="running",
            stage="evaluate",
            completed_epochs=100,
            latest_checkpoint_sha256=sha256_file(latest_path),
        )

        def state_callback(**values: Any) -> None:
            update_state(args.run_root, contract_sha256, **values)

        evaluation_path = evaluate_resumably(
            manifest_path,
            manifest,
            config,
            device,
            provenance,
            output_dir,
            log_path,
            state_callback,
        )
        update_state(
            args.run_root,
            contract_sha256,
            status="running",
            stage="audit",
            evaluation_sha256=sha256_file(evaluation_path),
            evaluation_completed=20,
        )
        audit_path = run_or_verify_independent_audit(
            data_dir,
            output_dir,
            args.source_root / "repro/src/audit_spinodal_attempt2b.py",
        )
        completion_path = completion_record(
            args.run_root, contract_sha256, manifest_path, output_dir
        )
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        update_state(
            args.run_root,
            contract_sha256,
            status="complete",
            stage="complete",
            completion_sha256=sha256_file(completion_path),
            audit_sha256=sha256_file(audit_path),
            verdict=completion["evaluation"]["verdict"],
        )
        report["status"] = "complete"
        report["completion"] = completion
        report["completion_sha256"] = sha256_file(completion_path)
    except TimeoutError as error:
        exit_code = 75
        report["status"] = "paused_timeout"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
        if contract_sha256 != "not_verified":
            update_state(
                args.run_root,
                contract_sha256,
                status="paused_timeout",
                current_attempt=attempt_id,
                resume_command="rerun the identical launcher command",
            )
    except BaseException as error:
        exit_code = 1
        report["status"] = "failed"
        report["errors"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback_tail": traceback.format_exc(limit=16)[-16000:],
            }
        )
        if contract_sha256 != "not_verified":
            update_state(
                args.run_root,
                contract_sha256,
                status="failed",
                current_attempt=attempt_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
    finally:
        signal.alarm(0)
        report["finished_at"] = utc_now()
        report["wall_seconds"] = time.monotonic() - started
        report["peak_rss_mib"] = peak_rss_mib()
        if torch.cuda.is_available():
            report["cuda_memory"] = {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            }
        atomic_json_new(attempt_report_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "attempt_report": str(attempt_report_path),
                    "attempt_report_sha256": sha256_file(attempt_report_path),
                    "resume": "rerun the identical launcher command",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
