"""Fail-closed HF T4 preflight for preregistered spinodal Attempt 2b.

This calibration job never writes a scientific dataset or checkpoint.  It
verifies the mounted sources, checks the C++ solver against an independent
NumPy oracle, times short solver samples, and measures the exact released
128x128 batch-16 FluxNet-D training computation on CUDA.  Its only durable
output is one JSON report in the mounted Hugging Face bucket.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 1
PAPER_ID = "1KRpajnd6u"
ATTEMPT = "claim3_spinodal_attempt2b"
OFFICIAL_COMMIT = "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c"
GRID = 128
FRAME_BYTES = GRID * GRID * np.dtype(np.float32).itemsize
T4_RATE_USD_PER_HOUR = 0.40
OUTER_TIMEOUT_MINUTES = 55
INTERNAL_TIMEOUT_MINUTES = 40
PREFLIGHT_MAX_COST_USD = OUTER_TIMEOUT_MINUTES / 60.0 * T4_RATE_USD_PER_HOUR
AGGREGATE_AUTHORIZED_CAP_USD = 10.0
PROJECTION_BUFFER = 1.25
CUDA_MEMORY_GATE_BYTES = 12 * 1024**3

CONFIG = {
    "seed": 42,
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-2,
    "ndt": 10,
    "unroll_steps": 2,
    "base_channels": 32,
    "num_blocks": 4,
    "kernel_size": 5,
    "neighborhood_size": 5,
    "scheduler_patience": 15,
    "scheduler_factor": 0.5,
    "p_loss_weight": 0.25,
    "stability_loss_weight": 0.25,
    "dcl_weight": 0.5,
    "dcl_n_weight": 0.5,
    "mass_drift_threshold": 1.0e-5,
    "final_mae_threshold": 4.32e-2,
    "radial_ratio_threshold": 1.25,
}
OFFICIAL_HPARAMS = {
    "base_channels": 32,
    "num_blocks": 4,
    "kernel_size": 5,
    "neighborhood_size": 5,
    "num_epochs": 100,
    "batch_size": 16,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-2,
    "ndt": 10,
    "num_workers": 4,
    "unroll_steps": 2,
    "loss_weight_mode": "manual",
    "dcl_weight": 0.5,
    "soft_cons_weight": 0.5,
    "loss_weights": {"p_loss": 0.5, "dcl_loss": 0.5, "stability_loss": 0.5},
}
SOURCE_RELATIVE_PATHS = {
    "wrapper": "repro/src/spinodal_attempt2b.py",
    "solver": "repro/src/spinodal_solver.cpp",
    "preflight": "repro/src/hf_spinodal_attempt2b_preflight.py",
    "preregistration": "docs/claim3_spinodal_attempt2b_preregistration.md",
    "released_generator": "official/dataset/spinodal_decomposition/phase_field_generator.cu",
    "released_test_generator": "official/dataset/spinodal_decomposition/phase_field_generator_test.cu",
    "released_model": "official/src/models/fluxnet_d_2d.py",
    "released_dataloader": "official/src/training/dataloader.py",
    "released_trainer": "official/src/training/trainer_unified.py",
    "released_100dt_config": "official/experiments/spinodal_decomposition/run_single_seed_100dt.py",
}

# The calibration workload is deliberately far below the fixed full protocol.
ORACLE_STEPS = 6
GENERATION_BENCHMARK_STEPS = 2000
TRAIN_SAVE_INTERVAL = 10
TEST_SAVE_INTERVAL = 100
TRAIN_WARMUP_BATCHES = 2
TRAIN_TIMED_BATCHES = 6
VALIDATION_TIMED_BATCHES = 6
ROLLOUT_WARMUP_STEPS = 3
ROLLOUT_TIMED_STEPS = 12
RADIAL_TIMED_FRAMES = 12
EXPECTED_MODEL_PARAMETERS = 216_434


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing preflight artifact: {path}")
    payload = json.dumps(value, indent=2, sort_keys=True, default=str).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        value /= 1024.0
    return value / 1024.0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timing_summary(values: list[float]) -> dict[str, Any]:
    if len(values) < 2 or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise RuntimeError(f"invalid timing sample: {values}")
    return {
        "samples_seconds": values,
        "count": len(values),
        "median_seconds": float(statistics.median(values)),
        "mean_seconds": float(statistics.fmean(values)),
        "p90_seconds": percentile(values, 90.0),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def load_hash_manifest(path: Path) -> tuple[dict[str, str], str]:
    records: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise RuntimeError(f"invalid source manifest line {line_number}")
        digest, relative = fields
        relative = relative.lstrip("*")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in records:
            raise RuntimeError(f"unsafe or duplicate source manifest path: {relative}")
        int(digest, 16)
        records[relative] = digest.lower()
    required = set(SOURCE_RELATIVE_PATHS.values())
    if set(records) != required:
        raise RuntimeError(
            f"source manifest inventory mismatch: missing={sorted(required - set(records))}, "
            f"extra={sorted(set(records) - required)}"
        )
    return records, sha256_file(path)


def extract_assignments(path: Path, class_name: str | None = None) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = tree.body
    if class_name is not None:
        matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(matches) != 1:
            raise RuntimeError(f"expected one class {class_name} in {path}")
        body = matches[0].body
    values: dict[str, Any] = {}
    for node in body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def extract_official_hparams(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "hparams"
        ):
            matches.append(ast.literal_eval(node.value))
    if len(matches) != 1:
        raise RuntimeError(f"expected one literal hparams mapping in {path}")
    return matches[0]


def verify_sources(source_root: Path, manifest_path: Path) -> dict[str, Any]:
    expected, manifest_sha256 = load_hash_manifest(manifest_path)
    observed = {}
    for relative, expected_digest in sorted(expected.items()):
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing mounted source: {relative}")
        actual = sha256_file(path)
        if actual != expected_digest:
            raise RuntimeError(f"mounted source hash mismatch: {relative}")
        observed[relative] = actual

    wrapper_values = extract_assignments(source_root / SOURCE_RELATIVE_PATHS["wrapper"])
    if wrapper_values.get("OFFICIAL_COMMIT") != OFFICIAL_COMMIT or wrapper_values.get("GRID") != GRID:
        raise RuntimeError("wrapper commit/grid constants do not match the preregistered values")
    wrapper_config = extract_assignments(
        source_root / SOURCE_RELATIVE_PATHS["wrapper"], class_name="RunConfig"
    )
    if wrapper_config != CONFIG:
        raise RuntimeError(f"RunConfig mismatch: {wrapper_config!r}")
    official_hparams = extract_official_hparams(
        source_root / SOURCE_RELATIVE_PATHS["released_100dt_config"]
    )
    if official_hparams != OFFICIAL_HPARAMS:
        raise RuntimeError(f"released 100dt hparams mismatch: {official_hparams!r}")
    preregistration = (source_root / SOURCE_RELATIVE_PATHS["preregistration"]).read_text(
        encoding="utf-8"
    )
    required_phrases = (
        "22 unique trajectories",
        "30,022 retained frames",
        "1,967,521,792 uncompressed field bytes",
        "batch 16",
        "100 epochs",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in preregistration]
    if missing_phrases:
        raise RuntimeError(f"preregistration contract phrases missing: {missing_phrases}")
    return {
        "verified": True,
        "official_commit": OFFICIAL_COMMIT,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "source_sha256": observed,
        "wrapper_run_config": wrapper_config,
        "released_100dt_hparams": official_hparams,
    }


def compile_solver(source: Path, output: Path) -> dict[str, Any]:
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required for the CPU solver oracle")
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-ffp-contract=off",
        "-fno-fast-math",
        str(source),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    return {
        "command": command,
        "compiler_version": subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
        "binary_sha256": sha256_file(output),
    }


def numpy_step(concentration: np.ndarray) -> np.ndarray:
    temperature = 973.15
    rt = 8.314 * temperature
    a0 = 15000.0 + 6.1 * temperature
    a1 = -7600.0 + 3.55 * temperature
    safe = np.clip(concentration, 1.0e-10, 1.0 - 1.0e-10)
    derivative = (
        rt * np.log(safe / (1.0 - safe))
        + (1.0 - 2.0 * safe) * a0
        + (-6.0 * safe + 6.0 * safe * safe + 1.0) * a1
    ) / rt
    laplacian = (
        np.roll(concentration, -1, axis=0)
        + np.roll(concentration, 1, axis=0)
        + np.roll(concentration, -1, axis=1)
        + np.roll(concentration, 1, axis=1)
        - 4.0 * concentration
    )
    potential = derivative - 2.0 * 3.57e-1 * laplacian
    laplacian_potential = (
        np.roll(potential, -1, axis=0)
        + np.roll(potential, 1, axis=0)
        + np.roll(potential, -1, axis=1)
        + np.roll(potential, 1, axis=1)
        - 4.0 * potential
    )
    return np.clip(concentration + 1.0e-2 * laplacian_potential, 0.0, 1.0)


def serialized(field: np.ndarray) -> np.ndarray:
    return (np.round(field * 1.0e6) / 1.0e6).astype(np.float32)


def run_solver(
    binary: Path,
    *,
    steps: int,
    save_start: int,
    save_interval: int,
    threads: int,
    seed: int | None = None,
    input_path: Path | None = None,
    timeout: float = 600.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if steps >= 52_000:
        raise RuntimeError("preflight guard forbids a full-protocol solver trajectory")
    command = [
        str(binary),
        "--steps",
        str(steps),
        "--save-start",
        str(save_start),
        "--save-interval",
        str(save_interval),
        "--threads",
        str(threads),
    ]
    if seed is not None:
        command.extend(("--seed", str(seed)))
    if input_path is not None:
        command.extend(("--input", str(input_path)))
    started = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, timeout=timeout)
    seconds = time.perf_counter() - started
    expected_frames = (steps - save_start) // save_interval + 1
    expected_bytes = expected_frames * FRAME_BYTES
    if len(completed.stdout) != expected_bytes:
        raise RuntimeError(
            f"solver byte count mismatch: {len(completed.stdout)} != {expected_bytes}"
        )
    expected_completion = f"complete steps={steps} frames={expected_frames} threads={threads}".encode()
    if expected_completion not in completed.stderr:
        raise RuntimeError("solver completion record missing")
    frames = np.frombuffer(completed.stdout, dtype=np.float32).reshape(
        expected_frames, GRID, GRID
    ).copy()
    return frames, {
        "command": command,
        "steps": steps,
        "frames": expected_frames,
        "streamed_bytes": expected_bytes,
        "seconds": seconds,
        "updates_per_second": steps / seconds,
        "stream_mib_per_second": expected_bytes / (1024**2) / seconds,
        "stderr_tail": completed.stderr.decode("utf-8", errors="replace")[-2000:],
    }


def solver_oracle(binary: Path, temporary: Path) -> dict[str, Any]:
    rows, columns = np.indices((GRID, GRID), dtype=np.float64)
    initial = 0.60 + 0.01 * np.sin(rows / 9.0) * np.cos(columns / 13.0)
    input_path = temporary / "oracle_initial.float64"
    initial.tofile(input_path)
    observed, timing = run_solver(
        binary,
        steps=ORACLE_STEPS,
        save_start=1,
        save_interval=1,
        threads=2,
        input_path=input_path,
    )
    state = initial.copy()
    expected = []
    for _ in range(ORACLE_STEPS):
        state = numpy_step(state)
        expected.append(serialized(state))
    expected_array = np.stack(expected)
    maximum_absolute_error = float(np.max(np.abs(observed.astype(np.float64) - expected_array)))
    tolerance = 1.0e-6
    if not np.allclose(observed, expected_array, rtol=0.0, atol=tolerance):
        raise RuntimeError(f"CPU/NumPy solver oracle failed: max abs {maximum_absolute_error}")
    return {
        "passed": True,
        "steps": ORACLE_STEPS,
        "shape": list(observed.shape),
        "rtol": 0.0,
        "atol": tolerance,
        "maximum_absolute_error": maximum_absolute_error,
        "solver_timing": timing,
    }


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("mounted_author_fluxnet_d_2d", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import mounted model source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict_with_dcl(model: torch.nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prediction, outflow, inflow = model(inputs)
    return prediction, torch.mean((outflow - inflow) ** 2)


def batch_losses(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    first, dcl = predict_with_dcl(model, inputs)
    prediction = torch.mean((first - targets[:, 0:1]) ** 2)
    current = inputs
    with torch.no_grad():
        for _ in range(CONFIG["unroll_steps"] - 1):
            current, _ = predict_with_dcl(model, current)
    terminal, dcl_n = predict_with_dcl(model, current)
    stability = torch.mean((terminal - targets[:, -1:]) ** 2)
    total = (
        CONFIG["p_loss_weight"] * prediction
        + CONFIG["stability_loss_weight"] * stability
        + CONFIG["dcl_weight"] * dcl
        + CONFIG["dcl_n_weight"] * dcl_n
    )
    return {
        "prediction": prediction,
        "stability": stability,
        "dcl": dcl,
        "dcl_n": dcl_n,
        "total": total,
    }


def make_numpy_batch(frames: np.ndarray, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    maximum_start = len(frames) - CONFIG["ndt"] * CONFIG["unroll_steps"] - 1
    indices = np.linspace(0, maximum_start, CONFIG["batch_size"], dtype=np.int64)
    if offset:
        indices = (indices + offset) % (maximum_start + 1)
    inputs = np.ascontiguousarray(frames[indices, None])
    targets = np.ascontiguousarray(
        np.stack(
            [
                frames[indices + step * CONFIG["ndt"]]
                for step in range(1, CONFIG["unroll_steps"] + 1)
            ],
            axis=1,
        )
    )
    if inputs.shape != (16, 1, 128, 128) or targets.shape != (16, 2, 128, 128):
        raise RuntimeError(f"paper-scale calibration batch shape mismatch: {inputs.shape}, {targets.shape}")
    return inputs, targets


def cuda_runtime() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"preflight requires exactly one CUDA device, found {torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(0)
    if "T4" not in properties.name.upper():
        raise RuntimeError(f"preflight is authorized only on NVIDIA T4, found {properties.name}")
    query = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,power.limit",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(query, check=True, capture_output=True, text=True, timeout=30)
    return {
        "device_count": torch.cuda.device_count(),
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": int(properties.total_memory),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_smi_query": query,
        "nvidia_smi_csv": completed.stdout.strip(),
    }


def time_cuda_training(model_path: Path, frames: np.ndarray) -> dict[str, Any]:
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    module = load_module(model_path)
    model = module.FluxNet_D(
        in_channels=1,
        base_channels=CONFIG["base_channels"],
        num_blocks=CONFIG["num_blocks"],
        kernel_size=CONFIG["kernel_size"],
        neighborhood_size=CONFIG["neighborhood_size"],
        lower_bound=0.0,
        upper_bound=1.0,
        learnable_lower_bound=False,
        learnable_upper_bound=False,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError(f"model parameter count mismatch: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    def training_batch(index: int) -> dict[str, float]:
        x_numpy, y_numpy = make_numpy_batch(frames, offset=index)
        inputs = torch.from_numpy(x_numpy).to(device)
        targets = torch.from_numpy(y_numpy).to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = batch_losses(model, inputs, targets)
        if not all(torch.isfinite(value) for value in losses.values()):
            raise FloatingPointError("non-finite exact-protocol training loss")
        losses["total"].backward()
        optimizer.step()
        return {name: float(value.detach().cpu()) for name, value in losses.items()}

    model.train()
    for index in range(TRAIN_WARMUP_BATCHES):
        training_batch(index)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_seconds = []
    last_losses: dict[str, float] | None = None
    for index in range(TRAIN_TIMED_BATCHES):
        started = time.perf_counter()
        last_losses = training_batch(index + TRAIN_WARMUP_BATCHES)
        torch.cuda.synchronize()
        train_seconds.append(time.perf_counter() - started)

    model.eval()
    validation_seconds = []
    with torch.no_grad():
        for index in range(VALIDATION_TIMED_BATCHES):
            x_numpy, y_numpy = make_numpy_batch(frames, offset=index)
            started = time.perf_counter()
            inputs = torch.from_numpy(x_numpy).to(device)
            targets = torch.from_numpy(y_numpy).to(device)
            losses = batch_losses(model, inputs, targets)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError("non-finite exact-protocol validation loss")
            torch.cuda.synchronize()
            validation_seconds.append(time.perf_counter() - started)

    current = torch.from_numpy(frames[0:1, None]).to(device)
    with torch.no_grad():
        for _ in range(ROLLOUT_WARMUP_STEPS):
            current, _ = predict_with_dcl(model, current)
        torch.cuda.synchronize()
        rollout_seconds = []
        for _ in range(ROLLOUT_TIMED_STEPS):
            started = time.perf_counter()
            current, _ = predict_with_dcl(model, current)
            torch.cuda.synchronize()
            rollout_seconds.append(time.perf_counter() - started)
    maximum_allocated = int(torch.cuda.max_memory_allocated(0))
    maximum_reserved = int(torch.cuda.max_memory_reserved(0))
    memory_gate_value = max(maximum_allocated, maximum_reserved)
    if memory_gate_value > CUDA_MEMORY_GATE_BYTES:
        raise RuntimeError(
            f"CUDA peak {memory_gate_value} bytes exceeds preregistered 12 GiB gate"
        )
    return {
        "device": "cuda:0",
        "model": "FluxNet_D_pf_100dt",
        "parameter_count": parameter_count,
        "input_shape": [16, 1, 128, 128],
        "target_shape": [16, 2, 128, 128],
        "warmup_training_batches": TRAIN_WARMUP_BATCHES,
        "training_batches": timing_summary(train_seconds),
        "validation_batches": timing_summary(validation_seconds),
        "rollout_steps_batch1": timing_summary(rollout_seconds),
        "last_training_losses": last_losses,
        "peak_cuda_allocated_bytes": maximum_allocated,
        "peak_cuda_reserved_bytes": maximum_reserved,
        "cuda_memory_gate_bytes": CUDA_MEMORY_GATE_BYTES,
        "cuda_memory_gate_passed": True,
    }


def radial_autocorrelation(field: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft2(field.astype(np.float64, copy=False))
    image = np.fft.fftshift(np.fft.ifft2(spectrum * np.conj(spectrum)).real) / field.size
    rows, columns = np.indices(field.shape, dtype=np.float64)
    radii = np.sqrt((rows - GRID // 2) ** 2 + (columns - GRID // 2) ** 2)
    result = np.empty(GRID // 2, dtype=np.float64)
    for radius in range(GRID // 2):
        mask = (radii >= radius - 0.5) & (radii < radius + 0.5)
        result[radius] = float(image[mask].mean())
    return result


def time_radial_metrics(frames: np.ndarray) -> dict[str, Any]:
    durations = []
    for frame in frames[:RADIAL_TIMED_FRAMES]:
        started = time.perf_counter()
        radial = radial_autocorrelation(frame)
        durations.append(time.perf_counter() - started)
        if radial.shape != (64,) or not np.isfinite(radial).all():
            raise FloatingPointError("invalid radial-autocorrelation calibration result")
    return timing_summary(durations)


def projection(
    train_generation: dict[str, Any],
    test_generation: dict[str, Any],
    cuda: dict[str, Any],
    radial: dict[str, Any],
) -> dict[str, Any]:
    train_generation_seconds = 2 * 52_000 * train_generation["seconds"] / train_generation["steps"]
    test_generation_seconds = 20 * 102_000 * test_generation["seconds"] / test_generation["steps"]
    generation_raw_seconds = train_generation_seconds + test_generation_seconds
    batches_per_split = math.ceil(4_981 / CONFIG["batch_size"])
    training_raw_seconds = CONFIG["epochs"] * batches_per_split * (
        cuda["training_batches"]["median_seconds"]
        + cuda["validation_batches"]["median_seconds"]
    )
    rollout_seconds = 20 * 1_000 * cuda["rollout_steps_batch1"]["median_seconds"]
    radial_seconds = 2 * 20 * 1_001 * radial["median_seconds"]
    evaluation_raw_seconds = rollout_seconds + radial_seconds
    raw_seconds = generation_raw_seconds + training_raw_seconds + evaluation_raw_seconds
    buffered_seconds = raw_seconds * PROJECTION_BUFFER
    full_cost = buffered_seconds / 3600.0 * T4_RATE_USD_PER_HOUR
    aggregate_with_preflight = full_cost + PREFLIGHT_MAX_COST_USD
    within_cap = aggregate_with_preflight <= AGGREGATE_AUTHORIZED_CAP_USD
    return {
        "method": "linear scaling from stage-inclusive measured medians, then 25 percent buffer",
        "buffer_multiplier": PROJECTION_BUFFER,
        "generation": {
            "train_val_seconds": train_generation_seconds,
            "test_seconds": test_generation_seconds,
            "raw_hours": generation_raw_seconds / 3600.0,
            "buffered_hours": generation_raw_seconds * PROJECTION_BUFFER / 3600.0,
            "formula": "2*52000*(train_sample_seconds/2000) + 20*102000*(test_sample_seconds/2000)",
        },
        "training": {
            "windows_per_split": 4_981,
            "batches_per_split": batches_per_split,
            "epochs": CONFIG["epochs"],
            "raw_hours": training_raw_seconds / 3600.0,
            "buffered_hours": training_raw_seconds * PROJECTION_BUFFER / 3600.0,
            "formula": "100*ceil(4981/16)*(median_train_batch_s+median_validation_batch_s)",
        },
        "evaluation": {
            "cuda_rollout_steps": 20_000,
            "radial_autocorrelations": 40_040,
            "cuda_rollout_seconds": rollout_seconds,
            "radial_metric_seconds": radial_seconds,
            "raw_hours": evaluation_raw_seconds / 3600.0,
            "buffered_hours": evaluation_raw_seconds * PROJECTION_BUFFER / 3600.0,
            "formula": "20000*median_batch1_forward_s + 40040*median_radial_frame_s",
        },
        "total_raw_hours": raw_seconds / 3600.0,
        "total_buffered_hours": buffered_seconds / 3600.0,
        "t4_rate_usd_per_hour": T4_RATE_USD_PER_HOUR,
        "estimated_full_campaign_cost_usd": full_cost,
        "preflight_timeout_max_cost_usd": PREFLIGHT_MAX_COST_USD,
        "aggregate_cost_including_preflight_worst_case_usd": aggregate_with_preflight,
        "authorized_aggregate_cap_usd": AGGREGATE_AUTHORIZED_CAP_USD,
        "within_authorized_cap": within_cap,
        "limitations": [
            "generation timing includes solver streaming but not full HDF5 gzip writes",
            "training projection omits checkpoint serialization and scheduler overhead",
            "evaluation projection measures rollout and radial FFT work but not full HDF5 artifact I/O",
            "the 25 percent buffer is applied to all three projected stages",
        ],
    }


def runtime_metadata() -> dict[str, Any]:
    total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "started_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "host_ram_bytes": int(total_ram),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HF_JOB_ID",
                "HF_JOB_NAME",
                "HF_HOME",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }


def validate_output_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    artifact_root = Path("/artifacts").resolve(strict=False)
    if artifact_root not in resolved.parents:
        raise RuntimeError(f"output must be inside mounted /artifacts bucket: {resolved}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("/workspace/hf_jobs/spinodal_attempt2b_preflight_sources.sha256"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/artifacts/hf-jobs/spinodal-attempt2b/preflight-v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_output_path(args.output)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "paper_id": PAPER_ID,
        "attempt": ATTEMPT,
        "status": "running",
        "runtime": runtime_metadata(),
        "job_contract": {
            "authorized_flavor": "t4-small",
            "outer_timeout_minutes": OUTER_TIMEOUT_MINUTES,
            "internal_timeout_minutes": INTERNAL_TIMEOUT_MINUTES,
            "t4_rate_usd_per_hour": T4_RATE_USD_PER_HOUR,
            "preflight_timeout_max_cost_usd": PREFLIGHT_MAX_COST_USD,
            "aggregate_authorized_cap_usd": AGGREGATE_AUTHORIZED_CAP_USD,
            "source_mount": "/workspace (read-only local volumes)",
            "artifact_mount": "/artifacts (read-write DineshAI/1KRpajnd6u-artifacts bucket)",
            "artifact_path": str(args.output),
        },
        "scientific_protocol": {
            **CONFIG,
            "grid": [GRID, GRID],
            "model": "FluxNet_D_pf_100dt",
            "trajectory_count": 22,
            "retained_frames": 30_022,
            "uncompressed_phi_bytes": 1_967_521_792,
            "full_solver_updates": 2_144_000,
        },
        "calibration_contract": {
            "oracle_steps": ORACLE_STEPS,
            "generation_benchmark_steps_per_schedule": GENERATION_BENCHMARK_STEPS,
            "training_warmup_batches": TRAIN_WARMUP_BATCHES,
            "training_timed_batches": TRAIN_TIMED_BATCHES,
            "validation_timed_batches": VALIDATION_TIMED_BATCHES,
            "rollout_timed_steps": ROLLOUT_TIMED_STEPS,
            "radial_timed_frames": RADIAL_TIMED_FRAMES,
            "full_dataset_generated": False,
            "scientific_checkpoint_written": False,
        },
        "errors": [],
    }
    started = time.monotonic()

    def deadline_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"internal {INTERNAL_TIMEOUT_MINUTES}-minute preflight deadline reached")

    signal.signal(signal.SIGALRM, deadline_handler)
    signal.alarm(INTERNAL_TIMEOUT_MINUTES * 60)
    exit_code = 0
    try:
        report["source_verification"] = verify_sources(args.source_root, args.source_manifest)
        report["cuda_runtime"] = cuda_runtime()
        with tempfile.TemporaryDirectory(prefix="fluxnet-spinodal-preflight-") as directory:
            temporary = Path(directory)
            binary = temporary / "spinodal_solver"
            report["solver_build"] = compile_solver(
                args.source_root / SOURCE_RELATIVE_PATHS["solver"], binary
            )
            report["solver_oracle"] = solver_oracle(binary, temporary)
            train_frames, train_generation = run_solver(
                binary,
                steps=GENERATION_BENCHMARK_STEPS,
                save_start=TRAIN_SAVE_INTERVAL,
                save_interval=TRAIN_SAVE_INTERVAL,
                threads=4,
                seed=12345,
            )
            test_frames, test_generation = run_solver(
                binary,
                steps=GENERATION_BENCHMARK_STEPS,
                save_start=TEST_SAVE_INTERVAL,
                save_interval=TEST_SAVE_INTERVAL,
                threads=4,
                seed=22345,
            )
            report["generation_benchmark"] = {
                "train_val_schedule": train_generation,
                "test_schedule": test_generation,
                "total_calibration_frames": len(train_frames) + len(test_frames),
                "total_calibration_bytes": int(train_frames.nbytes + test_frames.nbytes),
                "full_dataset_generated": False,
            }
            report["cuda_benchmark"] = time_cuda_training(
                args.source_root / SOURCE_RELATIVE_PATHS["released_model"], train_frames
            )
            report["radial_metric_benchmark"] = time_radial_metrics(test_frames)
        report["projection"] = projection(
            report["generation_benchmark"]["train_val_schedule"],
            report["generation_benchmark"]["test_schedule"],
            report["cuda_benchmark"],
            report["radial_metric_benchmark"],
        )
        report["gates"] = {
            "source_and_config_verified": report["source_verification"]["verified"],
            "nvidia_t4_verified": "T4" in report["cuda_runtime"]["name"].upper(),
            "cpu_numpy_oracle_passed": report["solver_oracle"]["passed"],
            "cuda_memory_under_12_gib": report["cuda_benchmark"]["cuda_memory_gate_passed"],
            "projection_within_aggregate_10_usd_cap": report["projection"]["within_authorized_cap"],
            "full_dataset_not_generated": not report["generation_benchmark"]["full_dataset_generated"],
        }
        report["gates"]["eligible_for_separately_authorized_full_campaign"] = all(
            report["gates"].values()
        )
        if not report["gates"]["eligible_for_separately_authorized_full_campaign"]:
            raise RuntimeError(f"preflight gate failed: {report['gates']}")
        report["status"] = "complete"
    except BaseException as error:
        exit_code = 1
        report["status"] = "failed"
        report["errors"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback_tail": traceback.format_exc(limit=12)[-12000:],
            }
        )
    finally:
        signal.alarm(0)
        report["finished_at"] = utc_now()
        report["preflight_wall_seconds"] = time.monotonic() - started
        report["peak_host_rss_mib"] = peak_rss_mib()
        artifact_sha256 = atomic_json(args.output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "artifact": str(args.output),
                    "sha256": artifact_sha256,
                    "bytes": args.output.stat().st_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
