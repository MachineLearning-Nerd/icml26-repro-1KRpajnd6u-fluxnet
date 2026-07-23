"""Checkpoint-safe data preparation for the preregistered spinodal Attempt 2b.

The released CUDA generator only writes text DAT files and is not runnable on
this host.  This wrapper compiles the independently tested C++/OpenMP port,
streams its float32 frames directly into atomic HDF5 files, and records enough
provenance to reject stale or mixed data before training begins.

Generation and training are explicit stages so their resource gates can be
enforced before work begins. Evaluation is added only after the one-epoch gate
has produced a validated, deterministically resumable checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch import nn


sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "official"
SOLVER_SOURCE = ROOT / "repro" / "src" / "spinodal_solver.cpp"
PREREGISTRATION = ROOT / "docs" / "claim3_spinodal_attempt2b_preregistration.md"
DEFAULT_DATA_DIR = ROOT / "data" / "spinodal_attempt2b"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "spinodal_attempt2b"
OFFICIAL_COMMIT = "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c"
GRID = 128
FRAME_BYTES = GRID * GRID * np.dtype(np.float32).itemsize
SOURCE_PATHS = {
    "released_generator": OFFICIAL / "dataset" / "spinodal_decomposition" / "phase_field_generator.cu",
    "released_test_generator": OFFICIAL / "dataset" / "spinodal_decomposition" / "phase_field_generator_test.cu",
    "released_model": OFFICIAL / "src" / "models" / "fluxnet_d_2d.py",
    "released_dataloader": OFFICIAL / "src" / "training" / "dataloader.py",
    "released_trainer": OFFICIAL / "src" / "training" / "trainer_unified.py",
    "released_100dt_config": OFFICIAL
    / "experiments"
    / "spinodal_decomposition"
    / "run_single_seed_100dt.py",
    "solver_port": SOLVER_SOURCE,
    "attempt2b_wrapper": Path(__file__).resolve(),
    "preregistration": PREREGISTRATION,
}


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    ndt: int = 10
    unroll_steps: int = 2
    base_channels: int = 32
    num_blocks: int = 4
    kernel_size: int = 5
    neighborhood_size: int = 5
    scheduler_patience: int = 15
    scheduler_factor: float = 0.5
    p_loss_weight: float = 0.25
    stability_loss_weight: float = 0.25
    dcl_weight: float = 0.5
    dcl_n_weight: float = 0.5
    mass_drift_threshold: float = 1.0e-5
    final_mae_threshold: float = 4.32e-2
    radial_ratio_threshold: float = 1.25


@dataclass(frozen=True)
class TrajectoryPlan:
    split: str
    seed: int
    steps: int
    save_start: int
    save_interval: int

    @property
    def frame_count(self) -> int:
        return (self.steps - self.save_start) // self.save_interval + 1

    @property
    def relative_path(self) -> Path:
        return Path(self.split) / f"seed_{self.seed}.h5"


def emit(event: str, log_path: Path | None = None, **values: Any) -> None:
    record = {"event": event, "unix_time": time.time(), **values}
    line = json.dumps(record, sort_keys=True, default=str)
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def blocked_pids(pids: Iterable[int]) -> list[int]:
    active = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            active.append(pid)
            continue
        active.append(pid)
    return active


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / (1024.0 * 1024.0)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_plan() -> list[TrajectoryPlan]:
    plans = [
        TrajectoryPlan("train", 12345, 52000, 2000, 10),
        TrajectoryPlan("val", 67890, 52000, 2000, 10),
    ]
    plans.extend(
        TrajectoryPlan("test", 22345 + 12345 * index, 102000, 2000, 100)
        for index in range(20)
    )
    return plans


def verify_official_pin() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official checkout is {commit}, expected {OFFICIAL_COMMIT}")
    status = subprocess.run(
        ["git", "-C", str(OFFICIAL), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("official checkout is not clean")
    missing = [str(path) for path in SOURCE_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned sources: {missing}")
    return {
        "official_commit": commit,
        "source_sha256": {name: sha256_file(path) for name, path in SOURCE_PATHS.items()},
    }


def compile_solver(output_path: Path) -> dict[str, Any]:
    libomp = Path("/opt/homebrew/opt/libomp")
    command = [
        "clang++",
        "-std=c++17",
        "-O3",
        "-Xpreprocessor",
        "-fopenmp",
        "-ffp-contract=off",
        "-fno-fast-math",
        f"-I{libomp / 'include'}",
        f"-L{libomp / 'lib'}",
        "-lomp",
        str(SOLVER_SOURCE),
        "-o",
        str(output_path),
    ]
    command_sha256 = canonical_sha256(command)
    metadata_path = output_path.with_suffix(".build.json")
    if output_path.is_file() and metadata_path.is_file():
        recorded = json.loads(metadata_path.read_text())
        if (
            recorded.get("command_sha256") == command_sha256
            and recorded.get("solver_source_sha256") == sha256_file(SOLVER_SOURCE)
            and recorded.get("binary_sha256") == sha256_file(output_path)
        ):
            return recorded
        raise RuntimeError(
            f"refusing to replace solver with incompatible build metadata: {metadata_path}"
        )
    if output_path.exists() or metadata_path.exists():
        raise RuntimeError(f"incomplete solver build state: {output_path}, {metadata_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    compile_command = [*command[:-1], str(temporary)]
    try:
        subprocess.run(compile_command, check=True, capture_output=True, text=True)
        os.chmod(temporary, 0o755)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    provenance = {
        "command": command,
        "command_sha256": command_sha256,
        "solver_source_sha256": sha256_file(SOLVER_SOURCE),
        "binary_sha256": sha256_file(output_path),
    }
    atomic_json(metadata_path, provenance)
    return provenance


def solver_command(binary: Path, plan: TrajectoryPlan, threads: int) -> list[str]:
    return [
        str(binary),
        "--seed",
        str(plan.seed),
        "--steps",
        str(plan.steps),
        "--save-start",
        str(plan.save_start),
        "--save-interval",
        str(plan.save_interval),
        "--threads",
        str(threads),
    ]


def _read_exact(stream: Any, byte_count: int) -> bytes:
    chunks = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def generate_trajectory(
    path: Path,
    binary: Path,
    plan: TrajectoryPlan,
    threads: int,
    provenance: dict[str, Any],
    compile_provenance: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    command = solver_command(binary, plan, threads)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout is not None
        with h5py.File(temporary, "w") as handle:
            dataset = handle.create_dataset(
                "phi_data",
                shape=(plan.frame_count, GRID, GRID),
                dtype=np.float32,
                chunks=(1, GRID, GRID),
                compression="gzip",
                shuffle=True,
            )
            for frame_index in range(plan.frame_count):
                payload = _read_exact(process.stdout, FRAME_BYTES)
                if len(payload) != FRAME_BYTES:
                    raise RuntimeError(
                        f"solver ended during frame {frame_index}: got {len(payload)} of {FRAME_BYTES} bytes"
                    )
                dataset[frame_index] = np.frombuffer(payload, dtype=np.float32).reshape(GRID, GRID)
            if process.stdout.read(1):
                raise RuntimeError("solver emitted more frames than the fixed plan")
            metadata = handle.create_group("metadata")
            metadata.attrs.update(
                {
                    "split": plan.split,
                    "seed": plan.seed,
                    "steps": plan.steps,
                    "save_start": plan.save_start,
                    "save_interval": plan.save_interval,
                    "base_dt": 0.01,
                    "grid": GRID,
                    "official_commit": provenance["official_commit"],
                    "solver_source_sha256": provenance["source_sha256"]["solver_port"],
                    "solver_binary_sha256": compile_provenance["binary_sha256"],
                    "rng": "std::mt19937_64 uniform_real_distribution<double>",
                    "serialization": "round-to-six-decimals then float32",
                }
            )
            handle.create_dataset(
                "base_steps",
                data=np.arange(
                    plan.save_start,
                    plan.steps + 1,
                    plan.save_interval,
                    dtype=np.int64,
                ),
            )
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"solver exited {return_code}: {stderr[-4000:]}")
        expected_completion = f"complete steps={plan.steps} frames={plan.frame_count} threads={threads}"
        if expected_completion not in stderr:
            raise RuntimeError(f"solver completion record missing: {stderr[-4000:]}")
        os.replace(temporary, path)
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()
    return inspect_trajectory(path, plan, provenance, compile_provenance)


def inspect_trajectory(
    path: Path,
    plan: TrajectoryPlan,
    provenance: dict[str, Any],
    compile_provenance: dict[str, Any],
) -> dict[str, Any]:
    minimum = math.inf
    maximum = -math.inf
    initial_mean = None
    maximum_relative_mass_drift = 0.0
    with h5py.File(path, "r") as handle:
        dataset = handle["phi_data"]
        if dataset.shape != (plan.frame_count, GRID, GRID) or dataset.dtype != np.dtype(np.float32):
            raise RuntimeError(f"invalid phi_data layout in {path}: {dataset.shape}, {dataset.dtype}")
        expected_steps = np.arange(plan.save_start, plan.steps + 1, plan.save_interval, dtype=np.int64)
        if not np.array_equal(handle["base_steps"][:], expected_steps):
            raise RuntimeError(f"base step schedule mismatch in {path}")
        metadata = handle["metadata"].attrs
        expected_metadata = {
            "split": plan.split,
            "seed": plan.seed,
            "steps": plan.steps,
            "save_start": plan.save_start,
            "save_interval": plan.save_interval,
            "grid": GRID,
            "official_commit": provenance["official_commit"],
            "solver_source_sha256": provenance["source_sha256"]["solver_port"],
            "solver_binary_sha256": compile_provenance["binary_sha256"],
        }
        for key, expected in expected_metadata.items():
            actual = metadata[key]
            if isinstance(expected, str):
                actual = str(actual)
            else:
                actual = int(actual)
            if actual != expected:
                raise RuntimeError(f"{key} mismatch in {path}: {actual!r} != {expected!r}")
        for frame_index in range(plan.frame_count):
            frame = dataset[frame_index]
            if not np.isfinite(frame).all():
                raise RuntimeError(f"non-finite frame {frame_index} in {path}")
            minimum = min(minimum, float(frame.min()))
            maximum = max(maximum, float(frame.max()))
            mean = float(frame.mean(dtype=np.float64))
            if initial_mean is None:
                initial_mean = mean
            scale = max(abs(initial_mean), 1.0e-12)
            maximum_relative_mass_drift = max(maximum_relative_mass_drift, abs(mean - initial_mean) / scale)
    if minimum < 0.0 or maximum > 1.0:
        raise RuntimeError(f"bound violation in generated reference {path}: [{minimum}, {maximum}]")
    return {
        **asdict(plan),
        "path": str(path),
        "shape": [plan.frame_count, GRID, GRID],
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "minimum": minimum,
        "maximum": maximum,
        "maximum_relative_mass_drift": maximum_relative_mass_drift,
    }


def dataset_digest(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        digest.update(f"{entry['path']}\0{entry['sha256']}\0{entry['bytes']}\n".encode())
    return digest.hexdigest()


def prepare_dataset(
    data_dir: Path,
    threads: int,
    provenance: dict[str, Any],
    log_path: Path | None,
) -> Path:
    binary = data_dir / "bin" / "spinodal_solver"
    compile_provenance = compile_solver(binary)
    entries = []
    plan = build_plan()
    for index, trajectory in enumerate(plan, start=1):
        path = data_dir / trajectory.relative_path
        if path.exists():
            entry = inspect_trajectory(path, trajectory, provenance, compile_provenance)
            disposition = "verified_existing"
        else:
            entry = generate_trajectory(
                path,
                binary,
                trajectory,
                threads,
                provenance,
                compile_provenance,
            )
            disposition = "generated"
        entry["path"] = str(trajectory.relative_path)
        entries.append(entry)
        emit(
            "spinodal_trajectory",
            log_path,
            index=index,
            total=len(plan),
            disposition=disposition,
            split=trajectory.split,
            seed=trajectory.seed,
            path=entry["path"],
            sha256=entry["sha256"],
        )
    manifest = {
        "schema_version": 1,
        "paper_target": "arXiv:2602.01941v1 Table 5, 100dt mechanism study",
        "claim_boundary": "128x128 released-training-scale mechanism study, not literal Table 5",
        **provenance,
        "compile": compile_provenance,
        "generator": {
            "grid": [GRID, GRID],
            "base_dt": 0.01,
            "precision": "float64 solver; six-decimal rounded float32 storage",
            "rng_deviation": "std::mt19937_64 replaces non-portable CURAND bitstream",
        },
        "split_counts": {
            split: sum(entry["split"] == split for entry in entries)
            for split in ("train", "val", "test")
        },
        "dataset_sha256": dataset_digest(entries),
        "files": entries,
    }
    manifest_path = data_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    emit(
        "spinodal_dataset_complete",
        log_path,
        manifest=str(manifest_path),
        dataset_sha256=manifest["dataset_sha256"],
    )
    return manifest_path


def verify_dataset(manifest_path: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest["official_commit"] != provenance["official_commit"]:
        raise RuntimeError("manifest official commit mismatch")
    if manifest["source_sha256"] != provenance["source_sha256"]:
        raise RuntimeError("manifest source hashes mismatch")
    if manifest["split_counts"] != {"train": 1, "val": 1, "test": 20}:
        raise RuntimeError(f"wrong fixed split counts: {manifest['split_counts']}")
    binary = manifest_path.parent / "bin" / "spinodal_solver"
    if not binary.is_file() or sha256_file(binary) != manifest["compile"]["binary_sha256"]:
        raise RuntimeError("compiled solver hash mismatch")
    expected_plan = {str(plan.relative_path): plan for plan in build_plan()}
    if {entry["path"] for entry in manifest["files"]} != set(expected_plan):
        raise RuntimeError("manifest paths do not match the fixed trajectory plan")
    entries = []
    for recorded in manifest["files"]:
        plan = expected_plan[recorded["path"]]
        path = manifest_path.parent / recorded["path"]
        inspected = inspect_trajectory(path, plan, provenance, manifest["compile"])
        inspected["path"] = recorded["path"]
        for key in ("sha256", "bytes", "shape"):
            if inspected[key] != recorded[key]:
                raise RuntimeError(f"{key} mismatch for {path}")
        entries.append(inspected)
    actual_digest = dataset_digest(entries)
    if actual_digest != manifest["dataset_sha256"]:
        raise RuntimeError("aggregate dataset hash mismatch")
    return manifest


def build_model(config: RunConfig) -> nn.Module:
    module = load_module(SOURCE_PATHS["released_model"], "author_fluxnet_d_2d")
    return module.FluxNet_D(
        in_channels=1,
        base_channels=config.base_channels,
        num_blocks=config.num_blocks,
        kernel_size=config.kernel_size,
        neighborhood_size=config.neighborhood_size,
        lower_bound=0.0,
        upper_bound=1.0,
        learnable_lower_bound=False,
        learnable_upper_bound=False,
    )


def load_training_arrays(manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for split, seed in (("train", 12345), ("val", 67890)):
        path = manifest_path.parent / split / f"seed_{seed}.h5"
        with h5py.File(path, "r") as handle:
            arrays.append(handle["phi_data"][:].astype(np.float32, copy=False))
    return arrays[0], arrays[1]


def window_count(trajectory: np.ndarray, config: RunConfig) -> int:
    return trajectory.shape[0] - config.ndt * config.unroll_steps


def make_batch(
    trajectory: np.ndarray,
    indices: np.ndarray,
    config: RunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = trajectory[indices, None]
    targets = np.stack(
        [trajectory[indices + offset * config.ndt] for offset in range(1, config.unroll_steps + 1)],
        axis=1,
    )
    return inputs, targets


def predict_with_dcl(model: nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prediction, outflow, inflow = model(inputs)
    return prediction, torch.mean((outflow - inflow) ** 2)


def batch_losses(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: RunConfig,
) -> dict[str, torch.Tensor]:
    first, dcl = predict_with_dcl(model, inputs)
    prediction = torch.mean((first - targets[:, 0:1]) ** 2)
    current = inputs
    with torch.no_grad():
        for _ in range(config.unroll_steps - 1):
            current, _ = predict_with_dcl(model, current)
    terminal, dcl_n = predict_with_dcl(model, current)
    stability = torch.mean((terminal - targets[:, -1:]) ** 2)
    total = (
        config.p_loss_weight * prediction
        + config.stability_loss_weight * stability
        + config.dcl_weight * dcl
        + config.dcl_n_weight * dcl_n
    )
    return {
        "prediction": prediction,
        "stability": stability,
        "dcl": dcl,
        "dcl_n": dcl_n,
        "total": total,
    }


def _epoch(
    model: nn.Module,
    trajectory: np.ndarray,
    indices: np.ndarray,
    config: RunConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    totals = {name: 0.0 for name in ("prediction", "stability", "dcl", "dcl_n", "total")}
    examples = 0
    model.train(optimizer is not None)
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for start in range(0, len(indices), config.batch_size):
            selected = indices[start : start + config.batch_size]
            x_numpy, y_numpy = make_batch(trajectory, selected, config)
            inputs = torch.from_numpy(x_numpy).to(device)
            targets = torch.from_numpy(y_numpy).to(device)
            losses = batch_losses(model, inputs, targets, config)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
            count = len(selected)
            examples += count
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu()) * count
    synchronize(device)
    return {name: value / examples for name, value in totals.items()}


def checkpoint_fingerprint(
    config: RunConfig,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    return canonical_sha256(
        {
            "model": "FluxNet_D_pf_100dt",
            "config": asdict(config),
            "dataset_sha256": manifest["dataset_sha256"],
            "source_sha256": provenance["source_sha256"],
            "official_commit": provenance["official_commit"],
        }
    )


def load_training_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("torch_rng_state", "mps_rng_state"):
        state = checkpoint.get(key)
        if isinstance(state, torch.Tensor):
            checkpoint[key] = state.cpu()
    return checkpoint


def train_model(
    train: np.ndarray,
    validation: np.ndarray,
    config: RunConfig,
    device: torch.device,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    output_dir: Path,
    log_path: Path,
    resume: bool,
    max_new_epochs: int | None,
) -> Path:
    if max_new_epochs is not None and max_new_epochs <= 0:
        raise ValueError("max-new-epochs must be positive")
    if train.shape != (5001, GRID, GRID) or validation.shape != (5001, GRID, GRID):
        raise RuntimeError(f"unexpected train/validation shapes: {train.shape}, {validation.shape}")
    set_determinism(config.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    model_dir = output_dir / "models" / "FluxNet_D_pf_100dt"
    latest_path = model_dir / "latest_checkpoint.pt"
    best_path = model_dir / "best_checkpoint.pt"
    fingerprint = checkpoint_fingerprint(config, manifest, provenance)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_loss = math.inf
    shuffle_rng = np.random.default_rng(config.seed)

    if resume and latest_path.exists():
        checkpoint = load_training_checkpoint(latest_path)
        if checkpoint["fingerprint"] != fingerprint:
            raise RuntimeError(f"refusing incompatible checkpoint {latest_path}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = checkpoint["history"]
        best_loss = float(checkpoint["best_validation_loss"])
        start_epoch = int(checkpoint["completed_epochs"])
        shuffle_rng.bit_generator.state = checkpoint["shuffle_rng_state"]
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "mps" and checkpoint.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng_state"])
        emit("spinodal_checkpoint_resumed", log_path, completed_epochs=start_epoch, path=str(latest_path))

    train_indices = np.arange(window_count(train, config), dtype=np.int64)
    validation_indices = np.arange(window_count(validation, config), dtype=np.int64)
    if len(train_indices) != 4981 or len(validation_indices) != 4981:
        raise RuntimeError("100dt training windows do not match the fixed released-scale protocol")
    end_epoch = config.epochs if max_new_epochs is None else min(config.epochs, start_epoch + max_new_epochs)
    started = time.monotonic()
    for epoch in range(start_epoch, end_epoch):
        epoch_started = time.monotonic()
        training_metrics = _epoch(
            model,
            train,
            shuffle_rng.permutation(train_indices),
            config,
            device,
            optimizer,
        )
        validation_metrics = _epoch(model, validation, validation_indices, config, device, None)
        scheduler.step(validation_metrics["total"])
        is_best = validation_metrics["total"] < best_loss
        if is_best:
            best_loss = validation_metrics["total"]
        record = {
            "epoch": epoch + 1,
            "epoch_seconds": time.monotonic() - epoch_started,
            "run_seconds": time.monotonic() - started,
            "peak_rss_mib": peak_rss_mib(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": training_metrics,
            "validation": validation_metrics,
            "is_best": is_best,
        }
        history.append(record)
        checkpoint = {
            "schema_version": 1,
            "model_name": "FluxNet_D_pf_100dt",
            "fingerprint": fingerprint,
            "completed_epochs": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_validation_loss": best_loss,
            "history": history,
            "shuffle_rng_state": shuffle_rng.bit_generator.state,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "mps_rng_state": torch.mps.get_rng_state() if device.type == "mps" else None,
            "config": asdict(config),
            "dataset_sha256": manifest["dataset_sha256"],
            **provenance,
        }
        atomic_torch_save(latest_path, checkpoint)
        if is_best:
            atomic_torch_save(best_path, checkpoint)
        atomic_json(model_dir / "training_history.json", history)
        emit("spinodal_training_epoch", log_path, **record)

    if not best_path.exists():
        raise RuntimeError("no best checkpoint produced")
    event = "spinodal_training_complete" if end_epoch >= config.epochs else "spinodal_training_paused"
    emit(
        event,
        log_path,
        completed_epochs=end_epoch,
        target_epochs=config.epochs,
        best_validation_loss=best_loss,
        best_checkpoint=str(best_path),
        best_checkpoint_sha256=sha256_file(best_path),
        peak_rss_mib=peak_rss_mib(),
    )
    return best_path


def run_metadata(config: RunConfig, provenance: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "config": asdict(config),
        **provenance,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "environment": {
            key: os.environ.get(key)
            for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "loss_formula": "0.25*p + 0.25*stability + 0.5*dcl + 0.5*dcl_n",
    }


def load_evaluation_checkpoint(
    config: RunConfig,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], Path, Path]:
    model_dir = output_dir / "models" / "FluxNet_D_pf_100dt"
    best_path = model_dir / "best_checkpoint.pt"
    latest_path = model_dir / "latest_checkpoint.pt"
    best = load_training_checkpoint(best_path)
    latest = load_training_checkpoint(latest_path)
    expected = checkpoint_fingerprint(config, manifest, provenance)
    for path, checkpoint in ((best_path, best), (latest_path, latest)):
        if checkpoint["fingerprint"] != expected:
            raise RuntimeError(f"checkpoint provenance mismatch: {path}")
    if int(latest["completed_epochs"]) != config.epochs:
        raise RuntimeError(
            f"refusing evaluation of incomplete training: {latest['completed_epochs']} of {config.epochs} epochs"
        )
    return best, best_path, latest_path


def radial_autocorrelation(field: np.ndarray) -> np.ndarray:
    if field.shape != (GRID, GRID):
        raise ValueError(f"radial autocorrelation expects {(GRID, GRID)}, got {field.shape}")
    spectrum = np.fft.fft2(field.astype(np.float64, copy=False))
    image = np.fft.fftshift(np.fft.ifft2(spectrum * np.conj(spectrum)).real) / field.size
    rows, columns = np.indices(field.shape, dtype=np.float64)
    radii = np.sqrt((rows - GRID // 2) ** 2 + (columns - GRID // 2) ** 2)
    result = np.empty(GRID // 2, dtype=np.float64)
    for radius in range(GRID // 2):
        mask = (radii >= radius - 0.5) & (radii < radius + 0.5)
        result[radius] = float(image[mask].mean())
    return result


def trajectory_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    base_steps: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if prediction.shape != truth.shape or prediction.shape != (1001, GRID, GRID):
        raise RuntimeError(f"unexpected rollout shapes: {prediction.shape}, {truth.shape}")
    absolute = np.abs(prediction.astype(np.float64) - truth.astype(np.float64))
    mae = absolute.mean(axis=(1, 2))
    initial_mass = float(prediction[0].sum(dtype=np.float64))
    mass = prediction.sum(axis=(1, 2), dtype=np.float64)
    relative_mass_drift = np.abs(mass - initial_mass) / max(abs(initial_mass), 1.0e-12)
    lower = prediction < 0.0
    upper = prediction > 1.0
    lower_count = lower.sum(axis=(1, 2))
    upper_count = upper.sum(axis=(1, 2))
    lower_magnitude = np.divide(
        np.maximum(-prediction, 0.0).sum(axis=(1, 2), dtype=np.float64),
        lower_count,
        out=np.zeros_like(relative_mass_drift),
        where=lower_count > 0,
    )
    upper_magnitude = np.divide(
        np.maximum(prediction - 1.0, 0.0).sum(axis=(1, 2), dtype=np.float64),
        upper_count,
        out=np.zeros_like(relative_mass_drift),
        where=upper_count > 0,
    )
    prediction_radial = np.stack([radial_autocorrelation(frame) for frame in prediction])
    truth_radial = np.stack([radial_autocorrelation(frame) for frame in truth])
    radial_error = np.abs(prediction_radial - truth_radial).mean(axis=1)
    phase_fraction = (prediction >= 0.6).mean(axis=(1, 2))
    late = base_steps >= 52000
    normalized_time = (base_steps[late] - 52000) / 50000.0
    radial_auc = float(np.trapezoid(radial_error[late], x=normalized_time))
    summary = {
        "finite": bool(np.isfinite(prediction).all()),
        "final_mae": float(mae[-1]),
        "maximum_relative_mass_drift": float(relative_mass_drift.max()),
        "minimum_prediction": float(np.nanmin(prediction)),
        "maximum_prediction": float(np.nanmax(prediction)),
        "lower_violation_rate": float(lower.mean()),
        "upper_violation_rate": float(upper.mean()),
        "maximum_conditional_lower_violation_magnitude": float(lower_magnitude.max()),
        "maximum_conditional_upper_violation_magnitude": float(upper_magnitude.max()),
        "radial_error_auc_T1_T2": radial_auc,
    }
    raw = {
        "mae": mae.astype(np.float32),
        "relative_mass_drift": relative_mass_drift.astype(np.float64),
        "lower_violation_rate": lower.mean(axis=(1, 2)).astype(np.float64),
        "upper_violation_rate": upper.mean(axis=(1, 2)).astype(np.float64),
        "conditional_lower_violation_magnitude": lower_magnitude.astype(np.float64),
        "conditional_upper_violation_magnitude": upper_magnitude.astype(np.float64),
        "phase_fraction_ge_0_6": phase_fraction.astype(np.float64),
        "prediction_radial": prediction_radial.astype(np.float64),
        "truth_radial": truth_radial.astype(np.float64),
        "radial_error": radial_error.astype(np.float64),
    }
    return summary, raw


def bootstrap_ratio_interval(
    matched_aucs: np.ndarray,
    intrinsic_aucs: np.ndarray,
    seed: int,
    replicates: int = 10000,
) -> dict[str, float]:
    if matched_aucs.ndim != 1 or intrinsic_aucs.ndim != 1 or not len(matched_aucs) or not len(intrinsic_aucs):
        raise ValueError("bootstrap inputs must be nonempty vectors")
    rng = np.random.default_rng(seed)
    ratios = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        numerator = rng.choice(matched_aucs, size=len(matched_aucs), replace=True).mean()
        denominator = rng.choice(intrinsic_aucs, size=len(intrinsic_aucs), replace=True).mean()
        ratios[index] = numerator / denominator if denominator > 0 else math.inf
    point_denominator = intrinsic_aucs.mean()
    point = matched_aucs.mean() / point_denominator if point_denominator > 0 else math.inf
    return {
        "point": float(point),
        "lower_95": float(np.percentile(ratios, 2.5)),
        "upper_95": float(np.percentile(ratios, 97.5)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def evaluate_model(
    manifest_path: Path,
    manifest: dict[str, Any],
    config: RunConfig,
    device: torch.device,
    provenance: dict[str, Any],
    output_dir: Path,
    log_path: Path,
) -> Path:
    checkpoint, checkpoint_path, latest_path = load_evaluation_checkpoint(
        config, manifest, provenance, output_dir
    )
    set_determinism(config.seed)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    test_plans = [item for item in build_plan() if item.split == "test"]
    entries = []
    radial_truth_by_seed: dict[int, np.ndarray] = {}
    evaluation_started = time.monotonic()
    for index, plan in enumerate(test_plans, start=1):
        source_path = manifest_path.parent / plan.relative_path
        with h5py.File(source_path, "r") as handle:
            truth = handle["phi_data"][:].astype(np.float32, copy=False)
            base_steps = handle["base_steps"][:]
        prediction = np.empty_like(truth)
        prediction[0] = truth[0]
        trajectory_started = time.monotonic()
        with torch.no_grad():
            current = torch.from_numpy(truth[0:1, None]).to(device)
            for step in range(1, len(truth)):
                current, _ = predict_with_dcl(model, current)
                prediction[step] = current[0, 0].detach().cpu().numpy()
        synchronize(device)
        summary, raw = trajectory_metrics(prediction, truth, base_steps)
        radial_truth_by_seed[plan.seed] = raw["truth_radial"]
        artifact_path = output_dir / "predictions" / f"seed_{plan.seed}.h5"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_name(f".{artifact_path.name}.{os.getpid()}.tmp")
        try:
            with h5py.File(temporary, "w") as handle:
                handle.create_dataset(
                    "prediction",
                    data=prediction,
                    chunks=(1, GRID, GRID),
                    compression="gzip",
                    shuffle=True,
                )
                handle.create_dataset("base_steps", data=base_steps)
                metrics_group = handle.create_group("metrics")
                for name, values in raw.items():
                    metrics_group.create_dataset(name, data=values, compression="gzip")
                metadata = handle.create_group("metadata")
                metadata.attrs.update(
                    {
                        "seed": plan.seed,
                        "source_path": str(plan.relative_path),
                        "source_sha256": sha256_file(source_path),
                        "checkpoint_sha256": checkpoint_sha256,
                        "dataset_sha256": manifest["dataset_sha256"],
                        "official_commit": provenance["official_commit"],
                    }
                )
            os.replace(temporary, artifact_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        entry = {
            "seed": plan.seed,
            "source_path": str(plan.relative_path),
            "prediction_path": str(artifact_path.relative_to(output_dir)),
            "prediction_sha256": sha256_file(artifact_path),
            "trajectory_seconds": time.monotonic() - trajectory_started,
            **summary,
        }
        entries.append(entry)
        emit("spinodal_evaluation_trajectory", log_path, index=index, total=len(test_plans), **entry)

    matched_aucs = np.asarray([entry["radial_error_auc_T1_T2"] for entry in entries], dtype=np.float64)
    intrinsic_aucs = []
    late_steps = np.arange(52000, 102001, 100, dtype=np.int64)
    normalized_time = (late_steps - 52000) / 50000.0
    for first, second in zip(test_plans[::2], test_plans[1::2]):
        first_radial = radial_truth_by_seed[first.seed][500:]
        second_radial = radial_truth_by_seed[second.seed][500:]
        per_time_error = np.abs(first_radial.astype(np.float64) - second_radial.astype(np.float64)).mean(axis=1)
        intrinsic_aucs.append(float(np.trapezoid(per_time_error, x=normalized_time)))
    intrinsic_aucs_array = np.asarray(intrinsic_aucs, dtype=np.float64)
    radial_ratio = bootstrap_ratio_interval(matched_aucs, intrinsic_aucs_array, config.seed)
    finite_all = all(entry["finite"] for entry in entries)
    maximum_mass_drift = max(entry["maximum_relative_mass_drift"] for entry in entries)
    mean_final_mae = float(np.mean([entry["final_mae"] for entry in entries]))
    supports = bool(
        finite_all
        and maximum_mass_drift <= config.mass_drift_threshold
        and mean_final_mae <= config.final_mae_threshold
        and radial_ratio["upper_95"] <= config.radial_ratio_threshold
    )
    falsified = bool(radial_ratio["lower_95"] > config.radial_ratio_threshold)
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
        "maximum_relative_mass_drift": maximum_mass_drift,
        "radial_auc_ratio": radial_ratio,
        "matched_radial_aucs": matched_aucs.tolist(),
        "intrinsic_pair_radial_aucs": intrinsic_aucs_array.tolist(),
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
        "evaluation_seconds": time.monotonic() - evaluation_started,
        "peak_rss_mib": peak_rss_mib(),
        "entries": entries,
        **provenance,
    }
    result_path = output_dir / "evaluation.json"
    atomic_json(result_path, result)
    emit("spinodal_evaluation_complete", log_path, result=str(result_path), **{k: result[k] for k in (
        "verdict", "mean_final_mae", "maximum_relative_mass_drift", "radial_auc_ratio", "evaluation_seconds", "peak_rss_mib"
    )})
    return result_path


def plan_summary(provenance: dict[str, Any]) -> dict[str, Any]:
    plan = build_plan()
    return {
        "status": "not_executed",
        "official_commit": provenance["official_commit"],
        "source_sha256": provenance["source_sha256"],
        "trajectory_count": len(plan),
        "split_counts": {
            split: sum(item.split == split for item in plan)
            for split in ("train", "val", "test")
        },
        "total_saved_frames": sum(item.frame_count for item in plan),
        "uncompressed_phi_bytes": sum(item.frame_count * FRAME_BYTES for item in plan),
        "plan": [{**asdict(item), "frame_count": item.frame_count, "path": str(item.relative_path)} for item in plan],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("plan", "generate", "verify", "train", "evaluate"),
        default="plan",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--block-pid", type=int, action="append", default=[])
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-epochs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    provenance = verify_official_pin()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "events.jsonl"
    if args.stage == "plan":
        summary = plan_summary(provenance)
        atomic_json(args.output_dir / "plan.json", summary)
        emit("spinodal_plan", log_path, **summary)
        return
    active = blocked_pids(args.block_pid)
    if active:
        raise RuntimeError(f"resource gate blocked by active PIDs: {active}")
    if args.stage == "generate":
        prepare_dataset(args.data_dir, args.threads, provenance, log_path)
        return
    manifest = verify_dataset(args.data_dir / "manifest.json", provenance)
    if args.stage in ("train", "evaluate"):
        if args.epochs <= 0 or args.batch_size <= 0:
            raise ValueError("epochs and batch-size must be positive")
        config = RunConfig(epochs=args.epochs, batch_size=args.batch_size)
        if config.epochs != 100:
            raise RuntimeError("target fingerprint is fixed at 100 epochs")
        if config.batch_size != 16:
            raise RuntimeError("preregistered training batch size is fixed at 16")
        device = resolve_device(args.device)
        if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0 for the measured MPS gate")
        atomic_json(args.output_dir / "run_metadata.json", run_metadata(config, provenance, device))
        if args.stage == "evaluate":
            evaluate_model(
                args.data_dir / "manifest.json",
                manifest,
                config,
                device,
                provenance,
                args.output_dir,
                log_path,
            )
            return
        train, validation = load_training_arrays(args.data_dir / "manifest.json")
        train_model(
            train,
            validation,
            config,
            device,
            manifest,
            provenance,
            args.output_dir,
            log_path,
            args.resume,
            args.max_new_epochs,
        )
        return
    emit(
        "spinodal_dataset_verified",
        log_path,
        dataset_sha256=manifest["dataset_sha256"],
        file_count=len(manifest["files"]),
    )


if __name__ == "__main__":
    main()
