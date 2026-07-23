"""Checkpointable preparation harness for FluxNet C3 shallow water Attempt 2.

This wrapper imports the pinned author generator and model classes without
modifying them. Its defaults are fixed to the two decisive arXiv-v1 Table 3
models and the complete 50/20/50 shallow-water split.
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
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "official"
DEFAULT_DATA_DIR = ROOT / "data" / "shallow_water_attempt2"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "shallow_water_attempt2"
OFFICIAL_COMMIT = "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c"
V1_PDF_SHA256 = "40b79be23b629ec42d41eae9bdd578120e9a401d9e2dbef54011b5ba6ec2a285"
MODEL_NAMES = ("FluxNet_SW_LAP_pf", "FNO_SW_Proj_box_mass_pf")
FIELD_NAMES = ("h", "mx", "my")
SOURCE_PATHS = {
    "generator": OFFICIAL / "dataset" / "shallow_water" / "dataset.py",
    "fluxnet_model": OFFICIAL / "src" / "models" / "fluxnet_sw_lap.py",
    "projection_baseline_model": OFFICIAL / "src" / "models" / "sw_baselines.py",
    "released_dataloader": OFFICIAL / "src" / "training" / "dataloader.py",
    "released_trainer": OFFICIAL / "src" / "training" / "trainer_unified.py",
    "released_single_seed_config": OFFICIAL / "experiments" / "shallow_water" / "run_single_seed.py",
}


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    unroll_steps: int = 5
    base_channels: int = 64
    num_blocks: int = 4
    kernel_size: int = 3
    neighborhood_size: int = 3
    fno_modes: int = 16
    fno_width: int = 64
    fno_layers: int = 4
    scheduler_patience: int = 15
    scheduler_factor: float = 0.5
    prediction_loss_weight: float = 0.5
    stability_loss_weight: float = 0.5
    train_horizon: float = 2.4
    test_horizon: float = 4.8
    divergence_threshold: float = 100.0
    mass_drift_threshold: float = 1e-5


def emit(event: str, log_path: Path | None = None, **values: Any) -> None:
    record = {"event": event, "unix_time": time.time(), **values}
    line = json.dumps(record, sort_keys=True, default=str)
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


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


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npz", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_official_pin() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official checkout is {commit}, expected {OFFICIAL_COMMIT}")
    for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        if subprocess.run(["git", "-C", str(OFFICIAL), *args], check=False).returncode:
            raise RuntimeError("official checkout has tracked changes")
    status = subprocess.run(
        ["git", "-C", str(OFFICIAL), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("official checkout is not clean")
    hashes = {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}
    return {"official_commit": commit, "source_sha256": hashes}


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / (1024.0 * 1024.0)


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


def build_split_plan(category_config: dict[str, dict[str, int]], global_seed: int) -> list[dict[str, Any]]:
    """Replicate the released seed draw and per-category stratified shuffle."""
    master_rng = np.random.default_rng(global_seed)
    plan: list[dict[str, Any]] = []
    for category, counts in category_config.items():
        seeds = [int(master_rng.integers(0, 2**31)) for _ in range(counts["total"])]
        indices = list(range(counts["total"]))
        master_rng.shuffle(indices)
        train_stop = counts["train"]
        val_stop = train_stop + counts["val"]
        split_indices = {
            "train": indices[:train_stop],
            "val": indices[train_stop:val_stop],
            "test": indices[val_stop:],
        }
        for split, selected in split_indices.items():
            for sample_index in selected:
                plan.append(
                    {
                        "split": split,
                        "category": category,
                        "sample_index": int(sample_index),
                        "sample_seed": seeds[sample_index],
                    }
                )
    return plan


def _write_trajectory(
    path: Path,
    generator: Any,
    record: dict[str, Any],
    horizon: float,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(record["sample_seed"])
    h0, mx0, my0, params = generator.GENERATORS[record["category"]](rng)
    result = generator.simulate_swe(
        h0,
        mx0,
        my0,
        horizon,
        generator.FIXED_DT,
        generator.TIME_DOWNSAMPLE,
    )
    times, h, mx, my, mass, momx, momy, h_min, h_max = result
    h = generator.downsample_spatial_conservative(h, generator.SPACE_DOWNSAMPLE).astype(np.float32)
    mx = generator.downsample_spatial_conservative(mx, generator.SPACE_DOWNSAMPLE).astype(np.float32)
    my = generator.downsample_spatial_conservative(my, generator.SPACE_DOWNSAMPLE).astype(np.float32)
    x_coarse, y_coarse, _, _ = generator.get_coarse_grid(generator.SPACE_DOWNSAMPLE)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as handle:
        for name, array in (("h", h), ("mx", mx), ("my", my)):
            handle.create_dataset(name, data=array, compression="gzip")
        handle.create_dataset("x", data=np.asarray(x_coarse, dtype=np.float32))
        handle.create_dataset("y", data=np.asarray(y_coarse, dtype=np.float32))
        handle.create_dataset("t", data=np.asarray(times, dtype=np.float32))
        for name, array in (
            ("mass", mass),
            ("momx", momx),
            ("momy", momy),
            ("h_min", h_min),
            ("h_max", h_max),
        ):
            handle.create_dataset(name, data=np.asarray(array, dtype=np.float32))
        metadata = handle.create_group("metadata")
        metadata.attrs.update(
            {
                "category": record["category"],
                "sample_index": record["sample_index"],
                "sample_seed": record["sample_seed"],
                "split": record["split"],
                "T_final": horizon,
                "dt": generator.FIXED_DT,
                "time_downsample": generator.TIME_DOWNSAMPLE,
                "space_downsample": generator.SPACE_DOWNSAMPLE,
                "Nx_original": generator.Nx,
                "Ny_original": generator.Ny,
                "Nx_saved": h.shape[-2],
                "Ny_saved": h.shape[-1],
                "official_commit": provenance["official_commit"],
                "generator_sha256": provenance["source_sha256"]["generator"],
            }
        )
        params_group = handle.create_group("params")
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                params_group.attrs[key] = value
    os.replace(temporary, path)
    return {
        **record,
        "path": str(path),
        "shape": list(h.shape),
        "horizon": horizon,
        "solver_max_absolute_drift": {
            "h": float(np.max(np.abs(mass - mass[0]))),
            "mx": float(np.max(np.abs(momx - momx[0]))),
            "my": float(np.max(np.abs(momy - momy[0]))),
        },
    }


def _inspect_existing(
    path: Path,
    record: dict[str, Any],
    horizon: float,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        shapes = [tuple(handle[name].shape) for name in FIELD_NAMES]
        if len(set(shapes)) != 1 or shapes[0][1:] != (64, 64):
            raise RuntimeError(f"invalid field shapes in {path}: {shapes}")
        metadata = handle["metadata"].attrs
        expected_metadata = {
            "category": record["category"],
            "sample_index": record["sample_index"],
            "split": record["split"],
        }
        for key, expected in expected_metadata.items():
            actual = metadata[key]
            if isinstance(expected, str):
                actual = str(actual)
            else:
                actual = int(actual)
            if actual != expected:
                raise RuntimeError(f"{key} mismatch in {path}: {actual!r} != {expected!r}")
        if int(metadata["sample_seed"]) != record["sample_seed"]:
            raise RuntimeError(f"sample seed mismatch in {path}")
        if not math.isclose(float(metadata["T_final"]), horizon):
            raise RuntimeError(f"horizon mismatch in {path}")
        if str(metadata["official_commit"]) != provenance["official_commit"]:
            raise RuntimeError(f"official commit mismatch in {path}")
        if str(metadata["generator_sha256"]) != provenance["source_sha256"]["generator"]:
            raise RuntimeError(f"generator hash mismatch in {path}")
        exact_numeric_metadata = {
            "dt": 0.004,
            "time_downsample": 10,
            "space_downsample": 2,
            "Nx_original": 128,
            "Ny_original": 128,
            "Nx_saved": 64,
            "Ny_saved": 64,
        }
        for key, expected in exact_numeric_metadata.items():
            actual = float(metadata[key]) if isinstance(expected, float) else int(metadata[key])
            if actual != expected:
                raise RuntimeError(f"{key} mismatch in {path}: {actual!r} != {expected!r}")
        times = handle["t"][:]
        if not math.isclose(float(times[0]), 0.0, abs_tol=1e-8) or not math.isclose(
            float(times[-1]), horizon, rel_tol=0.0, abs_tol=1e-5
        ):
            raise RuntimeError(f"time grid mismatch in {path}: {times[[0, -1]]}")
        drift = {}
        for field, key in (("h", "mass"), ("mx", "momx"), ("my", "momy")):
            values = handle[key][:]
            drift[field] = float(np.max(np.abs(values - values[0])))
    return {
        **record,
        "path": str(path),
        "shape": list(shapes[0]),
        "horizon": float(times[-1]),
        "solver_max_absolute_drift": drift,
    }


def prepare_dataset(data_dir: Path, provenance: dict[str, Any], log_path: Path | None) -> Path:
    generator = load_module(SOURCE_PATHS["generator"], "author_shallow_generator")
    plan = build_split_plan(generator.CATEGORY_CONFIG, generator.GLOBAL_SEED)
    entries = []
    for index, record in enumerate(plan, start=1):
        horizon = 4.8 if record["split"] == "test" else 2.4
        filename = (
            f"{record['category']}_sample{record['sample_index']:02d}_"
            f"seed{record['sample_seed']}.h5"
        )
        path = data_dir / record["split"] / filename
        if path.exists():
            entry = _inspect_existing(path, record, horizon, provenance)
            disposition = "verified_existing"
        else:
            entry = _write_trajectory(path, generator, record, horizon, provenance)
            disposition = "generated"
        entry["path"] = str(path.relative_to(data_dir))
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
        entries.append(entry)
        emit(
            "dataset_trajectory",
            log_path,
            index=index,
            total=len(plan),
            disposition=disposition,
            split=record["split"],
            category=record["category"],
            path=entry["path"],
            elapsed_peak_rss_mib=peak_rss_mib(),
        )

    dataset_digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        dataset_digest.update(f"{entry['path']}\0{entry['sha256']}\0{entry['bytes']}\n".encode())
    manifest = {
        "schema_version": 1,
        "paper_target": "arXiv:2602.01941v1 Table 3",
        "v1_pdf_sha256": V1_PDF_SHA256,
        **provenance,
        "generator_config": {
            "computation_grid": [128, 128],
            "saved_grid": [64, 64],
            "domain": [0.0, 10.0, 0.0, 10.0],
            "fixed_dt": 0.004,
            "saved_stride": 10,
            "train_horizon": 2.4,
            "test_horizon": 4.8,
            "global_seed": int(generator.GLOBAL_SEED),
        },
        "split_counts": {split: sum(item["split"] == split for item in entries) for split in ("train", "val", "test")},
        "category_counts": {
            split: {
                category: sum(item["split"] == split and item["category"] == category for item in entries)
                for category in generator.CATEGORY_CONFIG
            }
            for split in ("train", "val", "test")
        },
        "dataset_sha256": dataset_digest.hexdigest(),
        "files": entries,
    }
    manifest_path = data_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    emit("dataset_complete", log_path, manifest=str(manifest_path), dataset_sha256=manifest["dataset_sha256"])
    return manifest_path


def register_unique_trajectory_identity(
    entry: dict[str, Any], identities: set[tuple[str, int, int]]
) -> tuple[str, int, int]:
    """Reject reuse of one generated trajectory, including across data splits."""
    identity = (entry["category"], int(entry["sample_index"]), int(entry["sample_seed"]))
    if identity in identities:
        raise RuntimeError(f"duplicate trajectory identity across dataset splits: {identity}")
    identities.add(identity)
    return identity


def verify_dataset(manifest_path: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest["official_commit"] != provenance["official_commit"]:
        raise RuntimeError("dataset manifest official commit mismatch")
    if manifest["source_sha256"]["generator"] != provenance["source_sha256"]["generator"]:
        raise RuntimeError("dataset generator hash mismatch")
    if manifest["split_counts"] != {"train": 50, "val": 20, "test": 50}:
        raise RuntimeError(f"wrong split counts: {manifest['split_counts']}")
    expected_category_counts = {
        "train": {"CaseA1": 10, "CaseA2": 10, "CaseB1": 15, "CaseB2": 15},
        "val": {"CaseA1": 4, "CaseA2": 4, "CaseB1": 6, "CaseB2": 6},
        "test": {"CaseA1": 10, "CaseA2": 10, "CaseB1": 15, "CaseB2": 15},
    }
    if manifest["category_counts"] != expected_category_counts:
        raise RuntimeError(f"wrong stratification: {manifest['category_counts']}")
    digest = hashlib.sha256()
    identities: set[tuple[str, int, int]] = set()
    paths: set[str] = set()
    for entry in sorted(manifest["files"], key=lambda item: item["path"]):
        expected_shape = [121, 64, 64] if entry["split"] == "test" else [61, 64, 64]
        if entry["shape"] != expected_shape:
            raise RuntimeError(f"wrong trajectory shape for {entry['path']}: {entry['shape']}")
        path = manifest_path.parent / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if entry["path"] in paths:
            raise RuntimeError(f"duplicate data path: {entry['path']}")
        paths.add(entry["path"])
        register_unique_trajectory_identity(entry, identities)
        expected_horizon = 4.8 if entry["split"] == "test" else 2.4
        inspected = _inspect_existing(path, entry, expected_horizon, provenance)
        if inspected["shape"] != expected_shape:
            raise RuntimeError(f"HDF5 shape mismatch for {entry['path']}: {inspected['shape']}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"data hash mismatch: {path}")
        digest.update(f"{entry['path']}\0{actual}\0{path.stat().st_size}\n".encode())
    if len(identities) != 120:
        raise RuntimeError(f"expected 120 unique trajectories, got {len(identities)}")
    if digest.hexdigest() != manifest["dataset_sha256"]:
        raise RuntimeError("aggregate dataset hash mismatch")
    return manifest


def load_split(
    manifest_path: Path,
    manifest: dict[str, Any],
    split: str,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    entries = [entry for entry in manifest["files"] if entry["split"] == split]
    entries.sort(key=lambda item: (item["category"], item["sample_index"]))
    trajectories = []
    reference_times = None
    for entry in entries:
        with h5py.File(manifest_path.parent / entry["path"], "r") as handle:
            trajectories.append(np.stack([handle[name][:] for name in FIELD_NAMES], axis=1).astype(np.float32))
            times = handle["t"][:].astype(np.float64)
            if reference_times is None:
                reference_times = times
            elif not np.array_equal(reference_times, times):
                raise RuntimeError(f"nonuniform saved time grid in {split}")
    if reference_times is None:
        raise RuntimeError(f"empty split: {split}")
    return np.stack(trajectories), entries, reference_times


def window_count(trajectories: np.ndarray, unroll_steps: int) -> int:
    return trajectories.shape[0] * (trajectories.shape[1] - unroll_steps)


def make_batch(trajectories: np.ndarray, indices: np.ndarray, unroll_steps: int) -> tuple[np.ndarray, np.ndarray]:
    per_trajectory = trajectories.shape[1] - unroll_steps
    trajectory_indices = indices // per_trajectory
    starts = indices % per_trajectory
    inputs = trajectories[trajectory_indices, starts]
    targets = np.stack(
        [trajectories[trajectory_indices, starts + offset] for offset in range(1, unroll_steps + 1)],
        axis=1,
    )
    return inputs, targets


def build_model(name: str, config: RunConfig) -> nn.Module:
    if name == "FluxNet_SW_LAP_pf":
        module = load_module(SOURCE_PATHS["fluxnet_model"], "author_fluxnet_sw_lap")
        return module.FluxNet_SW_2D(
            base_channels=config.base_channels,
            num_blocks=config.num_blocks,
            kernel_size=config.kernel_size,
            neighborhood_size=config.neighborhood_size,
            lower_bound=0.0,
            head_config="LAP",
        )
    if name == "FNO_SW_Proj_box_mass_pf":
        module = load_module(SOURCE_PATHS["projection_baseline_model"], "author_sw_projection_baseline")
        return module.FNO_SW_Proj(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            num_layers=config.fno_layers,
            projection_mode="box_mass",
            prediction_mode="residual",
        )
    raise ValueError(f"unsupported model {name}; expected one of {MODEL_NAMES}")


def predict(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output = model(inputs)
    return output[0] if isinstance(output, tuple) else output


def batch_losses(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: RunConfig,
) -> dict[str, torch.Tensor]:
    first = predict(model, inputs)
    prediction = torch.mean((first - targets[:, 0]) ** 2)
    current = inputs
    with torch.no_grad():
        for _ in range(config.unroll_steps - 1):
            current = predict(model, current)
    terminal = predict(model, current)
    stability = torch.mean((terminal - targets[:, -1]) ** 2)
    total = config.prediction_loss_weight * prediction + config.stability_loss_weight * stability
    return {"prediction": prediction, "stability": stability, "total": total}


def _epoch(
    model: nn.Module,
    trajectories: np.ndarray,
    config: RunConfig,
    device: torch.device,
    indices: np.ndarray,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    totals = {"prediction": 0.0, "stability": 0.0, "total": 0.0}
    examples = 0
    model.train(optimizer is not None)
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for start in range(0, len(indices), config.batch_size):
            selected = indices[start : start + config.batch_size]
            x_numpy, y_numpy = make_batch(trajectories, selected, config.unroll_steps)
            x = torch.from_numpy(x_numpy).to(device)
            y = torch.from_numpy(y_numpy).to(device)
            losses = batch_losses(model, x, y, config)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
            count = len(selected)
            examples += count
            for key, value in losses.items():
                totals[key] += float(value.detach().cpu()) * count
    synchronize(device)
    return {key: value / examples for key, value in totals.items()}


def _checkpoint_fingerprint(config: RunConfig, manifest: dict[str, Any], provenance: dict[str, Any], name: str) -> str:
    return canonical_sha256(
        {
            "model": name,
            "config": asdict(config),
            "dataset_sha256": manifest["dataset_sha256"],
            "source_sha256": provenance["source_sha256"],
            "official_commit": provenance["official_commit"],
        }
    )


def load_training_checkpoint(path: Path) -> dict[str, Any]:
    """Load resume state on CPU so device RNG tensors remain valid ByteTensors."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("torch_rng_state", "mps_rng_state"):
        state = checkpoint.get(key)
        if isinstance(state, torch.Tensor):
            checkpoint[key] = state.cpu()
    cuda_states = checkpoint.get("cuda_rng_state_all")
    if cuda_states is not None:
        checkpoint["cuda_rng_state_all"] = [state.cpu() for state in cuda_states]
    return checkpoint


def train_model(
    name: str,
    train: np.ndarray,
    validation: np.ndarray,
    config: RunConfig,
    device: torch.device,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    output_dir: Path,
    resume: bool,
    log_path: Path,
    max_new_epochs: int | None = None,
) -> Path:
    if max_new_epochs is not None and max_new_epochs <= 0:
        raise ValueError("max_new_epochs must be positive")
    set_determinism(config.seed)
    model = build_model(name, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience
    )
    model_dir = output_dir / "models" / name
    latest_path = model_dir / "latest_checkpoint.pt"
    best_path = model_dir / "best_checkpoint.pt"
    fingerprint = _checkpoint_fingerprint(config, manifest, provenance, name)
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
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if device.type == "mps" and checkpoint.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng_state"])
        emit("checkpoint_resumed", log_path, model=name, completed_epochs=start_epoch, path=str(latest_path))

    train_indices = np.arange(window_count(train, config.unroll_steps), dtype=np.int64)
    validation_indices = np.arange(window_count(validation, config.unroll_steps), dtype=np.int64)
    started = time.monotonic()
    end_epoch = config.epochs if max_new_epochs is None else min(config.epochs, start_epoch + max_new_epochs)
    for epoch in range(start_epoch, end_epoch):
        epoch_started = time.monotonic()
        permutation = shuffle_rng.permutation(train_indices)
        training_metrics = _epoch(model, train, config, device, permutation, optimizer)
        validation_metrics = _epoch(model, validation, config, device, validation_indices, None)
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
            "model_name": name,
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
            "cuda_rng_state_all": (
                [state.cpu() for state in torch.cuda.get_rng_state_all()] if device.type == "cuda" else None
            ),
            "mps_rng_state": torch.mps.get_rng_state() if device.type == "mps" else None,
            "config": asdict(config),
            "dataset_sha256": manifest["dataset_sha256"],
            **provenance,
        }
        atomic_torch_save(latest_path, checkpoint)
        if is_best:
            atomic_torch_save(best_path, checkpoint)
        atomic_json(model_dir / "training_history.json", history)
        emit("training_epoch", log_path, model=name, **record)

    if not best_path.exists():
        raise RuntimeError(f"no best checkpoint produced for {name}")
    completed_epochs = end_epoch
    event = "training_complete" if completed_epochs >= config.epochs else "training_paused"
    emit(
        event,
        log_path,
        model=name,
        completed_epochs=completed_epochs,
        target_epochs=config.epochs,
        best_validation_loss=best_loss,
        best_checkpoint=str(best_path),
        best_checkpoint_sha256=sha256_file(best_path),
        peak_rss_mib=peak_rss_mib(),
    )
    return best_path


def compute_rollout_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    times: np.ndarray,
    config: RunConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    absolute = np.abs(prediction - truth)
    per_step_field_mae = absolute.mean(axis=(-2, -1))
    late = times > config.train_horizon + 1e-8
    finite = np.isfinite(prediction).all(axis=(1, 2, 3, 4))
    excessive = np.nanmax(np.abs(prediction), axis=(1, 2, 3, 4)) > config.divergence_threshold
    divergent = (~finite) | excessive
    totals = prediction.sum(axis=(-2, -1), dtype=np.float64)
    initial = totals[:, 0:1]
    absolute_drift = np.abs(totals - initial)
    scale = np.abs(prediction[:, 0]).sum(axis=(-2, -1), dtype=np.float64)[:, None, :] + 1e-12
    relative_l1_drift = absolute_drift / scale
    h_rollout = prediction[:, 1:, 0]
    h_violations = h_rollout < 0.0
    h_count = int(h_violations.sum())
    summary = {
        "trajectory_count": int(prediction.shape[0]),
        "rollout_steps": int(prediction.shape[1] - 1),
        "final_mae": {field: float(per_step_field_mae[:, -1, index].mean()) for index, field in enumerate(FIELD_NAMES)},
        "final_mae_trajectory_std": {
            field: float(per_step_field_mae[:, -1, index].std(ddof=1)) for index, field in enumerate(FIELD_NAMES)
        },
        "late_mae": {
            field: float(per_step_field_mae[:, late, index].mean()) for index, field in enumerate(FIELD_NAMES)
        },
        "divergent_trajectories": int(divergent.sum()),
        "h_violation_rate_percent": float(h_violations.mean() * 100.0),
        "h_conditional_violation_magnitude": (
            float(np.maximum(-h_rollout, 0.0).sum() / h_count) if h_count else 0.0
        ),
        "prediction_h_min": float(np.nanmin(prediction[:, :, 0])),
        "max_absolute_integral_drift": {
            field: float(absolute_drift[:, :, index].max()) for index, field in enumerate(FIELD_NAMES)
        },
        "max_relative_l1_normalized_integral_drift": {
            field: float(relative_l1_drift[:, :, index].max()) for index, field in enumerate(FIELD_NAMES)
        },
    }
    raw = {
        "per_step_field_mae": per_step_field_mae.astype(np.float32),
        "absolute_integral_drift": absolute_drift.astype(np.float32),
        "relative_l1_normalized_integral_drift": relative_l1_drift.astype(np.float32),
        "divergent": divergent,
    }
    return summary, raw


def load_evaluation_checkpoint(
    name: str,
    config: RunConfig,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], Path, Path]:
    checkpoint_path = output_dir / "models" / name / "best_checkpoint.pt"
    latest_path = output_dir / "models" / name / "latest_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    completion = torch.load(latest_path, map_location=device, weights_only=False)
    expected = _checkpoint_fingerprint(config, manifest, provenance, name)
    for path, candidate in ((checkpoint_path, checkpoint), (latest_path, completion)):
        if candidate["fingerprint"] != expected:
            raise RuntimeError(f"checkpoint provenance mismatch: {path}")
    if int(completion["completed_epochs"]) != config.epochs:
        raise RuntimeError(
            f"refusing evaluation of incomplete training: completed "
            f"{completion['completed_epochs']} of {config.epochs} epochs"
        )
    return checkpoint, checkpoint_path, latest_path


def evaluate_model(
    name: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    config: RunConfig,
    device: torch.device,
    provenance: dict[str, Any],
    output_dir: Path,
    evaluation_batch_size: int,
    log_path: Path,
) -> Path:
    checkpoint, checkpoint_path, latest_path = load_evaluation_checkpoint(
        name, config, manifest, provenance, output_dir, device
    )
    set_determinism(config.seed)
    model = build_model(name, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    truth, entries, times = load_split(manifest_path, manifest, "test")
    predictions = np.empty_like(truth)
    predictions[:, 0] = truth[:, 0]
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(truth), evaluation_batch_size):
            stop = min(start + evaluation_batch_size, len(truth))
            current = torch.from_numpy(truth[start:stop, 0]).to(device)
            for step in range(1, truth.shape[1]):
                current = predict(model, current)
                predictions[start:stop, step] = current.detach().cpu().numpy()
            emit(
                "evaluation_batch",
                log_path,
                model=name,
                completed=stop,
                total=len(truth),
                elapsed_seconds=time.monotonic() - started,
                peak_rss_mib=peak_rss_mib(),
            )
    synchronize(device)
    summary, raw_metrics = compute_rollout_metrics(predictions, truth, times, config)
    categories = np.asarray([entry["category"] for entry in entries])
    sample_ids = np.asarray(
        [f"{entry['category']}:{entry['sample_index']:02d}:seed{entry['sample_seed']}" for entry in entries]
    )
    per_case = {}
    for category in sorted(set(categories.tolist())):
        selected = categories == category
        per_case[category], _ = compute_rollout_metrics(predictions[selected], truth[selected], times, config)
    metrics = {
        "model": name,
        "summary": summary,
        "per_case": per_case,
        "config": asdict(config),
        "dataset_sha256": manifest["dataset_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_checkpoint_epoch": int(checkpoint["completed_epochs"]),
        "training_completion_checkpoint_sha256": sha256_file(latest_path),
        "official_commit": provenance["official_commit"],
        "source_sha256": provenance["source_sha256"],
        "evaluation_seconds": time.monotonic() - started,
        "peak_rss_mib": peak_rss_mib(),
    }
    metrics_path = output_dir / "metrics" / f"{name}.json"
    atomic_json(metrics_path, metrics)
    raw_path = output_dir / "raw" / f"{name}_rollouts.npz"
    atomic_npz(
        raw_path,
        model=np.asarray(name),
        sample_id=sample_ids,
        category=categories,
        times=times,
        truth=truth,
        prediction=predictions,
        **raw_metrics,
    )
    metrics["raw_artifact_sha256"] = sha256_file(raw_path)
    metrics["raw_artifact_bytes"] = raw_path.stat().st_size
    atomic_json(metrics_path, metrics)
    emit("evaluation_complete", log_path, model=name, metrics=str(metrics_path), raw=str(raw_path), **summary)
    return metrics_path


def compare_models(output_dir: Path, config: RunConfig) -> dict[str, Any]:
    metrics = {
        name: json.loads((output_dir / "metrics" / f"{name}.json").read_text()) for name in MODEL_NAMES
    }
    flux = metrics[MODEL_NAMES[0]]["summary"]
    projection = metrics[MODEL_NAMES[1]]["summary"]
    lower_final = all(flux["final_mae"][field] < projection["final_mae"][field] for field in FIELD_NAMES)
    lower_late = all(flux["late_mae"][field] < projection["late_mae"][field] for field in FIELD_NAMES)
    flux_mass_ok = all(
        flux["max_relative_l1_normalized_integral_drift"][field] <= config.mass_drift_threshold
        for field in FIELD_NAMES
    )
    direct_support = (
        lower_final
        and lower_late
        and flux["divergent_trajectories"] == 0
        and flux["h_violation_rate_percent"] == 0.0
        and flux_mass_ok
    )
    projection_wins = all(
        projection["final_mae"][field] <= flux["final_mae"][field]
        and projection["late_mae"][field] <= flux["late_mae"][field]
        for field in FIELD_NAMES
    )
    paper_flux = {"h": 3.12e-3, "mx": 4.41e-3, "my": 4.64e-3}
    paper_projection = {"h": 6.74e-3, "mx": 12.9e-3, "my": 11.5e-3}
    ratios = {field: flux["final_mae"][field] / projection["final_mae"][field] for field in FIELD_NAMES}
    anchor_consistent = all(flux["final_mae"][field] <= 2.0 * paper_flux[field] for field in FIELD_NAMES) and all(
        ratios[field] <= 2.0 * (paper_flux[field] / paper_projection[field]) for field in FIELD_NAMES
    )
    result = {
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
            "fluxnet_mass_drift_at_most_1e-5_all_fields": flux_mass_ok,
        },
        "paper_anchor_consistent": anchor_consistent,
        "controlled_result": "supports" if direct_support else ("contradicts" if projection_wins else "inconclusive"),
        "challenge_assessment_ceiling": "partial C3 evidence; spinodal is reserved for a separate protocol",
    }
    atomic_json(output_dir / "comparison.json", result)
    return result


def spinodal_boundary() -> dict[str, Any]:
    source_files = sorted((OFFICIAL / "dataset" / "spinodal_decomposition").glob("*.cu"))
    return {
        "status": "not_attempted_separate_protocol",
        "reason": (
            "the release has only a CUDA generator and no data/checkpoints/converter; "
            "a separately preregistered CPU/MPS 128x128 mechanism study is feasible, "
            "but the literal 1024x1024 evaluation and same-A800 timing are not local targets"
        ),
        "nvcc_path": shutil.which("nvcc"),
        "torch_cuda_available": torch.cuda.is_available(),
        "released_cuda_sources": [str(path.relative_to(OFFICIAL)) for path in source_files],
        "official_commit": OFFICIAL_COMMIT,
        "claim_boundary": "this shallow-water harness makes no spinodal empirical claim",
    }


def run_metadata(config: RunConfig, provenance: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "models": list(MODEL_NAMES),
        "device": str(device),
        "python": sys.version,
        "torch": torch.__version__,
        "execution_environment": {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "v1_pdf_sha256": V1_PDF_SHA256,
        **provenance,
        "known_version_drift": {
            "v1_architecture": {"base_channels": 64, "num_blocks": 4, "kernel_size": 3, "epochs": 100},
            "released_single_seed_architecture": {"base_channels": 64, "num_blocks": 6, "kernel_size": 5, "epochs": 100},
            "selected_attempt": "arXiv-v1 architecture and 100-epoch budget; post-v1 6/5 sensitivity is not run",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "prepare-data", "train", "evaluate", "all"), default="plan")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--model", action="append", choices=MODEL_NAMES)
    parser.add_argument(
        "--max-new-epochs",
        type=int,
        help="stop after this many additional epochs while retaining the fixed target budget in the checkpoint fingerprint",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--block-pid", action="append", type=int)
    args = parser.parse_args()
    if args.max_new_epochs is not None and args.stage != "train":
        parser.error("--max-new-epochs is only valid with --stage train")
    if args.max_new_epochs is not None and args.max_new_epochs <= 0:
        parser.error("--max-new-epochs must be positive")
    if args.stage != "plan" and (args.seed, args.epochs, args.batch_size) != (42, 100, 16):
        parser.error("scientific stages require the preregistered --seed 42 --epochs 100 --batch-size 16")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = (args.data_manifest or (data_dir / "manifest.json")).resolve()
    config = RunConfig(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
    provenance = verify_official_pin()
    device = resolve_device(args.device)
    selected_models = tuple(args.model or MODEL_NAMES)

    if args.stage != "plan":
        active = blocked_pids(args.block_pid if args.block_pid is not None else [18410])
        if active:
            raise RuntimeError(f"refusing heavy stage while blocked PID(s) are active: {active}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "events.jsonl"
    metadata = run_metadata(config, provenance, device)
    atomic_json(output_dir / "run_config.json", metadata)
    atomic_json(output_dir / "spinodal_boundary.json", spinodal_boundary())
    emit(
        "run_plan",
        log_path,
        stage=args.stage,
        models=selected_models,
        data_manifest=str(manifest_path),
        output_dir=str(output_dir),
        metadata=metadata,
    )
    if args.stage == "plan":
        return

    if args.stage in ("prepare-data", "all"):
        manifest_path = prepare_dataset(data_dir, provenance, log_path)
        if args.stage == "prepare-data":
            return

    manifest = verify_dataset(manifest_path, provenance)
    emit("dataset_verified", log_path, dataset_sha256=manifest["dataset_sha256"], manifest=str(manifest_path))

    if args.stage in ("train", "all"):
        train, _, _ = load_split(manifest_path, manifest, "train")
        validation, _, _ = load_split(manifest_path, manifest, "val")
        emit(
            "training_data_loaded",
            log_path,
            train_shape=train.shape,
            validation_shape=validation.shape,
            train_windows=window_count(train, config.unroll_steps),
            validation_windows=window_count(validation, config.unroll_steps),
            peak_rss_mib=peak_rss_mib(),
        )
        for name in selected_models:
            train_model(
                name,
                train,
                validation,
                config,
                device,
                manifest,
                provenance,
                output_dir,
                not args.no_resume,
                log_path,
                args.max_new_epochs,
            )
        if args.stage == "train":
            return

    if args.stage in ("evaluate", "all"):
        for name in selected_models:
            evaluate_model(
                name,
                manifest_path,
                manifest,
                config,
                device,
                provenance,
                output_dir,
                args.evaluation_batch_size,
                log_path,
            )
        if all((output_dir / "metrics" / f"{name}.json").is_file() for name in MODEL_NAMES):
            emit("comparison_complete", log_path, **compare_models(output_dir, config))


if __name__ == "__main__":
    main()
