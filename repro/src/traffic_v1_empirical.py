"""Bounded arXiv-v1 traffic-flow reproduction using unmodified author cores.

The author checkout is an ignored, pinned dependency. This module imports the
released generator and model files without modifying or copying their source.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


# Importing the ignored author checkout must not create files under official/.
sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "official"
DEFAULT_OUTPUT = ROOT / "outputs" / "traffic_v1_attempt1"
OFFICIAL_COMMIT = "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c"
V1_PDF_SHA256 = "40b79be23b629ec42d41eae9bdd578120e9a401d9e2dbef54011b5ba6ec2a285"


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 96
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    unroll_steps: int = 5
    dcl_weight: float = 0.1
    base_channels: int = 32
    num_blocks: int = 6
    kernel_size: int = 5
    neighborhood_size: int = 11
    scheduler_patience: int = 15
    scheduler_factor: float = 0.5


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_official_pin() -> str:
    completed = subprocess.run(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official checkout is {commit}, expected {OFFICIAL_COMMIT}")
    for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        completed = subprocess.run(["git", "-C", str(OFFICIAL), *args], check=False)
        if completed.returncode != 0:
            raise RuntimeError("official checkout has tracked changes; refusing an unpinned run")
    return commit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def generate_dataset(generator: Any) -> dict[str, dict[str, np.ndarray]]:
    """Reproduce the released split while extending test trajectories to T=8."""
    master_rng = np.random.default_rng(generator.GLOBAL_SEED)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for category, counts in generator.CATEGORY_CONFIG.items():
        category_samples: list[dict[str, Any]] = []
        make_case = generator.GENERATORS[category]
        for sample_index in range(counts["total"]):
            rho0, vmax, description, params = make_case(master_rng, generator.x_centers)
            category_samples.append(
                {
                    "rho0": rho0,
                    "vmax": vmax,
                    "description": description,
                    "params": params,
                    "category": category,
                    "sample_index": sample_index,
                }
            )

        indices = list(range(counts["total"]))
        master_rng.shuffle(indices)
        train_stop = counts["train"]
        val_stop = train_stop + counts["val"]
        for split, selected in (
            ("train", indices[:train_stop]),
            ("val", indices[train_stop:val_stop]),
            ("test", indices[val_stop:]),
        ):
            for index in selected:
                splits[split].append(category_samples[index])

    packed: dict[str, dict[str, np.ndarray]] = {}
    for split, samples in splits.items():
        horizon = 8.0 if split == "test" else 4.0
        histories = []
        velocities = []
        times_reference = None
        sample_ids = []
        categories = []
        mass_drifts = []
        for sample in samples:
            times, rho, mass = generator.simulate_lwr_fixed_dt(
                sample["rho0"],
                sample["vmax"],
                horizon,
                generator.dx,
                generator.DT_FIXED,
                time_downsample=generator.TIME_DOWNSAMPLE,
            )
            if times_reference is None:
                times_reference = times
            elif not np.array_equal(times_reference, times):
                raise AssertionError("nonuniform saved time grid")
            histories.append(rho.astype(np.float32))
            velocities.append(sample["vmax"].astype(np.float32))
            sample_ids.append(f"{sample['category']}:{sample['sample_index']:02d}")
            categories.append(sample["category"])
            mass_drifts.append(float(np.max(np.abs(mass - mass[0]))))

        packed[split] = {
            "rho": np.stack(histories),
            "vmax": np.stack(velocities),
            "times": np.asarray(times_reference, dtype=np.float64),
            "sample_id": np.asarray(sample_ids),
            "category": np.asarray(categories),
            "solver_max_absolute_mass_drift": np.asarray(mass_drifts),
        }

    expected = {"train": (100, 26, 256), "val": (50, 26, 256), "test": (100, 51, 256)}
    for split, shape in expected.items():
        if packed[split]["rho"].shape != shape:
            raise AssertionError(f"{split} shape {packed[split]['rho'].shape}, expected {shape}")
    return packed


def dataset_manifest(dataset: dict[str, dict[str, np.ndarray]], config: RunConfig) -> dict[str, Any]:
    split_details = {}
    for split, data in dataset.items():
        unique, counts = np.unique(data["category"], return_counts=True)
        split_details[split] = {
            "rho_shape": list(data["rho"].shape),
            "vmax_shape": list(data["vmax"].shape),
            "saved_times": data["times"].tolist(),
            "rho_min": float(data["rho"].min()),
            "rho_max": float(data["rho"].max()),
            "max_solver_absolute_mass_drift": float(data["solver_max_absolute_mass_drift"].max()),
            "category_counts": {str(k): int(v) for k, v in zip(unique, counts, strict=True)},
            "sample_ids": data["sample_id"].tolist(),
        }
    return {
        "paper_target": "arXiv:2602.01941v1 Table 4",
        "v1_pdf_sha256": V1_PDF_SHA256,
        "official_commit": OFFICIAL_COMMIT,
        "solver": {"domain": [0.0, 10.0], "nx": 256, "dt": 0.016, "saved_stride": 10},
        "config": asdict(config),
        "splits": split_details,
        "known_drift": {
            "released_generator_default_horizon": 4.0,
            "wrapper_test_horizon": 8.0,
            "released_script_batch_size": 16,
            "wrapper_batch_size": config.batch_size,
            "reason": "bounded local wall-clock ceiling with unrelated CPU jobs active",
        },
    }


def make_windows(data: dict[str, np.ndarray], unroll_steps: int) -> tuple[np.ndarray, np.ndarray]:
    rho = data["rho"]
    vmax = data["vmax"]
    inputs = []
    targets = []
    for trajectory, external in zip(rho, vmax, strict=True):
        for start in range(trajectory.shape[0] - unroll_steps):
            inputs.append(np.stack([trajectory[start], external], axis=0))
            targets.append(trajectory[start + 1 : start + 1 + unroll_steps])
    return np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32)


def predict(model: nn.Module, model_input: torch.Tensor, fluxnet: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(model_input)
    if fluxnet:
        state, outflow, inflow = output
        return state, torch.mean((outflow - inflow) ** 2)
    return output[0], None


def batch_losses(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: RunConfig,
    fluxnet: bool,
) -> dict[str, torch.Tensor]:
    first, first_dcl = predict(model, inputs, fluxnet)
    losses = {"prediction": torch.mean((first - targets[:, 0:1]) ** 2)}

    current = inputs[:, 0:1]
    external = inputs[:, 1:]
    with torch.no_grad():
        for _ in range(config.unroll_steps - 1):
            current, _ = predict(model, torch.cat([current, external], dim=1), fluxnet)
    terminal, terminal_dcl = predict(model, torch.cat([current, external], dim=1), fluxnet)
    losses["stability"] = torch.mean((terminal - targets[:, -1:]) ** 2)
    if fluxnet:
        assert first_dcl is not None and terminal_dcl is not None
        losses["dcl"] = config.dcl_weight * (first_dcl + terminal_dcl)
    losses["total"] = sum(losses.values())
    return losses


def epoch_metrics(
    model: nn.Module,
    arrays: tuple[np.ndarray, np.ndarray],
    config: RunConfig,
    device: torch.device,
    fluxnet: bool,
) -> dict[str, float]:
    inputs, targets = arrays
    totals: dict[str, float] = {}
    examples = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), config.batch_size):
            stop = min(start + config.batch_size, len(inputs))
            x = torch.from_numpy(inputs[start:stop]).to(device)
            y = torch.from_numpy(targets[start:stop]).to(device)
            losses = batch_losses(model, x, y, config, fluxnet)
            count = stop - start
            examples += count
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * count
    return {name: value / examples for name, value in totals.items()}


def train_model(
    name: str,
    model: nn.Module,
    train_arrays: tuple[np.ndarray, np.ndarray],
    val_arrays: tuple[np.ndarray, np.ndarray],
    config: RunConfig,
    device: torch.device,
    fluxnet: bool,
    output_dir: Path,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    set_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    inputs, targets = train_arrays
    shuffle_rng = np.random.default_rng(config.seed)
    best_loss = math.inf
    best_state = None
    history = []
    started = time.monotonic()

    for epoch in range(config.epochs):
        permutation = shuffle_rng.permutation(len(inputs))
        sums: dict[str, float] = {}
        examples = 0
        model.train()
        for start in range(0, len(permutation), config.batch_size):
            selected = permutation[start : start + config.batch_size]
            x = torch.from_numpy(inputs[selected]).to(device)
            y = torch.from_numpy(targets[selected]).to(device)
            losses = batch_losses(model, x, y, config, fluxnet)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            count = len(selected)
            examples += count
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * count

        train_values = {key: value / examples for key, value in sums.items()}
        val_values = epoch_metrics(model, val_arrays, config, device, fluxnet)
        scheduler.step(val_values["total"])
        elapsed = time.monotonic() - started
        record = {
            "epoch": epoch + 1,
            "elapsed_seconds": elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_values,
            "validation": val_values,
        }
        history.append(record)
        if val_values["total"] < best_loss:
            best_loss = val_values["total"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            print(
                f"{name} epoch={epoch + 1:03d}/{config.epochs} "
                f"train={train_values['total']:.6g} val={val_values['total']:.6g} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if best_state is None:
        raise AssertionError("no best model state captured")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "config": asdict(config),
            "official_commit": OFFICIAL_COMMIT,
            "best_validation_loss": best_loss,
        },
        output_dir / f"{name}_best.pt",
    )
    return model, history


def rollout(model: nn.Module, test: dict[str, np.ndarray], device: torch.device, fluxnet: bool) -> np.ndarray:
    truth = test["rho"]
    vmax = test["vmax"]
    result = np.empty_like(truth)
    result[:, 0] = truth[:, 0]
    model.eval()
    evaluation_batch = 20
    with torch.no_grad():
        for start in range(0, len(truth), evaluation_batch):
            stop = min(start + evaluation_batch, len(truth))
            current = torch.from_numpy(truth[start:stop, 0:1]).to(device)
            external = torch.from_numpy(vmax[start:stop, None]).to(device)
            for step in range(1, truth.shape[1]):
                current, _ = predict(model, torch.cat([current, external], dim=1), fluxnet)
                result[start:stop, step] = current[:, 0].detach().cpu().numpy()
    return result


def rollout_metrics(prediction: np.ndarray, truth: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    absolute = np.abs(prediction - truth)
    per_step_mae = absolute.mean(axis=2)
    initial_sum = prediction[:, 0].sum(axis=1, keepdims=True)
    predicted_sum = prediction.sum(axis=2)
    per_step_mass_drift_absolute = np.abs(predicted_sum - initial_sum)
    per_step_mass_drift_relative = per_step_mass_drift_absolute / (np.abs(initial_sum) + 1e-12)
    per_step_lower_rate = (prediction < 0.0).mean(axis=2)
    per_step_upper_rate = (prediction > 1.0).mean(axis=2)
    lower_magnitudes = np.maximum(-prediction, 0.0)
    upper_magnitudes = np.maximum(prediction - 1.0, 0.0)
    lower_count = int((prediction < 0.0).sum())
    upper_count = int((prediction > 1.0).sum())
    late_slice = slice((truth.shape[1] - 1) // 2 + 1, None)
    steps = np.arange(truth.shape[1], dtype=np.float64)
    slopes = []
    for values in per_step_mae:
        late_steps = steps[late_slice]
        late_values = values[late_slice]
        if len(late_steps) >= 2:
            slopes.append(float(np.polyfit(late_steps, late_values, 1)[0]))
        else:
            slopes.append(0.0)
    divergent = (~np.isfinite(prediction).all(axis=(1, 2))) | (np.abs(prediction).max(axis=(1, 2)) > 10.0)
    summary = {
        "final_mae_mean": float(per_step_mae[:, -1].mean()),
        "final_mae_std": float(per_step_mae[:, -1].std(ddof=1 if len(per_step_mae) > 1 else 0)),
        "late_horizon_mae_mean": float(per_step_mae[:, late_slice].mean()),
        "late_horizon_error_slope_mean": float(np.mean(slopes)),
        "divergent_trajectories": int(divergent.sum()),
        "max_absolute_sum_drift": float(per_step_mass_drift_absolute.max()),
        "max_relative_sum_drift": float(per_step_mass_drift_relative.max()),
        "lower_violation_rate_percent": float((prediction < 0.0).mean() * 100.0),
        "upper_violation_rate_percent": float((prediction > 1.0).mean() * 100.0),
        "conditional_lower_violation_magnitude": (
            float(lower_magnitudes.sum() / lower_count) if lower_count else 0.0
        ),
        "conditional_upper_violation_magnitude": (
            float(upper_magnitudes.sum() / upper_count) if upper_count else 0.0
        ),
        "prediction_min": float(np.nanmin(prediction)),
        "prediction_max": float(np.nanmax(prediction)),
    }
    raw = {
        "per_step_mae": per_step_mae.astype(np.float32),
        "per_step_mass_drift_absolute": per_step_mass_drift_absolute.astype(np.float32),
        "per_step_mass_drift_relative": per_step_mass_drift_relative.astype(np.float32),
        "per_step_lower_violation_rate": per_step_lower_rate.astype(np.float32),
        "per_step_upper_violation_rate": per_step_upper_rate.astype(np.float32),
        "late_horizon_error_slope": np.asarray(slopes, dtype=np.float32),
        "divergent": divergent,
    }
    return summary, raw


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = RunConfig(epochs=args.epochs, batch_size=args.batch_size)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    official_commit = verify_official_pin()
    device = resolve_device(args.device)
    print(f"official={official_commit} device={device} config={asdict(config)}", flush=True)

    generator = load_module(OFFICIAL / "dataset" / "traffic_flow" / "dataset.py", "author_traffic_dataset")
    fluxnet_module = load_module(OFFICIAL / "src" / "models" / "fluxnet_d_1d.py", "author_fluxnet_d")
    baseline_module = load_module(OFFICIAL / "src" / "models" / "cnn_baseline.py", "author_cnn")

    dataset = generate_dataset(generator)
    write_json(output_dir / "dataset_manifest.json", dataset_manifest(dataset, config))
    train_arrays = make_windows(dataset["train"], config.unroll_steps)
    val_arrays = make_windows(dataset["val"], config.unroll_steps)
    print(f"train_windows={train_arrays[0].shape} val_windows={val_arrays[0].shape}", flush=True)

    set_seed(config.seed)
    fluxnet = fluxnet_module.FluxNet_D_1D(
        in_channels=2,
        base_channels=config.base_channels,
        num_blocks=config.num_blocks,
        kernel_size=config.kernel_size,
        neighborhood_size=config.neighborhood_size,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    fluxnet, fluxnet_history = train_model(
        "fluxnet_d", fluxnet, train_arrays, val_arrays, config, device, True, output_dir
    )

    set_seed(config.seed)
    baseline = baseline_module.CNN_Baseline_1D(
        in_channels=2,
        out_channels=1,
        base_channels=config.base_channels,
        num_blocks=config.num_blocks,
        kernel_size=config.kernel_size,
        prediction_mode="residual",
        bound_mode="none",
    )
    baseline, baseline_history = train_model(
        "resnet_ar", baseline, train_arrays, val_arrays, config, device, False, output_dir
    )
    write_json(
        output_dir / "training_history.json",
        {"config": asdict(config), "fluxnet_d": fluxnet_history, "resnet_ar": baseline_history},
    )

    fluxnet_prediction = rollout(fluxnet, dataset["test"], device, True)
    baseline_prediction = rollout(baseline, dataset["test"], device, False)
    truth = dataset["test"]["rho"]
    fluxnet_summary, fluxnet_raw = rollout_metrics(fluxnet_prediction, truth)
    baseline_summary, baseline_raw = rollout_metrics(baseline_prediction, truth)

    ratio = fluxnet_summary["final_mae_mean"] / baseline_summary["final_mae_mean"]
    direct_improvement = (
        fluxnet_summary["final_mae_mean"] < baseline_summary["final_mae_mean"]
        and fluxnet_summary["late_horizon_mae_mean"] < baseline_summary["late_horizon_mae_mean"]
        and fluxnet_summary["divergent_trajectories"] <= baseline_summary["divergent_trajectories"]
        and fluxnet_summary["max_relative_sum_drift"] <= 1e-5
        and fluxnet_summary["conditional_lower_violation_magnitude"] <= 1e-2
        and fluxnet_summary["conditional_upper_violation_magnitude"] <= 1e-2
    )
    if direct_improvement:
        controlled_result = "supports"
    elif (
        baseline_summary["final_mae_mean"] <= fluxnet_summary["final_mae_mean"]
        and baseline_summary["late_horizon_mae_mean"] <= fluxnet_summary["late_horizon_mae_mean"]
    ):
        controlled_result = "contradicts"
    else:
        controlled_result = "inconclusive"

    summary = {
        "paper_target": {
            "version": "arXiv:2602.01941v1",
            "table": 4,
            "fluxnet_d_final_mae": 3.48e-3,
            "resnet_ar_final_mae": 15.9e-3,
            "reported_ratio": 3.48 / 15.9,
        },
        "official_commit": official_commit,
        "device": str(device),
        "config": asdict(config),
        "fluxnet_d": fluxnet_summary,
        "resnet_ar": baseline_summary,
        "observed_fluxnet_to_resnet_final_mae_ratio": ratio,
        "controlled_result": controlled_result,
        "decision_thresholds": {
            "max_relative_sum_drift": 1e-5,
            "max_conditional_bound_violation_magnitude": 1e-2,
            "divergence_absolute_prediction": 10.0,
        },
        "challenge_assessment": "toy",
        "assessment_reason": (
            "full released dataset breadth, architecture, v1 epoch count, and rollout horizon; "
            "single seed and batch-96/source-version drift prevent paper-table verification"
        ),
    }
    write_json(output_dir / "summary.json", summary)

    arrays: dict[str, np.ndarray] = {
        "sample_id": dataset["test"]["sample_id"],
        "category": dataset["test"]["category"],
        "times": dataset["test"]["times"],
        "vmax": dataset["test"]["vmax"],
        "truth": truth,
        "fluxnet_d_prediction": fluxnet_prediction,
        "resnet_ar_prediction": baseline_prediction,
    }
    arrays.update({f"fluxnet_d_{key}": value for key, value in fluxnet_raw.items()})
    arrays.update({f"resnet_ar_{key}": value for key, value in baseline_raw.items()})
    np.savez_compressed(output_dir / "traffic_rollouts_raw.npz", **arrays)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
