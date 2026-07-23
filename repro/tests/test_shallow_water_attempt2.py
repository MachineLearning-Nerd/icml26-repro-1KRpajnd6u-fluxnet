from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch
from torch import nn

import src.shallow_water_attempt2 as shallow_attempt
from src.shallow_water_attempt2 import (
    MODEL_NAMES,
    RunConfig,
    _inspect_existing,
    atomic_torch_save,
    batch_losses,
    build_model,
    build_split_plan,
    compute_rollout_metrics,
    load_evaluation_checkpoint,
    load_training_checkpoint,
    make_batch,
    train_model,
    window_count,
)
from src.audit_shallow_water_attempt2 import recompute as independently_recompute


def test_split_plan_is_complete_and_stratified() -> None:
    categories = {
        "CaseA1": {"total": 24, "train": 10, "val": 4, "test": 10},
        "CaseA2": {"total": 24, "train": 10, "val": 4, "test": 10},
        "CaseB1": {"total": 36, "train": 15, "val": 6, "test": 15},
        "CaseB2": {"total": 36, "train": 15, "val": 6, "test": 15},
    }
    plan = build_split_plan(categories, 42)
    assert len(plan) == 120
    assert {split: sum(row["split"] == split for row in plan) for split in ("train", "val", "test")} == {
        "train": 50,
        "val": 20,
        "test": 50,
    }
    assert len({(row["category"], row["sample_index"]) for row in plan}) == 120
    assert plan == build_split_plan(categories, 42)


def test_window_batch_keeps_five_future_states() -> None:
    trajectories = np.arange(2 * 8 * 3 * 2 * 2, dtype=np.float32).reshape(2, 8, 3, 2, 2)
    assert window_count(trajectories, 5) == 6
    inputs, targets = make_batch(trajectories, np.asarray([0, 2, 3, 5]), 5)
    assert inputs.shape == (4, 3, 2, 2)
    assert targets.shape == (4, 5, 3, 2, 2)
    np.testing.assert_array_equal(targets[0], trajectories[0, 1:6])
    np.testing.assert_array_equal(targets[2], trajectories[1, 1:6])


def test_pushforward_loss_matches_released_one_plus_terminal_definition() -> None:
    class Scale(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.25))

        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
            return (value * self.scale,)

    model = Scale()
    inputs = torch.ones(2, 3, 2, 2)
    targets = torch.stack([inputs * float(step) for step in range(2, 7)], dim=1)
    config = RunConfig(unroll_steps=5)
    losses = batch_losses(model, inputs, targets, config)
    expected_prediction = torch.mean((inputs * model.scale - targets[:, 0]) ** 2)
    detached_step_four = inputs * model.scale.detach() ** 4
    expected_stability = torch.mean((detached_step_four * model.scale - targets[:, -1]) ** 2)
    torch.testing.assert_close(losses["prediction"], expected_prediction)
    torch.testing.assert_close(losses["stability"], expected_stability)
    torch.testing.assert_close(losses["total"], 0.5 * expected_prediction + 0.5 * expected_stability)


def test_metrics_cover_accuracy_divergence_bounds_and_conservation() -> None:
    config = RunConfig(train_horizon=0.04)
    truth = np.zeros((2, 4, 3, 2, 2), dtype=np.float32)
    truth[:, :, 0] = 1.0
    prediction = truth.copy()
    times = np.arange(4) * 0.04
    summary, raw = compute_rollout_metrics(prediction, truth, times, config)
    assert summary["final_mae"] == {"h": 0.0, "mx": 0.0, "my": 0.0}
    assert summary["divergent_trajectories"] == 0
    assert summary["h_violation_rate_percent"] == 0.0
    assert summary["max_relative_l1_normalized_integral_drift"]["h"] == 0.0
    assert raw["per_step_field_mae"].shape == (2, 4, 3)
    independent = independently_recompute(
        prediction, truth, times, require_complete=False, train_horizon=config.train_horizon
    )
    assert independent["final_mae"] == summary["final_mae"]
    assert independent["late_mae"] == summary["late_mae"]
    assert independent["max_relative_l1_normalized_integral_drift"] == summary[
        "max_relative_l1_normalized_integral_drift"
    ]


def test_atomic_checkpoint_round_trip(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(path, {"completed_epochs": 7, "tensor": torch.arange(3)})
    loaded = torch.load(path, weights_only=False)
    assert loaded["completed_epochs"] == 7
    torch.testing.assert_close(loaded["tensor"], torch.arange(3))


def test_training_checkpoint_rng_states_are_loaded_on_cpu(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(
        path,
        {
            "torch_rng_state": torch.get_rng_state(),
            "mps_rng_state": torch.arange(4, dtype=torch.uint8),
        },
    )
    loaded = load_training_checkpoint(path)
    assert loaded["torch_rng_state"].device.type == "cpu"
    assert loaded["torch_rng_state"].dtype == torch.uint8
    assert loaded["mps_rng_state"].device.type == "cpu"
    assert loaded["mps_rng_state"].dtype == torch.uint8


def test_existing_hdf5_requires_embedded_semantic_provenance(tmp_path) -> None:
    path = tmp_path / "trajectory.h5"
    record = {"split": "train", "category": "CaseA1", "sample_index": 3, "sample_seed": 17}
    provenance = {"official_commit": "commit", "source_sha256": {"generator": "generator"}}
    with h5py.File(path, "w") as handle:
        for name in ("h", "mx", "my"):
            handle.create_dataset(name, data=np.zeros((2, 64, 64), dtype=np.float32))
        handle.create_dataset("t", data=np.asarray([0.0, 2.4], dtype=np.float32))
        for name in ("mass", "momx", "momy"):
            handle.create_dataset(name, data=np.zeros(2, dtype=np.float32))
        metadata = handle.create_group("metadata")
        metadata.attrs.update(
            {
                **record,
                "T_final": 2.4,
                "dt": 0.004,
                "time_downsample": 10,
                "space_downsample": 2,
                "Nx_original": 128,
                "Ny_original": 128,
                "Nx_saved": 64,
                "Ny_saved": 64,
                "official_commit": "commit",
                "generator_sha256": "generator",
            }
        )
    inspected = _inspect_existing(path, record, 2.4, provenance)
    assert inspected["shape"] == [2, 64, 64]
    with h5py.File(path, "r+") as handle:
        handle["metadata"].attrs["Nx_saved"] = 32
    with pytest.raises(RuntimeError, match="Nx_saved mismatch"):
        _inspect_existing(path, record, 2.4, provenance)


def test_trajectory_identity_reuse_across_splits_is_rejected() -> None:
    identities: set[tuple[str, int, int]] = set()
    train = {"split": "train", "category": "CaseA1", "sample_index": 3, "sample_seed": 17}
    test = {"split": "test", "category": "CaseA1", "sample_index": 3, "sample_seed": 17}

    shallow_attempt.register_unique_trajectory_identity(train, identities)
    with pytest.raises(RuntimeError, match="across dataset splits"):
        shallow_attempt.register_unique_trajectory_identity(test, identities)


def test_evaluation_uses_latest_for_completion_and_best_for_weights(tmp_path) -> None:
    config = replace(RunConfig(), epochs=2)
    manifest = {"dataset_sha256": "dataset"}
    provenance = {"official_commit": "commit", "source_sha256": {"model": "source"}}
    name = MODEL_NAMES[0]
    fingerprint = shallow_attempt._checkpoint_fingerprint(config, manifest, provenance, name)
    model_dir = tmp_path / "models" / name
    atomic_torch_save(
        model_dir / "best_checkpoint.pt",
        {"fingerprint": fingerprint, "completed_epochs": 1, "model_state": {"best": True}},
    )
    atomic_torch_save(
        model_dir / "latest_checkpoint.pt",
        {"fingerprint": fingerprint, "completed_epochs": 2, "model_state": {"best": False}},
    )
    checkpoint, best_path, latest_path = load_evaluation_checkpoint(
        name, config, manifest, provenance, tmp_path, torch.device("cpu")
    )
    assert checkpoint["model_state"] == {"best": True}
    assert best_path.name == "best_checkpoint.pt"
    assert latest_path.name == "latest_checkpoint.pt"
    atomic_torch_save(
        model_dir / "latest_checkpoint.pt",
        {"fingerprint": fingerprint, "completed_epochs": 1, "model_state": {}},
    )
    with pytest.raises(RuntimeError, match="incomplete training"):
        load_evaluation_checkpoint(name, config, manifest, provenance, tmp_path, torch.device("cpu"))


def test_bounded_epoch_gate_resumes_to_fixed_target(tmp_path, monkeypatch) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Conv2d(3, 3, 1)

        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
            return (self.layer(value),)

    monkeypatch.setattr(shallow_attempt, "build_model", lambda _name, _config: TinyModel())
    config = replace(RunConfig(), epochs=2, batch_size=1, unroll_steps=2)
    trajectories = np.ones((1, 4, 3, 2, 2), dtype=np.float32)
    manifest = {"dataset_sha256": "dataset"}
    provenance = {"official_commit": "commit", "source_sha256": {"model": "source"}}
    log_path = tmp_path / "events.jsonl"
    common = (
        MODEL_NAMES[0],
        trajectories,
        trajectories,
        config,
        torch.device("cpu"),
        manifest,
        provenance,
        tmp_path,
        True,
        log_path,
    )
    checkpoint_path = train_model(*common, max_new_epochs=1)
    first = torch.load(checkpoint_path, weights_only=False)
    assert first["completed_epochs"] == 1
    assert first["config"]["epochs"] == 2
    assert first["mps_rng_state"] is None
    train_model(*common)
    completed = torch.load(checkpoint_path, weights_only=False)
    assert completed["completed_epochs"] == 2
    assert [row["epoch"] for row in completed["history"]] == [1, 2]


def test_exact_two_v1_headline_models_have_matching_output_shape() -> None:
    config = replace(
        RunConfig(), base_channels=4, num_blocks=1, kernel_size=3, fno_modes=2, fno_width=4, fno_layers=1
    )
    state = torch.zeros(1, 3, 8, 8)
    state[:, 0] = 1.0
    assert MODEL_NAMES == ("FluxNet_SW_LAP_pf", "FNO_SW_Proj_box_mass_pf")
    for name in MODEL_NAMES:
        model = build_model(name, config).eval()
        with torch.no_grad():
            output = model(state)[0]
        assert output.shape == state.shape
    flux_h = build_model(MODEL_NAMES[0], config).eval()(state)[0][:, 0]
    assert bool((flux_h >= 0.0).all())
    torch.testing.assert_close(flux_h.sum(), state[:, 0].sum(), rtol=1e-6, atol=1e-6)
    projected_h = build_model(MODEL_NAMES[1], config).eval()(state)[0][:, 0]
    assert bool((projected_h >= 0.0).all())
    torch.testing.assert_close(projected_h.sum(), state[:, 0].sum(), rtol=1e-5, atol=1e-5)
