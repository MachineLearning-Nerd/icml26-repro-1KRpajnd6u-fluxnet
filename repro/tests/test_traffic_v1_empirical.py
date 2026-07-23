from __future__ import annotations

import numpy as np

from src.traffic_v1_empirical import make_windows, rollout_metrics


def test_make_windows_keeps_five_step_targets() -> None:
    rho = np.arange(2 * 7 * 4, dtype=np.float32).reshape(2, 7, 4)
    vmax = np.ones((2, 4), dtype=np.float32)
    inputs, targets = make_windows({"rho": rho, "vmax": vmax}, unroll_steps=5)
    assert inputs.shape == (4, 2, 4)
    assert targets.shape == (4, 5, 4)
    np.testing.assert_array_equal(inputs[0, 0], rho[0, 0])
    np.testing.assert_array_equal(targets[0], rho[0, 1:6])


def test_rollout_metrics_detect_stable_conservative_improvement() -> None:
    truth = np.full((2, 5, 4), 0.5, dtype=np.float32)
    prediction = truth.copy()
    summary, raw = rollout_metrics(prediction, truth)
    assert summary["final_mae_mean"] == 0.0
    assert summary["max_absolute_sum_drift"] == 0.0
    assert summary["divergent_trajectories"] == 0
    assert summary["lower_violation_rate_percent"] == 0.0
    assert summary["upper_violation_rate_percent"] == 0.0
    assert raw["per_step_mae"].shape == (2, 5)


def test_rollout_metrics_reports_bounds_and_mass_drift() -> None:
    truth = np.full((1, 3, 2), 0.5, dtype=np.float32)
    prediction = truth.copy()
    prediction[0, 1] = [-0.1, 1.2]
    summary, _ = rollout_metrics(prediction, truth)
    assert summary["lower_violation_rate_percent"] > 0.0
    assert summary["upper_violation_rate_percent"] > 0.0
    assert summary["max_absolute_sum_drift"] > 0.0
