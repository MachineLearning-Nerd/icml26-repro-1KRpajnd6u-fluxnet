from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

from src.audit_spinodal_attempt2b import independent_bootstrap
from src.spinodal_attempt2b import (
    FRAME_BYTES,
    RunConfig,
    batch_losses,
    bootstrap_ratio_interval,
    build_plan,
    canonical_sha256,
    compile_solver as compile_attempt2b_solver,
    make_batch,
    radial_autocorrelation,
    solver_command,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repro" / "src" / "spinodal_solver.cpp"
LIBOMP = Path("/opt/homebrew/opt/libomp")


def compile_solver(tmp_path: Path) -> Path:
    binary = tmp_path / "spinodal_solver"
    subprocess.run(
        [
            "clang++",
            "-std=c++17",
            "-O3",
            "-Xpreprocessor",
            "-fopenmp",
            "-ffp-contract=off",
            "-fno-fast-math",
            f"-I{LIBOMP / 'include'}",
            f"-L{LIBOMP / 'lib'}",
            "-lomp",
            str(SOURCE),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def numpy_step(concentration: np.ndarray) -> np.ndarray:
    temperature = 973.15
    gas_constant = 8.314
    rt = gas_constant * temperature
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
    chemical_potential = derivative - 2.0 * 3.57e-1 * laplacian
    laplacian_potential = (
        np.roll(chemical_potential, -1, axis=0)
        + np.roll(chemical_potential, 1, axis=0)
        + np.roll(chemical_potential, -1, axis=1)
        + np.roll(chemical_potential, 1, axis=1)
        - 4.0 * chemical_potential
    )
    return np.clip(concentration + 1.0e-2 * laplacian_potential, 0.0, 1.0)


def serialized(field: np.ndarray) -> np.ndarray:
    return (np.round(field * 1.0e6) / 1.0e6).astype(np.float32)


def test_cpp_solver_matches_independent_numpy_oracle(tmp_path) -> None:
    binary = compile_solver(tmp_path)
    rows, columns = np.indices((128, 128), dtype=np.float64)
    initial = 0.60 + 0.01 * np.sin(rows / 9.0) * np.cos(columns / 13.0)
    input_path = tmp_path / "initial.float64"
    initial.tofile(input_path)
    completed = subprocess.run(
        [
            str(binary),
            "--input",
            str(input_path),
            "--steps",
            "3",
            "--save-start",
            "0",
            "--save-interval",
            "1",
            "--threads",
            "2",
        ],
        check=True,
        capture_output=True,
    )
    observed = np.frombuffer(completed.stdout, dtype=np.float32).reshape(4, 128, 128)
    expected = [serialized(initial)]
    state = initial.copy()
    for _ in range(3):
        state = numpy_step(state)
        expected.append(serialized(state))
    np.testing.assert_allclose(observed, np.stack(expected), rtol=0.0, atol=1.0e-6)
    assert b"complete steps=3 frames=4 threads=2" in completed.stderr


def test_cpp_solver_rejects_misaligned_save_schedule(tmp_path) -> None:
    binary = compile_solver(tmp_path)
    completed = subprocess.run(
        [
            str(binary),
            "--steps",
            "10",
            "--save-start",
            "3",
            "--save-interval",
            "4",
        ],
        capture_output=True,
    )
    assert completed.returncode == 1
    assert b"must be divisible" in completed.stderr


def test_attempt2b_fixed_plan_matches_preregistration() -> None:
    plan = build_plan()
    assert len(plan) == 22
    assert [(item.split, item.seed) for item in plan[:2]] == [
        ("train", 12345),
        ("val", 67890),
    ]
    assert [item.seed for item in plan[2:]] == [22345 + 12345 * index for index in range(20)]
    assert [item.frame_count for item in plan[:2]] == [5001, 5001]
    assert all(item.frame_count == 1001 for item in plan[2:])
    assert sum(item.frame_count for item in plan) == 30022
    assert sum(item.frame_count * FRAME_BYTES for item in plan) == 1_967_521_792
    assert len({item.relative_path for item in plan}) == len(plan)


def test_solver_command_and_fingerprint_are_deterministic(tmp_path) -> None:
    trajectory = build_plan()[0]
    binary = tmp_path / "solver"
    command = solver_command(binary, trajectory, threads=3)
    assert command == [
        str(binary),
        "--seed",
        "12345",
        "--steps",
        "52000",
        "--save-start",
        "2000",
        "--save-interval",
        "10",
        "--threads",
        "3",
    ]
    assert canonical_sha256(command) == canonical_sha256(list(command))


def test_solver_build_is_reused_with_verified_metadata(tmp_path) -> None:
    binary = tmp_path / "bin" / "spinodal_solver"
    first = compile_attempt2b_solver(binary)
    first_mtime = binary.stat().st_mtime_ns
    second = compile_attempt2b_solver(binary)
    assert second == first
    assert binary.stat().st_mtime_ns == first_mtime


def test_100dt_windows_and_released_loss_formula() -> None:
    config = RunConfig()
    trajectory = np.broadcast_to(
        np.arange(30, dtype=np.float32)[:, None, None],
        (30, 2, 2),
    ).copy()
    inputs, targets = make_batch(trajectory, np.array([0, 3]), config)
    np.testing.assert_array_equal(inputs[:, 0, 0, 0], np.array([0.0, 3.0]))
    np.testing.assert_array_equal(targets[:, :, 0, 0], np.array([[10.0, 20.0], [13.0, 23.0]]))

    class FixedModel(torch.nn.Module):
        def forward(self, values):
            return values + 0.5, torch.zeros_like(values), torch.full_like(values, 2.0)

    losses = batch_losses(
        FixedModel(),
        torch.zeros((2, 1, 4, 4)),
        torch.stack(
            [torch.ones((2, 1, 4, 4)), torch.full((2, 1, 4, 4), 2.0)],
            dim=1,
        ).squeeze(2),
        config,
    )
    assert float(losses["prediction"]) == 0.25
    assert float(losses["stability"]) == 1.0
    assert float(losses["dcl"]) == 4.0
    assert float(losses["dcl_n"]) == 4.0
    assert float(losses["total"]) == 4.3125


def test_radial_statistic_and_whole_sample_bootstrap_are_deterministic() -> None:
    radial = radial_autocorrelation(np.full((128, 128), 0.6, dtype=np.float32))
    np.testing.assert_allclose(radial, np.full(64, 0.36), rtol=0.0, atol=3.0e-8)
    first = bootstrap_ratio_interval(
        np.full(20, 2.0),
        np.full(10, 1.0),
        seed=42,
        replicates=100,
    )
    second = bootstrap_ratio_interval(
        np.full(20, 2.0),
        np.full(10, 1.0),
        seed=42,
        replicates=100,
    )
    assert first == second == {
        "point": 2.0,
        "lower_95": 2.0,
        "upper_95": 2.0,
        "replicates": 100,
        "seed": 42,
    }

    matched = np.linspace(0.1, 0.4, 20)
    intrinsic = np.linspace(0.2, 0.3, 10)
    primary = bootstrap_ratio_interval(matched, intrinsic, seed=42, replicates=250)
    independent = independent_bootstrap(matched, intrinsic, seed=42, replicates=250)
    assert independent == primary
