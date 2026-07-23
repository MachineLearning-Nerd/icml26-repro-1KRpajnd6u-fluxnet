"""Independent raw-artifact audit for spinodal Attempt 2b.

This program deliberately does not import the experiment harness or its metric
functions.  It reopens every source and prediction HDF5 file, recomputes the
preregistered metrics and bootstrap verdict, and emits a standalone audit
certificate only after all provenance and numeric comparisons pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "official"
OFFICIAL_COMMIT = "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c"
DEFAULT_DATA_DIR = ROOT / "data" / "spinodal_attempt2b"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "spinodal_attempt2b"
GRID = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def assert_close(label: str, actual: float, expected: float, tolerance: float = 1.0e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")


def independent_radial(field: np.ndarray, radius_map: np.ndarray) -> np.ndarray:
    transformed = np.fft.fft2(np.asarray(field, dtype=np.float64))
    correlation = np.fft.fftshift(np.fft.ifft2(np.abs(transformed) ** 2).real) / (GRID * GRID)
    values = []
    for radius in range(GRID // 2):
        annulus = (radius_map >= radius - 0.5) & (radius_map < radius + 0.5)
        values.append(float(np.sum(correlation[annulus], dtype=np.float64) / np.count_nonzero(annulus)))
    return np.asarray(values, dtype=np.float64)


def independent_bootstrap(
    matched: np.ndarray,
    intrinsic: np.ndarray,
    seed: int = 42,
    replicates: int = 10000,
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        numerator_indices = generator.integers(0, len(matched), size=len(matched))
        denominator_indices = generator.integers(0, len(intrinsic), size=len(intrinsic))
        numerator = float(np.mean(matched[numerator_indices]))
        denominator = float(np.mean(intrinsic[denominator_indices]))
        samples.append(numerator / denominator if denominator > 0.0 else math.inf)
    denominator = float(np.mean(intrinsic))
    return {
        "point": float(np.mean(matched) / denominator) if denominator > 0.0 else math.inf,
        "lower_95": float(np.percentile(samples, 2.5)),
        "upper_95": float(np.percentile(samples, 97.5)),
        "replicates": replicates,
        "seed": seed,
    }


def audit_trajectory(
    source_path: Path,
    prediction_path: Path,
    recorded: dict[str, Any],
    radius_map: np.ndarray,
    dataset_sha256: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], np.ndarray]:
    with h5py.File(source_path, "r") as source, h5py.File(prediction_path, "r") as artifact:
        truth = source["phi_data"][:]
        prediction = artifact["prediction"][:]
        source_steps = source["base_steps"][:]
        artifact_steps = artifact["base_steps"][:]
        metadata = dict(artifact["metadata"].attrs.items())
        recorded_truth_radial = artifact["metrics/truth_radial"][:]
        recorded_prediction_radial = artifact["metrics/prediction_radial"][:]
    if truth.shape != (1001, GRID, GRID) or prediction.shape != truth.shape:
        raise RuntimeError(f"invalid rollout shape for seed {recorded['seed']}")
    expected_steps = np.arange(2000, 102001, 100, dtype=np.int64)
    if not np.array_equal(source_steps, expected_steps) or not np.array_equal(artifact_steps, expected_steps):
        raise RuntimeError(f"invalid time grid for seed {recorded['seed']}")
    if int(metadata["seed"]) != int(recorded["seed"]):
        raise RuntimeError(f"artifact seed mismatch for {prediction_path}")
    if str(metadata["source_sha256"]) != sha256_file(source_path):
        raise RuntimeError(f"artifact source hash mismatch for {prediction_path}")
    if str(metadata["dataset_sha256"]) != dataset_sha256:
        raise RuntimeError(f"artifact dataset hash mismatch for {prediction_path}")
    if str(metadata["checkpoint_sha256"]) != checkpoint_sha256:
        raise RuntimeError(f"artifact checkpoint hash mismatch for {prediction_path}")
    if str(metadata["official_commit"]) != OFFICIAL_COMMIT:
        raise RuntimeError(f"artifact official commit mismatch for {prediction_path}")
    if not np.array_equal(prediction[0], truth[0]):
        raise RuntimeError(f"rollout does not begin at the true warmup state for seed {recorded['seed']}")
    finite = bool(np.isfinite(prediction).all())
    difference = np.abs(prediction.astype(np.float64) - truth.astype(np.float64))
    final_mae = float(difference[-1].mean())
    initial_mass = float(prediction[0].sum(dtype=np.float64))
    mass = prediction.sum(axis=(1, 2), dtype=np.float64)
    drift = np.abs(mass - initial_mass) / max(abs(initial_mass), 1.0e-12)
    lower = prediction < 0.0
    upper = prediction > 1.0
    lower_counts = lower.sum(axis=(1, 2))
    upper_counts = upper.sum(axis=(1, 2))
    lower_magnitudes = np.divide(
        np.maximum(-prediction, 0.0).sum(axis=(1, 2), dtype=np.float64),
        lower_counts,
        out=np.zeros(1001, dtype=np.float64),
        where=lower_counts > 0,
    )
    upper_magnitudes = np.divide(
        np.maximum(prediction - 1.0, 0.0).sum(axis=(1, 2), dtype=np.float64),
        upper_counts,
        out=np.zeros(1001, dtype=np.float64),
        where=upper_counts > 0,
    )
    truth_radial = np.empty((1001, GRID // 2), dtype=np.float64)
    prediction_radial = np.empty_like(truth_radial)
    for frame in range(1001):
        truth_radial[frame] = independent_radial(truth[frame], radius_map)
        prediction_radial[frame] = independent_radial(prediction[frame], radius_map)
    np.testing.assert_allclose(recorded_truth_radial, truth_radial, rtol=2.0e-6, atol=2.0e-7)
    np.testing.assert_allclose(recorded_prediction_radial, prediction_radial, rtol=2.0e-6, atol=2.0e-7)
    radial_error = np.mean(np.abs(prediction_radial - truth_radial), axis=1)
    normalized_time = (source_steps[500:] - 52000) / 50000.0
    radial_auc = float(np.trapezoid(radial_error[500:], x=normalized_time))
    recomputed = {
        "finite": finite,
        "final_mae": final_mae,
        "maximum_relative_mass_drift": float(np.max(drift)),
        "minimum_prediction": float(np.nanmin(prediction)),
        "maximum_prediction": float(np.nanmax(prediction)),
        "lower_violation_rate": float(np.mean(lower)),
        "upper_violation_rate": float(np.mean(upper)),
        "maximum_conditional_lower_violation_magnitude": float(np.max(lower_magnitudes)),
        "maximum_conditional_upper_violation_magnitude": float(np.max(upper_magnitudes)),
        "radial_error_auc_T1_T2": radial_auc,
    }
    if recomputed["finite"] != recorded["finite"]:
        raise RuntimeError(f"finite verdict mismatch for seed {recorded['seed']}")
    for name, actual in recomputed.items():
        if name == "finite":
            continue
        assert_close(f"seed {recorded['seed']} {name}", actual, float(recorded[name]), tolerance=2.0e-7)
    return recomputed, truth_radial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--certificate", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    commit = subprocess.run(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official checkout is {commit}, expected {OFFICIAL_COMMIT}")
    if subprocess.run(
        ["git", "-C", str(OFFICIAL), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise RuntimeError("official checkout is not clean")
    manifest_path = args.data_dir / "manifest.json"
    evaluation_path = args.output_dir / "evaluation.json"
    manifest = json.loads(manifest_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    if manifest["dataset_sha256"] != evaluation["dataset_sha256"]:
        raise RuntimeError("evaluation dataset fingerprint mismatch")
    checkpoint_path = args.output_dir / evaluation["checkpoint"]
    if sha256_file(checkpoint_path) != evaluation["checkpoint_sha256"]:
        raise RuntimeError("evaluation checkpoint hash mismatch")
    manifest_entries = {entry["path"]: entry for entry in manifest["files"]}
    rows, columns = np.indices((GRID, GRID), dtype=np.float64)
    radius_map = np.sqrt((rows - GRID // 2) ** 2 + (columns - GRID // 2) ** 2)
    audited_entries = []
    truth_radial: dict[int, np.ndarray] = {}
    for recorded in evaluation["entries"]:
        source_relative = recorded["source_path"]
        source_path = args.data_dir / source_relative
        prediction_path = args.output_dir / recorded["prediction_path"]
        if source_relative not in manifest_entries:
            raise RuntimeError(f"unmanifested source: {source_relative}")
        if sha256_file(source_path) != manifest_entries[source_relative]["sha256"]:
            raise RuntimeError(f"source hash mismatch: {source_relative}")
        if sha256_file(prediction_path) != recorded["prediction_sha256"]:
            raise RuntimeError(f"prediction hash mismatch: {prediction_path}")
        audited, radial = audit_trajectory(
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
        raise RuntimeError("audit requires 20 unique completed trajectories")
    matched = np.asarray([entry["radial_error_auc_T1_T2"] for entry in audited_entries])
    intrinsic = []
    seeds = [entry["seed"] for entry in audited_entries]
    normalized_time = np.arange(501, dtype=np.float64) / 500.0
    for first, second in zip(seeds[::2], seeds[1::2]):
        error = np.mean(np.abs(truth_radial[first][500:] - truth_radial[second][500:]), axis=1)
        intrinsic.append(float(np.trapezoid(error, x=normalized_time)))
    intrinsic_array = np.asarray(intrinsic)
    ratio = independent_bootstrap(matched, intrinsic_array)
    for name in ("point", "lower_95", "upper_95"):
        assert_close(name, ratio[name], float(evaluation["radial_auc_ratio"][name]), tolerance=2.0e-7)
    mean_final_mae = float(np.mean([entry["final_mae"] for entry in audited_entries]))
    maximum_drift = float(np.max([entry["maximum_relative_mass_drift"] for entry in audited_entries]))
    finite_all = all(entry["finite"] for entry in audited_entries)
    supports = bool(finite_all and maximum_drift <= 1.0e-5 and mean_final_mae <= 4.32e-2 and ratio["upper_95"] <= 1.25)
    falsified = bool(ratio["lower_95"] > 1.25)
    verdict = "supports" if supports else "falsified" if falsified else "inconclusive"
    if verdict != evaluation["verdict"] or supports != evaluation["supports"] or falsified != evaluation["falsified"]:
        raise RuntimeError("independent verdict mismatch")
    certificate = {
        "schema_version": 1,
        "audit": "independent raw HDF5 recomputation without experiment metric imports",
        "audit_source_sha256": sha256_file(Path(__file__).resolve()),
        "official_commit": commit,
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
    certificate_path = args.certificate or args.output_dir / "audit_certificate.json"
    atomic_json(certificate_path, certificate)
    print(json.dumps({"event": "spinodal_audit_complete", "certificate": str(certificate_path), **certificate}, sort_keys=True))


if __name__ == "__main__":
    main()
