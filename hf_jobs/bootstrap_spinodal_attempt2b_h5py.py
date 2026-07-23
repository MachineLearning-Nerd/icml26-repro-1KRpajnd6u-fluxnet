"""Install the pinned h5py wheel into an isolated ephemeral target, then exec.

The base image and scientific sources remain unchanged. This bootstrap accepts
only the preflight-observed CPython/Linux/NumPy/Torch runtime, verifies every
mounted source listed in the full manifest, installs without indexes or
dependencies, checks the imported wheel, and replaces itself with the full
campaign driver.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


SOURCE_ROOT = Path("/workspace")
MANIFEST = SOURCE_ROOT / "hf_jobs/spinodal_attempt2b_full_sources.sha256"
REQUIREMENTS = SOURCE_ROOT / "hf_jobs/spinodal_attempt2b_h5py_requirements.txt"
WHEEL = SOURCE_ROOT / (
    "hf_jobs/wheels/"
    "h5py-3.14.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
TARGET = Path("/tmp/fluxnet-h5py-3.14.0")
DRIVER = SOURCE_ROOT / "repro/src/hf_spinodal_attempt2b_full.py"
WHEEL_SHA256 = "723a40ee6505bd354bfd26385f2dae7bbfa87655f4e61bab175a49d72ebfc06b"
WHEEL_BYTES = 4_516_618
PREFLIGHT_SHA256 = "ebc2bf33922783a4a06b1a48d02716d15be3740be67e4e0e6403146eb5f1df3e"
PREFLIGHT_REPORT = Path("/artifacts/hf-jobs/spinodal-attempt2b/preflight-v1.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime() -> dict[str, object]:
    libc_name, libc_version = platform.libc_ver()
    observed = {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "sys_platform": sys.platform,
        "machine": platform.machine(),
        "libc": [libc_name, libc_version],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    expected = {
        "implementation": "CPython",
        "python": "3.11.13",
        "sys_platform": "linux",
        "machine": "x86_64",
        "libc": ["glibc", "2.35"],
        "numpy": "2.2.6",
        "torch": "2.7.1+cu128",
        "torch_cuda": "12.8",
    }
    if observed != expected:
        raise RuntimeError(f"base-image runtime mismatch: {observed!r} != {expected!r}")
    return observed


def verify_mounted_sources() -> tuple[int, str]:
    records: dict[str, str] = {}
    for number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), start=1):
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
        path = SOURCE_ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"bootstrap mounted source hash mismatch: {relative}")
        records[relative] = digest
    required = {
        "hf_jobs/bootstrap_spinodal_attempt2b_h5py.py",
        "hf_jobs/spinodal_attempt2b_h5py_requirements.txt",
        str(WHEEL.relative_to(SOURCE_ROOT)),
        str(DRIVER.relative_to(SOURCE_ROOT)),
    }
    if not required.issubset(records):
        raise RuntimeError(f"bootstrap inventory missing: {sorted(required - set(records))}")
    return len(records), sha256_file(MANIFEST)


def main() -> None:
    runtime = verify_runtime()
    source_count, manifest_sha256 = verify_mounted_sources()
    if not WHEEL.is_file() or WHEEL.stat().st_size != WHEEL_BYTES:
        raise RuntimeError("staged h5py wheel size mismatch")
    if sha256_file(WHEEL) != WHEEL_SHA256:
        raise RuntimeError("staged h5py wheel SHA-256 mismatch")
    if sha256_file(PREFLIGHT_REPORT) != PREFLIGHT_SHA256:
        raise RuntimeError("returned preflight report SHA-256 mismatch")
    if TARGET.exists():
        raise RuntimeError(f"refusing pre-existing bootstrap target: {TARGET}")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--no-index",
        "--no-deps",
        "--no-compile",
        "--only-binary=:all:",
        "--require-hashes",
        "--ignore-installed",
        "--target",
        str(TARGET),
        "--find-links",
        str(WHEEL.parent),
        "-r",
        str(REQUIREMENTS),
    ]
    subprocess.run(command, check=True, timeout=300)
    sys.path.insert(0, str(TARGET))
    import h5py  # noqa: PLC0415

    imported = Path(h5py.__file__).resolve()
    if TARGET.resolve() not in imported.parents:
        raise RuntimeError(f"h5py imported outside isolated target: {imported}")
    if h5py.__version__ != "3.14.0":
        raise RuntimeError(f"unexpected h5py version: {h5py.__version__}")
    if np.__version__ != "2.2.6" or torch.__version__ != "2.7.1+cu128":
        raise RuntimeError("bootstrap changed a preflight-pinned scientific dependency")
    print(
        json.dumps(
            {
                "event": "spinodal_h5py_bootstrap_complete",
                "runtime": runtime,
                "source_count": source_count,
                "full_source_manifest_sha256": manifest_sha256,
                "wheel": str(WHEEL),
                "wheel_bytes": WHEEL_BYTES,
                "wheel_sha256": WHEEL_SHA256,
                "h5py": h5py.__version__,
                "hdf5": h5py.version.hdf5_version,
                "target": str(TARGET),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(TARGET)
    arguments = [
        sys.executable,
        str(DRIVER),
        "--source-root",
        str(SOURCE_ROOT),
        "--full-source-manifest",
        str(MANIFEST),
        "--preflight-source-manifest",
        str(SOURCE_ROOT / "hf_jobs/spinodal_attempt2b_preflight_sources.sha256"),
        "--preflight-report",
        str(PREFLIGHT_REPORT),
        "--run-root",
        "/artifacts/hf-jobs/spinodal-attempt2b/full-v1",
    ]
    os.execve(sys.executable, arguments, environment)


if __name__ == "__main__":
    main()
