"""Local-only, fail-closed gate before building a spinodal Space repair.

This checker performs no network request, HF mutation, publication, test, or
scientific computation. It binds the terminal import report to its immutable
hash-named snapshot and verifies that the exact published parent files remain
unchanged. It never grants publication authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PAPER_ID = "1KRpajnd6u"
EXISTING_SPACE = "DineshAI/1KRpajnd6u"
PARENT_SPACE_SHA = "e367aa22ce98c5926ee41ac5a74a7a2fbf78f364"
JOB_ID = "6a5dd080bee6ee1cf4ed215e"
DATASET_SHA256 = "d437b96f6c7f8b0b5ff4b530af0d1fd8b74f2e57a9422f59c5e008b0fb193b06"
IMPORTER_SHA256 = "154f322af249f2370646deb3d9981d5b7a79e5d1d4fa604ec567cb748461d972"
AUDITOR_SHA256 = "ead541330615c12005252f551e95f840a4abf79f04e19c6311ee2e3b709ae44f"
FULL_MANIFEST_SHA256 = "f40896a9d8871f649b37d1318aaf7de76a89f2e2c4e31128cbc798ffffc6eff5"
PARENT_MANIFEST = ROOT / "hf_jobs/spinodal_attempt2b_space_parent_sources.sha256"
REPORT = ROOT / "outputs/hf-jobs/spinodal-attempt2b/terminal-import-verification.json"
VERIFIED_ROOT = ROOT / "outputs/hf-jobs/spinodal-attempt2b/verified-returns"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_parent() -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in PARENT_MANIFEST.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        require(relative not in expected and len(digest) == 64, "malformed parent manifest")
        expected[relative] = digest
    require(len(expected) == 16, "published parent manifest must contain 16 files")
    for relative, digest in expected.items():
        path = ROOT / relative
        require(path.is_file() and sha256(path) == digest, f"published parent drift: {relative}")
    return expected


def main() -> None:
    parent = verify_parent()
    require(sha256(ROOT / "repro/src/import_verify_hf_spinodal_attempt2b.py") == IMPORTER_SHA256,
            "terminal importer drift")
    require(sha256(ROOT / "repro/src/audit_spinodal_attempt2b.py") == AUDITOR_SHA256,
            "independent raw-HDF5 auditor drift")
    require(sha256(ROOT / "hf_jobs/spinodal_attempt2b_full_sources.sha256") ==
            FULL_MANIFEST_SHA256, "full source manifest drift")
    if not REPORT.is_file():
        print(json.dumps({
            "status": "blocked_terminal_import_absent",
            "paper_id": PAPER_ID,
            "job_id": f"DineshAI/{JOB_ID}",
            "existing_space": EXISTING_SPACE,
            "expected_parent_sha": PARENT_SPACE_SHA,
            "preserved_parent_files": len(parent),
            "publication_authority": False,
        }, indent=2, sort_keys=True))
        raise SystemExit(3)

    report = json.loads(REPORT.read_text())
    require(report.get("schema_version") == 1 and report.get("status") == "verified",
            "terminal report is not verified")
    require(report.get("paper_id") == PAPER_ID, "terminal report paper mismatch")
    require(report.get("job", {}).get("id") == JOB_ID, "terminal report Job mismatch")
    require(report.get("job", {}).get("status") == "COMPLETED", "terminal Job is not completed")
    require(report.get("dataset_sha256") == DATASET_SHA256, "terminal dataset mismatch")
    require(report.get("source_pins") == {
        "full_source_manifest": FULL_MANIFEST_SHA256,
        "frozen_runner": "6e574948dcf9d50a8d21ecb62924d3d6ede82ca2d609490c2700abb44a1f9762",
        "independent_auditor": AUDITOR_SHA256,
    }, "terminal source pins mismatch")
    require(report.get("independent_raw_hdf5_reaudit") == "passed",
            "independent raw-HDF5 audit did not pass")
    require(set(report.get("negative_controls_detected", [])) == {
        "nonterminal_completion", "state_completion_hash", "manifest_digest",
        "prediction_hash", "verdict_flip", "recorded_metric",
    }, "terminal negative-control inventory mismatch")
    result = report.get("result", {})
    require(result.get("verdict") == "supports" and result.get("supports") is True
            and result.get("falsified") is False, "spinodal verdict is not supports")
    require(result.get("trajectory_count") == 20 and result.get("finite_trajectories") == 20,
            "spinodal trajectory completion mismatch")
    gate = report.get("publication_gate", {})
    require(gate == {
        "publication_performed": False,
        "existing_space_only": EXISTING_SPACE,
        "expected_pre_update_space_sha": PARENT_SPACE_SHA,
        "eligible_for_root_review": True,
        "requires_official_validator": True,
        "requires_root_approval": True,
        "new_space_forbidden": True,
    }, "terminal publication boundary mismatch")
    snapshot = Path(report.get("local_snapshot_path", ""))
    require(snapshot.parent == VERIFIED_ROOT and snapshot.name == report.get("local_snapshot_sha256"),
            "terminal snapshot is not in the immutable hash-named directory")
    require(snapshot.is_dir(), "terminal snapshot directory is absent")
    for relative in (
        "completion.json", "state.json", "provenance.json", "data/manifest.json",
        "outputs/evaluation.json", "outputs/audit_certificate.json",
    ):
        require((snapshot / relative).is_file(), f"terminal snapshot missing {relative}")
    require(sha256(snapshot / "completion.json") == report.get("completion_sha256"),
            "terminal completion hash mismatch")
    print(json.dumps({
        "status": "eligible_for_root_candidate_build_only",
        "paper_id": PAPER_ID,
        "job_id": f"DineshAI/{JOB_ID}",
        "existing_space": EXISTING_SPACE,
        "expected_parent_sha": PARENT_SPACE_SHA,
        "terminal_report_sha256": sha256(REPORT),
        "snapshot_sha256": report["local_snapshot_sha256"],
        "verdict": result["verdict"],
        "preserved_parent_files": len(parent),
        "publication_authority": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(4) from error
