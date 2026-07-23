#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=0.36,<2"]
# ///
"""Fail-closed existing-Space publisher for the terminal FluxNet C3 repair.

The tool has three modes. ``--verify-parent-only`` performs a read-only live
check of the complete pinned 18-file parent. ``--check-only`` additionally
requires a terminal supporting import and an exact full-tree candidate.
``--execute`` is the sole mutating mode and requires root approval values that
bind the parent, candidate tree, terminal report, and official validator pass.

No mode creates a Space, deletes a remote path, starts a Job, or imports Job
artifacts. The mutating mode uses Hugging Face compare-and-swap through
``HfApi.create_commit(parent_commit=...)`` and verifies the exact 18-file union
at the returned commit before writing a local receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tempfile

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parents[2]
SPACE_ID = "DineshAI/1KRpajnd6u"
PARENT_REVISION = "e367aa22ce98c5926ee41ac5a74a7a2fbf78f364"
PARENT_MANIFEST = ROOT / "hf_jobs/spinodal_attempt2b_space_parent_complete.sha256"
PARENT_MANIFEST_SHA256 = "a10b68fb0ea8e748f74c5cb3bd984da495cd3808c6c48844fc523968aff34696"
PARENT_SNAPSHOT = ROOT / "hf_jobs/spinodal_attempt2b_space_parent_snapshot.json"
TERMINAL_REPORT = ROOT / "outputs/hf-jobs/spinodal-attempt2b/terminal-import-verification.json"
READINESS_CHECKER = ROOT / "repro/src/check_spinodal_space_repair_readiness.py"
RECEIPT = ROOT / "outputs/hf-jobs/spinodal-attempt2b/space-repair-publication-receipt.json"
APPROVAL_ENV = "FLUXNET_SPACE_REPAIR_APPROVED"
OFFICIAL_VALIDATOR_COMMAND = (
    "python ../../tmp/validate_icml_logbook.py --space DineshAI/1KRpajnd6u"
)

# Only C3 and its dependent navigation/summaries may change. The remaining
# parent files, including C1, C2, .gitattributes, and style.css, are immutable.
ALLOWED_CHANGED_PATHS = {
    "logbook.json",
    "pages/claim-3-empirical-rollouts/page.md",
    "pages/conclusion/page.md",
    "pages/executive-summary/page.md",
    "pages/index.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_manifest() -> dict[str, str]:
    require(PARENT_MANIFEST.is_file(), "complete parent manifest is absent")
    require(sha256(PARENT_MANIFEST) == PARENT_MANIFEST_SHA256,
            "complete parent manifest hash drift")
    entries: dict[str, str] = {}
    for line in PARENT_MANIFEST.read_text().splitlines():
        pieces = line.split("  ", 1)
        require(len(pieces) == 2, "malformed complete parent manifest line")
        digest, relative = pieces
        pure = PurePosixPath(relative)
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "malformed complete parent digest")
        require(relative == pure.as_posix() and not pure.is_absolute()
                and ".." not in pure.parts, "unsafe complete parent path")
        require(relative not in entries, "duplicate complete parent path")
        entries[relative] = digest
    require(len(entries) == 18, "complete parent manifest must contain exactly 18 files")
    require(".gitattributes" in entries and "style.css" in entries,
            "complete parent manifest omits required root files")
    return entries


def verify_snapshot_contract(parent: dict[str, str]) -> None:
    contract = json.loads(PARENT_SNAPSHOT.read_text())
    require(contract == {
        "captured_at": "2026-07-20T12:45:02Z",
        "file_count": 18,
        "manifest": "hf_jobs/spinodal_attempt2b_space_parent_complete.sha256",
        "manifest_sha256": PARENT_MANIFEST_SHA256,
        "repo_id": SPACE_ID,
        "repo_type": "space",
        "schema_version": 1,
        "source": "immutable Hugging Face snapshot downloaded at the exact revision and independently hashed",
        "source_revision": PARENT_REVISION,
    }, "parent snapshot contract drift")
    require(len(parent) == contract["file_count"], "parent snapshot file-count drift")


def inventory(root: Path, *, reject_symlinks: bool) -> dict[str, str]:
    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if reject_symlinks and path.is_symlink():
            raise RuntimeError(f"symlink forbidden in candidate: {path.relative_to(root)}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            entries[relative] = sha256(path)
    return entries


def tree_sha256(entries: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"fluxnet-space-tree-v1\0")
    for relative, file_digest in sorted(entries.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def verify_live_parent(api: HfApi, parent: dict[str, str]) -> dict[str, object]:
    info = api.repo_info(repo_id=SPACE_ID, repo_type="space")
    require(info.sha == PARENT_REVISION,
            f"live Space parent drift: {info.sha} != {PARENT_REVISION}")
    remote_paths = set(api.list_repo_files(
        repo_id=SPACE_ID, repo_type="space", revision=PARENT_REVISION,
    ))
    require(remote_paths == set(parent), "live parent path inventory differs from frozen 18-file union")
    with tempfile.TemporaryDirectory(prefix="fluxnet-space-parent-") as temporary:
        snapshot = Path(api.snapshot_download(
            repo_id=SPACE_ID,
            repo_type="space",
            revision=PARENT_REVISION,
            cache_dir=temporary,
        ))
        observed = inventory(snapshot, reject_symlinks=False)
    require(observed == parent, "live parent content differs from frozen complete manifest")
    return {
        "space_id": SPACE_ID,
        "parent_revision": PARENT_REVISION,
        "parent_file_count": len(parent),
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_tree_sha256": tree_sha256(parent),
    }


def require_terminal_support() -> dict[str, object]:
    process = subprocess.run(
        [sys.executable, str(READINESS_CHECKER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    require(process.returncode == 0,
            "terminal-only readiness gate refused: " +
            (process.stderr.strip() or process.stdout.strip() or f"exit {process.returncode}"))
    readiness = json.loads(process.stdout)
    require(readiness.get("status") == "eligible_for_root_candidate_build_only"
            and readiness.get("publication_authority") is False,
            "terminal-only readiness output drift")
    require(readiness.get("expected_parent_sha") == PARENT_REVISION,
            "terminal readiness parent mismatch")
    require(TERMINAL_REPORT.is_file(), "terminal report disappeared after readiness check")
    return readiness


def verify_candidate(candidate: Path, parent: dict[str, str]) -> tuple[dict[str, str], list[str], str]:
    require(candidate.is_dir(), "candidate directory is absent")
    observed = inventory(candidate, reject_symlinks=True)
    require(set(observed) == set(parent), "candidate must contain the exact 18-file parent path union")
    changed = sorted(relative for relative in observed if observed[relative] != parent[relative])
    require(changed, "candidate does not change any file")
    require(set(changed).issubset(ALLOWED_CHANGED_PATHS),
            "candidate changes a forbidden parent path: " +
            ", ".join(sorted(set(changed) - ALLOWED_CHANGED_PATHS)))
    require(observed["pages/claim-1-conservation/page.md"] ==
            parent["pages/claim-1-conservation/page.md"], "C1 parent content changed")
    require(observed["pages/claim-2-structural-bounds/page.md"] ==
            parent["pages/claim-2-structural-bounds/page.md"], "C2 parent content changed")
    return observed, changed, tree_sha256(observed)


def approval_command(candidate: Path, candidate_tree: str, terminal_report_sha: str) -> str:
    return (
        f"{APPROVAL_ENV}=1 uv run repro/src/publish_spinodal_space_repair.py --execute "
        f"--candidate-dir {shlex.quote(str(candidate))} "
        f"--approve-parent-commit {PARENT_REVISION} "
        f"--approve-candidate-tree-sha256 {candidate_tree} "
        f"--approve-terminal-report-sha256 {terminal_report_sha} "
        "--confirm-official-validator-passed"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-parent-only", action="store_true")
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--approve-parent-commit")
    parser.add_argument("--approve-candidate-tree-sha256")
    parser.add_argument("--approve-terminal-report-sha256")
    parser.add_argument("--confirm-official-validator-passed", action="store_true")
    args = parser.parse_args()

    parent = load_manifest()
    verify_snapshot_contract(parent)
    api = HfApi()
    live = verify_live_parent(api, parent)
    if args.verify_parent_only:
        print(json.dumps({
            "status": "exact_18_file_parent_verified_read_only",
            **live,
            "publication_authority": False,
        }, indent=2, sort_keys=True))
        return

    require(args.candidate_dir is not None, "--candidate-dir is required")
    readiness = require_terminal_support()
    candidate = args.candidate_dir.resolve()
    candidate_inventory, changed, candidate_tree = verify_candidate(candidate, parent)
    terminal_report_sha = sha256(TERMINAL_REPORT)
    ready = {
        "status": "eligible_for_explicit_root_approval_only",
        **live,
        "candidate_dir": str(candidate),
        "candidate_file_count": len(candidate_inventory),
        "candidate_tree_sha256": candidate_tree,
        "changed_paths": changed,
        "terminal_report_sha256": terminal_report_sha,
        "terminal_snapshot_sha256": readiness["snapshot_sha256"],
        "official_validator_command": OFFICIAL_VALIDATOR_COMMAND,
        "publication_authority": False,
    }
    if args.check_only:
        ready["approval_command_after_validator_pass"] = approval_command(
            candidate, candidate_tree, terminal_report_sha,
        )
        print(json.dumps(ready, indent=2, sort_keys=True))
        return

    require(RECEIPT.exists() is False, "publication receipt already exists")
    require(os.environ.get(APPROVAL_ENV) == "1", f"{APPROVAL_ENV}=1 is required")
    require(args.approve_parent_commit == PARENT_REVISION,
            "explicit parent approval is absent or mismatched")
    require(args.approve_candidate_tree_sha256 == candidate_tree,
            "explicit candidate-tree approval is absent or mismatched")
    require(args.approve_terminal_report_sha256 == terminal_report_sha,
            "explicit terminal-report approval is absent or mismatched")
    require(args.confirm_official_validator_passed,
            "explicit official-validator confirmation is absent")

    operations = [CommitOperationAdd(
        path_in_repo=relative,
        path_or_fileobj=str(candidate / relative),
    ) for relative in changed]
    commit = api.create_commit(
        repo_id=SPACE_ID,
        repo_type="space",
        operations=operations,
        commit_message="Add terminal spinodal and shallow-water C3 evidence",
        commit_description=(
            "Existing-Space-only exact-parent repair. C1/C2 and all non-C3 parent files "
            "remain byte-identical; adverse and scale-limited evidence is retained."
        ),
        parent_commit=PARENT_REVISION,
    )
    commit_oid = commit.oid
    remote_paths = set(api.list_repo_files(
        repo_id=SPACE_ID, repo_type="space", revision=commit_oid,
    ))
    require(remote_paths == set(candidate_inventory),
            "post-commit remote inventory differs from exact candidate union")
    with tempfile.TemporaryDirectory(prefix="fluxnet-space-readback-") as temporary:
        readback = Path(api.snapshot_download(
            repo_id=SPACE_ID,
            repo_type="space",
            revision=commit_oid,
            cache_dir=temporary,
        ))
        readback_inventory = inventory(readback, reject_symlinks=False)
    require(readback_inventory == candidate_inventory,
            "post-commit full-tree readback differs from approved candidate")

    receipt = {
        **ready,
        "status": "existing_space_repair_committed_and_exact_union_read_back",
        "commit_oid": commit_oid,
        "commit_url": commit.commit_url,
        "readback_file_count": len(readback_inventory),
        "readback_tree_sha256": tree_sha256(readback_inventory),
        "parent_commit_cas": PARENT_REVISION,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = RECEIPT.with_suffix(".json.tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary_receipt.replace(RECEIPT)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(4) from error
