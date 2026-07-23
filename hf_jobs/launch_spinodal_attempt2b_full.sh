#!/usr/bin/env bash
set -euo pipefail

if [[ "${FLUXNET_ATTEMPT2B_LAUNCH_LOCK_HELD:-}" != "1" ]]; then
  exec python3 - "/tmp/fluxnet-spinodal-attempt2b-full-v1.launch.lock" "$0" "$@" <<'PY'
import fcntl
import os
import sys
from pathlib import Path

lock_path = Path(sys.argv[1])
script = str(Path(sys.argv[2]).resolve())
arguments = sys.argv[3:]
descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(descriptor, fcntl.LOCK_EX)
os.set_inheritable(descriptor, True)
environment = dict(os.environ)
environment["FLUXNET_ATTEMPT2B_LAUNCH_LOCK_HELD"] = "1"
os.execvpe("bash", ["bash", script, *arguments], environment)
PY
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEDGER="${REPO_ROOT}/../../icml-2026-reproduction-challenge/HF_JOBS_BUDGET.md"
MANIFEST="${SCRIPT_DIR}/spinodal_attempt2b_full_sources.sha256"
IMAGE="pytorch/pytorch@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9"
BUCKET_ID="DineshAI/1KRpajnd6u-artifacts"
BUCKET_MOUNT="/artifacts"
RUN_ROOT="${BUCKET_MOUNT}/hf-jobs/spinodal-attempt2b/full-v1"
TIMEOUT="12h"
MAX_COST_USD="4.80"
AUTHORIZED_CAMPAIGN_CAP_USD="40.00"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "full campaign source manifest is missing" >&2
  exit 1
fi
LEDGER_REMAINING_USD="$(python3 - "${LEDGER}" "${MAX_COST_USD}" <<'PY'
import re
import sys
from pathlib import Path

ledger = Path(sys.argv[1])
maximum_cost = float(sys.argv[2])
matches = re.findall(
    r"^Uncommitted balance:\s*\*\*USD ([0-9]+(?:\.[0-9]+)?)\*\*\s*$",
    ledger.read_text(encoding="utf-8"),
    flags=re.MULTILINE,
)
if len(matches) != 1:
    raise SystemExit("could not resolve exactly one uncommitted balance from the HF budget ledger")
remaining = float(matches[0])
if maximum_cost > remaining:
    raise SystemExit(
        f"job maximum cost USD {maximum_cost:.2f} exceeds ledger balance USD {remaining:.2f}"
    )
print(f"{remaining:.2f}")
PY
)"
python3 - "${REPO_ROOT}" "${MANIFEST}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    digest, relative = line.split(maxsplit=1)
    path = root / relative.lstrip("*")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != digest:
        raise SystemExit(f"source hash mismatch at manifest line {number}: {relative}")
    print(f"{relative}: OK")
preflight = root / "outputs/hf-jobs/spinodal-attempt2b/preflight-v1.json"
if hashlib.sha256(preflight.read_bytes()).hexdigest() != "ebc2bf33922783a4a06b1a48d02716d15be3740be67e4e0e6403146eb5f1df3e":
    raise SystemExit("local returned preflight report hash mismatch")
print(f"{preflight.relative_to(root)}: OK")
PY
OFFICIAL_HEAD="$(git -C "${REPO_ROOT}/official" rev-parse HEAD)"
if [[ "${OFFICIAL_HEAD}" != "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c" ]]; then
  echo "official checkout commit mismatch: ${OFFICIAL_HEAD}" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO_ROOT}/official" status --porcelain --untracked-files=all)" ]]; then
  echo "official checkout is not clean" >&2
  exit 1
fi

COMMAND=(
  hf jobs run
  --detach
  --name fluxnet-spinodal-attempt2b-full-v1
  --label paper=1KRpajnd6u
  --label purpose=spinodal-attempt2b-full-v1
  --label attempt=3
  --flavor t4-small
  --timeout "${TIMEOUT}"
  --env OMP_NUM_THREADS=4
  --env MKL_NUM_THREADS=1
  --env PYTHONDONTWRITEBYTECODE=1
  --volume "${REPO_ROOT}/official:/workspace/official:ro"
  --volume "${REPO_ROOT}/repro/src:/workspace/repro/src:ro"
  --volume "${REPO_ROOT}/docs:/workspace/docs:ro"
  --volume "${REPO_ROOT}/hf_jobs:/workspace/hf_jobs:ro"
  --volume "hf://buckets/${BUCKET_ID}:${BUCKET_MOUNT}:rw"
  "${IMAGE}"
  python /workspace/hf_jobs/bootstrap_spinodal_attempt2b_h5py.py
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'flavor=t4-small timeout=%s max_cost_usd=%s authorized_campaign_cap_usd=%s ledger_remaining_usd=%s\n' "${TIMEOUT}" "${MAX_COST_USD}" "${AUTHORIZED_CAMPAIGN_CAP_USD}" "${LEDGER_REMAINING_USD}"
  printf 'bucket=hf://buckets/%s:%s:rw\n' "${BUCKET_ID}" "${BUCKET_MOUNT}"
  printf 'run_root=%s/hf-jobs/spinodal-attempt2b/full-v1\n' "${BUCKET_ID}"
  printf 'resume_command=%q\n' "bash ${SCRIPT_DIR}/launch_spinodal_attempt2b_full.sh"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

command -v hf >/dev/null
hf auth whoami --format json >/dev/null
RATE="$(hf jobs hardware --format json | python3 -c '
import json, sys
match = [row for row in json.load(sys.stdin) if row.get("name") == "t4-small"]
if len(match) != 1 or match[0].get("cost/hour") != "$0.40":
    raise SystemExit("t4-small rate is not the authorized $0.40/hour")
print(match[0]["cost/hour"])
')"
if [[ "${RATE}" != '$0.40' ]]; then
  echo "unexpected t4-small rate: ${RATE}" >&2
  exit 1
fi
ACTIVE_JOBS_JSON="$(hf jobs list --status RUNNING,SCHEDULING --limit 0 --format json)"
python3 -c '
import json, sys
jobs = json.load(sys.stdin)
summary = [
    {
        "id": job.get("id"),
        "name": job.get("labels", {}).get("name"),
        "flavor": job.get("flavor"),
        "status": job.get("status"),
    }
    for job in jobs
]
print("active HF jobs (read-only inspection): " + json.dumps(summary, sort_keys=True), file=sys.stderr)
' <<<"${ACTIVE_JOBS_JSON}"
ACTIVE_DUPLICATES="$(python3 -c '
import json, sys
jobs = json.load(sys.stdin)
print(sum(job.get("labels", {}).get("name") == "fluxnet-spinodal-attempt2b-full-v1" for job in jobs))
' <<<"${ACTIVE_JOBS_JSON}")"
if [[ "${ACTIVE_DUPLICATES}" != "0" ]]; then
  echo "a FluxNet spinodal full-v1 job is already running or scheduling" >&2
  exit 1
fi
ACTIVE_T4_JOBS="$(python3 -c '
import json, sys
jobs = json.load(sys.stdin)
print(sum(job.get("flavor") in {"t4-small", "t4-medium"} for job in jobs))
' <<<"${ACTIVE_JOBS_JSON}")"
if [[ "${ACTIVE_T4_JOBS}" != "0" ]]; then
  echo "another T4 job is active; deferring FluxNet so the shared account retains capacity" >&2
  exit 1
fi
# The advisory lock serializes this launcher on one host. A final backend
# recheck narrows, but cannot eliminate, the HF API check-to-submit race across
# different hosts because Jobs currently provides no atomic idempotency key.
FINAL_ACTIVE_JOBS_JSON="$(hf jobs list --status RUNNING,SCHEDULING --limit 0 --format json)"
FINAL_ACTIVE_DUPLICATES="$(python3 -c '
import json, sys
jobs = json.load(sys.stdin)
print(sum(job.get("labels", {}).get("name") == "fluxnet-spinodal-attempt2b-full-v1" for job in jobs))
' <<<"${FINAL_ACTIVE_JOBS_JSON}")"
if [[ "${FINAL_ACTIVE_DUPLICATES}" != "0" ]]; then
  echo "a FluxNet spinodal full-v1 job appeared before submission" >&2
  exit 1
fi
FINAL_ACTIVE_T4_JOBS="$(python3 -c '
import json, sys
jobs = json.load(sys.stdin)
print(sum(job.get("flavor") in {"t4-small", "t4-medium"} for job in jobs))
' <<<"${FINAL_ACTIVE_JOBS_JSON}")"
if [[ "${FINAL_ACTIVE_T4_JOBS}" != "0" ]]; then
  echo "a T4 job appeared before submission; deferring FluxNet" >&2
  exit 1
fi
exec "${COMMAND[@]}"
