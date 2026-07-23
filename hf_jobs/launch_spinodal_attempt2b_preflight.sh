#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${SCRIPT_DIR}/spinodal_attempt2b_preflight_sources.sha256"
DRIVER="${REPO_ROOT}/repro/src/hf_spinodal_attempt2b_preflight.py"
IMAGE="pytorch/pytorch@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9"
BUCKET_ID="DineshAI/1KRpajnd6u-artifacts"
BUCKET_MOUNT="/artifacts"
BUCKET_OUTPUT="hf-jobs/spinodal-attempt2b/preflight-v1.json"
TIMEOUT="55m"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" || ! -f "${DRIVER}" ]]; then
  echo "preflight driver or source manifest is missing" >&2
  exit 1
fi

(
  cd "${REPO_ROOT}"
  shasum -a 256 -c "${MANIFEST}"
)

OFFICIAL_HEAD="$(git -C "${REPO_ROOT}/official" rev-parse HEAD)"
if [[ "${OFFICIAL_HEAD}" != "ec0cafe3bb48cb7f2497723c5e12c6ebc518442c" ]]; then
  echo "official checkout is ${OFFICIAL_HEAD}; expected ec0cafe3bb48cb7f2497723c5e12c6ebc518442c" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO_ROOT}/official" status --porcelain --untracked-files=all)" ]]; then
  echo "official checkout is not clean" >&2
  exit 1
fi

COMMAND=(
  hf jobs run
  --detach
  --name fluxnet-spinodal-attempt2b-preflight
  --label paper=1KRpajnd6u
  --label purpose=spinodal-attempt2b-preflight
  --flavor t4-small
  --timeout "${TIMEOUT}"
  --env OMP_NUM_THREADS=4
  --env MKL_NUM_THREADS=1
  --volume "${REPO_ROOT}/official:/workspace/official:ro"
  --volume "${REPO_ROOT}/repro/src:/workspace/repro/src:ro"
  --volume "${REPO_ROOT}/docs:/workspace/docs:ro"
  --volume "${REPO_ROOT}/hf_jobs:/workspace/hf_jobs:ro"
  --volume "hf://buckets/${BUCKET_ID}:${BUCKET_MOUNT}:rw"
  "${IMAGE}"
  python /workspace/repro/src/hf_spinodal_attempt2b_preflight.py
  --source-root /workspace
  --source-manifest /workspace/hf_jobs/spinodal_attempt2b_preflight_sources.sha256
  --output "${BUCKET_MOUNT}/${BUCKET_OUTPUT}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'local_volume=%s\n' "${REPO_ROOT}/official:/workspace/official:ro"
  printf 'local_volume=%s\n' "${REPO_ROOT}/repro/src:/workspace/repro/src:ro"
  printf 'local_volume=%s\n' "${REPO_ROOT}/docs:/workspace/docs:ro"
  printf 'local_volume=%s\n' "${REPO_ROOT}/hf_jobs:/workspace/hf_jobs:ro"
  printf 'bucket=hf://buckets/%s:%s:rw\n' "${BUCKET_ID}" "${BUCKET_MOUNT}"
  printf 'bucket_output=%s/%s\n' "${BUCKET_ID}" "${BUCKET_OUTPUT}"
  printf 'flavor=t4-small timeout=%s max_cost_usd=0.366667 aggregate_cap_usd=10.00\n' "${TIMEOUT}"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

command -v hf >/dev/null
RATE="$(hf jobs hardware --format json | python3 -c '
import json, sys
rows = json.load(sys.stdin)
match = [row for row in rows if row.get("name") == "t4-small"]
if len(match) != 1 or match[0].get("cost/hour") != "$0.40":
    raise SystemExit("t4-small rate is not the authorized $0.40/hour")
print(match[0]["cost/hour"])
')"
if [[ "${RATE}" != '$0.40' ]]; then
  echo "unexpected t4-small rate: ${RATE}" >&2
  exit 1
fi
hf auth whoami --format json >/dev/null
exec "${COMMAND[@]}"
