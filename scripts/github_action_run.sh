#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
HARNESS_CONFIG="${INPUT_HARNESS_CONFIG:-ci/harness.yaml}"
BASELINE_REF_INPUT="${INPUT_BASELINE_REF:-}"
INSTALL_EXTRAS="${INPUT_INSTALL_EXTRAS:-}"
RUN_ARGS_RAW="${INPUT_RUN_ARGS:-}"
DIFF_ARGS_RAW="${INPUT_DIFF_ARGS:-}"
FAIL_ON_CHANGES="${INPUT_FAIL_ON_CHANGES:-true}"
EVENT_PATH="${GITHUB_EVENT_PATH:-}"

BASELINE_SHA=""
IS_FORK_PR="false"
if [[ -n "${EVENT_PATH}" && -f "${EVENT_PATH}" ]]; then
  mapfile -t EVENT_META < <(
    python - "${EVENT_PATH}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    print("false")
    raise SystemExit(0)

pr = payload.get("pull_request") or {}
base = pr.get("base") or {}
head = pr.get("head") or {}
print(base.get("sha", ""))
print(str(bool((head.get("repo") or {}).get("fork", False))).lower())
PY
  )
  BASELINE_SHA="${EVENT_META[0]:-}"
  IS_FORK_PR="${EVENT_META[1]:-false}"
fi

BASELINE_SELECTOR=""
BASELINE_SOURCE=""
if [[ -n "${BASELINE_REF_INPUT}" ]]; then
  BASELINE_SELECTOR="${BASELINE_REF_INPUT}"
  BASELINE_SOURCE="input:baseline-ref"
elif [[ -n "${BASELINE_SHA}" ]]; then
  BASELINE_SELECTOR="${BASELINE_SHA}"
  BASELINE_SOURCE="event:pull_request.base.sha"
elif [[ -n "${GITHUB_BASE_REF:-}" ]]; then
  BASELINE_SELECTOR="${GITHUB_BASE_REF}"
  BASELINE_SOURCE="env:GITHUB_BASE_REF"
else
  BASELINE_SELECTOR="main"
  BASELINE_SOURCE="default:main"
fi

if [[ ! -f "${WORKSPACE}/${HARNESS_CONFIG}" ]]; then
  echo "Harness config not found: ${HARNESS_CONFIG}" >&2
  exit 1
fi

if [[ -n "${INSTALL_EXTRAS}" ]]; then
  EXTRA_SPEC="[${INSTALL_EXTRAS}]"
else
  EXTRA_SPEC=""
fi

BASE_WORKTREE="${RUNNER_TEMP:-/tmp}/insidellms-base-${GITHUB_RUN_ID:-local}-${RANDOM}"
BASE_RUN_DIR="${RUNNER_TEMP:-/tmp}/insidellms-run-base-${GITHUB_RUN_ID:-local}-${RANDOM}"
HEAD_RUN_DIR="${RUNNER_TEMP:-/tmp}/insidellms-run-head-${GITHUB_RUN_ID:-local}-${RANDOM}"
DIFF_JSON="${RUNNER_TEMP:-/tmp}/insidellms-diff-${GITHUB_RUN_ID:-local}-${RANDOM}.json"

cleanup() {
  git -C "${WORKSPACE}" worktree remove --force "${BASE_WORKTREE}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "${WORKSPACE}"

resolve_ref_to_commit() {
  local ref="$1"
  ref="${ref#refs/heads/}"
  ref="${ref#origin/}"
  if git fetch --no-tags --prune --depth=1 origin "${ref}" >/dev/null 2>&1; then
    git rev-parse FETCH_HEAD
    return 0
  fi
  if git fetch --no-tags --prune origin "${ref}" >/dev/null 2>&1; then
    git rev-parse FETCH_HEAD
    return 0
  fi
  return 1
}

BASELINE_COMMIT=""
if [[ "${BASELINE_SELECTOR}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  if ! git fetch --no-tags --prune --depth=1 origin "${BASELINE_SELECTOR}" >/dev/null 2>&1; then
    echo "Warning: unable to fetch baseline SHA from origin, using local object if available." >&2
  fi
  BASELINE_COMMIT="${BASELINE_SELECTOR}"
else
  if ! BASELINE_COMMIT="$(resolve_ref_to_commit "${BASELINE_SELECTOR}")"; then
    echo "Unable to resolve baseline ref '${BASELINE_SELECTOR}' from origin." >&2
    exit 1
  fi
fi

if ! git cat-file -e "${BASELINE_COMMIT}^{commit}" >/dev/null 2>&1; then
  echo "Baseline commit is not available locally: ${BASELINE_COMMIT}" >&2
  exit 1
fi

git worktree add --detach "${BASE_WORKTREE}" "${BASELINE_COMMIT}"

BASE_HARNESS_CONFIG="${BASE_WORKTREE}/${HARNESS_CONFIG}"
if [[ ! -f "${BASE_HARNESS_CONFIG}" ]]; then
  echo "Harness config not found on baseline commit '${BASELINE_COMMIT}' (${BASELINE_SOURCE}): ${HARNESS_CONFIG}" >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -e "${BASE_WORKTREE}${EXTRA_SPEC}"

RUN_ARGS=()
if [[ -n "${RUN_ARGS_RAW}" ]]; then
  # shellcheck disable=SC2206
  RUN_ARGS=(${RUN_ARGS_RAW})
fi

(
  cd "${BASE_WORKTREE}"
  python -m insideLLMs.cli harness \
    "${HARNESS_CONFIG}" \
    --run-dir "${BASE_RUN_DIR}" \
    --overwrite \
    --skip-report \
    "${RUN_ARGS[@]}"
)

python -m pip install -e "${WORKSPACE}${EXTRA_SPEC}"
(
  cd "${WORKSPACE}"
  python -m insideLLMs.cli harness \
    "${HARNESS_CONFIG}" \
    --run-dir "${HEAD_RUN_DIR}" \
    --overwrite \
    --skip-report \
    "${RUN_ARGS[@]}"
)

DIFF_ARGS=()
if [[ -n "${DIFF_ARGS_RAW}" ]]; then
  # shellcheck disable=SC2206
  DIFF_ARGS=(${DIFF_ARGS_RAW})
fi

DIFF_CMD=(
  python -m insideLLMs.cli diff
  "${BASE_RUN_DIR}"
  "${HEAD_RUN_DIR}"
  --format json
  --output "${DIFF_JSON}"
  "${DIFF_ARGS[@]}"
)
if [[ "${FAIL_ON_CHANGES}" == "true" ]]; then
  DIFF_CMD+=(--fail-on-changes)
fi

set +e
"${DIFF_CMD[@]}"
DIFF_EXIT_CODE=$?
set -e

if [[ ! -f "${DIFF_JSON}" ]]; then
  cat >"${DIFF_JSON}" <<EOF
{"error":"insidellms diff did not produce JSON output","exit_code":${DIFF_EXIT_CODE}}
EOF
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "diff_json=${DIFF_JSON}"
    echo "baseline_run_dir=${BASE_RUN_DIR}"
    echo "candidate_run_dir=${HEAD_RUN_DIR}"
    echo "diff_exit_code=${DIFF_EXIT_CODE}"
    echo "baseline_commit=${BASELINE_COMMIT}"
    echo "is_fork_pr=${IS_FORK_PR}"
  } >>"${GITHUB_OUTPUT}"
fi
