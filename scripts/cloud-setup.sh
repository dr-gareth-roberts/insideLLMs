#!/usr/bin/env bash
# Cloud coding agent bootstrap for insideLLMs.
# Offline-capable core env. No provider keys required.
#
# Usage:
#   bash scripts/cloud-setup.sh
#   CLOUD_EXTRAS=providers bash scripts/cloud-setup.sh   # optional overlays
#
# CLOUD_EXTRAS is a comma-separated list from:
#   providers | openai | anthropic | huggingface | nlp | visualization | serving | all
set -euo pipefail

cd "$(dirname "$0")/.."

export PATH="${HOME}/.local/bin:${PATH}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1

# --- Python ---
if command -v python3.12 >/dev/null 2>&1; then
  PY=python3.12
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "python3 not found" >&2
  exit 1
fi

# Optional: pin for tools that look for `python`
if ! command -v python >/dev/null 2>&1; then
  mkdir -p "${HOME}/.local/bin"
  ln -sfn "$(command -v "$PY")" "${HOME}/.local/bin/python"
fi

install_extras() {
  # Base always: editable [dev] + pydantic (validate/CLI schema path).
  local base=( -e ".[dev]" "pydantic>=2" )
  local extras=()
  local token
  IFS=',' read -r -a tokens <<< "${CLOUD_EXTRAS:-}"
  for token in "${tokens[@]}"; do
    token="$(echo "$token" | tr '[:upper:]' '[:lower:]' | xargs)"
    [[ -z "$token" ]] && continue
    case "$token" in
      providers|openai|anthropic|huggingface|nlp|visualization|serving|crypto|signing|langchain|all)
        extras+=( "$token" )
        ;;
      *)
        echo "cloud-setup: unknown CLOUD_EXTRAS entry: $token" >&2
        exit 2
        ;;
    esac
  done

  if ((${#extras[@]})); then
    local joined
    joined="$(IFS=,; echo "${extras[*]}")"
    base=( -e ".[dev,${joined}]" "pydantic>=2" )
  fi

  if command -v uv >/dev/null 2>&1; then
    uv pip install "${base[@]}"
  else
    python -m pip install "${base[@]}"
  fi
}

# --- Install ---
# Prefer uv (fast, respects lock when used). Fall back to venv+pip.
if command -v uv >/dev/null 2>&1; then
  uv venv .venv --python "$PY" 2>/dev/null || true
  # shellcheck disable=SC1091
  source .venv/bin/activate
  install_extras
else
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install -U pip setuptools wheel
  install_extras
fi

# Optional NLP corpora only when nlp/all was requested (large download).
if [[ ",${CLOUD_EXTRAS:-}," == *",nlp,"* || ",${CLOUD_EXTRAS:-}," == *",all,"* ]]; then
  python - <<'PY' || true
try:
    import nltk
    for pkg in ("punkt", "averaged_perceptron_tagger", "punkt_tab"):
        nltk.download(pkg, quiet=True)
except Exception as exc:  # noqa: BLE001 — best-effort bootstrap
    print(f"cloud-setup: nltk download skipped: {exc}")
PY
fi

# --- Env (no secrets) ---
mkdir -p .tmp/runs .cache/insidellms
if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  # Strip placeholder secrets so tools don't treat example values as real keys.
  if sed --version >/dev/null 2>&1; then
    sed -i -E 's/^(OPENAI_API_KEY|ANTHROPIC_API_KEY|COHERE_API_KEY|GOOGLE_API_KEY|HUGGINGFACE_TOKEN)=.*/\1=/' .env
  else
    sed -i.bak -E 's/^(OPENAI_API_KEY|ANTHROPIC_API_KEY|COHERE_API_KEY|GOOGLE_API_KEY|HUGGINGFACE_TOKEN)=.*/\1=/' .env
    rm -f .env.bak
  fi
fi
export INSIDELLMS_RUN_ROOT="${INSIDELLMS_RUN_ROOT:-.tmp/runs}"
export INSIDELLMS_CACHE_DIR="${INSIDELLMS_CACHE_DIR:-.cache/insidellms}"

# --- Smoke (must pass with zero API keys) ---
python -c "import insideLLMs; print('import ok', getattr(insideLLMs, '__version__', '?'))"
python -m ruff --version >/dev/null
python -m pytest -q tests/inference/test_architecture.py --tb=no
make golden-path

echo "cloud-setup: ready"
echo "  activate: source .venv/bin/activate"
echo "  gates:    make check-fast"
echo "  demo:     make golden-path"
echo "  extras:   CLOUD_EXTRAS=providers bash scripts/cloud-setup.sh"
