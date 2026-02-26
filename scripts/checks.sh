#!/usr/bin/env bash
# Full quality gates: lint, format, typecheck, test.
# For quick iteration, use: make check-fast
set -euo pipefail

ruff check .
ruff format --check .
mypy insideLLMs
pytest

