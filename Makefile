.PHONY: help lint format format-check typecheck test test-fast check check-fast golden-path

help:
	@echo "insideLLMs developer commands"
	@echo ""
	@echo "  make lint          - ruff check ."
	@echo "  make format        - ruff format . + ruff check --fix ."
	@echo "  make format-check  - ruff format --check ."
	@echo "  make typecheck     - mypy insideLLMs"
	@echo "  make test          - pytest"
	@echo "  make test-fast     - pytest -m \"not slow and not integration\""
	@echo "  make check         - lint + format-check + typecheck + test"
	@echo "  make check-fast    - lint + format-check + test-fast (quick pre-commit)"
	@echo "  make golden-path   - offline harness + diff (DummyModel)"

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

format-check:
	ruff format --check .

typecheck:
	mypy insideLLMs

test:
	pytest

test-fast:
	pytest -m "not slow and not integration"

check: lint format-check typecheck test

check-fast: lint format-check test-fast

golden-path:
	python -m insideLLMs.cli harness ci/harness.yaml --run-dir .tmp/runs/baseline --overwrite --skip-report
	python -m insideLLMs.cli harness ci/harness.yaml --run-dir .tmp/runs/candidate --overwrite --skip-report
	python -m insideLLMs.cli diff .tmp/runs/baseline .tmp/runs/candidate --fail-on-changes

