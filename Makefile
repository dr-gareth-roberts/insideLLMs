.PHONY: help lint format format-check typecheck typecheck-strict typecheck-report typecheck-module typecheck-coverage test test-fast test-determinism test-contract test-adapter test-performance docs-audit check check-fast golden-path

help:
	@echo "insideLLMs developer commands"
	@echo ""
	@echo "  make lint          - ruff check ."
	@echo "  make format        - ruff format . + ruff check --fix ."
	@echo "  make format-check  - ruff format --check ."
	@echo "  make typecheck     - mypy insideLLMs"
	@echo "  make test          - python -m pytest"
	@echo "  make test-fast     - python -m pytest -m \"not slow and not integration\""
	@echo "  make test-determinism - python -m pytest -m determinism"
	@echo "  make test-contract - python -m pytest -m contract"
	@echo "  make test-adapter  - python -m pytest -m adapter"
	@echo "  make test-performance - python -m pytest -m performance"
	@echo "  make docs-audit    - markdown/docs coverage + wiki link checks"
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

# Standard type checking (matches CI)
typecheck:
	mypy insideLLMs

# Strict type checking for new code
typecheck-strict:
	mypy --strict insideLLMs/injection.py
	mypy --strict insideLLMs/safety.py
	mypy --disallow-untyped-defs insideLLMs/runtime/

# Generate type checking report
typecheck-report:
	mypy insideLLMs --html-report ./mypy-report
	@echo "Report generated in ./mypy-report/index.html"

# Check specific module
typecheck-module:
	@test -n "$(MODULE)" || (echo "Usage: make typecheck-module MODULE=insideLLMs.injection" && exit 1)
	mypy --strict $(MODULE)

# Type coverage report
typecheck-coverage:
	mypy insideLLMs --any-exprs-report ./mypy-coverage
	@echo "Coverage report in ./mypy-coverage/"

test:
	python -m pytest

test-fast:
	python -m pytest -m "not slow and not integration"

test-determinism:
	python -m pytest -m determinism

test-contract:
	python -m pytest -m contract

test-adapter:
	python -m pytest -m adapter

test-performance:
	python -m pytest -m performance

docs-audit:
	python scripts/audit_docs.py
	python scripts/check_wiki_links.py

check: lint format-check typecheck test

check-fast: lint format-check test-fast

golden-path:
	python -m insideLLMs.cli harness ci/harness.yaml --run-dir .tmp/runs/baseline --overwrite --skip-report
	python -m insideLLMs.cli harness ci/harness.yaml --run-dir .tmp/runs/candidate --overwrite --skip-report
	python -m insideLLMs.cli diff .tmp/runs/baseline .tmp/runs/candidate --fail-on-changes
