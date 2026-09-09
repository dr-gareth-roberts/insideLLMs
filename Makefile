.PHONY: help lint format format-check typecheck typecheck-strict typecheck-report typecheck-module typecheck-coverage test test-fast test-determinism test-contract test-adapter test-performance docs-audit architecture architecture-update check check-fast golden-path package-smoke clean-install-golden-path

PYTHON ?= python3

help:
	@echo "insideLLMs developer commands"
	@echo ""
	@echo "  make lint          - ruff check ."
	@echo "  make format        - ruff format . + ruff check --fix ."
	@echo "  make format-check  - ruff format --check ."
	@echo "  make typecheck     - mypy insideLLMs"
	@echo "  make test          - \$$PYTHON -m pytest"
	@echo "  make test-fast     - \$$PYTHON -m pytest -m \"not slow and not integration\""
	@echo "  make test-determinism - \$$PYTHON -m pytest -m determinism"
	@echo "  make test-contract - \$$PYTHON -m pytest -m contract"
	@echo "  make test-adapter  - \$$PYTHON -m pytest -m adapter"
	@echo "  make test-performance - \$$PYTHON -m pytest -m performance"
	@echo "  make docs-audit    - markdown/docs coverage + wiki link checks"
	@echo "  make architecture  - check API/import evidence and architecture boundaries"
	@echo "  make architecture-update - regenerate API/import evidence"
	@echo "  make check         - lint + format-check + typecheck + test"
	@echo "  make check-fast    - lint + format-check + test-fast (quick pre-commit)"
	@echo "  make golden-path   - offline harness + diff (DummyModel)"
	@echo "  make package-smoke PYTHON=/path/to/clean-venv/bin/python - verify installed distribution"
	@echo "  make clean-install-golden-path - build/install core wheel in an isolated venv"

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

# Strict type checking on the security-critical modules. This is the gating
# strict check and is mirrored exactly by CI (which runs `make typecheck-strict`).
typecheck-strict:
	mypy --strict --follow-imports=silent insideLLMs/injection.py
	mypy --strict --follow-imports=silent insideLLMs/safety.py

# Aspirational: full untyped-def strictness on the runtime package. Not yet
# clean (tracked); run manually, not part of the gating typecheck-strict.
typecheck-strict-runtime:
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
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow and not integration"

test-determinism:
	$(PYTHON) -m pytest -m determinism

test-contract:
	$(PYTHON) -m pytest -m contract

test-adapter:
	$(PYTHON) -m pytest -m adapter

test-performance:
	$(PYTHON) -m pytest -m performance

docs-audit:
	$(PYTHON) scripts/audit_docs.py
	$(PYTHON) scripts/check_wiki_links.py

architecture:
	$(PYTHON) scripts/architecture_evidence.py --check
	$(PYTHON) -m pytest tests/architecture tests/inference/test_architecture.py

architecture-update:
	$(PYTHON) scripts/architecture_evidence.py --write

check: lint format-check typecheck test docs-audit architecture

check-fast: lint format-check test-fast

# Install the built wheel/sdist in a clean virtualenv first; this rejects an
# editable/source import and runs every init template without provider access.
package-smoke:
	$(PYTHON) scripts/smoke_installed_package.py

golden-path:
	$(PYTHON) -m insideLLMs.cli harness ci/harness.yaml --run-dir .tmp/runs/baseline --overwrite --skip-report
	$(PYTHON) -m insideLLMs.cli harness ci/harness.yaml --run-dir .tmp/runs/candidate --overwrite --skip-report
	$(PYTHON) -m insideLLMs.cli diff .tmp/runs/baseline .tmp/runs/candidate --fail-on-changes

clean-install-golden-path:
	PYTHON=$(PYTHON) bash scripts/clean_install_golden_path.sh
