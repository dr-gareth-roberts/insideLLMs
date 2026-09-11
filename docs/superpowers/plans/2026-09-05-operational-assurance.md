# Operational Assurance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Implement the five approved operational-assurance workstreams with explicit proof boundaries.

**Architecture:** Separate policy verification, invocation-wide budget admission,
provider metadata, partial-run finalization and extension packaging. Share only
typed contracts and reserve file ownership before parallel work.

**Tech Stack:** Python, Pydantic, pytest, Ruff, cosign adapter; TypeScript, Node/npm,
VS Code API and pinned VSIX packaging tooling.

**Spec:** `docs/superpowers/specs/2026-09-05-operational-assurance-design.md`

## Global Constraints

- Preserve the existing dirty working tree and its completed core fixes.
- Strict assurance and budget policies are opt-in; once requested, missing or unsupported evidence fails closed without a permissive fallback.
- Ordinary offline DummyModel workflows remain available and deterministic.
- No commits, tags, publication, real signing identities, or paid provider calls.
- Verify focused tests first, then integrated lint/type/docs/test gates. Do not substitute mock coverage for external-tool proof.

## Task 1: Cryptographic policy and publication gate

**Files:** `insideLLMs/policy/engine.py`, new `insideLLMs/policy/verification.py`,
`insideLLMs/signing/cosign.py`, `insideLLMs/runtime/_ultimate.py`,
`insideLLMs/attestations/steps/builders.py`, CLI verification/parser/dispatch,
new `tests/test_policy_verification.py`, affected policy/signing/ultimate tests,
new `docs/POLICY_ASSURANCE.md`.

**Interface:** Keep `run_policy(run_dir)` structural, with explicit assurance
metadata. Add `verify_policy(run_dir, policy)` for typed caller-owned strict
policy and a CLI projection. The exact typed policy lives in the new module.
No changes to configuration or high-level runner files owned by other tasks.

- [x] Add failing tests for unsigned/missing required stages, wrong identity or
  issuer, tampered envelope/manifest/records, unsupported receipt assurance,
  absent executable and timeouts. Assert no publication occurs after policy failure:
  `assert publisher.call_count == 0`.
- [x] Run `.venv/bin/python -m pytest -q tests/test_policy_verification.py` and record RED.
- [x] Implement strict verification and artifact binding using the existing
  detached-envelope signed-byte contract. Use `hashlib.sha256(records_bytes)`
  plus count in signed execution evidence; reject missing legacy commitments in
  strict mode, without rewriting old signatures. Validate cosign arguments
  against its official interface before relying on them.
- [x] Run focused policy/signing/ultimate/CLI tests; record GREEN. If no real
  verifier is available, add an explicit skipped external-tool integration gate
  and state that real verification remains unproven locally.
- [x] Document invocation, trust inputs, structural/cryptographic distinction,
  unsupported SCITT requirements and migration. Self-review then hand off.

## Task 2: Run-wide pre-dispatch budget admission

**Files:** new `insideLLMs/runtime/budget.py`, `insideLLMs/config_schema.py`,
`insideLLMs/runtime/_config_loader.py`, provider adapters and retry wrappers as
required, new `tests/test_budget_enforcement.py`, new `docs/RUN_BUDGETS.md`.
Coordinate `_high_level.py` edits with Task 4 owner before applying them.

**Interface:** `BudgetPolicy` validates explicit currency, finite nonnegative
allowance, exact pricing/request bounds and invocation scope. `BudgetLedger`
offers atomic `reserve`, idempotent `settle`, and conservative uncertain-outcome
accounting. `BudgetExceededError` is nonretryable and exposes an abort reason.
Provide shared model-factory/high-level integration without creating one ledger
per model. Reject budgeted resume and unsupported model/probe bypasses before calls.

- [x] Write tests whose fake SDK increments `dispatches`; assert
  `dispatches == 0` on absent/unknown bounds and exhausted allowance.
- [x] Run focused tests and record RED. Include thread/async contention,
  multiple models and judges, hidden retries, cancellation and duplicate settlement.
- [x] Implement a short `threading.Lock` critical section and Decimal accounting:
  `settled + uncertain + reserved + quote <= limit` is required for admission.
  Enforce output caps in the actual outgoing request; disable invisible SDK
  retries. Unsupported operations fail before dispatch, not after spending.
- [x] Add explicit configuration/factory integration and conservative supported
  adapter contracts. Do not advertise an adapter as strict-safe unless its
  input and all billable output bounds are defensible. Surface uncertainties.
- [x] Run focused config/model/middleware/budget tests and document exact initial
  support, policy assumptions and account-bill boundary. Hand off for review.

## Task 3: Provider catalogue and doctor

**Files:** new `insideLLMs/models/catalogue.py`, `insideLLMs/registry.py`,
`insideLLMs/cli/commands/doctor.py`, new `tests/test_provider_catalogue.py`,
new `docs/PROVIDER_CAPABILITIES.md`. Do not edit provider implementations or
budget files concurrently with Task 2.

**Interface:** immutable `ProviderSpec` and catalogue mapping of builtin registry
keys to lazy import, modules/distributions, credential alternatives, external
requirements and declared native/simulated capabilities. Registration and doctor
consume it. Budget support is separately unknown unless Task 2 implements it;
declared adapter support must not imply verified billing support.

- [x] Add failing registry parity and diagnostic tests: OpenRouter builtin,
  `CO_API_KEY` independently valid, missing Ollama/vLLM SDKs reported, unknown
  plugins not ready, and no constructor/network invocation during discovery.
- [x] Run `.venv/bin/python -m pytest -q tests/test_provider_catalogue.py` for RED.
- [x] Implement catalogue-driven registration and diagnostics with additive
  JSON fields. `live_verification = "not_checked"`; preserve existing runtime
  `can_*` methods and current doctor default exit policy.
- [x] Verify focused registry/doctor/model tests and a real offline doctor JSON
  invocation; ensure secrets are never included and documentation matches.
- [x] Self-review and hand off; separate metadata from constructor kwargs.

## Task 4: Fail-fast aggregate diagnostics

**Files:** `insideLLMs/exceptions.py`, `insideLLMs/runtime/_sync_runner.py`,
`insideLLMs/runtime/_async_runner.py`, `insideLLMs/runtime/_high_level.py`,
`insideLLMs/cli/commands/harness.py`, new `tests/test_fail_fast_diagnostics.py`.

**Interface:** aborted runner/harness exceptions carry a structured partial
result payload, preserving original cause. CLI catches the typed harness abort,
passes its partial payload through the existing artifact writer, and exits 1.
`run_completed=False`, health false, expected count retained; no fake successes.
Task 2's budget aborts feed the same mechanism.

- [x] Add first/middle/final-item and cross-cell failing tests. Expect completed
  records plus the failing record, original exception and no subsequent calls.
  `assert manifest["run_completed"] is False` and validate emitted schemas.
- [x] Run focused tests for RED.
- [x] Preserve slot-ordered partial runner results before raising; finalize
  harness aggregate data before exposing failure. Use existing result conversion
  and atomic output helpers, not a second independent serialization contract.
- [x] Test CLI/API, schema versions, deterministic repeats, async aborts and
  batch semantics; avoid claiming already-dispatched batch work was prevented.
- [x] Hand `_high_level.py` back to Task 2 and document proof boundaries.

## Task 5: Reproducible and safe editor packaging

**Files:** `extensions/vscode-insidellms/` manifest/lock/build/source/tests/docs,
new `.github/workflows/editor-extension.yml`, narrow `.gitignore` exception.

**Interface:** clean `npm ci`, `npm run build`, `npm test`, `npm run package`,
`npm run verify-package` entrypoints. Use VS Code argument-based execution and
workspace trust. Output a local `.vsix`, never publish.

- [x] Add failing package/launcher tests for entrypoint presence, archive
  allowlist, path metacharacters and invoking workspace selection.
- [x] Pin compatible tooling, install a lockfile, and produce a package using
  supported tooling; justify package-only dependencies in extension docs.
- [x] Build twice from clean copies with changed mtimes and assert whole-VSIX
  SHA-256 equality. Normalize archive ordering/timestamps if needed.
- [x] Add read-only CI; test isolated editor activation and real DummyModel
  harness artifacts when an editor host is available. Record runtime limitations
  explicitly when unavailable; mocks prove only launch contract behavior.
- [x] Document installation and verification commands and hand off.

## Integration and review

- [x] Review each task against its spec and diff; fix important findings.
- [x] Run full `make check-fast`, configured-target mypy, docs audit and focused
  integration tests. Run remaining safe tests marked integration separately.
- [x] Run offline golden path, strict-policy negatives, budget admission CLI
  checks, fail-fast CLI artifacts, doctor JSON and package verification.
- [x] Record actual commands/counts and unresolved external proof in
  `docs/RELEASE_READINESS.md`. Do not commit or publish this dirty worktree.

Completion note: checklist marks execution and review, not universal external
assurance. The broad suite had two legacy exception-contract assertions, fixed
and rerun in the harness suite; it was not rerun in full afterward. See release
readiness for exact counts, real-verifier skip and conditional budget scope.
