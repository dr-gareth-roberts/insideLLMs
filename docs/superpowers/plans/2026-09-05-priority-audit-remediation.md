# Priority Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six highest-priority core audit findings with behavioral proof, including their directly coupled publication/diagnostic defects, and independently block the satellite application's known-sanctions override.

**Architecture:** Use small, independently reviewable work packages: filesystem containment, CI retirement, runtime finalization, statistical correctness and measured compute assurance. Preserve public interfaces where safe; explicitly migrate interfaces that currently imply a guarantee they do not provide. Share the existing no-follow snapshot implementation and interruption contracts instead of adding parallel policy engines.

**Tech Stack:** Python >=3.10, stdlib, optional cryptography/ORAS, pytest, Ruff, mypy; Bash/GitHub composite actions; separate FastAPI/Pydantic satellite tests. No new core runtime dependencies planned.

**Spec:** [Repository audit](../../AUDIT_2026-09-05.md), findings A01–A06; coupled A09/A13; satellite A11. Preserve the completed contracts in [Operational assurance plan](2026-09-05-operational-assurance.md) and its [design](../specs/2026-09-05-operational-assurance-design.md).

## Global Constraints

- Preserve the existing dirty working tree and its completed core fixes.
- Strict assurance and budget policies are opt-in; once requested, missing or unsupported evidence fails closed without a permissive fallback.
- Ordinary offline DummyModel workflows remain available and deterministic.
- No commits, tags, publication, real signing identities, or paid provider calls.
- Verify focused tests first, then integrated lint/type/docs/test gates. Do not substitute mock coverage for external-tool proof.
- Keep the existing `requires-python = ">=3.10"` and Ruff `line-length = 100`; do not raise the package floor as incidental cleanup.
- This document is a plan, not implementation or proof that the defects are fixed.
- Existing generated/signed evidence is immutable historical data. Never silently rewrite it to satisfy new semantics.

## Scope, decisions and deliverables

Recommend executing the independent work packages as separate review units, not
one large patch. This master plan defines their shared acceptance gate.

| Work package | Tasks | Audit coverage | Deliverable |
| --- | --- | --- | --- |
| File safety | 1–2 | A01, A02 | Owned encryption staging; contained publication snapshot |
| CI gate | 3 | A03 | Legacy unsafe action fails closed with migration guidance |
| Finalization | 4–6 | A04, A09, A13 | Scoring before success; stable published payload; durable abort reporting |
| Statistics | 7 | A05 | Correct Student-t p-values and explicit invalid-input outcomes |
| Compute assurance | 8–9 | A06 | Truthful report semantics plus scoped dispatch evidence |
| Satellite veto | 10 | A11 | Hard block independent of live-model advice |
| Integration | 11 | All above | Fresh reproducible regression and release-readiness evidence |

Decisions locked for implementation:

1. **Retire, do not reimplement, the alternate action.** Keep its old path as an
   explicit failure with migration instructions. No current repository workflow
   uses it; maintaining a second shell execution/security path is unnecessary.
   This is intentionally breaking for external callers of the unsafe path.
2. **No-follow snapshots are common infrastructure, not strict-policy-only.**
   Raw publication must obey filesystem containment even without authenticity.
3. **Scoring is required before execution success.** Publishing is a subsequent
   operation with its own outcome; a failed remote write must not be described
   as a successful publication or retroactively mutate signed payload bytes.
4. **Published payload identity excludes its own publication receipt.** A09
   requires an explicit versioned identity contract, not merely moving one call.
5. **Declared equality is not verified equality.** Keep declarations visible but
   never infer verified token limits from profile values or user-supplied Spend.
6. **Statistics stays usable in a core-only install.** Implement a bounded,
   reference-tested Student-t tail helper; SciPy may be an optional test oracle,
   not a runtime fallback that changes results between installations.
7. **Satellite work is independently gated.** Its domain rules remain a demo;
   this task enforces the code's existing hard veto, not current legal screening.

Out of scope: A07/A08 resume repairs, A10 LaTeX escaping, A12 satellite concurrency,
extension dependency upgrades, account-wide billing enforcement, new provider
support and exhaustive dependency remediation. Keep them open in the audit.
Passing this plan does not make those findings disappear.

## Execution order and ownership

```text
File owner:       Task 1 ── Task 2 ───────────┐
Runtime owner:    Task 4 ── Task 5 ── Task 6 ├─ Task 11
CI owner:        Task 3 ─────────────────────┤
Analysis owner:  Task 7 ── Task 8 ── Task 9 ─┤
Satellite owner: Task 10 ───────────────────┘
```

Task 5 integrates Task 2's snapshot interface; settle that interface first.
Tasks 4/6 and 7 touch statistics consumers: the runtime owner must not edit
`analysis/statistics.py` concurrently with the analysis owner. Task 9 owns
inference changes; no imports of provider SDKs or models into that package.
With three worker slots, start file safety, runtime and analysis, then assign
CI/satellite when a slot becomes free. The coordinator owns integration/docs.

Before implementation, read `AGENTS.md`, root and affected-folder codemaps,
the audit and prior design. Run `git status --short` and `git diff --stat`.
Do not use a fresh HEAD-only worktree that drops the uncommitted fixes audited
here. Any isolation must explicitly include this working-tree baseline.
Record task-owned hunks; do not stage whole files containing unrelated changes.
Checkpoint with a reviewed diff and test log instead of committing.

## Task 1: Make encryption staging exclusively owned

**Files:** modify `insideLLMs/privacy/encryption.py`, `insideLLMs/privacy/codemap.md`;
extend `tests/test_encryption.py`; add `tests/test_encryption_staging.py`.
Inspect `insideLLMs/cli/commands/export.py` to preserve its outer atomic wrapper.

**Interfaces:** retain `encrypt_jsonl(path: Path | str, *, key: bytes | None = None) -> None`
and corresponding `decrypt_jsonl`. Add private
`_transform_jsonl(path: Path, transform: Callable[[bytes], bytes]) -> None` in
the same module; both public operations pass the Fernet transform to it.

- [ ] Add a failing synthetic symlink regression:

```python
from cryptography.fernet import Fernet
from insideLLMs.privacy.encryption import encrypt_jsonl

def test_encrypt_does_not_touch_preexisting_staging_link(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_bytes(b'{"value": 1}\n')
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"untouched")
    link = tmp_path / "records.jsonl.enc.tmp"
    link.symlink_to(sentinel)
    encrypt_jsonl(source, key=Fernet.generate_key())
    assert sentinel.read_bytes() == b"untouched"
    assert link.is_symlink()
    assert not source.is_symlink()
```

- [ ] Run `.venv/bin/python -m pytest -q tests/test_encryption_staging.py` and
  capture failure on sentinel/source assertions, not a missing import.
- [ ] Implement unique sibling staging via `tempfile.mkstemp`, immediately wrap
  the returned descriptor with `os.fdopen`, and use `os.fchmod` where supported.
  Keep temporary plaintext at mode 0600 throughout transformation; apply the
  preserved source mode only after transformation succeeds. Keep the existing
  blank-line/line-normalization semantics. The ownership primitive is:

```python
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
)
with os.fdopen(descriptor, "wb") as output:
    for line in source:
        if line.strip():
            output.write(transform(line.strip()) + b"\n")
    output.flush()
    os.fsync(output.fileno())
```

  `source` is the already-open binary input owned by `_transform_jsonl`.
  Use `os.replace` only after the stream closes successfully. Reject a symlink
  or nonregular source; on POSIX open with no-follow and verify `fstat`.
  Cleanup only the random pathname created by this invocation; preserve the
  original transform exception if cleanup also fails, logging a redacted
  secondary error. Never delete an old predictable staging file.
- [ ] Add decrypt-link, preexisting regular staging, invalid-token midway,
  source-link, mode-preservation and injected replace-failure tests. After
  failure assert original source bytes unchanged and no owned plaintext stage.
  Use barrier-controlled concurrent operations on distinct files to verify
  staging ownership. Explicitly document that simultaneous in-place transforms
  of the *same* source require caller serialization; unique staging is not a
  transactional concurrency guarantee against a malicious parent-directory owner.
- [ ] Run both encryption test files and export encryption tests, review diff,
  update codemap, and record a checkpoint. Crypto-enabled CI must run these;
  optional-dependency skips are not acceptance.

## Task 2: Unify contained publication snapshots

**Files:** create `insideLLMs/_artifact_snapshot.py` and
`tests/test_publication_containment.py`; modify `insideLLMs/policy/verification.py`,
`insideLLMs/publish/oras.py`, `tests/test_policy_verification.py`,
`docs/POLICY_ASSURANCE.md`, and affected codemaps.

**Interfaces:** move the existing directory-relative no-follow primitives into
the new internal module without weakening them. Add a context manager:

```python
@contextmanager
def regular_tree_snapshot(source: Path | str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="insidellms-snapshot-") as root:
        snapshot = Path(root) / "run"
        descriptor = _open_directory(source)
        try:
            _copy_regular_tree(descriptor, snapshot)
        finally:
            os.close(descriptor)
        yield snapshot
```

`_open_directory` and `_copy_regular_tree` are moved from verification.py into
this module, with their current signatures. Strict verification's individual
artifact reader uses the same primitives. Public push signatures stay unchanged.

- [ ] Turn the A02 synthetic-sentinel reproduction into a negative regression.
  Fake `oras_client.OciClient`, set `ORAS_AVAILABLE=True`, and assert the public
  push raises before the fake client's `push` is called for external-file links,
  directory links and FIFOs. All fixtures remain inside `tmp_path`.

```python
with pytest.raises((ValueError, OSError)):
    push_run_oci(run_dir, "example.invalid/test:local")
assert fake_client.push.call_count == 0
assert sentinel.read_bytes() == b"publication-sentinel"
```

- [ ] Run `.venv/bin/python -m pytest -q tests/test_publication_containment.py`
  and record RED for the existing external link case.
- [ ] Put ordinary push inside `regular_tree_snapshot` and enumerate sorted
  regular files from the private copy. Keep the copy alive until push returns.
  `publish_verified_run` verifies its private snapshot and passes that snapshot
  to public push; the second defensive copy is acceptable initially because
  the first copy remains private and unchanged. Do not expose an unchecked
  public push helper just to avoid copying twice.
- [ ] Reject all source symlinks, including the optional `results.jsonl` alias,
  with a clear message directing callers to remove/materialize aliases in a
  separate export copy. Do not alter the signed original. Fail closed on
  platforms without required no-follow operations; document platform support.
  Copy in bounded chunks rather than reading arbitrary files entirely into RAM.
  Compare file identity/size/mtime before/after copying and reject detected
  mutation; do not claim this is a transactional snapshot of hostile concurrent
  writes. Snapshot directory permissions must be private.
- [ ] Add swap-at-open tests against the OS boundary, verify the fake publisher
  reads original snapshot bytes after the source changes, and assert teardown
  after success/failure. Retain all strict signer/root/record-binding tests.
- [ ] Run containment and policy tests, review exact filenames/bytes handed to
  fake ORAS, update docs, checkpoint. No real upload is needed for containment
  acceptance; actual SDK layer encoding remains an independent integration gate.

## Task 3: Retire the unsafe alternate CI action

**Files:** modify `.github/actions/diff-gate/action.yml` and
`.github/actions/diff-gate/diff-gate.sh`; create
`tests/test_legacy_action_retirement.py`; extend
`tests/test_github_action_manifest.py`; update `wiki/tutorials/CI-Integration.md`.
Inspect maintained `action.yml`, `scripts/github_action_run.sh` and split workflows.

**Interfaces:** old inputs remain parseable but do not execute. Both the old
action and directly invoked old script exit nonzero with migration instructions.
No old output claims a successful diff. Maintained action behavior is unchanged.

- [ ] Add a subprocess regression invoking the legacy script with all provider
  and GitHub credentials absent, stub commands on a temporary PATH, and assert
  exit 1, a deprecation message, and zero Python/Git/gh invocations.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_legacy_action_retirement.py`
  and capture current failure.
- [ ] Replace the old composite steps with a single quoted-path invocation of
  its retirement script. Remove setup/install/comment steps and token exports.
  The script becomes:

```bash
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' '::error::This legacy action is retired. Use the repository-root action and the split comment workflow documented in wiki/tutorials/CI-Integration.md.' >&2
exit 1
```

- [ ] Add maintained-action coverage for omitted baseline resolving the PR base
  SHA, explicit baseline override and unknown ref failure. Use temporary local
  Git history and stub only installation/model boundaries. Assert recorded
  resolved revisions and fixture artifact differences, not merely checkout calls.
  A missing baseline must fail, never silently use the candidate. Explicitly
  requested identical refs may be allowed with a warning; PR defaults must use
  the actual event base even when a no-op PR happens to share the same commit.
- [ ] Document the migration and changed availability before release. Run
  `tests/test_legacy_action_retirement.py`, `tests/test_github_action_manifest.py`
  and `tests/test_github_action_comment.py`; checkpoint. Do not run a privileged
  workflow or alter repository settings as part of this task.

## Task 4: Make aggregate scoring part of finalization

**Files:** modify `insideLLMs/runtime/_sync_runner.py`, `_async_runner.py`,
`_high_level.py`, `_interruptions.py` and `_run_health.py` only as needed;
extend `tests/test_fail_fast_diagnostics.py` and
`tests/test_runtime_scoring_contract.py`; add `tests/test_aggregate_finalization.py`.
Reuse `insideLLMs/exceptions.py::RunnerExecutionError`.

**Interfaces:** `run()` return types stay unchanged. On first aggregate failure,
raise `RunnerExecutionError` with `original_error`, `partial_results`,
`partial_result`, and `score=None` on its partial experiment. Add a private keyword
`_skip_scoring: bool = False` to `create_experiment_result` for error reconstruction
only; it must not skip normal scoring or retry a failing scorer. Existing
`_abort_error` keeps original failures primary and captures secondary errors.

- [ ] Add a `LogicProbe`-based test whose `score` raises `ValueError("aggregate failed")`.
  Run one successful DummyModel item with artifacts. The required assertions are:

```python
with pytest.raises(RunnerExecutionError) as caught:
    runner.run(["hello"], run_dir=run_dir)
assert isinstance(caught.value.original_error, ValueError)
assert len(caught.value.partial_results) == 1
assert caught.value.partial_result["run_completed"] is False
manifest = json.loads((run_dir / "manifest.json").read_text())
assert manifest["run_completed"] is False
assert manifest["custom"]["health"]["healthy"] is False
publisher.assert_not_called()
```

  Define `runner = ProbeRunner(DummyModel(), failing_probe)` and mock only the
  ultimate boundary for `publisher`. Repeat as an async test with `await` and
  `AsyncProbeRunner`. Port the audit fixture, not its expected-defect assertions.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_aggregate_finalization.py`
  and capture the plain-exception/healthy-manifest failure.
- [ ] Move output validation and experiment aggregation before final success
  manifest construction and ultimate dispatch in both runners. Catch ordinary
  `Exception` from aggregation, not `BaseException`; create the existing typed
  error, reconstruct once with `_skip_scoring=True`, then finalize through the
  existing incomplete-artifact path. Example wrapping contract:

```python
error = RunnerExecutionError(
    "Aggregate scoring failed",
    run_id=resolved_run_id,
    original_error=exc,
)
```

  Do not change each successful record to an error: the calls succeeded while
  the aggregate failed. Do not invoke the scorer a second time. Preserve older
  schemas through `custom.health` when they lack top-level `run_completed`.
- [ ] Add harness-first/second/final-cell tests, artifact-disabled API tests,
  validation failure before success, first-vs-secondary scorer errors and failed
  diagnostic writes. The original cause remains primary even if summary writing
  fails. Retain in-memory partial results and report secondary failure explicitly.
  Successful repeat runs must remain byte-deterministic; score called exactly once.
- [ ] Run the three named test files plus run-health/schema suites. Verify CLI
  failure exit and on-disk summary, update runtime codemap, checkpoint. Hand off
  the finalized ordering to Task 5; do not publish inside this task.

## Task 5: Finalize the payload before publication

**Files:** modify `insideLLMs/runtime/_ultimate.py`,
`insideLLMs/crypto/canonical.py`, `insideLLMs/attestations/steps/builders.py`,
`tests/test_ultimate_integration.py`, `docs/POLICY_ASSURANCE.md`;
create `tests/test_publication_ordering.py` and `docs/PUBLICATION_CONTRACT.md`.
Inspect `insideLLMs/cli/commands/attest.py` and bundle-ID consumers before changing
identity fields. Task 2 owns publisher/snapshot code; coordinate changes there.

**Interfaces:** preserve `run_ultimate_post_artifact` call signature. Introduce a
versioned payload identity descriptor `integrity/bundle_identity.json` with
`version=2`, ordered relative payload paths and their SHA-256 digests. Exclude
the descriptor itself, `integrity/bundle_id.txt`, and postpublication stage 09
receipts to avoid self-reference. Hash the canonical descriptor for v2 bundle ID.
Keep legacy `run_bundle_id` behavior available for reading existing artifacts;
do not reinterpret old IDs as v2.

- [ ] Port A09: fake push captures every relative filename and byte at call time.
  Assert 08.policy, v2 descriptor and bundle_id already exist; their bytes match
  the finalized local payload. Run `tests/test_publication_ordering.py` for RED.
- [ ] Implement the lifecycle, preserving Task 4's prerequisite:

```text
score and validate → finalize execution manifest/summary
→ attestations 00–07 → policy verdict → attestation 08
→ fresh private publication tree → payload descriptor → bundle ID
→ contained push → separate attestation 09 publication receipt
```

  Build the publication tree from explicit current-run artifacts; do not blindly
  include an old 09 receipt, old bundle-ID metadata, temporary files or arbitrary
  leftover generated attestations. Define the path set in the new contract and
  test it. Reject conflicting/nonregular inputs rather than deleting originals.
  Stage 09 may remain at its established local path but is explicitly outside
  the v2 payload; record target, nullable returned digest, payload ID and outcome.
  No receipt falsely says published when ORAS returns no digest or throws.
- [ ] On push failure, preserve the immutable execution payload and report a
  failed publication outcome to the caller; on timeout use unknown outcome.
  Do not auto-retry an uncertain remote write. Keep execution health distinct
  from publication status, and never rewrite already-signed manifest bytes to
  attach the publication result.
- [ ] Add tests for policy rejection (zero push), scorer rejection (zero push),
  fresh/reused directories, old receipt exclusion, optional no-publish mode,
  unchanged v1 reader behavior, deterministic v2 IDs and failure/unknown receipt
  outcomes. Assert exact bytes at the mock push boundary, not only final presence.
- [ ] Run ultimate, containment and strict-policy suites together. Document
  v1/v2 migration and exclusions; keep structural/authenticity labels unchanged.
  Checkpoint only when a reviewer can reconstruct the payload ID independently
  from the descriptor and validate every listed file digest.

## Task 6: Preserve abort context when rebuilding reports

**Files:** modify `insideLLMs/cli/commands/report.py`,
`insideLLMs/cli/_report_builder.py` and the HTML report path in
`insideLLMs/analysis/visualization.py` only to show supplied health metadata;
create `tests/test_report_abort_preservation.py`; update CLI report documentation.

**Interfaces:** report reconstruction consumes records plus manifest health and
existing summary abort/secondary diagnostics. Add keyword-only
`run_health: Mapping[str, object] | None = None` to HTML builders as needed.
Never infer expected record count solely from observed records. Existing report
exit 0 means successful generation, not successful execution; show a warning and
visible incomplete banner. Malformed authoritative metadata fails without
overwriting the old report/summary.

- [ ] Port the actual CLI A13 reproduction (successful first cell, unknown second
  model). Preserve its before/after assertions as a regression:

```python
assert report_exit == 0
assert after["summary"]["run_completed"] is False
assert after["summary"]["abort"] == before["summary"]["abort"]
assert after["summary"]["health"]["expected_count"] == 2
assert manifest_after == manifest_before
assert "incomplete" in report_html.lower()
```

- [ ] Run `.venv/bin/python -m pytest -q tests/test_report_abort_preservation.py`
  and record missing-field failure.
- [ ] Merge diagnostic metadata after recomputing metrics. Manifest completion
  wins over a conflicting success assertion in an old summary; preserve conflict
  warnings. Without a manifest, preserve existing diagnostic context; if neither
  supplies completion evidence, label it unknown, not healthy. Write summary and
  HTML through owned temporary files, validating both before replacing outputs.
- [ ] Add completed-run, secondary-scorer-error, conflicting-metadata,
  missing-manifest and malformed-summary tests. Run report and CLI suites,
  inspect one incomplete HTML output, document semantics and checkpoint.

## Task 7: Correct Student-t probabilities and input boundaries

**Files:** create `insideLLMs/analysis/_student_t.py` and
`tests/test_student_t_reference.py`; modify `insideLLMs/analysis/statistics.py`,
`tests/test_statistics.py`, `tests/test_statistics_branch_coverage.py` and
`insideLLMs/analysis/codemap.md`. Document corrected p-values in release notes.

**Interfaces:** preserve public t-test signatures and result dataclass. Add private
`two_sided_t_probability(statistic: float, degrees_of_freedom: float) -> float`.
For n<2 return the existing non-significant insufficient-data result shape for
both tests. Invalid alpha, NaN/infinite samples or unequal paired lengths raise
ValueError. Preserve explicit zero-variance behavior only for valid n>=2 and
document it as a degenerate case, not a regular distribution estimate.

- [ ] Add independent analytical references:

```python
import math
import pytest
from insideLLMs.analysis.statistics import paired_t_test, welchs_t_test

def test_paired_small_sample_uses_student_t_tail():
    result = paired_t_test([1, 2, 3], [0, 0, 0])
    assert result.p_value == pytest.approx(1 - math.sqrt(6 / 7), abs=1e-12)
    assert result.significant is False

@pytest.mark.parametrize("test", [paired_t_test, welchs_t_test])
def test_single_observation_is_insufficient(test):
    result = test([1], [0])
    assert result.significant is False
    assert "insufficient" in result.conclusion.lower()
```

- [ ] Run `.venv/bin/python -m pytest -q tests/test_student_t_reference.py`
  and record incorrect probability/ZeroDivisionError.
- [ ] Implement the Student-t two-sided tail using the regularized incomplete
  beta expression `I_x(df/2, 1/2)`, `x=df/(df+t*t)`. Use log-gamma normalization,
  `log1p(-x)`, symmetry and a bounded continued fraction, returning endpoints
  explicitly and raising ArithmeticError on nonconvergence. Avoid `1-CDF`
  subtraction in small tails; rescale before squaring extreme finite t.
  The helper's integration into each public test is:

```python
p_value = two_sided_t_probability(t_stat, df)       # Welch's computed df
p_value = two_sided_t_probability(t_stat, n - 1)    # paired observations
```

  Implement recurrence from the [NIST DLMF incomplete-beta definitions and
  continued fraction](https://dlmf.nist.gov/8.17), with tolerance 1e-14 and a
  maximum 10,000 iterations. Guard near-zero denominators with signed 1e-300;
  only clamp final roundoff within 1e-14 of [0,1], otherwise raise.
  The [NIST Student-t reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm)
  supplies distribution semantics; implementation constants require numerical
  validation, not trust in the formula alone.
- [ ] Add df=1 Cauchy and df=2 closed-form tests, symmetry, monotonicity,
  t=0, extreme tails, noninteger Welch df and near-alpha decisions. Validate a
  fixed grid df={1,2,5,30,1000}, |t|={0,.1,1,2,5,20} against independently
  generated reference fixtures (record oracle/version); tolerance 1e-12 absolute
  and 1e-8 relative for non-underflow tails. Optional SciPy comparison may
  supplement, not replace, always-running analytical/reference tests.
- [ ] Run all three statistics files plus report/scoring consumers. Do not
  regenerate expected values from the new helper. Review numerical behavior
  separately, state that historical significance decisions need recomputation,
  and checkpoint. Broader confidence-interval approximations remain outside A05.

## Task 8: Stop overstating matched-compute assurance

**Files:** modify `insideLLMs/analysis/matched_compute.py`,
`tests/test_matched_compute.py`, `examples/matched_compute_evaluation.py`;
create `tests/test_matched_compute_limits.py` and `docs/MATCHED_COMPUTE_ASSURANCE.md`.

**Interfaces:** add report field `declared_limits_match: bool` and
`output_limit_assurance: Literal["verified", "unknown"]`. Keep legacy
`matching.max_output_tokens` as a boolean but make it true only for verified
matching request caps; `output_tokens_matched` adopts the same stricter meaning.
Under unknown assurance it is false, with an explicit limitation. Numeric
observed usage remains separately visible and is not required to be equal.

- [ ] Port A06's actual InferenceClient fake-model fixture into
  `tests/test_matched_compute_limits.py`: two clients share a model, kwargs
  caps 10/1000, profiles 10/10, actual usage 10/1000. Initially require rejection
  or false verified matching; assert it cannot return affirmative assurance.
- [ ] Run that file and capture current `matching.max_output_tokens=True` failure.
- [ ] Add an immediate aggregate consistency guard using each side's observed
  calls and output tokens, before emitting a report:

```python
if observed.output_tokens > observed.calls * profile.max_output_tokens_per_call:
    raise ComputeMismatchError("observed output tokens exceed declared call caps")
```

  This catches an obvious overflow but is not per-call enforcement. Reject
  negative/nonfinite counters and outputs with zero calls. Missing usage remains
  unknown, not a trustworthy zero. Default to unknown until Task 9 supplies
  complete dispatch evidence. Direct/custom runners cannot self-certify merely
  by returning a favorable Spend or provenance dictionary.
- [ ] Add serialization tests showing equal declarations with unknown assurance,
  observed overflow failure, genuine zero vs missing usage and unequal actual
  usage within identical limits. Update exact example JSON expectations and
  docs; warn consumers that older true values meant declarations only.
- [ ] Run matched-compute tests and example module; checkpoint as an independently
  shippable fail-honest fix. Do not claim A06 pre-call enforcement complete yet.

## Task 9: Bind request limits and collect scoped dispatch evidence

**Files:** create `insideLLMs/inference/_limits.py` and
`tests/inference/test_matched_limits.py`; modify
`insideLLMs/inference/adapters.py`, `client.py`,
`insideLLMs/analysis/matched_compute.py`, associated codemaps and Task 8 docs.

**Interfaces:** define in `_limits.py`:

```python
@dataclass(frozen=True)
class OutputLimitBinding:
    parameter: str
    maximum: int

@dataclass(frozen=True)
class DispatchEvidence:
    output_cap: int
    output_tokens: int | None
    completed: bool
```

Add optional `output_limit: OutputLimitBinding | None = None` to InferenceClient
and its proposer. This is an explicit adapter/request binding, not automatic
provider discovery. Initially accept only the explicit flat `max_tokens` and
`max_completion_tokens` keyword forms; exactly one binding may be active, reject
conflicting aliases. Other provider shapes are unsupported for verified matching.
Keep this module independent of `models`, SDKs and `runtime` imports.

- [ ] Add tests for cap mismatch before dispatch (fake model call count zero),
  mutated generation kwargs, missing binding, missing usage, usage exceeding a
  sent cap, and model-backed judge/verifier calls. Regression assertion:

```python
assert fake_model.calls == 0   # reject 1000 vs declared 10 before generation
```

- [ ] Run `.venv/bin/python -m pytest -q tests/inference/test_matched_limits.py`
  and record RED for missing enforcement/evidence, then add the typed contracts.
- [ ] Implement a context-local per-variant collector using ContextVar, reset
  with its token in `finally`. The collector records pre-dispatch reservations
  and terminal outcomes; use a short lock for worker callbacks. It is shared by
  concurrent child calls in that variant, never across separate evaluations.
  `ModelProposer._generate` copies kwargs for every call, validates that its bound
  cap equals the profile cap and that total admitted calls remain within the
  declared ceiling, then records evidence before invoking the selected method.
  Never silently clamp one variant's configured cap to make comparisons pass.
- [ ] Record completion tokens only from the real response boundary, never
  request metadata. Unknown/malformed usage yields unknown assurance. Timed-out
  sync work remains an unsettled dispatch, not a refunded call; cancellation must
  not let its eventual completion contaminate another variant. At report time
  require observed call count and scoped evidence count to agree, every dispatch
  to have a matching cap and known within-cap output, and no unfinished call.
  Otherwise unknown, or ComputeMismatchError on a definite violation.
- [ ] Make `one_shot_baseline` validate the binding before it executes;
  `single_result_variant` and manual variants use the enclosing scope. Unobserved
  judge/tool/foreign-client calls cannot gain verified assurance. Supply an
  explicit trusted-code contract: arbitrary Python callbacks can bypass client
  instrumentation; the collector is not a sandbox or proof against malicious
  callback code. Strict callers must restrict execution to instrumented clients.
- [ ] Add concurrent-variant isolation and sync/async exception tests, multiple
  clients sharing one model, two individually excessive calls hidden by a small
  total, and metadata-spoof tests. Update example to explicitly bind its offline
  model. Run all `tests/inference/`, matched-compute tests and architecture fences.
  Checkpoint only with zero-dispatch proof for pre-call mismatch and truthful
  assurance output for unsupported paths. No USD-ledger rewrite or billing claim.

## Task 10: Enforce satellite hard veto before live advice

**Files:** modify `compliance_intelligence/app/agents/decision.py`;
create `compliance_intelligence/tests/test_decision_policy.py`;
update `compliance_intelligence/README.md` and agent codemap.

**Interfaces:** retain `run_decision(state: PipelineState) -> PipelineState`.
Extract `_hard_block_decision(state: PipelineState) -> ComplianceDecision | None`
from the existing sanctions/embargo branch; it must not require a risk score.
Known hard blocks precede both missing-risk handling and live/simulation routing.

- [ ] Parameterize sanctions-only, embargo-only and both, with simulation on/off
  and risk present/absent. Patch only `_llm_decision` to return APPROVE. Assert
  `state.decision.verdict is DecisionVerdict.BLOCK` and zero live-boundary calls.
  Add an ordinary clear-state control that still reaches live decision logic.
- [ ] Run satellite tests in its installed dependency environment, record RED.
  If dependencies are unavailable, record blocked full-integration proof; the
  earlier audit's dependency stubs do not satisfy this final gate.
- [ ] Reuse the existing hard-block rationale/escalation fields, and route through
  the shared logging/alert code rather than returning before alerts are created:

```python
decision = _hard_block_decision(state)
if decision is None:
    if state.risk_score is None:
        decision = ComplianceDecision(
            verdict=DecisionVerdict.REQUEST_MORE_INFO,
            confidence=0.0,
            rationale="Risk score unavailable — cannot render decision.",
            needs_reanalysis=True,
            reanalysis_reason="Missing risk score",
        )
    elif settings.simulation_mode:
        decision = _rule_based_decision(state, state.risk_score)
    else:
        decision = _llm_decision(state, state.risk_score)
state.decision = decision
```

- [ ] Test critical alert, state status, no reanalysis overriding a block and
  malformed/missing-risk controls. Document that this enforces existing demo
  flags, not real sanctions-list accuracy. Review and checkpoint separately;
  unresolved A12 still blocks concurrent live deployment readiness.

## Task 11: Integrated acceptance and release handoff

**Files:** update `docs/RELEASE_READINESS.md`; append a dated resolution section
to `docs/AUDIT_2026-09-05.md` without deleting original findings. Tests and docs
from Tasks 1–10 are deliverables; local scratch scripts are not permanent gates.

- [ ] Review every task's scoped diff, public compatibility changes and fresh
  RED/GREEN logs. No checkbox is complete solely because a worker reported it.
- [ ] Run all focused test files named above together. Required adverse-path
  assertions: no sentinel modification, zero publication on rejected inputs,
  no legacy action execution, no healthy aggregate failure, no misleading
  matched-limit true value, correct reference p-value, no live veto override.
- [ ] Run the repository gates with the selected environment explicit:

```sh
PATH="$PWD/.venv/bin:$PATH" make lint format-check docs-audit
.venv/bin/mypy insideLLMs
MPLCONFIGDIR="$PWD/.tmp/test-cache/matplotlib" XDG_CACHE_HOME="$PWD/.tmp/test-cache/xdg" .venv/bin/python -m pytest -q -o addopts='--tb=short --strict-markers'
PATH="$PWD/.venv/bin:$PATH" make golden-path
git diff --check
```

  Audit baseline was 7,775 passed/73 skipped, not the required new count. Record
  actual counts and skip reasons. Default mypy previously failed on NumPy stub
  syntax; do not label it passed via another interpreter. Run the supported
  isolated-core typecheck too and report both, or resolve the environment in a
  separately reviewed tooling change. No uncontrolled dependency upgrades.
- [ ] Run crypto-enabled tests on Linux and macOS; for unavailable no-follow
  platforms assert explicit rejection. Use a clean core-only install to prove
  statistics does not accidentally require SciPy. Local wheel/sdist builds and
  installs may be used for this gate, without release/upload. Hosted checks
  require an authorized existing workflow execution, not a new privileged push.
- [ ] Re-run strict-policy tamper/identity negatives and runtime budget contention
  tests as nonregression gates. Keep real cosign identity, remote OCI digest and
  provider invoice tests separate and unproven unless independently authorized
  and executed. No live services are required for the offline fixes to be reviewed.
- [ ] Record each audit ID as fixed with exact regression/evidence or still open.
  Include migration notes for legacy action retirement, bundle identity v2,
  historical p-values and matched-compute assurance fields. Leave A07/A08/A10/A12
  and dependency findings open. No blanket “release-ready” claim while applicable
  blockers remain.

## Definition of done and handoff

The priority work is complete when all included findings have permanent tests
that fail on the audited baseline and pass on the corrected working tree; the
integrated gates have fresh outcomes; required unsupported-platform paths fail
closed; and compatibility/proof limitations are documented. Mocks establish
dispatch, filesystem and ordering contracts only, not external service behavior.

Recommended execution: subagent-driven work with file ownership above, one
spec-compliance review and one code-quality/security review per package, then
coordinator integration. Inline execution with checkpoints is also viable.
This plan does not authorize commits, uploads, signing or paid calls.
