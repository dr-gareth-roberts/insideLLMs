# Behavioural gate hardening

This document tracks the changes addressing the September 2026 release-readiness
audit. A passing test suite is evidence about the checked-out code; it does not
mean a package, GitHub Action tag, or release has been published.

## Behavioural scores and execution health

A successful model call and a correct answer are different results. A dataset
row opts into scored-probe evaluation with `reference_answer` (`reference` is a
compatibility alias). For example:

```json
{"question": "What is the capital of France?", "reference_answer": "Paris"}
```

The reference is kept out of the model prompt. The runner evaluates the response
and persists numeric `scores`, the `primary_metric`, and evaluation metadata in
`records.jsonl`. Sync, async, batch, and harness execution use the same scoring
contract. A missing reference means the result is unscored; successful execution
alone does not establish accuracy. An explicit null reference allows probes
whose evaluation does not require a reference answer.

Run health is checked separately. Empty runs, item errors, timeouts, skipped
items, incomplete execution, and mismatched expected counts fail the CLI/CI
health gate. Handled fail-fast errors and thrown batch exceptions now retain
completed outcomes, an incomplete manifest, and an atomic partial summary.
The harness preserves previous model/probe cells; failed batch calls without
returned outcomes do not fabricate item records. The CLI exits with failure;
Python callers receive `RunnerExecutionError` with the original exception and
partial payload. Process termination, disk failures, and arbitrary user-code
failures outside these execution paths are not crash-recovery guarantees.
API compatibility: harness model/probe initialization failures now use that
typed wrapper too, including failures before the first record. Inspect
`original_error` (also `__cause__`) for the previous registry/provider exception;
the partial payload explicitly reports zero observed records when none ran.
Earlier file/config/schema/dataset validation and budget-preflight errors still
propagate their existing exception types; this change is limited to model/probe
initialization and the handled execution-abort paths.
`run_completed` records
whether execution finished; `custom.health` records whether the finished run was
healthy. A healthy run may still contain incorrect answers and fail a score diff.

Use `diff --fail-on-regressions` when improvements are allowed. Use
`diff --fail-on-any-difference` when every reported change must block the gate,
including improvements and trace/trajectory changes. The older
`--fail-on-changes` retains its narrower compatibility behaviour. Duplicate
record identities are invalid input and fail comparison.

## Credential handling

Configuration snapshots, model metadata, execution/scoring attestations, and
tracker parameters pass through `insideLLMs._secrets.redact_config_secrets`
before persistence. It recursively replaces recognized credential fields,
credential-bearing HTTP headers, URL user information and secret query fields,
and secret-value wrappers with `[REDACTED]`.

Live configuration is separate from saved provenance, so providers still receive
their credentials. An environment variable name such as `api_key_env:
OPENAI_API_KEY` can be retained without reading or saving the variable's value.
Changing only a literal credential does not change the configuration-derived run
identity. This intentionally changes IDs for old configurations containing keys.

This is configuration redaction, not general data-loss prevention. Do not put
credentials in prompts, model responses, arbitrary free text, or input datasets.
Existing artefacts are not rewritten; previously persisted credentials require
separate review and rotation.

## CI permissions

Evaluation executes with `contents: read` and checkout credential persistence
disabled. The composite action no longer posts comments in the job that executes
candidate code. Repository PR comments are handled by a separate trusted
`workflow_run` workflow. It does not check out or execute candidate code; it
validates a small report against the triggering PR/commit and renders fixed
fields. See [the CI guide](../ci/README.md) for migration details.

## Configuration and installation

The configuration schema uses `config_version: '1'`; this is separate from output
artefact schema versions. `init`, `validate`, and runtime loading share validation
and resolve dataset files relative to their configuration file. Old public
`provider`/`model_id`/`source` configurations are translated through an explicit
compatibility path. Unsupported settings produce errors instead of being ignored.

Pydantic is a core dependency because configuration and artefact validation are
part of the core workflow. Generated templates must complete a config-validation
round trip. Package checks must install the built wheel and sdist in isolated
environments and exercise `init → run/harness → validate → diff`.

## Core-blocker verification snapshot — 5 September 2026

Before the operational-assurance changes below, the working tree passed:

- `make check-fast`: Ruff lint and formatting (575 files), then **7,653 tests
  passed, 72 skipped, 7 deselected**. This target excludes tests marked `slow`
  or `integration`; it is not a claim that those seven tests ran.
- Mypy: no issues in 240 source files, using the project's configured target
  and a clean core-dependency interpreter. The all-extras environment's NumPy
  stubs require newer syntax than that configured target.
- Documentation audit, wiki-link checks, shell syntax, and Git whitespace checks.
- Clean wheel and source-distribution installations outside the source tree:
  all five templates completed their CLI smoke flows. A real Paris-to-Lyon
  answer change lowered accuracy from 1 to 0 and made the regression gate exit 2.

Those historical local packages are in `.tmp/release-check/input-field-fixed-dist/`.
All 241 packaged Python/type-marker files matched the source at that checkpoint,
not the later operational-assurance implementation. Their SHA-256 digests were:

```text
f5f056d923ca7a16572b0bcb09e3a971f25977dd8a58212d296d96f922145e07  insidellms-0.2.0-py3-none-any.whl
19ddf1bd412a8889680f4431505bb85cdd234ac86aa944f308ffa05c29b501c7  insidellms-0.2.0.tar.gz
```

Package installation used an offline dependency cache; it does not prove a
fresh public-index installation. Live GitHub workflow execution and publication
remain unverified. No commit, release tag, or package publication was performed.

## Release boundary and remaining work

Use source-install instructions until package publication and a stable action
reference have been verified. Before publishing, review the complete diff, pass
the local and CI gates, and configure the intended package/repository release
destination. Local builds and tests do not constitute publication.

## Operational assurance — implementation and limits

- [Cryptographic policy](POLICY_ASSURANCE.md): explicit signer identity, OIDC
  issuer and trusted root, signed manifest/records binding, fail-closed verdicts,
  and snapshot-based strict publication. Structural policy is labelled separately.
  Strict snapshots reject symlinks and special files. Authentic SCITT verification
  remains unsupported; real cosign fixture verification remains an optional,
  explicitly skipped gate when the local verifier/fixture is unavailable.
- [Pre-call budgets](RUN_BUDGETS.md): one invocation ledger reserves each supported
  OpenAI/Anthropic attempt before dispatch, including retries and judges.
  Unknown operations, custom callouts, and budgeted resume fail closed. The
  guarantee depends on caller-supplied fixed rates and maximum billable context;
  it is not a provider invoice or account spending cap.
- [Provider catalogue](PROVIDER_CAPABILITIES.md): one immutable declaration drives
  registration and doctor. Local prerequisites, simulated/native operations and
  unverified external requirements are distinct. No live endpoint checks occur.
- Interrupted sync/async runs and harnesses preserve aggregate diagnostics as
  described above, including budget-abort accounting.
- [Editor extension](../extensions/vscode-insidellms/README.md): locked tooling,
  allowlisted normalized VSIX contents, argument-based task execution, workspace
  trust and per-file workspace selection. Two clean builds with different
  timestamps/timezones produced identical archives; real isolated VS Code
  activation and an offline DummyModel task succeeded. Other editors/platforms
  and authored CI execution remain unverified.

No cryptographic identity, price or spending guarantee is inferred from a
structurally valid attestation.

## Operational verification — 5 September 2026

Final-source focused tests: **158 passed, 1 skipped** across policy, budget,
provider catalogue, interrupted execution, scoring and schema compatibility.
The skip requires a real cosign executable and independently signed local
fixture. An independent broader policy/signing suite passed **77 tests with the
same explicit external-verifier skip**. Neither count proves signer authenticity.

All **7** separately marked offline integration tests passed. The DummyModel
golden path produced 12 common record keys and no differences. Full-project
mypy passed **244 source files** using the core-dependency interpreter; Ruff
lint and formatting passed **584 files**. Documentation/wiki and Git whitespace
checks passed. Missing OpenAI/Anthropic SDK simulation ran **9 tests and skipped
28 provider-specific tests**, without skipping the pure ledger cases.

Fresh wheel and sdist installations from an offline dependency cache both ran
all five template smoke flows, schema validation and a real accuracy regression
(1.0 to 0.0, regression-gate exit 2). Their installed package trees and archived
package contents matched all **245 Python/type-marker files** in the final
source. This is local install proof, not public-index or publication proof.

The final VSIX activated in isolated VS Code **1.135.0** and ran a real offline
DummyModel task, exiting 0 with one record and a completed manifest. Its test
workspace included literal shell metacharacters. The test launcher supplies its
environment explicitly after a login-shell-resolution timeout in an earlier
attempt; this does not change the extension launcher or normal editor profile.
Proof and tested archive digest are retained in
`extensions/vscode-insidellms/.editor-test/session-p1ZFg8/proof.json`.

The broad `make check-fast` run finished with **7,751 passed, 73 skipped,
7 deselected and 2 failed**. Both failures were the old initialization-exception
expectations described above, not lost diagnostics. Those assertions were
updated to require the typed wrapper, original cause and explicit zero-record
unhealthy payload. The full harness file then passed **31 tests**; the final
strengthened two cases also passed separately. The broad suite was not rerun
after these test-only expectation updates and is not reported as a green
full-suite run. The final-source focused run above covers the later diagnostic
and budget fixes.

Final local artifact SHA-256 digests:

| Artifact | Local path | SHA-256 |
| --- | --- | --- |
| Wheel | `.tmp/operational-dist/insidellms-0.2.0-py3-none-any.whl` | `d5007c68083f5efbff2a1fdea3e24256ed58098da47ae6eeec6467245d102abd` |
| Sdist, including updated harness tests | `.tmp/operational-final-dist/insidellms-0.2.0.tar.gz` | `91debf3cf0f7058e31e3ed007eb29e557bc6801f7dc61582d273ad9b2156ad93` |
| VSIX | `extensions/vscode-insidellms/insidellms-tools-0.1.0.vsix` | `4a577774e7a0a2e3480476f107cb6f9647ae2c4e552d4e94d72109f9a7dfe761` |

No commit, tag, Marketplace upload or package publication was performed.

## Priority audit remediation — 5 September 2026

This section supersedes the older test-count snapshots above for the priority
remediation working tree. It does not supersede their historical package or
editor-host evidence. The initial remediation addresses A01, A02, A03, A04, A05,
A06, A09, A11 and A13 within the boundaries below. A later follow-up, recorded
after the historical evidence in this section, addresses A07, A08, A10 and A12.
Dependency and development-environment follow-ups remain open, so the repository
is **not broadly release-ready**.

### Implemented boundaries

- Encryption uses exclusive, unpredictable owned staging files with
  descriptor-based permissions and owned-file cleanup. Raw stage descriptors
  are closed if stream construction fails. Publication separately
  uses contained no-follow snapshots that reject hostile links, special files
  and unsupported no-follow operations before a push. These protections are not
  a transaction or lock against concurrent writers. The tests establish local
  filesystem and dispatch ordering; Linux was not executed on this macOS host.
- The alternate `.github/actions/diff-gate` entry point is retired and fails
  before executing tools. Migrate callers to the maintained root action and its
  separate trusted-comment workflow. No hosted workflow was run in this work.
- Aggregate scoring and output validation now finish before a run is declared
  healthy or ultimate processing begins. Failures retain typed interruption
  data, completed items, earlier harness cells and secondary diagnostics.
  Persisted health checks honor explicit abort/unhealthy evidence before the
  schema-1.0.0 timestamp fallback; genuinely completed legacy runs still pass.
- Publication identity v2 names an immutable, finalized payload before push;
  later stage-09 publication receipts remain separate. Existing v1 artifacts
  and their receipts are historical evidence and must not be rewritten or
  silently upgraded. Conflicting historical bytes reject a new attempt.
- Report reconstruction preserves incomplete-run diagnostics and refuses to
  rebuild in a sealed v2 directory; generate a derivative/export artifact
  instead. Abort diagnostic arrays retain manifest entries followed by distinct
  summary entries, preserving full details without duplication across rebuilds.
  Both report-stage allocations have cleanup coverage. The fallback HTML was
  inspected in a real local browser and visibly showed the incomplete banner,
  unknown model, and expected-versus-observed
  record count. This is local rendering evidence, not hosted-browser proof.
- Welch and paired tests use Student-t tails with the applicable degrees of
  freedom. Samples with fewer than two values preserve the existing contract by
  returning a non-significant insufficient-data result; non-finite inputs,
  invalid alpha and unequal paired lengths raise. Verification included a fixed
  240-point, 120-digit mpmath oracle grid plus six extreme-tail cases.
  Recompute historical p-values into separate derived artifacts; do not mutate
  signed or otherwise preserved historical results.
- Matched-compute reporting separates declarations from measured assurance.
  Historical serialized `matching.max_output_tokens: true` (exposed by the
  `output_tokens_matched` property) meant only that limits were declared equal.
  A new true assurance value requires complete, instrumented dispatch-cap and
  response-usage evidence under trusted callbacks. Missing usage is unknown/null
  and known zero remains zero. This is not sandbox, provider-internal,
  input-token-total, price, USD or invoice proof.
- A known satellite sanctions veto can no longer be overridden by a live-model
  recommendation. The test used the real installed satellite dependencies and
  a fake live-advice boundary; it does not establish sanctions-data accuracy or
  real provider behavior.

### Fresh acceptance evidence

- The initial whole-change review found three Important gaps: schema-1.0.0
  persisted failure health, destructive merging of secondary-diagnostic arrays,
  and encryption-test collection without optional crypto. It also identified
  three Minor cleanups: raw descriptor ownership on `fdopen` failure, report-stage
  allocation cleanup, and incomplete public push exception documentation.
  All six received bounded fixes and fresh local regression/documentation
  evidence. These results alone do not constitute independent review or release
  approval.
- The fresh covering/security/runtime/publication suite passed **317 tests with
  1 explicit real-cosign skip**. Crypto was enabled and encryption cases executed.
  In a separate environment with cryptography actually absent, the old staging
  module failed collection; the amended modules and unrelated controls produced
  **10 passes and 2 explicit module skips**. This minimal environment is not a
  full bare-`[dev]` installation. No core/dev dependency was changed to hide the
  absence path. The refreshed satellite suite passed **16 tests** with installed
  dependencies and a fake live-advice boundary.
- Earlier priority checkpoints remain historical: **690 passed, 1 skipped**;
  strict-policy/budget checks **67 passed with the same cosign skip**; macOS
  filesystem/crypto/publication **73 passed**. Unsupported no-follow simulation
  rejected with zero fake pushes; this is not Linux or registry execution.
- The DummyModel golden path passed with **12 common keys and 0 changes**.
- Ruff lint, formatting for **598 files**, documentation/wiki audits and Git
  whitespace checks passed. Isolated-core mypy passed **247 source files**.
  The default `.venv/bin/mypy insideLLMs` command still exits 2 before checking
  project code because the installed NumPy stubs use target-incompatible syntax;
  this environment issue remains open.
- Superseding local wheel and sdist builds in `.tmp/priority-final-dist-JDzY0S`
  include the final-review source/test fixes. Both were force-installed
  offline, ran all five template flows, and made an actual 1-to-0 accuracy
  regression exit 2. Core-only statistical probes passed without SciPy or NumPy.
  Archive and installed payloads matched all **248** Python/type-marker source
  files, and the sdist's four final-wave test files plus the earlier runner
  fixture matched their current root bytes exactly.
  The wheel SHA-256 is
  `b15d8ba3a0d75fda919f514ba88261ef794bd06fa33b511d60b62eeb9150c8f9`;
  the sdist SHA-256 is
  `eceb25dc182e189f76a700b2e46e67ae4d659ed41356bd7dc111756f75582940`.
  Earlier `.tmp/priority-dist-VDvw9T` packages and other attempts, including the
  invalid mixed-version build in `.tmp/priority-dist-RU9yt6`, are historical only
  and not final evidence.
- The first completed broad integration run had **2 failures, 8,111 passes and
  73 skips**. It is retained as diagnostic evidence, not a green gate: one
  failure came from a controller-only plugin-disable environment override and
  the other from a stale Task 4 test fixture. Both were subsequently corrected
  or isolated. The subsequent **8,113 passed, 73 skipped, 7 warnings** checkpoint
  in 192.62 seconds is historical. After the final-review fixes, the superseding
  full suite exited 0 with **8,132 passed, 73 skipped and 7 existing warnings**
  in 278.51 seconds. It included all seven offline integration tests with no
  deselections. Skips cover optional
  dependencies/resources, installed-dependency negative paths, a complex import
  mock and the real-cosign fixture; warnings are from the legacy visualization
  facade, Starlette/httpx and five Seaborn/Matplotlib deprecations.
- The amended report command regenerated the local incomplete-run preview with
  exit 0 and an explicit incomplete warning. Browser DOM and a 1280×720 screenshot
  confirmed the incomplete/unknown-model/expected-2-found-1 banner above the
  successful first-cell metrics. The owned preview tab and local server were closed.

### Remaining release work

The extension development-dependency advisory, the Python/satellite dependency
inventory, and a reproducible supported development environment for the default
typecheck remain open. No Python 3.10 or
Linux execution, hosted CI, real cosign identity verification, remote OCI digest
verification, provider invoice reconciliation, fresh VSIX build, or fresh editor
host run was performed here. Older editor-host and archive proof remains historical
only. No commit, tag, upload, signing, publication or paid provider call occurred.
The initial whole-change review is complete and its six bounded fixes have fresh
local evidence. Implementation and test results alone do not constitute
independent review or release approval.

## Follow-up remediation — 5 September 2026

This follow-up addresses A07, A08, A10 and A12 in the same dirty working tree.
Each fix received scoped independent approval. That approval covers the changed
code only; it is not global release approval.

- Async fail-fast records now carry scheduler-owned provenance for items never
  dispatched. Resume accepts only a verified contiguous suffix after an attempted
  error and rejects ambiguous legacy skipped histories before mutation. It retains
  exact attempted-prefix bytes through same-directory atomic replacement, but
  provides neither cryptographic history identity nor concurrent-writer control.
- Attack, Bias and Agent aggregate scoring accepts validated persisted mappings as
  well as live typed outputs. Required fields and types remain enforced; output
  classes, schemas and serialized output formats are unchanged.
- LaTeX export escapes literal headers and cells after the existing 30-character
  truncation while retaining Unicode and the 20-row cap.
- Both satellite analysis endpoints offload synchronous pipeline work and share a
  process-local four-call admission cap. Overflow returns HTTP 503 before dispatch;
  cancellation retains capacity until the underlying work actually completes. The
  cap does not cover other processes, direct CLI calls or account-wide spending.
  See the satellite README for the endpoint contract.

Fresh integrated evidence for the amended checkout is **8,179 passed, 73 skipped
and 7 existing warnings in 353.90 seconds**, with all seven offline integrations
included and no deselections. The credential-free installed-dependency satellite
suite passed **27 tests in 1.36 seconds**. The DummyModel golden path returned 12
common keys and no changes. Ruff lint, Ruff formatting for **602 files**, and Git
whitespace checks passed. Isolated-core mypy passed **248 source files**; the
default invocation still exits 2 on installed NumPy target-incompatible syntax
before project checking and is not a green gate.

Fresh local wheel and sdist installs exercised all five templates, validation,
run/harness/diff, and an actual accuracy regression from 1 to 0 with gate exit 2.
All **249** Python/type-marker source files matched both archives and installed
trees byte-for-byte. These are local artifacts, not a release or a two-build
reproducibility result:

```text
31036dbc105ec752df2fa03e2806038abfa78f56c54ca89d34bf7c68a9a43928  .tmp/followup-dist-0lm0uatk/insidellms-0.2.0-py3-none-any.whl
17b825f2f2a6b93da0e96fe42610ab3e3c1275e1f0d00f2be642fee9f02b5b1b  .tmp/followup-dist-0lm0uatk/insidellms-0.2.0.tar.gz
```

The dependency advisory/inventory and default-mypy environment remain open. This
follow-up did not establish publication, Linux or Python 3.10 behavior, hosted CI,
real signing, paid-provider billing, or fresh VSIX/editor acceptance.
