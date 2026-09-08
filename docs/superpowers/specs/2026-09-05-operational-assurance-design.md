# Operational assurance: approved design

Approved in the current task on 5 September 2026. These are five independently
testable workstreams; no publication, paid calls, or real identity enrollment is
authorized by implementation approval.

## Shared constraints

- Preserve the existing dirty working tree and its completed core fixes.
- Strict assurance and budget policies are opt-in; once requested, missing or
  unsupported evidence fails closed without a permissive fallback.
- Ordinary offline DummyModel workflows remain available and deterministic.
- Never equate structural artifacts, installed SDKs, estimated prices, mocked
  verification, or unpacked package parity with stronger proof.
- Python remains compatible with the project's declared version floor; use
  existing Pydantic, pytest and Ruff. Justify new dependencies.
- No commits, tags, publication, real signing identities, or paid provider calls.

## 1. Cryptographic policy assurance

Keep structural completeness explicitly labelled. Add a strict post-signing
verification path using caller-owned exact signer identity, OIDC issuer and
explicit trust source. Verify the required stage set independently of discovered
files, validate envelope and statement structure, and bind signed execution
evidence to the actual manifest and exact records bytes/count. Verification must
not rewrite signed inputs. Missing tools, roots, required bundles, malformed
evidence and cryptographic failures return failed/unavailable checks and nonzero
CLI status. SCITT authenticity remains unsupported and must fail if required.
Policy failure must block automatic OCI publication. Signature authenticity is
not evidence that model execution or the scientific claims themselves are true.

## 2. Pre-call budgets

One invocation-scoped ledger is shared by subject models, supported judges,
retries, async tasks and worker threads. Each provider attempt atomically reserves
its conservative monetary liability before dispatch. Cache hits reserve nothing.
Unknown prices, unsupported endpoints/request shapes, absent enforceable output
caps, and hidden retries deny strict operation. Pricing and request-bound policy
must be explicit and fixed for the invocation, not inferred from heuristic tables.
Budget rejection is not retryable. Confirmed unused reservations may be released;
ambiguous timeout/cancellation/partial-stream liability is retained. A detected
underquote is a visible breach and prevents further admissions. Budgeted resume
is refused until durable reservation recovery exists. Unsupported arbitrary
plugin/probe callouts must not be allowed to bypass the ledger. The guarantee is
admission control under the supplied pricing/bounds, not an account invoice cap.

## 3. Provider catalogue

An immutable lightweight catalogue is the source for builtin registration,
dependency/credential requirements and declared capabilities. Doctor projects
this metadata without constructing clients or making network calls. Preserve
runtime dispatch predicates. Distinguish registration provenance, prerequisites,
native/simulated/unsupported operations and live verification not checked.
Unknown plugin metadata remains unknown. Include OpenRouter, Cohere aliases,
Ollama and vLLM SDK requirements. Documentation must agree with the catalogue.

## 4. Fail-fast diagnostics

Stop further work after a fail-fast error, retain prior completed results and the
failing item, and finalize an incomplete manifest and partial summary before
returning CLI failure or reraising the API exception. Preserve the original error
cause. Do not fabricate successful execution for unattempted work. Include a
structured abort reason and expected/observed counts. Atomic writes and existing
schema/version contracts continue to apply. Budget stops use the same aggregate
finalization path.

## 5. Editor packaging

Lock dependencies and toolchain, use clean installs and builds, allowlist runtime
package contents, and create a local VSIX without Marketplace publication.
Verify byte-identical archives from two clean builds, including differing source
timestamps. Add read-only CI packaging checks. Replace shell command strings with
argument-based process execution, respect workspace trust and the invoking file's
workspace. Verify activation and offline harness execution in an isolated editor
host when available; report unavailable host proof rather than substitute mocks.
