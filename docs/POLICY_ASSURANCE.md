# Policy assurance

`run_policy(run_dir)` checks structural completeness and labels its verdict
`assurance: structural`. It does not authenticate signatures or SCITT receipts.
Automatic ultimate-mode OCI publication is blocked when this policy fails.
Ultimate finalizes policy attestation 08, a version-2 payload descriptor and its
bundle ID before the push. Stage 09 is a separate publication receipt; failed
or unknown publication never changes execution health or signed manifest bytes.
See [Publication payload contract](PUBLICATION_CONTRACT.md) for exact selection,
independent ID reconstruction, receipt locations, and v1 migration rules.
An ordinary ultimate run still produces unsigned drafts; structural success is
not permission from a cryptographic trust policy.

## Optional claims-quality gate

`run_policy(run_dir, require_claims_pass=True)` additionally requires recorded
metric acceptance. The default is `False`: existing structural verdicts and
Ultimate payload selection remain unchanged, and claim results are not checked.
These are separate assurance questions:

| Question | Result or API | Scope |
| --- | --- | --- |
| Is the expected evidence structurally complete? | `structural_passed` when the gate is enabled | Existing completeness and receipt-shape checks |
| Do the recorded metrics satisfy the declared thresholds? | `claims_quality`, labelled `assurance: metric_acceptance` | Complete, valid and internally consistent claim results |
| Does signed evidence satisfy independently selected trust requirements? | `verify_policy` | Existing signature and artefact-binding checks |

With the gate enabled, the top-level `passed` is the conjunction of
`structural_passed` and `claims_quality.passed`; `checks.claims_quality` records
the additional check. `claims_quality` includes `reasons` and per-claim `checks`.
The top-level `assurance: structural` still describes the provenance checks;
`signature_authenticity` and `scitt_authenticity` remain `not_checked`.

The gate reads the compiler's `claims.json` and `verification.json`. It requires
a nonempty claims list with unique, nonblank string IDs, nonblank metric names
and exactly one result per declared claim. Each result must be an object with
a boolean `passed`, finite numeric `metric_value` and `threshold` (booleans are not numbers here),
and a supported operator (`>`, `>=`, `<`, `<=`, `==`). The threshold and operator
must match the declaration, the recorded comparison must agree with `passed`,
and every claim must pass. An absent, null or empty-string `error` is permitted;
other `error` values reject the result. Missing or unreadable evidence, malformed
JSON or result shapes, duplicate keys, missing/extra results and empty evidence
fail closed.

```python
from insideLLMs.policy import run_policy

verdict = run_policy("./claims-export", require_claims_pass=True)
if not verdict["passed"]:
    raise RuntimeError("; ".join(verdict["reasons"]))
```

For a fresh, unsealed execution export containing precompiled `claims.json`
and `verification.json`, enforce the same requirement when generating evidence:

```sh
insidellms attest ./claims-export --require-claims-pass
```

This exits 1 on a failed combined policy, including when publication is not
requested. The verdict, stage 08 and payload are retained for inspection; no
new stage 09 is produced. This retention applies once policy evaluation is
reached; filesystem containment or input-preflight errors can reject earlier.
Ultimate admits both claim files into its private
payload only when opted in, checks that snapshot and blocks a requested OCI
push on failure. `RunConfig(require_claims_pass=True, run_mode="ultimate",
emit_run_artifacts=True)` enables this requirement for both direct runners;
other run modes, disabled artefact output and nonboolean flag values reject.
The runners do not invoke the claims compiler automatically.

`insideLLMs.contrib.claims.compile_claims(claims_yaml_path, run_dir)` can prepare
the two files before finalisation. Its input is a compiler-compatible
`summary.json` with `metrics[metric_name]["mean"]`; it does not automatically
adapt the runner's standard summary layout. A successful compilation can still
contain failed claims. The gate does not recompute statistics from the summary
or records, establish that evidence belongs to an execution, or validate
measurement methods, experimental design or scientific truth. It adds no
signature or SCITT verification. Stage 07 and the separate cryptographic APIs
retain their existing contracts; a v2 payload digest is not authentication.
Keep historical evidence intact and use a fresh export when changing gate
settings or claim evidence would conflict with already finalised artefacts.

## Cryptographic verification and publication

For post-signing verification, provision a Sigstore TrustedRoot JSON from your
independent trust source and choose the exact signer identity and OIDC issuer.
Do not obtain those trust decisions from the run you are verifying.

```sh
insidellms verify-policy ./run \
  --identity builder@example.com \
  --oidc-issuer https://issuer.example \
  --trusted-root /trusted/config/trusted-root.json
```

The command prints a JSON verdict and exits nonzero on failed or unavailable
checks. The Python API is `insideLLMs.policy.verify_policy(run_dir,
VerificationPolicy(identity, oidc_issuer, trusted_root))`. Its default required
stages are 00–07, independently of which files happen to exist. Python callers
may choose a smaller core stage set, but execution is always required. Stages
08–09 are downstream metadata and are outside this verification policy.

Each required stage must have its detached bundle under `signing/`. The signed
object is the exact full `.dsse.json` file, not the DSSE PAE or only its decoded
payload. Verification snapshots those bytes and the caller's root locally and
does not reserialize or rewrite the signed originals. Every stage must bind the
manifest using the existing canonical `digest_obj(..., purpose="manifest")`
contract. Execution additionally commits SHA-256 of exact `records.jsonl` bytes
and the count of nonblank JSON-object lines. Whitespace changes in records are
therefore detectable. Manifest formatting alone does not change its canonical
commitment. The verdict covers the inspected snapshot; callers must protect
artifacts against concurrent/subsequent mutation before distributing them.

The adapter uses modern cosign `verify-blob --bundle`,
`--certificate-identity`, `--certificate-oidc-issuer`, and `--trusted-root`.
Use a current cosign release with modern Sigstore bundles and explicit root
support. The local trusted root avoids implicit public-root retrieval. No
insecure skip-verification flags are used. Missing executables, unsupported
flags/bundles, root failures, timeouts and verifier rejection fail closed.
The interface was checked against [cosign's official command reference](https://github.com/sigstore/cosign/blob/main/doc/cosign_verify-blob.md)
and [Sigstore verification documentation](https://docs.sigstore.dev/cosign/verifying/verify/).

`--require-scitt` (Python `require_scitt=True`) fails because cryptographic
SCITT receipt verification is unsupported. A receipt that looks well formed
does not satisfy that requirement. Authenticated signatures establish signer
and artifact binding, not that a model call happened or that scientific claims
are true.

Legacy executions without exact records commitments/count fail strict policy.
Regenerate attestations and sign them again through an independently authorized
signing workflow; never patch already signed payloads to make them pass. Keep
old bundles as historical evidence. Strict verification itself never signs,
submits to a transparency service, or publishes.

For explicitly authorized post-signing publication, Python callers can use
`publish_verified_run(run_dir, reference, policy)`. It copies the run into a
private snapshot, verifies that snapshot, and invokes OCI publication only on
success. This prevents changes to the original run after verification from
changing the published bytes. All OCI pushes, including ordinary automatic
publication, use a private regular-file snapshot kept alive until push returns
and removed on success or failure. Files are enumerated in sorted path order.
Strict publication makes a second defensive copy of the admitted private tree.

All publication refuses every symlink and special file, including directory
links, dangling links, FIFOs and the legacy `results.jsonl` alias. Remove or
materialize aliases in a separate export copy; never alter signed originals.
Evidence reads also reject links in required paths. No-follow directory-relative
file access prevents link swapping from reading outside the run. Snapshot
directories have private permissions; files are copied in bounded 1 MiB chunks.
File identity, size and modification time are checked before and after copying;
detected mutation aborts publication. This is not a transactional snapshot of
hostile concurrent writes: callers must keep source trees stable during copying.

This requires POSIX-style `O_NOFOLLOW`, `O_DIRECTORY`, `O_NONBLOCK`, directory
descriptors for `open`/`stat`, no-follow `stat`, and descriptor-based `listdir`
(available on supported Linux/macOS environments). Platforms lacking any of
these operations, including typical Windows Python environments, fail closed.
The ordinary ultimate automatic publication path remains labelled structural
and does not authenticate its unsigned drafts. Containment tests use a fake
ORAS client; actual SDK layer encoding and registry interoperability remain an
independent integration gate and are not established by those tests.

The focused suite mocks verifier outcomes to test enforcement, not authenticity.
For real verification, set `INSIDELLMS_COSIGN_TEST_FIXTURE` to a locally prepared
fixture directory with `run/`, `trusted-root.json` and `policy.json` containing
`identity` and `oidc_issuer`, with cosign on PATH. The optional integration gate
verifies that fixture and rejects a wrong identity. It explicitly skips without
both prerequisites. No real verification proof is claimed from mocked tests.
