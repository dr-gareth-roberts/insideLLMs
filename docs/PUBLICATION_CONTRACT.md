# Publication payload contract

Ultimate mode completes scoring and output validation, writes the execution
manifest and successful summary, then constructs attestations 00–07, the
structural policy verdict, and attestation 08. Only then does it finalize a
private publication tree, its v2 identity descriptor and bundle ID. A requested
OCI push receives this finalized tree. The subsequent stage 09 receipt is local
publication metadata outside that payload. No scoring happens during sealing.

## Exact path selection

The execution inputs admitted from the source directory are `manifest.json`,
`records.jsonl`, `summary.json`, `config.resolved.yaml`, and `receipts/calls.jsonl`.
Only present inputs are included. Newly completed direct ultimate runners emit
the already-scored summary; `insidellms attest` also supports previously
un-attested runs without a summary, without synthesizing or rescoring one.
Structural policy blocks a requested push if manifest or records is missing.

Fresh generated files are:

- `attestations/00.source.dsse.json`, `01.env.dsse.json`, `02.dataset.dsse.json`,
  `03.promptset.dsse.json`, `04.execution.dsse.json`, `05.scoring.dsse.json`,
  `06.report.dsse.json`, `07.claims.dsse.json`, `08.policy.dsse.json` (all within
  `attestations/`).
- `policy/verdict.json`.
- `integrity/records.merkle.json` and `integrity/receipts.merkle.json` when the
  respective source JSONL or explicit root is supplied; `integrity/dataset.merkle.json`
  and `integrity/promptset.merkle.json` only when their explicit roots are supplied.
- When this invocation requests SCITT submission, the newly returned
  `receipts/scitt/04.execution.receipt.json` and `07.claims.receipt.json`.
- `integrity/bundle_identity.json` and `integrity/bundle_id.txt`.

Existing reports, `explain.json`, extra attestations, signing bundles, old SCITT
receipts, old publication receipts, temporary files, arbitrary generated roots,
and all other files are not selected. Original identity metadata is compared
for conflicts, never used as input to the new identity. The entire source is
first snapshotted using contained no-follow I/O: symlinks (including legacy
`results.jsonl`) and special files anywhere in it cause rejection, even at
unselected paths. Materialize aliases only in a separate export copy.

## Identity version 2

`integrity/bundle_identity.json` contains precisely an object with integer
`version: 2` and `files`, an array sorted lexicographically by relative POSIX
path. Each entry has `path` and the lowercase hexadecimal `sha256` digest of
the file's exact bytes. Neither the descriptor itself nor `integrity/bundle_id.txt`
appears in this array. Stage 09 receipts and everything below `publication/`
are outside the identity and outside automatic ultimate pushes.

The bundle ID is SHA-256 of the canonical descriptor: UTF-8 JSON, sorted object
keys, compact separators `(',', ':')`, ASCII escaping enabled, no trailing
newline. All descriptor paths selected by this producer are ASCII. The
descriptor file contains these exact canonical bytes. `bundle_id.txt` contains
the lowercase hexadecimal ID followed by one newline. The following independent
Python code reconstructs and checks a payload without importing insideLLMs:

```python
import hashlib
import json
from pathlib import Path

root = Path("run")
descriptor = json.loads((root / "integrity/bundle_identity.json").read_bytes())
assert set(descriptor) == {"version", "files"} and descriptor["version"] == 2
paths = [entry["path"] for entry in descriptor["files"]]
assert paths == sorted(set(paths))
for entry in descriptor["files"]:
    assert set(entry) == {"path", "sha256"}
    path = Path(entry["path"])
    assert not path.is_absolute() and ".." not in path.parts
    assert entry["sha256"] == hashlib.sha256((root / path).read_bytes()).hexdigest()
canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
bundle_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
assert (root / "integrity/bundle_id.txt").read_text() == bundle_id + "\n"
```

This identity is independent of Merkle `canon_v1` / `canon_v2`. Those version
tags describe Merkle construction; descriptor `version: 2` describes the payload
identity. Identical selected bytes yield identical v2 IDs. Raw call-receipt
latency remains part of selected bytes, so varying latency can change the v2 ID
even though its normalized receipts Merkle root stays constant.

## Outcomes and caller behavior

All selected bytes, descriptor, ID and policy attestation are installed locally
before push; the private tree survives through the push call. Public ORAS push
takes its own defensive contained copy. No automatic retry occurs.

| Result | Stage 09 outcome | Caller result |
| --- | --- | --- |
| No publication requested | `not_requested` | Returns `None` |
| Push returns a nonempty digest | `published` | Returns `None` |
| Push returns no digest | `unknown` | Raises `PublicationError` |
| Push times out (`TimeoutError` or `subprocess.TimeoutExpired`) | `unknown` | Raises `PublicationError`, chained to timeout |
| Other push exception | `failed` | Raises `PublicationError`, chained to original |
| Failed structural policy | No new stage 09 | Raises `RuntimeError`; zero pushes; verdict, 08 and payload preserved |
| Aggregate scoring/output validation rejects | No publication payload | Existing runner interruption error; zero pushes |

The stage 09 predicate records `oci_ref` (requested target, nullable),
`oci_digest` (returned digest, nullable), `payload_id`, and `outcome`.
`PublicationError` exposes `outcome`, `payload_id`, and `receipt_path`. Failed
or uncertain publication preserves the completed execution manifest, summary,
records and attestations exactly. Direct callers retain `runner.last_experiment`.
Execution health is not changed into publication health, and no signed manifest
is rewritten to add remote-write metadata. Current ORAS integration does not
extract a registry digest; its `None` result is therefore honestly reported as
unknown, requiring independent reconciliation, not an automatic retry.

The first receipt occupies `attestations/09.publish.dsse.json`. If it already
exists, it remains byte-for-byte intact. A new explicitly requested attempt
reserves a unique `publication/attempts/<uuid>.09.publish.dsse.json` using
exclusive creation. Consumers must discover both locations. Repeating
no-publication finalization preserves the existing receipt without creating
another. Receipt names are not chronological ordering evidence. A process
crash or failure writing the receipt can leave an empty/partial reserved file;
that is not proof of publication. Reconcile uncertain remote writes externally.

## Historical evidence and migration

The legacy `crypto.run_bundle_id(manifest_digest, roots, attestation_digests)`
algorithm remains unchanged for v1 readers. An ID without a v2 descriptor must
never be interpreted as a v2 identity. Existing generated/signed evidence is
immutable: before adding any payload files, ultimate compares every generated
destination, including ID, descriptor, roots, 00–08 and verdict. Any conflict
rejects with fresh-export guidance before source writes or OCI push. Matching
v2 files are preserved without rewriting them. Existing unselected evidence
also remains untouched.

Consequently, `insidellms attest` can finalize an un-attested existing run or
revisit byte-identical v2 evidence, but cannot silently regenerate conflicting
v1 or signed artifacts. Prepare a separate export containing only the intended
execution inputs, retain the historical tree and IDs, and explicitly authorize
any new signing workflow.

Direct sync/async runner `resume` and `overwrite` now reject directories carrying
either identity file or any `attestations/` or `signing/` marker before model
dispatch or artifact writes. Dangling links and special-file markers also
reject; unsafe integrity containers reject conservatively. Use a fresh run
directory. Ordinary unsealed interrupted runs remain resumable. This guard is
not a general lock against concurrent writers or a universal guard for every
CLI writer. Reporting a sealed run must use a separate derivative/export tree
or refuse to mutate any listed summary/report bytes; downstream commands must
honor the descriptor rather than rewriting evidence in place.

The v2 curated selection is performed by automatic ultimate orchestration.
The generic `push_run_oci` and `publish_verified_run` APIs retain their public
signatures and snapshot the directory their caller explicitly supplies; they
do not rebuild a v2 payload or incorporate later signatures/receipts into its
ID. Adding detached signatures or publication receipts does not change the
listed payload identity, and those additions are not authenticated merely by
the descriptor. Keep structural/authenticity distinctions in
[Policy assurance](POLICY_ASSURANCE.md).

Snapshots detect file changes during copying but are not transactional across
hostile concurrent writes. Keep source trees stable throughout preparation and
installation. Conflict preflight prevents historical rewrites; a disk failure
while adding new files can still leave a partially installed local tree. No
registry or real-signature interoperability proof is claimed by fake-push tests.
