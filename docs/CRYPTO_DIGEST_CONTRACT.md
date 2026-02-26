# Digest Descriptor Contract

Compact reference for external verifiers interpreting digest descriptors emitted by insideLLMs. Used for records, receipts, attestations, Merkle roots, and run bundle IDs.

## Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `digest` | string | Lowercase hex hash of canonicalized content |
| `algo` | string | Hash algorithm. Currently only `sha256` is supported |
| `canon_version` | string | Canonicalization scheme version (e.g. `canon_v1`) |
| `purpose` | string | Semantic tag for the digest (see below) |
| `created_by` | string \| null | insideLLMs version that produced the digest; may be null in some contexts |

## Purpose Tags

| Purpose | Use |
|---------|-----|
| `record` | Single record in `records.jsonl` |
| `manifest` | Run `manifest.json` |
| `attestation` | DSSE attestation statement |
| `policy_verdict` | Policy evaluation verdict |
| `policy_file` | Policy file digest |
| `dataset_example` | Dataset item commitment |
| `scitt_submission` | SCITT submission payload |

## Merkle Root Descriptors

Merkle roots use `root`, `count`, `algo`, and `canon_version`:

| Field | Type | Description |
|-------|------|-------------|
| `root` | string | Hex Merkle root |
| `count` | int \| null | Number of leaves; null when not applicable |
| `algo` | string | Hash algorithm (`sha256`) |
| `canon_version` | string | Canonicalization version (`canon_v1`) |

## Related

- `insideLLMs.crypto.canonical` — `digest_obj`, `canonical_json_bytes`
- `insideLLMs.crypto.merkle` — `merkle_root_from_items`
- `wiki/advanced/Verifiable-Evaluation.md` — Attest/sign/verify workflows
