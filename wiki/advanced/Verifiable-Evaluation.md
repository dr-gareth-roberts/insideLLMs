---
title: Verifiable Evaluation
parent: Advanced Features
nav_order: 6
---

# Verifiable Evaluation

Build provenance-oriented evaluation runs with attestations, signatures, and transparency evidence.

## What this covers

- Deterministic run artifacts (`records.jsonl`, `manifest.json`, `summary.json`, `report.html`)
- DSSE attestation generation (`insidellms attest`)
- Sigstore signing (`insidellms sign`)
- Signature verification (`insidellms verify-signatures`)
- Optional transparency and distribution paths (SCITT receipts, OCI publish)

## Prerequisites

- A completed run directory (must include `manifest.json`)
- `cosign` installed for signing/verification workflows
- Optional: `oras` for OCI publishing workflows

Use:

```bash
insidellms doctor --format text
```

to check common readiness items (`ultimate:tuf`, `ultimate:cosign`, `ultimate:oras`).

## Quickstart: attest → sign → verify

```bash
# 1) Generate DSSE attestations for an existing run
insidellms attest ./baseline

# 2) Sign generated attestations
insidellms sign ./baseline

# 3) Verify signature bundles
insidellms verify-signatures ./baseline
```

To enforce signer identity constraints:

```bash
insidellms verify-signatures ./baseline --identity "issuer=https://token.actions.githubusercontent.com"
```

## Expected run-directory additions

After `attest`:

- `attestations/*.dsse.json`

After `sign`:

- `signing/*.sigstore.bundle.json`

## Recommended CI posture

1. Generate deterministic baseline/candidate runs.
2. Generate attestations for both runs.
3. Verify signatures in CI.
4. Run `insidellms diff --fail-on-changes` as a behavioral gate.

## Related docs

- [Determinism and CI](../Determinism-and-CI.md)
- [Tracing and Fingerprinting](../Tracing-and-Fingerprinting.md)
- [CLI Reference](../reference/CLI.md)
- `docs/DETERMINISM.md`
- `docs/ARTIFACT_CONTRACT.md`
- `docs/CRYPTO_DIGEST_CONTRACT.md` — Digest descriptor fields for external verifiers
