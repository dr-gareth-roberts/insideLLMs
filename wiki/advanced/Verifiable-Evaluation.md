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

## Trust boundaries

- `insidellms attest` creates DSSE envelopes with no embedded signatures. They
  are drafts until detached cosign bundles are created and verified.
- `insidellms verify-signatures` checks each DSSE file it finds against its
  detached bundle and fails when that corresponding bundle is missing. It does
  not require the complete expected attestation set.
- Optional `--identity` applies one cosign certificate-identity constraint. The
  command does not enforce an issuer, organization allowlist, or signer policy.
- SCITT receipt checks in the current client are structural only. They do not
  verify a COSE signature, Merkle inclusion proof, issuer, or key.
- The TUF dataset client has no production verification path. It fails closed
  unless tests explicitly request a mock result labelled `verified=False`.

Do not treat an unsigned envelope, a structurally checked receipt, or a TUF
mock as verified provenance.

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
insidellms verify-signatures ./baseline --identity "EXPECTED_CERTIFICATE_IDENTITY"
```

Replace the value with the exact certificate identity you trust. It is not an
issuer expression; issuer constraints and organizational authorization must be
enforced separately.

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
