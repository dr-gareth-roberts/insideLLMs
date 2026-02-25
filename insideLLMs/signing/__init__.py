"""Signing with Sigstore (keyless) for attestations."""

from insideLLMs.signing.cosign import sign_blob, verify_bundle

__all__ = ["sign_blob", "verify_bundle"]
