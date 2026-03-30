"""Cosign (Sigstore) wrapper for signing and verifying blobs.

Shells out to the cosign CLI when available. Use for signing attestation
envelopes and verifying signature bundles.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _cosign_path() -> Optional[Path]:
    """Return path to cosign binary or None if not found."""
    result = shutil.which("cosign")
    return Path(result) if result is not None else None


def _validate_identity_constraints(identity_constraints: str) -> None:
    """Validate identity constraints parameter to prevent command injection.

    Parameters
    ----------
    identity_constraints : str
        The identity constraints string to validate.

    Raises
    ------
    ValueError
        If identity_constraints contains invalid characters or patterns.
    """
    # Only allow alphanumeric, @, -, _, ., =, +, :, and /
    # These are safe characters for identity values like email addresses, URLs, etc.
    if not re.match(r'^[a-zA-Z0-9@._:+=/-]+$', identity_constraints):
        raise ValueError(
            "Invalid identity_constraints: contains disallowed characters. "
            "Only alphanumeric, @, -, _, ., =, +, :, and / are allowed."
        )


def sign_blob(blob_path: Path | str, output_bundle_path: Path | str) -> None:
    """Sign a blob with cosign and write the Sigstore bundle.

    Parameters
    ----------
    blob_path : Path or str
        Path to the file to sign (e.g. an attestation .dsse.json).
    output_bundle_path : Path or str
        Where to write the .sigstore.bundle.json.

    Raises
    ------
    FileNotFoundError
        If cosign is not installed.
    RuntimeError
        If cosign sign fails.
    """
    cosign = _cosign_path()
    if not cosign:
        raise FileNotFoundError(
            "cosign not found. Install cosign for keyless signing: https://docs.sigstore.dev/cosign/installation/"
        )
    blob_path = Path(blob_path)
    output_bundle_path = Path(output_bundle_path)
    if not blob_path.exists():
        raise FileNotFoundError(f"Blob to sign not found: {blob_path}")
    output_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(cosign), "sign-blob", str(blob_path), "--bundle", str(output_bundle_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cosign sign-blob failed: {result.stderr or result.stdout}")


def verify_bundle(
    blob_path: Path | str,
    bundle_path: Path | str,
    identity_constraints: Optional[str] = None,
) -> bool:
    """Verify a Sigstore bundle against the signed blob.

    Parameters
    ----------
    blob_path : Path or str
        Path to the original blob (e.g. attestation .dsse.json).
    bundle_path : Path or str
        Path to the .sigstore.bundle.json (or .bundle).
    identity_constraints : str or None
        Optional identity constraints (e.g. issuer=example@example.com).
        Only alphanumeric characters and @, -, _, ., =, +, :, / are allowed.

    Returns
    -------
    bool
        True if verification succeeded.

    Raises
    ------
    ValueError
        If identity_constraints contains invalid or unsafe characters.
    """
    cosign = _cosign_path()
    if not cosign:
        raise FileNotFoundError("cosign not found. Install cosign for verification.")
    blob_path = Path(blob_path)
    bundle_path = Path(bundle_path)
    if not bundle_path.exists() or not blob_path.exists():
        return False
    cmd = [str(cosign), "verify-blob", "--bundle", str(bundle_path), str(blob_path)]
    if identity_constraints:
        _validate_identity_constraints(identity_constraints)
        cmd.extend(["--cert-identity", identity_constraints])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.returncode == 0
