"""SCITT client for transparency log receipts.

Submit DSSE envelopes to a SCITT service and run structural checks on the
stored submission results. This module does NOT perform cryptographic receipt
verification: there is no COSE countersignature check, no Merkle
inclusion-proof check, and no issuer/key validation. Policy can require
execution + claims attestations to have well-formed receipts, but that is a
structural completeness signal only.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import warnings
from typing import Any

from insideLLMs.crypto.canonical import digest_obj

DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 2


class ScittError(Exception):
    """Base exception for SCITT operations."""


class ScittTimeoutError(ScittError):
    """SCITT request timed out."""


class ScittSubmissionError(ScittError):
    """SCITT submission failed (HTTP error or invalid response)."""


def submit_statement(
    dsse_envelope: dict[str, Any],
    service_url: str | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    """Submit a DSSE envelope to the transparency log; return receipt.

    Args:
        dsse_envelope: DSSE envelope to submit.
        service_url: SCITT service base URL (required).
        timeout: Request timeout in seconds.
        retries: Number of retries on transient failure.

    Returns:
        Dict with status, statement_digest, receipt (from service), service_url.
        On error: status="error", message=..., statement_digest=...

    Raises:
        ValueError: If service_url is empty.
    """
    if not service_url:
        raise ValueError("SCITT service_url is required for submission")

    envelope_digest = digest_obj(dsse_envelope, purpose="scitt_submission")["digest"]
    url = f"{service_url.rstrip('/')}/entries"
    body = json.dumps(dsse_envelope).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(  # noqa: S310
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
                response_data = json.loads(response.read().decode("utf-8"))
            return {
                "status": "success",
                "statement_digest": envelope_digest,
                "receipt": response_data,
                "service_url": service_url,
            }
        except urllib.error.HTTPError as e:
            last_error = ScittSubmissionError(f"SCITT HTTP {e.code}: {e.reason}")
            if e.code and 400 <= e.code < 500:
                break
        except TimeoutError:
            last_error = ScittTimeoutError(f"SCITT request timed out after {timeout}s")
        except OSError as e:
            last_error = ScittSubmissionError(f"SCITT request failed: {e}")
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))

    return {
        "status": "error",
        "message": str(last_error) if last_error else "Unknown error",
        "statement_digest": envelope_digest,
    }


def receipt_looks_well_formed(receipt: dict[str, Any], statement_digest: str) -> bool:
    """Structurally check a stored submission result against a statement digest.

    This is NOT cryptographic receipt verification. It only checks fields that
    :func:`submit_statement` itself wrote client-side: status=success, the
    statement_digest matches, and the service returned a non-empty payload.
    It proves nothing about transparency-log inclusion or the service's
    countersignature.
    """
    if receipt.get("status") != "success":
        return False
    rd = receipt.get("statement_digest")
    if not rd or rd != statement_digest:
        return False
    inner = receipt.get("receipt")
    if not isinstance(inner, dict) or not inner:
        return False
    return True


def verify_receipt(receipt: dict[str, Any], statement_digest: str) -> bool:
    """Deprecated alias for :func:`receipt_looks_well_formed`.

    Deprecated because the name overclaimed: no cryptographic verification is
    performed. Use :func:`receipt_looks_well_formed` instead.
    """
    warnings.warn(
        "verify_receipt is deprecated: it performs structural checks only, not "
        "cryptographic receipt verification. Use receipt_looks_well_formed.",
        DeprecationWarning,
        stacklevel=2,
    )
    return receipt_looks_well_formed(receipt, statement_digest)
