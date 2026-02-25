"""SCITT client for transparency log receipts.

Submit signed statements to a SCITT service and verify receipts.
Policy can require execution + claims attestations to have valid receipts.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
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
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
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
        except TimeoutError as e:
            last_error = ScittTimeoutError(
                f"SCITT request timed out after {timeout}s"
            )
        except OSError as e:
            last_error = ScittSubmissionError(f"SCITT request failed: {e}")
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))

    return {
        "status": "error",
        "message": str(last_error) if last_error else "Unknown error",
        "statement_digest": envelope_digest,
    }


def verify_receipt(receipt: dict[str, Any], statement_digest: str) -> bool:
    """Verify a receipt against a statement digest.

    Checks: status=success, statement_digest present and matches, receipt payload
    has expected structure (non-empty receipt from service).
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
