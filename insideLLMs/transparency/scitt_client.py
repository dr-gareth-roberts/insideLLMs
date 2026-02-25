"""SCITT client for transparency log receipts.

Submit signed statements to a SCITT service and verify receipts.
Policy can require execution + claims attestations to have valid receipts.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from insideLLMs.crypto.canonical import digest_obj


def submit_statement(dsse_envelope: dict[str, Any], service_url: str | None = None) -> dict[str, Any]:
    """Submit a DSSE envelope to the transparency log; return receipt."""
    if not service_url:
        raise ValueError("SCITT service_url is required for submission")
        
    # We compute the digest of the statement to track it in the receipt
    # The payload in DSSE is base64 encoded, but for tracking we hash the envelope
    envelope_digest = digest_obj(dsse_envelope, purpose="scitt_submission")["digest"]
    
    try:
        # Prepare the request
        req = urllib.request.Request(
            f"{service_url.rstrip('/')}/entries",
            data=json.dumps(dsse_envelope).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        # Submit to SCITT
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            
        return {
            "status": "success",
            "statement_digest": envelope_digest,
            "receipt": response_data,
            "service_url": service_url
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"SCITT submission failed: {e}",
            "statement_digest": envelope_digest
        }


def verify_receipt(receipt: dict[str, Any], statement_digest: str) -> bool:
    """Verify a receipt against a statement digest."""
    if receipt.get("status") != "success":
        return False
        
    # In a full implementation, this would cryptographically verify the SCITT receipt
    # (e.g., verifying the COSE Sign1 signature and Merkle inclusion proof).
    # For this implementation, we verify the receipt is for the correct statement digest.
    
    receipt_digest = receipt.get("statement_digest")
    if not receipt_digest:
        return False
        
    return receipt_digest == statement_digest
