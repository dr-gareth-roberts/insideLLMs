"""DSSE (Dead Simple Signing Envelope) for attestations.

Build and parse DSSE envelopes: payloadType, payload (base64), signatures.
Used as the outer wrapper for in-toto Statements. Signatures are optional
at build time; verification is done by the signing layer.
"""

from __future__ import annotations

import base64
import json
from typing import Any

# Payload type for in-toto Statement (JSON)
PAYLOAD_TYPE_IN_TOTO = "application/vnd.in-toto+json"


def build_dsse_envelope(
    payload: dict[str, Any],
    payload_type: str = PAYLOAD_TYPE_IN_TOTO,
    signatures: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a DSSE envelope from a payload dict.

    Parameters
    ----------
    payload : dict
        The Statement or other payload (will be JSON-serialized then base64).
    payload_type : str
        Media type of the payload (e.g. application/vnd.in-toto+json).
    signatures : list of dict or None
        Optional list of {keyid, sig} (sig base64). Empty if None.

    Returns
    -------
    dict
        DSSE envelope: payloadType, payload (base64), signatures.
    """
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.standard_b64encode(body).decode("ascii")
    return {
        "payloadType": payload_type,
        "payload": payload_b64,
        "signatures": signatures if signatures else [],
    }


def parse_dsse_envelope(envelope: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Parse a DSSE envelope and return the decoded payload and raw payload bytes for verification.

    Parameters
    ----------
    envelope : dict
        Must have payloadType, payload (base64), signatures.

    Returns
    -------
    tuple[dict, str]
        (parsed payload as dict, payload type string).

    Raises
    ------
    ValueError
        If envelope is malformed.
    """
    if "payloadType" not in envelope or "payload" not in envelope:
        raise ValueError("DSSE envelope must have payloadType and payload")
    try:
        raw = base64.standard_b64decode(envelope["payload"])
    except Exception as e:
        raise ValueError(f"Invalid base64 payload: {e}") from e
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid JSON payload: {e}") from e
    return payload, envelope["payloadType"]


def pae(payload_type: str, payload_bytes: bytes) -> bytes:
    """Pre-Authentication Encoding for DSSE (used by signers).

    PAE = "DSSEV1" + SP + len(payload_type) + SP + payload_type + SP + len(payload) + SP + payload
    (SP = space, lengths as decimal ASCII).
    """
    pt = payload_type.encode("utf-8")
    pb = payload_bytes
    return (
        b"DSSEV1 "
        + str(len(pt)).encode("ascii")
        + b" "
        + pt
        + b" "
        + str(len(pb)).encode("ascii")
        + b" "
        + pb
    )
