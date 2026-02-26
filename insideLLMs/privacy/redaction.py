"""PII redaction for exports."""

from __future__ import annotations

from typing import Any

from insideLLMs.safety import mask_pii


def redact_pii(obj: Any) -> Any:
    """Recursively redact PII in strings nested within dict/list/tuple structures."""
    if isinstance(obj, str):
        return mask_pii(obj)

    elif isinstance(obj, dict):
        return {k: redact_pii(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [redact_pii(item) for item in obj]

    elif isinstance(obj, tuple):
        return tuple(redact_pii(item) for item in obj)

    # Return other types (int, float, bool, None) unmodified
    return obj
