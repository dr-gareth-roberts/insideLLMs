"""Internal serialization utilities for deterministic JSON output.

This module provides shared utilities for serializing Python objects to
JSON-compatible representations and generating deterministic fingerprints.
These utilities are used by both the CLI and the runtime runner to ensure
consistent, reproducible serialization across the codebase.

This is an internal module - external code should not import from here directly.
"""

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


def serialize_value(value: Any) -> Any:
    """Recursively serialize a value to JSON-compatible types.

    Handles conversion of complex Python types (datetime, Path, Enum, dataclass,
    sets, etc.) into JSON-serializable primitives.

    Parameters
    ----------
    value : Any
        The value to serialize. Can be any Python type including nested
        structures like dicts, lists, dataclasses, etc.

    Returns
    -------
    Any
        A JSON-serializable representation of the input value:

        - datetime -> ISO format string
        - Path -> string path
        - Enum -> enum value
        - dataclass -> dict (via asdict)
        - dict/list/tuple -> recursively serialized
        - set/frozenset -> sorted list
        - None/str/int/float/bool -> unchanged
        - Other types -> str(value)

    Examples
    --------
    Serializing datetime and Path objects:

        >>> from datetime import datetime, timezone
        >>> from pathlib import Path
        >>> serialize_value(datetime(2024, 1, 15, tzinfo=timezone.utc))
        '2024-01-15T00:00:00+00:00'
        >>> serialize_value(Path("/home/user/data"))
        '/home/user/data'

    Serializing nested structures:

        >>> data = {"key": [1, 2, {3, 4}], "path": Path("/tmp")}
        >>> serialize_value(data)
        {'key': [1, 2, [3, 4]], 'path': '/tmp'}

    Handling sets (returns sorted list for determinism):

        >>> serialize_value({3, 1, 2})
        [1, 2, 3]
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        normalized = [serialize_value(v) for v in value]
        try:
            return sorted(normalized)
        except TypeError:
            return sorted(
                normalized,
                key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), default=str),
            )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def stable_json_dumps(data: Any) -> str:
    """Serialize data to a stable, deterministic JSON string.

    Produces identical JSON output for equivalent data structures regardless
    of dict ordering or internal Python representation. Used for hashing
    and fingerprinting where consistency is required.

    Parameters
    ----------
    data : Any
        Data to serialize. Passed through serialize_value for type conversion.

    Returns
    -------
    str
        Compact JSON string with sorted keys and minimal separators.

    Examples
    --------
    Consistent output regardless of dict order:

        >>> stable_json_dumps({"b": 2, "a": 1})
        '{"a":1,"b":2}'
        >>> stable_json_dumps({"a": 1, "b": 2})
        '{"a":1,"b":2}'

    Nested structures:

        >>> stable_json_dumps({"outer": {"z": 3, "y": 2}})
        '{"outer":{"y":2,"z":3}}'
    """
    return json.dumps(
        serialize_value(data),
        sort_keys=True,
        separators=(",", ":"),
        default=serialize_value,
    )


def fingerprint_value(value: Any) -> Optional[str]:
    """Generate a short fingerprint hash of a value.

    Produces a 12-character hash suitable for use as a compact identifier
    or deduplication key. Returns None for None input.

    Parameters
    ----------
    value : Any
        Value to fingerprint. Can be any JSON-serializable type.

    Returns
    -------
    Optional[str]
        12-character hexadecimal fingerprint, or None if value is None.

    Examples
    --------
    Fingerprinting a value:

        >>> fp = fingerprint_value({"key": "value"})
        >>> len(fp)
        12
        >>> fp.isalnum()
        True

    None returns None:

        >>> fingerprint_value(None) is None
        True

    Consistent fingerprints:

        >>> fingerprint_value([1, 2, 3]) == fingerprint_value([1, 2, 3])
        True
    """
    if value is None:
        return None
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()[:12]
