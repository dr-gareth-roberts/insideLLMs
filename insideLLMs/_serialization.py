"""Internal serialization utilities for deterministic JSON output.

This module provides shared utilities for serializing Python objects to
JSON-compatible representations and generating deterministic fingerprints.
These utilities are used by both the CLI and the runtime runner to ensure
consistent, reproducible serialization across the codebase.

This is an internal module - external code should not import from here directly.
"""

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class StrictSerializationError(TypeError):
    """Raised when strict serialization encounters a non-deterministic value."""


def _path_label(path: tuple[str, ...]) -> str:
    if not path:
        return "<root>"
    return ".".join(path)


def _serialize_dict_key(key: Any, *, strict: bool, path: tuple[str, ...]) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, Enum):
        return str(key.value)
    if isinstance(key, Path):
        return str(key)
    if isinstance(key, datetime):
        return key.isoformat()
    if isinstance(key, (int, bool)):
        return str(key)
    if isinstance(key, float):
        if not math.isfinite(key):
            if strict:
                raise StrictSerializationError(
                    f"Non-finite float key at {_path_label(path)} is not allowed in strict mode."
                )
            return "null"
        return str(key)
    if strict:
        raise StrictSerializationError(
            f"Non-string dict key of type {type(key).__name__} at {_path_label(path)}."
        )
    return str(key)


def serialize_value(value: Any, *, strict: bool = False, _path: tuple[str, ...] = ()) -> Any:
    """Recursively serialize a value to JSON-compatible types.

    Handles conversion of complex Python types (datetime, Path, Enum, dataclass,
    sets, etc.) into JSON-serializable primitives.

    Parameters
    ----------
    value : Any
        The value to serialize. Can be any Python type including nested
        structures like dicts, lists, dataclasses, etc.
    strict : bool, default False
        If True, raise StrictSerializationError when encountering values that
        would fall back to ``str(value)``.

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
        - Other types -> str(value) (or error in strict mode)

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
        return serialize_value(asdict(value), strict=strict, _path=_path)
    if isinstance(value, dict):
        # JSON object keys must be strings; coerce for determinism and robustness.
        result: dict[str, Any] = {}
        seen_keys: dict[str, Any] = {}
        for k, v in value.items():
            key = _serialize_dict_key(k, strict=strict, path=_path)
            if strict and key in seen_keys and k != seen_keys[key]:
                raise StrictSerializationError(
                    f"Dict key collision at {_path_label(_path)}: "
                    f"{seen_keys[key]!r} and {k!r} -> {key!r}."
                )
            seen_keys.setdefault(key, k)
            child_path = (*_path, key)
            result[key] = serialize_value(v, strict=strict, _path=child_path)
        return result
    if isinstance(value, (list, tuple)):
        return [
            serialize_value(v, strict=strict, _path=(*_path, str(i))) for i, v in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        normalized = [
            serialize_value(v, strict=strict, _path=(*_path, f"set[{i}]"))
            for i, v in enumerate(value)
        ]
        try:
            return sorted(normalized)
        except TypeError:
            return sorted(
                normalized,
                key=lambda v: stable_json_dumps(v, strict=strict),
            )
    if isinstance(value, float):
        # Ensure emitted JSON is standards-compliant; Python's json.dumps can emit
        # NaN/Infinity which is not valid JSON.
        if not math.isfinite(value):
            return None
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if strict:
        raise StrictSerializationError(
            f"Unsupported value type {type(value).__name__} at {_path_label(_path)}."
        )
    return str(value)


def stable_json_dumps(data: Any, *, strict: bool = False) -> str:
    """Serialize data to a stable, deterministic JSON string.

    Produces identical JSON output for equivalent data structures regardless
    of dict ordering or internal Python representation. Used for hashing
    and fingerprinting where consistency is required.

    Parameters
    ----------
    data : Any
        Data to serialize. Passed through serialize_value for type conversion.
    strict : bool, default False
        If True, raise StrictSerializationError when serialization would
        otherwise fall back to ``str(value)``.

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
    serialized = serialize_value(data, strict=strict)
    return json.dumps(
        serialized,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda v: serialize_value(v, strict=strict),
        allow_nan=False,
    )


def fingerprint_value(value: Any, *, strict: bool = False) -> Optional[str]:
    """Generate a short fingerprint hash of a value.

    Produces a 12-character hash suitable for use as a compact identifier
    or deduplication key. Returns None for None input.

    Parameters
    ----------
    value : Any
        Value to fingerprint. Can be any JSON-serializable type.
    strict : bool, default False
        If True, raise StrictSerializationError when serialization would
        otherwise fall back to ``str(value)``.

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
    return hashlib.sha256(stable_json_dumps(value, strict=strict).encode("utf-8")).hexdigest()[:12]
