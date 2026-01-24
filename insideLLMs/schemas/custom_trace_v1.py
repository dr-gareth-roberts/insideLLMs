"""
Custom Trace Schema V1 - Pydantic models for LLM execution trace bundles.

This module defines the data structures and validation logic for the
``insideLLMs.custom.trace@1`` schema format, which captures detailed
execution traces from Large Language Model (LLM) interactions including
tool calls, contract violations, and derived analytics.

Overview
--------
The schema is designed around a hierarchical structure:

- **TraceBundleV1**: The root container holding the entire trace
- **TraceCounts**: Statistics about events in the trace
- **TraceFingerprint**: Cryptographic hash for trace deduplication
- **TraceNormaliser**: Information about how the trace was normalised
- **TraceContractsSummary**: Summary of contract checking results
- **TraceViolation**: Individual contract violations
- **TraceEventStored**: Individual events captured during execution
- **TraceTruncation**: Information about data truncation policies
- **TraceDerived**: Computed analytics derived from the trace

Modes
-----
The schema supports two primary modes:

- ``compact``: Metadata-only mode without raw events (for storage efficiency)
- ``full``: Complete trace with all events stored

Examples
--------
Creating a minimal compact trace bundle:

>>> from insideLLMs.schemas.custom_trace_v1 import (
...     TraceBundleV1, TraceCounts, TraceFingerprint,
...     TraceNormaliser, TraceContractsSummary
... )
>>> bundle = TraceBundleV1(
...     schema_version="insideLLMs.custom.trace@1",
...     mode="compact",
...     counts=TraceCounts(events_total=10, events_stored=0),
...     fingerprint=TraceFingerprint(enabled=False),
...     normaliser=TraceNormaliser(
...         kind="builtin",
...         name="default",
...         config_hash="a" * 64
...     ),
...     contracts=TraceContractsSummary(
...         enabled=False,
...         violations_total=0,
...         violations_stored=0
...     ),
... )

Creating a full trace with events:

>>> from insideLLMs.schemas.custom_trace_v1 import TraceEventStored
>>> full_bundle = TraceBundleV1(
...     schema_version="insideLLMs.custom.trace@1",
...     mode="full",
...     counts=TraceCounts(events_total=2, events_stored=2),
...     fingerprint=TraceFingerprint(
...         enabled=True,
...         value="abc123" + "0" * 58,
...         basis="normalised_full_trace"
...     ),
...     normaliser=TraceNormaliser(
...         kind="builtin",
...         name="default",
...         config_hash="b" * 64
...     ),
...     contracts=TraceContractsSummary(
...         enabled=True,
...         violations_total=0,
...         violations_stored=0
...     ),
...     events_view="normalised",
...     events=[
...         TraceEventStored(seq=0, kind="user_message", payload={"text": "Hello"}),
...         TraceEventStored(seq=1, kind="assistant_message", payload={"text": "Hi!"})
...     ]
... )

Notes
-----
- All hash values (fingerprint, config_hash) must be 64-character hex strings
  representing SHA-256 digests. The optional ``sha256:`` prefix is automatically
  stripped during normalisation.
- Event sequences must be non-decreasing (events are ordered chronologically)
- The schema enforces strict consistency between counts and actual stored items
- All payload and metadata fields must be JSON-serialisable

See Also
--------
insideLLMs.tracing : High-level tracing API
insideLLMs.contracts : Contract definition and checking
pydantic.BaseModel : Base class for all schema models

References
----------
.. [1] Pydantic V2 Documentation: https://docs.pydantic.dev/latest/
.. [2] JSON Schema Specification: https://json-schema.org/
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Regular expression pattern for validating 64-character hexadecimal SHA-256 hashes.
#: Matches exactly 64 hex digits (0-9, a-f, A-F).
HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _canonical_json_bytes(value: Any) -> bytes:
    """
    Convert a value to canonical JSON bytes for consistent hashing.

    This function serialises a Python value to JSON using a deterministic
    format suitable for cryptographic hashing. The output is guaranteed to
    be identical for semantically equivalent inputs.

    Parameters
    ----------
    value : Any
        Any JSON-serialisable Python value (dict, list, str, int, float,
        bool, None, or nested combinations thereof).

    Returns
    -------
    bytes
        UTF-8 encoded JSON bytes with:
        - Keys sorted alphabetically
        - No whitespace (compact separators)
        - Non-ASCII characters preserved (not escaped)

    Raises
    ------
    TypeError
        If the value contains non-JSON-serialisable types (e.g., datetime,
        custom objects, bytes, sets).

    Examples
    --------
    Serialising a simple dictionary:

    >>> _canonical_json_bytes({"b": 2, "a": 1})
    b'{"a":1,"b":2}'

    Nested structures are handled recursively:

    >>> _canonical_json_bytes({"users": [{"name": "Alice"}, {"name": "Bob"}]})
    b'{"users":[{"name":"Alice"},{"name":"Bob"}]}'

    Unicode is preserved:

    >>> _canonical_json_bytes({"greeting": "Hello"})
    b'{"greeting":"Hello"}'

    See Also
    --------
    _assert_jsonable : Validates JSON serialisability with error context
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _assert_jsonable(value: Any, *, where: str) -> Any:
    """
    Validate that a value is JSON-serialisable, with contextual error messages.

    This function attempts to serialise the value to canonical JSON and raises
    a descriptive ValueError if serialisation fails. It is primarily used in
    Pydantic field validators to ensure payload and metadata fields contain
    only JSON-compatible data.

    Parameters
    ----------
    value : Any
        The value to validate for JSON serialisability.
    where : str
        A human-readable description of where this value appears in the schema,
        used in error messages (e.g., "events[].payload", "violations[].meta").

    Returns
    -------
    Any
        The original value, unchanged, if it is JSON-serialisable.

    Raises
    ------
    ValueError
        If the value cannot be serialised to JSON. The error message includes
        the ``where`` context and the underlying serialisation error.

    Examples
    --------
    Validating a simple JSON-compatible dictionary:

    >>> result = _assert_jsonable({"key": "value"}, where="test.field")
    >>> result
    {"key": "value"}

    Validating nested structures:

    >>> _assert_jsonable(
    ...     {"tools": [{"name": "search", "args": {"query": "test"}}]},
    ...     where="events[].payload"
    ... )
    {"tools": [{"name": "search", "args": {"query": "test"}}]}

    Non-serialisable values raise ValueError:

    >>> from datetime import datetime
    >>> _assert_jsonable({"timestamp": datetime.now()}, where="events[].payload")
    Traceback (most recent call last):
        ...
    ValueError: events[].payload must be JSON-serialisable: ...

    See Also
    --------
    _canonical_json_bytes : The underlying serialisation function
    TraceEventStored : Uses this for payload validation
    TraceViolation : Uses this for meta validation
    """
    try:
        _canonical_json_bytes(value)
    except TypeError as e:
        raise ValueError(f"{where} must be JSON-serialisable: {e}") from e
    return value


def _normalise_sha256_value(value: Optional[str]) -> Optional[str]:
    """
    Normalise a SHA-256 hash value by stripping whitespace and optional prefix.

    This function prepares SHA-256 hash strings for validation and storage by:
    1. Stripping leading/trailing whitespace
    2. Removing the optional ``sha256:`` prefix (commonly used in content-addressable
       storage systems)

    Parameters
    ----------
    value : str or None
        A SHA-256 hash string, optionally prefixed with "sha256:".
        If None, returns None unchanged.

    Returns
    -------
    str or None
        The normalised hash string (without prefix), or None if input was None.
        Note: This function does NOT validate the hash format; use ``HEX64_RE``
        for validation after normalisation.

    Examples
    --------
    Basic hash normalisation:

    >>> _normalise_sha256_value("abc123def456" + "0" * 52)
    'abc123def4560000000000000000000000000000000000000000000000000000'

    Stripping the sha256: prefix:

    >>> _normalise_sha256_value("sha256:abc123def456" + "0" * 52)
    'abc123def4560000000000000000000000000000000000000000000000000000'

    Handling whitespace:

    >>> _normalise_sha256_value("  sha256:abc123  ")
    'abc123'

    None passthrough:

    >>> _normalise_sha256_value(None) is None
    True

    See Also
    --------
    HEX64_RE : Regex pattern for validating normalised hashes
    TraceFingerprint : Uses this for fingerprint value normalisation
    TraceNormaliser : Uses this for config_hash normalisation
    """
    if value is None:
        return None
    v = value.strip()
    if v.startswith("sha256:"):
        v = v[len("sha256:") :]
    return v


class TraceCounts(BaseModel):
    """
    Statistics about events captured during trace collection.

    This model tracks the total number of events observed during an LLM
    interaction, how many were stored (which may be fewer due to truncation
    or filtering), and a breakdown by event kind.

    Parameters
    ----------
    events_total : int
        Total number of events observed during the trace, regardless of
        whether they were stored. Must be >= 0.
    events_stored : int
        Number of events actually stored in the trace bundle. Must be >= 0
        and <= events_total. In "compact" mode, this is always 0.
    by_kind : dict[str, int], optional
        Breakdown of event counts by kind (e.g., "tool_call", "user_message").
        Keys must be non-empty strings, values must be >= 0. Defaults to {}.

    Attributes
    ----------
    events_total : int
        Total events observed.
    events_stored : int
        Events actually stored.
    by_kind : dict[str, int]
        Event counts by kind.

    Examples
    --------
    Creating counts for a trace with 100 events, 50 stored:

    >>> counts = TraceCounts(
    ...     events_total=100,
    ...     events_stored=50,
    ...     by_kind={"tool_call": 30, "user_message": 20}
    ... )
    >>> counts.events_total
    100

    Minimal counts for a compact trace:

    >>> compact_counts = TraceCounts(events_total=500, events_stored=0)
    >>> compact_counts.by_kind
    {}

    Validation ensures stored <= total:

    >>> TraceCounts(events_total=10, events_stored=20)
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: ...

    See Also
    --------
    TraceBundleV1 : The parent model containing counts
    TruncationEvents : Explains why events_stored may be less than events_total

    Notes
    -----
    The sum of ``by_kind`` values does not need to equal ``events_total``.
    The ``by_kind`` dict may contain a subset of event kinds, or may be
    empty even when events_total > 0.
    """

    model_config = ConfigDict(extra="forbid")

    events_total: int = Field(ge=0)
    events_stored: int = Field(ge=0)
    by_kind: Dict[str, int] = Field(default_factory=dict)

    @field_validator("by_kind")
    @classmethod
    def _validate_by_kind(cls, v: Dict[str, int]) -> Dict[str, int]:
        """
        Validate the by_kind dictionary structure.

        Parameters
        ----------
        v : dict[str, int]
            The by_kind dictionary to validate.

        Returns
        -------
        dict[str, int]
            The validated dictionary, unchanged.

        Raises
        ------
        ValueError
            If any key is empty or not a string, or if any value is
            negative or not an integer.

        Examples
        --------
        Valid by_kind dictionary:

        >>> TraceCounts._validate_by_kind({"tool_call": 5, "message": 10})
        {"tool_call": 5, "message": 10}

        Empty keys are rejected:

        >>> TraceCounts._validate_by_kind({"": 5})
        Traceback (most recent call last):
            ...
        ValueError: counts.by_kind keys must be non-empty strings
        """
        for k, n in v.items():
            if not isinstance(k, str) or not k:
                raise ValueError("counts.by_kind keys must be non-empty strings")
            if not isinstance(n, int) or n < 0:
                raise ValueError("counts.by_kind values must be integers >= 0")
        return v

    @model_validator(mode="after")
    def _validate_totals(self) -> "TraceCounts":
        """
        Validate that events_stored does not exceed events_total.

        Returns
        -------
        TraceCounts
            Self, if validation passes.

        Raises
        ------
        ValueError
            If events_stored > events_total.

        Examples
        --------
        Valid relationship:

        >>> counts = TraceCounts(events_total=100, events_stored=50)
        >>> counts._validate_totals()
        TraceCounts(events_total=100, events_stored=50, by_kind={})

        Invalid relationship raises error:

        >>> counts = TraceCounts.__new__(TraceCounts)
        >>> counts.events_total = 10
        >>> counts.events_stored = 20
        >>> counts._validate_totals()
        Traceback (most recent call last):
            ...
        ValueError: counts.events_total must be >= counts.events_stored
        """
        if self.events_total < self.events_stored:
            raise ValueError("counts.events_total must be >= counts.events_stored")
        return self


class TraceFingerprint(BaseModel):
    """
    Cryptographic fingerprint for trace deduplication and integrity verification.

    The fingerprint provides a unique identifier for a trace based on its
    normalised content. This allows detection of duplicate traces even when
    metadata differs, and can verify trace integrity during storage/retrieval.

    Parameters
    ----------
    enabled : bool
        Whether fingerprinting is enabled for this trace. When True, ``value``
        and ``basis`` must be provided. When False, both must be None.
    alg : {"sha256"}, optional
        The hashing algorithm used. Currently only "sha256" is supported.
        Defaults to "sha256".
    value : str or None, optional
        The 64-character hexadecimal hash value (without prefix). Required
        when enabled=True. Can include optional "sha256:" prefix which is
        stripped during normalisation.
    basis : {"normalised_full_trace"} or None, optional
        What data the fingerprint is computed from. Currently only
        "normalised_full_trace" is supported. Required when enabled=True.

    Attributes
    ----------
    enabled : bool
        Whether fingerprinting is active.
    alg : str
        The algorithm used (always "sha256").
    value : str or None
        The normalised hash value in lowercase.
    basis : str or None
        The fingerprint computation basis.

    Examples
    --------
    Creating an enabled fingerprint:

    >>> fp = TraceFingerprint(
    ...     enabled=True,
    ...     value="abc123" + "0" * 58,
    ...     basis="normalised_full_trace"
    ... )
    >>> fp.value
    'abc1230000000000000000000000000000000000000000000000000000000000'
    >>> fp.alg
    'sha256'

    Creating a disabled fingerprint:

    >>> fp_disabled = TraceFingerprint(enabled=False)
    >>> fp_disabled.value is None
    True
    >>> fp_disabled.basis is None
    True

    Hash values are normalised to lowercase:

    >>> fp = TraceFingerprint(
    ...     enabled=True,
    ...     value="sha256:ABCDEF" + "0" * 58,
    ...     basis="normalised_full_trace"
    ... )
    >>> fp.value
    'abcdef0000000000000000000000000000000000000000000000000000000000'

    Validation enforces consistency:

    >>> TraceFingerprint(enabled=True, value=None)
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: ...

    See Also
    --------
    TraceNormaliser : Defines how traces are normalised before fingerprinting
    _normalise_sha256_value : Hash value normalisation function

    Notes
    -----
    Fingerprints computed from the same normalised trace content will always
    be identical, regardless of when or where the trace was captured. This
    enables efficient deduplication in storage systems.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    alg: Literal["sha256"] = "sha256"
    value: Optional[str] = None
    basis: Optional[Literal["normalised_full_trace"]] = None

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate and normalise the fingerprint hash value.

        Parameters
        ----------
        v : str or None
            The hash value to validate, optionally prefixed with "sha256:".

        Returns
        -------
        str or None
            The normalised (lowercase, no prefix) hash, or None.

        Raises
        ------
        ValueError
            If the value is not a valid 64-character hex string.

        Examples
        --------
        Valid hash value:

        >>> TraceFingerprint._validate_value("abc" + "0" * 61)
        'abc0000000000000000000000000000000000000000000000000000000000000'

        Prefix is stripped:

        >>> TraceFingerprint._validate_value("sha256:def" + "0" * 61)
        'def0000000000000000000000000000000000000000000000000000000000000'

        Invalid hash raises error:

        >>> TraceFingerprint._validate_value("invalid")
        Traceback (most recent call last):
            ...
        ValueError: fingerprint.value must be a 64-hex sha256 ...
        """
        v = _normalise_sha256_value(v)
        if v is None:
            return None
        if not HEX64_RE.match(v):
            raise ValueError(
                "fingerprint.value must be a 64-hex sha256 (optionally prefixed with 'sha256:')"
            )
        return v.lower()

    @model_validator(mode="after")
    def _validate_enabled_semantics(self) -> "TraceFingerprint":
        """
        Validate semantic consistency between enabled flag and other fields.

        When enabled=True, both value and basis must be present.
        When enabled=False, both must be None.

        Returns
        -------
        TraceFingerprint
            Self, if validation passes.

        Raises
        ------
        ValueError
            If the enabled flag is inconsistent with value/basis presence.

        Examples
        --------
        Enabled requires value and basis:

        >>> fp = TraceFingerprint(enabled=True, value="a"*64, basis="normalised_full_trace")
        >>> fp._validate_enabled_semantics()
        TraceFingerprint(enabled=True, ...)

        Disabled must not have value or basis:

        >>> fp = TraceFingerprint(enabled=False)
        >>> fp._validate_enabled_semantics()
        TraceFingerprint(enabled=False, ...)
        """
        if self.enabled:
            if self.value is None:
                raise ValueError("fingerprint.enabled=true requires fingerprint.value")
            if self.basis is None:
                raise ValueError("fingerprint.enabled=true requires fingerprint.basis")
        else:
            if self.value is not None:
                raise ValueError("fingerprint.enabled=false requires fingerprint.value=null")
            if self.basis is not None:
                raise ValueError("fingerprint.enabled=false requires fingerprint.basis=null")
        return self


class TraceNormaliser(BaseModel):
    """
    Metadata about the normalisation process applied to trace events.

    Normalisation transforms raw trace events into a canonical form before
    fingerprinting. This model records which normaliser was used and its
    configuration, enabling reproducible fingerprint computation.

    Parameters
    ----------
    kind : {"builtin", "import"}
        The type of normaliser used:
        - "builtin": A normaliser included with the insideLLMs package
        - "import": A custom normaliser loaded from an external module
    name : str or None, optional
        The name of the builtin normaliser (e.g., "default", "strict").
        Required when kind="builtin", forbidden when kind="import".
    import_path : str or None, optional
        The Python import path for custom normalisers (e.g.,
        "myproject.normalisers.custom_normaliser"). Required when
        kind="import", forbidden when kind="builtin". Aliased as "import"
        in JSON serialisation.
    config_hash : str
        SHA-256 hash of the normaliser's configuration. This ensures
        fingerprints are reproducible even if normaliser behaviour changes
        between versions. Must be a 64-character hex string.

    Attributes
    ----------
    kind : str
        The normaliser type.
    name : str or None
        Builtin normaliser name.
    import_path : str or None
        Custom normaliser import path.
    config_hash : str
        Configuration hash in lowercase.

    Examples
    --------
    Using a builtin normaliser:

    >>> normaliser = TraceNormaliser(
    ...     kind="builtin",
    ...     name="default",
    ...     config_hash="abc123" + "0" * 58
    ... )
    >>> normaliser.kind
    'builtin'
    >>> normaliser.name
    'default'
    >>> normaliser.import_path is None
    True

    Using a custom imported normaliser:

    >>> custom_normaliser = TraceNormaliser(
    ...     kind="import",
    ...     import_path="myproject.normalisers.sensitive_data_filter",
    ...     config_hash="def456" + "0" * 58
    ... )
    >>> custom_normaliser.name is None
    True
    >>> custom_normaliser.import_path
    'myproject.normalisers.sensitive_data_filter'

    JSON serialisation uses "import" alias:

    >>> normaliser = TraceNormaliser(
    ...     kind="import",
    ...     import_path="custom.norm",
    ...     config_hash="0" * 64
    ... )
    >>> normaliser.model_dump(by_alias=True)
    {'kind': 'import', 'name': None, 'import': 'custom.norm', 'config_hash': '0'*64}

    Validation enforces kind-specific requirements:

    >>> TraceNormaliser(kind="builtin", import_path="invalid")
    Traceback (most recent call last):
        ...
    pydantic_core._pydantic_core.ValidationError: ...

    See Also
    --------
    TraceFingerprint : Uses the normalised trace for hash computation
    insideLLMs.normalisation : Normalisation implementation module

    Notes
    -----
    The config_hash captures all normaliser settings, including version,
    enabled transformations, and any parameters. Two normalisers with the
    same config_hash will produce identical output for the same input.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["builtin", "import"]
    name: Optional[str] = None
    import_path: Optional[str] = Field(default=None, alias="import")
    config_hash: str

    @field_validator("config_hash")
    @classmethod
    def _validate_config_hash(cls, v: str) -> str:
        """
        Validate and normalise the configuration hash.

        Parameters
        ----------
        v : str
            The hash value to validate, optionally prefixed with "sha256:".

        Returns
        -------
        str
            The normalised (lowercase, no prefix) hash.

        Raises
        ------
        ValueError
            If the value is not a valid 64-character hex string.

        Examples
        --------
        Valid hash:

        >>> TraceNormaliser._validate_config_hash("ABC" + "0" * 61)
        'abc0000000000000000000000000000000000000000000000000000000000000'

        With prefix:

        >>> TraceNormaliser._validate_config_hash("sha256:" + "f" * 64)
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'

        Invalid hash:

        >>> TraceNormaliser._validate_config_hash("not-a-hash")
        Traceback (most recent call last):
            ...
        ValueError: normaliser.config_hash must be a 64-hex sha256 ...
        """
        v2 = _normalise_sha256_value(v) or ""
        if not HEX64_RE.match(v2):
            raise ValueError(
                "normaliser.config_hash must be a 64-hex sha256 "
                "(optionally prefixed with 'sha256:')"
            )
        return v2.lower()

    @model_validator(mode="after")
    def _validate_kind_semantics(self) -> "TraceNormaliser":
        """
        Validate that kind-specific fields are correctly set.

        Returns
        -------
        TraceNormaliser
            Self, if validation passes.

        Raises
        ------
        ValueError
            If:
            - kind="builtin" but name is missing or import_path is set
            - kind="import" but import_path is missing or name is set

        Examples
        --------
        Builtin normaliser with name:

        >>> n = TraceNormaliser(kind="builtin", name="default", config_hash="0"*64)
        >>> n._validate_kind_semantics()
        TraceNormaliser(kind='builtin', name='default', ...)

        Import normaliser with path:

        >>> n = TraceNormaliser(kind="import", import_path="mod.norm", config_hash="0"*64)
        >>> n._validate_kind_semantics()
        TraceNormaliser(kind='import', import_path='mod.norm', ...)
        """
        if self.kind == "builtin":
            if not self.name:
                raise ValueError("normaliser.kind='builtin' requires normaliser.name")
            if self.import_path is not None:
                raise ValueError("normaliser.kind='builtin' forbids normaliser.import")
        if self.kind == "import":
            if not self.import_path:
                raise ValueError("normaliser.kind='import' requires normaliser.import")
            if self.name is not None:
                raise ValueError("normaliser.kind='import' forbids normaliser.name")
        return self


class TraceContractsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    fail_fast: bool = False
    violations_total: int = Field(ge=0)
    violations_stored: int = Field(ge=0)
    by_code: Dict[str, int] = Field(default_factory=dict)

    @field_validator("by_code")
    @classmethod
    def _validate_by_code(cls, v: Dict[str, int]) -> Dict[str, int]:
        for k, n in v.items():
            if not isinstance(k, str) or not k:
                raise ValueError("contracts.by_code keys must be non-empty strings")
            if not isinstance(n, int) or n < 0:
                raise ValueError("contracts.by_code values must be integers >= 0")
        return v

    @model_validator(mode="after")
    def _validate_totals(self) -> "TraceContractsSummary":
        if self.violations_total < self.violations_stored:
            raise ValueError("contracts.violations_total must be >= contracts.violations_stored")
        return self


class TraceViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    event_seq: Optional[int] = Field(default=None, ge=0)
    path: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("meta")
    @classmethod
    def _validate_meta_jsonable(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        _assert_jsonable(v, where="violations[].meta")
        return v


class TraceEventStored(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=0)
    kind: str = Field(min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _validate_payload_jsonable(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        _assert_jsonable(v, where="events[].payload")
        return v


class TruncationEvents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applied: bool
    policy: Literal["head"] = "head"
    max_events: Optional[int] = Field(default=None, ge=0)
    dropped: int = Field(default=0, ge=0)
    dropped_by_kind: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_applied(self) -> "TruncationEvents":
        if self.applied:
            if self.max_events is None:
                raise ValueError(
                    "truncation.events.applied=true requires truncation.events.max_events"
                )
        return self


class TruncationPayloads(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applied: bool
    max_bytes: Optional[int] = Field(default=None, gt=0)
    omitted_fields: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_applied(self) -> "TruncationPayloads":
        if self.applied and self.max_bytes is None:
            raise ValueError(
                "truncation.payloads.applied=true requires truncation.payloads.max_bytes"
            )
        return self


class TruncationViolations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applied: bool
    max_violations: Optional[int] = Field(default=None, ge=0)
    dropped: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_applied(self) -> "TruncationViolations":
        if self.applied and self.max_violations is None:
            raise ValueError(
                "truncation.violations.applied=true requires truncation.violations.max_violations"
            )
        return self


class TraceTruncation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: TruncationEvents = Field(default_factory=lambda: TruncationEvents(applied=False))
    payloads: TruncationPayloads = Field(default_factory=lambda: TruncationPayloads(applied=False))
    violations: TruncationViolations = Field(
        default_factory=lambda: TruncationViolations(applied=False)
    )


class DerivedToolCalls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    sequence: List[str] = Field(default_factory=list)
    by_tool: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "DerivedToolCalls":
        if self.count != len(self.sequence):
            raise ValueError("derived.tool_calls.count must equal len(derived.tool_calls.sequence)")
        total = sum(self.by_tool.values())
        if total != self.count:
            raise ValueError(
                "sum(derived.tool_calls.by_tool.values()) must equal derived.tool_calls.count"
            )
        for k, n in self.by_tool.items():
            if not k:
                raise ValueError("derived.tool_calls.by_tool keys must be non-empty strings")
            if n < 0:
                raise ValueError("derived.tool_calls.by_tool counts must be >= 0")
        return self


class TraceDerived(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: DerivedToolCalls = Field(default_factory=lambda: DerivedToolCalls(count=0))


class TraceBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["insideLLMs.custom.trace@1"]
    mode: Literal["compact", "full"]
    counts: TraceCounts
    fingerprint: TraceFingerprint
    normaliser: TraceNormaliser
    contracts: TraceContractsSummary
    violations: List[TraceViolation] = Field(default_factory=list)
    events_view: Optional[Literal["raw", "normalised"]] = None
    events: Optional[List[TraceEventStored]] = None
    truncation: TraceTruncation = Field(default_factory=TraceTruncation)
    derived: TraceDerived = Field(default_factory=TraceDerived)

    @model_validator(mode="after")
    def _validate_bundle_semantics(self) -> "TraceBundleV1":
        if self.mode == "compact":
            if self.events is not None:
                raise ValueError("mode='compact' forbids events")
            if self.events_view is not None:
                raise ValueError("mode='compact' forbids events_view")
            if self.counts.events_stored != 0:
                raise ValueError("mode='compact' requires counts.events_stored == 0")

        if self.mode == "full":
            if self.events is None:
                raise ValueError("mode='full' requires events (can be empty list)")
            if self.events_view is None:
                raise ValueError("mode='full' requires events_view")
            if self.counts.events_stored != len(self.events):
                raise ValueError("counts.events_stored must equal len(events) in mode='full'")

            for i in range(1, len(self.events)):
                if self.events[i].seq < self.events[i - 1].seq:
                    raise ValueError("events[].seq must be non-decreasing")

        if not self.contracts.enabled:
            if self.violations:
                raise ValueError("contracts.enabled=false requires violations=[]")
            if self.contracts.violations_total != 0 or self.contracts.violations_stored != 0:
                raise ValueError(
                    "contracts.enabled=false requires violations_total=violations_stored=0"
                )

        if self.contracts.violations_stored != len(self.violations):
            raise ValueError("contracts.violations_stored must equal len(violations)")

        return self
