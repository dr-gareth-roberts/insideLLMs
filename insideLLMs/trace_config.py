"""Trace configuration for deterministic CI enforcement.

This module provides the public configuration surface for trace recording
and contract validation. Configuration is loaded from YAML and compiled
to the internal validator formats.

Example:
    >>> config = load_trace_config({
    ...     "version": 1,
    ...     "enabled": True,
    ...     "store": {"mode": "full"},
    ...     "contracts": {"enabled": True, "fail_fast": False},
    ... })
    >>> schemas, rules, toggles = config.to_contracts()
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Union

# Type alias for event-kind-aware normaliser function
TracePayloadNormaliserFunc = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


class OnViolationMode(str, Enum):
    """What to do when contracts are violated."""
    RECORD = "record"       # Write violations, continue
    FAIL_PROBE = "fail_probe"  # Mark example as error
    FAIL_RUN = "fail_run"   # Stop harness early


class StoreMode(str, Enum):
    """How much trace data to persist."""
    NONE = "none"           # No storage (renamed from OFF for clarity)
    COMPACT = "compact"     # Hash + violations only (renamed from FINGERPRINT)
    FULL = "full"           # Full events + fingerprint
    # Keep old names as aliases for backwards compatibility
    OFF = "none"
    FINGERPRINT = "compact"


class NormaliserKind(str, Enum):
    """Type of normaliser to use."""
    BUILTIN = "builtin"     # Library-provided normaliser
    IMPORT = "import"       # User-imported function


# =============================================================================
# Normaliser Configuration
# =============================================================================


@dataclass
class NormaliserConfig:
    """Configuration for the payload normaliser.

    The normaliser transforms payloads before fingerprinting to strip
    noise (timestamps, request IDs, etc.) and create stable hashes.

    Attributes:
        kind: "builtin" or "import"
        name: Builtin normaliser name (when kind=builtin)
        import_path: Python import path "pkg.module:callable" (when kind=import)
        drop_keys: Exact key names to drop from payloads
        drop_key_regex: Regex patterns for keys to drop
        hash_paths: Dotpaths to hash instead of storing raw
        hash_strings_over: Hash strings longer than this length
    """
    kind: NormaliserKind = NormaliserKind.BUILTIN
    name: str = "structural_v1"
    import_path: Optional[str] = None
    drop_keys: list[str] = field(default_factory=lambda: [
        "request_id", "response_id", "created", "timestamp", "latency_ms"
    ])
    drop_key_regex: list[str] = field(default_factory=lambda: [])
    hash_paths: list[str] = field(default_factory=lambda: ["result", "raw"])
    hash_strings_over: int = 512


@dataclass
class FingerprintConfig:
    """Configuration for trace fingerprinting.

    Attributes:
        enabled: Whether to compute fingerprints
        algorithm: Hash algorithm (currently only sha256)
        normaliser: Normaliser configuration
    """
    enabled: bool = True
    algorithm: str = "sha256"
    normaliser: NormaliserConfig = field(default_factory=NormaliserConfig)


# =============================================================================
# Store Configuration (backwards compatible)
# =============================================================================


@dataclass
class TraceRedactConfig:
    """Configuration for payload redaction (legacy, use normaliser instead)."""
    enabled: bool = False
    json_pointers: list[str] = field(default_factory=list)
    replacement: str = "<redacted>"


@dataclass
class TraceStoreConfig:
    """Configuration for trace storage.

    Attributes:
        mode: Storage mode (none, compact, full)
        max_events: Maximum events to store (None = unlimited)
        max_event_payload_bytes: Max payload size before truncation
        include_payloads: Whether to include payloads in storage
        redact: Legacy redaction config (use fingerprint.normaliser instead)
    """
    mode: StoreMode = StoreMode.FULL
    max_events: Optional[int] = None
    max_event_payload_bytes: Optional[int] = None
    include_payloads: bool = True
    redact: TraceRedactConfig = field(default_factory=TraceRedactConfig)


# =============================================================================
# Contracts Configuration
# =============================================================================


@dataclass
class GenerateBoundariesConfig:
    """Configuration for generate boundary validation."""
    enabled: bool = True
    start_kind: str = "generate_start"
    end_kind: str = "generate_end"


@dataclass
class StreamBoundariesConfig:
    """Configuration for stream boundary validation."""
    enabled: bool = True
    start_kind: str = "stream_start"
    chunk_kind: str = "stream_chunk"
    end_kind: str = "stream_end"
    stream_id_key: str = "stream_id"
    chunk_index_key: str = "chunk_index"
    first_chunk_index: int = 0
    require_end: bool = True
    require_monotonic_chunks: bool = True


@dataclass
class ToolResultsConfig:
    """Configuration for tool result validation."""
    enabled: bool = True
    call_start_kind: str = "tool_call_start"
    call_result_kind: str = "tool_result"
    call_id_key: str = "call_id"
    require_exactly_one_result: bool = True


@dataclass
class ToolPayloadSchemaConfig:
    """JSON Schema for a tool's arguments."""
    args_schema: dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass
class ToolPayloadsConfig:
    """Configuration for tool payload validation."""
    enabled: bool = True
    tool_key: str = "tool"
    args_key: str = "args"
    tools: dict[str, ToolPayloadSchemaConfig] = field(default_factory=dict)


@dataclass
class ToolOrderConfig:
    """Configuration for tool ordering validation."""
    enabled: bool = True
    allow_unknown_tools: bool = False
    must_follow: dict[str, list[str]] = field(default_factory=dict)
    must_precede: dict[str, list[str]] = field(default_factory=dict)
    forbidden_sequences: list[list[str]] = field(default_factory=list)


@dataclass
class TraceContractsConfig:
    """Configuration for all contract validators.

    Attributes:
        enabled: Master switch for all contract validation
        fail_fast: Stop on first violation (faster, less info)
        generate_boundaries: Config for generate event validation
        stream_boundaries: Config for stream event validation
        tool_results: Config for tool result validation
        tool_payloads: Config for tool payload validation
        tool_order: Config for tool ordering validation
    """
    enabled: bool = True
    fail_fast: bool = False
    generate_boundaries: GenerateBoundariesConfig = field(
        default_factory=GenerateBoundariesConfig
    )
    stream_boundaries: StreamBoundariesConfig = field(
        default_factory=StreamBoundariesConfig
    )
    tool_results: ToolResultsConfig = field(default_factory=ToolResultsConfig)
    tool_payloads: ToolPayloadsConfig = field(default_factory=ToolPayloadsConfig)
    tool_order: ToolOrderConfig = field(default_factory=ToolOrderConfig)


@dataclass
class OnViolationConfig:
    """Configuration for violation handling."""
    mode: OnViolationMode = OnViolationMode.RECORD


# =============================================================================
# Top-Level TraceConfig
# =============================================================================


@dataclass
class TraceConfig:
    """Top-level trace configuration.

    This is the public configuration surface for trace recording and validation.
    Load from YAML with load_trace_config(), then compile to validators with
    to_contracts().

    Attributes:
        version: Config schema version (bump when semantics change)
        enabled: Master switch for tracing
        store: Storage configuration
        fingerprint: Fingerprinting configuration
        contracts: Contract validation configuration
        on_violation: Violation handling configuration
    """
    version: int = 1
    enabled: bool = True
    store: TraceStoreConfig = field(default_factory=TraceStoreConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    contracts: TraceContractsConfig = field(default_factory=TraceContractsConfig)
    on_violation: OnViolationConfig = field(default_factory=OnViolationConfig)

    def to_contracts(self) -> dict[str, Any]:
        """Compile config to validator inputs.

        Returns:
            Dictionary containing:
            - tool_schemas: dict[str, ToolSchema] for validate_tool_payloads
            - tool_order_rules: Optional[ToolOrderRule] for validate_tool_order
            - toggles: dict[str, bool] for each validator
            - fail_fast: bool for early termination
        """
        from insideLLMs.trace_contracts import ToolSchema, ToolOrderRule

        # If contracts disabled, return empty
        if not self.contracts.enabled:
            return {
                "tool_schemas": {},
                "tool_order_rules": None,
                "toggles": {
                    "generate_boundaries": False,
                    "stream_boundaries": False,
                    "tool_results": False,
                    "tool_payloads": False,
                    "tool_order": False,
                },
                "fail_fast": False,
            }

        # Compile tool schemas from JSON Schema to ToolSchema
        tool_schemas: dict[str, "ToolSchema"] = {}
        for tool_name, schema_cfg in self.contracts.tool_payloads.tools.items():
            args_schema = schema_cfg.args_schema
            required_args = args_schema.get("required", [])
            properties = args_schema.get("properties", {})

            # Map JSON Schema types to Python types
            arg_types: dict[str, type] = {}
            for arg_name, prop in properties.items():
                json_type = prop.get("type")
                if json_type == "string":
                    arg_types[arg_name] = str
                elif json_type == "integer":
                    arg_types[arg_name] = int
                elif json_type == "number":
                    arg_types[arg_name] = float
                elif json_type == "boolean":
                    arg_types[arg_name] = bool
                elif json_type == "array":
                    arg_types[arg_name] = list
                elif json_type == "object":
                    arg_types[arg_name] = dict

            # Determine optional args (in properties but not required)
            optional_args = [
                name for name in properties if name not in required_args
            ]

            tool_schemas[tool_name] = ToolSchema(
                name=tool_name,
                required_args=required_args,
                arg_types=arg_types,
                optional_args=optional_args,
            )

        # Compile tool order rules
        tool_order_cfg = self.contracts.tool_order
        tool_order_rules: Optional["ToolOrderRule"] = None
        if tool_order_cfg.enabled and (
            tool_order_cfg.must_follow
            or tool_order_cfg.must_precede
            or tool_order_cfg.forbidden_sequences
        ):
            tool_order_rules = ToolOrderRule(
                name="trace_config_rules",
                must_precede=tool_order_cfg.must_precede,
                must_follow=tool_order_cfg.must_follow,
                forbidden_sequences=tool_order_cfg.forbidden_sequences,
            )

        # Build toggles
        toggles = {
            "generate_boundaries": self.contracts.generate_boundaries.enabled,
            "stream_boundaries": self.contracts.stream_boundaries.enabled,
            "tool_results": self.contracts.tool_results.enabled,
            "tool_payloads": self.contracts.tool_payloads.enabled,
            "tool_order": self.contracts.tool_order.enabled,
        }

        return {
            "tool_schemas": tool_schemas,
            "tool_order_rules": tool_order_rules,
            "toggles": toggles,
            "fail_fast": self.contracts.fail_fast,
        }


# =============================================================================
# TracePayloadNormaliser (Event-Kind-Aware)
# =============================================================================


def _stable_hash(obj: Any) -> dict[str, Any]:
    """Create a stable hash summary of an object.

    Args:
        obj: Any JSON-serialisable object

    Returns:
        Dict with sha256 hash and original length
    """
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    h = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return {"sha256": h, "len": len(data)}


class TracePayloadNormaliser:
    """Event-kind-aware normaliser for trace payloads.

    Transforms payloads before fingerprinting to strip noise and create
    stable, deterministic hashes. Supports:
    - Dropping keys by exact name or regex
    - Hashing large blobs at specific paths
    - Auto-hashing long strings
    - Event-kind-specific handling (e.g., stream chunks)

    Example:
        >>> normaliser = TracePayloadNormaliser(
        ...     drop_keys=["timestamp", "request_id"],
        ...     hash_paths=["result"],
        ...     hash_strings_over=256,
        ... )
        >>> result = normaliser.normalise("tool_call_result", {"result": "long text..."})
    """

    def __init__(
        self,
        drop_keys: Optional[list[str]] = None,
        drop_key_regex: Optional[list[str]] = None,
        hash_paths: Optional[list[str]] = None,
        hash_strings_over: int = 512,
        # Legacy parameters for backwards compatibility
        redact_enabled: bool = False,
        json_pointers: Optional[list[str]] = None,
        replacement: str = "<redacted>",
    ):
        """Initialize the normaliser.

        Args:
            drop_keys: Exact key names to drop from payloads
            drop_key_regex: Regex patterns for keys to drop
            hash_paths: Dotpaths to hash instead of storing raw
            hash_strings_over: Hash strings longer than this
            redact_enabled: Legacy redaction flag
            json_pointers: Legacy JSON pointers for redaction
            replacement: Legacy replacement string
        """
        self._drop_keys = set(drop_keys or [])
        self._drop_key_regex = [re.compile(r) for r in (drop_key_regex or [])]
        self._hash_paths = set(hash_paths or [])
        self._hash_strings_over = hash_strings_over

        # Legacy support
        self._redact_enabled = redact_enabled
        self._json_pointers = json_pointers or []
        self._replacement = replacement

    @classmethod
    def from_config(cls, config: "TraceConfig") -> "TracePayloadNormaliser":
        """Create normaliser from TraceConfig.

        Args:
            config: The trace configuration.

        Returns:
            TracePayloadNormaliser configured from the config.
        """
        norm_cfg = config.fingerprint.normaliser
        redact_cfg = config.store.redact

        return cls(
            drop_keys=norm_cfg.drop_keys,
            drop_key_regex=norm_cfg.drop_key_regex,
            hash_paths=norm_cfg.hash_paths,
            hash_strings_over=norm_cfg.hash_strings_over,
            # Legacy support
            redact_enabled=redact_cfg.enabled,
            json_pointers=redact_cfg.json_pointers,
            replacement=redact_cfg.replacement,
        )

    def _should_drop_key(self, key: str) -> bool:
        """Check if a key should be dropped."""
        if key in self._drop_keys:
            return True
        return any(r.search(key) for r in self._drop_key_regex)

    def _walk(self, kind: str, obj: Any, path: str = "") -> Any:
        """Recursively walk and transform an object.

        Args:
            kind: Event kind for context-specific handling
            obj: The object to transform
            path: Current dotpath within the payload

        Returns:
            Transformed object
        """
        # Hash selected paths
        if path in self._hash_paths:
            return _stable_hash(obj)

        # Hash long strings
        if isinstance(obj, str) and len(obj) > self._hash_strings_over:
            return _stable_hash(obj)

        # Recurse into lists
        if isinstance(obj, list):
            return [self._walk(kind, v, path=path) for v in obj]

        # Recurse into dicts, dropping keys as configured
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                if self._should_drop_key(k):
                    continue
                child_path = f"{path}.{k}" if path else k
                out[k] = self._walk(kind, v, path=child_path)
            return out

        return obj

    def _apply_legacy_redaction(self, obj: dict[str, Any], pointer: str) -> None:
        """Apply legacy JSON pointer redaction."""
        if not pointer.startswith("/"):
            return

        parts = pointer[1:].split("/")
        if not parts:
            return

        current: Any = obj
        for part in parts[:-1]:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return
                except ValueError:
                    return
            else:
                return

        final_key = parts[-1].replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and final_key in current:
            current[final_key] = self._replacement
        elif isinstance(current, list):
            try:
                idx = int(final_key)
                if 0 <= idx < len(current):
                    current[idx] = self._replacement
            except ValueError:
                pass

    def normalise(
        self,
        payload: dict[str, Any],
        kind: Optional[str] = None,
    ) -> dict[str, Any]:
        """Normalise a payload with dropping, hashing, and canonicalisation.

        Args:
            payload: The payload to normalise
            kind: Event kind for context-specific handling (optional)

        Returns:
            Normalised payload ready for fingerprinting
        """
        result = copy.deepcopy(payload)
        event_kind = kind or ""

        # Special handling for stream chunks
        if event_kind == "stream_chunk":
            # Keep structural info, summarise text content
            text = result.get("text") or result.get("chunk")
            if isinstance(text, str):
                result["text_len"] = len(text)
                result.pop("text", None)
                result.pop("chunk", None)

        # Apply legacy redaction if enabled
        if self._redact_enabled:
            for pointer in self._json_pointers:
                self._apply_legacy_redaction(result, pointer)

        # Apply new normalisation
        result = self._walk(event_kind, result)

        # Canonicalise via JSON round-trip with sorted keys
        canonical = json.dumps(result, sort_keys=True, ensure_ascii=False)
        return json.loads(canonical)

    def __call__(
        self,
        kind: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Callable interface for use as TracePayloadNormaliserFunc.

        Args:
            kind: Event kind
            payload: Event payload

        Returns:
            Normalised payload
        """
        return self.normalise(dict(payload), kind=kind)


def make_structural_v1_normaliser(
    *,
    drop_keys: Optional[list[str]] = None,
    drop_key_regex: Optional[list[str]] = None,
    hash_paths: Optional[list[str]] = None,
    hash_strings_over: int = 512,
) -> TracePayloadNormaliser:
    """Factory for the structural_v1 builtin normaliser.

    This normaliser is designed for behavioural drift detection:
    - Drops obviously noisy keys (timestamps, request IDs)
    - Hashes large blobs (tool results, raw provider payloads)
    - Summarises stream chunk text to length only

    Args:
        drop_keys: Keys to drop (defaults to common noise fields)
        drop_key_regex: Regex patterns for keys to drop
        hash_paths: Paths to hash (defaults to ["result", "raw"])
        hash_strings_over: Hash strings longer than this (default 512)

    Returns:
        Configured TracePayloadNormaliser
    """
    return TracePayloadNormaliser(
        drop_keys=drop_keys or [
            "request_id", "response_id", "created", "timestamp",
            "latency_ms", "x-request-id",
        ],
        drop_key_regex=drop_key_regex or ["^x-.*", ".*_request_id$"],
        hash_paths=hash_paths or ["result", "raw"],
        hash_strings_over=hash_strings_over,
    )


# =============================================================================
# Config Loading
# =============================================================================


def load_trace_config(yaml_dict: dict[str, Any]) -> TraceConfig:
    """Load TraceConfig from a YAML-parsed dictionary.

    Args:
        yaml_dict: Dictionary parsed from YAML trace: block.

    Returns:
        TraceConfig with values from dict, defaults for missing keys.

    Example:
        >>> config = load_trace_config({
        ...     "version": 1,
        ...     "enabled": True,
        ...     "store": {"mode": "full"},
        ... })
        >>> config.enabled
        True
    """
    if not yaml_dict:
        return TraceConfig()

    # Parse store config
    store_dict = yaml_dict.get("store", {})
    redact_dict = store_dict.get("redact", {})
    redact = TraceRedactConfig(
        enabled=redact_dict.get("enabled", False),
        json_pointers=redact_dict.get("json_pointers", []),
        replacement=redact_dict.get("replacement", "<redacted>"),
    )

    # Handle mode with backwards compatibility
    mode_str = store_dict.get("mode", "full")
    # Map old names to new
    mode_map = {"off": "none", "fingerprint": "compact"}
    mode_str = mode_map.get(mode_str, mode_str)

    store = TraceStoreConfig(
        mode=StoreMode(mode_str),
        max_events=store_dict.get("max_events"),
        max_event_payload_bytes=store_dict.get("max_event_payload_bytes"),
        include_payloads=store_dict.get("include_payloads", True),
        redact=redact,
    )

    # Parse fingerprint config
    fp_dict = yaml_dict.get("fingerprint", {})
    norm_dict = fp_dict.get("normaliser", {})

    normaliser = NormaliserConfig(
        kind=NormaliserKind(norm_dict.get("kind", "builtin")),
        name=norm_dict.get("name", "structural_v1"),
        import_path=norm_dict.get("import"),
        drop_keys=norm_dict.get("config", {}).get("drop_keys", [
            "request_id", "response_id", "created", "timestamp", "latency_ms"
        ]),
        drop_key_regex=norm_dict.get("config", {}).get("drop_key_regex", []),
        hash_paths=norm_dict.get("config", {}).get("hash_paths", ["result", "raw"]),
        hash_strings_over=norm_dict.get("config", {}).get("hash_strings_over", 512),
    )

    fingerprint = FingerprintConfig(
        enabled=fp_dict.get("enabled", True),
        algorithm=fp_dict.get("algorithm", "sha256"),
        normaliser=normaliser,
    )

    # Parse contracts config
    contracts_dict = yaml_dict.get("contracts", {})
    contracts = _parse_contracts_config(contracts_dict)

    # Parse on_violation config
    on_violation_dict = yaml_dict.get("on_violation", {})
    mode_str = on_violation_dict.get("mode", "record")
    on_violation = OnViolationConfig(mode=OnViolationMode(mode_str))

    return TraceConfig(
        version=yaml_dict.get("version", 1),
        enabled=yaml_dict.get("enabled", True),
        store=store,
        fingerprint=fingerprint,
        contracts=contracts,
        on_violation=on_violation,
    )


def _parse_contracts_config(contracts_dict: dict[str, Any]) -> TraceContractsConfig:
    """Parse the contracts section of config."""
    if not contracts_dict:
        return TraceContractsConfig()

    # Generate boundaries
    gen_dict = contracts_dict.get("generate_boundaries", {})
    generate_boundaries = GenerateBoundariesConfig(
        enabled=gen_dict.get("enabled", True),
        start_kind=gen_dict.get("start_kind", "generate_start"),
        end_kind=gen_dict.get("end_kind", "generate_end"),
    )

    # Stream boundaries
    stream_dict = contracts_dict.get("stream_boundaries", {})
    stream_boundaries = StreamBoundariesConfig(
        enabled=stream_dict.get("enabled", True),
        start_kind=stream_dict.get("start_kind", "stream_start"),
        chunk_kind=stream_dict.get("chunk_kind", "stream_chunk"),
        end_kind=stream_dict.get("end_kind", "stream_end"),
        stream_id_key=stream_dict.get("stream_id_key", "stream_id"),
        chunk_index_key=stream_dict.get("chunk_index_key", "chunk_index"),
        first_chunk_index=stream_dict.get("first_chunk_index", 0),
        require_end=stream_dict.get("require_end", True),
        require_monotonic_chunks=stream_dict.get("require_monotonic_chunks", True),
    )

    # Tool results
    results_dict = contracts_dict.get("tool_results", {})
    tool_results = ToolResultsConfig(
        enabled=results_dict.get("enabled", True),
        call_start_kind=results_dict.get("call_start_kind", "tool_call_start"),
        call_result_kind=results_dict.get("call_result_kind", "tool_result"),
        call_id_key=results_dict.get("call_id_key", "call_id"),
        require_exactly_one_result=results_dict.get("require_exactly_one_result", True),
    )

    # Tool payloads
    payloads_dict = contracts_dict.get("tool_payloads", {})
    tools_dict = payloads_dict.get("tools", {})
    tools = {
        name: ToolPayloadSchemaConfig(
            args_schema=spec.get("args_schema", {}),
            required=spec.get("required", True),
        )
        for name, spec in tools_dict.items()
    }
    tool_payloads = ToolPayloadsConfig(
        enabled=payloads_dict.get("enabled", True),
        tool_key=payloads_dict.get("tool_key", "tool"),
        args_key=payloads_dict.get("args_key", "args"),
        tools=tools,
    )

    # Tool order
    order_dict = contracts_dict.get("tool_order", {})
    tool_order = ToolOrderConfig(
        enabled=order_dict.get("enabled", True),
        allow_unknown_tools=order_dict.get("allow_unknown_tools", False),
        must_follow=order_dict.get("must_follow", {}),
        must_precede=order_dict.get("must_precede", {}),
        forbidden_sequences=order_dict.get("forbidden_sequences", []),
    )

    return TraceContractsConfig(
        enabled=contracts_dict.get("enabled", True),
        fail_fast=contracts_dict.get("fail_fast", False),
        generate_boundaries=generate_boundaries,
        stream_boundaries=stream_boundaries,
        tool_results=tool_results,
        tool_payloads=tool_payloads,
        tool_order=tool_order,
    )


# =============================================================================
# Validation Helper
# =============================================================================


def validate_with_config(
    events: list[Any],
    config: TraceConfig,
) -> list[Any]:
    """Run trace contract validators using config toggles.

    This is a convenience helper that compiles the config and runs
    only the enabled validators.

    Args:
        events: List of trace events (TraceEvent or dict).
        config: The TraceConfig with enabled/disabled validators.

    Returns:
        List of Violation objects from all enabled validators.

    Example:
        >>> config = load_trace_config({"contracts": {"stream_boundaries": {"enabled": False}}})
        >>> violations = validate_with_config(events, config)
    """
    from insideLLMs.trace_contracts import (
        validate_generate_boundaries,
        validate_stream_boundaries,
        validate_tool_results,
        validate_tool_payloads,
        validate_tool_order,
        Violation,
    )

    if not config.enabled or not config.contracts.enabled:
        return []

    compiled = config.to_contracts()
    toggles = compiled["toggles"]
    tool_schemas = compiled["tool_schemas"]
    tool_order_rules = compiled["tool_order_rules"]
    fail_fast = compiled["fail_fast"]

    all_violations: list[Violation] = []

    def _check_fail_fast() -> bool:
        """Return True if we should stop validation."""
        return fail_fast and len(all_violations) > 0

    if toggles.get("generate_boundaries", True):
        all_violations.extend(validate_generate_boundaries(events))
        if _check_fail_fast():
            return all_violations

    if toggles.get("stream_boundaries", True):
        all_violations.extend(validate_stream_boundaries(events))
        if _check_fail_fast():
            return all_violations

    if toggles.get("tool_results", True):
        all_violations.extend(validate_tool_results(events))
        if _check_fail_fast():
            return all_violations

    if toggles.get("tool_payloads", True) and tool_schemas:
        all_violations.extend(validate_tool_payloads(events, tool_schemas))
        if _check_fail_fast():
            return all_violations

    if toggles.get("tool_order", True) and tool_order_rules:
        all_violations.extend(validate_tool_order(events, tool_order_rules))

    # Sort by event sequence for stable output
    all_violations.sort(key=lambda v: v.event_seq)
    return all_violations
