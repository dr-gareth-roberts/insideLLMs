"""Trace configuration for deterministic CI enforcement.

This module provides the public configuration surface for trace recording
and contract validation. Configuration is loaded from YAML and compiled
to the internal validator formats.

Example:
    >>> config = load_trace_config({
    ...     "enabled": True,
    ...     "store": {"mode": "full"},
    ...     "contracts": {"enabled": True},
    ... })
    >>> schemas, rules, toggles = config.to_contracts()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class OnViolationMode(str, Enum):
    """What to do when contracts are violated."""
    RECORD = "record"       # Write violations, continue
    FAIL_PROBE = "fail_probe"  # Mark example as error
    FAIL_RUN = "fail_run"   # Stop harness early


class StoreMode(str, Enum):
    """How much trace data to persist."""
    OFF = "off"             # No storage
    FINGERPRINT = "fingerprint"  # Hash + violations only
    FULL = "full"           # Full events + fingerprint


@dataclass
class TraceRedactConfig:
    """Configuration for payload redaction."""
    enabled: bool = False
    json_pointers: list[str] = field(default_factory=list)
    replacement: str = "<redacted>"


@dataclass
class TraceStoreConfig:
    """Configuration for trace storage."""
    mode: StoreMode = StoreMode.FULL
    max_events: int = 5000
    include_payloads: bool = True
    redact: TraceRedactConfig = field(default_factory=TraceRedactConfig)


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
    must_follow: dict[str, list[str]] = field(default_factory=dict)
    must_precede: dict[str, list[str]] = field(default_factory=dict)
    forbidden_sequences: list[list[str]] = field(default_factory=list)


@dataclass
class TraceContractsConfig:
    """Configuration for all contract validators."""
    enabled: bool = True
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


@dataclass
class TraceConfig:
    """Top-level trace configuration.

    This is the public configuration surface for trace recording and validation.
    Load from YAML with load_trace_config(), then compile to validators with
    to_contracts().
    """
    enabled: bool = True
    store: TraceStoreConfig = field(default_factory=TraceStoreConfig)
    contracts: TraceContractsConfig = field(default_factory=TraceContractsConfig)
    on_violation: OnViolationConfig = field(default_factory=OnViolationConfig)
