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

    def to_contracts(self) -> dict[str, Any]:
        """Compile config to validator inputs.

        Returns:
            Dictionary containing:
            - tool_schemas: dict[str, ToolSchema] for validate_tool_payloads
            - tool_order_rules: Optional[ToolOrderRule] for validate_tool_order
            - toggles: dict[str, bool] for each validator
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
        }


def load_trace_config(yaml_dict: dict[str, Any]) -> TraceConfig:
    """Load TraceConfig from a YAML-parsed dictionary.

    Args:
        yaml_dict: Dictionary parsed from YAML trace: block.

    Returns:
        TraceConfig with values from dict, defaults for missing keys.

    Example:
        >>> config = load_trace_config({"enabled": True, "store": {"mode": "full"}})
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
    store = TraceStoreConfig(
        mode=StoreMode(store_dict.get("mode", "full")),
        max_events=store_dict.get("max_events", 5000),
        include_payloads=store_dict.get("include_payloads", True),
        redact=redact,
    )

    # Parse contracts config
    contracts_dict = yaml_dict.get("contracts", {})
    contracts = _parse_contracts_config(contracts_dict)

    # Parse on_violation config
    on_violation_dict = yaml_dict.get("on_violation", {})
    mode_str = on_violation_dict.get("mode", "record")
    on_violation = OnViolationConfig(mode=OnViolationMode(mode_str))

    return TraceConfig(
        enabled=yaml_dict.get("enabled", True),
        store=store,
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
        name: ToolPayloadSchemaConfig(args_schema=spec.get("args_schema", {}))
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
        must_follow=order_dict.get("must_follow", {}),
        must_precede=order_dict.get("must_precede", {}),
        forbidden_sequences=order_dict.get("forbidden_sequences", []),
    )

    return TraceContractsConfig(
        enabled=contracts_dict.get("enabled", True),
        generate_boundaries=generate_boundaries,
        stream_boundaries=stream_boundaries,
        tool_results=tool_results,
        tool_payloads=tool_payloads,
        tool_order=tool_order,
    )
