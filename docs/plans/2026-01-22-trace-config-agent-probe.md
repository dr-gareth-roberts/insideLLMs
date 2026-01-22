# Trace Config & AgentProbe Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `TraceConfig` dataclass that loads from YAML, compiles to existing validators, and implement `AgentProbe` for tool-using agents with trace integration.

**Architecture:** Create `trace_config.py` with `TraceConfig` as the public config surface, `load_trace_config()` loader, and `to_contracts()` compiler. `AgentProbe` extends `Probe[str]` with a tool-calling loop that records events via `TraceRecorder`.

**Tech Stack:** Python 3.11+, dataclasses, existing `trace_contracts.py` validators, existing `tracing.py` infrastructure.

---

## Task 1: TraceConfig Dataclasses

**Files:**
- Create: `insideLLMs/trace_config.py`
- Test: `tests/test_trace_config.py`

**Step 1: Write the failing test for basic config structure**

```python
# tests/test_trace_config.py
"""Tests for insideLLMs.trace_config module."""

import pytest

from insideLLMs.trace_config import (
    TraceConfig,
    TraceStoreConfig,
    TraceContractsConfig,
    TraceRedactConfig,
    OnViolationMode,
)


class TestTraceConfigStructure:
    """Tests for TraceConfig dataclass structure."""

    def test_default_config(self):
        """Test TraceConfig with defaults."""
        config = TraceConfig()
        assert config.enabled is True
        assert config.store.mode == "full"
        assert config.contracts.enabled is True
        assert config.on_violation.mode == OnViolationMode.RECORD

    def test_store_config_defaults(self):
        """Test TraceStoreConfig defaults."""
        store = TraceStoreConfig()
        assert store.mode == "full"
        assert store.max_events == 5000
        assert store.include_payloads is True
        assert store.redact.enabled is False

    def test_redact_config(self):
        """Test TraceRedactConfig."""
        redact = TraceRedactConfig(
            enabled=True,
            json_pointers=["/payload/headers/authorization"],
            replacement="<redacted>",
        )
        assert redact.enabled is True
        assert len(redact.json_pointers) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_config.py::TestTraceConfigStructure -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'insideLLMs.trace_config'"

**Step 3: Write minimal implementation**

```python
# insideLLMs/trace_config.py
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
    mode: str = "full"  # StoreMode value
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_config.py::TestTraceConfigStructure -v`
Expected: PASS

**Step 5: Commit**

```bash
git add insideLLMs/trace_config.py tests/test_trace_config.py
git commit -m "feat(trace): add TraceConfig dataclasses for YAML config surface"
```

---

## Task 2: load_trace_config() Function

**Files:**
- Modify: `insideLLMs/trace_config.py:1-150` (add function at end)
- Test: `tests/test_trace_config.py`

**Step 1: Write the failing test for YAML loading**

```python
# Add to tests/test_trace_config.py

from insideLLMs.trace_config import load_trace_config


class TestLoadTraceConfig:
    """Tests for load_trace_config function."""

    def test_load_minimal_config(self):
        """Test loading minimal config dict."""
        yaml_dict = {"enabled": True}
        config = load_trace_config(yaml_dict)
        assert config.enabled is True
        assert config.store.mode == "full"  # default

    def test_load_full_config(self):
        """Test loading full config dict."""
        yaml_dict = {
            "enabled": True,
            "store": {
                "mode": "fingerprint",
                "max_events": 1000,
                "include_payloads": False,
                "redact": {
                    "enabled": True,
                    "json_pointers": ["/secret"],
                    "replacement": "[HIDDEN]",
                },
            },
            "contracts": {
                "enabled": True,
                "tool_payloads": {
                    "enabled": True,
                    "tool_key": "tool",
                    "args_key": "args",
                    "tools": {
                        "search": {
                            "args_schema": {
                                "type": "object",
                                "required": ["query"],
                                "properties": {
                                    "query": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "tool_order": {
                    "enabled": True,
                    "must_follow": {"summarise": ["search"]},
                    "forbidden_sequences": [["send_email", "search"]],
                },
            },
            "on_violation": {"mode": "fail_probe"},
        }
        config = load_trace_config(yaml_dict)
        assert config.store.mode == "fingerprint"
        assert config.store.max_events == 1000
        assert config.store.redact.enabled is True
        assert "search" in config.contracts.tool_payloads.tools
        assert config.contracts.tool_order.must_follow == {"summarise": ["search"]}
        assert config.on_violation.mode == OnViolationMode.FAIL_PROBE

    def test_load_disabled_config(self):
        """Test loading disabled config."""
        yaml_dict = {"enabled": False}
        config = load_trace_config(yaml_dict)
        assert config.enabled is False

    def test_load_empty_dict(self):
        """Test loading empty dict uses defaults."""
        config = load_trace_config({})
        assert config.enabled is True  # default
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_config.py::TestLoadTraceConfig -v`
Expected: FAIL with "ImportError: cannot import name 'load_trace_config'"

**Step 3: Write implementation**

```python
# Add to insideLLMs/trace_config.py (after TraceConfig class)

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
        mode=store_dict.get("mode", "full"),
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_config.py::TestLoadTraceConfig -v`
Expected: PASS

**Step 5: Commit**

```bash
git add insideLLMs/trace_config.py tests/test_trace_config.py
git commit -m "feat(trace): add load_trace_config() YAML loader"
```

---

## Task 3: TraceConfig.to_contracts() Compiler

**Files:**
- Modify: `insideLLMs/trace_config.py` (add method to TraceConfig)
- Test: `tests/test_trace_config.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_trace_config.py

from insideLLMs.trace_contracts import ToolSchema, ToolOrderRule


class TestTraceConfigToContracts:
    """Tests for TraceConfig.to_contracts() compiler."""

    def test_to_contracts_basic(self):
        """Test basic compilation to contracts."""
        config = TraceConfig()
        result = config.to_contracts()
        assert "tool_schemas" in result
        assert "tool_order_rules" in result
        assert "toggles" in result

    def test_to_contracts_with_tool_schemas(self):
        """Test compilation with tool schemas."""
        config = load_trace_config({
            "contracts": {
                "tool_payloads": {
                    "tools": {
                        "search": {
                            "args_schema": {
                                "type": "object",
                                "required": ["query", "top_k"],
                                "properties": {
                                    "query": {"type": "string"},
                                    "top_k": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            },
        })
        result = config.to_contracts()
        schemas = result["tool_schemas"]
        assert "search" in schemas
        assert isinstance(schemas["search"], ToolSchema)
        assert "query" in schemas["search"].required_args
        assert "top_k" in schemas["search"].required_args

    def test_to_contracts_with_tool_order(self):
        """Test compilation with tool order rules."""
        config = load_trace_config({
            "contracts": {
                "tool_order": {
                    "must_follow": {"summarise": ["search"]},
                    "must_precede": {"search": ["summarise"]},
                    "forbidden_sequences": [["send_email", "search"]],
                },
            },
        })
        result = config.to_contracts()
        rules = result["tool_order_rules"]
        assert isinstance(rules, ToolOrderRule)
        assert rules.must_follow == {"summarise": ["search"]}
        assert rules.forbidden_sequences == [["send_email", "search"]]

    def test_to_contracts_toggles(self):
        """Test toggles are set correctly."""
        config = load_trace_config({
            "contracts": {
                "generate_boundaries": {"enabled": False},
                "stream_boundaries": {"enabled": True},
            },
        })
        result = config.to_contracts()
        toggles = result["toggles"]
        assert toggles["generate_boundaries"] is False
        assert toggles["stream_boundaries"] is True

    def test_to_contracts_disabled(self):
        """Test contracts disabled returns empty."""
        config = load_trace_config({"contracts": {"enabled": False}})
        result = config.to_contracts()
        assert result["tool_schemas"] == {}
        assert result["tool_order_rules"] is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_config.py::TestTraceConfigToContracts -v`
Expected: FAIL with "AttributeError: 'TraceConfig' object has no attribute 'to_contracts'"

**Step 3: Write implementation**

```python
# Add to TraceConfig class in insideLLMs/trace_config.py

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
        tool_schemas: dict[str, ToolSchema] = {}
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
                    arg_types[arg_name] = (int, float)  # Accept both
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
        tool_order_rules: Optional[ToolOrderRule] = None
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
```

Also add the import at the top and Optional type hint:

```python
# At top of file, update imports
from typing import Any, Optional
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_config.py::TestTraceConfigToContracts -v`
Expected: PASS

**Step 5: Commit**

```bash
git add insideLLMs/trace_config.py tests/test_trace_config.py
git commit -m "feat(trace): add TraceConfig.to_contracts() compiler"
```

---

## Task 4: TracePayloadNormaliser

**Files:**
- Modify: `insideLLMs/trace_config.py` (add normaliser class)
- Test: `tests/test_trace_config.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_trace_config.py

from insideLLMs.trace_config import TracePayloadNormaliser


class TestTracePayloadNormaliser:
    """Tests for TracePayloadNormaliser."""

    def test_normalise_no_redaction(self):
        """Test normalisation without redaction."""
        normaliser = TracePayloadNormaliser(redact_enabled=False)
        payload = {"key": "value", "secret": "password123"}
        result = normaliser.normalise(payload)
        assert result == {"key": "value", "secret": "password123"}

    def test_normalise_with_redaction(self):
        """Test normalisation with redaction."""
        normaliser = TracePayloadNormaliser(
            redact_enabled=True,
            json_pointers=["/secret", "/nested/token"],
            replacement="<REDACTED>",
        )
        payload = {
            "key": "value",
            "secret": "password123",
            "nested": {"token": "abc123", "other": "keep"},
        }
        result = normaliser.normalise(payload)
        assert result["key"] == "value"
        assert result["secret"] == "<REDACTED>"
        assert result["nested"]["token"] == "<REDACTED>"
        assert result["nested"]["other"] == "keep"

    def test_normalise_canonicalises_json(self):
        """Test normalisation produces deterministic output."""
        normaliser = TracePayloadNormaliser(redact_enabled=False)
        payload1 = {"b": 2, "a": 1}
        payload2 = {"a": 1, "b": 2}
        # Both should produce same canonical form
        result1 = normaliser.normalise(payload1)
        result2 = normaliser.normalise(payload2)
        assert result1 == result2

    def test_from_config(self):
        """Test creating normaliser from TraceConfig."""
        config = load_trace_config({
            "store": {
                "redact": {
                    "enabled": True,
                    "json_pointers": ["/auth"],
                    "replacement": "[HIDDEN]",
                },
            },
        })
        normaliser = TracePayloadNormaliser.from_config(config)
        assert normaliser._redact_enabled is True
        assert "/auth" in normaliser._json_pointers
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_config.py::TestTracePayloadNormaliser -v`
Expected: FAIL with "ImportError: cannot import name 'TracePayloadNormaliser'"

**Step 3: Write implementation**

```python
# Add to insideLLMs/trace_config.py (after load_trace_config)

import copy
import json


class TracePayloadNormaliser:
    """Normalises and optionally redacts trace payloads.

    Ensures deterministic output by canonicalising JSON and applying
    configurable redaction for sensitive fields.
    """

    def __init__(
        self,
        redact_enabled: bool = False,
        json_pointers: Optional[list[str]] = None,
        replacement: str = "<redacted>",
    ):
        """Initialize the normaliser.

        Args:
            redact_enabled: Whether to redact sensitive fields.
            json_pointers: JSON pointer paths to redact (e.g., "/secret").
            replacement: Replacement string for redacted values.
        """
        self._redact_enabled = redact_enabled
        self._json_pointers = json_pointers or []
        self._replacement = replacement

    @classmethod
    def from_config(cls, config: TraceConfig) -> "TracePayloadNormaliser":
        """Create normaliser from TraceConfig."""
        redact = config.store.redact
        return cls(
            redact_enabled=redact.enabled,
            json_pointers=redact.json_pointers,
            replacement=redact.replacement,
        )

    def normalise(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalise a payload for deterministic storage.

        Args:
            payload: The payload dict to normalise.

        Returns:
            Normalised (and optionally redacted) payload.
        """
        # Deep copy to avoid mutating original
        result = copy.deepcopy(payload)

        # Apply redaction if enabled
        if self._redact_enabled:
            for pointer in self._json_pointers:
                self._apply_redaction(result, pointer)

        # Canonicalise by round-tripping through sorted JSON
        canonical = json.dumps(result, sort_keys=True, ensure_ascii=False)
        return json.loads(canonical)

    def _apply_redaction(self, obj: dict[str, Any], pointer: str) -> None:
        """Apply redaction at a JSON pointer path.

        Args:
            obj: The object to redact in place.
            pointer: JSON pointer path (e.g., "/nested/secret").
        """
        if not pointer.startswith("/"):
            return

        parts = pointer[1:].split("/")
        current = obj

        # Navigate to parent
        for part in parts[:-1]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return  # Path doesn't exist

        # Redact final key
        final_key = parts[-1]
        if isinstance(current, dict) and final_key in current:
            current[final_key] = self._replacement
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_config.py::TestTracePayloadNormaliser -v`
Expected: PASS

**Step 5: Commit**

```bash
git add insideLLMs/trace_config.py tests/test_trace_config.py
git commit -m "feat(trace): add TracePayloadNormaliser for deterministic payloads"
```

---

## Task 5: validate_with_config() Helper

**Files:**
- Modify: `insideLLMs/trace_config.py` (add helper function)
- Test: `tests/test_trace_config.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_trace_config.py

from insideLLMs.trace_config import validate_with_config
from insideLLMs.tracing import TraceEvent


class TestValidateWithConfig:
    """Tests for validate_with_config helper."""

    def test_validate_with_all_enabled(self):
        """Test validation with all validators enabled."""
        config = load_trace_config({"contracts": {"enabled": True}})
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            TraceEvent(seq=1, kind="generate_end", payload={}),
        ]
        violations = validate_with_config(events, config)
        assert len(violations) == 0

    def test_validate_with_violations(self):
        """Test validation catches violations."""
        config = load_trace_config({"contracts": {"enabled": True}})
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            # Missing generate_end!
        ]
        violations = validate_with_config(events, config)
        assert len(violations) == 1

    def test_validate_with_contracts_disabled(self):
        """Test validation skipped when disabled."""
        config = load_trace_config({"contracts": {"enabled": False}})
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            # Would be violation, but contracts disabled
        ]
        violations = validate_with_config(events, config)
        assert len(violations) == 0

    def test_validate_with_specific_validators_disabled(self):
        """Test individual validators can be disabled."""
        config = load_trace_config({
            "contracts": {
                "enabled": True,
                "generate_boundaries": {"enabled": False},
            },
        })
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            # Would violate generate_boundaries, but it's disabled
        ]
        violations = validate_with_config(events, config)
        # Should not find generate violations
        gen_violations = [v for v in violations if "generate" in v.code.lower()]
        assert len(gen_violations) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_config.py::TestValidateWithConfig -v`
Expected: FAIL with "ImportError: cannot import name 'validate_with_config'"

**Step 3: Write implementation**

```python
# Add to insideLLMs/trace_config.py (after TracePayloadNormaliser)

from insideLLMs.trace_contracts import (
    Violation,
    validate_generate_boundaries,
    validate_stream_boundaries,
    validate_tool_results,
    validate_tool_payloads,
    validate_tool_order,
)
from insideLLMs.tracing import TraceEvent


def validate_with_config(
    events: list[TraceEvent] | list[dict[str, Any]],
    config: TraceConfig,
) -> list[Violation]:
    """Run contract validation using TraceConfig settings.

    This is a convenience wrapper that respects config toggles and compiles
    schemas/rules from the config.

    Args:
        events: List of trace events to validate.
        config: The TraceConfig controlling which validators run.

    Returns:
        List of Violation objects, sorted by event sequence.
    """
    if not config.contracts.enabled:
        return []

    violations: list[Violation] = []
    contracts = config.to_contracts()
    toggles = contracts["toggles"]

    # Run structural validators based on toggles
    if toggles.get("generate_boundaries", True):
        violations.extend(validate_generate_boundaries(events))

    if toggles.get("stream_boundaries", True):
        violations.extend(validate_stream_boundaries(events))

    if toggles.get("tool_results", True):
        violations.extend(validate_tool_results(events))

    # Run schema validator if enabled and schemas exist
    tool_schemas = contracts["tool_schemas"]
    if toggles.get("tool_payloads", True) and tool_schemas:
        violations.extend(validate_tool_payloads(events, tool_schemas))

    # Run order validator if enabled and rules exist
    tool_order_rules = contracts["tool_order_rules"]
    if toggles.get("tool_order", True) and tool_order_rules:
        violations.extend(validate_tool_order(events, tool_order_rules))

    # Sort by event sequence
    violations.sort(key=lambda v: v.event_seq)
    return violations
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_config.py::TestValidateWithConfig -v`
Expected: PASS

**Step 5: Commit**

```bash
git add insideLLMs/trace_config.py tests/test_trace_config.py
git commit -m "feat(trace): add validate_with_config() helper"
```

---

## Task 6: AgentProbe Base Implementation

**Files:**
- Create: `insideLLMs/probes/agent.py`
- Test: `tests/test_agent_probe.py`

**Step 1: Write the failing test**

```python
# tests/test_agent_probe.py
"""Tests for AgentProbe."""

import json
import pytest

from insideLLMs.probes.agent import AgentProbe
from insideLLMs.types import ResultStatus


class MockModel:
    """Mock model for testing."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._call_index = 0

    def stream(self, prompt: str, **kwargs):
        """Return pre-configured response as chunks."""
        if self._call_index >= len(self._responses):
            response = '{"final": "No more responses"}'
        else:
            response = self._responses[self._call_index]
            self._call_index += 1
        # Yield as single chunk for simplicity
        yield response


class TestAgentProbeBasic:
    """Basic tests for AgentProbe."""

    def test_simple_final_answer(self):
        """Test agent that returns final answer immediately."""
        model = MockModel(['{"final": "The answer is 42"}'])
        tools = {}
        probe = AgentProbe(tools=tools)

        result = probe.run(model, {"question": "What is the answer?"})
        assert result.status == ResultStatus.SUCCESS
        assert result.output == "The answer is 42"

    def test_tool_call_then_final(self):
        """Test agent that calls a tool then returns final."""
        model = MockModel([
            '{"tool": "search", "args": {"query": "test"}}',
            '{"final": "Found: test result"}',
        ])

        search_called = []

        def search_tool(args):
            search_called.append(args)
            return {"results": ["item1", "item2"]}

        tools = {"search": search_tool}
        probe = AgentProbe(tools=tools, max_steps=4)

        result = probe.run(model, {"question": "Search for test"})
        assert result.status == ResultStatus.SUCCESS
        assert "Found: test result" in result.output
        assert len(search_called) == 1
        assert search_called[0]["query"] == "test"

    def test_max_steps_limit(self):
        """Test agent stops after max_steps."""
        # Model always returns tool call, never final
        model = MockModel([
            '{"tool": "search", "args": {"query": "1"}}',
            '{"tool": "search", "args": {"query": "2"}}',
            '{"tool": "search", "args": {"query": "3"}}',
        ])

        tools = {"search": lambda args: {"result": "ok"}}
        probe = AgentProbe(tools=tools, max_steps=2)

        result = probe.run(model, {"question": "Loop forever"})
        # Should hit max_steps and return default message
        assert "No final answer" in result.output

    def test_unknown_tool_error(self):
        """Test error on unknown tool."""
        model = MockModel(['{"tool": "unknown_tool", "args": {}}'])
        tools = {}  # No tools registered
        probe = AgentProbe(tools=tools)

        result = probe.run(model, {"question": "Use unknown tool"})
        assert result.status == ResultStatus.ERROR
        assert "Unknown tool" in result.error
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_probe.py::TestAgentProbeBasic -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'insideLLMs.probes.agent'"

**Step 3: Write implementation**

```python
# insideLLMs/probes/agent.py
"""AgentProbe for testing tool-using LLM agents.

This module provides a minimal, framework-agnostic agent probe that:
- Runs a tool-calling loop with configurable max_steps
- Records tool calls and results via TraceRecorder
- Supports trace config for contract validation
- Integrates with the insideLLMs probe infrastructure
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from insideLLMs.probes.base import Probe
from insideLLMs.types import ProbeCategory, ProbeResult, ResultStatus

# Type alias for tool functions
ToolFn = Callable[[dict[str, Any]], Any]


class AgentProbe(Probe[str]):
    """Probe for testing tool-using LLM agents.

    Runs a simple agent loop:
    1. Send prompt to model
    2. Parse response as JSON: {"tool": ..., "args": ...} or {"final": ...}
    3. If tool call, execute tool and loop
    4. If final, return the answer

    Attributes:
        name: Probe name (default: "agent").
        tools: Dict mapping tool names to callable functions.
        max_steps: Maximum number of agent steps before forcing stop.
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        tools: dict[str, ToolFn],
        max_steps: int = 4,
        name: str = "AgentProbe",
    ):
        """Initialize the agent probe.

        Args:
            tools: Dict mapping tool names to callable functions.
            max_steps: Maximum number of agent steps (default: 4).
            name: Probe name (default: "AgentProbe").
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)
        self._tools = tools
        self._max_steps = max_steps

    def run(
        self,
        model: Any,
        data: Any,
        **kwargs: Any,
    ) -> ProbeResult[str]:
        """Run the agent probe.

        Args:
            model: Model with stream(prompt, **kwargs) method.
            data: Input data, expected to have "question" key if dict.
            **kwargs: Additional arguments passed to model.

        Returns:
            ProbeResult with the agent's final answer or error.
        """
        # Extract question from data
        if isinstance(data, dict):
            user_question = data.get("question", str(data))
        else:
            user_question = str(data)

        scratch: list[dict[str, Any]] = []  # Tool results
        final_text: Optional[str] = None

        for step in range(self._max_steps):
            prompt = self._render_prompt(user_question, scratch)

            # Stream response from model
            chunks: list[str] = []
            try:
                for chunk in model.stream(prompt, **kwargs):
                    chunks.append(chunk)
            except Exception as e:
                return ProbeResult(
                    input=data,
                    status=ResultStatus.ERROR,
                    error=str(e),
                )

            assistant_text = "".join(chunks).strip()
            action = self._parse_action_json(assistant_text)

            # Check for final answer
            if "final" in action:
                final_text = str(action["final"])
                break

            # Handle tool call
            tool_name = str(action.get("tool", "unknown"))
            tool_args = action.get("args", {})

            if tool_name not in self._tools:
                return ProbeResult(
                    input=data,
                    status=ResultStatus.ERROR,
                    error=f"Unknown tool '{tool_name}'",
                )

            # Execute tool
            try:
                tool_result = self._tools[tool_name](tool_args)
            except Exception as e:
                return ProbeResult(
                    input=data,
                    status=ResultStatus.ERROR,
                    error=f"Tool '{tool_name}' failed: {e}",
                )

            # Canonicalise result and add to scratch
            tool_result_canon = self._canonicalise(tool_result)
            scratch.append({"tool": tool_name, "result": tool_result_canon})

        if final_text is None:
            final_text = "No final answer produced."

        return ProbeResult(
            input=data,
            output=final_text,
            status=ResultStatus.SUCCESS,
        )

    def _render_prompt(
        self,
        question: str,
        scratch: list[dict[str, Any]],
    ) -> str:
        """Render the agent prompt.

        Args:
            question: The user's question.
            scratch: Tool results so far.

        Returns:
            Formatted prompt string.
        """
        tool_state = json.dumps(scratch, sort_keys=True, ensure_ascii=False)
        return (
            "You are an agent. You must respond ONLY with a single JSON object.\n"
            "Either:\n"
            '  {"tool": "<name>", "args": {...}}\n'
            'or {"final": "<answer>"}\n\n'
            f"Question: {question}\n"
            f"Tool state (JSON): {tool_state}\n"
        )

    def _parse_action_json(self, text: str) -> dict[str, Any]:
        """Parse JSON action from model response.

        Args:
            text: Raw model response text.

        Returns:
            Parsed action dict with "tool"/"args" or "final" key.
        """
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"final": text}

        blob = text[start : end + 1]
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            return {"final": text}

        if isinstance(obj, dict) and ("tool" in obj or "final" in obj):
            return obj
        return {"final": text}

    def _canonicalise(self, value: Any) -> Any:
        """Canonicalise a value for deterministic storage.

        Args:
            value: Value to canonicalise.

        Returns:
            Canonicalised value (JSON round-tripped if possible).
        """
        try:
            s = json.dumps(value, sort_keys=True, ensure_ascii=False)
            return json.loads(s)
        except (TypeError, ValueError):
            return value
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_probe.py::TestAgentProbeBasic -v`
Expected: PASS

**Step 5: Commit**

```bash
git add insideLLMs/probes/agent.py tests/test_agent_probe.py
git commit -m "feat(probes): add AgentProbe for tool-using agents"
```

---

## Task 7: AgentProbe Trace Integration

**Files:**
- Modify: `insideLLMs/probes/agent.py` (add trace recording)
- Test: `tests/test_agent_probe.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_agent_probe.py

from insideLLMs.tracing import TraceRecorder, TraceEventKind
from insideLLMs.trace_config import load_trace_config


class TestAgentProbeTracing:
    """Tests for AgentProbe trace integration."""

    def test_records_tool_calls(self):
        """Test that tool calls are recorded in trace."""
        model = MockModel([
            '{"tool": "search", "args": {"query": "test"}}',
            '{"final": "Done"}',
        ])
        tools = {"search": lambda args: {"results": []}}

        recorder = TraceRecorder()
        probe = AgentProbe(tools=tools)
        result = probe.run(model, {"question": "Test"}, _trace_recorder=recorder)

        # Check tool events were recorded
        events = recorder.events
        tool_starts = [e for e in events if e.kind == TraceEventKind.TOOL_CALL_START.value]
        tool_results = [e for e in events if e.kind == TraceEventKind.TOOL_RESULT.value]

        assert len(tool_starts) == 1
        assert len(tool_results) == 1
        assert tool_starts[0].payload["tool_name"] == "search"

    def test_trace_in_metadata(self):
        """Test that trace fingerprint appears in result metadata."""
        model = MockModel(['{"final": "42"}'])
        tools = {}

        trace_config = load_trace_config({"enabled": True})
        probe = AgentProbe(tools=tools, trace_config=trace_config)
        result = probe.run(model, {"question": "Test"})

        assert "custom" in result.metadata
        assert "trace" in result.metadata["custom"]
        assert "fingerprint" in result.metadata["custom"]["trace"]

    def test_trace_violations_recorded(self):
        """Test that violations are recorded in metadata."""
        model = MockModel([
            '{"tool": "search", "args": {}}',  # Missing required "query"
            '{"final": "Done"}',
        ])
        tools = {"search": lambda args: {}}

        trace_config = load_trace_config({
            "enabled": True,
            "contracts": {
                "tool_payloads": {
                    "tools": {
                        "search": {
                            "args_schema": {
                                "type": "object",
                                "required": ["query"],
                            },
                        },
                    },
                },
            },
        })
        probe = AgentProbe(tools=tools, trace_config=trace_config)
        result = probe.run(model, {"question": "Test"})

        violations = result.metadata.get("custom", {}).get("trace", {}).get("violations", [])
        assert len(violations) > 0

    def test_trace_disabled(self):
        """Test trace can be disabled."""
        model = MockModel(['{"final": "42"}'])
        tools = {}

        trace_config = load_trace_config({"enabled": False})
        probe = AgentProbe(tools=tools, trace_config=trace_config)
        result = probe.run(model, {"question": "Test"})

        # No trace in metadata when disabled
        custom = result.metadata.get("custom", {})
        assert "trace" not in custom or custom.get("trace") is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_probe.py::TestAgentProbeTracing -v`
Expected: FAIL (AgentProbe doesn't support trace_config or _trace_recorder yet)

**Step 3: Write implementation**

```python
# Update insideLLMs/probes/agent.py - replace the class

"""AgentProbe for testing tool-using LLM agents.

This module provides a minimal, framework-agnostic agent probe that:
- Runs a tool-calling loop with configurable max_steps
- Records tool calls and results via TraceRecorder
- Supports trace config for contract validation
- Integrates with the insideLLMs probe infrastructure
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from insideLLMs.probes.base import Probe
from insideLLMs.tracing import TraceRecorder, TraceEventKind, trace_fingerprint
from insideLLMs.trace_config import (
    TraceConfig,
    load_trace_config,
    validate_with_config,
    TracePayloadNormaliser,
)
from insideLLMs.trace_contracts import violations_to_custom_field
from insideLLMs.types import ProbeCategory, ProbeResult, ResultStatus

# Type alias for tool functions
ToolFn = Callable[[dict[str, Any]], Any]


class AgentProbe(Probe[str]):
    """Probe for testing tool-using LLM agents.

    Runs a simple agent loop:
    1. Send prompt to model
    2. Parse response as JSON: {"tool": ..., "args": ...} or {"final": ...}
    3. If tool call, execute tool and loop
    4. If final, return the answer

    Supports trace recording and contract validation via TraceConfig.

    Attributes:
        name: Probe name (default: "agent").
        tools: Dict mapping tool names to callable functions.
        max_steps: Maximum number of agent steps before forcing stop.
        trace_config: Optional trace configuration.
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        tools: dict[str, ToolFn],
        max_steps: int = 4,
        name: str = "AgentProbe",
        trace_config: Optional[TraceConfig] = None,
    ):
        """Initialize the agent probe.

        Args:
            tools: Dict mapping tool names to callable functions.
            max_steps: Maximum number of agent steps (default: 4).
            name: Probe name (default: "AgentProbe").
            trace_config: Optional TraceConfig for tracing/validation.
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)
        self._tools = tools
        self._max_steps = max_steps
        self._trace_config = trace_config

    def run(
        self,
        model: Any,
        data: Any,
        **kwargs: Any,
    ) -> ProbeResult[str]:
        """Run the agent probe.

        Args:
            model: Model with stream(prompt, **kwargs) method.
            data: Input data, expected to have "question" key if dict.
            **kwargs: Additional arguments passed to model.
                - _trace_recorder: Optional TraceRecorder to use.

        Returns:
            ProbeResult with the agent's final answer or error.
        """
        # Handle trace recorder - may be injected or created
        recorder: Optional[TraceRecorder] = kwargs.pop("_trace_recorder", None)

        trace_enabled = bool(
            self._trace_config and self._trace_config.enabled
        )
        if trace_enabled and recorder is None:
            recorder = TraceRecorder()

        # Create normaliser if tracing
        normaliser: Optional[TracePayloadNormaliser] = None
        if trace_enabled and self._trace_config:
            normaliser = TracePayloadNormaliser.from_config(self._trace_config)

        # Extract question from data
        if isinstance(data, dict):
            user_question = data.get("question", str(data))
        else:
            user_question = str(data)

        scratch: list[dict[str, Any]] = []  # Tool results
        final_text: Optional[str] = None

        for step in range(self._max_steps):
            prompt = self._render_prompt(user_question, scratch)

            # Record generate start if tracing
            if recorder:
                recorder.record_generate_start(prompt)

            # Stream response from model
            chunks: list[str] = []
            try:
                for chunk in model.stream(prompt, **kwargs):
                    chunks.append(chunk)
            except Exception as e:
                if recorder:
                    recorder.record_error(str(e), error_type=type(e).__name__)
                return self._build_result(
                    data, None, ResultStatus.ERROR, str(e), recorder, normaliser
                )

            assistant_text = "".join(chunks).strip()

            # Record generate end if tracing
            if recorder:
                recorder.record_generate_end(assistant_text)

            action = self._parse_action_json(assistant_text)

            # Check for final answer
            if "final" in action:
                final_text = str(action["final"])
                break

            # Handle tool call
            tool_name = str(action.get("tool", "unknown"))
            tool_args = action.get("args", {})

            if tool_name not in self._tools:
                error_msg = f"Unknown tool '{tool_name}'"
                if recorder:
                    recorder.record_error(error_msg, error_type="UnknownTool")
                return self._build_result(
                    data, None, ResultStatus.ERROR, error_msg, recorder, normaliser
                )

            # Generate deterministic call ID
            call_id = f"call_{step}_{tool_name}"

            # Record tool call start
            if recorder:
                recorder.record_tool_call(tool_name, tool_args, tool_call_id=call_id)

            # Execute tool
            try:
                tool_result = self._tools[tool_name](tool_args)
            except Exception as e:
                if recorder:
                    recorder.record_tool_result(
                        tool_name, None, tool_call_id=call_id, error=str(e)
                    )
                return self._build_result(
                    data, None, ResultStatus.ERROR,
                    f"Tool '{tool_name}' failed: {e}", recorder, normaliser
                )

            # Canonicalise result
            tool_result_canon = self._canonicalise(tool_result)

            # Record tool result
            if recorder:
                recorder.record_tool_result(
                    tool_name, tool_result_canon, tool_call_id=call_id
                )

            scratch.append({"tool": tool_name, "result": tool_result_canon})

        if final_text is None:
            final_text = "No final answer produced."

        return self._build_result(
            data, final_text, ResultStatus.SUCCESS, None, recorder, normaliser
        )

    def _build_result(
        self,
        data: Any,
        output: Optional[str],
        status: ResultStatus,
        error: Optional[str],
        recorder: Optional[TraceRecorder],
        normaliser: Optional[TracePayloadNormaliser],
    ) -> ProbeResult[str]:
        """Build ProbeResult with optional trace metadata.

        Args:
            data: Original input data.
            output: Output text (if success).
            status: Result status.
            error: Error message (if error).
            recorder: Optional TraceRecorder with events.
            normaliser: Optional normaliser for payloads.

        Returns:
            ProbeResult with trace info in metadata if applicable.
        """
        metadata: dict[str, Any] = {}

        if recorder and self._trace_config and self._trace_config.enabled:
            events = recorder.events

            # Validate contracts
            violations = validate_with_config(events, self._trace_config)

            # Build trace custom field
            trace_custom: dict[str, Any] = {
                "fingerprint": trace_fingerprint(events),
                "violations": violations_to_custom_field(violations),
                "tool_sequence": recorder.get_tool_sequence(),
                "event_count": len(events),
            }

            # Optionally include full events
            store_cfg = self._trace_config.store
            if store_cfg.mode == "full":
                event_dicts = [e.to_dict() for e in events]
                # Apply normalisation to payloads
                if normaliser:
                    for ed in event_dicts:
                        ed["payload"] = normaliser.normalise(ed.get("payload", {}))
                # Truncate if needed
                if store_cfg.max_events > 0:
                    event_dicts = event_dicts[:store_cfg.max_events]
                trace_custom["events"] = event_dicts

            metadata["custom"] = {"trace": trace_custom}

        return ProbeResult(
            input=data,
            output=output,
            status=status,
            error=error,
            metadata=metadata,
        )

    def _render_prompt(
        self,
        question: str,
        scratch: list[dict[str, Any]],
    ) -> str:
        """Render the agent prompt."""
        tool_state = json.dumps(scratch, sort_keys=True, ensure_ascii=False)
        return (
            "You are an agent. You must respond ONLY with a single JSON object.\n"
            "Either:\n"
            '  {"tool": "<name>", "args": {...}}\n'
            'or {"final": "<answer>"}\n\n'
            f"Question: {question}\n"
            f"Tool state (JSON): {tool_state}\n"
        )

    def _parse_action_json(self, text: str) -> dict[str, Any]:
        """Parse JSON action from model response."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"final": text}

        blob = text[start : end + 1]
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            return {"final": text}

        if isinstance(obj, dict) and ("tool" in obj or "final" in obj):
            return obj
        return {"final": text}

    def _canonicalise(self, value: Any) -> Any:
        """Canonicalise a value for deterministic storage."""
        try:
            s = json.dumps(value, sort_keys=True, ensure_ascii=False)
            return json.loads(s)
        except (TypeError, ValueError):
            return value
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_probe.py::TestAgentProbeTracing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add insideLLMs/probes/agent.py tests/test_agent_probe.py
git commit -m "feat(probes): add trace integration to AgentProbe"
```

---

## Task 8: Export AgentProbe in __init__.py

**Files:**
- Modify: `insideLLMs/probes/__init__.py:1-83` (add import and export)

**Step 1: Write the failing test**

```python
# Add to tests/test_agent_probe.py

class TestAgentProbeImport:
    """Test AgentProbe is exported correctly."""

    def test_import_from_probes(self):
        """Test AgentProbe can be imported from probes package."""
        from insideLLMs.probes import AgentProbe
        assert AgentProbe is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_probe.py::TestAgentProbeImport -v`
Expected: FAIL with "ImportError: cannot import name 'AgentProbe'"

**Step 3: Write implementation**

```python
# Update insideLLMs/probes/__init__.py - add import and __all__ entry

# Add to imports section (after line 22):
from insideLLMs.probes.agent import AgentProbe

# Add to __all__ list (after "CustomProbe"):
    "AgentProbe",
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_probe.py::TestAgentProbeImport -v`
Expected: PASS

**Step 5: Commit**

```bash
git add insideLLMs/probes/__init__.py
git commit -m "feat(probes): export AgentProbe from probes package"
```

---

## Task 9: Export trace_config from Package

**Files:**
- Modify: `insideLLMs/__init__.py` (if exists, add exports)

**Step 1: Write the failing test**

```python
# Add to tests/test_trace_config.py

class TestTraceConfigImport:
    """Test trace_config exports."""

    def test_import_from_package(self):
        """Test key symbols are importable."""
        from insideLLMs.trace_config import (
            TraceConfig,
            load_trace_config,
            validate_with_config,
            TracePayloadNormaliser,
        )
        assert TraceConfig is not None
        assert load_trace_config is not None
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_trace_config.py::TestTraceConfigImport -v`
Expected: PASS (already importable from trace_config module)

**Step 3: Commit all tests pass**

```bash
pytest tests/test_trace_config.py tests/test_agent_probe.py -v
git add .
git commit -m "test: verify all trace_config and AgentProbe tests pass"
```

---

## Task 10: Full Integration Test

**Files:**
- Create: `tests/test_trace_integration.py`

**Step 1: Write integration test**

```python
# tests/test_trace_integration.py
"""Integration tests for trace config + AgentProbe."""

import pytest

from insideLLMs.probes.agent import AgentProbe
from insideLLMs.trace_config import load_trace_config, TraceConfig
from insideLLMs.trace_contracts import ViolationCode
from insideLLMs.types import ResultStatus


class MockStreamModel:
    """Mock model that streams responses."""

    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    def stream(self, prompt: str, **kwargs):
        try:
            response = next(self._responses)
        except StopIteration:
            response = '{"final": "exhausted"}'
        for char in response:
            yield char


class TestTraceIntegration:
    """Integration tests for full trace pipeline."""

    def test_full_pipeline_with_contracts(self):
        """Test full pipeline: config -> agent -> validation -> result."""
        # Configure trace with tool schema
        config = load_trace_config({
            "enabled": True,
            "store": {"mode": "full", "max_events": 100},
            "contracts": {
                "enabled": True,
                "tool_payloads": {
                    "tools": {
                        "search": {
                            "args_schema": {
                                "type": "object",
                                "required": ["query"],
                                "properties": {
                                    "query": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "tool_order": {
                    "must_follow": {"summarise": ["search"]},
                },
            },
            "on_violation": {"mode": "record"},
        })

        # Create model that follows correct order
        model = MockStreamModel([
            '{"tool": "search", "args": {"query": "test"}}',
            '{"tool": "summarise", "args": {"text": "results"}}',
            '{"final": "Summary complete"}',
        ])

        tools = {
            "search": lambda args: {"results": ["a", "b"]},
            "summarise": lambda args: {"summary": "done"},
        }

        probe = AgentProbe(tools=tools, trace_config=config, max_steps=5)
        result = probe.run(model, {"question": "Search and summarise"})

        assert result.status == ResultStatus.SUCCESS
        assert "Summary complete" in result.output

        # Check trace metadata
        trace = result.metadata["custom"]["trace"]
        assert trace["fingerprint"].startswith("sha256:")
        assert trace["tool_sequence"] == ["search", "summarise"]
        assert len(trace["violations"]) == 0  # No violations expected

    def test_catches_tool_order_violation(self):
        """Test that tool order violations are caught."""
        config = load_trace_config({
            "enabled": True,
            "contracts": {
                "tool_order": {
                    "must_follow": {"summarise": ["search"]},
                },
            },
        })

        # Model calls summarise BEFORE search (violation)
        model = MockStreamModel([
            '{"tool": "summarise", "args": {}}',
            '{"tool": "search", "args": {}}',
            '{"final": "Done"}',
        ])

        tools = {
            "search": lambda args: {},
            "summarise": lambda args: {},
        }

        probe = AgentProbe(tools=tools, trace_config=config, max_steps=5)
        result = probe.run(model, {"question": "Wrong order"})

        violations = result.metadata["custom"]["trace"]["violations"]
        assert len(violations) > 0
        assert any(v["code"] == ViolationCode.TOOL_ORDER_VIOLATION.value for v in violations)

    def test_catches_missing_required_arg(self):
        """Test that missing required args are caught."""
        config = load_trace_config({
            "enabled": True,
            "contracts": {
                "tool_payloads": {
                    "tools": {
                        "search": {
                            "args_schema": {
                                "type": "object",
                                "required": ["query"],
                            },
                        },
                    },
                },
            },
        })

        # Model calls search without required "query" arg
        model = MockStreamModel([
            '{"tool": "search", "args": {"limit": 10}}',
            '{"final": "Done"}',
        ])

        tools = {"search": lambda args: {}}

        probe = AgentProbe(tools=tools, trace_config=config)
        result = probe.run(model, {"question": "Missing arg"})

        violations = result.metadata["custom"]["trace"]["violations"]
        assert len(violations) > 0
        assert any(
            v["code"] == ViolationCode.TOOL_MISSING_REQUIRED_ARG.value
            for v in violations
        )

    def test_redaction_applied(self):
        """Test that redaction is applied to stored events."""
        config = load_trace_config({
            "enabled": True,
            "store": {
                "mode": "full",
                "redact": {
                    "enabled": True,
                    "json_pointers": ["/secret"],
                    "replacement": "[REDACTED]",
                },
            },
        })

        model = MockStreamModel([
            '{"tool": "auth", "args": {"secret": "password123"}}',
            '{"final": "Done"}',
        ])

        tools = {"auth": lambda args: {"token": "abc"}}

        probe = AgentProbe(tools=tools, trace_config=config)
        result = probe.run(model, {"question": "Authenticate"})

        events = result.metadata["custom"]["trace"]["events"]
        # Find tool_call_start event
        tool_event = next(
            (e for e in events if e["kind"] == "tool_call_start"), None
        )
        assert tool_event is not None
        # Secret should be redacted in arguments
        assert tool_event["payload"]["arguments"]["secret"] == "[REDACTED]"
```

**Step 2: Run integration test**

Run: `pytest tests/test_trace_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_trace_integration.py
git commit -m "test: add integration tests for trace config + AgentProbe"
```

---

## Task 11: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/test_trace_config.py tests/test_agent_probe.py tests/test_trace_integration.py tests/test_trace_contracts.py tests/test_tracing.py -v`
Expected: All PASS

**Step 2: Final commit**

```bash
git add .
git commit -m "feat: complete trace config and AgentProbe implementation"
```

---

## Summary

**Files Created:**
- `insideLLMs/trace_config.py` - TraceConfig, load_trace_config, to_contracts, TracePayloadNormaliser, validate_with_config
- `insideLLMs/probes/agent.py` - AgentProbe with trace integration
- `tests/test_trace_config.py` - Config tests
- `tests/test_agent_probe.py` - AgentProbe tests
- `tests/test_trace_integration.py` - Integration tests

**Files Modified:**
- `insideLLMs/probes/__init__.py` - Export AgentProbe

**Key Components:**
1. `TraceConfig` - Public config surface matching YAML schema
2. `load_trace_config()` - YAML dict → TraceConfig
3. `TraceConfig.to_contracts()` - Compiles to ToolSchema, ToolOrderRule, toggles
4. `TracePayloadNormaliser` - Redaction + canonicalisation
5. `validate_with_config()` - Run validators with config toggles
6. `AgentProbe` - Tool-using agent with trace recording
