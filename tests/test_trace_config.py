"""Tests for insideLLMs.trace_config module."""

import pytest

from insideLLMs.trace_config import (
    TraceConfig,
    TraceStoreConfig,
    TraceContractsConfig,
    TraceRedactConfig,
    OnViolationMode,
    StoreMode,
    load_trace_config,
)


class TestTraceConfigStructure:
    """Tests for TraceConfig dataclass structure."""

    def test_default_config(self):
        """Test TraceConfig with defaults."""
        config = TraceConfig()
        assert config.enabled is True
        assert config.store.mode == StoreMode.FULL
        assert config.contracts.enabled is True
        assert config.on_violation.mode == OnViolationMode.RECORD

    def test_store_config_defaults(self):
        """Test TraceStoreConfig defaults."""
        store = TraceStoreConfig()
        assert store.mode == StoreMode.FULL
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


class TestLoadTraceConfig:
    """Tests for load_trace_config function."""

    def test_load_minimal_config(self):
        """Test loading minimal config dict."""
        yaml_dict = {"enabled": True}
        config = load_trace_config(yaml_dict)
        assert config.enabled is True
        assert config.store.mode == StoreMode.FULL  # default

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
        assert config.store.mode == StoreMode.FINGERPRINT
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
