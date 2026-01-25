"""Tests for insideLLMs.trace_config module."""

import pytest

from insideLLMs.trace_config import (
    FingerprintConfig,
    NormaliserConfig,
    NormaliserKind,
    OnViolationMode,
    StoreMode,
    TraceConfig,
    TraceContractsConfig,
    TracePayloadNormaliser,
    TraceRedactConfig,
    TraceStoreConfig,
    load_trace_config,
    make_structural_v1_normaliser,
    validate_with_config,
)
from insideLLMs.trace_contracts import ToolOrderRule, ToolSchema, ViolationCode
from insideLLMs.tracing import TraceEvent


class TestTraceConfigStructure:
    """Tests for TraceConfig dataclass structure."""

    def test_default_config(self):
        """Test TraceConfig with defaults."""
        config = TraceConfig()
        assert config.version == 1
        assert config.enabled is True
        assert config.store.mode == StoreMode.FULL
        assert config.contracts.enabled is True
        assert config.contracts.fail_fast is False
        assert config.on_violation.mode == OnViolationMode.RECORD

    def test_store_config_defaults(self):
        """Test TraceStoreConfig defaults."""
        store = TraceStoreConfig()
        assert store.mode == StoreMode.FULL
        assert store.max_events is None  # Unlimited by default
        assert store.include_payloads is True
        assert store.redact.enabled is False

    def test_fingerprint_config_defaults(self):
        """Test FingerprintConfig defaults."""
        config = TraceConfig()
        assert config.fingerprint.enabled is True
        assert config.fingerprint.algorithm == "sha256"
        assert config.fingerprint.normaliser.kind == NormaliserKind.BUILTIN
        assert config.fingerprint.normaliser.name == "structural_v1"
        assert "request_id" in config.fingerprint.normaliser.drop_keys
        assert config.fingerprint.normaliser.hash_strings_over == 512

    def test_normaliser_config_defaults(self):
        """Test NormaliserConfig defaults."""
        normaliser = NormaliserConfig()
        assert normaliser.kind == NormaliserKind.BUILTIN
        assert normaliser.name == "structural_v1"
        assert normaliser.import_path is None
        assert "timestamp" in normaliser.drop_keys
        assert normaliser.hash_paths == ["result", "raw"]
        assert normaliser.hash_strings_over == 512

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
        assert config.store.mode == StoreMode.COMPACT  # "fingerprint" maps to COMPACT
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
        config = load_trace_config(
            {
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
            }
        )
        result = config.to_contracts()
        schemas = result["tool_schemas"]
        assert "search" in schemas
        assert isinstance(schemas["search"], ToolSchema)
        assert "query" in schemas["search"].required_args
        assert "top_k" in schemas["search"].required_args

    def test_to_contracts_with_tool_order(self):
        """Test compilation with tool order rules."""
        config = load_trace_config(
            {
                "contracts": {
                    "tool_order": {
                        "must_follow": {"summarise": ["search"]},
                        "must_precede": {"search": ["summarise"]},
                        "forbidden_sequences": [["send_email", "search"]],
                    },
                },
            }
        )
        result = config.to_contracts()
        rules = result["tool_order_rules"]
        assert isinstance(rules, ToolOrderRule)
        assert rules.must_follow == {"summarise": ["search"]}
        assert rules.forbidden_sequences == [["send_email", "search"]]

    def test_to_contracts_toggles(self):
        """Test toggles are set correctly."""
        config = load_trace_config(
            {
                "contracts": {
                    "generate_boundaries": {"enabled": False},
                    "stream_boundaries": {"enabled": True},
                },
            }
        )
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


class TestTracePayloadNormaliser:
    """Tests for TracePayloadNormaliser class."""

    def test_normalise_no_redaction(self):
        """Test normalise without redaction just canonicalises."""
        normaliser = TracePayloadNormaliser()
        payload = {"b": 2, "a": 1}
        result = normaliser.normalise(payload)
        # Should be canonicalised (sorted keys)
        assert list(result.keys()) == ["a", "b"]
        assert result == {"a": 1, "b": 2}

    def test_normalise_with_redaction(self):
        """Test normalise with redaction enabled."""
        normaliser = TracePayloadNormaliser(
            redact_enabled=True,
            json_pointers=["/headers/authorization"],
            replacement="<redacted>",
        )
        payload = {
            "headers": {"authorization": "Bearer secret123", "content-type": "json"},
            "body": "data",
        }
        result = normaliser.normalise(payload)
        assert result["headers"]["authorization"] == "<redacted>"
        assert result["headers"]["content-type"] == "json"
        assert result["body"] == "data"

    def test_normalise_nested_redaction(self):
        """Test redaction with nested paths."""
        normaliser = TracePayloadNormaliser(
            redact_enabled=True,
            json_pointers=["/deep/nested/secret"],
        )
        payload = {"deep": {"nested": {"secret": "xyz", "public": "abc"}}}
        result = normaliser.normalise(payload)
        assert result["deep"]["nested"]["secret"] == "<redacted>"
        assert result["deep"]["nested"]["public"] == "abc"

    def test_normalise_missing_path_no_error(self):
        """Test redaction path that doesn't exist doesn't error."""
        normaliser = TracePayloadNormaliser(
            redact_enabled=True,
            json_pointers=["/does/not/exist"],
        )
        payload = {"data": "value"}
        result = normaliser.normalise(payload)
        assert result == {"data": "value"}

    def test_normalise_canonicalises_json(self):
        """Test that output is deterministic."""
        normaliser = TracePayloadNormaliser()
        payload1 = {"z": 1, "a": 2, "m": 3}
        payload2 = {"a": 2, "m": 3, "z": 1}
        result1 = normaliser.normalise(payload1)
        result2 = normaliser.normalise(payload2)
        # Should produce identical output
        import json

        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_from_config(self):
        """Test creating normaliser from TraceConfig."""
        config = load_trace_config(
            {
                "store": {
                    "redact": {
                        "enabled": True,
                        "json_pointers": ["/secret"],
                        "replacement": "[HIDDEN]",
                    },
                },
            }
        )
        normaliser = TracePayloadNormaliser.from_config(config)
        payload = {"secret": "password", "public": "data"}
        result = normaliser.normalise(payload)
        assert result["secret"] == "[HIDDEN]"
        assert result["public"] == "data"


class TestValidateWithConfig:
    """Tests for validate_with_config helper."""

    def test_disabled_config_returns_empty(self):
        """Test disabled config returns no violations."""
        config = load_trace_config({"enabled": False})
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            # Missing generate_end - would be violation if enabled
        ]
        violations = validate_with_config(events, config)
        assert violations == []

    def test_disabled_contracts_returns_empty(self):
        """Test disabled contracts returns no violations."""
        config = load_trace_config({"contracts": {"enabled": False}})
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
        ]
        violations = validate_with_config(events, config)
        assert violations == []

    def test_respects_toggle_disabled(self):
        """Test toggle disables specific validator."""
        config = load_trace_config(
            {
                "contracts": {
                    "generate_boundaries": {"enabled": False},
                    "stream_boundaries": {"enabled": True},
                },
            }
        )
        events = [
            # generate_start without end - should NOT cause violation (disabled)
            TraceEvent(seq=0, kind="generate_start", payload={}),
            # stream_start without end - SHOULD cause violation (enabled)
            TraceEvent(seq=1, kind="stream_start", payload={}),
        ]
        violations = validate_with_config(events, config)
        # Should only have stream violation
        codes = [v.code for v in violations]
        assert ViolationCode.GENERATE_NO_END.value not in codes
        assert ViolationCode.STREAM_NO_END.value in codes

    def test_all_validators_enabled(self):
        """Test all validators run when enabled."""
        config = load_trace_config({})  # Defaults to all enabled
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            TraceEvent(seq=1, kind="stream_start", payload={}),
            TraceEvent(seq=2, kind="tool_call_start", payload={"tool_name": "test"}),
            # All unclosed
        ]
        violations = validate_with_config(events, config)
        codes = [v.code for v in violations]
        assert ViolationCode.GENERATE_NO_END.value in codes
        assert ViolationCode.STREAM_NO_END.value in codes
        assert ViolationCode.TOOL_NO_RESULT.value in codes

    def test_valid_trace_no_violations(self):
        """Test valid trace returns no violations."""
        config = load_trace_config({})
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={"prompt": "Hi"}),
            TraceEvent(seq=1, kind="generate_end", payload={"response": "Hello"}),
        ]
        violations = validate_with_config(events, config)
        assert violations == []

    def test_fail_fast_stops_early(self):
        """Test fail_fast option stops on first violation."""
        config = load_trace_config(
            {
                "contracts": {
                    "fail_fast": True,
                },
            }
        )
        events = [
            # Multiple violations: missing generate_end, stream_end, tool_result
            TraceEvent(seq=0, kind="generate_start", payload={}),
            TraceEvent(seq=1, kind="stream_start", payload={}),
            TraceEvent(seq=2, kind="tool_call_start", payload={"tool_name": "test"}),
        ]
        violations = validate_with_config(events, config)
        # Should stop after first violation
        assert len(violations) == 1
        assert violations[0].code == ViolationCode.GENERATE_NO_END.value

    def test_fail_fast_disabled_finds_all(self):
        """Test fail_fast=False finds all violations."""
        config = load_trace_config(
            {
                "contracts": {
                    "fail_fast": False,
                },
            }
        )
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            TraceEvent(seq=1, kind="stream_start", payload={}),
            TraceEvent(seq=2, kind="tool_call_start", payload={"tool_name": "test"}),
        ]
        violations = validate_with_config(events, config)
        # Should find all violations
        assert len(violations) >= 3

    def test_respects_custom_event_kinds_and_payload_keys(self):
        """validate_with_config should honor TraceConfig kind/key mappings."""
        config = load_trace_config(
            {
                "contracts": {
                    "generate_boundaries": {
                        "start_kind": "completion_start",
                        "end_kind": "completion_end",
                    },
                    "stream_boundaries": {
                        "start_kind": "sse_open",
                        "chunk_kind": "sse_message",
                        "end_kind": "sse_close",
                        "chunk_index_key": "message_index",
                        "first_chunk_index": 1,
                    },
                    "tool_results": {
                        "call_start_kind": "function_invoke",
                        "call_result_kind": "function_return",
                    },
                    "tool_payloads": {
                        "tool_key": "tool",
                        "args_key": "args",
                        "tools": {
                            "search": {
                                "args_schema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                }
                            }
                        },
                    },
                }
            }
        )
        events = [
            TraceEvent(seq=0, kind="completion_start", payload={"prompt": "hi"}),
            TraceEvent(seq=1, kind="completion_end", payload={"response": "ok"}),
            TraceEvent(seq=2, kind="sse_open", payload={"prompt": "x"}),
            TraceEvent(seq=3, kind="sse_message", payload={"chunk": "a", "message_index": 1}),
            TraceEvent(seq=4, kind="sse_message", payload={"chunk": "b", "message_index": 2}),
            TraceEvent(seq=5, kind="sse_close", payload={}),
            # Tool call uses custom kind + custom payload key names, and is missing required arg + result.
            TraceEvent(seq=6, kind="function_invoke", payload={"tool": "search", "args": {}}),
        ]
        violations = validate_with_config(events, config)
        codes = {v.code for v in violations}
        assert ViolationCode.TOOL_MISSING_REQUIRED_ARG.value in codes
        assert ViolationCode.TOOL_NO_RESULT.value in codes


class TestEnhancedNormaliser:
    """Tests for enhanced TracePayloadNormaliser features."""

    def test_drop_keys(self):
        """Test drop_keys removes exact key matches."""
        normaliser = TracePayloadNormaliser(
            drop_keys=["timestamp", "request_id"],
        )
        payload = {
            "data": "value",
            "timestamp": "2025-01-01T00:00:00Z",
            "request_id": "abc123",
        }
        result = normaliser.normalise(payload)
        assert "data" in result
        assert "timestamp" not in result
        assert "request_id" not in result

    def test_drop_key_regex(self):
        """Test drop_key_regex removes regex-matched keys."""
        normaliser = TracePayloadNormaliser(
            drop_key_regex=[r"^x-.*", r".*_id$"],
        )
        payload = {
            "data": "value",
            "x-request-id": "abc",
            "x-custom-header": "xyz",
            "user_id": "123",
            "name": "test",
        }
        result = normaliser.normalise(payload)
        assert "data" in result
        assert "name" in result
        assert "x-request-id" not in result
        assert "x-custom-header" not in result
        assert "user_id" not in result

    def test_hash_paths(self):
        """Test hash_paths hashes values at specified paths."""
        normaliser = TracePayloadNormaliser(
            hash_paths=["result"],
        )
        payload = {
            "data": "small",
            "result": {"large": "content" * 100},
        }
        result = normaliser.normalise(payload)
        assert result["data"] == "small"
        assert "sha256" in result["result"]
        assert "len" in result["result"]

    def test_hash_strings_over(self):
        """Test hash_strings_over hashes long strings."""
        normaliser = TracePayloadNormaliser(
            hash_strings_over=10,
        )
        payload = {
            "short": "abc",
            "long": "this is a very long string that exceeds the limit",
        }
        result = normaliser.normalise(payload)
        assert result["short"] == "abc"
        assert "sha256" in result["long"]

    def test_stream_chunk_normalisation(self):
        """Test stream_chunk events get special handling."""
        normaliser = TracePayloadNormaliser()
        payload = {
            "text": "Hello, this is streaming content",
            "chunk_index": 0,
        }
        result = normaliser.normalise(payload, kind="stream_chunk")
        # Text should be replaced with length
        assert "text" not in result
        assert "text_len" in result
        assert result["text_len"] == 32
        assert result["chunk_index"] == 0

    def test_stream_chunk_with_chunk_key(self):
        """Test stream_chunk normalisation with 'chunk' key."""
        normaliser = TracePayloadNormaliser()
        payload = {
            "chunk": "Content chunk",
            "index": 1,
        }
        result = normaliser.normalise(payload, kind="stream_chunk")
        assert "chunk" not in result
        assert "text_len" in result
        assert result["text_len"] == 13

    def test_callable_interface(self):
        """Test normaliser can be called as TracePayloadNormaliserFunc."""
        normaliser = TracePayloadNormaliser(
            drop_keys=["noise"],
        )
        # Use __call__ interface
        result = normaliser("custom_event", {"data": "value", "noise": "xyz"})
        assert "data" in result
        assert "noise" not in result


class TestMakeStructuralV1Normaliser:
    """Tests for make_structural_v1_normaliser factory."""

    def test_default_configuration(self):
        """Test default factory configuration."""
        normaliser = make_structural_v1_normaliser()
        # Test default drop_keys
        payload = {
            "data": "value",
            "timestamp": "123",
            "request_id": "abc",
            "latency_ms": 100,
        }
        result = normaliser.normalise(payload)
        assert "data" in result
        assert "timestamp" not in result
        assert "request_id" not in result
        assert "latency_ms" not in result

    def test_default_regex_patterns(self):
        """Test default regex patterns."""
        normaliser = make_structural_v1_normaliser()
        payload = {
            "data": "value",
            "x-custom-header": "xyz",
            "foo_request_id": "abc",
        }
        result = normaliser.normalise(payload)
        assert "data" in result
        assert "x-custom-header" not in result
        assert "foo_request_id" not in result

    def test_custom_configuration(self):
        """Test factory with custom configuration."""
        normaliser = make_structural_v1_normaliser(
            drop_keys=["custom_noise"],
            hash_strings_over=5,
        )
        payload = {
            "data": "short",
            "longer": "this is longer",
            "custom_noise": "remove me",
        }
        result = normaliser.normalise(payload)
        assert result["data"] == "short"
        assert "sha256" in result["longer"]
        assert "custom_noise" not in result


class TestVersionAndFailFast:
    """Tests for version field and fail_fast option."""

    def test_version_in_config(self):
        """Test version field in config."""
        config = load_trace_config({"version": 2})
        assert config.version == 2

    def test_version_defaults_to_1(self):
        """Test version defaults to 1."""
        config = load_trace_config({})
        assert config.version == 1

    def test_fail_fast_in_contracts(self):
        """Test fail_fast is loaded correctly."""
        config = load_trace_config(
            {
                "contracts": {"fail_fast": True},
            }
        )
        assert config.contracts.fail_fast is True

    def test_fail_fast_in_to_contracts(self):
        """Test fail_fast is compiled to contracts."""
        config = load_trace_config(
            {
                "contracts": {"fail_fast": True},
            }
        )
        result = config.to_contracts()
        assert result["fail_fast"] is True
