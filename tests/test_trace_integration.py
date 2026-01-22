"""Integration tests for trace configuration and AgentProbe.

These tests verify the complete flow from YAML config through
validation and AgentProbe execution.
"""

import pytest
from typing import Any

from insideLLMs import (
    AgentProbe,
    AgentProbeResult,
    ToolDefinition,
    TraceConfig,
    load_trace_config,
    validate_with_config,
    TracePayloadNormaliser,
    OnViolationMode,
    StoreMode,
)
from insideLLMs.tracing import TraceRecorder, TraceEvent, trace_fingerprint
from insideLLMs.trace_contracts import ViolationCode


class MockAgentModel:
    """Mock model that simulates an agent with tool use."""

    def __init__(self, tool_sequence: list[str] = None, response: str = "Done"):
        self.tool_sequence = tool_sequence or []
        self.response = response
        self.calls = []

    def run_with_tools(self, prompt: str, tools: list[ToolDefinition]) -> str:
        self.calls.append({"prompt": prompt, "tools": [t.name for t in tools]})
        return self.response


class SearchSummarizeAgentProbe(AgentProbe):
    """Agent probe that simulates search-then-summarize workflow."""

    def run_agent(
        self,
        model: Any,
        prompt: str,
        tools: list[ToolDefinition],
        recorder: TraceRecorder,
        **kwargs,
    ) -> str:
        # Record generation start
        recorder.record_generate_start(prompt)

        # Simulate search tool call
        search_tool = self.get_tool_by_name("search")
        if search_tool:
            recorder.record_tool_call("search", {"query": prompt}, tool_call_id="call_1")
            recorder.record_tool_result(
                "search",
                {"results": ["result1", "result2"]},
                tool_call_id="call_1",
            )

        # Simulate summarize tool call
        summarize_tool = self.get_tool_by_name("summarize")
        if summarize_tool:
            recorder.record_tool_call(
                "summarize",
                {"text": "result1 result2"},
                tool_call_id="call_2",
            )
            recorder.record_tool_result(
                "summarize",
                {"summary": "Combined results"},
                tool_call_id="call_2",
            )

        # Get final response
        response = model.run_with_tools(prompt, tools)

        # Record generation end
        recorder.record_generate_end(response)

        return response


class TestFullIntegration:
    """Full integration tests for trace config and AgentProbe."""

    def test_complete_workflow_no_violations(self):
        """Test complete workflow with valid trace produces no violations."""
        # Define tools
        tools = [
            ToolDefinition(
                name="search",
                description="Search for information",
                parameters={
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            ),
            ToolDefinition(
                name="summarize",
                description="Summarize text",
                parameters={
                    "type": "object",
                    "required": ["text"],
                    "properties": {"text": {"type": "string"}},
                },
            ),
        ]

        # Load config with tool order rules
        config_dict = {
            "enabled": True,
            "contracts": {
                "enabled": True,
                "tool_order": {
                    "must_follow": {"summarize": ["search"]},
                },
                "tool_payloads": {
                    "tools": {
                        "search": {
                            "args_schema": {
                                "type": "object",
                                "required": ["query"],
                                "properties": {"query": {"type": "string"}},
                            },
                        },
                        "summarize": {
                            "args_schema": {
                                "type": "object",
                                "required": ["text"],
                                "properties": {"text": {"type": "string"}},
                            },
                        },
                    },
                },
            },
        }

        # Create probe
        model = MockAgentModel()
        probe = SearchSummarizeAgentProbe(
            name="search_summarize",
            tools=tools,
            trace_config=config_dict,
        )

        # Run probe
        result = probe.run(model, "Find and summarize Python tutorials")

        # Assertions
        assert result.final_response == "Done"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["tool_name"] == "search"
        assert result.tool_calls[1]["tool_name"] == "summarize"
        assert result.violations == []  # No violations
        assert result.trace_fingerprint.startswith("sha256:")

    def test_tool_order_violation_detected(self):
        """Test that tool order violations are detected."""

        class BadOrderAgentProbe(AgentProbe):
            """Probe that calls summarize before search."""

            def run_agent(self, model, prompt, tools, recorder, **kwargs):
                recorder.record_generate_start(prompt)
                # Call summarize first (violates must_follow)
                recorder.record_tool_call("summarize", {"text": "test"})
                recorder.record_tool_result("summarize", {"summary": "test"})
                # Then search
                recorder.record_tool_call("search", {"query": prompt})
                recorder.record_tool_result("search", {"results": []})
                recorder.record_generate_end("Done")
                return "Done"

        config_dict = {
            "contracts": {
                "tool_order": {
                    "must_follow": {"summarize": ["search"]},
                },
            },
        }

        probe = BadOrderAgentProbe(
            name="bad_order",
            tools=[],
            trace_config=config_dict,
        )

        result = probe.run(MockAgentModel(), "test")

        # Should have tool order violation
        assert len(result.violations) > 0
        codes = [v["code"] for v in result.violations]
        assert ViolationCode.TOOL_ORDER_VIOLATION.value in codes

    def test_payload_validation_violation(self):
        """Test that missing required arguments are detected."""

        class MissingArgAgentProbe(AgentProbe):
            """Probe that calls tool without required argument."""

            def run_agent(self, model, prompt, tools, recorder, **kwargs):
                recorder.record_generate_start(prompt)
                # Missing 'query' argument
                recorder.record_tool_call("search", {})
                recorder.record_tool_result("search", {})
                recorder.record_generate_end("Done")
                return "Done"

        config_dict = {
            "contracts": {
                "tool_payloads": {
                    "tools": {
                        "search": {
                            "args_schema": {
                                "type": "object",
                                "required": ["query"],
                                "properties": {"query": {"type": "string"}},
                            },
                        },
                    },
                },
            },
        }

        probe = MissingArgAgentProbe(
            name="missing_arg",
            tools=[],
            trace_config=config_dict,
        )

        result = probe.run(MockAgentModel(), "test")

        # Should have missing required arg violation
        codes = [v["code"] for v in result.violations]
        assert ViolationCode.TOOL_MISSING_REQUIRED_ARG.value in codes

    def test_redaction_in_trace(self):
        """Test that sensitive data is redacted in trace events."""

        class SensitiveDataProbe(AgentProbe):
            """Probe that includes sensitive data in payloads."""

            def run_agent(self, model, prompt, tools, recorder, **kwargs):
                recorder.record_generate_start(prompt)
                recorder.record_tool_call(
                    "api_call",
                    {"url": "https://api.example.com", "api_key": "secret123"},
                )
                recorder.record_tool_result(
                    "api_call",
                    {"data": "response", "api_key": "secret123"},
                )
                recorder.record_generate_end("Done")
                return "Done"

        config_dict = {
            "store": {
                "redact": {
                    "enabled": True,
                    "json_pointers": ["/api_key"],
                    "replacement": "[REDACTED]",
                },
            },
        }

        probe = SensitiveDataProbe(
            name="sensitive",
            tools=[],
            trace_config=config_dict,
        )

        result = probe.run(MockAgentModel(), "test")

        # Check that api_key is redacted in all events
        for event in result.trace_events:
            payload = event.get("payload", {})
            if "api_key" in payload:
                assert payload["api_key"] == "[REDACTED]"

    def test_fail_probe_mode(self):
        """Test that fail_probe mode raises exception on violation."""

        class IncompleteProbe(AgentProbe):
            """Probe with incomplete trace."""

            def run_agent(self, model, prompt, tools, recorder, **kwargs):
                recorder.record_generate_start(prompt)
                # Missing generate_end!
                return "Done"

        config_dict = {
            "on_violation": {"mode": "fail_probe"},
        }

        probe = IncompleteProbe(
            name="incomplete",
            tools=[],
            trace_config=config_dict,
        )

        with pytest.raises(ValueError, match="Trace contract violations"):
            probe.run(MockAgentModel(), "test")

    def test_fingerprint_determinism(self):
        """Test that same trace produces same fingerprint."""

        class DeterministicProbe(AgentProbe):
            def run_agent(self, model, prompt, tools, recorder, **kwargs):
                recorder.record_generate_start(prompt)
                recorder.record_tool_call("tool", {"arg": "value"})
                recorder.record_tool_result("tool", {"result": "data"})
                recorder.record_generate_end("response")
                return "response"

        probe = DeterministicProbe(name="deterministic", tools=[])
        model = MockAgentModel()

        # Run twice
        result1 = probe.run(model, "same prompt")
        result2 = probe.run(model, "same prompt")

        # Fingerprints should be identical
        assert result1.trace_fingerprint == result2.trace_fingerprint

    def test_validate_with_config_direct(self):
        """Test validate_with_config function directly."""
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={"prompt": "Hi"}),
            TraceEvent(seq=1, kind="stream_start", payload={}),
            # Missing stream_end and generate_end
        ]

        config = load_trace_config({
            "contracts": {
                "generate_boundaries": {"enabled": True},
                "stream_boundaries": {"enabled": True},
            },
        })

        violations = validate_with_config(events, config)

        codes = [v.code for v in violations]
        assert ViolationCode.GENERATE_NO_END.value in codes
        assert ViolationCode.STREAM_NO_END.value in codes

    def test_config_toggles_respected(self):
        """Test that disabled validators don't produce violations."""
        events = [
            TraceEvent(seq=0, kind="generate_start", payload={}),
            # Missing generate_end - but generate_boundaries is disabled
        ]

        config = load_trace_config({
            "contracts": {
                "generate_boundaries": {"enabled": False},
            },
        })

        violations = validate_with_config(events, config)

        # Should not have generate violation since it's disabled
        codes = [v.code for v in violations]
        assert ViolationCode.GENERATE_NO_END.value not in codes


class TestYAMLConfigLoading:
    """Tests for loading YAML-style config dicts."""

    def test_full_yaml_style_config(self):
        """Test loading a complete YAML-style configuration."""
        yaml_dict = {
            "enabled": True,
            "store": {
                "mode": "full",
                "max_events": 10000,
                "include_payloads": True,
                "redact": {
                    "enabled": True,
                    "json_pointers": [
                        "/payload/headers/authorization",
                        "/payload/api_key",
                    ],
                    "replacement": "<redacted>",
                },
            },
            "contracts": {
                "enabled": True,
                "generate_boundaries": {"enabled": True},
                "stream_boundaries": {"enabled": True},
                "tool_results": {"enabled": True},
                "tool_payloads": {
                    "enabled": True,
                    "tools": {
                        "search": {
                            "args_schema": {
                                "type": "object",
                                "required": ["query"],
                                "properties": {
                                    "query": {"type": "string"},
                                    "limit": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
                "tool_order": {
                    "enabled": True,
                    "must_follow": {"summarize": ["search", "retrieve"]},
                    "must_precede": {"search": ["summarize"]},
                    "forbidden_sequences": [["delete", "commit"]],
                },
            },
            "on_violation": {"mode": "fail_probe"},
        }

        config = load_trace_config(yaml_dict)

        # Verify all settings loaded correctly
        assert config.enabled is True
        assert config.store.mode == StoreMode.FULL
        assert config.store.max_events == 10000
        assert config.store.redact.enabled is True
        assert len(config.store.redact.json_pointers) == 2

        assert config.contracts.enabled is True
        assert "search" in config.contracts.tool_payloads.tools
        assert config.contracts.tool_order.must_follow == {
            "summarize": ["search", "retrieve"]
        }
        assert config.contracts.tool_order.forbidden_sequences == [
            ["delete", "commit"]
        ]

        assert config.on_violation.mode == OnViolationMode.FAIL_PROBE

        # Verify to_contracts compilation
        compiled = config.to_contracts()
        assert "search" in compiled["tool_schemas"]
        assert compiled["tool_order_rules"] is not None
        assert compiled["toggles"]["generate_boundaries"] is True
