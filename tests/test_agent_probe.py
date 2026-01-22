"""Tests for AgentProbe class."""

import pytest
from typing import Any

from insideLLMs.probes.agent_probe import (
    AgentProbe,
    AgentProbeResult,
    ToolDefinition,
)
from insideLLMs.tracing import TraceRecorder, TraceEventKind
from insideLLMs.trace_config import TraceConfig, load_trace_config, OnViolationMode
from insideLLMs.types import ProbeCategory, ProbeResult, ResultStatus


class MockModel:
    """Mock model for testing."""

    def __init__(self, response: str = "Mock response"):
        self.response = response
        self.calls = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return self.response


class SimpleAgentProbe(AgentProbe):
    """Simple test implementation of AgentProbe."""

    def run_agent(
        self,
        model: Any,
        prompt: str,
        tools: list[ToolDefinition],
        recorder: TraceRecorder,
        **kwargs,
    ) -> str:
        # Record generate start
        recorder.record_generate_start(prompt)

        # Simulate tool usage
        if tools:
            for tool in tools:
                recorder.record_tool_call(tool.name, {"query": prompt})
                recorder.record_tool_result(tool.name, {"result": "tool result"})

        # Get response
        response = model.generate(prompt)

        # Record generate end
        recorder.record_generate_end(response)

        return response


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_basic_creation(self):
        """Test basic tool definition."""
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "required": ["query"]},
        )
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert "query" in tool.parameters.get("required", [])

    def test_default_values(self):
        """Test default values."""
        tool = ToolDefinition(name="test", description="Test tool")
        assert tool.parameters == {}
        assert tool.handler is None


class TestAgentProbeResult:
    """Tests for AgentProbeResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = AgentProbeResult(
            prompt="Test prompt",
            final_response="Test response",
        )
        assert result.prompt == "Test prompt"
        assert result.final_response == "Test response"
        assert result.tool_calls == []
        assert result.trace_events == []
        assert result.violations == []

    def test_with_full_data(self):
        """Test result with all fields."""
        result = AgentProbeResult(
            prompt="Search for X",
            final_response="Found X",
            tool_calls=[{"tool_name": "search", "arguments": {"q": "X"}}],
            trace_fingerprint="sha256:abc123",
            violations=[{"code": "ERROR", "detail": "test"}],
            metadata={"key": "value"},
        )
        assert len(result.tool_calls) == 1
        assert result.trace_fingerprint == "sha256:abc123"
        assert len(result.violations) == 1


class TestAgentProbe:
    """Tests for AgentProbe class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        probe = SimpleAgentProbe(name="test_probe")
        assert probe.name == "test_probe"
        assert probe.category == ProbeCategory.REASONING
        assert probe.tools == []
        assert probe.trace_config.enabled is True

    def test_init_with_tools(self):
        """Test initialization with tools."""
        tools = [
            ToolDefinition(name="search", description="Search"),
            ToolDefinition(name="calc", description="Calculate"),
        ]
        probe = SimpleAgentProbe(name="test", tools=tools)
        assert len(probe.tools) == 2
        assert probe.tools[0].name == "search"

    def test_init_with_trace_config_dict(self):
        """Test initialization with config dict."""
        probe = SimpleAgentProbe(
            name="test",
            trace_config={"contracts": {"stream_boundaries": {"enabled": False}}},
        )
        assert probe.trace_config.contracts.stream_boundaries.enabled is False

    def test_init_with_trace_config_object(self):
        """Test initialization with TraceConfig object."""
        config = load_trace_config({"enabled": False})
        probe = SimpleAgentProbe(name="test", trace_config=config)
        assert probe.trace_config.enabled is False

    def test_run_basic(self):
        """Test basic run execution."""
        model = MockModel(response="Agent response")
        probe = SimpleAgentProbe(name="test")

        result = probe.run(model, "Test prompt")

        assert result.prompt == "Test prompt"
        assert result.final_response == "Agent response"
        assert result.trace_fingerprint is not None
        assert result.trace_fingerprint.startswith("sha256:")

    def test_run_with_tools(self):
        """Test run with tool usage."""
        model = MockModel()
        tools = [ToolDefinition(name="search", description="Search")]
        probe = SimpleAgentProbe(name="test", tools=tools)

        result = probe.run(model, "Search for X")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool_name"] == "search"
        # Trace events should include tool_call_start and tool_result
        kinds = [e["kind"] for e in result.trace_events]
        assert TraceEventKind.TOOL_CALL_START.value in kinds
        assert TraceEventKind.TOOL_RESULT.value in kinds

    def test_run_records_trace_events(self):
        """Test that run records trace events."""
        model = MockModel()
        probe = SimpleAgentProbe(name="test")

        result = probe.run(model, "Test")

        # Should have generate_start and generate_end
        kinds = [e["kind"] for e in result.trace_events]
        assert TraceEventKind.GENERATE_START.value in kinds
        assert TraceEventKind.GENERATE_END.value in kinds

    def test_run_validates_contracts(self):
        """Test that violations are detected."""
        class BadAgentProbe(AgentProbe):
            """Probe that creates invalid traces."""

            def run_agent(self, model, prompt, tools, recorder, **kwargs):
                # Record generate_start but no generate_end
                recorder.record_generate_start(prompt)
                return "response"

        model = MockModel()
        probe = BadAgentProbe(name="bad")

        result = probe.run(model, "Test")

        # Should have violation for missing generate_end
        assert len(result.violations) > 0
        codes = [v["code"] for v in result.violations]
        assert "GENERATE_NO_END" in codes

    def test_run_fail_probe_on_violation(self):
        """Test fail_probe mode raises on violation."""
        class BadAgentProbe(AgentProbe):
            def run_agent(self, model, prompt, tools, recorder, **kwargs):
                recorder.record_generate_start(prompt)
                return "response"

        model = MockModel()
        probe = BadAgentProbe(
            name="bad",
            trace_config={"on_violation": {"mode": "fail_probe"}},
        )

        with pytest.raises(ValueError, match="Trace contract violations"):
            probe.run(model, "Test")

    def test_run_with_dict_data(self):
        """Test run accepts dict input."""
        model = MockModel()
        probe = SimpleAgentProbe(name="test")

        result = probe.run(model, {"prompt": "Test prompt"})

        assert result.prompt == "Test prompt"

    def test_run_with_run_id(self):
        """Test run_id is passed to recorder."""
        model = MockModel()
        probe = SimpleAgentProbe(name="test")

        result = probe.run(model, "Test", run_id="run_123", example_id="ex_001")

        # Check that events have run_id
        for event in result.trace_events:
            assert event.get("run_id") == "run_123"
            assert event.get("example_id") == "ex_001"

    def test_get_tool_by_name(self):
        """Test get_tool_by_name method."""
        tools = [
            ToolDefinition(name="search", description="Search"),
            ToolDefinition(name="calc", description="Calculate"),
        ]
        probe = SimpleAgentProbe(name="test", tools=tools)

        search = probe.get_tool_by_name("search")
        assert search is not None
        assert search.name == "search"

        unknown = probe.get_tool_by_name("unknown")
        assert unknown is None

    def test_info(self):
        """Test info method includes tools."""
        tools = [ToolDefinition(name="search", description="Search")]
        probe = SimpleAgentProbe(
            name="test",
            tools=tools,
            trace_config={"contracts": {"enabled": False}},
        )

        info = probe.info()

        assert info["name"] == "test"
        assert info["trace_enabled"] is True
        assert info["contracts_enabled"] is False
        assert len(info["tools"]) == 1
        assert info["tools"][0]["name"] == "search"

    def test_score_includes_violation_rate(self):
        """Test score calculates violation rate."""
        probe = SimpleAgentProbe(name="test")

        # Create mock results
        results = [
            ProbeResult(
                input="a",
                output=AgentProbeResult(prompt="a", final_response="b", violations=[]),
                status=ResultStatus.SUCCESS,
            ),
            ProbeResult(
                input="c",
                output=AgentProbeResult(
                    prompt="c",
                    final_response="d",
                    violations=[{"code": "ERR"}],
                ),
                status=ResultStatus.SUCCESS,
            ),
        ]

        score = probe.score(results)

        assert "violation_rate" in score.custom_metrics
        assert score.custom_metrics["violation_rate"] == 0.5


class TestAgentProbePayloadRedaction:
    """Tests for payload redaction in AgentProbe."""

    def test_redacts_sensitive_payloads(self):
        """Test that sensitive data is redacted in trace."""
        class SecretAgentProbe(AgentProbe):
            def run_agent(self, model, prompt, tools, recorder, **kwargs):
                recorder.record_generate_start(prompt)
                recorder.record("custom", {"secret": "password123", "public": "data"})
                recorder.record_generate_end("response")
                return "response"

        model = MockModel()
        probe = SecretAgentProbe(
            name="secret",
            trace_config={
                "store": {
                    "redact": {
                        "enabled": True,
                        "json_pointers": ["/secret"],
                    },
                },
            },
        )

        result = probe.run(model, "Test")

        # Find the custom event
        custom_events = [e for e in result.trace_events if e["kind"] == "custom"]
        assert len(custom_events) == 1
        assert custom_events[0]["payload"]["secret"] == "<redacted>"
        assert custom_events[0]["payload"]["public"] == "data"
