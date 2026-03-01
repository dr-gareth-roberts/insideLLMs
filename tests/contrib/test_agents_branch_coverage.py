"""Additional branch coverage for agent edge paths."""

from __future__ import annotations

from insideLLMs.contrib.agents import (
    AgentConfig,
    AgentStatus,
    BaseAgent,
    ChainOfThoughtAgent,
    ReActAgent,
    SimpleAgent,
    Tool,
    create_calculator_tool,
    create_python_tool,
    create_react_agent,
    create_simple_agent,
)
from insideLLMs.models import DummyModel


class StaticModel:
    def __init__(self, response: str):
        self.response = response

    def generate(self, _prompt: str, **_kwargs):
        return self.response


def test_tool_inference_skips_self_and_tool_call_and_non_dict_execute():
    def method_like(self, value: int) -> int:
        return value

    inferred = Tool(name="method_like", func=method_like)
    assert [p.name for p in inferred.parameters] == ["value"]

    add = Tool(name="add", func=lambda a, b: a + b)
    assert add(2, 3) == 5

    doubled = Tool(name="double", func=lambda x: x * 2)
    result = doubled.execute(4)
    assert result.success is True
    assert result.output == 8


def test_base_agent_tool_registration_callable_paths_and_abstract_base_methods():
    def square(x: int) -> int:
        return x * x

    agent = ReActAgent(DummyModel(), tools=[square])
    assert any(t.name == "square" for t in agent.tools)

    agent.add_tool(lambda y: y)
    assert any(t.name == "<lambda>" for t in agent.tools)

    # Explicitly execute abstract base defaults for branch coverage.
    assert BaseAgent.run(agent, "query") is None
    assert BaseAgent._plan(agent, "query") is None


def test_react_agent_timeout_and_non_retry_error_paths(monkeypatch):
    timeout_agent = ReActAgent(
        DummyModel(),
        config=AgentConfig(max_iterations=3, max_execution_time=0.0),
    )
    timed_out = timeout_agent.run("q")
    assert timed_out.status == AgentStatus.MAX_ITERATIONS
    assert timed_out.total_iterations == 0

    error_agent = ReActAgent(
        DummyModel(),
        config=AgentConfig(max_iterations=2, retry_on_error=False),
    )
    monkeypatch.setattr(error_agent, "_plan", lambda _q: (_ for _ in ()).throw(ValueError("boom")))
    errored = error_agent.run("q")
    assert errored.status == AgentStatus.ERROR
    assert "Error: boom" in errored.steps[0].observation


def test_simple_agent_error_and_fallback_answer_branches():
    tool_error_agent = SimpleAgent(StaticModel("Tool: missing_tool\nInput: payload"))
    tool_error = tool_error_agent.run("query")
    assert tool_error.status == AgentStatus.ERROR
    assert "Error:" in tool_error.steps[0].observation

    fallback_agent = SimpleAgent(StaticModel("Direct response without parser markers"))
    fallback = fallback_agent.run("query")
    assert fallback.status == AgentStatus.FINISHED
    assert fallback.answer == "Direct response without parser markers"
    assert fallback_agent._plan("unused") == ""


def test_chain_of_thought_plan_delegates_to_model():
    agent = ChainOfThoughtAgent(StaticModel("reasoning"))
    assert agent._plan("what?") == "reasoning"


def test_calculator_and_python_tool_error_branches():
    calculator = create_calculator_tool()
    bad_expr = calculator.execute("1/(")
    assert bad_expr.success is True
    assert "Error:" in bad_expr.output

    python_tool = create_python_tool(allow_exec=True, sandbox_contract="unit-test-sandbox")
    success = python_tool.execute("result = 41 + 1")
    assert success.output == "42"

    failure = python_tool.execute("result = 1/0")
    assert "Error:" in failure.output


def test_agent_factory_conversion_branches_for_tool_and_callable():
    def triple(x: int) -> int:
        return x * 3

    react_agent = create_react_agent(DummyModel(), tools=[Tool("noop", lambda: None), triple])
    react_names = [t.name for t in react_agent.tools]
    assert "noop" in react_names
    assert "triple" in react_names

    simple_agent = create_simple_agent(DummyModel(), tools=[Tool("noop2", lambda: None), triple])
    simple_names = [t.name for t in simple_agent.tools]
    assert "noop2" in simple_names
    assert "triple" in simple_names
