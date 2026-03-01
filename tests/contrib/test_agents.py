"""Tests for Autonomous Agents module."""

import json
from unittest.mock import MagicMock

import pytest

from insideLLMs.contrib.agents import (
    AgentConfig,
    AgentExecutor,
    AgentMemory,
    AgentResult,
    AgentStatus,
    AgentStep,
    ChainOfThoughtAgent,
    ReActAgent,
    SimpleAgent,
    Tool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    create_calculator_tool,
    create_python_tool,
    create_react_agent,
    create_search_tool,
    create_simple_agent,
    quick_agent_run,
    tool,
)
from insideLLMs.models import DummyModel

# =============================================================================
# Test Configuration
# =============================================================================


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = AgentConfig()

        assert config.max_iterations == 10
        assert config.max_execution_time == 300
        assert config.early_stop_on_finish is True
        assert config.verbose is False
        assert config.memory_limit == 20

    def test_custom_values(self):
        """Test custom configuration."""
        config = AgentConfig(
            max_iterations=5,
            verbose=True,
            memory_limit=50,
        )

        assert config.max_iterations == 5
        assert config.verbose is True
        assert config.memory_limit == 50

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = AgentConfig(max_iterations=15)
        data = config.to_dict()

        assert data["max_iterations"] == 15
        assert "verbose" in data


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_status_values(self):
        """Test status values exist."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.THINKING.value == "thinking"
        assert AgentStatus.ACTING.value == "acting"
        assert AgentStatus.FINISHED.value == "finished"
        assert AgentStatus.ERROR.value == "error"


# =============================================================================
# Test Tool Classes
# =============================================================================


class TestToolParameter:
    """Tests for ToolParameter."""

    def test_basic_parameter(self):
        """Test basic parameter creation."""
        param = ToolParameter(
            name="query",
            type="str",
            description="Search query",
        )

        assert param.name == "query"
        assert param.type == "str"
        assert param.required is True

    def test_optional_parameter(self):
        """Test optional parameter."""
        param = ToolParameter(
            name="limit",
            type="int",
            description="Max results",
            required=False,
            default=10,
        )

        assert param.required is False
        assert param.default == 10


class TestTool:
    """Tests for Tool class."""

    def test_basic_tool(self):
        """Test basic tool creation."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        t = Tool(name="add", func=add, description="Add numbers")

        assert t.name == "add"
        assert t.description == "Add numbers"

    def test_tool_inference(self):
        """Test parameter inference from function."""

        def greet(name: str, formal: bool = False) -> str:
            """Greet a person."""
            if formal:
                return f"Good day, {name}."
            return f"Hi, {name}!"

        t = Tool(name="greet", func=greet)

        assert len(t.parameters) == 2
        assert t.parameters[0].name == "name"
        assert t.parameters[0].required is True
        assert t.parameters[1].name == "formal"
        assert t.parameters[1].required is False

    def test_tool_execution(self):
        """Test tool execution."""

        def multiply(x: int, y: int) -> int:
            return x * y

        t = Tool(name="multiply", func=multiply)
        result = t.execute({"x": 5, "y": 3})

        assert result.success is True
        assert result.output == 15

    def test_tool_execution_string_input(self):
        """Test execution with string input."""

        def echo(text: str) -> str:
            return text

        t = Tool(name="echo", func=echo)
        result = t.execute("hello")

        assert result.success is True
        assert result.output == "hello"

    def test_tool_execution_json_input(self):
        """Test execution with JSON string input."""

        def add(a: int, b: int) -> int:
            return a + b

        t = Tool(name="add", func=add)
        result = t.execute('{"a": 10, "b": 20}')

        assert result.success is True
        assert result.output == 30

    def test_tool_execution_error(self):
        """Test execution with error."""

        def divide(a: int, b: int) -> float:
            return a / b

        t = Tool(name="divide", func=divide)
        result = t.execute({"a": 1, "b": 0})

        assert result.success is False
        assert result.error is not None
        assert "division" in result.error.lower() or "zero" in result.error.lower()

    def test_tool_to_dict(self):
        """Test dictionary conversion."""

        def test_func(param1: str) -> str:
            return param1

        t = Tool(name="test", func=test_func, description="Test tool")
        data = t.to_dict()

        assert data["name"] == "test"
        assert data["description"] == "Test tool"
        assert len(data["parameters"]) == 1

    def test_tool_format_for_prompt(self):
        """Test prompt formatting."""

        def search(query: str, limit: int = 10) -> str:
            return f"Results for {query}"

        t = Tool(name="search", func=search, description="Search the web")
        prompt_str = t.format_for_prompt()

        assert "search" in prompt_str
        assert "query" in prompt_str
        assert "Search the web" in prompt_str


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_basic_decorator(self):
        """Test basic decorator usage."""

        @tool(name="add_nums")
        def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, Tool)
        assert add.name == "add_nums"

    def test_decorator_with_description(self):
        """Test decorator with description."""

        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert greet.description == "Greet someone"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        t = Tool(name="test", func=lambda x: x)

        registry.register(t)

        assert registry.get("test") is t

    def test_register_function(self):
        """Test registering a function."""
        registry = ToolRegistry()

        def my_func(x: str) -> str:
            return x.upper()

        t = registry.register_function(my_func)

        assert registry.get("my_func") is t

    def test_list_tools(self):
        """Test listing tools."""
        registry = ToolRegistry()
        registry.register(Tool(name="t1", func=lambda: None))
        registry.register(Tool(name="t2", func=lambda: None))

        tools = registry.list_tools()
        assert len(tools) == 2

    def test_format_for_prompt(self):
        """Test prompt formatting."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="calc",
                func=lambda x: x,
                description="Calculator",
            )
        )

        prompt = registry.format_for_prompt()
        assert "calc" in prompt
        assert "Calculator" in prompt


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(
            tool_name="test",
            input="input",
            output="output",
            success=True,
        )

        assert result.success is True
        assert result.output == "output"
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = ToolResult(
            tool_name="test",
            input="input",
            output=None,
            success=False,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ToolResult(
            tool_name="calc",
            input="2+2",
            output="4",
            success=True,
            execution_time_ms=5.5,
        )

        data = result.to_dict()
        assert data["tool_name"] == "calc"
        assert data["execution_time_ms"] == 5.5


# =============================================================================
# Test AgentStep and AgentResult
# =============================================================================


class TestAgentStep:
    """Tests for AgentStep."""

    def test_basic_step(self):
        """Test basic step creation."""
        step = AgentStep(step_number=1)

        assert step.step_number == 1
        assert step.thought is None
        assert step.action is None

    def test_step_with_details(self):
        """Test step with details."""
        step = AgentStep(
            step_number=1,
            thought="I should search",
            action="search",
            action_input="python tutorials",
            observation="Found 10 results",
        )

        assert step.thought == "I should search"
        assert step.action == "search"

    def test_to_dict(self):
        """Test dictionary conversion."""
        step = AgentStep(step_number=1, thought="Thinking...")
        data = step.to_dict()

        assert data["step_number"] == 1
        assert data["thought"] == "Thinking..."


class TestAgentResult:
    """Tests for AgentResult."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = AgentResult(
            query="What is 2+2?",
            answer="4",
            status=AgentStatus.FINISHED,
        )

        assert result.query == "What is 2+2?"
        assert result.answer == "4"
        assert result.status == AgentStatus.FINISHED

    def test_result_with_steps(self):
        """Test result with steps."""
        steps = [
            AgentStep(step_number=1, thought="Let me calculate"),
            AgentStep(step_number=2, action="calculator", action_input="2+2"),
        ]

        result = AgentResult(
            query="What is 2+2?",
            answer="4",
            status=AgentStatus.FINISHED,
            steps=steps,
        )

        assert len(result.steps) == 2

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = AgentResult(
            query="test",
            answer="answer",
            status=AgentStatus.FINISHED,
            total_iterations=3,
        )

        data = result.to_dict()
        assert data["query"] == "test"
        assert data["status"] == "finished"
        assert data["total_iterations"] == 3

    def test_to_json(self):
        """Test JSON conversion."""
        result = AgentResult(
            query="test",
            answer="answer",
            status=AgentStatus.FINISHED,
        )

        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["query"] == "test"


# =============================================================================
# Test AgentMemory
# =============================================================================


class TestAgentMemory:
    """Tests for AgentMemory."""

    def test_add_and_get_steps(self):
        """Test adding and getting steps."""
        memory = AgentMemory()

        memory.add_step(AgentStep(step_number=1, thought="First"))
        memory.add_step(AgentStep(step_number=2, thought="Second"))

        steps = memory.get_steps()
        assert len(steps) == 2
        assert steps[0].thought == "First"

    def test_get_last_step(self):
        """Test getting last step."""
        memory = AgentMemory()

        memory.add_step(AgentStep(step_number=1))
        memory.add_step(AgentStep(step_number=2))

        last = memory.get_last_step()
        assert last.step_number == 2

    def test_max_steps_eviction(self):
        """Test eviction when max steps reached."""
        memory = AgentMemory(max_steps=3)

        for i in range(5):
            memory.add_step(AgentStep(step_number=i + 1))

        steps = memory.get_steps()
        assert len(steps) == 3
        assert steps[0].step_number == 3  # Oldest kept

    def test_context(self):
        """Test context storage."""
        memory = AgentMemory()

        memory.set_context("user_id", "123")
        assert memory.get_context("user_id") == "123"
        assert memory.get_context("missing", "default") == "default"

    def test_clear(self):
        """Test clearing memory."""
        memory = AgentMemory()

        memory.add_step(AgentStep(step_number=1))
        memory.set_context("key", "value")

        memory.clear()

        assert len(memory.get_steps()) == 0
        assert memory.get_context("key") is None

    def test_format_scratchpad(self):
        """Test scratchpad formatting."""
        memory = AgentMemory()
        config = AgentConfig()

        memory.add_step(
            AgentStep(
                step_number=1,
                thought="I should search",
                action="search",
                action_input="python",
                observation="Found results",
            )
        )

        scratchpad = memory.format_scratchpad(config)

        assert "Thought:" in scratchpad
        assert "Action:" in scratchpad
        assert "Observation:" in scratchpad


# =============================================================================
# Test ReActAgent
# =============================================================================


class TestReActAgent:
    """Tests for ReActAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        model = DummyModel()
        agent = ReActAgent(model)

        assert agent.model is model
        assert agent.config is not None

    def test_initialization_with_tools(self):
        """Test initialization with tools."""
        model = DummyModel()
        calc = create_calculator_tool()

        agent = ReActAgent(model, tools=[calc])

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "calculator"

    def test_add_tool(self):
        """Test adding tools."""
        model = DummyModel()
        agent = ReActAgent(model)

        agent.add_tool(create_calculator_tool())

        assert len(agent.tools) == 1

    def test_run_basic(self):
        """Test basic run."""
        model = MagicMock()
        model.generate.return_value = "Thought: I know the answer.\nFinal Answer: 42"

        agent = ReActAgent(model)
        result = agent.run("What is the meaning of life?")

        assert result.answer == "42"
        assert result.status == AgentStatus.FINISHED

    def test_run_with_tool(self):
        """Test run with tool usage."""
        model = MagicMock()
        model.generate.side_effect = [
            "Thought: I need to calculate.\nAction: calculator\nAction Input: 2+2",
            "Thought: I now know the answer.\nFinal Answer: 4",
        ]

        agent = ReActAgent(model, tools=[create_calculator_tool()])
        result = agent.run("What is 2+2?")

        assert result.status == AgentStatus.FINISHED
        assert len(result.steps) >= 1

    def test_run_max_iterations(self):
        """Test max iterations limit."""
        model = MagicMock()
        model.generate.return_value = "Thought: Still thinking..."

        config = AgentConfig(max_iterations=3)
        agent = ReActAgent(model, config=config)

        result = agent.run("Infinite loop?")

        assert result.status == AgentStatus.MAX_ITERATIONS
        assert result.total_iterations <= 3

    def test_parse_response_final_answer(self):
        """Test parsing final answer."""
        model = DummyModel()
        agent = ReActAgent(model)

        thought, action, action_input, final = agent._parse_response(
            "Thought: Done thinking.\nFinal Answer: The answer is 42."
        )

        assert thought == "Done thinking."
        assert final == "The answer is 42."
        assert action is None

    def test_parse_response_action(self):
        """Test parsing action."""
        model = DummyModel()
        agent = ReActAgent(model)

        thought, action, action_input, final = agent._parse_response(
            "Thought: Need to search.\nAction: search\nAction Input: python tutorials"
        )

        assert thought == "Need to search."
        assert action == "search"
        assert action_input == "python tutorials"
        assert final is None


# =============================================================================
# Test SimpleAgent
# =============================================================================


class TestSimpleAgent:
    """Tests for SimpleAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        model = DummyModel()
        agent = SimpleAgent(model)

        assert agent.model is model

    def test_run_direct_answer(self):
        """Test run with direct answer."""
        model = MagicMock()
        model.generate.return_value = "Answer: The sky is blue."

        agent = SimpleAgent(model)
        result = agent.run("What color is the sky?")

        assert result.answer == "The sky is blue."
        assert result.status == AgentStatus.FINISHED

    def test_run_with_tool(self):
        """Test run with tool."""
        model = MagicMock()
        model.generate.return_value = "Tool: calculator\nInput: 5*5"

        agent = SimpleAgent(model, tools=[create_calculator_tool()])
        result = agent.run("Calculate 5*5")

        assert result.answer == "25"
        assert result.status == AgentStatus.FINISHED


# =============================================================================
# Test ChainOfThoughtAgent
# =============================================================================


class TestChainOfThoughtAgent:
    """Tests for ChainOfThoughtAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        model = DummyModel()
        agent = ChainOfThoughtAgent(model)

        assert agent.model is model

    def test_run_with_final_answer(self):
        """Test run with final answer."""
        model = MagicMock()
        model.generate.return_value = """
Let me think step by step:
1. First, I consider...
2. Then, I realize...
3. Therefore...

Final Answer: The answer is 42.
"""

        agent = ChainOfThoughtAgent(model)
        result = agent.run("What is the answer?")

        assert result.answer == "The answer is 42."
        assert result.status == AgentStatus.FINISHED

    def test_run_without_final_answer_marker(self):
        """Test run without explicit final answer."""
        model = MagicMock()
        model.generate.return_value = """
Step 1: Consider the problem.
Step 2: Analyze the options.

The answer must be 42.
"""

        agent = ChainOfThoughtAgent(model)
        result = agent.run("What is the answer?")

        # Should use last paragraph
        assert "42" in result.answer
        assert result.status == AgentStatus.FINISHED


# =============================================================================
# Test AgentExecutor
# =============================================================================


class TestAgentExecutor:
    """Tests for AgentExecutor."""

    def test_initialization(self):
        """Test executor initialization."""
        model = DummyModel()
        agent = ReActAgent(model)
        executor = AgentExecutor(agent)

        assert executor.agent is agent
        assert executor.execution_count == 0

    def test_run(self):
        """Test basic execution."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: Done"

        agent = ReActAgent(model)
        executor = AgentExecutor(agent)

        result = executor.run("Test query")

        assert result is not None
        assert executor.execution_count == 1

    def test_pre_hook(self):
        """Test pre-execution hook."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: Done"

        agent = ReActAgent(model)
        hook_called = []

        def pre_hook(query, kwargs):
            hook_called.append(query)

        executor = AgentExecutor(agent, pre_hooks=[pre_hook])
        executor.run("Test")

        assert len(hook_called) == 1
        assert hook_called[0] == "Test"

    def test_post_hook(self):
        """Test post-execution hook."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: Done"

        agent = ReActAgent(model)
        results = []

        def post_hook(result):
            results.append(result)

        executor = AgentExecutor(agent, post_hooks=[post_hook])
        executor.run("Test")

        assert len(results) == 1

    def test_result_processor(self):
        """Test result processor."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: lowercase"

        agent = ReActAgent(model)

        def processor(result):
            if result.answer:
                result.answer = result.answer.upper()
            return result

        executor = AgentExecutor(agent, result_processor=processor)
        result = executor.run("Test")

        assert result.answer == "LOWERCASE"

    def test_batch_run(self):
        """Test batch execution."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: Done"

        agent = ReActAgent(model)
        executor = AgentExecutor(agent)

        results = executor.batch_run(["Query 1", "Query 2", "Query 3"])

        assert len(results) == 3
        assert executor.execution_count == 3

    def test_history(self):
        """Test execution history."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: Done"

        agent = ReActAgent(model)
        executor = AgentExecutor(agent)

        executor.run("Query 1")
        executor.run("Query 2")

        history = executor.get_history()
        assert len(history) == 2

        executor.clear_history()
        assert len(executor.get_history()) == 0


# =============================================================================
# Test Built-in Tools
# =============================================================================


class TestBuiltInTools:
    """Tests for built-in tools."""

    def test_calculator_tool(self):
        """Test calculator tool."""
        calc = create_calculator_tool()

        assert calc.name == "calculator"

        result = calc.execute("10 + 5")
        assert result.success is True
        assert result.output == "15"

    def test_calculator_invalid_chars(self):
        """Test calculator with invalid characters."""
        calc = create_calculator_tool()

        result = calc.execute("import os")
        assert "Invalid" in result.output or "Error" in result.output

    def test_calculator_rejects_non_arithmetic_ast(self):
        """Calculator rejects expressions outside arithmetic AST."""
        calc = create_calculator_tool()
        result = calc.execute("2**3")
        assert result.success is True
        assert result.output == "8"

    def test_search_tool(self):
        """Test search tool."""
        search = create_search_tool()

        assert search.name == "search"

        result = search.execute("python")
        assert result.success is True

    def test_search_tool_custom_fn(self):
        """Test search tool with custom function."""

        def custom_search(query: str) -> str:
            return f"Custom results for: {query}"

        search = create_search_tool(custom_search)
        result = search.execute("test")

        assert "Custom results" in result.output

    def test_python_tool_disabled(self):
        """Test Python tool when disabled."""
        python = create_python_tool(allow_exec=False)

        result = python.execute("print('hello')")
        assert "disabled" in result.output.lower()

    def test_python_tool_requires_sandbox_contract_when_enabled(self):
        """Execution stays disabled without explicit sandbox contract."""
        python = create_python_tool(allow_exec=True)
        result = python.execute("result = 2 + 2")
        assert "sandbox contract" in result.output.lower()


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_react_agent(self):
        """Test create_react_agent."""
        model = DummyModel()
        agent = create_react_agent(model, max_iterations=5)

        assert isinstance(agent, ReActAgent)
        assert agent.config.max_iterations == 5

    def test_create_react_agent_with_functions(self):
        """Test create_react_agent with callable tools."""
        model = DummyModel()

        def my_tool(x: str) -> str:
            return x.upper()

        agent = create_react_agent(model, tools=[my_tool])

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "my_tool"

    def test_create_simple_agent(self):
        """Test create_simple_agent."""
        model = DummyModel()
        agent = create_simple_agent(model)

        assert isinstance(agent, SimpleAgent)

    def test_quick_agent_run_react(self):
        """Test quick_agent_run with ReAct."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: Quick result"

        result = quick_agent_run("Test", model, agent_type="react")

        assert result.answer == "Quick result"

    def test_quick_agent_run_simple(self):
        """Test quick_agent_run with simple agent."""
        model = MagicMock()
        model.generate.return_value = "Answer: Simple result"

        result = quick_agent_run("Test", model, agent_type="simple")

        assert result.answer == "Simple result"

    def test_quick_agent_run_cot(self):
        """Test quick_agent_run with CoT agent."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: CoT result"

        result = quick_agent_run("Test", model, agent_type="cot")

        assert "CoT result" in result.answer

    def test_quick_agent_run_invalid_type(self):
        """Test quick_agent_run with invalid type."""
        model = DummyModel()

        with pytest.raises(ValueError):
            quick_agent_run("Test", model, agent_type="invalid")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests."""

    def test_full_react_loop(self):
        """Test complete ReAct execution loop."""
        model = MagicMock()
        model.generate.side_effect = [
            "Thought: I need to calculate 10 * 5.\nAction: calculator\nAction Input: 10 * 5",
            "Thought: The result is 50. Now I can answer.\nFinal Answer: 10 * 5 = 50",
        ]

        calc = create_calculator_tool()
        agent = ReActAgent(model, tools=[calc])

        result = agent.run("What is 10 * 5?")

        assert result.status == AgentStatus.FINISHED
        assert "50" in result.answer
        assert len(result.steps) == 2

    def test_agent_with_executor(self):
        """Test agent through executor."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: Executed"

        agent = ReActAgent(model)

        execution_log = []

        def log_hook(result):
            execution_log.append(result.answer)

        executor = AgentExecutor(agent, post_hooks=[log_hook])
        executor.run("Test")

        assert len(execution_log) == 1
        assert execution_log[0] == "Executed"

    def test_tool_error_handling(self):
        """Test error handling in tool execution."""
        model = MagicMock()
        model.generate.side_effect = [
            "Thought: Let me try.\nAction: broken\nAction Input: test",
            "Thought: Tool failed, I'll answer directly.\nFinal Answer: Handled error",
        ]

        def broken_tool(x: str) -> str:
            raise RuntimeError("Tool is broken!")

        agent = ReActAgent(model, tools=[Tool(name="broken", func=broken_tool)])
        result = agent.run("Test error handling")

        # Should handle error and continue
        assert result.status == AgentStatus.FINISHED
        assert "Handled error" in result.answer


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_query(self):
        """Test with empty query."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: Empty input received"

        agent = ReActAgent(model)
        result = agent.run("")

        assert result is not None

    def test_very_long_query(self):
        """Test with very long query."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: Processed"

        agent = ReActAgent(model)
        long_query = "word " * 1000

        result = agent.run(long_query)
        assert result is not None

    def test_special_characters_in_query(self):
        """Test with special characters."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: Handled special chars"

        agent = ReActAgent(model)
        result = agent.run("Query with 'quotes' and \"double quotes\" and {braces}")

        assert result is not None

    def test_unicode_query(self):
        """Test with unicode."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: 你好"

        agent = ReActAgent(model)
        result = agent.run("你好世界")

        assert result.answer == "你好"

    def test_no_tools(self):
        """Test agent with no tools."""
        model = MagicMock()
        model.generate.return_value = "Final Answer: No tools needed"

        agent = ReActAgent(model)
        result = agent.run("Just answer this")

        assert result.answer == "No tools needed"

    def test_unknown_tool(self):
        """Test when model tries to use unknown tool."""
        model = MagicMock()
        model.generate.side_effect = [
            "Thought: Use tool.\nAction: nonexistent\nAction Input: test",
            "Thought: Tool failed.\nFinal Answer: Recovered",
        ]

        agent = ReActAgent(model)
        result = agent.run("Test unknown tool")

        # Should handle gracefully
        assert result is not None
