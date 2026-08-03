"""ReAct, simple, and chain-of-thought agent implementations."""

import re
import time
from typing import Any, Optional

from insideLLMs.contrib._agents.base import BaseAgent
from insideLLMs.contrib._agents.models import AgentConfig, AgentResult, AgentStatus, AgentStep
from insideLLMs.contrib._agents.tools import Tool

REACT_PROMPT_TEMPLATE = """Answer the following question using the tools available. Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Available Tools:
{tools}

Question: {question}
{scratchpad}"""


class ReActAgent(BaseAgent):
    """ReAct (Reasoning + Acting) Agent implementation.

    Implements the ReAct paradigm from "ReAct: Synergizing Reasoning and Acting
    in Language Models" (Yao et al., 2022). The agent iteratively:
    1. Thinks about what to do next (Reasoning)
    2. Takes an action using a tool (Acting)
    3. Observes the result
    4. Repeats until a final answer is reached or max iterations exceeded

    This is the primary agent type for tasks requiring multi-step reasoning
    with tool use.

    Args:
        model: LLM to use for reasoning. Must have a `generate(prompt)` method.
        tools: List of tools available to the agent. Can be Tool objects or
            callable functions.
        config: AgentConfig for customizing behavior (iterations, timeouts, etc.).
        system_prompt: Optional custom system prompt to override the default.

    Attributes:
        model: The LLM model instance.
        config: The agent configuration.
        memory: AgentMemory for execution history.
        _system_prompt: The system prompt used (if custom).

    Examples:
        Basic usage with calculator:

            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> model = DummyModel()
            >>> calc = create_calculator_tool()
            >>> agent = ReActAgent(model, tools=[calc])
            >>>
            >>> result = agent.run("What is 15 * 4?")
            >>> print(result.status)
            AgentStatus.FINISHED

        With custom configuration:

            >>> from insideLLMs.contrib.agents import ReActAgent, AgentConfig, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> config = AgentConfig(
            ...     max_iterations=5,
            ...     verbose=True,
            ...     max_execution_time=60
            ... )
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()], config=config)
            >>> result = agent.run("Calculate 100 / 4")

        With multiple tools:

            >>> from insideLLMs.contrib.agents import ReActAgent, Tool, create_calculator_tool, create_search_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> tools = [
            ...     create_calculator_tool(),
            ...     create_search_tool(),
            ...     Tool("greet", lambda name: f"Hello, {name}!")
            ... ]
            >>> agent = ReActAgent(DummyModel(), tools=tools)
            >>> print(len(agent.tools))
            3

        Analyzing execution steps:

            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> result = agent.run("What is 5 + 5?")
            >>>
            >>> # Examine the reasoning trace
            >>> for step in result.steps:
            ...     if step.thought:
            ...         print(f"Thought: {step.thought[:50]}...")
            ...     if step.action:
            ...         print(f"Action: {step.action}({step.action_input})")
            ...     if step.observation:
            ...         print(f"Observation: {step.observation}")

    Note:
        The agent uses a ReAct-style prompt that expects the model to output
        responses in a specific format with "Thought:", "Action:", "Action Input:",
        "Observation:", and "Final Answer:" prefixes.

    See Also:
        :class:`SimpleAgent`: A simpler agent without iterative reasoning.
        :class:`ChainOfThoughtAgent`: Agent focused on reasoning without tools.
        :func:`create_react_agent`: Convenience function for creating ReActAgent.
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[list[Tool]] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the ReAct agent.

        Args:
            model: LLM model with generate() method.
            tools: Optional list of tools.
            config: Optional configuration.
            system_prompt: Optional custom system prompt.
        """
        super().__init__(model, tools, config)
        self._system_prompt = system_prompt

    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Run the agent on a query using the ReAct loop.

        Executes the ReAct loop: Think -> Act -> Observe -> Repeat until
        a final answer is found or limits are reached.

        Args:
            query: The question or task for the agent.
            **kwargs: Additional arguments (currently unused).

        Returns:
            AgentResult: Contains the answer, execution status, step trace,
                and timing information.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> result = agent.run("What is 2 + 2?")
            >>>
            >>> print(f"Query: {result.query}")
            Query: What is 2 + 2?
            >>> print(f"Status: {result.status}")
            >>> print(f"Iterations: {result.total_iterations}")
            >>> print(f"Time: {result.execution_time_ms:.2f}ms")
        """
        start_time = time.time()
        self.memory.clear()

        result = AgentResult(
            query=query,
            answer=None,
            status=AgentStatus.THINKING,
        )

        for iteration in range(self.config.max_iterations):
            # Check timeout
            if time.time() - start_time > self.config.max_execution_time:
                result.status = AgentStatus.MAX_ITERATIONS
                break

            step = AgentStep(step_number=iteration + 1)

            try:
                # Get agent response
                response = self._plan(query)

                # Parse response
                thought, action, action_input, final_answer = self._parse_response(response)

                step.thought = thought
                step.action = action
                step.action_input = action_input

                # Check for final answer
                if final_answer:
                    result.answer = final_answer
                    result.status = AgentStatus.FINISHED
                    self.memory.add_step(step)
                    result.steps.append(step)
                    break

                # Execute action
                if action:
                    result.status = AgentStatus.ACTING
                    tool_result = self._execute_tool(action, action_input)
                    step.tool_result = tool_result

                    if tool_result.success:
                        step.observation = str(tool_result.output)
                    else:
                        step.observation = f"Error: {tool_result.error}"

                    result.status = AgentStatus.OBSERVING

                self.memory.add_step(step)
                result.steps.append(step)

            except Exception as e:
                step.observation = f"Error: {str(e)}"
                self.memory.add_step(step)
                result.steps.append(step)

                if not self.config.retry_on_error:
                    result.status = AgentStatus.ERROR
                    break

        if result.status not in (AgentStatus.FINISHED, AgentStatus.ERROR):
            result.status = AgentStatus.MAX_ITERATIONS

        result.total_iterations = len(result.steps)
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def _plan(self, query: str) -> str:
        """Generate the next reasoning step using the model.

        Builds a prompt including the question and scratchpad (previous steps),
        then calls the model to generate the next thought/action.

        Args:
            query: The original query being processed.

        Returns:
            str: The model's response containing thought/action/answer.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>> response = agent._plan("What is 2 + 2?")
            >>> print(type(response))
            <class 'str'>
        """
        prompt = self._build_prompt(query)
        return self.model.generate(prompt)

    def _build_prompt(self, query: str) -> str:
        """Build the ReAct prompt for the model.

        Constructs a prompt that includes the tool descriptions, the question,
        and any previous reasoning steps (scratchpad).

        Args:
            query: The question to answer.

        Returns:
            str: The complete prompt string.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> prompt = agent._build_prompt("What is 5 + 5?")
            >>> print("calculator" in prompt)
            True
            >>> print("What is 5 + 5?" in prompt)
            True
        """
        tool_names = ", ".join(t.name for t in self.tools)
        tools_desc = self._registry.format_for_prompt()
        scratchpad = ""

        if self.config.include_scratchpad and self.memory.get_steps():
            scratchpad = self.memory.format_scratchpad(self.config)

        return REACT_PROMPT_TEMPLATE.format(
            tool_names=tool_names or "none",
            tools=tools_desc or "No tools available",
            question=query,
            scratchpad=scratchpad,
        )

    def _parse_response(
        self,
        response: str,
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse the model response into structured components.

        Extracts thought, action, action_input, and final_answer from the
        model's ReAct-formatted response using regex patterns.

        Args:
            response: The raw model response string.

        Returns:
            tuple: A tuple of (thought, action, action_input, final_answer),
                where any component may be None if not found.

        Examples:
            >>> from insideLLMs.contrib.agents import ReActAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ReActAgent(DummyModel())
            >>>
            >>> # Parse a thought + action response
            >>> response = '''Thought: I need to calculate this.
            ... Action: calculator
            ... Action Input: 2 + 2'''
            >>> thought, action, action_input, final = agent._parse_response(response)
            >>> print(thought)
            I need to calculate this.
            >>> print(action)
            calculator
            >>> print(action_input)
            2 + 2
            >>> print(final)
            None
            >>>
            >>> # Parse a final answer response
            >>> response = '''Thought: I now know the answer.
            ... Final Answer: The result is 4.'''
            >>> thought, action, action_input, final = agent._parse_response(response)
            >>> print(final)
            The result is 4.
        """
        thought = None
        action = None
        action_input = None
        final_answer = None

        # Extract thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract final answer
        final_match = re.search(
            r"Final Answer:\s*(.+?)$",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match:
            final_answer = final_match.group(1).strip()
            return thought, None, None, final_answer

        # Extract action
        action_match = re.search(
            r"Action:\s*(.+?)(?=Action Input:|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            action = action_match.group(1).strip()

        # Extract action input
        input_match = re.search(
            r"Action Input:\s*(.+?)(?=Observation:|Thought:|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if input_match:
            action_input = input_match.group(1).strip()

        return thought, action, action_input, final_answer


# =============================================================================
# Simple Agent (No ReAct)
# =============================================================================


class SimpleAgent(BaseAgent):
    """Simple agent that executes tools based on direct instructions.

    Unlike the ReActAgent, the SimpleAgent does not iterate through multiple
    reasoning cycles. It makes a single decision about which tool to use
    (if any) and returns the result. This is suitable for simple tasks
    where complex multi-step reasoning is not required.

    Args:
        model: LLM to use for reasoning. Must have a `generate(prompt)` method.
        tools: Optional list of tools available to the agent.
        config: Optional AgentConfig for customizing behavior.

    Attributes:
        model: The LLM model instance.
        config: The agent configuration.
        memory: AgentMemory (less utilized in SimpleAgent).

    Examples:
        Basic usage:

            >>> from insideLLMs.contrib.agents import SimpleAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = SimpleAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> result = agent.run("Calculate 5 + 3")
            >>> print(result.total_iterations)
            1

        Without tools (direct answer):

            >>> from insideLLMs.contrib.agents import SimpleAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = SimpleAgent(DummyModel())
            >>> result = agent.run("What is the capital of France?")
            >>> print(result.status)
            AgentStatus.FINISHED

        Handling tool errors:

            >>> from insideLLMs.contrib.agents import SimpleAgent, Tool, AgentStatus
            >>> from insideLLMs import DummyModel
            >>>
            >>> def faulty_tool(x: int) -> int:
            ...     raise ValueError("Something went wrong")
            >>>
            >>> agent = SimpleAgent(DummyModel(), tools=[Tool("faulty", faulty_tool)])
            >>> # If model chooses the faulty tool, result.status will be ERROR

        Comparing with ReActAgent:

            >>> from insideLLMs.contrib.agents import SimpleAgent, ReActAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> # SimpleAgent: single step, no iteration
            >>> simple = SimpleAgent(DummyModel())
            >>> result = simple.run("Hello")
            >>> print(result.total_iterations)
            1
            >>>
            >>> # ReActAgent: may take multiple steps
            >>> react = ReActAgent(DummyModel())
            >>> result = react.run("Hello")
            >>> # May have multiple iterations

    Note:
        SimpleAgent is best for tasks where:
        - A single tool call is sufficient
        - No multi-step reasoning is needed
        - Speed is prioritized over complex problem-solving

    See Also:
        :class:`ReActAgent`: For tasks requiring multi-step reasoning.
        :class:`ChainOfThoughtAgent`: For reasoning-heavy tasks without tools.
    """

    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Run the agent on a query with a single decision step.

        Makes a single decision about which tool to use (if any) based on
        the query, executes the tool, and returns the result.

        Args:
            query: The question or task for the agent.
            **kwargs: Additional arguments (currently unused).

        Returns:
            AgentResult: Contains the answer, execution status, and single step.

        Examples:
            >>> from insideLLMs.contrib.agents import SimpleAgent, create_calculator_tool
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = SimpleAgent(DummyModel(), tools=[create_calculator_tool()])
            >>> result = agent.run("What is 10 + 5?")
            >>>
            >>> print(f"Query: {result.query}")
            Query: What is 10 + 5?
            >>> print(f"Steps: {len(result.steps)}")
            Steps: 1
        """
        start_time = time.time()

        result = AgentResult(
            query=query,
            answer=None,
            status=AgentStatus.THINKING,
        )

        # Ask model what to do
        tool_prompt = f"""Given this query: "{query}"

Available tools:
{self._registry.format_for_prompt()}

Which tool should be used? Respond with:
Tool: <tool_name>
Input: <input_to_tool>

Or if no tool is needed, respond with:
Answer: <direct_answer>"""

        response = self.model.generate(tool_prompt)

        step = AgentStep(step_number=1, thought=response)

        # Parse response
        tool_match = re.search(r"Tool:\s*(\w+)", response, re.IGNORECASE)
        input_match = re.search(r"Input:\s*(.+?)(?=\n|$)", response, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"Answer:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)

        if answer_match:
            result.answer = answer_match.group(1).strip()
            result.status = AgentStatus.FINISHED
        elif tool_match:
            tool_name = tool_match.group(1).strip()
            tool_input = input_match.group(1).strip() if input_match else ""

            step.action = tool_name
            step.action_input = tool_input

            tool_result = self._execute_tool(tool_name, tool_input)
            step.tool_result = tool_result

            if tool_result.success:
                result.answer = str(tool_result.output)
                step.observation = str(tool_result.output)
                result.status = AgentStatus.FINISHED
            else:
                result.status = AgentStatus.ERROR
                step.observation = f"Error: {tool_result.error}"
        else:
            result.answer = response
            result.status = AgentStatus.FINISHED

        result.steps.append(step)
        result.total_iterations = 1
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def _plan(self, query: str) -> str:
        """Not used in SimpleAgent.

        The SimpleAgent does not use a planning method as it makes direct
        tool decisions within the run method.

        Args:
            query: The query (unused).

        Returns:
            str: Always returns an empty string.
        """
        return ""


# =============================================================================
# Chain of Thought Agent
# =============================================================================


class ChainOfThoughtAgent(BaseAgent):
    """Agent that uses chain-of-thought (CoT) reasoning.

    This agent prompts the model to break down complex problems into explicit
    reasoning steps before arriving at a final answer. Unlike ReActAgent,
    it does not use tools - it relies purely on the model's reasoning ability.

    Chain-of-thought prompting has been shown to improve performance on
    complex reasoning tasks by encouraging step-by-step thinking.

    Args:
        model: LLM to use for reasoning. Must have a `generate(prompt)` method.
        tools: Optional list of tools (note: CoT agent doesn't typically use tools).
        config: Optional AgentConfig for customizing behavior.
        cot_prompt: Optional custom chain-of-thought prompt template.
            Must include {question} placeholder.

    Attributes:
        model: The LLM model instance.
        config: The agent configuration.
        memory: AgentMemory for execution history.
        _cot_prompt: The chain-of-thought prompt template.

    Examples:
        Basic usage:

            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel())
            >>> result = agent.run("If I have 3 apples and buy 5 more, how many do I have?")
            >>> print(result.status)
            AgentStatus.FINISHED

        With custom prompt:

            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> custom_prompt = '''Think carefully about this question:
            ... {question}
            ...
            ... Break it down:
            ... Step 1:
            ... Step 2:
            ... Step 3:
            ... Final Answer:'''
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel(), cot_prompt=custom_prompt)
            >>> result = agent.run("What is 15% of 200?")

        Examining the reasoning:

            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel())
            >>> result = agent.run("Solve: x + 5 = 12")
            >>>
            >>> # The reasoning is captured in the step's thought
            >>> if result.steps:
            ...     print(f"Reasoning: {result.steps[0].thought[:100]}...")

        Comparing with other agents:

            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent, ReActAgent, SimpleAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> # CoT: Best for reasoning without tools
            >>> cot = ChainOfThoughtAgent(DummyModel())
            >>>
            >>> # ReAct: Best for tasks requiring tool use
            >>> react = ReActAgent(DummyModel())
            >>>
            >>> # Simple: Best for straightforward tool invocations
            >>> simple = SimpleAgent(DummyModel())

    Note:
        Chain-of-thought reasoning works best for:
        - Mathematical problems
        - Logical reasoning tasks
        - Multi-step word problems
        - Tasks that benefit from explicit reasoning

    See Also:
        :class:`ReActAgent`: For tasks requiring tool use.
        :class:`SimpleAgent`: For simple, direct tasks.
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[list[Tool]] = None,
        config: Optional[AgentConfig] = None,
        cot_prompt: Optional[str] = None,
    ):
        """Initialize the Chain-of-Thought agent.

        Args:
            model: LLM model with generate() method.
            tools: Optional list of tools (rarely used with CoT).
            config: Optional configuration.
            cot_prompt: Optional custom prompt template with {question} placeholder.
        """
        super().__init__(model, tools, config)
        self._cot_prompt = cot_prompt or self._default_cot_prompt()

    def _default_cot_prompt(self) -> str:
        """Return the default chain-of-thought prompt template.

        Returns:
            str: The default CoT prompt with {question} placeholder.

        Examples:
            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel())
            >>> prompt = agent._default_cot_prompt()
            >>> print("{question}" in prompt)
            True
        """
        return """Let's solve this step by step:

Question: {question}

Please think through this carefully:
1. First, let me understand what is being asked...
2. Then, I'll break down the problem...
3. Now, let me work through each part...
4. Finally, I'll combine my findings...

Show your reasoning at each step, then provide your final answer prefixed with "Final Answer:"
"""

    def run(self, query: str, **kwargs: Any) -> AgentResult:
        """Run chain-of-thought reasoning on a query.

        Prompts the model with the CoT template and extracts the final answer
        from the response.

        Args:
            query: The question or problem to reason through.
            **kwargs: Additional arguments (currently unused).

        Returns:
            AgentResult: Contains the answer, reasoning trace, and status.

        Examples:
            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel())
            >>> result = agent.run("What is 2 + 2?")
            >>>
            >>> print(f"Query: {result.query}")
            Query: What is 2 + 2?
            >>> print(f"Answer: {result.answer}")
            >>> print(f"Status: {result.status}")
            AgentStatus.FINISHED
        """
        start_time = time.time()

        result = AgentResult(
            query=query,
            answer=None,
            status=AgentStatus.THINKING,
        )

        prompt = self._cot_prompt.format(question=query)
        response = self.model.generate(prompt)

        step = AgentStep(step_number=1, thought=response)

        # Extract final answer
        final_match = re.search(
            r"Final Answer:\s*(.+?)$",
            response,
            re.DOTALL | re.IGNORECASE,
        )

        if final_match:
            result.answer = final_match.group(1).strip()
        else:
            # Use last paragraph as answer
            paragraphs = response.strip().split("\n\n")
            result.answer = paragraphs[-1] if paragraphs else response

        result.status = AgentStatus.FINISHED
        result.steps.append(step)
        result.total_iterations = 1
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def _plan(self, query: str) -> str:
        """Generate the chain-of-thought reasoning for a query.

        Args:
            query: The question to reason about.

        Returns:
            str: The model's chain-of-thought response.

        Examples:
            >>> from insideLLMs.contrib.agents import ChainOfThoughtAgent
            >>> from insideLLMs import DummyModel
            >>>
            >>> agent = ChainOfThoughtAgent(DummyModel())
            >>> response = agent._plan("What is 5 + 5?")
            >>> print(type(response))
            <class 'str'>
        """
        return self.model.generate(self._cot_prompt.format(question=query))
