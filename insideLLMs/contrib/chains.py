"""Prompt chain and workflow orchestration for LLM pipelines.

This module provides a comprehensive framework for building complex multi-step
LLM workflows with support for sequential execution, parallel processing,
conditional branching, loops, and workflow composition.

Overview
--------
The chains module implements a flexible pipeline architecture for orchestrating
LLM calls and data transformations. It follows a step-based execution model
where each step can be an LLM call, data transformation, validation, conditional
branch, router, loop, or nested subchain.

Key Components
--------------
- **Chain**: The main orchestrator that executes a sequence of steps
- **ChainStep**: Abstract base class for all step types
- **ChainBuilder**: Fluent API for constructing chains
- **ChainRegistry**: Central registry for reusable chains
- **WorkflowTemplate**: Pre-built templates for common patterns

Step Types
----------
- **LLMStep**: Calls an LLM with a prompt template
- **TransformStep**: Applies a transformation function to data
- **ValidatorStep**: Validates data and controls chain flow
- **ConditionalStep**: Branches based on a condition
- **RouterStep**: Routes to different steps based on content classification
- **LoopStep**: Repeats a step until an exit condition is met
- **ParallelStep**: Executes multiple steps and aggregates results
- **SubchainStep**: Embeds another chain as a step

State Management
----------------
ChainState maintains variables and step results across execution, allowing
steps to share data and enabling complex workflows with dependencies.

Examples
--------
Basic sequential chain with LLM calls:

>>> from insideLLMs.contrib.chains import create_chain
>>>
>>> def mock_llm(prompt: str) -> str:
...     return f"Response to: {prompt[:50]}..."
>>>
>>> chain = (
...     create_chain("summarize_pipeline", mock_llm)
...     .llm("extract", "Extract key points from: {input}")
...     .llm("summarize", "Summarize these points: {input}")
...     .build()
... )
>>>
>>> result = chain.run("Long document text here...")
>>> print(result.success)
True

Chain with validation and error handling:

>>> def validate_length(text):
...     if len(text) < 10:
...         return False, "Response too short"
...     return True, None
>>>
>>> chain = (
...     create_chain("validated_chain", mock_llm)
...     .llm("generate", "Generate a response for: {input}")
...     .validate("check_length", validate_length, on_failure="fail")
...     .build()
... )

Conditional branching based on content:

>>> def is_question(data, state):
...     return "?" in str(data)
>>>
>>> from insideLLMs.contrib.chains import TransformStep
>>> answer_step = TransformStep("answer", lambda d, s: f"Answer: {d}")
>>> statement_step = TransformStep("acknowledge", lambda d, s: f"Noted: {d}")
>>>
>>> chain = (
...     create_chain("branching_chain")
...     .branch("route", is_question, answer_step, statement_step)
...     .build()
... )

Router with keyword-based classification:

>>> from insideLLMs.contrib.chains import LLMStep, RouterStrategy
>>>
>>> routes = {
...     "technical": LLMStep("tech", "Explain technically: {input}", mock_llm),
...     "simple": LLMStep("simple", "Explain simply: {input}", mock_llm),
... }
>>>
>>> chain = (
...     create_chain("router_chain")
...     .route("classify", routes, RouterStrategy.KEYWORD, default_route="simple")
...     .build()
... )

Loop with exit condition:

>>> from insideLLMs.contrib.chains import TransformStep
>>>
>>> refine_step = TransformStep("refine", lambda d, s: d + " [refined]")
>>>
>>> def should_exit(data, state, iteration):
...     return iteration >= 3 or "[refined]" * 2 in data
>>>
>>> chain = (
...     create_chain("refinement_loop")
...     .loop("refine", refine_step, should_exit, max_iterations=5)
...     .build()
... )

Using pre-built workflow templates:

>>> from insideLLMs.contrib.chains import WorkflowTemplate
>>>
>>> chain = WorkflowTemplate.summarize_and_answer(mock_llm)
>>> result = chain.run("Document to summarize and answer questions about")

Parallel execution with aggregation:

>>> from insideLLMs.contrib.chains import TransformStep
>>>
>>> steps = [
...     TransformStep("upper", lambda d, s: d.upper()),
...     TransformStep("lower", lambda d, s: d.lower()),
...     TransformStep("title", lambda d, s: d.title()),
... ]
>>>
>>> chain = (
...     create_chain("parallel_transforms")
...     .parallel("variants", steps, aggregator=lambda r: {"variants": r})
...     .build()
... )

Notes
-----
- Steps are executed sequentially by default; ParallelStep simulates parallelism
- Chain state persists across steps and can be used for inter-step communication
- The ChainBuilder provides a fluent API for chain construction
- Pre/post hooks allow instrumentation and logging
- Error handlers can recover from failures or provide fallback behavior

See Also
--------
insideLLMs.prompts : Prompt template construction and management
insideLLMs.tracer : Tracing and observability for chain execution
insideLLMs.cache : Caching for LLM responses within chains
"""

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ChainStatus(Enum):
    """Status of a chain or step execution.

    This enumeration represents the lifecycle states of both chains and
    individual steps during execution. It enables tracking of execution
    progress and handling of different completion states.

    Attributes
    ----------
    PENDING : str
        Initial state before execution begins. Steps and chains start in
        this state when created but not yet run.
    RUNNING : str
        Active execution state. The step or chain is currently processing.
    COMPLETED : str
        Successful completion. The execution finished without errors.
    FAILED : str
        Execution failed due to an error. Check the error field for details.
    SKIPPED : str
        Execution was skipped, typically due to a conditional branch or
        validation failure with on_failure="skip".
    CANCELLED : str
        Execution was cancelled before completion, either manually or due
        to a timeout.

    Examples
    --------
    Check if a chain completed successfully:

    >>> result = chain.run("input data")
    >>> if result.status == ChainStatus.COMPLETED:
    ...     print("Chain finished successfully")
    ... elif result.status == ChainStatus.FAILED:
    ...     print(f"Chain failed with errors: {result.errors}")

    Use status in conditional logic:

    >>> for step_result in chain_state.step_results:
    ...     if step_result.status == ChainStatus.SKIPPED:
    ...         print(f"Step {step_result.step_name} was skipped")

    See Also
    --------
    ChainResult : Contains the final status of a chain execution
    StepResult : Contains the status of individual step executions
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ChainStepType(Enum):
    """Types of chain steps for categorization and logging.

    This enumeration categorizes the different types of steps that can be
    included in a chain. It is used for logging, debugging, and analytics
    to understand chain composition and execution patterns.

    Attributes
    ----------
    LLM_CALL : str
        A step that invokes a language model with a prompt template.
        Used by LLMStep to track model invocations.
    TRANSFORM : str
        A step that transforms data using a custom function.
        Used by TransformStep for data manipulation.
    CONDITION : str
        A step that conditionally branches based on input or state.
        Used by ConditionalStep for if/else logic.
    PARALLEL : str
        A step that executes multiple sub-steps (simulated parallelism).
        Used by ParallelStep for fan-out operations.
    LOOP : str
        A step that repeats execution until a condition is met.
        Used by LoopStep for iterative refinement.
    SUBCHAIN : str
        A step that embeds and executes another chain.
        Used by SubchainStep for workflow composition.
    VALIDATOR : str
        A step that validates data and can halt or modify chain flow.
        Used by ValidatorStep for quality gates.
    ROUTER : str
        A step that routes to different sub-steps based on classification.
        Used by RouterStep for content-based routing.

    Examples
    --------
    Check step type in results:

    >>> for result in chain_state.step_results:
    ...     if result.step_type == ChainStepType.LLM_CALL:
    ...         print(f"LLM step '{result.step_name}' took {result.duration:.2f}s")

    Filter steps by type for analysis:

    >>> llm_steps = [
    ...     r for r in chain_state.step_results
    ...     if r.step_type == ChainStepType.LLM_CALL
    ... ]
    >>> total_llm_time = sum(s.duration for s in llm_steps)

    See Also
    --------
    ChainStep : Base class that uses step_type attribute
    StepResult : Contains step_type for executed steps
    """

    LLM_CALL = "llm_call"
    TRANSFORM = "transform"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    SUBCHAIN = "subchain"
    VALIDATOR = "validator"
    ROUTER = "router"


class RouterStrategy(Enum):
    """Strategies for routing in conditional chains.

    This enumeration defines the available strategies for RouterStep to
    classify input and determine which route to take. Each strategy provides
    a different mechanism for content-based routing.

    Attributes
    ----------
    KEYWORD : str
        Simple keyword matching. Checks if route keys appear as substrings
        in the input (case-insensitive). Fast but limited to exact matches.
    REGEX : str
        Regular expression matching. Route keys are treated as regex patterns
        and matched against the input. More flexible than keyword matching.
    CLASSIFIER : str
        Uses a custom classifier function to categorize input. The function
        should return a route key. Useful for ML-based classification.
    CUSTOM : str
        Fully custom routing logic via a classifier function. Allows arbitrary
        routing decisions based on input analysis.

    Examples
    --------
    Keyword-based routing (simple substring matching):

    >>> from insideLLMs.contrib.chains import RouterStep, RouterStrategy, TransformStep
    >>>
    >>> routes = {
    ...     "error": TransformStep("handle_error", lambda d, s: f"Error: {d}"),
    ...     "warning": TransformStep("handle_warn", lambda d, s: f"Warning: {d}"),
    ... }
    >>> router = RouterStep("log_router", routes, RouterStrategy.KEYWORD)
    >>> # Input "error occurred" matches "error" route

    Regex-based routing (pattern matching):

    >>> routes = {
    ...     r"\\d{3}-\\d{4}": TransformStep("phone", lambda d, s: f"Phone: {d}"),
    ...     r"[\\w.]+@[\\w.]+": TransformStep("email", lambda d, s: f"Email: {d}"),
    ... }
    >>> router = RouterStep("contact_router", routes, RouterStrategy.REGEX)

    Classifier-based routing (custom function):

    >>> def sentiment_classifier(text):
    ...     if "happy" in text.lower() or "great" in text.lower():
    ...         return "positive"
    ...     elif "sad" in text.lower() or "bad" in text.lower():
    ...         return "negative"
    ...     return "neutral"
    >>>
    >>> routes = {
    ...     "positive": TransformStep("pos", lambda d, s: "Glad to hear!"),
    ...     "negative": TransformStep("neg", lambda d, s: "Sorry to hear that."),
    ...     "neutral": TransformStep("neu", lambda d, s: "I see."),
    ... }
    >>> router = RouterStep(
    ...     "sentiment_router", routes, RouterStrategy.CLASSIFIER,
    ...     classifier_fn=sentiment_classifier
    ... )

    See Also
    --------
    RouterStep : Step class that uses these strategies
    ConditionalStep : Alternative for simple binary branching
    """

    KEYWORD = "keyword"
    REGEX = "regex"
    CLASSIFIER = "classifier"
    CUSTOM = "custom"


@dataclass
class StepResult:
    """Result from a single chain step execution.

    StepResult captures all information about a completed step execution,
    including timing, input/output data, status, and any errors. It is
    automatically created by the Chain during execution and stored in
    ChainState.step_results.

    Parameters
    ----------
    step_name : str
        The name identifier of the step that was executed.
    step_type : ChainStepType
        The type classification of the step (LLM_CALL, TRANSFORM, etc.).
    status : ChainStatus
        The execution status (COMPLETED, FAILED, SKIPPED, etc.).
    input_data : Any
        The data that was passed into the step.
    output_data : Any
        The data produced by the step (None if failed).
    start_time : float
        Unix timestamp when step execution began.
    end_time : float
        Unix timestamp when step execution completed.
    error : Optional[str], default=None
        Error message if the step failed, None otherwise.
    metadata : dict[str, Any], default={}
        Additional metadata about the step execution.

    Attributes
    ----------
    duration : float
        Computed property returning execution time in seconds.
    success : bool
        Computed property indicating if the step completed successfully.

    Examples
    --------
    Access step results after chain execution:

    >>> result = chain.run("input data")
    >>> for step in result.state.step_results:
    ...     print(f"{step.step_name}: {step.status.value} ({step.duration:.3f}s)")
    extract: completed (0.523s)
    transform: completed (0.001s)
    validate: completed (0.002s)

    Check for failures and get error details:

    >>> failed_steps = [s for s in result.state.step_results if not s.success]
    >>> for step in failed_steps:
    ...     print(f"Step '{step.step_name}' failed: {step.error}")

    Convert to dictionary for logging or serialization:

    >>> step_result = result.state.step_results[0]
    >>> step_dict = step_result.to_dict()
    >>> print(step_dict["step_name"], step_dict["duration"])

    Analyze LLM step performance:

    >>> llm_results = [
    ...     s for s in result.state.step_results
    ...     if s.step_type == ChainStepType.LLM_CALL
    ... ]
    >>> avg_llm_time = sum(s.duration for s in llm_results) / len(llm_results)
    >>> print(f"Average LLM call time: {avg_llm_time:.2f}s")

    See Also
    --------
    ChainState : Contains the list of StepResults
    ChainResult : Contains the final result including all step results
    ChainStatus : Possible status values for steps
    """

    step_name: str
    step_type: ChainStepType
    status: ChainStatus
    input_data: Any
    output_data: Any
    start_time: float
    end_time: float
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the step result to a dictionary representation.

        Creates a serializable dictionary containing all step result
        information. Input and output data are truncated to 500 characters
        for readability in logs and debugging output.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: step_name, step_type, status, input_data,
            output_data, duration, error, metadata.

        Examples
        --------
        >>> step_result = StepResult(
        ...     step_name="extract",
        ...     step_type=ChainStepType.LLM_CALL,
        ...     status=ChainStatus.COMPLETED,
        ...     input_data="Long input text...",
        ...     output_data="Extracted content...",
        ...     start_time=1000.0,
        ...     end_time=1000.5,
        ... )
        >>> d = step_result.to_dict()
        >>> print(d["step_name"], d["duration"])
        extract 0.5
        """
        return {
            "step_name": self.step_name,
            "step_type": self.step_type.value,
            "status": self.status.value,
            "input_data": str(self.input_data)[:500],  # Truncate for readability
            "output_data": str(self.output_data)[:500],
            "duration": self.duration,
            "error": self.error,
            "metadata": self.metadata,
        }

    @property
    def duration(self) -> float:
        """Duration of step execution in seconds.

        Computed as the difference between end_time and start_time.

        Returns
        -------
        float
            Execution duration in seconds.

        Examples
        --------
        >>> step_result = StepResult(
        ...     step_name="test", step_type=ChainStepType.TRANSFORM,
        ...     status=ChainStatus.COMPLETED, input_data="x", output_data="y",
        ...     start_time=100.0, end_time=100.5
        ... )
        >>> step_result.duration
        0.5
        """
        return self.end_time - self.start_time

    @property
    def success(self) -> bool:
        """Whether the step completed successfully.

        Returns
        -------
        bool
            True if status is COMPLETED, False otherwise.

        Examples
        --------
        >>> step_result = StepResult(
        ...     step_name="test", step_type=ChainStepType.TRANSFORM,
        ...     status=ChainStatus.COMPLETED, input_data="x", output_data="y",
        ...     start_time=100.0, end_time=100.5
        ... )
        >>> step_result.success
        True

        >>> failed_result = StepResult(
        ...     step_name="test", step_type=ChainStepType.TRANSFORM,
        ...     status=ChainStatus.FAILED, input_data="x", output_data=None,
        ...     start_time=100.0, end_time=100.5, error="Something went wrong"
        ... )
        >>> failed_result.success
        False
        """
        return self.status == ChainStatus.COMPLETED


@dataclass
class ChainState:
    """State maintained across chain execution.

    ChainState serves as the shared context that persists throughout chain
    execution. It provides variable storage for inter-step communication,
    tracks execution progress, and accumulates step results for analysis.

    Parameters
    ----------
    variables : dict[str, Any], default={}
        Key-value store for sharing data between steps. Steps can read
        and write variables to pass information along the chain.
    step_results : list[StepResult], default=[]
        Accumulated results from each executed step. Populated automatically
        during chain execution.
    current_step : int, default=0
        Index of the currently executing step (0-based).
    start_time : Optional[float], default=None
        Unix timestamp when chain execution began. Set automatically.
    end_time : Optional[float], default=None
        Unix timestamp when chain execution completed. Set automatically.
    status : ChainStatus, default=PENDING
        Current execution status of the chain.

    Attributes
    ----------
    variables : dict[str, Any]
        The variable store accessible via get() and set() methods.
    step_results : list[StepResult]
        List of completed step results.

    Notes
    -----
    Special variable names prefixed with underscore are used internally:

    - ``_skip_remaining``: When True, skips remaining steps
    - ``_validation_error``: Contains the last validation error message
    - ``_validation_failed``: True if validation has failed
    - ``_loop_iteration``: Current iteration count in a loop
    - ``_loop_total_iterations``: Total iterations completed in a loop
    - ``_selected_route``: The route key selected by a RouterStep

    Examples
    --------
    Create and use state for variable sharing:

    >>> state = ChainState()
    >>> state.set("user_id", "12345")
    >>> state.set("context", {"key": "value"})
    >>> print(state.get("user_id"))
    12345
    >>> print(state.get("missing_key", "default"))
    default

    Access state within a transform step:

    >>> def transform_with_state(data, state):
    ...     user_id = state.get("user_id")
    ...     result = f"Processing for user {user_id}: {data}"
    ...     state.set("processed_count", state.get("processed_count", 0) + 1)
    ...     return result

    Initialize chain with pre-populated state:

    >>> initial_state = ChainState()
    >>> initial_state.set("config", {"max_length": 1000})
    >>> initial_state.set("user_preferences", {"language": "en"})
    >>> result = chain.run("input", initial_state=initial_state)

    Inspect execution results:

    >>> result = chain.run("input")
    >>> state = result.state
    >>> print(f"Executed {len(state.step_results)} steps")
    >>> print(f"Final status: {state.status.value}")
    >>> print(f"Duration: {state.end_time - state.start_time:.2f}s")

    Get the last successful output:

    >>> last_output = state.get_last_output()
    >>> if last_output:
    ...     print(f"Last output: {last_output}")

    See Also
    --------
    ChainResult : Contains the state after chain completion
    StepResult : Individual step results stored in step_results
    Chain.run : Method that creates and populates ChainState
    """

    variables: dict[str, Any] = field(default_factory=dict)
    step_results: list[StepResult] = field(default_factory=list)
    current_step: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: ChainStatus = ChainStatus.PENDING

    def set(self, key: str, value: Any) -> None:
        """Set a variable in the state.

        Stores a value that can be accessed by subsequent steps in the
        chain. Variables persist throughout the entire chain execution.

        Args
        ----
        key : str
            The variable name. Can be any string; names starting with
            underscore are reserved for internal use.
        value : Any
            The value to store. Can be any Python object.

        Examples
        --------
        >>> state = ChainState()
        >>> state.set("counter", 0)
        >>> state.set("results", [])
        >>> state.get("counter")
        0

        Use in a transform step to accumulate data:

        >>> def accumulating_transform(data, state):
        ...     results = state.get("results", [])
        ...     results.append(data)
        ...     state.set("results", results)
        ...     return data
        """
        self.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable from the state.

        Retrieves a previously stored variable value, or returns a default
        if the variable does not exist.

        Args
        ----
        key : str
            The variable name to retrieve.
        default : Any, optional
            Value to return if the key is not found. Defaults to None.

        Returns
        -------
        Any
            The stored value, or the default if not found.

        Examples
        --------
        >>> state = ChainState()
        >>> state.set("name", "Alice")
        >>> state.get("name")
        'Alice'
        >>> state.get("age", 25)
        25
        >>> state.get("missing") is None
        True

        Safely access nested data:

        >>> state.set("config", {"timeout": 30})
        >>> config = state.get("config", {})
        >>> timeout = config.get("timeout", 60)
        """
        return self.variables.get(key, default)

    def get_last_output(self) -> Any:
        """Get the output from the last successfully completed step.

        Searches through step results in reverse order to find the most
        recent step that completed successfully, and returns its output.

        Returns
        -------
        Any
            The output_data from the last successful step, or None if
            no steps have completed successfully.

        Examples
        --------
        >>> result = chain.run("input")
        >>> last_output = result.state.get_last_output()
        >>> if last_output:
        ...     print(f"Final result: {last_output}")
        ... else:
        ...     print("No successful steps")

        Use for error recovery:

        >>> def error_handler(error, state):
        ...     # Return the last good output instead of failing
        ...     return state.get_last_output() or "default value"
        """
        for result in reversed(self.step_results):
            if result.success:
                return result.output_data
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert the state to a dictionary representation.

        Creates a serializable dictionary containing state information.
        Variable values are truncated to 200 characters for readability.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: variables, step_count, current_step,
            status, duration.

        Examples
        --------
        >>> state = ChainState()
        >>> state.set("key", "value")
        >>> state.status = ChainStatus.COMPLETED
        >>> d = state.to_dict()
        >>> print(d["status"])
        completed

        Log state for debugging:

        >>> import json
        >>> print(json.dumps(state.to_dict(), indent=2))
        """
        return {
            "variables": {k: str(v)[:200] for k, v in self.variables.items()},
            "step_count": len(self.step_results),
            "current_step": self.current_step,
            "status": self.status.value,
            "duration": (self.end_time - self.start_time)
            if self.end_time and self.start_time
            else None,
        }


@dataclass
class ChainResult:
    """Result from a complete chain execution.

    ChainResult encapsulates all information about a chain's execution,
    including the final output, execution statistics, timing, and any
    errors encountered. It is returned by Chain.run() after execution
    completes (or fails).

    Parameters
    ----------
    chain_name : str
        The name of the chain that was executed.
    status : ChainStatus
        The final execution status (COMPLETED, FAILED, etc.).
    final_output : Any
        The output from the last successfully executed step, or None
        if the chain failed before any steps completed.
    state : ChainState
        The complete execution state including all variables and step
        results. Provides access to intermediate outputs and timing.
    total_steps : int
        The total number of steps in the chain definition.
    successful_steps : int
        The number of steps that completed successfully.
    failed_steps : int
        The number of steps that failed during execution.
    total_duration : float
        Total execution time in seconds from start to finish.
    errors : list[str], default=[]
        List of error messages from failed steps.

    Attributes
    ----------
    success : bool
        Computed property indicating if the chain completed successfully.

    Examples
    --------
    Basic usage after chain execution:

    >>> result = chain.run("input data")
    >>> if result.success:
    ...     print(f"Output: {result.final_output}")
    ... else:
    ...     print(f"Failed with errors: {result.errors}")

    Access detailed execution statistics:

    >>> result = chain.run("input")
    >>> print(f"Chain '{result.chain_name}' completed in {result.total_duration:.2f}s")
    >>> print(f"Steps: {result.successful_steps}/{result.total_steps} succeeded")
    >>> if result.failed_steps > 0:
    ...     print(f"Failures: {result.errors}")

    Access intermediate results through state:

    >>> result = chain.run("input")
    >>> for step_result in result.state.step_results:
    ...     print(f"  {step_result.step_name}: {step_result.status.value}")
    ...     if step_result.success:
    ...         print(f"    Output: {str(step_result.output_data)[:100]}")

    Access stored variables:

    >>> result = chain.run("input")
    >>> accumulated_data = result.state.get("accumulated_results", [])
    >>> metrics = result.state.get("metrics", {})

    Convert to dictionary for logging or API responses:

    >>> result_dict = result.to_dict()
    >>> import json
    >>> print(json.dumps(result_dict, indent=2))

    Error handling pattern:

    >>> result = chain.run("input")
    >>> if not result.success:
    ...     for error in result.errors:
    ...         logging.error(f"Chain error: {error}")
    ...     # Attempt recovery or fallback
    ...     fallback_output = handle_chain_failure(result)

    See Also
    --------
    Chain.run : Method that returns ChainResult
    ChainState : Detailed state accessible via result.state
    ChainStatus : Possible status values
    """

    chain_name: str
    status: ChainStatus
    final_output: Any
    state: ChainState
    total_steps: int
    successful_steps: int
    failed_steps: int
    total_duration: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the chain result to a dictionary representation.

        Creates a serializable dictionary containing the chain result
        information. The final_output is truncated to 1000 characters
        for readability.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: chain_name, status, final_output,
            total_steps, successful_steps, failed_steps, total_duration,
            errors.

        Examples
        --------
        >>> result = chain.run("input")
        >>> d = result.to_dict()
        >>> print(f"Chain: {d['chain_name']}, Status: {d['status']}")

        Serialize for logging:

        >>> import json
        >>> log_entry = json.dumps(result.to_dict())
        >>> logger.info(f"Chain completed: {log_entry}")

        API response formatting:

        >>> @app.route("/process")
        ... def process():
        ...     result = chain.run(request.data)
        ...     return jsonify(result.to_dict())
        """
        return {
            "chain_name": self.chain_name,
            "status": self.status.value,
            "final_output": str(self.final_output)[:1000],
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "total_duration": self.total_duration,
            "errors": self.errors,
        }

    @property
    def success(self) -> bool:
        """Whether the chain completed successfully.

        Returns
        -------
        bool
            True if the chain status is COMPLETED, False otherwise.

        Examples
        --------
        >>> result = chain.run("input")
        >>> if result.success:
        ...     process_output(result.final_output)
        ... else:
        ...     handle_failure(result.errors)

        Use in conditional expressions:

        >>> output = result.final_output if result.success else "default"
        """
        return self.status == ChainStatus.COMPLETED


class ChainStep(ABC):
    """Abstract base class for chain steps.

    ChainStep defines the interface that all step implementations must follow.
    It provides the foundation for creating custom step types that can be
    added to chains. Each step receives input data and the chain state,
    performs some operation, and returns output data.

    Parameters
    ----------
    name : str
        A unique identifier for the step within the chain. Used for
        logging, debugging, and accessing step results.
    description : str, default=""
        An optional human-readable description of what the step does.

    Attributes
    ----------
    name : str
        The step's identifier.
    description : str
        The step's description.
    step_type : ChainStepType
        The type classification of the step. Subclasses should set this
        to the appropriate value.

    Notes
    -----
    To create a custom step type:

    1. Inherit from ChainStep
    2. Set self.step_type in __init__
    3. Implement the execute() method
    4. Optionally override validate_input() for input validation

    Examples
    --------
    Create a custom step that formats text:

    >>> class FormatStep(ChainStep):
    ...     def __init__(self, name: str, format_string: str):
    ...         super().__init__(name, f"Formats using: {format_string}")
    ...         self.format_string = format_string
    ...         self.step_type = ChainStepType.TRANSFORM
    ...
    ...     def execute(self, input_data: Any, state: ChainState) -> Any:
    ...         return self.format_string.format(data=input_data)
    ...
    ...     def validate_input(self, input_data: Any) -> tuple[bool, Optional[str]]:
    ...         if input_data is None:
    ...             return False, "Input cannot be None"
    ...         return True, None

    Use the custom step in a chain:

    >>> format_step = FormatStep("format", "Processed: {data}")
    >>> chain = Chain("example")
    >>> chain.add_step(format_step)

    Create a step with state manipulation:

    >>> class CountingStep(ChainStep):
    ...     def __init__(self, name: str):
    ...         super().__init__(name, "Counts executions")
    ...         self.step_type = ChainStepType.TRANSFORM
    ...
    ...     def execute(self, input_data: Any, state: ChainState) -> Any:
    ...         count = state.get("execution_count", 0)
    ...         state.set("execution_count", count + 1)
    ...         return input_data

    See Also
    --------
    LLMStep : Step for LLM calls
    TransformStep : Step for data transformations
    ValidatorStep : Step for validation
    ConditionalStep : Step for conditional branching
    RouterStep : Step for content-based routing
    LoopStep : Step for iterative execution
    ParallelStep : Step for parallel execution
    SubchainStep : Step for nested chains
    """

    def __init__(self, name: str, description: str = ""):
        """Initialize a chain step.

        Args
        ----
        name : str
            A unique identifier for the step within the chain.
        description : str, default=""
            Optional human-readable description of the step.
        """
        self.name = name
        self.description = description
        self.step_type = ChainStepType.TRANSFORM

    @abstractmethod
    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute the step's logic.

        This method must be implemented by all subclasses. It receives
        the input data from the previous step (or chain input for the
        first step) and the current chain state.

        Args
        ----
        input_data : Any
            The data to process. For the first step, this is the chain's
            input. For subsequent steps, this is the output of the
            previous step.
        state : ChainState
            The chain's execution state. Can be used to read/write
            variables and access previous step results.

        Returns
        -------
        Any
            The output data, which becomes the input for the next step.

        Raises
        ------
        Exception
            Any exception raised will cause the step to fail. The chain's
            error handler (if set) will be invoked.

        Examples
        --------
        Simple transformation:

        >>> def execute(self, input_data: Any, state: ChainState) -> Any:
        ...     return input_data.upper()

        Using state variables:

        >>> def execute(self, input_data: Any, state: ChainState) -> Any:
        ...     prefix = state.get("prefix", "")
        ...     result = f"{prefix}{input_data}"
        ...     state.set("last_processed", result)
        ...     return result
        """
        pass

    def validate_input(self, input_data: Any) -> tuple[bool, Optional[str]]:
        """Validate input data before execution.

        Override this method to add input validation. If validation fails,
        the step will not execute and the chain will handle the error
        according to its configuration.

        Args
        ----
        input_data : Any
            The input data to validate.

        Returns
        -------
        tuple[bool, Optional[str]]
            A tuple of (is_valid, error_message). If is_valid is False,
            error_message should describe why validation failed.

        Examples
        --------
        Validate that input is a non-empty string:

        >>> def validate_input(self, input_data: Any) -> tuple[bool, Optional[str]]:
        ...     if not isinstance(input_data, str):
        ...         return False, f"Expected str, got {type(input_data).__name__}"
        ...     if not input_data.strip():
        ...         return False, "Input cannot be empty"
        ...     return True, None

        Validate numeric ranges:

        >>> def validate_input(self, input_data: Any) -> tuple[bool, Optional[str]]:
        ...     if not isinstance(input_data, (int, float)):
        ...         return False, "Input must be numeric"
        ...     if input_data < 0:
        ...         return False, "Input must be non-negative"
        ...     return True, None
        """
        return True, None


class LLMStep(ChainStep):
    """Step that calls a language model with a formatted prompt.

    LLMStep is the primary step type for interacting with language models.
    It takes a prompt template, formats it with the input data and state
    variables, calls the model, and optionally parses the response.

    Parameters
    ----------
    name : str
        A unique identifier for the step within the chain.
    prompt_template : str
        The prompt template with placeholders for variable substitution.
        Use {input} for the step's input data, and {variable_name} for
        any variables stored in the chain state.
    model_fn : Callable[[str], str]
        A function that takes a prompt string and returns the model's
        response. This allows flexibility in which model backend is used.
    output_parser : Optional[Callable[[str], Any]], default=None
        An optional function to parse/transform the model's response.
        If provided, the parser's output becomes the step's output.
    description : str, default=""
        Optional human-readable description of the step.

    Attributes
    ----------
    prompt_template : str
        The prompt template used for formatting.
    model_fn : Callable[[str], str]
        The model function for generating responses.
    output_parser : Optional[Callable[[str], Any]]
        The optional output parser function.

    Notes
    -----
    Template substitution uses simple string replacement with {key} syntax.
    Available variables include:

    - {input}: The input data passed to the step
    - Any variable stored in state.variables

    For more complex templating needs, consider using a TransformStep to
    prepare the prompt before the LLMStep.

    Examples
    --------
    Basic LLM step for summarization:

    >>> def call_gpt(prompt: str) -> str:
    ...     # Your model API call here
    ...     return openai.chat(prompt)
    >>>
    >>> summarize = LLMStep(
    ...     name="summarize",
    ...     prompt_template="Summarize the following text in 3 sentences:\\n{input}",
    ...     model_fn=call_gpt,
    ... )
    >>> chain = Chain("summarizer")
    >>> chain.add_step(summarize)

    Using state variables in the prompt:

    >>> # Set up initial state with context
    >>> state = ChainState()
    >>> state.set("language", "Spanish")
    >>> state.set("tone", "formal")
    >>>
    >>> translate = LLMStep(
    ...     name="translate",
    ...     prompt_template="Translate to {language} in a {tone} tone:\\n{input}",
    ...     model_fn=call_gpt,
    ... )
    >>> result = chain.run("Hello, world!", initial_state=state)

    With output parsing for structured data:

    >>> import json
    >>>
    >>> def parse_json(response: str) -> dict:
    ...     # Extract JSON from response
    ...     return json.loads(response)
    >>>
    >>> extract_entities = LLMStep(
    ...     name="extract",
    ...     prompt_template=\"\"\"Extract named entities from the text as JSON:
    ...     Text: {input}
    ...     Output format: {{"people": [...], "places": [...], "dates": [...]}}\"\"\",
    ...     model_fn=call_gpt,
    ...     output_parser=parse_json,
    ... )

    Chaining multiple LLM calls:

    >>> chain = (
    ...     create_chain("multi_step", call_gpt)
    ...     .llm("draft", "Write a first draft about: {input}")
    ...     .llm("review", "Review and improve this draft: {input}")
    ...     .llm("polish", "Polish and finalize: {input}")
    ...     .build()
    ... )

    See Also
    --------
    ChainBuilder.llm : Convenient method for adding LLMStep to chains
    TransformStep : For custom data transformations
    create_llm_step : Factory function for creating LLMStep instances
    """

    def __init__(
        self,
        name: str,
        prompt_template: str,
        model_fn: Callable[[str], str],
        output_parser: Optional[Callable[[str], Any]] = None,
        description: str = "",
    ):
        """Initialize an LLM step.

        Args
        ----
        name : str
            A unique identifier for the step.
        prompt_template : str
            Template string with {placeholders} for variable substitution.
        model_fn : Callable[[str], str]
            Function that calls the LLM and returns its response.
        output_parser : Optional[Callable[[str], Any]], default=None
            Optional function to parse the model response.
        description : str, default=""
            Optional description of the step.
        """
        super().__init__(name, description)
        self.step_type = ChainStepType.LLM_CALL
        self.prompt_template = prompt_template
        self.model_fn = model_fn
        self.output_parser = output_parser

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute the LLM call with the formatted prompt.

        Formats the prompt template by substituting {input} with the
        input_data and any state variables, then calls the model function
        and optionally parses the response.

        Args
        ----
        input_data : Any
            The input data to include in the prompt as {input}.
        state : ChainState
            The chain state containing variables for template substitution.

        Returns
        -------
        Any
            The model's response, optionally parsed if output_parser is set.

        Raises
        ------
        Exception
            Re-raises any exception from the model function or parser.

        Examples
        --------
        >>> step = LLMStep("test", "Echo: {input}", lambda x: f"Response: {x}")
        >>> state = ChainState()
        >>> result = step.execute("hello", state)
        >>> print(result)
        Response: Echo: hello
        """
        # Format prompt with input and state variables
        format_vars = {"input": input_data, **state.variables}

        # Simple template substitution
        prompt = self.prompt_template
        for key, value in format_vars.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        # Call model
        response = self.model_fn(prompt)

        # Parse output if parser provided
        if self.output_parser:
            return self.output_parser(response)

        return response


class TransformStep(ChainStep):
    """Step that transforms data using a custom function.

    TransformStep provides maximum flexibility for data manipulation within
    a chain. It accepts any function that takes input data and the chain
    state, allowing for arbitrary transformations, filtering, aggregation,
    or side effects.

    Parameters
    ----------
    name : str
        A unique identifier for the step within the chain.
    transform_fn : Callable[[Any, ChainState], Any]
        A function that receives the input data and chain state, and
        returns the transformed output. The function signature must be
        `fn(input_data: Any, state: ChainState) -> Any`.
    description : str, default=""
        Optional human-readable description of the step.

    Attributes
    ----------
    transform_fn : Callable[[Any, ChainState], Any]
        The transformation function.

    Examples
    --------
    Simple text transformation:

    >>> uppercase = TransformStep(
    ...     name="uppercase",
    ...     transform_fn=lambda data, state: data.upper(),
    ... )

    Extract specific fields from structured data:

    >>> def extract_summary(data, state):
    ...     if isinstance(data, dict):
    ...         return data.get("summary", data.get("content", str(data)))
    ...     return str(data)
    >>>
    >>> extract = TransformStep("extract_summary", extract_summary)

    Transform with state access:

    >>> def add_metadata(data, state):
    ...     return {
    ...         "content": data,
    ...         "user_id": state.get("user_id"),
    ...         "timestamp": time.time(),
    ...         "step_count": len(state.step_results),
    ...     }
    >>>
    >>> metadata_step = TransformStep("add_metadata", add_metadata)

    Store intermediate results in state:

    >>> def cache_result(data, state):
    ...     # Store for later use by other steps
    ...     results = state.get("all_results", [])
    ...     results.append(data)
    ...     state.set("all_results", results)
    ...     return data
    >>>
    >>> cache_step = TransformStep("cache", cache_result)

    Filter and clean data:

    >>> def clean_text(data, state):
    ...     text = str(data)
    ...     # Remove extra whitespace
    ...     text = " ".join(text.split())
    ...     # Limit length
    ...     max_len = state.get("max_length", 1000)
    ...     if len(text) > max_len:
    ...         text = text[:max_len] + "..."
    ...     return text
    >>>
    >>> clean_step = TransformStep("clean", clean_text)

    Combine data from state:

    >>> def merge_results(data, state):
    ...     original = state.get("original_input")
    ...     return {
    ...         "original": original,
    ...         "processed": data,
    ...         "comparison": len(str(data)) / len(str(original))
    ...     }
    >>>
    >>> merge_step = TransformStep("merge", merge_results)

    Using lambdas for simple transforms:

    >>> chain = (
    ...     create_chain("pipeline")
    ...     .transform("lower", lambda d, s: d.lower())
    ...     .transform("strip", lambda d, s: d.strip())
    ...     .transform("split", lambda d, s: d.split())
    ...     .build()
    ... )

    See Also
    --------
    LLMStep : For LLM-based transformations
    ValidatorStep : For validation with flow control
    ChainBuilder.transform : Convenient method for adding transforms
    create_transform_step : Factory function for creating TransformStep
    """

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[Any, ChainState], Any],
        description: str = "",
    ):
        """Initialize a transform step.

        Args
        ----
        name : str
            A unique identifier for the step.
        transform_fn : Callable[[Any, ChainState], Any]
            Function that transforms the input data.
        description : str, default=""
            Optional description of the step.
        """
        super().__init__(name, description)
        self.step_type = ChainStepType.TRANSFORM
        self.transform_fn = transform_fn

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute the transformation function.

        Calls the transform function with the input data and state,
        returning the transformed result.

        Args
        ----
        input_data : Any
            The data to transform.
        state : ChainState
            The chain state for accessing/storing variables.

        Returns
        -------
        Any
            The transformed data returned by transform_fn.

        Examples
        --------
        >>> step = TransformStep("double", lambda d, s: d * 2)
        >>> state = ChainState()
        >>> step.execute(5, state)
        10
        >>> step.execute("ab", state)
        'abab'
        """
        return self.transform_fn(input_data, state)


class ValidatorStep(ChainStep):
    """Step that validates data and controls chain flow on failure.

    ValidatorStep acts as a quality gate within a chain. It evaluates the
    input data against validation criteria and determines whether the chain
    should continue, skip remaining steps, or fail entirely.

    Parameters
    ----------
    name : str
        A unique identifier for the step within the chain.
    validator_fn : Callable[[Any], tuple[bool, Optional[str]]]
        A function that validates the input data. Must return a tuple of
        (is_valid: bool, error_message: Optional[str]). If is_valid is
        False, error_message should describe the validation failure.
    on_failure : str, default="fail"
        Action to take when validation fails:

        - "fail": Raise an exception and stop the chain
        - "skip": Skip remaining steps and complete with current output
        - "retry": (Not implemented) Could be used for retry logic
    description : str, default=""
        Optional human-readable description of the step.

    Attributes
    ----------
    validator_fn : Callable[[Any], tuple[bool, Optional[str]]]
        The validation function.
    on_failure : str
        The failure handling strategy.

    Notes
    -----
    When validation fails, the following state variables are set:

    - ``_validation_error``: The error message from the validator
    - ``_validation_failed``: Set to True
    - ``_skip_remaining``: Set to True if on_failure="skip"

    Examples
    --------
    Validate response length:

    >>> def check_length(data):
    ...     if len(str(data)) < 10:
    ...         return False, "Response too short (minimum 10 characters)"
    ...     if len(str(data)) > 1000:
    ...         return False, "Response too long (maximum 1000 characters)"
    ...     return True, None
    >>>
    >>> length_validator = ValidatorStep(
    ...     name="check_length",
    ...     validator_fn=check_length,
    ...     on_failure="fail",
    ... )

    Validate JSON structure:

    >>> import json
    >>>
    >>> def validate_json(data):
    ...     try:
    ...         if isinstance(data, str):
    ...             parsed = json.loads(data)
    ...         else:
    ...             parsed = data
    ...         if "result" not in parsed:
    ...             return False, "Missing required 'result' field"
    ...         return True, None
    ...     except json.JSONDecodeError as e:
    ...         return False, f"Invalid JSON: {e}"
    >>>
    >>> json_validator = ValidatorStep("validate_json", validate_json)

    Validate with skip on failure (graceful degradation):

    >>> def check_quality(data):
    ...     # Some quality check
    ...     score = compute_quality_score(data)
    ...     if score < 0.5:
    ...         return False, f"Quality score {score} below threshold"
    ...     return True, None
    >>>
    >>> quality_gate = ValidatorStep(
    ...     name="quality_check",
    ...     validator_fn=check_quality,
    ...     on_failure="skip",  # Continue with what we have
    ... )

    Validate required fields in structured output:

    >>> def validate_entity_extraction(data):
    ...     required_fields = ["entities", "relationships", "confidence"]
    ...     missing = [f for f in required_fields if f not in data]
    ...     if missing:
    ...         return False, f"Missing required fields: {missing}"
    ...     if data["confidence"] < 0.7:
    ...         return False, f"Low confidence: {data['confidence']}"
    ...     return True, None
    >>>
    >>> entity_validator = ValidatorStep("validate_entities", validate_entity_extraction)

    Chain with multiple validators:

    >>> chain = (
    ...     create_chain("validated_pipeline", model_fn)
    ...     .llm("generate", "Generate a response for: {input}")
    ...     .validate("not_empty", lambda d: (bool(d.strip()), "Empty response"))
    ...     .validate("no_errors", lambda d: ("error" not in d.lower(), "Response contains error"))
    ...     .validate("length", check_length)
    ...     .build()
    ... )

    See Also
    --------
    ChainBuilder.validate : Convenient method for adding validators
    create_validator_step : Factory function for creating ValidatorStep
    ChainStep.validate_input : Pre-execution validation on step inputs
    """

    def __init__(
        self,
        name: str,
        validator_fn: Callable[[Any], tuple[bool, Optional[str]]],
        on_failure: str = "fail",  # "fail", "skip", "retry"
        description: str = "",
    ):
        """Initialize a validator step.

        Args
        ----
        name : str
            A unique identifier for the step.
        validator_fn : Callable[[Any], tuple[bool, Optional[str]]]
            Function that validates input, returning (is_valid, error_msg).
        on_failure : str, default="fail"
            Action on validation failure: "fail", "skip", or "retry".
        description : str, default=""
            Optional description of the step.
        """
        super().__init__(name, description)
        self.step_type = ChainStepType.VALIDATOR
        self.validator_fn = validator_fn
        self.on_failure = on_failure

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute the validation check.

        Runs the validator function on the input data and handles the
        result according to the on_failure strategy.

        Args
        ----
        input_data : Any
            The data to validate.
        state : ChainState
            The chain state for storing validation results.

        Returns
        -------
        Any
            The input data (unchanged) if validation passes or
            on_failure="skip".

        Raises
        ------
        ValueError
            If validation fails and on_failure="fail".

        Examples
        --------
        >>> def check_positive(data):
        ...     if data > 0:
        ...         return True, None
        ...     return False, f"Expected positive, got {data}"
        >>>
        >>> step = ValidatorStep("positive", check_positive)
        >>> state = ChainState()
        >>> step.execute(5, state)  # Passes
        5
        >>> step.execute(-1, state)  # Raises ValueError
        Traceback (most recent call last):
            ...
        ValueError: Validation failed: Expected positive, got -1
        """
        is_valid, error = self.validator_fn(input_data)

        if is_valid:
            return input_data

        state.set("_validation_error", error)
        state.set("_validation_failed", True)

        if self.on_failure == "fail":
            raise ValueError(f"Validation failed: {error}")
        elif self.on_failure == "skip":
            state.set("_skip_remaining", True)
            return input_data

        return input_data


class ConditionalStep(ChainStep):
    """Step that executes different branches based on a condition.

    ConditionalStep implements if/else branching logic within a chain.
    It evaluates a condition function and executes one of two possible
    steps based on the result, enabling dynamic workflow paths.

    Parameters
    ----------
    name : str
        A unique identifier for the step within the chain.
    condition_fn : Callable[[Any, ChainState], bool]
        A function that evaluates the condition. Receives the input data
        and chain state, returns True or False.
    if_true : ChainStep
        The step to execute when the condition is True.
    if_false : Optional[ChainStep], default=None
        The step to execute when the condition is False. If None, the
        input data passes through unchanged when condition is False.
    description : str, default=""
        Optional human-readable description of the step.

    Attributes
    ----------
    condition_fn : Callable[[Any, ChainState], bool]
        The condition evaluation function.
    if_true : ChainStep
        The step for the True branch.
    if_false : Optional[ChainStep]
        The step for the False branch (may be None).

    Examples
    --------
    Simple conditional based on content:

    >>> def is_question(data, state):
    ...     return "?" in str(data)
    >>>
    >>> answer_step = TransformStep("answer", lambda d, s: f"Answer: {d}")
    >>> acknowledge_step = TransformStep("ack", lambda d, s: f"Noted: {d}")
    >>>
    >>> conditional = ConditionalStep(
    ...     name="route_by_type",
    ...     condition_fn=is_question,
    ...     if_true=answer_step,
    ...     if_false=acknowledge_step,
    ... )

    Conditional based on state variables:

    >>> def is_premium_user(data, state):
    ...     return state.get("user_tier") == "premium"
    >>>
    >>> detailed_step = LLMStep("detailed", "Provide detailed analysis: {input}", model_fn)
    >>> basic_step = LLMStep("basic", "Provide brief summary: {input}", model_fn)
    >>>
    >>> tier_conditional = ConditionalStep(
    ...     name="tier_routing",
    ...     condition_fn=is_premium_user,
    ...     if_true=detailed_step,
    ...     if_false=basic_step,
    ... )

    Conditional with pass-through on False:

    >>> def needs_translation(data, state):
    ...     # Check if content is in English
    ...     return state.get("source_language") != "en"
    >>>
    >>> translate_step = LLMStep("translate", "Translate to English: {input}", model_fn)
    >>>
    >>> # If already English, pass through unchanged
    >>> translation_conditional = ConditionalStep(
    ...     name="maybe_translate",
    ...     condition_fn=needs_translation,
    ...     if_true=translate_step,
    ...     if_false=None,  # Pass through unchanged
    ... )

    Conditional based on previous results:

    >>> def confidence_check(data, state):
    ...     confidence = state.get("confidence_score", 0)
    ...     return confidence < 0.8
    >>>
    >>> refine_step = LLMStep("refine", "Improve this response: {input}", model_fn)
    >>>
    >>> confidence_conditional = ConditionalStep(
    ...     name="refine_if_low_confidence",
    ...     condition_fn=confidence_check,
    ...     if_true=refine_step,
    ... )

    Using in a chain with ChainBuilder:

    >>> chain = (
    ...     create_chain("conditional_pipeline", model_fn)
    ...     .llm("generate", "Generate response for: {input}")
    ...     .branch(
    ...         "quality_gate",
    ...         lambda d, s: len(d) > 100,  # Check if substantial
    ...         TransformStep("pass", lambda d, s: d),
    ...         LLMStep("expand", "Expand on this: {input}", model_fn),
    ...     )
    ...     .build()
    ... )

    Nested conditionals:

    >>> inner_conditional = ConditionalStep(
    ...     "inner", lambda d, s: "urgent" in d.lower(),
    ...     urgent_step, normal_step
    ... )
    >>> outer_conditional = ConditionalStep(
    ...     "outer", lambda d, s: state.get("priority") == "high",
    ...     inner_conditional, low_priority_step
    ... )

    See Also
    --------
    RouterStep : For multi-way branching based on content classification
    ChainBuilder.branch : Convenient method for adding conditionals
    """

    def __init__(
        self,
        name: str,
        condition_fn: Callable[[Any, ChainState], bool],
        if_true: ChainStep,
        if_false: Optional[ChainStep] = None,
        description: str = "",
    ):
        """Initialize a conditional step.

        Args
        ----
        name : str
            A unique identifier for the step.
        condition_fn : Callable[[Any, ChainState], bool]
            Function that evaluates the branching condition.
        if_true : ChainStep
            Step to execute when condition is True.
        if_false : Optional[ChainStep], default=None
            Step to execute when condition is False.
        description : str, default=""
            Optional description of the step.
        """
        super().__init__(name, description)
        self.step_type = ChainStepType.CONDITION
        self.condition_fn = condition_fn
        self.if_true = if_true
        self.if_false = if_false

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute the appropriate branch based on the condition.

        Evaluates the condition function and executes either if_true
        or if_false step accordingly.

        Args
        ----
        input_data : Any
            The data to pass to the branch step.
        state : ChainState
            The chain state for condition evaluation and step execution.

        Returns
        -------
        Any
            The output from the executed branch step, or the input_data
            unchanged if condition is False and no if_false step is set.

        Examples
        --------
        >>> def is_long(data, state):
        ...     return len(str(data)) > 50
        >>>
        >>> truncate = TransformStep("truncate", lambda d, s: d[:50])
        >>> step = ConditionalStep("check_length", is_long, truncate)
        >>> state = ChainState()
        >>> step.execute("short", state)  # Passes through
        'short'
        >>> step.execute("x" * 100, state)  # Gets truncated
        'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        """
        if self.condition_fn(input_data, state):
            return self.if_true.execute(input_data, state)
        elif self.if_false:
            return self.if_false.execute(input_data, state)
        return input_data


class RouterStep(ChainStep):
    """Step that routes to different steps based on content classification.

    RouterStep enables multi-way branching by classifying the input content
    and routing to the appropriate handler step. It supports several
    classification strategies including keyword matching, regex patterns,
    and custom classifier functions.

    Parameters
    ----------
    name : str
        A unique identifier for the step within the chain.
    routes : dict[str, ChainStep]
        A dictionary mapping route keys to handler steps. The keys are
        used for classification based on the selected strategy.
    strategy : RouterStrategy, default=RouterStrategy.KEYWORD
        The classification strategy to use:

        - KEYWORD: Simple substring matching (case-insensitive)
        - REGEX: Regular expression pattern matching
        - CLASSIFIER: Uses the classifier_fn for classification
        - CUSTOM: Uses the classifier_fn for classification
    default_route : Optional[str], default=None
        The route key to use when no match is found. If None and no
        match is found, uses the first route key.
    classifier_fn : Optional[Callable[[Any], str]], default=None
        A custom function that returns a route key for the input.
        Required when strategy is CLASSIFIER or CUSTOM.
    description : str, default=""
        Optional human-readable description of the step.

    Attributes
    ----------
    routes : dict[str, ChainStep]
        The route mapping.
    strategy : RouterStrategy
        The classification strategy.
    default_route : Optional[str]
        The default route key.
    classifier_fn : Optional[Callable[[Any], str]]
        The custom classifier function.

    Notes
    -----
    The selected route is stored in state as ``_selected_route`` for
    debugging and analysis purposes.

    Examples
    --------
    Keyword-based routing for intent handling:

    >>> routes = {
    ...     "help": TransformStep("help", lambda d, s: "Here's how I can help..."),
    ...     "price": TransformStep("price", lambda d, s: "Our pricing is..."),
    ...     "support": TransformStep("support", lambda d, s: "Contact support at..."),
    ... }
    >>>
    >>> router = RouterStep(
    ...     name="intent_router",
    ...     routes=routes,
    ...     strategy=RouterStrategy.KEYWORD,
    ...     default_route="help",
    ... )
    >>> # Input "What are your prices?" matches "price" route

    Regex-based routing for format detection:

    >>> routes = {
    ...     r"\\d{3}-\\d{2}-\\d{4}": TransformStep("ssn", lambda d, s: "SSN detected"),
    ...     r"\\d{3}-\\d{3}-\\d{4}": TransformStep("phone", lambda d, s: "Phone detected"),
    ...     r"[\\w.]+@[\\w.]+\\.[a-z]+": TransformStep("email", lambda d, s: "Email detected"),
    ... }
    >>>
    >>> format_router = RouterStep(
    ...     name="format_detector",
    ...     routes=routes,
    ...     strategy=RouterStrategy.REGEX,
    ... )

    Custom classifier-based routing:

    >>> def classify_sentiment(text):
    ...     text_lower = text.lower()
    ...     positive_words = ["great", "excellent", "happy", "good", "love"]
    ...     negative_words = ["bad", "terrible", "hate", "awful", "poor"]
    ...
    ...     pos_count = sum(1 for w in positive_words if w in text_lower)
    ...     neg_count = sum(1 for w in negative_words if w in text_lower)
    ...
    ...     if pos_count > neg_count:
    ...         return "positive"
    ...     elif neg_count > pos_count:
    ...         return "negative"
    ...     return "neutral"
    >>>
    >>> routes = {
    ...     "positive": LLMStep("pos", "Respond enthusiastically: {input}", model_fn),
    ...     "negative": LLMStep("neg", "Respond empathetically: {input}", model_fn),
    ...     "neutral": LLMStep("neu", "Respond neutrally: {input}", model_fn),
    ... }
    >>>
    >>> sentiment_router = RouterStep(
    ...     name="sentiment_router",
    ...     routes=routes,
    ...     strategy=RouterStrategy.CLASSIFIER,
    ...     classifier_fn=classify_sentiment,
    ... )

    LLM-based classification:

    >>> def llm_classifier(text):
    ...     prompt = f"Classify this text as 'technical', 'casual', or 'formal': {text}"
    ...     response = model_fn(prompt).strip().lower()
    ...     if response in ["technical", "casual", "formal"]:
    ...         return response
    ...     return "casual"  # Default
    >>>
    >>> style_routes = {
    ...     "technical": LLMStep("tech", "Respond technically: {input}", model_fn),
    ...     "casual": LLMStep("casual", "Respond casually: {input}", model_fn),
    ...     "formal": LLMStep("formal", "Respond formally: {input}", model_fn),
    ... }
    >>>
    >>> style_router = RouterStep(
    ...     "style_router", style_routes, RouterStrategy.CLASSIFIER,
    ...     classifier_fn=llm_classifier
    ... )

    Using in a chain:

    >>> chain = (
    ...     create_chain("routed_pipeline")
    ...     .route("intent", routes, RouterStrategy.KEYWORD, default_route="help")
    ...     .transform("format", lambda d, s: f"Response: {d}")
    ...     .build()
    ... )

    See Also
    --------
    ConditionalStep : For simple binary branching
    RouterStrategy : Available classification strategies
    ChainBuilder.route : Convenient method for adding routers
    """

    def __init__(
        self,
        name: str,
        routes: dict[str, ChainStep],
        strategy: RouterStrategy = RouterStrategy.KEYWORD,
        default_route: Optional[str] = None,
        classifier_fn: Optional[Callable[[Any], str]] = None,
        description: str = "",
    ):
        """Initialize a router step.

        Args
        ----
        name : str
            A unique identifier for the step.
        routes : dict[str, ChainStep]
            Mapping of route keys to handler steps.
        strategy : RouterStrategy, default=KEYWORD
            The classification strategy to use.
        default_route : Optional[str], default=None
            Route key to use when no match is found.
        classifier_fn : Optional[Callable[[Any], str]], default=None
            Custom classifier function for CLASSIFIER/CUSTOM strategies.
        description : str, default=""
            Optional description of the step.
        """
        super().__init__(name, description)
        self.step_type = ChainStepType.ROUTER
        self.routes = routes
        self.strategy = strategy
        self.default_route = default_route
        self.classifier_fn = classifier_fn

    def _classify_input(self, input_data: Any) -> str:
        """Classify input to determine which route to take.

        Applies the configured strategy to classify the input and
        determine the appropriate route key.

        Args
        ----
        input_data : Any
            The data to classify.

        Returns
        -------
        str
            The route key for the matching route.

        Examples
        --------
        >>> routes = {"error": step1, "warning": step2}
        >>> router = RouterStep("r", routes, RouterStrategy.KEYWORD)
        >>> router._classify_input("An error occurred")
        'error'
        """
        input_str = str(input_data).lower()

        if self.strategy == RouterStrategy.KEYWORD:
            for route_key in self.routes:
                if route_key.lower() in input_str:
                    return route_key

        elif self.strategy == RouterStrategy.REGEX:
            for route_key in self.routes:
                if re.search(route_key, input_str, re.IGNORECASE):
                    return route_key

        elif (
            self.strategy == RouterStrategy.CLASSIFIER
            and self.classifier_fn
            or self.strategy == RouterStrategy.CUSTOM
            and self.classifier_fn
        ):
            return self.classifier_fn(input_data)

        return self.default_route or list(self.routes.keys())[0]

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Classify input and execute the appropriate route step.

        Classifies the input, stores the selected route in state,
        and executes the corresponding handler step.

        Args
        ----
        input_data : Any
            The data to route and process.
        state : ChainState
            The chain state for storing the selected route.

        Returns
        -------
        Any
            The output from the executed route step, or input_data
            unchanged if no matching route is found.

        Examples
        --------
        >>> routes = {
        ...     "upper": TransformStep("up", lambda d, s: d.upper()),
        ...     "lower": TransformStep("lo", lambda d, s: d.lower()),
        ... }
        >>> router = RouterStep("r", routes, RouterStrategy.KEYWORD)
        >>> state = ChainState()
        >>> router.execute("convert to upper case", state)
        'CONVERT TO UPPER CASE'
        >>> state.get("_selected_route")
        'upper'
        """
        route = self._classify_input(input_data)
        state.set("_selected_route", route)

        if route in self.routes:
            return self.routes[route].execute(input_data, state)

        if self.default_route and self.default_route in self.routes:
            return self.routes[self.default_route].execute(input_data, state)

        return input_data


class LoopStep(ChainStep):
    """Step that loops until an exit condition is met.

    LoopStep enables iterative processing by repeatedly executing a body
    step until an exit condition returns True or the maximum iteration
    count is reached. This is useful for refinement loops, retry logic,
    and iterative algorithms.

    Parameters
    ----------
    name : str
        A unique identifier for the step within the chain.
    body_step : ChainStep
        The step to execute on each iteration. The output from each
        iteration becomes the input for the next.
    exit_condition : Callable[[Any, ChainState, int], bool]
        A function that determines when to exit the loop. Receives the
        current data, chain state, and iteration count (0-indexed).
        Returns True to exit, False to continue.
    max_iterations : int, default=10
        Maximum number of iterations to prevent infinite loops. The loop
        will exit when this count is reached even if exit_condition
        returns False.
    description : str, default=""
        Optional human-readable description of the step.

    Attributes
    ----------
    body_step : ChainStep
        The step executed on each iteration.
    exit_condition : Callable[[Any, ChainState, int], bool]
        The exit condition function.
    max_iterations : int
        The maximum iteration limit.

    Notes
    -----
    The loop sets the following state variables:

    - ``_loop_iteration``: Current iteration count (updated after each iteration)
    - ``_loop_total_iterations``: Total iterations completed when loop exits

    The exit condition is checked BEFORE each iteration, so if it returns
    True initially, the body will not execute at all.

    Examples
    --------
    Iterative refinement until quality threshold:

    >>> def quality_check(data, state, iteration):
    ...     # Exit when quality is high enough or after 5 iterations
    ...     quality = compute_quality_score(data)
    ...     state.set("last_quality", quality)
    ...     return quality > 0.9 or iteration >= 5
    >>>
    >>> refine_step = LLMStep(
    ...     "refine",
    ...     "Improve this response, making it clearer and more accurate:\\n{input}",
    ...     model_fn,
    ... )
    >>>
    >>> refinement_loop = LoopStep(
    ...     name="quality_refinement",
    ...     body_step=refine_step,
    ...     exit_condition=quality_check,
    ...     max_iterations=10,
    ... )

    Simple iteration count-based exit:

    >>> def exit_after_3(data, state, iteration):
    ...     return iteration >= 3
    >>>
    >>> process_step = TransformStep("process", lambda d, s: f"[{d}]")
    >>> loop = LoopStep("bracket_loop", process_step, exit_after_3)
    >>>
    >>> # "hello" -> "[hello]" -> "[[hello]]" -> "[[[hello]]]"
    >>> state = ChainState()
    >>> result = loop.execute("hello", state)

    Content-based exit condition:

    >>> def contains_done(data, state, iteration):
    ...     return "DONE" in str(data) or iteration > 10
    >>>
    >>> generate_step = LLMStep(
    ...     "generate",
    ...     "Continue writing. Say DONE when finished:\\n{input}",
    ...     model_fn,
    ... )
    >>>
    >>> writing_loop = LoopStep(
    ...     "write_until_done",
    ...     generate_step,
    ...     contains_done,
    ...     max_iterations=20,
    ... )

    Retry with exponential backoff (using state):

    >>> def retry_condition(data, state, iteration):
    ...     # Exit if we got a valid result or exceeded retries
    ...     if is_valid_response(data):
    ...         return True
    ...     if iteration >= 3:
    ...         return True
    ...     # Wait before retry (exponential backoff)
    ...     time.sleep(2 ** iteration)
    ...     return False
    >>>
    >>> api_call_step = TransformStep("call_api", make_api_call)
    >>> retry_loop = LoopStep("api_retry", api_call_step, retry_condition, 5)

    Accumulating results across iterations:

    >>> def accumulate_step(data, state):
    ...     collected = state.get("collected", [])
    ...     collected.append(process(data))
    ...     state.set("collected", collected)
    ...     return get_next_item(data)
    >>>
    >>> def all_collected(data, state, iteration):
    ...     return data is None or iteration >= 100
    >>>
    >>> collector = TransformStep("collect", accumulate_step)
    >>> collection_loop = LoopStep("collector", collector, all_collected)

    Using in a chain with ChainBuilder:

    >>> chain = (
    ...     create_chain("iterative_pipeline", model_fn)
    ...     .llm("initial", "Generate initial response: {input}")
    ...     .loop(
    ...         "refine_loop",
    ...         LLMStep("refine", "Improve: {input}", model_fn),
    ...         lambda d, s, i: i >= 3,  # 3 refinement iterations
    ...         max_iterations=5,
    ...     )
    ...     .build()
    ... )

    See Also
    --------
    ChainBuilder.loop : Convenient method for adding loops
    WorkflowTemplate.iterative_refinement : Pre-built refinement template
    """

    def __init__(
        self,
        name: str,
        body_step: ChainStep,
        exit_condition: Callable[[Any, ChainState, int], bool],
        max_iterations: int = 10,
        description: str = "",
    ):
        """Initialize a loop step.

        Args
        ----
        name : str
            A unique identifier for the step.
        body_step : ChainStep
            The step to execute on each iteration.
        exit_condition : Callable[[Any, ChainState, int], bool]
            Function that returns True to exit the loop.
        max_iterations : int, default=10
            Maximum iterations before forced exit.
        description : str, default=""
            Optional description of the step.
        """
        super().__init__(name, description)
        self.step_type = ChainStepType.LOOP
        self.body_step = body_step
        self.exit_condition = exit_condition
        self.max_iterations = max_iterations

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute the loop until exit condition is met.

        Repeatedly executes the body step, passing the output of each
        iteration as input to the next, until the exit condition
        returns True or max_iterations is reached.

        Args
        ----
        input_data : Any
            The initial input data for the first iteration.
        state : ChainState
            The chain state for storing loop progress variables.

        Returns
        -------
        Any
            The output from the final iteration.

        Examples
        --------
        >>> step = TransformStep("add_x", lambda d, s: d + "x")
        >>> loop = LoopStep("add_xs", step, lambda d, s, i: i >= 3)
        >>> state = ChainState()
        >>> loop.execute("start", state)
        'startxxx'
        >>> state.get("_loop_total_iterations")
        3
        """
        current_data = input_data
        iteration = 0

        while iteration < self.max_iterations:
            # Check exit condition
            if self.exit_condition(current_data, state, iteration):
                break

            # Execute body
            current_data = self.body_step.execute(current_data, state)
            iteration += 1

            state.set("_loop_iteration", iteration)

        state.set("_loop_total_iterations", iteration)
        return current_data


class ParallelStep(ChainStep):
    """Step that executes multiple steps in parallel (simulated)."""

    def __init__(
        self,
        name: str,
        steps: list[ChainStep],
        aggregator: Optional[Callable[[list[Any]], Any]] = None,
        description: str = "",
    ):
        super().__init__(name, description)
        self.step_type = ChainStepType.PARALLEL
        self.steps = steps
        self.aggregator = aggregator or (lambda results: results)

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute steps (sequentially, simulating parallel)."""
        results = []
        for step in self.steps:
            result = step.execute(input_data, state)
            results.append(result)

        return self.aggregator(results)


class SubchainStep(ChainStep):
    """Step that executes another chain as a subchain."""

    def __init__(
        self,
        name: str,
        chain: "Chain",
        description: str = "",
    ):
        super().__init__(name, description)
        self.step_type = ChainStepType.SUBCHAIN
        self.chain = chain

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute subchain."""
        result = self.chain.run(input_data)
        return result.final_output


class Chain:
    """A chain of steps to execute sequentially."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: list[ChainStep] = []
        self.error_handler: Optional[Callable[[Exception, ChainState], Any]] = None
        self.pre_hooks: list[Callable[[Any, ChainState], Any]] = []
        self.post_hooks: list[Callable[[Any, ChainState], Any]] = []

    def add_step(self, step: ChainStep) -> "Chain":
        """Add a step to the chain."""
        self.steps.append(step)
        return self

    def add_llm_step(
        self,
        name: str,
        prompt_template: str,
        model_fn: Callable[[str], str],
        output_parser: Optional[Callable[[str], Any]] = None,
    ) -> "Chain":
        """Add an LLM call step."""
        step = LLMStep(name, prompt_template, model_fn, output_parser)
        return self.add_step(step)

    def add_transform(
        self,
        name: str,
        transform_fn: Callable[[Any, ChainState], Any],
    ) -> "Chain":
        """Add a transformation step."""
        step = TransformStep(name, transform_fn)
        return self.add_step(step)

    def add_validator(
        self,
        name: str,
        validator_fn: Callable[[Any], tuple[bool, Optional[str]]],
        on_failure: str = "fail",
    ) -> "Chain":
        """Add a validation step."""
        step = ValidatorStep(name, validator_fn, on_failure)
        return self.add_step(step)

    def add_conditional(
        self,
        name: str,
        condition_fn: Callable[[Any, ChainState], bool],
        if_true: ChainStep,
        if_false: Optional[ChainStep] = None,
    ) -> "Chain":
        """Add a conditional step."""
        step = ConditionalStep(name, condition_fn, if_true, if_false)
        return self.add_step(step)

    def add_router(
        self,
        name: str,
        routes: dict[str, ChainStep],
        strategy: RouterStrategy = RouterStrategy.KEYWORD,
        default_route: Optional[str] = None,
    ) -> "Chain":
        """Add a router step."""
        step = RouterStep(name, routes, strategy, default_route)
        return self.add_step(step)

    def add_loop(
        self,
        name: str,
        body_step: ChainStep,
        exit_condition: Callable[[Any, ChainState, int], bool],
        max_iterations: int = 10,
    ) -> "Chain":
        """Add a loop step."""
        step = LoopStep(name, body_step, exit_condition, max_iterations)
        return self.add_step(step)

    def add_parallel(
        self,
        name: str,
        steps: list[ChainStep],
        aggregator: Optional[Callable[[list[Any]], Any]] = None,
    ) -> "Chain":
        """Add parallel execution step."""
        step = ParallelStep(name, steps, aggregator)
        return self.add_step(step)

    def add_subchain(self, name: str, chain: "Chain") -> "Chain":
        """Add a subchain step."""
        step = SubchainStep(name, chain)
        return self.add_step(step)

    def set_error_handler(
        self,
        handler: Callable[[Exception, ChainState], Any],
    ) -> "Chain":
        """Set error handler for the chain."""
        self.error_handler = handler
        return self

    def add_pre_hook(self, hook: Callable[[Any, ChainState], Any]) -> "Chain":
        """Add a hook to run before each step."""
        self.pre_hooks.append(hook)
        return self

    def add_post_hook(self, hook: Callable[[Any, ChainState], Any]) -> "Chain":
        """Add a hook to run after each step."""
        self.post_hooks.append(hook)
        return self

    def run(self, input_data: Any, initial_state: Optional[ChainState] = None) -> ChainResult:
        """Run the chain."""
        state = initial_state or ChainState()
        state.status = ChainStatus.RUNNING
        state.start_time = time.time()
        state.set("input", input_data)

        current_data = input_data
        errors = []
        successful = 0
        failed = 0

        for i, step in enumerate(self.steps):
            state.current_step = i

            # Check for skip flag
            if state.get("_skip_remaining"):
                break

            # Run pre-hooks
            for hook in self.pre_hooks:
                current_data = hook(current_data, state)

            start_time = time.time()
            step_status = ChainStatus.RUNNING

            try:
                # Validate input
                is_valid, error = step.validate_input(current_data)
                if not is_valid:
                    raise ValueError(f"Input validation failed: {error}")

                # Execute step
                output_data = step.execute(current_data, state)
                step_status = ChainStatus.COMPLETED
                successful += 1
                current_data = output_data

            except Exception as e:
                step_status = ChainStatus.FAILED
                failed += 1
                error_msg = f"{step.name}: {str(e)}"
                errors.append(error_msg)

                if self.error_handler:
                    try:
                        current_data = self.error_handler(e, state)
                        step_status = ChainStatus.COMPLETED
                    except Exception:
                        state.status = ChainStatus.FAILED
                        break
                else:
                    state.status = ChainStatus.FAILED
                    output_data = None
                    # Record result and break
                    end_time = time.time()
                    result = StepResult(
                        step_name=step.name,
                        step_type=step.step_type,
                        status=step_status,
                        input_data=current_data,
                        output_data=None,
                        start_time=start_time,
                        end_time=end_time,
                        error=str(e),
                    )
                    state.step_results.append(result)
                    break

            end_time = time.time()

            # Record result
            result = StepResult(
                step_name=step.name,
                step_type=step.step_type,
                status=step_status,
                input_data=current_data if step_status == ChainStatus.FAILED else input_data,
                output_data=current_data if step_status == ChainStatus.COMPLETED else None,
                start_time=start_time,
                end_time=end_time,
            )
            state.step_results.append(result)

            # Run post-hooks
            for hook in self.post_hooks:
                current_data = hook(current_data, state)

        state.end_time = time.time()
        if state.status == ChainStatus.RUNNING:
            state.status = ChainStatus.COMPLETED

        return ChainResult(
            chain_name=self.name,
            status=state.status,
            final_output=current_data,
            state=state,
            total_steps=len(self.steps),
            successful_steps=successful,
            failed_steps=failed,
            total_duration=state.end_time - state.start_time,
            errors=errors,
        )


class ChainBuilder:
    """Fluent builder for creating chains."""

    def __init__(self, name: str, model_fn: Optional[Callable[[str], str]] = None):
        self.chain = Chain(name)
        self.model_fn = model_fn

    def with_description(self, description: str) -> "ChainBuilder":
        """Set chain description."""
        self.chain.description = description
        return self

    def with_model(self, model_fn: Callable[[str], str]) -> "ChainBuilder":
        """Set default model function."""
        self.model_fn = model_fn
        return self

    def llm(
        self,
        name: str,
        prompt_template: str,
        output_parser: Optional[Callable[[str], Any]] = None,
        model_fn: Optional[Callable[[str], str]] = None,
    ) -> "ChainBuilder":
        """Add LLM step."""
        model_callable = model_fn or self.model_fn
        if not model_callable:
            raise ValueError("No model function provided")
        self.chain.add_llm_step(name, prompt_template, model_callable, output_parser)
        return self

    def transform(
        self,
        name: str,
        transform_fn: Callable[[Any, ChainState], Any],
    ) -> "ChainBuilder":
        """Add transform step."""
        self.chain.add_transform(name, transform_fn)
        return self

    def validate(
        self,
        name: str,
        validator_fn: Callable[[Any], tuple[bool, Optional[str]]],
        on_failure: str = "fail",
    ) -> "ChainBuilder":
        """Add validation step."""
        self.chain.add_validator(name, validator_fn, on_failure)
        return self

    def branch(
        self,
        name: str,
        condition_fn: Callable[[Any, ChainState], bool],
        if_true: ChainStep,
        if_false: Optional[ChainStep] = None,
    ) -> "ChainBuilder":
        """Add conditional branch."""
        self.chain.add_conditional(name, condition_fn, if_true, if_false)
        return self

    def route(
        self,
        name: str,
        routes: dict[str, ChainStep],
        strategy: RouterStrategy = RouterStrategy.KEYWORD,
        default_route: Optional[str] = None,
    ) -> "ChainBuilder":
        """Add router step."""
        self.chain.add_router(name, routes, strategy, default_route)
        return self

    def loop(
        self,
        name: str,
        body_step: ChainStep,
        exit_condition: Callable[[Any, ChainState, int], bool],
        max_iterations: int = 10,
    ) -> "ChainBuilder":
        """Add loop step."""
        self.chain.add_loop(name, body_step, exit_condition, max_iterations)
        return self

    def parallel(
        self,
        name: str,
        steps: list[ChainStep],
        aggregator: Optional[Callable[[list[Any]], Any]] = None,
    ) -> "ChainBuilder":
        """Add parallel step."""
        self.chain.add_parallel(name, steps, aggregator)
        return self

    def subchain(self, name: str, chain: Chain) -> "ChainBuilder":
        """Add subchain."""
        self.chain.add_subchain(name, chain)
        return self

    def on_error(self, handler: Callable[[Exception, ChainState], Any]) -> "ChainBuilder":
        """Set error handler."""
        self.chain.set_error_handler(handler)
        return self

    def before_each(self, hook: Callable[[Any, ChainState], Any]) -> "ChainBuilder":
        """Add pre-step hook."""
        self.chain.add_pre_hook(hook)
        return self

    def after_each(self, hook: Callable[[Any, ChainState], Any]) -> "ChainBuilder":
        """Add post-step hook."""
        self.chain.add_post_hook(hook)
        return self

    def build(self) -> Chain:
        """Build and return the chain."""
        return self.chain


class ChainRegistry:
    """Registry for reusable chains."""

    def __init__(self):
        self.chains: dict[str, Chain] = {}

    def register(self, chain: Chain) -> None:
        """Register a chain."""
        self.chains[chain.name] = chain

    def get(self, name: str) -> Optional[Chain]:
        """Get a chain by name."""
        return self.chains.get(name)

    def list_chains(self) -> list[str]:
        """List all registered chain names."""
        return list(self.chains.keys())

    def run(self, name: str, input_data: Any) -> Optional[ChainResult]:
        """Run a registered chain."""
        chain = self.get(name)
        if chain:
            return chain.run(input_data)
        return None


class WorkflowTemplate:
    """Template for common workflow patterns."""

    @staticmethod
    def summarize_and_answer(
        model_fn: Callable[[str], str],
        name: str = "summarize_and_answer",
    ) -> Chain:
        """Create a summarize-then-answer chain."""
        return (
            ChainBuilder(name, model_fn)
            .llm(
                "summarize",
                "Summarize the following text:\n{input}",
            )
            .llm(
                "answer",
                "Based on this summary: {input}\n\nAnswer any questions concisely.",
            )
            .build()
        )

    @staticmethod
    def extract_and_validate(
        model_fn: Callable[[str], str],
        validator_fn: Callable[[Any], tuple[bool, Optional[str]]],
        name: str = "extract_and_validate",
    ) -> Chain:
        """Create an extract-then-validate chain."""
        return (
            ChainBuilder(name, model_fn)
            .llm(
                "extract",
                "Extract the key information from:\n{input}",
            )
            .validate("validate", validator_fn)
            .build()
        )

    @staticmethod
    def iterative_refinement(
        model_fn: Callable[[str], str],
        max_iterations: int = 3,
        quality_threshold: float = 0.8,
        name: str = "iterative_refinement",
    ) -> Chain:
        """Create an iterative refinement chain."""
        refine_step = LLMStep(
            "refine",
            "Improve the following text, making it clearer and more accurate:\n{input}",
            model_fn,
        )

        def exit_condition(output: Any, state: ChainState, iteration: int) -> bool:
            # Simple exit: just check iteration count
            return iteration >= max_iterations

        return (
            ChainBuilder(name, model_fn)
            .llm("initial", "Generate an initial response for:\n{input}")
            .loop("refine_loop", refine_step, exit_condition, max_iterations)
            .build()
        )

    @staticmethod
    def classify_and_route(
        model_fn: Callable[[str], str],
        routes: dict[str, str],  # route_name -> prompt_template
        name: str = "classify_and_route",
    ) -> Chain:
        """Create a classification-based routing chain."""
        route_steps = {
            route: LLMStep(route, template, model_fn) for route, template in routes.items()
        }

        return (
            ChainBuilder(name, model_fn)
            .route("router", route_steps, RouterStrategy.KEYWORD)
            .build()
        )


# Convenience functions


def create_chain(name: str, model_fn: Optional[Callable[[str], str]] = None) -> ChainBuilder:
    """Create a new chain builder."""
    return ChainBuilder(name, model_fn)


def create_llm_step(
    name: str,
    prompt_template: str,
    model_fn: Callable[[str], str],
    output_parser: Optional[Callable[[str], Any]] = None,
) -> LLMStep:
    """Create an LLM step."""
    return LLMStep(name, prompt_template, model_fn, output_parser)


def create_transform_step(
    name: str,
    transform_fn: Callable[[Any, ChainState], Any],
) -> TransformStep:
    """Create a transform step."""
    return TransformStep(name, transform_fn)


def create_validator_step(
    name: str,
    validator_fn: Callable[[Any], tuple[bool, Optional[str]]],
    on_failure: str = "fail",
) -> ValidatorStep:
    """Create a validator step."""
    return ValidatorStep(name, validator_fn, on_failure)


def run_chain(
    chain: Chain,
    input_data: Any,
    initial_state: Optional[ChainState] = None,
) -> ChainResult:
    """Run a chain and return the result."""
    return chain.run(input_data, initial_state)


def simple_chain(
    steps: list[Callable[[Any], Any]],
    name: str = "simple_chain",
) -> Chain:
    """Create a simple chain from a list of functions."""
    chain = Chain(name)
    for i, fn in enumerate(steps):
        step = TransformStep(
            f"step_{i}",
            lambda data, state, f=fn: f(data),
        )
        chain.add_step(step)
    return chain


def sequential_llm_chain(
    prompts: list[str],
    model_fn: Callable[[str], str],
    name: str = "sequential_llm",
) -> Chain:
    """Create a chain of sequential LLM calls."""
    builder = ChainBuilder(name, model_fn)
    for i, prompt in enumerate(prompts):
        builder.llm(f"llm_{i}", prompt)
    return builder.build()


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests use StepType. The canonical name is ChainStepType.
StepType = ChainStepType
