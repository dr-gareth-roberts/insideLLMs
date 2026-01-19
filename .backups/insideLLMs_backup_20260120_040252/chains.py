"""Prompt chain and workflow orchestration for LLM pipelines.

This module provides tools for building complex multi-step LLM workflows:
- Sequential and parallel chain execution
- Conditional branching based on outputs
- Loop constructs with exit conditions
- State management between steps
- Error handling and retry logic
- Workflow composition and reuse
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
    """Status of a chain execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class StepType(Enum):
    """Types of chain steps."""

    LLM_CALL = "llm_call"
    TRANSFORM = "transform"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    SUBCHAIN = "subchain"
    VALIDATOR = "validator"
    ROUTER = "router"


class RouterStrategy(Enum):
    """Strategies for routing in conditional chains."""

    KEYWORD = "keyword"
    REGEX = "regex"
    CLASSIFIER = "classifier"
    CUSTOM = "custom"


@dataclass
class StepResult:
    """Result from a single chain step."""

    step_name: str
    step_type: StepType
    status: ChainStatus
    input_data: Any
    output_data: Any
    start_time: float
    end_time: float
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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
        """Duration in seconds."""
        return self.end_time - self.start_time

    @property
    def success(self) -> bool:
        """Whether the step succeeded."""
        return self.status == ChainStatus.COMPLETED


@dataclass
class ChainState:
    """State maintained across chain execution."""

    variables: dict[str, Any] = field(default_factory=dict)
    step_results: list[StepResult] = field(default_factory=list)
    current_step: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: ChainStatus = ChainStatus.PENDING

    def set(self, key: str, value: Any) -> None:
        """Set a variable."""
        self.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable."""
        return self.variables.get(key, default)

    def get_last_output(self) -> Any:
        """Get the output from the last completed step."""
        for result in reversed(self.step_results):
            if result.success:
                return result.output_data
        return None

    def to_dict(self) -> dict[str, Any]:
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
    """Result from a complete chain execution."""

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
        """Whether the chain completed successfully."""
        return self.status == ChainStatus.COMPLETED


class ChainStep(ABC):
    """Abstract base class for chain steps."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.step_type = StepType.TRANSFORM

    @abstractmethod
    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute the step."""
        pass

    def validate_input(self, input_data: Any) -> tuple[bool, Optional[str]]:
        """Validate input data. Override for custom validation."""
        return True, None


class LLMStep(ChainStep):
    """Step that calls an LLM."""

    def __init__(
        self,
        name: str,
        prompt_template: str,
        model_fn: Callable[[str], str],
        output_parser: Optional[Callable[[str], Any]] = None,
        description: str = "",
    ):
        super().__init__(name, description)
        self.step_type = StepType.LLM_CALL
        self.prompt_template = prompt_template
        self.model_fn = model_fn
        self.output_parser = output_parser

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute LLM call."""
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
    """Step that transforms data."""

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[Any, ChainState], Any],
        description: str = "",
    ):
        super().__init__(name, description)
        self.step_type = StepType.TRANSFORM
        self.transform_fn = transform_fn

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute transformation."""
        return self.transform_fn(input_data, state)


class ValidatorStep(ChainStep):
    """Step that validates output and can halt chain."""

    def __init__(
        self,
        name: str,
        validator_fn: Callable[[Any], tuple[bool, Optional[str]]],
        on_failure: str = "fail",  # "fail", "skip", "retry"
        description: str = "",
    ):
        super().__init__(name, description)
        self.step_type = StepType.VALIDATOR
        self.validator_fn = validator_fn
        self.on_failure = on_failure

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute validation."""
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
    """Step that executes different branches based on condition."""

    def __init__(
        self,
        name: str,
        condition_fn: Callable[[Any, ChainState], bool],
        if_true: ChainStep,
        if_false: Optional[ChainStep] = None,
        description: str = "",
    ):
        super().__init__(name, description)
        self.step_type = StepType.CONDITION
        self.condition_fn = condition_fn
        self.if_true = if_true
        self.if_false = if_false

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute conditional branch."""
        if self.condition_fn(input_data, state):
            return self.if_true.execute(input_data, state)
        elif self.if_false:
            return self.if_false.execute(input_data, state)
        return input_data


class RouterStep(ChainStep):
    """Step that routes to different steps based on content."""

    def __init__(
        self,
        name: str,
        routes: dict[str, ChainStep],
        strategy: RouterStrategy = RouterStrategy.KEYWORD,
        default_route: Optional[str] = None,
        classifier_fn: Optional[Callable[[Any], str]] = None,
        description: str = "",
    ):
        super().__init__(name, description)
        self.step_type = StepType.ROUTER
        self.routes = routes
        self.strategy = strategy
        self.default_route = default_route
        self.classifier_fn = classifier_fn

    def _classify_input(self, input_data: Any) -> str:
        """Classify input to determine route."""
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
        """Route and execute appropriate step."""
        route = self._classify_input(input_data)
        state.set("_selected_route", route)

        if route in self.routes:
            return self.routes[route].execute(input_data, state)

        if self.default_route and self.default_route in self.routes:
            return self.routes[self.default_route].execute(input_data, state)

        return input_data


class LoopStep(ChainStep):
    """Step that loops until condition is met."""

    def __init__(
        self,
        name: str,
        body_step: ChainStep,
        exit_condition: Callable[[Any, ChainState, int], bool],
        max_iterations: int = 10,
        description: str = "",
    ):
        super().__init__(name, description)
        self.step_type = StepType.LOOP
        self.body_step = body_step
        self.exit_condition = exit_condition
        self.max_iterations = max_iterations

    def execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute loop."""
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
        self.step_type = StepType.PARALLEL
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
        self.step_type = StepType.SUBCHAIN
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
        fn = model_fn or self.model_fn
        if not fn:
            raise ValueError("No model function provided")
        self.chain.add_llm_step(name, prompt_template, fn, output_parser)
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
