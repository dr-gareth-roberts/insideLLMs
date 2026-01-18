"""Tests for the chains module."""

import pytest

from insideLLMs.chains import (
    # Chain classes
    Chain,
    ChainBuilder,
    ChainRegistry,
    ChainResult,
    ChainState,
    # Enums
    ChainStatus,
    # Step classes
    ConditionalStep,
    LLMStep,
    LoopStep,
    ParallelStep,
    RouterStep,
    RouterStrategy,
    # Dataclasses
    StepResult,
    StepType,
    SubchainStep,
    TransformStep,
    ValidatorStep,
    WorkflowTemplate,
    # Convenience functions
    create_chain,
    create_llm_step,
    create_transform_step,
    create_validator_step,
    run_chain,
    sequential_llm_chain,
    simple_chain,
)

# ============================================================================
# Enum Tests
# ============================================================================


class TestChainEnums:
    """Test chain-related enums."""

    def test_chain_status_values(self):
        assert ChainStatus.PENDING.value == "pending"
        assert ChainStatus.RUNNING.value == "running"
        assert ChainStatus.COMPLETED.value == "completed"
        assert ChainStatus.FAILED.value == "failed"
        assert ChainStatus.SKIPPED.value == "skipped"

    def test_step_type_values(self):
        assert StepType.LLM_CALL.value == "llm_call"
        assert StepType.TRANSFORM.value == "transform"
        assert StepType.CONDITION.value == "condition"
        assert StepType.PARALLEL.value == "parallel"
        assert StepType.LOOP.value == "loop"

    def test_router_strategy_values(self):
        assert RouterStrategy.KEYWORD.value == "keyword"
        assert RouterStrategy.REGEX.value == "regex"
        assert RouterStrategy.CLASSIFIER.value == "classifier"
        assert RouterStrategy.CUSTOM.value == "custom"


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestStepResult:
    """Test StepResult dataclass."""

    def test_creation(self):
        result = StepResult(
            step_name="test_step",
            step_type=StepType.TRANSFORM,
            status=ChainStatus.COMPLETED,
            input_data="input",
            output_data="output",
            start_time=1000.0,
            end_time=1001.0,
        )
        assert result.step_name == "test_step"
        assert result.success is True

    def test_duration(self):
        result = StepResult(
            step_name="test",
            step_type=StepType.LLM_CALL,
            status=ChainStatus.COMPLETED,
            input_data="",
            output_data="",
            start_time=1000.0,
            end_time=1005.0,
        )
        assert result.duration == 5.0

    def test_failed_result(self):
        result = StepResult(
            step_name="failed",
            step_type=StepType.VALIDATOR,
            status=ChainStatus.FAILED,
            input_data="",
            output_data=None,
            start_time=0.0,
            end_time=0.0,
            error="Validation failed",
        )
        assert result.success is False
        assert result.error is not None

    def test_to_dict(self):
        result = StepResult(
            step_name="test",
            step_type=StepType.TRANSFORM,
            status=ChainStatus.COMPLETED,
            input_data="in",
            output_data="out",
            start_time=0.0,
            end_time=1.0,
        )
        d = result.to_dict()
        assert d["step_name"] == "test"
        assert d["status"] == "completed"


class TestChainState:
    """Test ChainState dataclass."""

    def test_creation(self):
        state = ChainState()
        assert state.status == ChainStatus.PENDING
        assert state.current_step == 0

    def test_set_and_get(self):
        state = ChainState()
        state.set("key", "value")
        assert state.get("key") == "value"
        assert state.get("nonexistent", "default") == "default"

    def test_get_last_output(self):
        state = ChainState()
        state.step_results = [
            StepResult("s1", StepType.TRANSFORM, ChainStatus.COMPLETED, "", "output1", 0, 1),
            StepResult("s2", StepType.TRANSFORM, ChainStatus.COMPLETED, "", "output2", 1, 2),
        ]
        assert state.get_last_output() == "output2"

    def test_get_last_output_with_failed(self):
        state = ChainState()
        state.step_results = [
            StepResult("s1", StepType.TRANSFORM, ChainStatus.COMPLETED, "", "output1", 0, 1),
            StepResult("s2", StepType.TRANSFORM, ChainStatus.FAILED, "", None, 1, 2),
        ]
        assert state.get_last_output() == "output1"


class TestChainResult:
    """Test ChainResult dataclass."""

    def test_creation(self):
        state = ChainState()
        result = ChainResult(
            chain_name="test_chain",
            status=ChainStatus.COMPLETED,
            final_output="result",
            state=state,
            total_steps=3,
            successful_steps=3,
            failed_steps=0,
            total_duration=5.0,
        )
        assert result.chain_name == "test_chain"
        assert result.success is True

    def test_failed_result(self):
        state = ChainState()
        result = ChainResult(
            chain_name="failed_chain",
            status=ChainStatus.FAILED,
            final_output=None,
            state=state,
            total_steps=3,
            successful_steps=2,
            failed_steps=1,
            total_duration=3.0,
            errors=["Step 3 failed"],
        )
        assert result.success is False
        assert len(result.errors) == 1


# ============================================================================
# Step Tests
# ============================================================================


class TestLLMStep:
    """Test LLMStep class."""

    def test_creation(self):
        step = LLMStep(
            "test_llm",
            "Hello {input}",
            lambda x: f"Response: {x}",
        )
        assert step.name == "test_llm"
        assert step.step_type == StepType.LLM_CALL

    def test_execute(self):
        step = LLMStep(
            "echo",
            "Say: {input}",
            lambda prompt: f"LLM says: {prompt}",
        )
        state = ChainState()
        result = step.execute("hello", state)

        assert "Say: hello" in result

    def test_with_parser(self):
        step = LLMStep(
            "parse",
            "{input}",
            lambda x: "raw output",
            output_parser=lambda x: x.upper(),
        )
        state = ChainState()
        result = step.execute("test", state)

        assert result == "RAW OUTPUT"

    def test_with_state_variables(self):
        step = LLMStep(
            "with_state",
            "Input: {input}, Name: {name}",
            lambda x: x,
        )
        state = ChainState()
        state.set("name", "Alice")
        result = step.execute("hello", state)

        assert "Alice" in result


class TestTransformStep:
    """Test TransformStep class."""

    def test_creation(self):
        step = TransformStep(
            "transform",
            lambda x, s: x.upper(),
        )
        assert step.step_type == StepType.TRANSFORM

    def test_execute(self):
        step = TransformStep(
            "uppercase",
            lambda x, s: x.upper(),
        )
        state = ChainState()
        result = step.execute("hello", state)

        assert result == "HELLO"

    def test_with_state_access(self):
        def transform_fn(data, state):
            prefix = state.get("prefix", "")
            return f"{prefix}{data}"

        step = TransformStep("prefix", transform_fn)
        state = ChainState()
        state.set("prefix", ">> ")
        result = step.execute("hello", state)

        assert result == ">> hello"


class TestValidatorStep:
    """Test ValidatorStep class."""

    def test_valid_input(self):
        step = ValidatorStep(
            "length_check",
            lambda x: (len(x) > 3, None if len(x) > 3 else "Too short"),
        )
        state = ChainState()
        result = step.execute("hello", state)

        assert result == "hello"

    def test_invalid_input_fail(self):
        step = ValidatorStep(
            "length_check",
            lambda x: (len(x) > 10, "Too short"),
            on_failure="fail",
        )
        state = ChainState()

        with pytest.raises(ValueError, match="Validation failed"):
            step.execute("short", state)

    def test_invalid_input_skip(self):
        step = ValidatorStep(
            "length_check",
            lambda x: (len(x) > 10, "Too short"),
            on_failure="skip",
        )
        state = ChainState()
        result = step.execute("short", state)

        assert result == "short"
        assert state.get("_skip_remaining") is True


class TestConditionalStep:
    """Test ConditionalStep class."""

    def test_true_branch(self):
        if_true = TransformStep("upper", lambda x, s: x.upper())
        if_false = TransformStep("lower", lambda x, s: x.lower())

        step = ConditionalStep(
            "conditional",
            lambda x, s: len(x) > 3,
            if_true,
            if_false,
        )
        state = ChainState()
        result = step.execute("hello", state)

        assert result == "HELLO"

    def test_false_branch(self):
        if_true = TransformStep("upper", lambda x, s: x.upper())
        if_false = TransformStep("lower", lambda x, s: x.lower())

        step = ConditionalStep(
            "conditional",
            lambda x, s: len(x) > 10,
            if_true,
            if_false,
        )
        state = ChainState()
        result = step.execute("HELLO", state)

        assert result == "hello"

    def test_no_false_branch(self):
        if_true = TransformStep("upper", lambda x, s: x.upper())

        step = ConditionalStep(
            "conditional",
            lambda x, s: False,
            if_true,
        )
        state = ChainState()
        result = step.execute("hello", state)

        # Should return unchanged
        assert result == "hello"


class TestRouterStep:
    """Test RouterStep class."""

    def test_keyword_routing(self):
        routes = {
            "question": TransformStep("q", lambda x, s: f"Answer: {x}"),
            "command": TransformStep("c", lambda x, s: f"Executing: {x}"),
        }
        step = RouterStep("router", routes, RouterStrategy.KEYWORD)
        state = ChainState()

        result = step.execute("This is a question?", state)
        assert "Answer:" in result

    def test_regex_routing(self):
        routes = {
            r"\d+": TransformStep("num", lambda x, s: f"Number: {x}"),
            r"[a-z]+": TransformStep("text", lambda x, s: f"Text: {x}"),
        }
        step = RouterStep("router", routes, RouterStrategy.REGEX)
        state = ChainState()

        result = step.execute("123", state)
        assert "Number:" in result

    def test_custom_routing(self):
        routes = {
            "a": TransformStep("a", lambda x, s: "Route A"),
            "b": TransformStep("b", lambda x, s: "Route B"),
        }

        def classifier(x):
            return "a" if x.startswith("A") else "b"

        step = RouterStep(
            "router",
            routes,
            RouterStrategy.CUSTOM,
            classifier_fn=classifier,
        )
        state = ChainState()

        assert step.execute("Apple", state) == "Route A"
        assert step.execute("Banana", state) == "Route B"

    def test_default_route(self):
        routes = {
            "specific": TransformStep("s", lambda x, s: "Specific"),
            "default": TransformStep("d", lambda x, s: "Default"),
        }
        step = RouterStep(
            "router",
            routes,
            RouterStrategy.KEYWORD,
            default_route="default",
        )
        state = ChainState()

        result = step.execute("unknown input", state)
        assert result == "Default"


class TestLoopStep:
    """Test LoopStep class."""

    def test_basic_loop(self):
        body = TransformStep("add_x", lambda x, s: x + "x")

        def exit_when_long(output, state, iteration):
            return len(output) >= 5

        step = LoopStep("loop", body, exit_when_long, max_iterations=10)
        state = ChainState()
        result = step.execute("a", state)

        assert len(result) >= 5
        assert result.count("x") >= 4

    def test_max_iterations(self):
        body = TransformStep("inc", lambda x, s: x + 1)

        # Never exit naturally
        def never_exit(output, state, iteration):
            return False

        step = LoopStep("loop", body, never_exit, max_iterations=5)
        state = ChainState()
        result = step.execute(0, state)

        assert result == 5
        assert state.get("_loop_total_iterations") == 5

    def test_immediate_exit(self):
        body = TransformStep("never_run", lambda x, s: "should not see this")

        def immediate_exit(output, state, iteration):
            return True

        step = LoopStep("loop", body, immediate_exit)
        state = ChainState()
        result = step.execute("original", state)

        assert result == "original"


class TestParallelStep:
    """Test ParallelStep class."""

    def test_basic_parallel(self):
        steps = [
            TransformStep("upper", lambda x, s: x.upper()),
            TransformStep("lower", lambda x, s: x.lower()),
            TransformStep("reverse", lambda x, s: x[::-1]),
        ]
        step = ParallelStep("parallel", steps)
        state = ChainState()

        result = step.execute("Hello", state)

        assert result == ["HELLO", "hello", "olleH"]

    def test_with_aggregator(self):
        steps = [
            TransformStep("len1", lambda x, s: len(x)),
            TransformStep("len2", lambda x, s: len(x) * 2),
        ]

        def sum_aggregator(results):
            return sum(results)

        step = ParallelStep("parallel", steps, sum_aggregator)
        state = ChainState()

        result = step.execute("hello", state)  # len=5
        assert result == 15  # 5 + 10


class TestSubchainStep:
    """Test SubchainStep class."""

    def test_basic_subchain(self):
        # Create a simple subchain
        subchain = Chain("sub")
        subchain.add_step(TransformStep("upper", lambda x, s: x.upper()))
        subchain.add_step(TransformStep("exclaim", lambda x, s: x + "!"))

        step = SubchainStep("run_sub", subchain)
        state = ChainState()

        result = step.execute("hello", state)
        assert result == "HELLO!"


# ============================================================================
# Chain Tests
# ============================================================================


class TestChain:
    """Test Chain class."""

    def test_simple_chain(self):
        chain = Chain("simple")
        chain.add_step(TransformStep("upper", lambda x, s: x.upper()))
        chain.add_step(TransformStep("exclaim", lambda x, s: x + "!"))

        result = chain.run("hello")

        assert result.success
        assert result.final_output == "HELLO!"
        assert result.successful_steps == 2

    def test_chain_with_state(self):
        chain = Chain("with_state")

        def store_original(x, state):
            state.set("original", x)
            return x

        def combine(x, state):
            original = state.get("original")
            return f"{original} -> {x}"

        chain.add_step(TransformStep("store", store_original))
        chain.add_step(TransformStep("upper", lambda x, s: x.upper()))
        chain.add_step(TransformStep("combine", combine))

        result = chain.run("hello")

        assert "hello -> HELLO" in result.final_output

    def test_chain_failure(self):
        chain = Chain("failing")
        chain.add_step(TransformStep("ok", lambda x, s: x))
        chain.add_step(TransformStep("fail", lambda x, s: 1 / 0))  # Will fail

        result = chain.run("input")

        assert not result.success
        assert result.status == ChainStatus.FAILED
        assert result.failed_steps == 1

    def test_error_handler(self):
        chain = Chain("with_handler")
        chain.add_step(TransformStep("fail", lambda x, s: 1 / 0))

        def handler(e, state):
            return "recovered"

        chain.set_error_handler(handler)

        result = chain.run("input")

        # Error was handled, so chain should continue
        assert result.final_output == "recovered"

    def test_pre_post_hooks(self):
        chain = Chain("with_hooks")
        chain.add_step(TransformStep("main", lambda x, s: x.upper()))

        calls = []

        def pre_hook(x, state):
            calls.append("pre")
            return x

        def post_hook(x, state):
            calls.append("post")
            return x

        chain.add_pre_hook(pre_hook)
        chain.add_post_hook(post_hook)

        chain.run("test")

        assert calls == ["pre", "post"]


# ============================================================================
# ChainBuilder Tests
# ============================================================================


class TestChainBuilder:
    """Test ChainBuilder class."""

    def test_basic_builder(self):
        def mock_model(prompt):
            return f"Response to: {prompt[:20]}"

        chain = (
            create_chain("builder_test", mock_model)
            .llm("step1", "Process: {input}")
            .transform("clean", lambda x, s: x.strip())
            .build()
        )

        result = chain.run("test input")
        assert result.success

    def test_builder_with_validation(self):
        def mock_model(prompt):
            return "valid response"

        chain = (
            create_chain("validated", mock_model)
            .llm("generate", "{input}")
            .validate("check", lambda x: (len(x) > 5, "Too short"))
            .build()
        )

        result = chain.run("test")
        assert result.success

    def test_builder_with_branch(self):
        def mock_model(prompt):
            return prompt

        upper_step = TransformStep("upper", lambda x, s: x.upper())
        lower_step = TransformStep("lower", lambda x, s: x.lower())

        chain = (
            create_chain("branched", mock_model)
            .branch(
                "size_check",
                lambda x, s: len(x) > 3,
                upper_step,
                lower_step,
            )
            .build()
        )

        assert chain.run("hello").final_output == "HELLO"
        assert chain.run("hi").final_output == "hi"

    def test_builder_fluent_interface(self):
        def mock_model(prompt):
            return prompt

        builder = create_chain("fluent", mock_model)
        builder = builder.with_description("A fluent chain")
        builder = builder.llm("s1", "{input}")
        builder = builder.transform("s2", lambda x, s: x + "!")
        chain = builder.build()

        assert chain.description == "A fluent chain"
        assert len(chain.steps) == 2


# ============================================================================
# ChainRegistry Tests
# ============================================================================


class TestChainRegistry:
    """Test ChainRegistry class."""

    def test_register_and_get(self):
        registry = ChainRegistry()
        chain = Chain("test_chain")

        registry.register(chain)

        assert registry.get("test_chain") is chain
        assert registry.get("nonexistent") is None

    def test_list_chains(self):
        registry = ChainRegistry()
        registry.register(Chain("chain1"))
        registry.register(Chain("chain2"))

        chains = registry.list_chains()
        assert "chain1" in chains
        assert "chain2" in chains

    def test_run_registered_chain(self):
        registry = ChainRegistry()
        chain = Chain("runner")
        chain.add_step(TransformStep("upper", lambda x, s: x.upper()))

        registry.register(chain)
        result = registry.run("runner", "hello")

        assert result is not None
        assert result.final_output == "HELLO"


# ============================================================================
# WorkflowTemplate Tests
# ============================================================================


class TestWorkflowTemplate:
    """Test WorkflowTemplate class."""

    def test_summarize_and_answer(self):
        def mock_model(prompt):
            if "Summarize" in prompt:
                return "This is a summary."
            return "This is an answer."

        chain = WorkflowTemplate.summarize_and_answer(mock_model)
        result = chain.run("Long text here...")

        assert result.success
        assert result.total_steps == 2

    def test_extract_and_validate(self):
        def mock_model(prompt):
            return "extracted: value=42"

        def validator(x):
            return ("value=" in x, "No value found")

        chain = WorkflowTemplate.extract_and_validate(mock_model, validator)
        result = chain.run("raw data")

        assert result.success

    def test_iterative_refinement(self):
        call_count = [0]

        def mock_model(prompt):
            call_count[0] += 1
            return f"Refinement {call_count[0]}"

        chain = WorkflowTemplate.iterative_refinement(
            mock_model,
            max_iterations=3,
        )
        result = chain.run("initial prompt")

        assert result.success
        # Initial + 3 refinements = 4 calls
        assert call_count[0] == 4


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_create_chain(self):
        def model(x):
            return x

        builder = create_chain("test", model)
        assert isinstance(builder, ChainBuilder)

    def test_create_llm_step(self):
        step = create_llm_step(
            "test",
            "{input}",
            lambda x: x,
        )
        assert isinstance(step, LLMStep)

    def test_create_transform_step(self):
        step = create_transform_step(
            "test",
            lambda x, s: x.upper(),
        )
        assert isinstance(step, TransformStep)

    def test_create_validator_step(self):
        step = create_validator_step(
            "test",
            lambda x: (True, None),
        )
        assert isinstance(step, ValidatorStep)

    def test_run_chain(self):
        chain = Chain("test")
        chain.add_step(TransformStep("id", lambda x, s: x))

        result = run_chain(chain, "input")
        assert result.final_output == "input"

    def test_simple_chain(self):
        chain = simple_chain(
            [
                str.upper,
                lambda x: x + "!",
            ]
        )

        result = chain.run("hello")
        assert result.final_output == "HELLO!"

    def test_sequential_llm_chain(self):
        def model(prompt):
            return f"[{prompt[:10]}]"

        chain = sequential_llm_chain(
            ["First: {input}", "Second: {input}"],
            model,
        )

        result = chain.run("test")
        assert result.success
        assert len(chain.steps) == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for chain workflows."""

    def test_complex_workflow(self):
        def mock_model(prompt):
            return f"Processed: {prompt[:30]}"

        # Build a complex chain
        chain = (
            create_chain("complex", mock_model)
            .transform("clean", lambda x, s: x.strip())
            .llm("process", "Handle: {input}")
            .validate("check", lambda x: (len(x) > 0, "Empty output"))
            .transform("format", lambda x, s: f"[{x}]")
            .build()
        )

        result = chain.run("  test input  ")

        assert result.success
        assert result.successful_steps == 4
        assert "[" in result.final_output

    def test_nested_chains(self):
        # Create inner chains
        preprocess = Chain("preprocess")
        preprocess.add_step(TransformStep("lower", lambda x, s: x.lower()))
        preprocess.add_step(TransformStep("strip", lambda x, s: x.strip()))

        postprocess = Chain("postprocess")
        postprocess.add_step(TransformStep("upper", lambda x, s: x.upper()))
        postprocess.add_step(TransformStep("exclaim", lambda x, s: x + "!"))

        # Create main chain with subchains
        main = Chain("main")
        main.add_step(SubchainStep("pre", preprocess))
        main.add_step(TransformStep("process", lambda x, s: f"<{x}>"))
        main.add_step(SubchainStep("post", postprocess))

        result = main.run("  Hello World  ")

        assert result.success
        assert result.final_output == "<HELLO WORLD>!"

    def test_state_persistence(self):
        chain = Chain("stateful")

        def step1(x, state):
            state.set("seen_input", x)
            return x.upper()

        def step2(x, state):
            original = state.get("seen_input")
            return f"{original} became {x}"

        chain.add_step(TransformStep("s1", step1))
        chain.add_step(TransformStep("s2", step2))

        result = chain.run("hello")
        assert "hello became HELLO" in result.final_output


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_chain(self):
        chain = Chain("empty")
        result = chain.run("input")

        assert result.success
        assert result.final_output == "input"
        assert result.total_steps == 0

    def test_none_input(self):
        chain = Chain("none_input")
        chain.add_step(TransformStep("check", lambda x, s: x is None))

        result = chain.run(None)
        assert result.final_output is True

    def test_chain_with_initial_state(self):
        chain = Chain("with_state")

        def use_preset(x, state):
            preset = state.get("preset", "default")
            return f"{preset}: {x}"

        chain.add_step(TransformStep("use", use_preset))

        initial_state = ChainState()
        initial_state.set("preset", "custom")

        result = chain.run("input", initial_state)
        assert "custom: input" in result.final_output

    def test_validation_error_info(self):
        chain = Chain("validation_test")
        chain.add_step(
            ValidatorStep(
                "must_be_long",
                lambda x: (len(x) > 100, "Input must be longer than 100 characters"),
                on_failure="fail",
            )
        )

        result = chain.run("short")
        assert not result.success
        assert "must be longer" in str(result.errors)
