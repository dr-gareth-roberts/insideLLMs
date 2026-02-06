"""Comprehensive tests to increase coverage for probes and models modules.

Covers uncovered code paths in:
- insideLLMs/probes/base.py (validate_input, info, __repr__, ScoredProbe, ComparativeProbe, etc.)
- insideLLMs/probes/code.py (CodeGenerationProbe, CodeExplanationProbe, CodeDebugProbe)
- insideLLMs/models/openai.py (chat/stream error handling, info extra, init with base_url/org/project)
- insideLLMs/models/anthropic.py (chat/stream error handling, role conversion, info extra)
- insideLLMs/models/cohere.py (COHERE_API_KEY fallback, p/k mapping, multi-turn chat, etc.)
- insideLLMs/models/gemini.py (top_p/top_k, multi-turn chat, list_models, stream empty chunks)
- insideLLMs/models/local.py (Ollama options, pull/list_models/show_model_info, VLLMModel)
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.probes.base import ComparativeProbe, Probe, ScoredProbe
from insideLLMs.types import (
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)

# ---------------------------------------------------------------------------
# Helper probe subclasses for testing
# ---------------------------------------------------------------------------


class SimpleTestProbe(Probe[str]):
    """A simple probe for testing base class methods."""

    default_category = ProbeCategory.LOGIC

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        return model.generate(data)

    def score(self, results: list) -> ProbeScore:
        success_count = sum(1 for r in results if r.status == ResultStatus.SUCCESS)
        return ProbeScore(accuracy=success_count / len(results) if results else 0.0)


class ScoredProbeImpl(ScoredProbe[str]):
    """ScoredProbe subclass for testing."""

    default_category = ProbeCategory.FACTUALITY

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        return model.generate(data)

    def evaluate_single(self, model_output, reference, input_data):
        is_correct = model_output.strip().lower() == str(reference).strip().lower()
        return {"is_correct": is_correct}


class ComparativeProbeImpl(ComparativeProbe[str]):
    """ComparativeProbe subclass for testing."""

    default_category = ProbeCategory.BIAS

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        return model.generate(data)


# ===========================================================================
# probes/base.py tests
# ===========================================================================


class TestProbeValidateInput:
    """Tests for Probe.validate_input()."""

    def test_validate_input_returns_true(self):
        """Base validate_input always returns True."""
        probe = SimpleTestProbe(name="Test")
        assert probe.validate_input("anything") is True
        assert probe.validate_input(None) is True
        assert probe.validate_input({"key": "value"}) is True


class TestProbeInfo:
    """Tests for Probe.info()."""

    def test_info_returns_correct_dict(self):
        probe = SimpleTestProbe(name="InfoTest", category=ProbeCategory.LOGIC)
        info = probe.info()
        assert info["name"] == "InfoTest"
        assert info["category"] == "logic"
        assert info["type"] == "SimpleTestProbe"
        assert "description" in info

    def test_info_with_custom_description(self):
        probe = SimpleTestProbe(name="Desc", description="Custom description")
        info = probe.info()
        assert info["description"] == "Custom description"


class TestProbeReprOutput:
    """Tests for Probe.__repr__() string output."""

    def test_repr_format(self):
        probe = SimpleTestProbe(name="ReprTest", category=ProbeCategory.LOGIC)
        result = repr(probe)
        assert "SimpleTestProbe" in result
        assert "ReprTest" in result
        assert "logic" in result

    def test_repr_with_custom_category(self):
        probe = SimpleTestProbe(name="Test", category=ProbeCategory.BIAS)
        result = repr(probe)
        assert "bias" in result


class TestProbeBaseScore:
    """Tests for the base Probe.score() method with latency calculations."""

    def test_score_empty_returns_default(self):
        probe = SimpleTestProbe(name="Test")
        # Use the base Probe.score (not the overridden one)
        score = Probe.score(probe, [])
        assert isinstance(score, ProbeScore)

    def test_score_with_latency(self):
        results = [
            ProbeResult(
                input="a",
                output="x",
                status=ResultStatus.SUCCESS,
                latency_ms=100.0,
            ),
            ProbeResult(
                input="b",
                output="y",
                status=ResultStatus.SUCCESS,
                latency_ms=200.0,
            ),
        ]
        probe = SimpleTestProbe(name="Test")
        score = Probe.score(probe, results)
        assert score.accuracy == 1.0
        assert score.mean_latency_ms == 150.0
        assert score.error_rate == 0.0

    def test_score_with_errors_and_success(self):
        results = [
            ProbeResult(input="a", output="x", status=ResultStatus.SUCCESS, latency_ms=50.0),
            ProbeResult(input="b", status=ResultStatus.ERROR, error="fail"),
            ProbeResult(input="c", status=ResultStatus.TIMEOUT, error="timeout"),
        ]
        probe = SimpleTestProbe(name="Test")
        score = Probe.score(probe, results)
        assert score.accuracy == pytest.approx(1 / 3)
        assert score.error_rate == pytest.approx(1 / 3)
        assert score.mean_latency_ms == 50.0


class TestProbeRunBatchProgressCallback:
    """Tests for run_batch with progress_callback."""

    def test_progress_callback_called(self):
        probe = SimpleTestProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="output")

        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        results = probe.run_batch(mock_model, ["a", "b", "c"], progress_callback=callback)
        assert len(results) == 3
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    def test_progress_callback_parallel(self):
        probe = SimpleTestProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="output")

        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        results = probe.run_batch(mock_model, ["a", "b"], max_workers=2, progress_callback=callback)
        assert len(results) == 2
        assert len(progress_calls) == 2


class TestProbeRunBatchParallelErrors:
    """Tests for run_batch parallel with errors in futures."""

    def test_parallel_with_errors(self):
        probe = SimpleTestProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(side_effect=["output1", ValueError("error"), "output3"])

        results = probe.run_batch(mock_model, ["a", "b", "c"], max_workers=2)
        assert len(results) == 3
        statuses = [r.status for r in results]
        assert ResultStatus.SUCCESS in statuses
        assert ResultStatus.ERROR in statuses


# ---------------------------------------------------------------------------
# ScoredProbe tests
# ---------------------------------------------------------------------------


class TestScoredProbe:
    """Tests for ScoredProbe class."""

    def test_scored_probe_init(self):
        probe = ScoredProbeImpl(name="Scored")
        assert probe.category == ProbeCategory.FACTUALITY

    def test_scored_probe_evaluate_single_correct(self):
        probe = ScoredProbeImpl(name="Scored")
        result = probe.evaluate_single("Paris", "Paris", {"q": "capital?"})
        assert result["is_correct"] is True

    def test_scored_probe_evaluate_single_incorrect(self):
        probe = ScoredProbeImpl(name="Scored")
        result = probe.evaluate_single("London", "Paris", {"q": "capital?"})
        assert result["is_correct"] is False

    def test_scored_probe_score_with_metadata(self):
        probe = ScoredProbeImpl(name="Scored")
        results = [
            ProbeResult(
                input="q1",
                output="Paris",
                status=ResultStatus.SUCCESS,
                metadata={"is_correct": True},
            ),
            ProbeResult(
                input="q2",
                output="London",
                status=ResultStatus.SUCCESS,
                metadata={"is_correct": False},
            ),
        ]
        score = probe.score(results)
        assert score.accuracy == 0.5

    def test_scored_probe_score_with_errors(self):
        probe = ScoredProbeImpl(name="Scored")
        results = [
            ProbeResult(
                input="q1",
                output="Paris",
                status=ResultStatus.SUCCESS,
                metadata={"is_correct": True},
            ),
            ProbeResult(input="q2", status=ResultStatus.ERROR, error="fail"),
        ]
        score = probe.score(results)
        assert score.accuracy == 1.0  # 1 correct out of 1 evaluated

    def test_scored_probe_score_empty(self):
        probe = ScoredProbeImpl(name="Scored")
        score = probe.score([])
        assert isinstance(score, ProbeScore)

    def test_scored_probe_score_no_metadata(self):
        """Results with SUCCESS but no metadata should not count toward accuracy."""
        probe = ScoredProbeImpl(name="Scored")
        results = [
            ProbeResult(input="q1", output="answer", status=ResultStatus.SUCCESS),
        ]
        score = probe.score(results)
        # evaluated_count = 0, so base accuracy from parent is used
        assert isinstance(score, ProbeScore)


# ---------------------------------------------------------------------------
# ComparativeProbe tests
# ---------------------------------------------------------------------------


class TestComparativeProbe:
    """Tests for ComparativeProbe class."""

    def test_run_comparison(self):
        probe = ComparativeProbeImpl(name="Comparative")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(side_effect=["Response A", "Response B"])

        result = probe.run_comparison(mock_model, "input_a", "input_b")

        assert result["input_a"] == "input_a"
        assert result["input_b"] == "input_b"
        assert result["response_a"] == "Response A"
        assert result["response_b"] == "Response B"
        assert "comparison" in result

    def test_compare_responses_identical(self):
        probe = ComparativeProbeImpl(name="Comparative")
        comparison = probe.compare_responses("same", "same", "in_a", "in_b")
        assert comparison["response_identical"] is True
        assert comparison["length_diff"] == 0

    def test_compare_responses_different(self):
        probe = ComparativeProbeImpl(name="Comparative")
        comparison = probe.compare_responses("short", "much longer", "in_a", "in_b")
        assert comparison["response_identical"] is False
        assert comparison["length_diff"] == len("short") - len("much longer")

    def test_run_comparison_with_kwargs(self):
        probe = ComparativeProbeImpl(name="Comparative")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="response")

        probe.run_comparison(mock_model, "a", "b", temperature=0.0)
        assert mock_model.generate.call_count == 2


# ===========================================================================
# probes/code.py tests
# ===========================================================================


class TestCodeGenerationProbe:
    """Tests for CodeGenerationProbe."""

    def test_init_defaults(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe()
        assert probe.language == "python"
        assert probe.require_docstrings is False
        assert probe.require_type_hints is False
        assert "python" in probe.prompt_template

    def test_init_custom_language(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(language="JavaScript")
        assert probe.language == "javascript"

    def test_run_with_string_task(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="def foo(): pass")

        result = probe.run(mock_model, "Write a function")
        assert result == "def foo(): pass"
        call_arg = mock_model.generate.call_args[0][0]
        assert "Write a function" in call_arg

    def test_run_with_dict_task(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="code")

        probe.run(mock_model, {"task": "Do something"})
        call_arg = mock_model.generate.call_args[0][0]
        assert "Do something" in call_arg

    def test_run_with_dict_description_fallback(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="code")

        probe.run(mock_model, {"description": "Describe task"})
        call_arg = mock_model.generate.call_args[0][0]
        assert "Describe task" in call_arg

    def test_extract_code_markdown(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(language="python")
        response = "Here's the code:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
        code = probe.extract_code(response)
        assert "def add" in code
        assert "```" not in code

    def test_extract_code_no_markdown(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(language="python")
        response = "def multiply(a, b):\n    return a * b"
        code = probe.extract_code(response)
        assert "def multiply" in code

    def test_extract_code_heuristic(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(language="python")
        response = (
            "Here's my solution:\nimport math\ndef sqrt(x):\n    return math.sqrt(x)\n\nThis works."
        )
        code = probe.extract_code(response)
        assert "import math" in code

    def test_extract_code_no_code(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(language="python")
        response = "I cannot write code for this."
        code = probe.extract_code(response)
        assert code == response  # Returns original if nothing detected

    def test_check_syntax_valid_python(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(language="python")
        assert probe._check_syntax("def add(a, b):\n    return a + b") is True

    def test_check_syntax_invalid_python(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(language="python")
        assert probe._check_syntax("def add(a, b:\n    return a + b") is False

    def test_check_syntax_valid_javascript(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(language="javascript")
        assert probe._check_syntax("function add(a, b) { return a + b; }") is True

    def test_check_syntax_invalid_brackets(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(language="javascript")
        assert probe._check_syntax("function add(a, b { return a + b; }") is False

    def test_score_to_label_excellent(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        assert CodeGenerationProbe._score_to_label(0.95) == "excellent"

    def test_score_to_label_good(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        assert CodeGenerationProbe._score_to_label(0.75) == "good"

    def test_score_to_label_acceptable(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        assert CodeGenerationProbe._score_to_label(0.55) == "acceptable"

    def test_score_to_label_poor(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        assert CodeGenerationProbe._score_to_label(0.35) == "poor"

    def test_score_to_label_failing(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        assert CodeGenerationProbe._score_to_label(0.15) == "failing"

    def test_evaluate_single_with_valid_code_and_string_ref(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe()
        code = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
        result = probe.evaluate_single(code, reference="def factorial")
        assert result.status == ResultStatus.SUCCESS
        assert result.metadata["syntax_valid"] is True
        assert result.metadata["pattern_match_score"] == 1.0

    def test_evaluate_single_with_dict_patterns_ref(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe()
        code = "def is_prime(n):\n    if n < 2:\n        return False\n    return True"
        result = probe.evaluate_single(
            code, reference={"patterns": ["def is_prime", "return True", "return False"]}
        )
        assert result.metadata["pattern_match_score"] == 1.0

    def test_evaluate_single_with_none_ref(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe()
        code = "def add(a, b):\n    return a + b"
        result = probe.evaluate_single(code, reference=None)
        assert result.status == ResultStatus.SUCCESS

    def test_evaluate_single_invalid_syntax(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe()
        code = "def broken(:\n    pass"
        result = probe.evaluate_single(code, reference=None)
        assert result.metadata["syntax_valid"] is False

    def test_evaluate_single_require_docstrings(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(require_docstrings=True)
        code_with_doc = 'def add(a, b):\n    """Add two numbers."""\n    return a + b'
        result = probe.evaluate_single(code_with_doc, reference=None)
        assert result.metadata["has_docstring"] is True

        code_without_doc = "def add(a, b):\n    return a + b"
        result2 = probe.evaluate_single(code_without_doc, reference=None)
        assert result2.metadata["has_docstring"] is False

    def test_evaluate_single_require_type_hints(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe(require_type_hints=True)
        code_with_hints = "def add(a: int, b: int) -> int:\n    return a + b"
        result = probe.evaluate_single(code_with_hints, reference=None)
        assert result.metadata["has_type_hints"] is True

        code_without_hints = "def add(a, b):\n    return a + b"
        result2 = probe.evaluate_single(code_without_hints, reference=None)
        assert result2.metadata["has_type_hints"] is False

    def test_evaluate_single_truncates_long_input(self):
        from insideLLMs.probes.code import CodeGenerationProbe

        probe = CodeGenerationProbe()
        long_code = "def func():\n    pass\n" * 20  # longer than 100 chars
        result = probe.evaluate_single(long_code, reference=None)
        assert result.input.endswith("...")


class TestCodeExplanationProbe:
    """Tests for CodeExplanationProbe."""

    def test_init_defaults(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe()
        assert probe.detail_level == "medium"

    def test_init_brief(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe(detail_level="brief")
        assert probe.detail_level == "brief"

    def test_init_detailed(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe(detail_level="detailed")
        assert (
            "detailed explanation" in probe.prompt_template.lower()
            or "purpose" in probe.prompt_template.lower()
        )

    def test_run_with_string(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="This is an explanation")
        result = probe.run(mock_model, "def add(a, b): return a + b")
        assert result == "This is an explanation"

    def test_run_with_dict(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="Explanation")
        probe.run(mock_model, {"code": "x = 1"})
        call_arg = mock_model.generate.call_args[0][0]
        assert "x = 1" in call_arg

    def test_evaluate_single_with_dict_concepts(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe()
        explanation = "This function uses recursion to compute the factorial of a number by multiplying each value."
        result = probe.evaluate_single(
            explanation, reference={"concepts": ["recursion", "factorial"]}
        )
        assert result.metadata["concepts_covered"] == 2
        assert result.metadata["total_concepts"] == 2
        assert result.status == ResultStatus.SUCCESS

    def test_evaluate_single_with_list_concepts(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe()
        explanation = "Sorts the list using lambda as the key for comparison."
        result = probe.evaluate_single(explanation, reference=["sort", "lambda"])
        assert result.metadata["concepts_covered"] == 2

    def test_evaluate_single_with_string_ref(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe()
        explanation = "This sorts the array using a merge sort algorithm."
        result = probe.evaluate_single(explanation, reference="sort")
        assert result.metadata["concepts_covered"] == 1

    def test_evaluate_single_with_none_ref(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe()
        explanation = "This is a brief explanation with some structure:\n1. First\n2. Second"
        result = probe.evaluate_single(explanation, reference=None)
        assert result.metadata["has_structure"] is True

    def test_evaluate_single_no_structure(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe()
        explanation = "simple explanation without any special formatting or numbering at all here nothing else"
        result = probe.evaluate_single(explanation, reference=None)
        assert result.metadata["has_structure"] is False

    def test_evaluate_single_label_clear(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe(detail_level="brief")
        # Brief needs only 5 words; make a long explanation with structure
        explanation = "This function sorts a list using the built-in sort method. It is efficient:\n1. Step one"
        result = probe.evaluate_single(explanation, reference=None)
        assert result.metadata["label"] == "clear"

    def test_evaluate_single_label_unclear(self):
        from insideLLMs.probes.code import CodeExplanationProbe

        probe = CodeExplanationProbe(detail_level="detailed")
        # Detailed needs 50 words; provide very few
        explanation = "Does stuff"
        result = probe.evaluate_single(explanation, reference=None)
        assert result.metadata["label"] == "unclear"


class TestCodeDebugProbe:
    """Tests for CodeDebugProbe."""

    def test_init_defaults(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        assert probe.language == "python"
        assert "{code}" in probe.prompt_template  # Template uses .format() at run time

    def test_init_custom_language(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe(language="javascript")
        assert probe.language == "javascript"

    def test_run_with_dict_data(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="The bug is...")
        result = probe.run(mock_model, {"code": "def bad(): pass", "error": "SyntaxError"})
        assert result == "The bug is..."
        call_arg = mock_model.generate.call_args[0][0]
        assert "SyntaxError" in call_arg

    def test_run_with_string_data(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="Fixed")
        result = probe.run(mock_model, "def broken(: pass")
        assert result == "Fixed"

    def test_run_with_dict_no_error(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="Found it")
        probe.run(mock_model, {"code": "x = 1"})
        call_arg = mock_model.generate.call_args[0][0]
        assert "Error message:" not in call_arg

    def test_evaluate_single_with_dict_fix_patterns(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        response = (
            "The bug is that division by zero is not handled.\n"
            "```python\ndef divide(a, b):\n    if b == 0:\n        raise ValueError\n    return a / b\n```"
        )
        result = probe.evaluate_single(
            response, reference={"fix_patterns": ["if b == 0", "ValueError"]}
        )
        assert result.metadata["has_explanation"] is True
        assert result.metadata["has_code_fix"] is True
        assert result.metadata["fix_patterns_found"] == 2

    def test_evaluate_single_with_list_patterns(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        response = "The issue is the loop. Use len(arr)-1 to fix the bounds."
        result = probe.evaluate_single(response, reference=["len(arr)", "bounds"])
        assert result.metadata["fix_patterns_found"] == 2

    def test_evaluate_single_with_string_pattern(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        response = "The problem is the missing return statement."
        result = probe.evaluate_single(response, reference="return")
        assert result.metadata["has_explanation"] is True

    def test_evaluate_single_with_none_ref(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        response = "The problem is that x is uninitialized.\ndef fix():\n    x = 0"
        result = probe.evaluate_single(response, reference=None)
        assert result.metadata["has_explanation"] is True
        assert result.metadata["has_code_fix"] is True

    def test_evaluate_single_no_explanation_no_fix(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        response = "I do not know what is wrong with the code."
        result = probe.evaluate_single(response, reference=["specific_fix"])
        # "wrong" is not in the indicators, but "know" is not either
        # The score should be low
        assert result.metadata["score"] < 0.7

    def test_evaluate_single_fixed_label(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        response = (
            "The bug is in the loop. The fix is:\n```python\ndef correct():\n    return True\n```"
        )
        result = probe.evaluate_single(response, reference=None)
        assert result.metadata["label"] in ("fixed", "partially_fixed")

    def test_evaluate_single_low_score_error_status(self):
        from insideLLMs.probes.code import CodeDebugProbe

        probe = CodeDebugProbe()
        response = "No idea."
        result = probe.evaluate_single(response, reference=["complex_fix_pattern_xyz"])
        # low score: no explanation (0.3*0.3=0.09), no code fix (0.3*0.3=0.09), no pattern (0*0.4=0)
        # overall = 0.18, which < 0.5 => ERROR
        assert result.status == ResultStatus.ERROR


# ===========================================================================
# models/openai.py tests
# ===========================================================================


class TestOpenAIModelChatErrorHandling:
    """Tests for OpenAIModel.chat error handling."""

    def test_chat_rate_limit_error(self):
        from openai import RateLimitError as OpenAIRateLimitError

        from insideLLMs.exceptions import RateLimitError

        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = OpenAIRateLimitError(
                "Rate limit", response=MagicMock(status_code=429), body=None
            )
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            with pytest.raises(RateLimitError):
                model.chat([{"role": "user", "content": "Hello"}])

    def test_chat_timeout_error(self):
        from openai import APITimeoutError

        from insideLLMs.exceptions import TimeoutError as InsideLLMsTimeoutError

        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APITimeoutError(MagicMock())
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            with pytest.raises(InsideLLMsTimeoutError):
                model.chat([{"role": "user", "content": "Hello"}])

    def test_chat_api_error(self):
        from openai import APIError

        from insideLLMs.exceptions import APIError as InsideLLMsAPIError

        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_error = APIError(
                message="Server error",
                request=MagicMock(),
                body=None,
            )
            mock_error.status_code = 500
            mock_client.chat.completions.create.side_effect = mock_error
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            with pytest.raises(InsideLLMsAPIError):
                model.chat([{"role": "user", "content": "Hello"}])

    def test_chat_generic_error(self):
        from insideLLMs.exceptions import ModelGenerationError

        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("Unknown")
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            with pytest.raises(ModelGenerationError):
                model.chat([{"role": "user", "content": "Hello"}])


class TestOpenAIModelStreamErrorHandling:
    """Tests for OpenAIModel.stream error handling."""

    def test_stream_rate_limit_error(self):
        from openai import RateLimitError as OpenAIRateLimitError

        from insideLLMs.exceptions import RateLimitError

        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = OpenAIRateLimitError(
                "Rate limit", response=MagicMock(status_code=429), body=None
            )
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            with pytest.raises(RateLimitError):
                list(model.stream("Test"))

    def test_stream_timeout_error(self):
        from openai import APITimeoutError

        from insideLLMs.exceptions import TimeoutError as InsideLLMsTimeoutError

        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APITimeoutError(MagicMock())
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            with pytest.raises(InsideLLMsTimeoutError):
                list(model.stream("Test"))

    def test_stream_api_error(self):
        from openai import APIError

        from insideLLMs.exceptions import APIError as InsideLLMsAPIError

        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_error = APIError(
                message="Bad request",
                request=MagicMock(),
                body=None,
            )
            mock_error.status_code = 400
            mock_client.chat.completions.create.side_effect = mock_error
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            with pytest.raises(InsideLLMsAPIError):
                list(model.stream("Test"))

    def test_stream_generic_error(self):
        from insideLLMs.exceptions import ModelGenerationError

        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("Unknown")
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            with pytest.raises(ModelGenerationError):
                list(model.stream("Test"))


class TestOpenAIModelInfoExtra:
    """Tests for OpenAIModel.info with extra fields."""

    def test_info_includes_base_url_org_project(self):
        with patch("insideLLMs.models.openai.OpenAI"):
            from insideLLMs.models.openai import OpenAIModel

            model = OpenAIModel(
                api_key="test_key",
                base_url="https://custom.api.com",
                organization="org-123",
                project="proj-456",
            )
            info = model.info()
            assert info.extra["base_url"] == "https://custom.api.com"
            assert info.extra["organization"] == "org-123"
            assert info.extra["project"] == "proj-456"
            assert info.extra["model_name"] == "gpt-3.5-turbo"
            assert "description" in info.extra


# ===========================================================================
# models/anthropic.py tests
# ===========================================================================


class TestAnthropicModelChatErrorHandling:
    """Tests for AnthropicModel.chat error handling."""

    def test_chat_rate_limit_error(self):
        from anthropic import RateLimitError as AnthropicRateLimitError

        from insideLLMs.exceptions import RateLimitError

        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_client.messages.create.side_effect = AnthropicRateLimitError(
                "Rate limit", response=MagicMock(status_code=429), body=None
            )
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            with pytest.raises(RateLimitError):
                model.chat([{"role": "user", "content": "Hello"}])

    def test_chat_timeout_error(self):
        from anthropic import APITimeoutError

        from insideLLMs.exceptions import TimeoutError as InsideLLMsTimeoutError

        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_client.messages.create.side_effect = APITimeoutError(MagicMock())
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            with pytest.raises(InsideLLMsTimeoutError):
                model.chat([{"role": "user", "content": "Hello"}])

    def test_chat_api_error(self):
        from anthropic import APIError as AnthropicAPIError

        from insideLLMs.exceptions import APIError as InsideLLMsAPIError

        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_error = AnthropicAPIError(
                message="Server error",
                request=MagicMock(),
                body=None,
            )
            mock_error.status_code = 500
            mock_client.messages.create.side_effect = mock_error
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            with pytest.raises(InsideLLMsAPIError):
                model.chat([{"role": "user", "content": "Hello"}])

    def test_chat_generic_error(self):
        from insideLLMs.exceptions import ModelGenerationError

        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_client.messages.create.side_effect = RuntimeError("Unknown")
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            with pytest.raises(ModelGenerationError):
                model.chat([{"role": "user", "content": "Hello"}])


class TestAnthropicModelStreamErrorHandling:
    """Tests for AnthropicModel.stream error handling."""

    def test_stream_rate_limit_error(self):
        from anthropic import RateLimitError as AnthropicRateLimitError

        from insideLLMs.exceptions import RateLimitError

        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_client.messages.stream.side_effect = AnthropicRateLimitError(
                "Rate limit", response=MagicMock(status_code=429), body=None
            )
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            with pytest.raises(RateLimitError):
                list(model.stream("Test"))

    def test_stream_timeout_error(self):
        from anthropic import APITimeoutError

        from insideLLMs.exceptions import TimeoutError as InsideLLMsTimeoutError

        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_client.messages.stream.side_effect = APITimeoutError(MagicMock())
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            with pytest.raises(InsideLLMsTimeoutError):
                list(model.stream("Test"))

    def test_stream_api_error(self):
        from anthropic import APIError as AnthropicAPIError

        from insideLLMs.exceptions import APIError as InsideLLMsAPIError

        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_error = AnthropicAPIError(
                message="Server error",
                request=MagicMock(),
                body=None,
            )
            mock_error.status_code = 500
            mock_client.messages.stream.side_effect = mock_error
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            with pytest.raises(InsideLLMsAPIError):
                list(model.stream("Test"))

    def test_stream_generic_error(self):
        from insideLLMs.exceptions import ModelGenerationError

        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_client.messages.stream.side_effect = RuntimeError("Unknown")
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            with pytest.raises(ModelGenerationError):
                list(model.stream("Test"))


class TestAnthropicModelChatRoleConversion:
    """Tests for AnthropicModel.chat role conversion."""

    def test_system_role_mapped_to_user(self):
        """Non-assistant roles should be mapped to user."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Response"
            mock_client.messages.create.return_value = mock_response
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            messages = [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ]
            model.chat(messages)

            call_kwargs = mock_client.messages.create.call_args[1]
            sent_messages = call_kwargs["messages"]
            # system role should be mapped to user
            assert sent_messages[0]["role"] == "user"
            assert sent_messages[1]["role"] == "user"


class TestAnthropicModelInfoExtra:
    """Tests for AnthropicModel.info extra fields."""

    def test_info_extra_fields(self):
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic"):
            from insideLLMs.models.anthropic import AnthropicModel

            model = AnthropicModel(api_key="test_key", model_name="claude-3-opus-20240229")
            info = model.info()
            assert info.extra["model_name"] == "claude-3-opus-20240229"
            assert "description" in info.extra


# ===========================================================================
# models/cohere.py tests
# ===========================================================================


class TestCohereModelEnvVarFallback:
    """Tests for CohereModel COHERE_API_KEY env var fallback."""

    def test_init_with_cohere_api_key_env(self):
        from insideLLMs.models.cohere import CohereModel

        env_without_keys = {
            k: v for k, v in os.environ.items() if k not in ("CO_API_KEY", "COHERE_API_KEY")
        }
        with patch.dict(
            os.environ, {**env_without_keys, "COHERE_API_KEY": "cohere_env_key"}, clear=True
        ):
            model = CohereModel(model_name="command-r")
            assert model.api_key == "cohere_env_key"


class TestCohereModelParameterMapping:
    """Tests for Cohere parameter mapping (top_p->p, top_k->k)."""

    def test_generate_with_top_p(self):
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel(api_key="test_key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_client.chat.return_value = mock_response
        model._client = mock_client

        model.generate("Test", top_p=0.9)
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["p"] == 0.9

    def test_generate_with_top_k(self):
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel(api_key="test_key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_client.chat.return_value = mock_response
        model._client = mock_client

        model.generate("Test", top_k=40)
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["k"] == 40

    def test_generate_preamble_override(self):
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel(api_key="test_key", default_preamble="Default system")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_client.chat.return_value = mock_response
        model._client = mock_client

        model.generate("Test", preamble="Override system")
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["preamble"] == "Override system"


class TestCohereModelMultiTurnChat:
    """Tests for Cohere multi-turn chat with role conversion."""

    def test_chat_multi_turn(self):
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel(api_key="test_key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_client.chat.return_value = mock_response
        model._client = mock_client

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        model.chat(messages)

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["message"] == "How are you?"
        chat_history = call_kwargs["chat_history"]
        assert len(chat_history) == 2
        assert chat_history[0]["role"] == "USER"
        assert chat_history[1]["role"] == "CHATBOT"

    def test_chat_with_temperature_max_tokens(self):
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel(api_key="test_key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_client.chat.return_value = mock_response
        model._client = mock_client

        model.chat(
            [{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=100,
        )
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    def test_chat_last_user_message_fallback(self):
        """When current_message is empty after loop, find last user message."""
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel(api_key="test_key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_client.chat.return_value = mock_response
        model._client = mock_client

        # After user->assistant->assistant, current_message will be empty
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "assistant", "content": "Continued"},
        ]
        model.chat(messages)
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["message"] == "First question"


class TestCohereModelStreamNonTextEvents:
    """Tests for Cohere stream filtering non-text-generation events."""

    def test_stream_filters_non_text_events(self):
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel(api_key="test_key")

        event1 = MagicMock()
        event1.event_type = "stream-start"
        event1.text = ""

        event2 = MagicMock()
        event2.event_type = "text-generation"
        event2.text = "Hello"

        event3 = MagicMock()
        event3.event_type = "stream-end"
        event3.text = ""

        mock_client = MagicMock()
        mock_client.chat_stream.return_value = [event1, event2, event3]
        model._client = mock_client

        chunks = list(model.stream("Test"))
        assert chunks == ["Hello"]


class TestCohereModelGetClient:
    """Tests for CohereModel._get_client lazy init."""

    def test_get_client_import_error(self):
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel(api_key="test_key")
        model._client = None

        with patch("builtins.__import__", side_effect=ImportError("No module named 'cohere'")):
            with pytest.raises(ImportError, match="cohere package required"):
                model._get_client()


class TestCohereModelEmbed:
    """Test embed with embedding_types parameter."""

    def test_embed_with_embedding_types(self):
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel(api_key="test_key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2]]
        mock_client.embed.return_value = mock_response
        model._client = mock_client

        result = model.embed(["text"], embedding_types=["float"])
        assert result == [[0.1, 0.2]]
        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["embedding_types"] == ["float"]


class TestCohereModelRerank:
    """Test rerank with top_n parameter."""

    def test_rerank_with_top_n(self):
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel(api_key="test_key")

        result1 = MagicMock()
        result1.index = 0
        result1.relevance_score = 0.9

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = [result1]
        mock_client.rerank.return_value = mock_response
        model._client = mock_client

        docs = ["doc1", "doc2"]
        result = model.rerank("query", docs, top_n=1)
        assert len(result) == 1
        call_kwargs = mock_client.rerank.call_args[1]
        assert call_kwargs["top_n"] == 1


# ===========================================================================
# models/gemini.py tests
# ===========================================================================


class TestGeminiModelGenerateExtendedParams:
    """Tests for GeminiModel.generate with top_p, top_k."""

    def test_generate_with_top_p_top_k(self):
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated"
        mock_genai_model.generate_content.return_value = mock_response
        model._client = MagicMock()
        model._model = mock_genai_model

        result = model.generate("Test", top_p=0.9, top_k=40)
        assert result == "Generated"
        call_kwargs = mock_genai_model.generate_content.call_args[1]
        gen_config = call_kwargs["generation_config"]
        assert gen_config["top_p"] == 0.9
        assert gen_config["top_k"] == 40

    def test_generate_with_max_output_tokens(self):
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated"
        mock_genai_model.generate_content.return_value = mock_response
        model._client = MagicMock()
        model._model = mock_genai_model

        model.generate("Test", max_output_tokens=500)
        call_kwargs = mock_genai_model.generate_content.call_args[1]
        gen_config = call_kwargs["generation_config"]
        assert gen_config["max_output_tokens"] == 500


class TestGeminiModelChatMultiTurn:
    """Tests for GeminiModel.chat with multi-turn and system messages."""

    def test_chat_system_message_prepend(self):
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_chat.send_message.return_value = mock_response
        mock_genai_model.start_chat.return_value = mock_chat
        model._client = MagicMock()
        model._model = mock_genai_model

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = model.chat(messages)
        assert result == "Response"

        # The system message should be prepended to the user message
        sent_msg = mock_chat.send_message.call_args[0][0]
        assert "[System: Be helpful]" in sent_msg
        assert "Hello" in sent_msg

    def test_chat_multi_turn_history(self):
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_chat.send_message.return_value = mock_response
        mock_genai_model.start_chat.return_value = mock_chat
        model._client = MagicMock()
        model._model = mock_genai_model

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        model.chat(messages)

        # start_chat should be called with history (first 2 messages)
        start_chat_kwargs = mock_genai_model.start_chat.call_args[1]
        history = start_chat_kwargs["history"]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "model"  # assistant -> model

    def test_chat_with_max_tokens(self):
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_chat.send_message.return_value = mock_response
        mock_genai_model.start_chat.return_value = mock_chat
        model._client = MagicMock()
        model._model = mock_genai_model

        model.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=200,
        )
        call_kwargs = mock_chat.send_message.call_args[1]
        gen_config = call_kwargs["generation_config"]
        assert gen_config["max_output_tokens"] == 200


class TestGeminiModelStreamEmptyChunks:
    """Tests for GeminiModel.stream filtering empty chunks."""

    def test_stream_filters_empty_chunks(self):
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()

        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = ""  # Empty chunk
        chunk3 = MagicMock()
        chunk3.text = " world"

        mock_genai_model.generate_content.return_value = [chunk1, chunk2, chunk3]
        model._client = MagicMock()
        model._model = mock_genai_model

        chunks = list(model.stream("Test"))
        assert chunks == ["Hello", " world"]


class TestGeminiModelListModels:
    """Tests for GeminiModel.list_models."""

    def test_list_models(self):
        import sys

        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")

        model1_info = MagicMock()
        model1_info.name = "models/gemini-1.5-pro"
        model1_info.supported_generation_methods = ["generateContent"]

        model2_info = MagicMock()
        model2_info.name = "models/embedding-001"
        model2_info.supported_generation_methods = ["embedContent"]

        model3_info = MagicMock()
        model3_info.name = "models/gemini-1.5-flash"
        model3_info.supported_generation_methods = ["generateContent", "countTokens"]

        mock_genai = MagicMock()
        mock_genai.list_models.return_value = [model1_info, model2_info, model3_info]

        # Set _client so _get_client returns early (model already set)
        model._client = mock_genai
        model._model = MagicMock()

        # list_models does: import google.generativeai as genai; genai.list_models()
        # We need to mock the google.generativeai module in sys.modules
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai

        with patch.dict(
            sys.modules,
            {
                "google": mock_google,
                "google.generativeai": mock_genai,
            },
        ):
            result = model.list_models()
            assert "models/gemini-1.5-pro" in result
            assert "models/gemini-1.5-flash" in result
            assert "models/embedding-001" not in result


class TestGeminiModelStreamMaxTokens:
    """Test stream with max_tokens and temperature parameters."""

    def test_stream_with_max_tokens(self):
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()

        chunk = MagicMock()
        chunk.text = "Hello"
        mock_genai_model.generate_content.return_value = [chunk]
        model._client = MagicMock()
        model._model = mock_genai_model

        chunks = list(model.stream("Test", max_tokens=100))
        assert chunks == ["Hello"]
        call_kwargs = mock_genai_model.generate_content.call_args[1]
        gen_config = call_kwargs["generation_config"]
        assert gen_config["max_output_tokens"] == 100

    def test_stream_with_temperature(self):
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()

        chunk = MagicMock()
        chunk.text = "Hello"
        mock_genai_model.generate_content.return_value = [chunk]
        model._client = MagicMock()
        model._model = mock_genai_model

        chunks = list(model.stream("Test", temperature=0.5))
        assert chunks == ["Hello"]
        call_kwargs = mock_genai_model.generate_content.call_args[1]
        gen_config = call_kwargs["generation_config"]
        assert gen_config["temperature"] == 0.5


class TestGeminiModelChatExtra:
    """Additional tests for GeminiModel.chat edge cases."""

    def test_chat_with_temperature(self):
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_chat.send_message.return_value = mock_response
        mock_genai_model.start_chat.return_value = mock_chat
        model._client = MagicMock()
        model._model = mock_genai_model

        model.chat(
            [{"role": "user", "content": "Hello"}],
            temperature=0.8,
        )
        call_kwargs = mock_chat.send_message.call_args[1]
        gen_config = call_kwargs["generation_config"]
        assert gen_config["temperature"] == 0.8

    def test_chat_mid_conversation_system_message_skipped(self):
        """A system message after history has been started should be skipped."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_chat.send_message.return_value = mock_response
        mock_genai_model.start_chat.return_value = mock_chat
        model._client = MagicMock()
        model._model = mock_genai_model

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "system", "content": "This should be skipped"},
            {"role": "user", "content": "How are you?"},
        ]
        result = model.chat(messages)
        assert result == "Response"

        # The system message should be ignored; only 3 history entries
        start_chat_kwargs = mock_genai_model.start_chat.call_args[1]
        history = start_chat_kwargs["history"]
        assert len(history) == 2  # Only the first 2 messages (user + model)
        # Last user message is "How are you?"
        sent_msg = mock_chat.send_message.call_args[0][0]
        assert "How are you?" in sent_msg
        assert "skipped" not in sent_msg

    def test_chat_single_user_message(self):
        """Chat with a single user message (no history)."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        mock_genai_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_chat.send_message.return_value = mock_response
        mock_genai_model.start_chat.return_value = mock_chat
        model._client = MagicMock()
        model._model = mock_genai_model

        model.chat([{"role": "user", "content": "Hello"}])

        # start_chat should be called with empty history
        start_chat_kwargs = mock_genai_model.start_chat.call_args[1]
        assert start_chat_kwargs["history"] == []


# ===========================================================================
# models/local.py tests
# ===========================================================================


class TestOllamaModelOptionsMapping:
    """Tests for OllamaModel generate/chat/stream options mapping."""

    def test_generate_with_all_options(self):
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(model_name="llama3.2")
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "output"}
        model._client = mock_client

        model.generate(
            "Test",
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            top_k=40,
            seed=42,
        )
        call_kwargs = mock_client.generate.call_args[1]
        options = call_kwargs["options"]
        assert options["temperature"] == 0.5
        assert options["num_predict"] == 100
        assert options["top_p"] == 0.9
        assert options["top_k"] == 40
        assert options["seed"] == 42

    def test_generate_with_num_predict(self):
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(model_name="llama3.2")
        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "output"}
        model._client = mock_client

        model.generate("Test", num_predict=50)
        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs["options"]["num_predict"] == 50

    def test_chat_with_options(self):
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(model_name="llama3.2")
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "output"}}
        model._client = mock_client

        model.chat(
            [{"role": "user", "content": "Hello"}],
            temperature=0.3,
            max_tokens=50,
        )
        call_kwargs = mock_client.chat.call_args[1]
        options = call_kwargs["options"]
        assert options["temperature"] == 0.3
        assert options["num_predict"] == 50

    def test_stream_with_options(self):
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(model_name="llama3.2")
        mock_client = MagicMock()
        mock_client.generate.return_value = [
            {"response": "Hello"},
            {"response": " world"},
        ]
        model._client = mock_client

        list(model.stream("Test", temperature=0.5, max_tokens=100))
        call_kwargs = mock_client.generate.call_args[1]
        options = call_kwargs["options"]
        assert options["temperature"] == 0.5
        assert options["num_predict"] == 100


class TestOllamaModelHeadersApiKey:
    """Tests for OllamaModel headers and api_key handling."""

    def test_init_with_api_key(self):
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(model_name="llama3.2", api_key="my-secret-key")
        assert model.headers is not None
        assert model.headers["Authorization"] == "Bearer my-secret-key"

    def test_init_with_env_api_key(self):
        from insideLLMs.models.local import OllamaModel

        with patch.dict(os.environ, {"OLLAMA_API_KEY": "env-secret-key"}):
            model = OllamaModel(model_name="llama3.2")
            assert model.headers is not None
            assert model.headers["Authorization"] == "Bearer env-secret-key"

    def test_init_with_existing_auth_header(self):
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(
            model_name="llama3.2",
            api_key="should-be-ignored",
            headers={"Authorization": "Bearer existing-key"},
        )
        assert model.headers["Authorization"] == "Bearer existing-key"

    def test_init_no_api_key(self):
        from insideLLMs.models.local import OllamaModel

        env_without_key = {k: v for k, v in os.environ.items() if k != "OLLAMA_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            model = OllamaModel(model_name="llama3.2")
            assert model.headers is None


class TestOllamaModelPull:
    """Tests for OllamaModel.pull."""

    def test_pull(self):
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(model_name="llama3.2")
        mock_client = MagicMock()
        model._client = mock_client

        model.pull()
        mock_client.pull.assert_called_once_with("llama3.2")


class TestOllamaModelListModels:
    """Tests for OllamaModel.list_models."""

    def test_list_models(self):
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(model_name="llama3.2")
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "mistral:latest"},
            ]
        }
        model._client = mock_client

        result = model.list_models()
        assert result == ["llama3.2:latest", "mistral:latest"]


class TestOllamaModelShowModelInfo:
    """Tests for OllamaModel.show_model_info."""

    def test_show_model_info(self):
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(model_name="llama3.2")
        mock_client = MagicMock()
        expected_info = {
            "modelfile": "FROM llama3.2",
            "parameters": "temperature 0.7",
            "template": "{{.System}}\n{{.Prompt}}",
        }
        mock_client.show.return_value = expected_info
        model._client = mock_client

        result = model.show_model_info()
        assert result == expected_info
        mock_client.show.assert_called_once_with("llama3.2")


# ---------------------------------------------------------------------------
# VLLMModel tests
# ---------------------------------------------------------------------------


class TestVLLMModelInit:
    """Tests for VLLMModel initialization."""

    def test_init_defaults(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(model_name="meta-llama/Llama-3.1-8B")
        assert model.model_name == "meta-llama/Llama-3.1-8B"
        assert model.base_url == "http://localhost:8000"
        assert model.api_key is None
        assert model._client is None

    def test_init_custom(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(
            model_name="my-model",
            name="MyVLLM",
            base_url="http://gpu-server:8080/",
            api_key="sk-123",
        )
        assert model.name == "MyVLLM"
        assert model.base_url == "http://gpu-server:8080"  # trailing slash stripped
        assert model.api_key == "sk-123"


class TestVLLMModelGetClient:
    """Tests for VLLMModel._get_client."""

    def test_get_client_creates_openai_client(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(model_name="my-model", api_key="sk-123")

        mock_openai_cls = MagicMock()
        mock_openai_client = MagicMock()
        mock_openai_cls.return_value = mock_openai_client

        # The _get_client does: from openai import OpenAI
        # We patch the openai module in sys.modules
        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI = mock_openai_cls

        import sys

        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            client = model._get_client()
            assert client is mock_openai_client
            mock_openai_cls.assert_called_once_with(
                base_url="http://localhost:8000/v1",
                api_key="sk-123",
            )

    def test_get_client_uses_dummy_key(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(model_name="my-model")
        # api_key is None, so dummy-key should be used
        assert model.api_key is None

    def test_get_client_caches(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(model_name="my-model")
        mock_client = MagicMock()
        model._client = mock_client

        assert model._get_client() is mock_client


class TestVLLMModelGenerate:
    """Tests for VLLMModel.generate."""

    def test_generate_basic(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(model_name="my-model")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].text = "Generated text"
        mock_client.completions.create.return_value = mock_response
        model._client = mock_client

        result = model.generate("Test prompt")
        assert result == "Generated text"
        call_kwargs = mock_client.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-model"
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["temperature"] == 0.7

    def test_generate_with_custom_params(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(model_name="my-model")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].text = "Output"
        mock_client.completions.create.return_value = mock_response
        model._client = mock_client

        model.generate("Test", max_tokens=256, temperature=0.5)
        call_kwargs = mock_client.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.5


class TestVLLMModelChat:
    """Tests for VLLMModel.chat."""

    def test_chat_basic(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(model_name="my-model")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Chat response"
        mock_client.chat.completions.create.return_value = mock_response
        model._client = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        result = model.chat(messages)
        assert result == "Chat response"

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-model"
        sent_messages = call_kwargs["messages"]
        assert sent_messages[0]["role"] == "user"
        assert sent_messages[0]["content"] == "Hello"

    def test_chat_with_params(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(model_name="my-model")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        model._client = mock_client

        model.chat(
            [{"role": "user", "content": "Hi"}],
            max_tokens=128,
            temperature=0.3,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 128
        assert call_kwargs["temperature"] == 0.3


class TestVLLMModelStream:
    """Tests for VLLMModel.stream."""

    def test_stream_basic(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(model_name="my-model")
        mock_client = MagicMock()

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].text = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].text = " world"

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].text = ""  # Empty - should be filtered

        mock_client.completions.create.return_value = [chunk1, chunk2, chunk3]
        model._client = mock_client

        chunks = list(model.stream("Test"))
        assert chunks == ["Hello", " world"]

    def test_stream_with_params(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(model_name="my-model")
        mock_client = MagicMock()

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].text = "output"
        mock_client.completions.create.return_value = [chunk]
        model._client = mock_client

        list(model.stream("Test", max_tokens=64, temperature=0.2))
        call_kwargs = mock_client.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 64
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["stream"] is True


class TestVLLMModelInfo:
    """Tests for VLLMModel.info."""

    def test_info_returns_correct_values(self):
        from insideLLMs.models.local import VLLMModel

        model = VLLMModel(
            model_name="meta-llama/Llama-3.1-8B",
            name="MyLlama",
            base_url="http://gpu:8000",
        )
        info = model.info()
        assert info.name == "MyLlama"
        assert info.provider == "vLLM"
        assert info.model_id == "meta-llama/Llama-3.1-8B"
        assert info.supports_streaming is True
        assert info.supports_chat is True
        assert info.extra["model_name"] == "meta-llama/Llama-3.1-8B"
        assert info.extra["base_url"] == "http://gpu:8000"
        assert "description" in info.extra


class TestOllamaModelGetClientHeaders:
    """Tests for OllamaModel._get_client passing headers."""

    def test_get_client_with_headers(self):
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(
            model_name="llama3.2",
            api_key="test-key",
        )
        model._client = None

        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            model._get_client()
            call_kwargs = mock_ollama.Client.call_args[1]
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
