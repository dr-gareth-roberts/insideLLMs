"""Tests for prompt optimization and tuning utilities."""

import pytest

from insideLLMs.optimization import (
    AblationResult,
    CompressionResult,
    ExampleScore,
    ExampleSelectionResult,
    FewShotSelector,
    InstructionOptimizer,
    OptimizationReport,
    OptimizationStrategy,
    PromptAblator,
    PromptCompressor,
    PromptOptimizer,
    TokenBudgetOptimizer,
    ablate_prompt,
    compress_prompt,
    optimize_for_budget,
    optimize_instruction,
    optimize_prompt,
    select_examples,
)


class TestOptimizationStrategy:
    """Tests for OptimizationStrategy enum."""

    def test_all_strategies_exist(self):
        """Test that all strategies are defined."""
        assert OptimizationStrategy.COMPRESSION.value == "compression"
        assert OptimizationStrategy.CLARITY.value == "clarity"
        assert OptimizationStrategy.SPECIFICITY.value == "specificity"
        assert OptimizationStrategy.STRUCTURE.value == "structure"
        assert OptimizationStrategy.EXAMPLE_SELECTION.value == "example_selection"


class TestCompressionResult:
    """Tests for CompressionResult."""

    def test_tokens_saved(self):
        """Test tokens saved calculation."""
        result = CompressionResult(
            original="test",
            compressed="t",
            original_tokens=100,
            compressed_tokens=80,
            compression_ratio=0.2,
        )

        assert result.tokens_saved == 20

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = CompressionResult(
            original="test",
            compressed="t",
            original_tokens=100,
            compressed_tokens=75,
            compression_ratio=0.25,
            removed_elements=["filler"],
            preserved_elements=["keyword"],
        )

        d = result.to_dict()
        assert d["original_tokens"] == 100
        assert d["compression_ratio"] == 0.25
        assert d["tokens_saved"] == 25


class TestPromptCompressor:
    """Tests for PromptCompressor."""

    def test_remove_filler_phrases(self):
        """Test removal of filler phrases."""
        compressor = PromptCompressor()
        prompt = "Please note that basically the answer is essentially correct."

        result = compressor.compress(prompt)

        assert result.compressed_tokens < result.original_tokens
        assert "please note that" not in result.compressed.lower()
        assert "basically" not in result.compressed.lower()

    def test_simplify_verbose_patterns(self):
        """Test simplification of verbose patterns."""
        compressor = PromptCompressor()
        prompt = "In order to accomplish this task, due to the fact that we need results."

        result = compressor.compress(prompt)

        # Should simplify "in order to" -> "to" and "due to the fact that" -> "because"
        assert "in order to" not in result.compressed.lower()
        assert result.compression_ratio > 0

    def test_preserve_keywords(self):
        """Test preservation of keywords."""
        compressor = PromptCompressor()
        prompt = "Please note that the API key is important."

        result = compressor.compress(prompt, preserve_keywords={"API", "key"})

        assert "api" in result.compressed.lower() or "API" in result.compressed
        assert "key" in result.compressed.lower()

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        compressor = PromptCompressor()
        prompt = "Basically, essentially, in order to achieve this goal, please note that we need to consider."

        result = compressor.compress(prompt)

        assert 0 <= result.compression_ratio <= 1
        assert result.compressed_tokens <= result.original_tokens

    def test_empty_prompt(self):
        """Test with empty prompt."""
        compressor = PromptCompressor()
        result = compressor.compress("")

        assert result.compressed == ""
        assert result.compression_ratio == 0


class TestInstructionOptimizer:
    """Tests for InstructionOptimizer."""

    def test_strengthen_weak_verbs(self):
        """Test strengthening of weak verbs."""
        optimizer = InstructionOptimizer()
        instruction = "Try to analyze the data and consider the results."

        optimized, changes = optimizer.optimize(instruction)

        assert "try to" not in optimized.lower()
        assert len(changes) > 0

    def test_flag_ambiguous_terms(self):
        """Test flagging of ambiguous terms."""
        optimizer = InstructionOptimizer()
        instruction = "Provide a good analysis with appropriate detail."

        optimized, changes = optimizer.optimize(instruction)

        # Should flag 'good' and 'appropriate'
        ambiguous_flags = [c for c in changes if "ambiguous" in c.lower()]
        assert len(ambiguous_flags) > 0

    def test_add_punctuation(self):
        """Test adding ending punctuation."""
        optimizer = InstructionOptimizer()
        instruction = "Write a summary"

        optimized, changes = optimizer.optimize(instruction)

        assert optimized.endswith(".")

    def test_analyze_clarity(self):
        """Test clarity analysis."""
        optimizer = InstructionOptimizer()

        # Clear instruction
        clear = "Analyze the dataset and provide statistical summary."
        clear_result = optimizer.analyze_clarity(clear)

        assert clear_result["score"] > 0.5
        assert clear_result["has_action_verb"] is True

        # Unclear instruction
        unclear = "Try to maybe consider thinking about stuff."
        unclear_result = optimizer.analyze_clarity(unclear)

        assert unclear_result["score"] < clear_result["score"]
        assert len(unclear_result["issues"]) > 0


class TestFewShotSelector:
    """Tests for FewShotSelector."""

    def test_select_relevant_examples(self):
        """Test selection of relevant examples."""
        selector = FewShotSelector()
        query = "What is the capital of France?"
        examples = [
            {"input": "What is the capital of Germany?", "output": "Berlin"},
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What is the capital of Spain?", "output": "Madrid"},
            {"input": "Define photosynthesis.", "output": "Process by plants..."},
        ]

        result = selector.select(query, examples, n=2)

        assert len(result.selected_examples) == 2
        # Geography questions should be selected over math
        selected_inputs = [e["input"] for e in result.selected_examples]
        assert any("capital" in inp for inp in selected_inputs)

    def test_diversity_in_selection(self):
        """Test diversity consideration in selection."""
        selector = FewShotSelector(diversity_weight=0.5)
        query = "Explain machine learning."
        examples = [
            {"input": "Explain machine learning basics.", "output": "ML is..."},
            {"input": "What is machine learning?", "output": "Machine learning..."},
            {"input": "Describe neural networks.", "output": "Neural networks..."},
            {"input": "What is AI?", "output": "Artificial intelligence..."},
        ]

        result = selector.select(query, examples, n=3)

        assert result.diversity_score > 0
        assert len(result.selected_examples) == 3

    def test_empty_examples(self):
        """Test with no examples."""
        selector = FewShotSelector()
        result = selector.select("query", [], n=3)

        assert result.selected_examples == []
        assert result.coverage_score == 0.0

    def test_fewer_examples_than_requested(self):
        """Test when fewer examples available than requested."""
        selector = FewShotSelector()
        examples = [
            {"input": "test", "output": "result"},
        ]

        result = selector.select("query", examples, n=5)

        assert len(result.selected_examples) == 1

    def test_example_scores(self):
        """Test that example scores are calculated."""
        selector = FewShotSelector()
        examples = [
            {"input": "What is X?", "output": "X is..."},
            {"input": "Explain Y.", "output": "Y means..."},
        ]

        result = selector.select("What is X?", examples, n=2)

        assert len(result.example_scores) == 2
        for score in result.example_scores:
            assert 0 <= score.relevance_score <= 1
            assert 0 <= score.quality_score <= 1
            assert 0 <= score.overall_score <= 1


class TestExampleSelectionResult:
    """Tests for ExampleSelectionResult."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ExampleSelectionResult(
            query="test",
            selected_examples=[{"input": "a", "output": "b"}],
            example_scores=[
                ExampleScore(
                    example={"input": "a", "output": "b"},
                    relevance_score=0.8,
                    diversity_score=1.0,
                    quality_score=0.7,
                    overall_score=0.83,
                )
            ],
            coverage_score=0.9,
            diversity_score=1.0,
        )

        d = result.to_dict()
        assert d["num_selected"] == 1
        assert d["coverage_score"] == 0.9


class TestPromptAblator:
    """Tests for PromptAblator."""

    def test_ablate_multi_component(self):
        """Test ablation with multiple components."""
        ablator = PromptAblator()
        prompt = """You are a helpful assistant.

Please analyze the following text carefully.

Provide a detailed response with examples."""

        result = ablator.ablate(prompt)

        assert len(result.components) == 3
        assert len(result.component_scores) == 3
        assert len(result.importance_ranking) == 3

    def test_ablate_single_component(self):
        """Test ablation with single component."""
        ablator = PromptAblator()
        prompt = "Single line prompt."

        result = ablator.ablate(prompt)

        assert len(result.components) == 1
        assert result.minimal_prompt == prompt

    def test_custom_scorer(self):
        """Test with custom scorer."""
        def custom_scorer(prompt):
            return len(prompt) / 100  # Score based on length

        ablator = PromptAblator(scorer=custom_scorer)
        prompt = """Part one.

Part two which is longer.

Part three."""

        result = ablator.ablate(prompt)

        assert len(result.component_scores) == 3

    def test_identify_essential_components(self):
        """Test identification of essential vs removable components."""
        ablator = PromptAblator()
        prompt = """Critical instruction: Do X.

Optional context that could be removed.

Another important requirement."""

        result = ablator.ablate(prompt)

        # Should have some ranking
        assert len(result.importance_ranking) > 0


class TestAblationResult:
    """Tests for AblationResult."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = AblationResult(
            original_prompt="test",
            components=["a", "b"],
            component_scores={"a...": 0.5, "b...": 0.3},
            essential_components=["a..."],
            removable_components=["b..."],
            minimal_prompt="a",
            importance_ranking=[("a...", 0.5), ("b...", 0.3)],
        )

        d = result.to_dict()
        assert d["num_components"] == 2
        assert "essential_components" in d


class TestTokenBudgetOptimizer:
    """Tests for TokenBudgetOptimizer."""

    def test_within_budget(self):
        """Test prompt within budget."""
        optimizer = TokenBudgetOptimizer(max_tokens=1000)
        prompt = "Short prompt."

        result = optimizer.optimize(prompt, reserve_for_response=100)

        assert result["over_budget"] is False
        assert result["final_prompt"] == prompt

    def test_compress_when_over_budget(self):
        """Test compression when over budget."""
        optimizer = TokenBudgetOptimizer(max_tokens=100)
        prompt = "Basically, essentially, in order to do this very important task, please note that " * 5

        result = optimizer.optimize(prompt, reserve_for_response=20)

        assert len(result["actions_taken"]) > 0
        assert result["final_tokens"] <= result["available_tokens"]

    def test_reduce_examples_when_needed(self):
        """Test example reduction when over budget."""
        optimizer = TokenBudgetOptimizer(max_tokens=200)
        prompt = "Main prompt."
        examples = [
            {"input": "example " * 20, "output": "result " * 20}
            for _ in range(5)
        ]

        result = optimizer.optimize(
            prompt, examples=examples, reserve_for_response=50
        )

        assert len(result["final_examples"]) < len(examples)


class TestPromptOptimizer:
    """Tests for PromptOptimizer."""

    def test_apply_compression_strategy(self):
        """Test compression strategy."""
        optimizer = PromptOptimizer()
        prompt = "Basically, in order to achieve this, please note that we need to do things."

        report = optimizer.optimize(
            prompt, strategies=[OptimizationStrategy.COMPRESSION]
        )

        assert len(report.optimized_prompt) <= len(prompt)
        if OptimizationStrategy.COMPRESSION in report.strategies_applied:
            assert "compression" in report.improvements

    def test_apply_clarity_strategy(self):
        """Test clarity strategy."""
        optimizer = PromptOptimizer()
        prompt = "Try to maybe consider analyzing the good data."

        report = optimizer.optimize(
            prompt, strategies=[OptimizationStrategy.CLARITY]
        )

        assert len(report.suggestions) > 0

    def test_apply_multiple_strategies(self):
        """Test applying multiple strategies."""
        optimizer = PromptOptimizer()
        prompt = "Basically, try to consider the appropriate approach in order to achieve good results."

        report = optimizer.optimize(prompt)

        assert len(report.strategies_applied) > 0

    def test_token_reduction_tracking(self):
        """Test token reduction tracking."""
        optimizer = PromptOptimizer()
        prompt = "Basically essentially in order to " * 10

        report = optimizer.optimize(prompt)

        # May or may not reduce tokens depending on strategies
        assert isinstance(report.token_reduction, int)


class TestOptimizationReport:
    """Tests for OptimizationReport."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = OptimizationReport(
            original_prompt="original",
            optimized_prompt="optimized",
            strategies_applied=[OptimizationStrategy.COMPRESSION],
            improvements={"compression": 0.2},
            suggestions=["Consider X"],
            token_reduction=10,
            estimated_quality_change=0.1,
        )

        d = report.to_dict()
        assert d["strategies_applied"] == ["compression"]
        assert d["token_reduction"] == 10


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_compress_prompt(self):
        """Test compress_prompt function."""
        result = compress_prompt("Basically, the answer is correct.")

        assert isinstance(result, CompressionResult)
        assert result.compression_ratio >= 0

    def test_optimize_instruction(self):
        """Test optimize_instruction function."""
        optimized, changes = optimize_instruction("Try to analyze this.")

        assert isinstance(optimized, str)
        assert isinstance(changes, list)

    def test_select_examples(self):
        """Test select_examples function."""
        examples = [
            {"input": "test1", "output": "result1"},
            {"input": "test2", "output": "result2"},
        ]

        result = select_examples("query", examples, n=1)

        assert isinstance(result, ExampleSelectionResult)
        assert len(result.selected_examples) == 1

    def test_ablate_prompt(self):
        """Test ablate_prompt function."""
        prompt = "Part one.\n\nPart two."

        result = ablate_prompt(prompt)

        assert isinstance(result, AblationResult)

    def test_optimize_prompt(self):
        """Test optimize_prompt function."""
        result = optimize_prompt("Test prompt to optimize.")

        assert isinstance(result, OptimizationReport)

    def test_optimize_for_budget(self):
        """Test optimize_for_budget function."""
        result = optimize_for_budget(
            "Test prompt",
            max_tokens=1000,
            reserve_for_response=100,
        )

        assert "final_prompt" in result
        assert "over_budget" in result


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_prompt_compression(self):
        """Test compression of empty prompt."""
        result = compress_prompt("")

        assert result.compressed == ""

    def test_whitespace_only_prompt(self):
        """Test with whitespace-only prompt."""
        compressor = PromptCompressor()
        result = compressor.compress("   \n\n   ")

        assert result.compressed.strip() == ""

    def test_unicode_in_prompt(self):
        """Test with unicode characters."""
        compressor = PromptCompressor()
        result = compressor.compress("Basically, this is important: 日本語テスト 🎉")

        assert "日本語テスト" in result.compressed
        assert "🎉" in result.compressed

    def test_very_long_prompt_compression(self):
        """Test compression of very long prompt."""
        compressor = PromptCompressor()
        long_prompt = ("In order to achieve this goal, basically, " * 50)

        result = compressor.compress(long_prompt)

        assert result.compression_ratio > 0
        assert result.compressed_tokens < result.original_tokens

    def test_no_filler_prompt(self):
        """Test prompt with no fillers."""
        compressor = PromptCompressor()
        clean_prompt = "Analyze the data. Provide results."

        result = compressor.compress(clean_prompt)

        # Should not change much
        assert result.compression_ratio < 0.1

    def test_custom_delimiter_ablation(self):
        """Test ablation with custom delimiter."""
        prompt = "Part A---Part B---Part C"
        result = ablate_prompt(prompt, component_delimiter="---")

        assert len(result.components) == 3
