"""Tests for prompt sensitivity analysis module."""

from insideLLMs.sensitivity import (
    ComparativeSensitivity,
    ComparativeSensitivityAnalyzer,
    FormatSensitivityTester,
    OutputChangeType,
    OutputComparator,
    OutputComparison,
    # Dataclasses
    Perturbation,
    # Enums
    PerturbationType,
    # Classes
    PromptPerturbator,
    SensitivityAnalyzer,
    SensitivityLevel,
    SensitivityProfile,
    SensitivityResult,
    # Functions
    analyze_prompt_sensitivity,
    check_format_sensitivity,
    compare_prompt_sensitivity,
    generate_perturbations,
    quick_sensitivity_check,
)

# ============================================================================
# Enum Tests
# ============================================================================


class TestPerturbationType:
    """Tests for PerturbationType enum."""

    def test_all_types_defined(self):
        """Test all perturbation types are defined."""
        assert PerturbationType.CASE_CHANGE.value == "case_change"
        assert PerturbationType.WHITESPACE.value == "whitespace"
        assert PerturbationType.PUNCTUATION.value == "punctuation"
        assert PerturbationType.SYNONYM.value == "synonym"
        assert PerturbationType.PARAPHRASE.value == "paraphrase"
        assert PerturbationType.WORD_ORDER.value == "word_order"
        assert PerturbationType.TYPO.value == "typo"
        assert PerturbationType.FORMATTING.value == "formatting"
        assert PerturbationType.INSTRUCTION_STYLE.value == "instruction_style"

    def test_type_count(self):
        """Test correct number of types."""
        assert len(PerturbationType) == 9


class TestSensitivityLevel:
    """Tests for SensitivityLevel enum."""

    def test_all_levels_defined(self):
        """Test all sensitivity levels are defined."""
        assert SensitivityLevel.VERY_LOW.value == "very_low"
        assert SensitivityLevel.LOW.value == "low"
        assert SensitivityLevel.MODERATE.value == "moderate"
        assert SensitivityLevel.HIGH.value == "high"
        assert SensitivityLevel.VERY_HIGH.value == "very_high"


class TestOutputChangeType:
    """Tests for OutputChangeType enum."""

    def test_all_types_defined(self):
        """Test all change types are defined."""
        assert OutputChangeType.NO_CHANGE.value == "no_change"
        assert OutputChangeType.MINOR_VARIATION.value == "minor_variation"
        assert OutputChangeType.SEMANTIC_EQUIVALENT.value == "semantic_equivalent"
        assert OutputChangeType.DIFFERENT_CONTENT.value == "different_content"


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestPerturbation:
    """Tests for Perturbation dataclass."""

    def test_creation(self):
        """Test creating a perturbation."""
        perturbation = Perturbation(
            original="Explain this",
            perturbed="explain this",
            perturbation_type=PerturbationType.CASE_CHANGE,
            change_description="Changed letter casing",
            change_magnitude=0.1,
        )
        assert perturbation.original == "Explain this"
        assert perturbation.perturbed == "explain this"
        assert perturbation.perturbation_type == PerturbationType.CASE_CHANGE

    def test_to_dict(self):
        """Test dictionary conversion."""
        perturbation = Perturbation(
            original="Test",
            perturbed="test",
            perturbation_type=PerturbationType.CASE_CHANGE,
            change_description="Changed case",
            change_magnitude=0.05,
        )
        d = perturbation.to_dict()
        assert d["type"] == "case_change"
        assert d["magnitude"] == 0.05


class TestOutputComparison:
    """Tests for OutputComparison dataclass."""

    def test_creation(self):
        """Test creating an output comparison."""
        comparison = OutputComparison(
            original_output="The answer is 42.",
            perturbed_output="The answer is 42.",
            change_type=OutputChangeType.NO_CHANGE,
            similarity_score=1.0,
            semantic_similarity=1.0,
            length_ratio=1.0,
            key_differences=[],
        )
        assert comparison.similarity_score == 1.0
        assert comparison.change_type == OutputChangeType.NO_CHANGE

    def test_to_dict_truncates(self):
        """Test dictionary truncates long outputs."""
        long_output = "x" * 500
        comparison = OutputComparison(
            original_output=long_output,
            perturbed_output=long_output,
            change_type=OutputChangeType.NO_CHANGE,
            similarity_score=1.0,
            semantic_similarity=1.0,
            length_ratio=1.0,
            key_differences=[],
        )
        d = comparison.to_dict()
        assert len(d["original_output"]) == 200


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""

    def test_creation(self):
        """Test creating a sensitivity result."""
        perturbation = Perturbation(
            original="Test",
            perturbed="test",
            perturbation_type=PerturbationType.CASE_CHANGE,
            change_description="Changed case",
            change_magnitude=0.05,
        )
        comparison = OutputComparison(
            original_output="Response A",
            perturbed_output="Response A",
            change_type=OutputChangeType.NO_CHANGE,
            similarity_score=1.0,
            semantic_similarity=1.0,
            length_ratio=1.0,
            key_differences=[],
        )
        result = SensitivityResult(
            perturbation=perturbation,
            output_comparison=comparison,
            sensitivity_score=0.0,
            is_robust=True,
        )
        assert result.is_robust is True
        assert result.sensitivity_score == 0.0


class TestSensitivityProfile:
    """Tests for SensitivityProfile dataclass."""

    def test_creation(self):
        """Test creating a sensitivity profile."""
        profile = SensitivityProfile(
            prompt="Test prompt",
            results=[],
            overall_sensitivity=SensitivityLevel.LOW,
            overall_score=0.15,
            by_perturbation_type={PerturbationType.CASE_CHANGE: 0.1},
            most_sensitive_to=[],
            most_robust_to=[PerturbationType.CASE_CHANGE],
            recommendations=["Prompt appears robust"],
        )
        assert profile.overall_score == 0.15
        assert profile.overall_sensitivity == SensitivityLevel.LOW

    def test_to_dict(self):
        """Test dictionary conversion."""
        profile = SensitivityProfile(
            prompt="Test",
            results=[],
            overall_sensitivity=SensitivityLevel.MODERATE,
            overall_score=0.4,
            by_perturbation_type={},
            most_sensitive_to=[PerturbationType.TYPO],
            most_robust_to=[],
            recommendations=[],
        )
        d = profile.to_dict()
        assert d["overall_sensitivity"] == "moderate"
        assert d["most_sensitive_to"] == ["typo"]


# ============================================================================
# PromptPerturbator Tests
# ============================================================================


class TestPromptPerturbator:
    """Tests for PromptPerturbator class."""

    def test_case_perturbation(self):
        """Test case change perturbation."""
        perturbator = PromptPerturbator(seed=42)
        perturbations = perturbator.perturb(
            "Explain this concept",
            [PerturbationType.CASE_CHANGE],
            n_variations=1,
        )
        assert len(perturbations) >= 1
        assert perturbations[0].perturbation_type == PerturbationType.CASE_CHANGE
        assert perturbations[0].perturbed != perturbations[0].original

    def test_whitespace_perturbation(self):
        """Test whitespace perturbation."""
        perturbator = PromptPerturbator(seed=42)
        perturbations = perturbator.perturb(
            "Test prompt",
            [PerturbationType.WHITESPACE],
            n_variations=1,
        )
        assert len(perturbations) >= 1

    def test_punctuation_perturbation(self):
        """Test punctuation perturbation."""
        perturbator = PromptPerturbator(seed=42)
        perturbations = perturbator.perturb(
            "What is the answer?",
            [PerturbationType.PUNCTUATION],
            n_variations=1,
        )
        assert len(perturbations) >= 1

    def test_synonym_perturbation(self):
        """Test synonym replacement."""
        perturbator = PromptPerturbator(seed=42)
        perturbations = perturbator.perturb(
            "Explain the concept",
            [PerturbationType.SYNONYM],
            n_variations=1,
        )
        # Should find "explain" -> synonym
        assert len(perturbations) >= 1
        if perturbations:
            assert (
                "explain" not in perturbations[0].perturbed.lower()
                or perturbations[0].perturbed != perturbations[0].original
            )

    def test_typo_perturbation(self):
        """Test typo introduction."""
        perturbator = PromptPerturbator(seed=42)
        perturbations = perturbator.perturb(
            "Calculate the result",
            [PerturbationType.TYPO],
            n_variations=1,
        )
        assert len(perturbations) >= 1
        # Perturbed should differ by small amount
        assert perturbations[0].perturbed != perturbations[0].original

    def test_formatting_perturbation(self):
        """Test formatting changes."""
        perturbator = PromptPerturbator(seed=42)
        perturbations = perturbator.perturb(
            "List the items",
            [PerturbationType.FORMATTING],
            n_variations=1,
        )
        assert len(perturbations) >= 1

    def test_instruction_style_perturbation(self):
        """Test instruction style changes."""
        perturbator = PromptPerturbator(seed=42)
        perturbations = perturbator.perturb(
            "Explain the topic",
            [PerturbationType.INSTRUCTION_STYLE],
            n_variations=1,
        )
        assert len(perturbations) >= 1

    def test_multiple_types(self):
        """Test multiple perturbation types."""
        perturbator = PromptPerturbator(seed=42)
        perturbations = perturbator.perturb(
            "Explain this in detail",
            [PerturbationType.CASE_CHANGE, PerturbationType.TYPO],
            n_variations=2,
        )
        # Should have multiple perturbations
        assert len(perturbations) >= 2

    def test_all_types(self):
        """Test generating all perturbation types."""
        perturbator = PromptPerturbator(seed=42)
        perturbations = perturbator.perturb(
            "Explain the important concept clearly",
            n_variations=1,
        )
        # Should generate at least some perturbations
        assert len(perturbations) >= 5

    def test_reproducibility_with_seed(self):
        """Test that seed provides reproducibility."""
        p1 = PromptPerturbator(seed=123)
        p2 = PromptPerturbator(seed=123)

        perturb1 = p1.perturb("Test prompt", [PerturbationType.CASE_CHANGE], 1)
        perturb2 = p2.perturb("Test prompt", [PerturbationType.CASE_CHANGE], 1)

        assert perturb1[0].perturbed == perturb2[0].perturbed


# ============================================================================
# OutputComparator Tests
# ============================================================================


class TestOutputComparator:
    """Tests for OutputComparator class."""

    def test_identical_outputs(self):
        """Test comparing identical outputs."""
        comparator = OutputComparator()
        comparison = comparator.compare(
            "The answer is 42",
            "The answer is 42",
        )
        assert comparison.similarity_score == 1.0
        assert comparison.change_type == OutputChangeType.NO_CHANGE

    def test_similar_outputs(self):
        """Test comparing similar outputs."""
        comparator = OutputComparator()
        comparison = comparator.compare(
            "The answer is forty-two",
            "The answer is 42",
        )
        assert comparison.similarity_score > 0.5
        assert comparison.change_type in [
            OutputChangeType.MINOR_VARIATION,
            OutputChangeType.SEMANTIC_EQUIVALENT,
            OutputChangeType.DIFFERENT_CONTENT,
        ]

    def test_different_outputs(self):
        """Test comparing different outputs."""
        comparator = OutputComparator()
        comparison = comparator.compare(
            "The sky is blue because of light scattering",
            "Python is a programming language",
        )
        assert comparison.similarity_score < 0.5
        assert comparison.change_type != OutputChangeType.NO_CHANGE

    def test_empty_outputs(self):
        """Test comparing empty outputs."""
        comparator = OutputComparator()
        comparison = comparator.compare("", "")
        assert comparison.similarity_score == 1.0

    def test_length_ratio(self):
        """Test length ratio calculation."""
        comparator = OutputComparator()
        comparison = comparator.compare(
            "Short",
            "This is a much longer response with many more words",
        )
        assert comparison.length_ratio < 0.5

    def test_key_differences(self):
        """Test finding key differences."""
        comparator = OutputComparator()
        comparison = comparator.compare(
            "The cat sat on the mat",
            "The dog ran in the park",
        )
        assert len(comparison.key_differences) > 0

    def test_custom_similarity_function(self):
        """Test with custom similarity function."""

        def always_half(t1: str, t2: str) -> float:
            return 0.5

        comparator = OutputComparator(similarity_fn=always_half)
        comparison = comparator.compare("anything", "anything else")
        assert comparison.similarity_score == 0.5


# ============================================================================
# SensitivityAnalyzer Tests
# ============================================================================


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer class."""

    def test_analyze_robust_prompt(self):
        """Test analyzing a robust prompt."""

        # Mock response that ignores case changes
        def get_response(prompt: str) -> str:
            return "The answer is always the same"

        analyzer = SensitivityAnalyzer()
        profile = analyzer.analyze(
            "What is the answer?",
            get_response,
            [PerturbationType.CASE_CHANGE],
            n_variations=2,
        )

        assert profile.overall_score < 0.3
        assert profile.overall_sensitivity in [
            SensitivityLevel.VERY_LOW,
            SensitivityLevel.LOW,
        ]

    def test_analyze_sensitive_prompt(self):
        """Test analyzing a sensitive prompt."""
        # Mock response that changes dramatically based on any prompt change
        # Use completely different word sets to ensure low word-level Jaccard similarity
        responses = [
            "alpha beta gamma delta epsilon",
            "one two three four five six",
            "apple banana cherry date elderberry",
            "red green blue yellow purple orange",
        ]
        call_count = [0]  # Use list to allow mutation in closure

        def get_response(prompt: str) -> str:
            # Return a completely different response for each unique prompt
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            return responses[idx]

        analyzer = SensitivityAnalyzer()
        profile = analyzer.analyze(
            "Tell me something",
            get_response,
            [PerturbationType.CASE_CHANGE],
            n_variations=2,
        )

        # Since each call returns completely different words, sensitivity should be high
        assert profile.overall_score > 0.3

    def test_custom_robustness_threshold(self):
        """Test custom robustness threshold."""

        def get_response(prompt: str) -> str:
            return f"Response to: {prompt[:10]}"

        analyzer = SensitivityAnalyzer(robustness_threshold=0.9)
        profile = analyzer.analyze(
            "Test prompt",
            get_response,
            [PerturbationType.TYPO],
            n_variations=1,
        )

        # Results should reflect the custom threshold
        assert profile.results is not None

    def test_recommendations_generated(self):
        """Test that recommendations are generated."""

        def get_response(prompt: str) -> str:
            return "Standard response"

        analyzer = SensitivityAnalyzer()
        profile = analyzer.analyze(
            "Explain something",
            get_response,
            [PerturbationType.CASE_CHANGE],
            n_variations=1,
        )

        assert len(profile.recommendations) >= 1

    def test_by_perturbation_type_scores(self):
        """Test scores by perturbation type."""

        def get_response(prompt: str) -> str:
            return "Response"

        analyzer = SensitivityAnalyzer()
        profile = analyzer.analyze(
            "Test",
            get_response,
            [PerturbationType.CASE_CHANGE, PerturbationType.TYPO],
            n_variations=1,
        )

        assert len(profile.by_perturbation_type) >= 1


# ============================================================================
# ComparativeSensitivityAnalyzer Tests
# ============================================================================


class TestComparativeSensitivityAnalyzer:
    """Tests for ComparativeSensitivityAnalyzer class."""

    def test_compare_prompts(self):
        """Test comparing multiple prompts."""

        def get_response(prompt: str) -> str:
            return f"Response: {len(prompt)}"

        analyzer = ComparativeSensitivityAnalyzer()
        comparison = analyzer.compare_prompts(
            ["Short", "A longer prompt here"],
            get_response,
            [PerturbationType.CASE_CHANGE],
        )

        assert len(comparison.profiles) == 2
        assert comparison.most_robust is not None
        assert comparison.most_sensitive is not None

    def test_compare_models(self):
        """Test comparing multiple models."""

        def model_a(prompt: str) -> str:
            return "Model A always says this"

        def model_b(prompt: str) -> str:
            return f"Model B responds to: {prompt}"

        analyzer = ComparativeSensitivityAnalyzer()
        comparison = analyzer.compare_models(
            "Test prompt",
            {"model_a": model_a, "model_b": model_b},
            [PerturbationType.CASE_CHANGE],
        )

        assert len(comparison.profiles) == 2
        assert comparison.ranking is not None

    def test_ranking_order(self):
        """Test that ranking is ordered correctly."""

        def get_response(prompt: str) -> str:
            return "Same"

        analyzer = ComparativeSensitivityAnalyzer()
        comparison = analyzer.compare_prompts(
            ["Prompt one", "Prompt two"],
            get_response,
            [PerturbationType.CASE_CHANGE],
        )

        # Ranking should be sorted by sensitivity score
        scores = [score for _, score in comparison.ranking]
        assert scores == sorted(scores)


# ============================================================================
# FormatSensitivityTester Tests
# ============================================================================


class TestFormatSensitivityTester:
    """Tests for FormatSensitivityTester class."""

    def test_format_sensitivity(self):
        """Test format sensitivity testing."""

        def get_response(prompt: str) -> str:
            if "JSON" in prompt:
                return '{"key": "value"}'
            elif "Markdown" in prompt:
                return "# Heading\n**Bold**"
            elif "bullet" in prompt:
                return "- Item 1\n- Item 2"
            elif "numbered" in prompt:
                return "1. First\n2. Second"
            return "Plain text response"

        tester = FormatSensitivityTester()
        results = tester.test_format_sensitivity(
            "Explain something",
            get_response,
        )

        assert "baseline" in results
        assert "variations" in results
        assert "format_adherence_rate" in results

    def test_format_detection(self):
        """Test format detection logic."""

        def get_response(prompt: str) -> str:
            if "JSON" in prompt:
                return '{"result": 42}'
            return "text"

        tester = FormatSensitivityTester()
        results = tester.test_format_sensitivity("Test", get_response)

        # JSON format should be detected
        assert results["variations"]["json"]["format_followed"] is True


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_analyze_prompt_sensitivity(self):
        """Test analyze_prompt_sensitivity function."""

        def get_response(prompt: str) -> str:
            return "Response"

        profile = analyze_prompt_sensitivity(
            "Test prompt",
            get_response,
            [PerturbationType.CASE_CHANGE],
        )

        assert isinstance(profile, SensitivityProfile)

    def test_compare_prompt_sensitivity(self):
        """Test compare_prompt_sensitivity function."""

        def get_response(prompt: str) -> str:
            return "Response"

        comparison = compare_prompt_sensitivity(
            ["Prompt 1", "Prompt 2"],
            get_response,
        )

        assert isinstance(comparison, ComparativeSensitivity)

    def test_generate_perturbations(self):
        """Test generate_perturbations function."""
        perturbations = generate_perturbations(
            "Explain the concept",
            [PerturbationType.CASE_CHANGE, PerturbationType.TYPO],
            n_variations=2,
        )

        assert len(perturbations) >= 2
        assert all(isinstance(p, Perturbation) for p in perturbations)

    def test_quick_sensitivity_check(self):
        """Test quick_sensitivity_check function."""

        def get_response(prompt: str) -> str:
            return "Quick response"

        result = quick_sensitivity_check("Test", get_response)

        assert "overall_sensitivity" in result
        assert "overall_score" in result
        assert "is_robust" in result
        assert "recommendations" in result

    def test_check_format_sensitivity(self):
        """Test check_format_sensitivity function."""

        def get_response(prompt: str) -> str:
            return "Response"

        result = check_format_sensitivity("Base prompt", get_response)

        assert "baseline" in result
        assert "variations" in result


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_prompt(self):
        """Test with empty prompt."""
        perturbator = PromptPerturbator()
        perturbations = perturbator.perturb("", n_variations=1)
        # Should handle gracefully
        assert isinstance(perturbations, list)

    def test_very_short_prompt(self):
        """Test with very short prompt."""
        perturbator = PromptPerturbator()
        perturbations = perturbator.perturb("Hi", n_variations=1)
        assert isinstance(perturbations, list)

    def test_prompt_with_special_characters(self):
        """Test prompt with special characters."""
        perturbator = PromptPerturbator()
        perturbations = perturbator.perturb(
            "What is 2 + 2? @#$%",
            [PerturbationType.CASE_CHANGE],
            n_variations=1,
        )
        assert isinstance(perturbations, list)

    def test_unicode_prompt(self):
        """Test prompt with unicode characters."""
        perturbator = PromptPerturbator()
        perturbations = perturbator.perturb(
            "Explain 你好 and مرحبا",
            [PerturbationType.CASE_CHANGE],
            n_variations=1,
        )
        assert isinstance(perturbations, list)

    def test_multiline_prompt(self):
        """Test multiline prompt."""
        perturbator = PromptPerturbator()
        perturbations = perturbator.perturb(
            "Line 1\nLine 2\nLine 3",
            [PerturbationType.WHITESPACE],
            n_variations=1,
        )
        assert isinstance(perturbations, list)

    def test_no_applicable_perturbations(self):
        """Test when no perturbations can be applied."""
        perturbator = PromptPerturbator(seed=42)
        # Single character prompt
        perturbations = perturbator.perturb(
            "a",
            [PerturbationType.WORD_ORDER],  # Can't reorder single character
            n_variations=1,
        )
        # Should return empty or unchanged
        assert isinstance(perturbations, list)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for sensitivity module."""

    def test_full_sensitivity_workflow(self):
        """Test complete sensitivity analysis workflow."""
        # Setup mock model
        responses: dict[str, str] = {}

        def mock_model(prompt: str) -> str:
            if prompt not in responses:
                # Generate somewhat deterministic response
                responses[prompt] = f"Response to prompt of length {len(prompt)}"
            return responses[prompt]

        # Generate perturbations
        prompt = "Explain the concept of machine learning"
        generate_perturbations(
            prompt,
            [PerturbationType.CASE_CHANGE, PerturbationType.SYNONYM],
            n_variations=2,
        )

        # Analyze sensitivity
        profile = analyze_prompt_sensitivity(
            prompt,
            mock_model,
            [PerturbationType.CASE_CHANGE, PerturbationType.SYNONYM],
        )

        # Verify results
        assert profile.prompt == prompt
        assert len(profile.results) >= 1
        assert profile.overall_sensitivity is not None
        assert isinstance(profile.overall_score, float)
        assert 0 <= profile.overall_score <= 1

    def test_model_comparison_workflow(self):
        """Test comparing multiple models."""

        def consistent_model(prompt: str) -> str:
            return "Always the same response"

        def variable_model(prompt: str) -> str:
            return f"Variable: {hash(prompt) % 100}"

        analyzer = ComparativeSensitivityAnalyzer()
        comparison = analyzer.compare_models(
            "Test prompt for comparison",
            {
                "consistent": consistent_model,
                "variable": variable_model,
            },
            [PerturbationType.CASE_CHANGE],
        )

        # Consistent model should be more robust
        assert comparison.most_robust == "consistent"
        assert len(comparison.ranking) == 2

    def test_serialization_roundtrip(self):
        """Test that all results can be serialized."""

        def get_response(prompt: str) -> str:
            return "Test response"

        profile = analyze_prompt_sensitivity(
            "Test prompt",
            get_response,
            [PerturbationType.CASE_CHANGE],
        )

        # Convert to dict
        d = profile.to_dict()

        # Verify serializable
        import json

        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Verify structure
        loaded = json.loads(json_str)
        assert "overall_sensitivity" in loaded
        assert "recommendations" in loaded
