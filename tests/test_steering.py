"""Tests for the steering module."""

import pytest
import math
from insideLLMs.steering import (
    # Enums
    SteeringMethod,
    ActivationLayer,
    RepresentationSpace,
    SteeringStrength,
    # Dataclasses
    SteeringVector,
    ActivationPattern,
    SteeringExperiment,
    ContrastPair,
    SteeringReport,
    # Classes
    SteeringVectorExtractor,
    PromptSteerer,
    ActivationAnalyzer,
    BehavioralShiftMeasurer,
    SteeringExperimenter,
    RepresentationAnalyzer,
    # Convenience functions
    extract_steering_vector,
    create_contrast_pair,
    apply_prompt_steering,
    measure_behavioral_shift,
    analyze_activation_patterns,
    quick_steering_analysis,
)


# ============================================================================
# Enum Tests
# ============================================================================

class TestSteeringEnums:
    """Test steering-related enums."""

    def test_steering_method_values(self):
        assert SteeringMethod.PROMPT_PREFIX.value == "prompt_prefix"
        assert SteeringMethod.PROMPT_SUFFIX.value == "prompt_suffix"
        assert SteeringMethod.SYSTEM_MESSAGE.value == "system_message"
        assert SteeringMethod.FEW_SHOT.value == "few_shot"
        assert SteeringMethod.CONTRAST_PAIR.value == "contrast_pair"
        assert SteeringMethod.ACTIVATION_ADDITION.value == "activation_addition"
        assert SteeringMethod.SOFT_PROMPT.value == "soft_prompt"

    def test_activation_layer_values(self):
        assert ActivationLayer.INPUT_EMBEDDING.value == "input_embedding"
        assert ActivationLayer.ATTENTION.value == "attention"
        assert ActivationLayer.MLP.value == "mlp"
        assert ActivationLayer.RESIDUAL.value == "residual"
        assert ActivationLayer.OUTPUT.value == "output"
        assert ActivationLayer.ALL.value == "all"

    def test_representation_space_values(self):
        assert RepresentationSpace.TOKEN.value == "token"
        assert RepresentationSpace.SEQUENCE.value == "sequence"
        assert RepresentationSpace.SEMANTIC.value == "semantic"
        assert RepresentationSpace.TASK.value == "task"

    def test_steering_strength_values(self):
        assert SteeringStrength.MINIMAL.value == "minimal"
        assert SteeringStrength.LIGHT.value == "light"
        assert SteeringStrength.MODERATE.value == "moderate"
        assert SteeringStrength.STRONG.value == "strong"
        assert SteeringStrength.MAXIMUM.value == "maximum"


# ============================================================================
# Dataclass Tests
# ============================================================================

class TestSteeringVector:
    """Test SteeringVector dataclass."""

    def test_creation(self):
        vector = SteeringVector(
            name="formality",
            direction=[0.5, 0.5, 0.707],
            magnitude=1.0,
            source="contrast_pair",
            target_behavior="increase formality",
        )
        assert vector.name == "formality"
        assert len(vector.direction) == 3
        assert vector.magnitude == 1.0
        assert vector.source == "contrast_pair"

    def test_to_dict(self):
        vector = SteeringVector(
            name="test",
            direction=[1.0, 0.0],
            magnitude=1.0,
            source="manual",
            target_behavior="test behavior",
            layer="layer_5",
            metadata={"info": "test"},
        )
        d = vector.to_dict()
        assert d["name"] == "test"
        assert d["direction"] == [1.0, 0.0]
        assert d["layer"] == "layer_5"
        assert d["metadata"]["info"] == "test"

    def test_from_dict(self):
        data = {
            "name": "test",
            "direction": [0.6, 0.8],
            "magnitude": 1.0,
            "source": "examples",
            "target_behavior": "be helpful",
            "layer": "output",
            "metadata": {"key": "value"},
        }
        vector = SteeringVector.from_dict(data)
        assert vector.name == "test"
        assert vector.direction == [0.6, 0.8]
        assert vector.layer == "output"


class TestActivationPattern:
    """Test ActivationPattern dataclass."""

    def test_creation(self):
        pattern = ActivationPattern(
            prompt="Hello world",
            layer="attention_5",
            activations=[0.1, 0.5, 0.3, 0.8],
            token_positions=[0, 1, 2, 3],
        )
        assert pattern.prompt == "Hello world"
        assert pattern.layer == "attention_5"
        assert len(pattern.activations) == 4

    def test_mean_activation(self):
        pattern = ActivationPattern(
            prompt="test",
            layer="test",
            activations=[1.0, 2.0, 3.0, 4.0],
            token_positions=[0, 1, 2, 3],
        )
        assert pattern.mean_activation == 2.5

    def test_mean_activation_empty(self):
        pattern = ActivationPattern(
            prompt="test",
            layer="test",
            activations=[],
            token_positions=[],
        )
        assert pattern.mean_activation == 0.0

    def test_activation_variance(self):
        pattern = ActivationPattern(
            prompt="test",
            layer="test",
            activations=[1.0, 2.0, 3.0, 4.0],
            token_positions=[0, 1, 2, 3],
        )
        # Mean is 2.5, variance = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2)/4
        # = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
        assert pattern.activation_variance == 1.25

    def test_to_dict(self):
        pattern = ActivationPattern(
            prompt="test",
            layer="mlp",
            activations=[0.5, 0.5],
            token_positions=[0, 1],
            attention_weights=[[0.5, 0.5], [0.3, 0.7]],
        )
        d = pattern.to_dict()
        assert d["prompt"] == "test"
        assert d["attention_weights"] is not None


class TestSteeringExperiment:
    """Test SteeringExperiment dataclass."""

    def test_creation(self):
        exp = SteeringExperiment(
            original_prompt="Hello",
            steering_method=SteeringMethod.PROMPT_PREFIX,
            steering_config={"instruction": "Be formal"},
            original_output="Hi there!",
            steered_output="Greetings. How may I assist you?",
            behavioral_shift=0.7,
            direction_alignment=0.8,
            side_effects=["Length increased"],
        )
        assert exp.steering_method == SteeringMethod.PROMPT_PREFIX
        assert exp.behavioral_shift == 0.7

    def test_to_dict(self):
        exp = SteeringExperiment(
            original_prompt="test",
            steering_method=SteeringMethod.FEW_SHOT,
            steering_config={},
            original_output="out1",
            steered_output="out2",
            behavioral_shift=0.5,
            direction_alignment=0.5,
            side_effects=[],
        )
        d = exp.to_dict()
        assert d["steering_method"] == "few_shot"


class TestContrastPair:
    """Test ContrastPair dataclass."""

    def test_creation(self):
        pair = ContrastPair(
            positive_prompt="Please respond formally: Hello",
            negative_prompt="Feel free to be casual: Hello",
            target_dimension="formality",
            expected_difference="More formal vs more casual",
        )
        assert "formally" in pair.positive_prompt
        assert "casual" in pair.negative_prompt

    def test_to_dict(self):
        pair = ContrastPair(
            positive_prompt="pos",
            negative_prompt="neg",
            target_dimension="test",
            expected_difference="diff",
        )
        d = pair.to_dict()
        assert d["positive_prompt"] == "pos"
        assert d["target_dimension"] == "test"


class TestSteeringReport:
    """Test SteeringReport dataclass."""

    def test_creation(self):
        exp = SteeringExperiment(
            original_prompt="test",
            steering_method=SteeringMethod.PROMPT_PREFIX,
            steering_config={},
            original_output="out1",
            steered_output="out2",
            behavioral_shift=0.6,
            direction_alignment=0.7,
            side_effects=[],
        )
        report = SteeringReport(
            experiments=[exp],
            vectors_extracted=[],
            effective_methods=[(SteeringMethod.PROMPT_PREFIX, 0.6)],
            behavioral_dimensions={"formality": 0.5},
            recommendations=["Use prefix steering"],
        )
        assert len(report.experiments) == 1
        assert report.overall_steerability == 0.6

    def test_overall_steerability_empty(self):
        report = SteeringReport(
            experiments=[],
            vectors_extracted=[],
            effective_methods=[],
            behavioral_dimensions={},
            recommendations=[],
        )
        assert report.overall_steerability == 0.0


# ============================================================================
# SteeringVectorExtractor Tests
# ============================================================================

class TestSteeringVectorExtractor:
    """Test SteeringVectorExtractor class."""

    def test_extract_from_contrast_pair(self):
        extractor = SteeringVectorExtractor(normalize=False)
        positive = [1.0, 2.0, 3.0]
        negative = [0.5, 1.0, 1.5]

        vector = extractor.extract_from_contrast_pair(
            positive, negative, "test", "increase values"
        )

        assert vector.name == "test"
        assert vector.direction == [0.5, 1.0, 1.5]
        assert vector.source == "contrast_pair"

    def test_extract_normalized(self):
        extractor = SteeringVectorExtractor(normalize=True)
        positive = [3.0, 0.0]
        negative = [0.0, 0.0]

        vector = extractor.extract_from_contrast_pair(
            positive, negative, "test", "test"
        )

        # Should be normalized to unit length
        magnitude = math.sqrt(sum(x ** 2 for x in vector.direction))
        assert abs(magnitude - 1.0) < 0.001

    def test_extract_mismatched_lengths(self):
        extractor = SteeringVectorExtractor()
        with pytest.raises(ValueError):
            extractor.extract_from_contrast_pair(
                [1.0, 2.0], [1.0, 2.0, 3.0], "test", "test"
            )

    def test_extract_from_examples(self):
        extractor = SteeringVectorExtractor(normalize=False)
        positive_examples = [[2.0, 4.0], [4.0, 6.0]]  # mean: [3.0, 5.0]
        negative_examples = [[1.0, 2.0], [1.0, 2.0]]  # mean: [1.0, 2.0]

        vector = extractor.extract_from_examples(
            positive_examples, negative_examples, "test", "test"
        )

        assert vector.direction == [2.0, 3.0]

    def test_extract_from_examples_empty(self):
        extractor = SteeringVectorExtractor()
        with pytest.raises(ValueError):
            extractor.extract_from_examples([], [[1.0]], "test", "test")

    def test_combine_vectors(self):
        extractor = SteeringVectorExtractor(normalize=False)
        v1 = SteeringVector("v1", [1.0, 0.0], 1.0, "manual", "t1")
        v2 = SteeringVector("v2", [0.0, 1.0], 1.0, "manual", "t2")

        combined = extractor.combine_vectors([v1, v2], [0.5, 0.5])

        assert combined.direction == [0.5, 0.5]
        assert combined.source == "combination"

    def test_combine_vectors_empty(self):
        extractor = SteeringVectorExtractor()
        with pytest.raises(ValueError):
            extractor.combine_vectors([])

    def test_combine_vectors_mismatched_weights(self):
        extractor = SteeringVectorExtractor()
        v1 = SteeringVector("v1", [1.0], 1.0, "manual", "t1")
        with pytest.raises(ValueError):
            extractor.combine_vectors([v1], [0.5, 0.5])


# ============================================================================
# PromptSteerer Tests
# ============================================================================

class TestPromptSteerer:
    """Test PromptSteerer class."""

    def test_steer_with_prefix(self):
        steerer = PromptSteerer()
        result = steerer.steer_with_prefix("Hello", "Be formal.")
        assert result.startswith("Be formal.")
        assert "Hello" in result

    def test_steer_with_suffix(self):
        steerer = PromptSteerer()
        result = steerer.steer_with_suffix("Hello", "Be formal.")
        assert result.endswith("Be formal.")
        assert "Hello" in result

    def test_steer_with_system_message(self):
        steerer = PromptSteerer()
        system, user = steerer.steer_with_system_message("Hello", "You are formal")
        assert system == "You are formal"
        assert user == "Hello"

    def test_steer_with_few_shot(self):
        steerer = PromptSteerer()
        examples = [("Hi", "Hello!"), ("Thanks", "You're welcome!")]
        result = steerer.steer_with_few_shot("Bye", examples)

        assert "Example 1:" in result
        assert "Example 2:" in result
        assert "Bye" in result

    def test_apply_steering_prefix(self):
        steerer = PromptSteerer()
        result = steerer.apply_steering(
            "test",
            SteeringMethod.PROMPT_PREFIX,
            {"instruction": "Be brief."},
        )
        assert "Be brief." in result
        assert "test" in result

    def test_apply_steering_suffix(self):
        steerer = PromptSteerer()
        result = steerer.apply_steering(
            "test",
            SteeringMethod.PROMPT_SUFFIX,
            {"instruction": "Be brief."},
        )
        assert result.endswith("Be brief.")

    def test_apply_steering_system_message(self):
        steerer = PromptSteerer()
        result = steerer.apply_steering(
            "test",
            SteeringMethod.SYSTEM_MESSAGE,
            {"system_message": "You are helpful"},
        )
        assert isinstance(result, tuple)
        assert result[0] == "You are helpful"

    def test_apply_steering_few_shot(self):
        steerer = PromptSteerer()
        result = steerer.apply_steering(
            "query",
            SteeringMethod.FEW_SHOT,
            {"examples": [("a", "b")]},
        )
        assert "Example 1:" in result

    def test_get_preset_steering(self):
        steerer = PromptSteerer()
        formal = steerer.get_preset_steering("formal")
        assert "formal" in formal.lower()

        casual = steerer.get_preset_steering("casual")
        assert "casual" in casual.lower()

        unknown = steerer.get_preset_steering("nonexistent")
        assert unknown == ""

    def test_create_contrast_pair(self):
        steerer = PromptSteerer()
        pair = steerer.create_contrast_pair("Hello", "formality")

        assert pair.target_dimension == "formality"
        assert pair.positive_prompt != pair.negative_prompt


# ============================================================================
# ActivationAnalyzer Tests
# ============================================================================

class TestActivationAnalyzer:
    """Test ActivationAnalyzer class."""

    def test_record_pattern(self):
        analyzer = ActivationAnalyzer()
        pattern = analyzer.record_pattern(
            "test", "layer1", [0.1, 0.2, 0.3]
        )

        assert len(analyzer.patterns) == 1
        assert pattern.prompt == "test"

    def test_compare_patterns_identical(self):
        analyzer = ActivationAnalyzer()
        p1 = ActivationPattern("a", "l1", [1.0, 0.0], [0, 1])
        p2 = ActivationPattern("b", "l1", [1.0, 0.0], [0, 1])

        result = analyzer.compare_patterns(p1, p2)
        assert result["cosine_similarity"] == 1.0
        assert result["l2_distance"] == 0.0

    def test_compare_patterns_orthogonal(self):
        analyzer = ActivationAnalyzer()
        p1 = ActivationPattern("a", "l1", [1.0, 0.0], [0, 1])
        p2 = ActivationPattern("b", "l1", [0.0, 1.0], [0, 1])

        result = analyzer.compare_patterns(p1, p2)
        assert abs(result["cosine_similarity"]) < 0.001

    def test_compare_patterns_different_lengths(self):
        analyzer = ActivationAnalyzer()
        p1 = ActivationPattern("a", "l1", [1.0, 0.0, 0.5], [0, 1, 2])
        p2 = ActivationPattern("b", "l1", [1.0, 0.0], [0, 1])

        result = analyzer.compare_patterns(p1, p2)
        # Should use minimum length
        assert "cosine_similarity" in result

    def test_find_salient_positions(self):
        analyzer = ActivationAnalyzer()
        # Create pattern with one clear outlier
        # Mean of [1.0, 1.0, 1.0, 10.0, 1.0] = 2.8
        # Std deviation should make 10.0 a clear outlier
        activations = [1.0, 1.0, 1.0, 10.0, 1.0]
        pattern = ActivationPattern("test", "l1", activations, list(range(5)))

        salient = analyzer.find_salient_positions(pattern, threshold=1.5)
        assert 3 in salient

    def test_find_salient_positions_empty(self):
        analyzer = ActivationAnalyzer()
        pattern = ActivationPattern("test", "l1", [], [])
        salient = analyzer.find_salient_positions(pattern)
        assert salient == []

    def test_cluster_patterns(self):
        analyzer = ActivationAnalyzer()
        patterns = [
            ActivationPattern("p1", "l1", [1.0, 0.0], [0, 1]),
            ActivationPattern("p2", "l1", [0.9, 0.1], [0, 1]),
            ActivationPattern("p3", "l1", [0.0, 1.0], [0, 1]),
        ]

        clusters = analyzer.cluster_patterns(patterns, n_clusters=2)
        assert len(clusters) <= 2

    def test_cluster_patterns_empty(self):
        analyzer = ActivationAnalyzer()
        clusters = analyzer.cluster_patterns([])
        assert clusters == {}


# ============================================================================
# BehavioralShiftMeasurer Tests
# ============================================================================

class TestBehavioralShiftMeasurer:
    """Test BehavioralShiftMeasurer class."""

    def test_measure_length_shift_doubled(self):
        measurer = BehavioralShiftMeasurer()
        original = "one two three"
        steered = "one two three four five six"

        shift = measurer.measure_shift(original, steered, "length")
        assert shift > 0  # Got longer

    def test_measure_length_shift_halved(self):
        measurer = BehavioralShiftMeasurer()
        original = "one two three four"
        steered = "one two"

        shift = measurer.measure_shift(original, steered, "length")
        assert shift < 0  # Got shorter

    def test_measure_formality_shift(self):
        measurer = BehavioralShiftMeasurer()
        original = "Yeah that's gonna work kinda well"
        steered = "Therefore, this approach will consequently function effectively"

        shift = measurer.measure_shift(original, steered, "formality")
        assert shift > 0  # More formal

    def test_measure_generic_shift_same(self):
        measurer = BehavioralShiftMeasurer()
        text = "hello world"
        shift = measurer.measure_shift(text, text, "generic")
        assert shift == 0.0

    def test_measure_generic_shift_different(self):
        measurer = BehavioralShiftMeasurer()
        original = "hello world"
        steered = "completely different text here"

        shift = measurer.measure_shift(original, steered, "generic")
        assert shift > 0.5  # Significant difference

    def test_detect_side_effects(self):
        measurer = BehavioralShiftMeasurer()
        original = "Hello"
        steered = "Therefore, consequently and furthermore, greetings to you"

        side_effects = measurer.detect_side_effects(
            original, steered, "sentiment"  # targeting sentiment
        )
        # Should detect formality and length as side effects
        assert len(side_effects) > 0


# ============================================================================
# SteeringExperimenter Tests
# ============================================================================

class TestSteeringExperimenter:
    """Test SteeringExperimenter class."""

    def test_creation_without_model(self):
        experimenter = SteeringExperimenter()
        assert experimenter.model_fn is None

    def test_set_model(self):
        experimenter = SteeringExperimenter()
        mock_model = lambda x: f"Response to: {x}"
        experimenter.set_model(mock_model)
        assert experimenter.model_fn is not None

    def test_run_experiment_no_model(self):
        experimenter = SteeringExperimenter()
        with pytest.raises(ValueError, match="Model function not set"):
            experimenter.run_experiment(
                "test", SteeringMethod.PROMPT_PREFIX, {}
            )

    def test_run_experiment(self):
        def mock_model(prompt):
            if "formal" in prompt.lower():
                return "Greetings. I shall assist you."
            return "Hey there! What's up?"

        experimenter = SteeringExperimenter(model_fn=mock_model)
        experiment = experimenter.run_experiment(
            "Hello",
            SteeringMethod.PROMPT_PREFIX,
            {"instruction": "Please respond in a formal, professional tone."},
            "formality",
        )

        assert experiment.original_output == "Hey there! What's up?"
        assert "Greetings" in experiment.steered_output

    def test_run_method_comparison(self):
        def mock_model(prompt):
            return f"Response: {len(prompt)}"

        experimenter = SteeringExperimenter(model_fn=mock_model)
        experiments = experimenter.run_method_comparison(
            "test prompt",
            "formality",
            [SteeringMethod.PROMPT_PREFIX, SteeringMethod.PROMPT_SUFFIX],
        )

        assert len(experiments) == 2

    def test_analyze_steerability(self):
        responses = iter([
            "casual response",
            "formal response",
            "short",
            "very long and detailed response",
        ])

        def mock_model(prompt):
            try:
                return next(responses)
            except StopIteration:
                return "default response"

        experimenter = SteeringExperimenter(model_fn=mock_model)
        report = experimenter.analyze_steerability(
            ["test"],
            ["formality"],
        )

        assert isinstance(report, SteeringReport)
        assert len(report.experiments) > 0


# ============================================================================
# RepresentationAnalyzer Tests
# ============================================================================

class TestRepresentationAnalyzer:
    """Test RepresentationAnalyzer class."""

    def test_store_representation(self):
        analyzer = RepresentationAnalyzer()
        analyzer.store_representation("test", [1.0, 2.0, 3.0])
        assert "test" in analyzer.representations

    def test_compute_similarity_matrix(self):
        analyzer = RepresentationAnalyzer()
        analyzer.store_representation("a", [1.0, 0.0])
        analyzer.store_representation("b", [1.0, 0.0])
        analyzer.store_representation("c", [0.0, 1.0])

        matrix = analyzer.compute_similarity_matrix(["a", "b", "c"])

        assert len(matrix) == 3
        assert matrix[0][0] == 1.0  # Self-similarity
        assert matrix[0][1] == 1.0  # a and b identical
        assert abs(matrix[0][2]) < 0.001  # a and c orthogonal

    def test_find_similar_representations(self):
        analyzer = RepresentationAnalyzer()
        analyzer.store_representation("query", [1.0, 0.0])
        analyzer.store_representation("similar", [0.9, 0.1])
        analyzer.store_representation("different", [0.0, 1.0])

        similar = analyzer.find_similar_representations("query", top_k=2)

        assert len(similar) == 2
        assert similar[0][0] == "similar"  # Most similar first

    def test_find_similar_missing_query(self):
        analyzer = RepresentationAnalyzer()
        similar = analyzer.find_similar_representations("nonexistent")
        assert similar == []

    def test_project_to_2d(self):
        analyzer = RepresentationAnalyzer()
        analyzer.store_representation("a", [1.0, 2.0, 3.0])
        analyzer.store_representation("b", [4.0, 5.0, 6.0])

        projections = analyzer.project_to_2d(["a", "b"])

        assert len(projections) == 2
        assert projections[0] == ("a", 1.0, 2.0)
        assert projections[1] == ("b", 4.0, 5.0)

    def test_project_to_2d_empty(self):
        analyzer = RepresentationAnalyzer()
        projections = analyzer.project_to_2d([])
        assert projections == []


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_extract_steering_vector(self):
        vector = extract_steering_vector(
            [1.0, 2.0],
            [0.0, 1.0],
            "test",
            "increase",
            normalize=False,
        )
        assert vector.direction == [1.0, 1.0]

    def test_create_contrast_pair(self):
        pair = create_contrast_pair("Hello", "formality")
        assert pair.target_dimension == "formality"
        assert pair.positive_prompt != pair.negative_prompt

    def test_apply_prompt_steering(self):
        result = apply_prompt_steering(
            "test",
            SteeringMethod.PROMPT_PREFIX,
            {"instruction": "Be formal"},
        )
        assert "Be formal" in result

    def test_measure_behavioral_shift(self):
        shift = measure_behavioral_shift(
            "short text",
            "this is a much much longer text with many more words",
            "length",
        )
        assert shift > 0

    def test_analyze_activation_patterns(self):
        patterns = [
            ActivationPattern("p1", "l1", [0.5, 0.5], [0, 1]),
            ActivationPattern("p2", "l1", [0.6, 0.4], [0, 1]),
        ]

        analysis = analyze_activation_patterns(patterns)

        assert analysis["num_patterns"] == 2
        assert "mean_activation_avg" in analysis

    def test_analyze_activation_patterns_empty(self):
        analysis = analyze_activation_patterns([])
        assert "error" in analysis

    def test_quick_steering_analysis(self):
        def mock_model(prompt):
            return f"Response to: {prompt[:20]}"

        report = quick_steering_analysis(
            mock_model,
            ["test prompt"],
            ["formality"],
        )

        assert isinstance(report, SteeringReport)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_activations(self):
        pattern = ActivationPattern("test", "l1", [], [])
        assert pattern.mean_activation == 0.0
        assert pattern.activation_variance == 0.0

    def test_single_activation(self):
        pattern = ActivationPattern("test", "l1", [5.0], [0])
        assert pattern.mean_activation == 5.0
        assert pattern.activation_variance == 0.0

    def test_zero_magnitude_vector(self):
        extractor = SteeringVectorExtractor(normalize=True)
        # Same positive and negative should give zero vector
        vector = extractor.extract_from_contrast_pair(
            [1.0, 1.0], [1.0, 1.0], "zero", "none"
        )
        assert vector.magnitude == 0.0

    def test_empty_text_shift(self):
        measurer = BehavioralShiftMeasurer()
        shift = measurer.measure_shift("", "", "generic")
        assert shift == 0.0

    def test_empty_text_length_shift(self):
        measurer = BehavioralShiftMeasurer()
        shift = measurer.measure_shift("", "hello", "length")
        assert shift == 1.0  # Something from nothing

    def test_cosine_similarity_zero_vectors(self):
        analyzer = ActivationAnalyzer()
        p1 = ActivationPattern("a", "l1", [0.0, 0.0], [0, 1])
        p2 = ActivationPattern("b", "l1", [0.0, 0.0], [0, 1])

        result = analyzer.compare_patterns(p1, p2)
        assert result["cosine_similarity"] == 0.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for steering workflows."""

    def test_full_steering_workflow(self):
        # 1. Create contrast pair
        pair = create_contrast_pair("What is AI?", "formality")

        # 2. Mock model responses
        def mock_model(prompt):
            if "formal" in prompt.lower():
                return "Artificial Intelligence refers to computational systems..."
            return "AI is basically computers being smart!"

        # 3. Run experiment
        experimenter = SteeringExperimenter(model_fn=mock_model)
        experiment = experimenter.run_experiment(
            "What is AI?",
            SteeringMethod.PROMPT_PREFIX,
            {"instruction": pair.positive_prompt.split("\n")[0]},
            "formality",
        )

        # 4. Verify results
        assert experiment.original_output != experiment.steered_output

    def test_activation_analysis_workflow(self):
        analyzer = ActivationAnalyzer()

        # Record several patterns
        patterns = []
        for i in range(5):
            pattern = analyzer.record_pattern(
                f"prompt_{i}",
                "layer_1",
                [float(i) * 0.1, float(i) * 0.2, float(i) * 0.3],
            )
            patterns.append(pattern)

        # Compare first and last
        comparison = analyzer.compare_patterns(patterns[0], patterns[-1])
        assert comparison["l2_distance"] > 0

        # Cluster patterns
        clusters = analyzer.cluster_patterns(patterns, n_clusters=2)
        total_patterns = sum(len(c) for c in clusters.values())
        assert total_patterns == 5

    def test_representation_analysis_workflow(self):
        analyzer = RepresentationAnalyzer()

        # Store representations for different prompts
        analyzer.store_representation("formal_greeting", [0.9, 0.1, 0.0])
        analyzer.store_representation("casual_greeting", [0.1, 0.9, 0.0])
        analyzer.store_representation("formal_farewell", [0.85, 0.15, 0.0])
        analyzer.store_representation("casual_farewell", [0.15, 0.85, 0.0])

        # Find similar to formal greeting
        similar = analyzer.find_similar_representations("formal_greeting")

        # Formal farewell should be most similar
        assert similar[0][0] == "formal_farewell"

        # Compute similarity matrix
        matrix = analyzer.compute_similarity_matrix()
        assert len(matrix) == 4
