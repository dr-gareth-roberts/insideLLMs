"""Tests for multi-model ensemble evaluation module."""

from insideLLMs.ensemble import (
    AggregatedOutput,
    # Enums
    AggregationMethod,
    AgreementLevel,
    EnsembleEvaluator,
    EnsembleReport,
    ModelAgreementAnalyzer,
    ModelEnsemble,
    ModelOutput,
    ResponseAggregator,
    # Classes
    ResponseNormalizer,
    SimilarityCalculator,
    aggregate_responses,
    analyze_model_agreement,
    # Functions
    create_ensemble,
    evaluate_ensemble,
    quick_ensemble_check,
)

# ============================================================================
# Enum Tests
# ============================================================================


class TestAggregationMethod:
    """Tests for AggregationMethod enum."""

    def test_all_methods_defined(self):
        """Test all aggregation methods are defined."""
        assert AggregationMethod.MAJORITY_VOTE.value == "majority_vote"
        assert AggregationMethod.WEIGHTED_VOTE.value == "weighted_vote"
        assert AggregationMethod.BEST_OF_N.value == "best_of_n"
        assert AggregationMethod.CONSENSUS.value == "consensus"
        assert AggregationMethod.LONGEST.value == "longest"
        assert AggregationMethod.SHORTEST.value == "shortest"

    def test_method_count(self):
        """Test correct number of methods."""
        assert len(AggregationMethod) == 8


class TestAgreementLevel:
    """Tests for AgreementLevel enum."""

    def test_all_levels_defined(self):
        """Test all agreement levels are defined."""
        assert AgreementLevel.UNANIMOUS.value == "unanimous"
        assert AgreementLevel.STRONG.value == "strong"
        assert AgreementLevel.MODERATE.value == "moderate"
        assert AgreementLevel.WEAK.value == "weak"
        assert AgreementLevel.NONE.value == "none"


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestModelOutput:
    """Tests for ModelOutput dataclass."""

    def test_creation(self):
        """Test creating a model output."""
        output = ModelOutput(
            model_id="gpt-4",
            response="The answer is 42.",
            confidence=0.95,
            latency=1.5,
        )
        assert output.model_id == "gpt-4"
        assert output.response == "The answer is 42."
        assert output.confidence == 0.95

    def test_default_values(self):
        """Test default values."""
        output = ModelOutput(model_id="model", response="response")
        assert output.confidence == 1.0
        assert output.latency == 0.0
        assert output.metadata == {}

    def test_to_dict(self):
        """Test dictionary conversion."""
        output = ModelOutput(model_id="test", response="test response")
        d = output.to_dict()
        assert d["model_id"] == "test"
        assert d["response"] == "test response"


class TestAggregatedOutput:
    """Tests for AggregatedOutput dataclass."""

    def test_creation(self):
        """Test creating an aggregated output."""
        outputs = [
            ModelOutput(model_id="m1", response="answer"),
            ModelOutput(model_id="m2", response="answer"),
        ]
        aggregated = AggregatedOutput(
            final_response="answer",
            method=AggregationMethod.MAJORITY_VOTE,
            source_outputs=outputs,
            agreement_level=AgreementLevel.UNANIMOUS,
            agreement_score=1.0,
            selected_model="m1",
            vote_distribution={"0": 2},
        )
        assert aggregated.final_response == "answer"
        assert aggregated.agreement_score == 1.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        aggregated = AggregatedOutput(
            final_response="test",
            method=AggregationMethod.CONSENSUS,
            source_outputs=[],
            agreement_level=AgreementLevel.MODERATE,
            agreement_score=0.5,
            selected_model="m1",
            vote_distribution={},
        )
        d = aggregated.to_dict()
        assert d["method"] == "consensus"
        assert d["agreement_level"] == "moderate"


class TestEnsembleReport:
    """Tests for EnsembleReport dataclass."""

    def test_creation(self):
        """Test creating an ensemble report."""
        report = EnsembleReport(
            n_prompts=10,
            n_models=3,
            model_ids=["m1", "m2", "m3"],
            aggregation_method=AggregationMethod.MAJORITY_VOTE,
            overall_agreement=0.8,
            per_model_selection_rate={"m1": 0.5, "m2": 0.3, "m3": 0.2},
            per_model_agreement={"m1": 0.85, "m2": 0.75, "m3": 0.65},
            best_performing_model="m1",
            most_agreeable_model="m1",
            ensemble_diversity=0.3,
            recommendations=["Test recommendation"],
        )
        assert report.n_prompts == 10
        assert report.best_performing_model == "m1"

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = EnsembleReport(
            n_prompts=5,
            n_models=2,
            model_ids=["a", "b"],
            aggregation_method=AggregationMethod.CONSENSUS,
            overall_agreement=0.7,
            per_model_selection_rate={},
            per_model_agreement={},
            best_performing_model="a",
            most_agreeable_model="b",
            ensemble_diversity=0.4,
            recommendations=[],
        )
        d = report.to_dict()
        assert d["n_prompts"] == 5
        assert d["aggregation_method"] == "consensus"


# ============================================================================
# ResponseNormalizer Tests
# ============================================================================


class TestResponseNormalizer:
    """Tests for ResponseNormalizer class."""

    def test_lowercase(self):
        """Test lowercase normalization."""
        normalizer = ResponseNormalizer(lowercase=True)
        assert normalizer.normalize("HELLO World") == "hello world"

    def test_strip_whitespace(self):
        """Test whitespace stripping."""
        normalizer = ResponseNormalizer(strip_whitespace=True)
        assert normalizer.normalize("  hello   world  ") == "hello world"

    def test_remove_punctuation(self):
        """Test punctuation removal."""
        normalizer = ResponseNormalizer(remove_punctuation=True)
        result = normalizer.normalize("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_all_options(self):
        """Test all normalization options."""
        normalizer = ResponseNormalizer(
            lowercase=True,
            strip_whitespace=True,
            remove_punctuation=True,
        )
        result = normalizer.normalize("  HELLO,   World!  ")
        assert result == "hello world"


# ============================================================================
# SimilarityCalculator Tests
# ============================================================================


class TestSimilarityCalculator:
    """Tests for SimilarityCalculator class."""

    def test_identical_responses(self):
        """Test identical responses have similarity 1.0."""
        calculator = SimilarityCalculator()
        assert calculator.calculate("hello world", "hello world") == 1.0

    def test_completely_different(self):
        """Test completely different responses."""
        calculator = SimilarityCalculator()
        sim = calculator.calculate("abc def", "xyz uvw")
        assert sim == 0.0

    def test_partial_overlap(self):
        """Test partial word overlap."""
        calculator = SimilarityCalculator()
        sim = calculator.calculate("hello world today", "hello world tomorrow")
        assert 0 < sim < 1

    def test_case_insensitive(self):
        """Test case insensitivity."""
        calculator = SimilarityCalculator()
        sim = calculator.calculate("HELLO WORLD", "hello world")
        assert sim == 1.0

    def test_empty_responses(self):
        """Test empty responses."""
        calculator = SimilarityCalculator()
        assert calculator.calculate("", "") == 1.0


# ============================================================================
# ResponseAggregator Tests
# ============================================================================


class TestResponseAggregator:
    """Tests for ResponseAggregator class."""

    def test_majority_vote_unanimous(self):
        """Test majority vote with unanimous outputs."""
        outputs = [
            ModelOutput(model_id="m1", response="answer"),
            ModelOutput(model_id="m2", response="answer"),
            ModelOutput(model_id="m3", response="answer"),
        ]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs, AggregationMethod.MAJORITY_VOTE)

        assert result.final_response == "answer"
        assert result.agreement_level == AgreementLevel.UNANIMOUS

    def test_majority_vote_split(self):
        """Test majority vote with split outputs."""
        outputs = [
            ModelOutput(model_id="m1", response="answer A"),
            ModelOutput(model_id="m2", response="answer A"),
            ModelOutput(model_id="m3", response="answer B"),
        ]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs, AggregationMethod.MAJORITY_VOTE)

        assert result.final_response == "answer A"

    def test_weighted_vote(self):
        """Test weighted vote by confidence."""
        outputs = [
            ModelOutput(model_id="m1", response="low confidence", confidence=0.3),
            ModelOutput(model_id="m2", response="high confidence", confidence=0.9),
        ]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs, AggregationMethod.WEIGHTED_VOTE)

        assert result.selected_model == "m2"

    def test_longest(self):
        """Test longest response selection."""
        outputs = [
            ModelOutput(model_id="m1", response="short"),
            ModelOutput(model_id="m2", response="much longer response here"),
        ]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs, AggregationMethod.LONGEST)

        assert result.selected_model == "m2"

    def test_shortest(self):
        """Test shortest response selection."""
        outputs = [
            ModelOutput(model_id="m1", response="short"),
            ModelOutput(model_id="m2", response="much longer response here"),
        ]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs, AggregationMethod.SHORTEST)

        assert result.selected_model == "m1"

    def test_most_confident(self):
        """Test most confident selection."""
        outputs = [
            ModelOutput(model_id="m1", response="unsure", confidence=0.5),
            ModelOutput(model_id="m2", response="confident", confidence=0.95),
        ]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs, AggregationMethod.MOST_CONFIDENT)

        assert result.selected_model == "m2"

    def test_consensus(self):
        """Test consensus selection."""
        outputs = [
            ModelOutput(model_id="m1", response="common answer here"),
            ModelOutput(model_id="m2", response="common answer here"),
            ModelOutput(model_id="m3", response="different outlier"),
        ]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs, AggregationMethod.CONSENSUS)

        # Should select one of the common answers
        assert result.selected_model in ["m1", "m2"]

    def test_best_of_n_with_scorer(self):
        """Test best of n with custom scorer."""
        outputs = [
            ModelOutput(model_id="m1", response="short"),
            ModelOutput(model_id="m2", response="medium length"),
            ModelOutput(model_id="m3", response="longest response here"),
        ]

        # Score by word count
        def scorer(x):
            return len(x.split())

        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs, AggregationMethod.BEST_OF_N, scorer)

        assert result.selected_model == "m3"

    def test_empty_outputs(self):
        """Test with empty outputs list."""
        aggregator = ResponseAggregator()
        result = aggregator.aggregate([], AggregationMethod.MAJORITY_VOTE)

        assert result.final_response == ""
        assert result.agreement_level == AgreementLevel.NONE

    def test_single_output(self):
        """Test with single output."""
        outputs = [ModelOutput(model_id="m1", response="only one")]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs, AggregationMethod.MAJORITY_VOTE)

        assert result.final_response == "only one"
        assert result.agreement_score == 1.0


# ============================================================================
# ModelAgreementAnalyzer Tests
# ============================================================================


class TestModelAgreementAnalyzer:
    """Tests for ModelAgreementAnalyzer class."""

    def test_perfect_agreement(self):
        """Test perfect agreement analysis."""
        outputs = [
            ModelOutput(model_id="m1", response="same answer"),
            ModelOutput(model_id="m2", response="same answer"),
            ModelOutput(model_id="m3", response="same answer"),
        ]
        analyzer = ModelAgreementAnalyzer()
        result = analyzer.analyze(outputs)

        assert result["overall_agreement"] == 1.0

    def test_no_agreement(self):
        """Test no agreement analysis."""
        outputs = [
            ModelOutput(model_id="m1", response="apple orange banana"),
            ModelOutput(model_id="m2", response="car truck bus"),
            ModelOutput(model_id="m3", response="red green blue"),
        ]
        analyzer = ModelAgreementAnalyzer()
        result = analyzer.analyze(outputs)

        assert result["overall_agreement"] < 0.5

    def test_partial_agreement(self):
        """Test partial agreement."""
        outputs = [
            ModelOutput(model_id="m1", response="the answer is correct"),
            ModelOutput(model_id="m2", response="the answer is correct"),
            ModelOutput(model_id="m3", response="the answer is wrong"),
        ]
        analyzer = ModelAgreementAnalyzer()
        result = analyzer.analyze(outputs)

        assert 0 < result["overall_agreement"] < 1

    def test_single_output(self):
        """Test single output."""
        outputs = [ModelOutput(model_id="m1", response="only one")]
        analyzer = ModelAgreementAnalyzer()
        result = analyzer.analyze(outputs)

        assert result["overall_agreement"] == 1.0

    def test_agreement_matrix(self):
        """Test agreement matrix is populated."""
        outputs = [
            ModelOutput(model_id="m1", response="response A"),
            ModelOutput(model_id="m2", response="response B"),
        ]
        analyzer = ModelAgreementAnalyzer()
        result = analyzer.analyze(outputs)

        assert len(result["agreement_matrix"]) > 0


# ============================================================================
# EnsembleEvaluator Tests
# ============================================================================


class TestEnsembleEvaluator:
    """Tests for EnsembleEvaluator class."""

    def test_evaluate_single_prompt(self):
        """Test evaluating single prompt."""
        prompt_outputs = [
            [
                ModelOutput(model_id="m1", response="answer"),
                ModelOutput(model_id="m2", response="answer"),
            ]
        ]
        evaluator = EnsembleEvaluator()
        report = evaluator.evaluate(prompt_outputs)

        assert report.n_prompts == 1
        assert report.n_models == 2
        assert report.overall_agreement == 1.0

    def test_evaluate_multiple_prompts(self):
        """Test evaluating multiple prompts."""
        prompt_outputs = [
            [
                ModelOutput(model_id="m1", response="yes"),
                ModelOutput(model_id="m2", response="yes"),
            ],
            [
                ModelOutput(model_id="m1", response="no"),
                ModelOutput(model_id="m2", response="maybe"),
            ],
        ]
        evaluator = EnsembleEvaluator()
        report = evaluator.evaluate(prompt_outputs)

        assert report.n_prompts == 2
        assert report.n_models == 2

    def test_evaluate_empty(self):
        """Test evaluating empty data."""
        evaluator = EnsembleEvaluator()
        report = evaluator.evaluate([])

        assert report.n_prompts == 0
        assert report.n_models == 0

    def test_selection_rates(self):
        """Test selection rate calculation."""
        # m1 always wins
        prompt_outputs = [
            [
                ModelOutput(model_id="m1", response="winner answer here"),
                ModelOutput(model_id="m2", response="short"),
            ],
            [
                ModelOutput(model_id="m1", response="another winner answer"),
                ModelOutput(model_id="m2", response="tiny"),
            ],
        ]
        evaluator = EnsembleEvaluator()
        report = evaluator.evaluate(prompt_outputs, AggregationMethod.LONGEST)

        assert report.per_model_selection_rate.get("m1", 0) > 0

    def test_recommendations_generated(self):
        """Test recommendations are generated."""
        prompt_outputs = [
            [
                ModelOutput(model_id="m1", response="same"),
                ModelOutput(model_id="m2", response="same"),
            ]
        ]
        evaluator = EnsembleEvaluator()
        report = evaluator.evaluate(prompt_outputs)

        assert len(report.recommendations) >= 1


# ============================================================================
# ModelEnsemble Tests
# ============================================================================


class TestModelEnsemble:
    """Tests for ModelEnsemble class."""

    def test_query(self):
        """Test querying the ensemble."""
        models = {
            "m1": lambda p: "response from m1",
            "m2": lambda p: "response from m2",
        }
        ensemble = ModelEnsemble(models)
        result = ensemble.query("test prompt")

        assert result.final_response in ["response from m1", "response from m2"]
        assert len(result.source_outputs) == 2

    def test_query_with_method(self):
        """Test querying with specific method."""
        models = {
            "m1": lambda p: "short",
            "m2": lambda p: "much longer response",
        }
        ensemble = ModelEnsemble(models)
        result = ensemble.query("test", method=AggregationMethod.LONGEST)

        assert result.final_response == "much longer response"
        assert result.selected_model == "m2"

    def test_compare_methods(self):
        """Test comparing aggregation methods."""
        models = {
            "m1": lambda p: "response A",
            "m2": lambda p: "response B",
        }
        ensemble = ModelEnsemble(models)
        results = ensemble.compare_methods(
            "test",
            [AggregationMethod.LONGEST, AggregationMethod.SHORTEST],
        )

        assert AggregationMethod.LONGEST in results
        assert AggregationMethod.SHORTEST in results

    def test_model_failure_handling(self):
        """Test handling model failures."""

        def failing_model(p):
            raise ValueError("Model failed")

        models = {
            "m1": lambda p: "working response",
            "m2": failing_model,
        }
        ensemble = ModelEnsemble(models)
        result = ensemble.query("test")

        # Should still work with remaining model
        assert result.final_response == "working response"
        assert len(result.source_outputs) == 1


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_ensemble(self):
        """Test create_ensemble function."""
        models = {"m1": lambda p: "test"}
        ensemble = create_ensemble(models)

        assert isinstance(ensemble, ModelEnsemble)

    def test_aggregate_responses(self):
        """Test aggregate_responses function."""
        outputs = [
            ModelOutput(model_id="m1", response="answer"),
            ModelOutput(model_id="m2", response="answer"),
        ]
        result = aggregate_responses(outputs)

        assert isinstance(result, AggregatedOutput)
        assert result.final_response == "answer"

    def test_analyze_model_agreement(self):
        """Test analyze_model_agreement function."""
        outputs = [
            ModelOutput(model_id="m1", response="same"),
            ModelOutput(model_id="m2", response="same"),
        ]
        result = analyze_model_agreement(outputs)

        assert "overall_agreement" in result
        assert result["overall_agreement"] == 1.0

    def test_evaluate_ensemble(self):
        """Test evaluate_ensemble function."""
        prompt_outputs = [
            [
                ModelOutput(model_id="m1", response="test"),
                ModelOutput(model_id="m2", response="test"),
            ]
        ]
        report = evaluate_ensemble(prompt_outputs)

        assert isinstance(report, EnsembleReport)
        assert report.n_prompts == 1

    def test_quick_ensemble_check(self):
        """Test quick_ensemble_check function."""
        models = {
            "m1": lambda p: "response",
            "m2": lambda p: "response",
        }
        result = quick_ensemble_check(models, ["prompt1", "prompt2"])

        assert "n_prompts" in result
        assert "avg_agreement" in result
        assert result["n_prompts"] == 2


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_model_ensemble(self):
        """Test ensemble with single model."""
        models = {"solo": lambda p: "only response"}
        ensemble = ModelEnsemble(models)
        result = ensemble.query("test")

        assert result.final_response == "only response"
        assert result.agreement_score == 1.0

    def test_all_models_fail(self):
        """Test when all models fail."""

        def failing(p):
            raise ValueError("fail")

        models = {"m1": failing, "m2": failing}
        ensemble = ModelEnsemble(models)
        result = ensemble.query("test")

        assert result.final_response == ""
        assert len(result.source_outputs) == 0

    def test_empty_responses(self):
        """Test with empty responses."""
        outputs = [
            ModelOutput(model_id="m1", response=""),
            ModelOutput(model_id="m2", response=""),
        ]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs)

        assert result.final_response == ""

    def test_very_long_responses(self):
        """Test with very long responses."""
        long_response = "word " * 1000
        outputs = [
            ModelOutput(model_id="m1", response=long_response),
            ModelOutput(model_id="m2", response=long_response),
        ]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs)

        assert result.agreement_score == 1.0

    def test_unicode_responses(self):
        """Test with unicode responses."""
        outputs = [
            ModelOutput(model_id="m1", response="你好世界"),
            ModelOutput(model_id="m2", response="你好世界"),
        ]
        aggregator = ResponseAggregator()
        result = aggregator.aggregate(outputs)

        assert result.agreement_score == 1.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for ensemble module."""

    def test_full_ensemble_workflow(self):
        """Test complete ensemble workflow."""
        # Define mock models
        models = {
            "fast": lambda p: f"Quick answer to: {p[:20]}",
            "detailed": lambda p: f"Here is a detailed response about {p}. " * 3,
            "concise": lambda p: "Yes" if "?" in p else "Statement",
        }

        # Create ensemble
        ensemble = create_ensemble(models, AggregationMethod.CONSENSUS)

        # Query
        prompts = [
            "What is the capital of France?",
            "Explain machine learning",
            "Is water wet?",
        ]

        results = []
        for prompt in prompts:
            result = ensemble.query(prompt)
            results.append(result)

        # Verify results
        assert len(results) == 3
        for result in results:
            assert result.final_response
            assert result.agreement_score >= 0

    def test_ensemble_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Collect outputs
        prompt_outputs = []
        for i in range(5):
            outputs = [
                ModelOutput(model_id="m1", response=f"Answer {i} from m1"),
                ModelOutput(model_id="m2", response=f"Answer {i} from m2"),
                ModelOutput(model_id="m3", response=f"Answer {i} from m3"),
            ]
            prompt_outputs.append(outputs)

        # Evaluate
        report = evaluate_ensemble(prompt_outputs, AggregationMethod.MAJORITY_VOTE)

        # Verify report
        assert report.n_prompts == 5
        assert report.n_models == 3
        assert len(report.recommendations) >= 1

    def test_serialization_roundtrip(self):
        """Test that all results can be serialized."""
        outputs = [
            ModelOutput(model_id="m1", response="test"),
            ModelOutput(model_id="m2", response="test"),
        ]
        result = aggregate_responses(outputs)

        # Convert to dict
        d = result.to_dict()

        # Verify serializable
        import json

        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Verify structure
        loaded = json.loads(json_str)
        assert "final_response" in loaded
        assert "agreement_level" in loaded
