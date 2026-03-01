"""Additional branch coverage for ensemble internals."""

from __future__ import annotations

from insideLLMs.models.ensemble import (
    AggregationMethod,
    AgreementLevel,
    EnsembleEvaluator,
    ModelComparison,
    ModelEnsemble,
    ModelOutput,
    ResponseAggregator,
    ResponseNormalizer,
    SimilarityCalculator,
)


class _FalseyModelOutput(ModelOutput):
    """ModelOutput variant that evaluates to False for fallback branch tests."""

    def __bool__(self) -> bool:
        return False


def test_model_comparison_to_dict_truncates_prompt():
    comparison = ModelComparison(
        prompt="x" * 300,
        outputs=[ModelOutput(model_id="m1", response="ok")],
        best_model="m1",
        worst_model="m1",
        ranking=[("m1", 1.0)],
        agreement_matrix={},
        diversity_score=0.0,
    )
    data = comparison.to_dict()
    assert len(data["prompt"]) == 200
    assert data["n_models"] == 1


def test_similarity_calculator_empty_word_sets_and_normalizer_flags():
    normalizer = ResponseNormalizer(lowercase=False, strip_whitespace=False)
    calc = SimilarityCalculator(normalizer=normalizer)
    assert calc.calculate(" ", "\t") == 1.0
    assert normalizer.normalize("AbC") == "AbC"


def test_response_aggregator_private_selection_edge_paths():
    aggregator = ResponseAggregator()
    outputs = [
        ModelOutput(model_id="a", response="short", confidence=0.2),
        ModelOutput(model_id="b", response="a bit longer", confidence=0.9),
    ]

    assert aggregator._majority_vote([], outputs) == ("", "")
    assert aggregator._weighted_vote([], outputs) == ("", "")
    assert aggregator._best_of_n([], None) == ("", "")
    assert aggregator._consensus([], []) == ("", "")
    assert aggregator._longest([]) == ("", "")
    assert aggregator._shortest([]) == ("", "")
    assert aggregator._most_confident([]) == ("", "")
    assert aggregator._diverse_selection([]) == ("", "")

    # scorer=None path selects longest response.
    best_resp, best_model = aggregator._best_of_n(outputs, scorer=None)
    assert best_resp == "a bit longer"
    assert best_model == "b"


def test_score_to_level_boundaries_and_falsey_fallbacks():
    aggregator = ResponseAggregator()
    assert aggregator._score_to_level(0.80) == AgreementLevel.STRONG
    assert aggregator._score_to_level(0.60) == AgreementLevel.MODERATE
    assert aggregator._score_to_level(0.30) == AgreementLevel.WEAK
    assert aggregator._score_to_level(0.10) == AgreementLevel.NONE

    falsey_outputs = [
        _FalseyModelOutput(model_id="m1", response="alpha"),
        _FalseyModelOutput(model_id="m2", response="beta"),
    ]
    consensus = aggregator._consensus([], falsey_outputs)
    diverse = aggregator._diverse_selection(falsey_outputs)
    assert consensus == ("alpha", "m1")
    assert diverse == ("alpha", "m1")


def test_aggregate_diverse_and_unknown_method_defaults_to_majority():
    aggregator = ResponseAggregator()
    outputs = [
        ModelOutput(model_id="m1", response="answer one"),
        ModelOutput(model_id="m2", response="answer two"),
    ]

    diverse = aggregator.aggregate(outputs, AggregationMethod.DIVERSE_SELECTION)
    assert diverse.selected_model in {"m1", "m2"}

    # Unknown method falls back to majority vote.
    fallback = aggregator.aggregate(outputs, "unknown-method")  # type: ignore[arg-type]
    assert fallback.selected_model in {"m1", "m2"}


def test_ensemble_evaluator_diversity_and_recommendation_paths():
    evaluator = EnsembleEvaluator()

    # Includes an empty prompt output to exercise selected_model missing branch.
    report = evaluator.evaluate(
        [
            [],
            [ModelOutput(model_id="m1", response="x"), ModelOutput(model_id="m2", response="y")],
            [ModelOutput(model_id="m1", response="solo")],
        ]
    )
    assert report.n_prompts == 3
    assert report.ensemble_diversity >= 0.0

    # Recommendation generation: default and non-default branches.
    default_rec = evaluator._generate_recommendations({}, {}, 0.5)
    assert default_rec == ["Ensemble appears well-balanced"]

    mixed = evaluator._generate_recommendations({"m1": 0.9}, {"m1": 0.2}, 0.8)
    assert any("dominates selection" in rec for rec in mixed)
    assert any("low agreement" in rec for rec in mixed)
    assert any("High ensemble diversity" in rec for rec in mixed)

    # max_rate <= 0.8 branch and agreements absent branch.
    neutral = evaluator._generate_recommendations({"m1": 0.6}, {}, 0.5)
    assert neutral == ["Ensemble appears well-balanced"]


def test_compare_methods_defaults_and_model_exception_skip():
    def good_model(_prompt: str) -> str:
        return "good"

    def bad_model(_prompt: str) -> str:
        raise RuntimeError("failure")

    ensemble = ModelEnsemble({"ok": good_model, "bad": bad_model})
    comparison = ensemble.compare_methods("prompt")
    assert len(comparison) == len(AggregationMethod)
    assert all(result.final_response == "good" for result in comparison.values())
