"""Multi-model ensemble evaluation for LLM comparison.

This module provides tools for evaluating and combining outputs
from multiple language models:

- Response aggregation and voting
- Confidence-weighted ensemble outputs
- Model agreement analysis
- Diversity-based selection
- Ensemble performance evaluation
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class AggregationMethod(Enum):
    """Methods for aggregating model outputs."""

    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    BEST_OF_N = "best_of_n"
    CONSENSUS = "consensus"
    LONGEST = "longest"
    SHORTEST = "shortest"
    MOST_CONFIDENT = "most_confident"
    DIVERSE_SELECTION = "diverse_selection"


class AgreementLevel(Enum):
    """Levels of model agreement."""

    UNANIMOUS = "unanimous"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class EnsembleStrategy(Enum):
    """Strategies for ensemble model selection."""

    ALL = "all"
    TOP_K = "top_k"
    THRESHOLD = "threshold"
    DIVERSE = "diverse"
    RANDOM_SUBSET = "random_subset"


@dataclass
class ModelOutput:
    """Output from a single model."""

    model_id: str
    response: str
    confidence: float = 1.0
    latency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "response": self.response[:500],
            "confidence": self.confidence,
            "latency": self.latency,
            "metadata": self.metadata,
        }


@dataclass
class AggregatedOutput:
    """Aggregated output from multiple models."""

    final_response: str
    method: AggregationMethod
    source_outputs: list[ModelOutput]
    agreement_level: AgreementLevel
    agreement_score: float
    selected_model: str | None
    vote_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_response": self.final_response[:500],
            "method": self.method.value,
            "n_models": len(self.source_outputs),
            "agreement_level": self.agreement_level.value,
            "agreement_score": self.agreement_score,
            "selected_model": self.selected_model,
            "vote_distribution": self.vote_distribution,
        }


@dataclass
class ModelComparison:
    """Comparison between models on a task."""

    prompt: str
    outputs: list[ModelOutput]
    best_model: str
    worst_model: str
    ranking: list[tuple[str, float]]
    agreement_matrix: dict[tuple[str, str], float]
    diversity_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt[:200],
            "n_models": len(self.outputs),
            "best_model": self.best_model,
            "worst_model": self.worst_model,
            "ranking": self.ranking,
            "diversity_score": self.diversity_score,
        }


@dataclass
class EnsembleReport:
    """Complete report on ensemble performance."""

    n_prompts: int
    n_models: int
    model_ids: list[str]
    aggregation_method: AggregationMethod
    overall_agreement: float
    per_model_selection_rate: dict[str, float]
    per_model_agreement: dict[str, float]
    best_performing_model: str
    most_agreeable_model: str
    ensemble_diversity: float
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_prompts": self.n_prompts,
            "n_models": self.n_models,
            "model_ids": self.model_ids,
            "aggregation_method": self.aggregation_method.value,
            "overall_agreement": self.overall_agreement,
            "per_model_selection_rate": self.per_model_selection_rate,
            "per_model_agreement": self.per_model_agreement,
            "best_performing_model": self.best_performing_model,
            "most_agreeable_model": self.most_agreeable_model,
            "ensemble_diversity": self.ensemble_diversity,
            "recommendations": self.recommendations,
        }


class ResponseNormalizer:
    """Normalize responses for comparison."""

    def __init__(
        self,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        remove_punctuation: bool = False,
    ):
        """Initialize normalizer.

        Args:
            lowercase: Convert to lowercase
            strip_whitespace: Strip extra whitespace
            remove_punctuation: Remove punctuation
        """
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.remove_punctuation = remove_punctuation

    def normalize(self, response: str) -> str:
        """Normalize a response.

        Args:
            response: Raw response

        Returns:
            Normalized response
        """
        result = response

        if self.lowercase:
            result = result.lower()

        if self.strip_whitespace:
            result = " ".join(result.split())

        if self.remove_punctuation:
            import string

            result = result.translate(str.maketrans("", "", string.punctuation))

        return result


class SimilarityCalculator:
    """Calculate similarity between responses."""

    def __init__(
        self,
        normalizer: ResponseNormalizer | None = None,
    ):
        """Initialize calculator.

        Args:
            normalizer: Optional response normalizer
        """
        self._normalizer = normalizer or ResponseNormalizer()

    def calculate(self, response1: str, response2: str) -> float:
        """Calculate similarity between two responses.

        Args:
            response1: First response
            response2: Second response

        Returns:
            Similarity score between 0 and 1
        """
        norm1 = self._normalizer.normalize(response1)
        norm2 = self._normalizer.normalize(response2)

        if norm1 == norm2:
            return 1.0

        # Word-level Jaccard similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        if not words1 and not words2:
            return 1.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class ResponseAggregator:
    """Aggregate responses from multiple models."""

    def __init__(
        self,
        similarity_calculator: SimilarityCalculator | None = None,
        similarity_threshold: float = 0.8,
    ):
        """Initialize aggregator.

        Args:
            similarity_calculator: Calculator for response similarity
            similarity_threshold: Threshold for considering responses similar
        """
        self._calculator = similarity_calculator or SimilarityCalculator()
        self._threshold = similarity_threshold

    def aggregate(
        self,
        outputs: list[ModelOutput],
        method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
        scorer: Callable[[str], float] | None = None,
    ) -> AggregatedOutput:
        """Aggregate outputs from multiple models.

        Args:
            outputs: List of model outputs
            method: Aggregation method to use
            scorer: Optional scoring function for best_of_n

        Returns:
            AggregatedOutput object
        """
        if not outputs:
            return AggregatedOutput(
                final_response="",
                method=method,
                source_outputs=[],
                agreement_level=AgreementLevel.NONE,
                agreement_score=0.0,
                selected_model=None,
                vote_distribution={},
            )

        # Calculate agreement
        agreement_score = self._calculate_agreement(outputs)
        agreement_level = self._score_to_level(agreement_score)

        # Group similar responses
        groups = self._group_similar_responses(outputs)

        # Select response based on method
        if method == AggregationMethod.MAJORITY_VOTE:
            result, selected = self._majority_vote(groups, outputs)
        elif method == AggregationMethod.WEIGHTED_VOTE:
            result, selected = self._weighted_vote(groups, outputs)
        elif method == AggregationMethod.BEST_OF_N:
            result, selected = self._best_of_n(outputs, scorer)
        elif method == AggregationMethod.CONSENSUS:
            result, selected = self._consensus(groups, outputs)
        elif method == AggregationMethod.LONGEST:
            result, selected = self._longest(outputs)
        elif method == AggregationMethod.SHORTEST:
            result, selected = self._shortest(outputs)
        elif method == AggregationMethod.MOST_CONFIDENT:
            result, selected = self._most_confident(outputs)
        elif method == AggregationMethod.DIVERSE_SELECTION:
            result, selected = self._diverse_selection(outputs)
        else:
            result, selected = self._majority_vote(groups, outputs)

        # Build vote distribution
        vote_dist = {str(i): len(g) for i, g in enumerate(groups)}

        return AggregatedOutput(
            final_response=result,
            method=method,
            source_outputs=outputs,
            agreement_level=agreement_level,
            agreement_score=agreement_score,
            selected_model=selected,
            vote_distribution=vote_dist,
        )

    def _calculate_agreement(self, outputs: list[ModelOutput]) -> float:
        """Calculate overall agreement score."""
        if len(outputs) <= 1:
            return 1.0

        similarities = []
        for i, out1 in enumerate(outputs):
            for out2 in outputs[i + 1 :]:
                sim = self._calculator.calculate(out1.response, out2.response)
                similarities.append(sim)

        return statistics.mean(similarities) if similarities else 0.0

    def _group_similar_responses(
        self,
        outputs: list[ModelOutput],
    ) -> list[list[ModelOutput]]:
        """Group similar responses together."""
        groups: list[list[ModelOutput]] = []

        for output in outputs:
            found_group = False
            for group in groups:
                if (
                    self._calculator.calculate(output.response, group[0].response)
                    >= self._threshold
                ):
                    group.append(output)
                    found_group = True
                    break

            if not found_group:
                groups.append([output])

        return groups

    @staticmethod
    def _score_to_level(score: float) -> AgreementLevel:
        """Convert agreement score to level."""
        if score >= 0.95:
            return AgreementLevel.UNANIMOUS
        elif score >= 0.75:
            return AgreementLevel.STRONG
        elif score >= 0.5:
            return AgreementLevel.MODERATE
        elif score >= 0.25:
            return AgreementLevel.WEAK
        return AgreementLevel.NONE

    @staticmethod
    def _majority_vote(
        groups: list[list[ModelOutput]],
        outputs: list[ModelOutput],
    ) -> tuple[str, str]:
        """Select response by majority vote."""
        if not groups:
            return "", ""

        largest_group = max(groups, key=len)
        selected = largest_group[0]
        return selected.response, selected.model_id

    @staticmethod
    def _weighted_vote(
        groups: list[list[ModelOutput]],
        outputs: list[ModelOutput],
    ) -> tuple[str, str]:
        """Select response by confidence-weighted vote."""
        if not groups:
            return "", ""

        group_weights = []
        for group in groups:
            weight = sum(o.confidence for o in group)
            group_weights.append((group, weight))

        best_group = max(group_weights, key=lambda x: x[1])[0]
        # Select highest confidence within group
        selected = max(best_group, key=lambda x: x.confidence)
        return selected.response, selected.model_id

    @staticmethod
    def _best_of_n(
        outputs: list[ModelOutput],
        scorer: Callable[[str], float] | None,
    ) -> tuple[str, str]:
        """Select best response using scorer."""
        if not outputs:
            return "", ""

        if scorer is None:
            # Default to longest response
            selected = max(outputs, key=lambda x: len(x.response))
        else:
            selected = max(outputs, key=lambda x: scorer(x.response))

        return selected.response, selected.model_id

    @staticmethod
    def _consensus(
        groups: list[list[ModelOutput]],
        outputs: list[ModelOutput],
    ) -> tuple[str, str]:
        """Find consensus response (most similar to all others)."""
        if not outputs:
            return "", ""

        calculator = SimilarityCalculator()
        best_output = None
        best_avg_sim = -1

        for output in outputs:
            sims = [
                calculator.calculate(output.response, o.response) for o in outputs if o != output
            ]
            avg_sim = statistics.mean(sims) if sims else 0

            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_output = output

        if best_output:
            return best_output.response, best_output.model_id
        return outputs[0].response, outputs[0].model_id

    @staticmethod
    def _longest(outputs: list[ModelOutput]) -> tuple[str, str]:
        """Select longest response."""
        if not outputs:
            return "", ""
        selected = max(outputs, key=lambda x: len(x.response))
        return selected.response, selected.model_id

    @staticmethod
    def _shortest(outputs: list[ModelOutput]) -> tuple[str, str]:
        """Select shortest response."""
        if not outputs:
            return "", ""
        selected = min(outputs, key=lambda x: len(x.response))
        return selected.response, selected.model_id

    @staticmethod
    def _most_confident(outputs: list[ModelOutput]) -> tuple[str, str]:
        """Select most confident response."""
        if not outputs:
            return "", ""
        selected = max(outputs, key=lambda x: x.confidence)
        return selected.response, selected.model_id

    def _diverse_selection(
        self,
        outputs: list[ModelOutput],
    ) -> tuple[str, str]:
        """Select response that maximizes diversity coverage."""
        if not outputs:
            return "", ""

        # Select response most different from others (for diversity in selection)
        best_output = None
        best_diversity = -1

        for output in outputs:
            sims = [
                self._calculator.calculate(output.response, o.response)
                for o in outputs
                if o != output
            ]
            # Lower average similarity = more diverse
            diversity = 1 - (statistics.mean(sims) if sims else 0)

            if diversity > best_diversity:
                best_diversity = diversity
                best_output = output

        if best_output:
            return best_output.response, best_output.model_id
        return outputs[0].response, outputs[0].model_id


class ModelAgreementAnalyzer:
    """Analyze agreement between models."""

    def __init__(
        self,
        similarity_calculator: SimilarityCalculator | None = None,
    ):
        """Initialize analyzer.

        Args:
            similarity_calculator: Calculator for response similarity
        """
        self._calculator = similarity_calculator or SimilarityCalculator()

    def analyze(
        self,
        outputs: list[ModelOutput],
    ) -> dict[str, Any]:
        """Analyze agreement between model outputs.

        Args:
            outputs: List of model outputs

        Returns:
            Dictionary with agreement analysis
        """
        if len(outputs) < 2:
            return {
                "n_models": len(outputs),
                "overall_agreement": 1.0,
                "agreement_matrix": {},
                "clusters": [],
            }

        # Build agreement matrix
        matrix: dict[tuple[str, str], float] = {}
        for i, out1 in enumerate(outputs):
            for out2 in outputs[i + 1 :]:
                sim = self._calculator.calculate(out1.response, out2.response)
                matrix[(out1.model_id, out2.model_id)] = sim
                matrix[(out2.model_id, out1.model_id)] = sim

        # Calculate overall agreement
        overall = statistics.mean(matrix.values()) if matrix else 1.0

        # Find most/least agreeable models
        model_agreements: dict[str, list[float]] = defaultdict(list)
        for (m1, _m2), sim in matrix.items():
            model_agreements[m1].append(sim)

        avg_agreements = {m: statistics.mean(sims) for m, sims in model_agreements.items()}

        return {
            "n_models": len(outputs),
            "overall_agreement": overall,
            "agreement_matrix": {f"{k[0]}-{k[1]}": v for k, v in matrix.items()},
            "per_model_agreement": avg_agreements,
            "most_agreeable": max(avg_agreements.items(), key=lambda x: x[1])[0]
            if avg_agreements
            else None,
            "least_agreeable": min(avg_agreements.items(), key=lambda x: x[1])[0]
            if avg_agreements
            else None,
        }


class EnsembleEvaluator:
    """Evaluate ensemble model performance."""

    def __init__(
        self,
        aggregator: ResponseAggregator | None = None,
        analyzer: ModelAgreementAnalyzer | None = None,
    ):
        """Initialize evaluator.

        Args:
            aggregator: Response aggregator
            analyzer: Agreement analyzer
        """
        self._aggregator = aggregator or ResponseAggregator()
        self._analyzer = analyzer or ModelAgreementAnalyzer()

    def evaluate(
        self,
        prompt_outputs: list[list[ModelOutput]],
        method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
        scorer: Callable[[str], float] | None = None,
    ) -> EnsembleReport:
        """Evaluate ensemble across multiple prompts.

        Args:
            prompt_outputs: List of outputs per prompt
            method: Aggregation method
            scorer: Optional scoring function

        Returns:
            EnsembleReport object
        """
        if not prompt_outputs:
            return self._empty_report(method)

        # Collect all model IDs
        all_models = set()
        for outputs in prompt_outputs:
            for output in outputs:
                all_models.add(output.model_id)

        model_ids = sorted(all_models)

        # Track selection counts and agreements
        selection_counts: dict[str, int] = Counter()
        all_agreements = []
        per_model_agreements: dict[str, list[float]] = defaultdict(list)

        for outputs in prompt_outputs:
            # Aggregate
            aggregated = self._aggregator.aggregate(outputs, method, scorer)
            if aggregated.selected_model:
                selection_counts[aggregated.selected_model] += 1

            # Analyze agreement
            analysis = self._analyzer.analyze(outputs)
            all_agreements.append(analysis["overall_agreement"])

            for model, agreement in analysis.get("per_model_agreement", {}).items():
                per_model_agreements[model].append(agreement)

        # Calculate metrics
        n_prompts = len(prompt_outputs)
        selection_rates = {m: count / n_prompts for m, count in selection_counts.items()}
        avg_model_agreements = {
            m: statistics.mean(agreements) for m, agreements in per_model_agreements.items()
        }

        # Calculate diversity
        diversity = self._calculate_diversity(prompt_outputs)

        # Find best models
        best_model = max(selection_rates.items(), key=lambda x: x[1])[0] if selection_rates else ""
        most_agreeable = (
            max(avg_model_agreements.items(), key=lambda x: x[1])[0] if avg_model_agreements else ""
        )

        return EnsembleReport(
            n_prompts=n_prompts,
            n_models=len(model_ids),
            model_ids=model_ids,
            aggregation_method=method,
            overall_agreement=statistics.mean(all_agreements) if all_agreements else 0.0,
            per_model_selection_rate=selection_rates,
            per_model_agreement=avg_model_agreements,
            best_performing_model=best_model,
            most_agreeable_model=most_agreeable,
            ensemble_diversity=diversity,
            recommendations=self._generate_recommendations(
                selection_rates, avg_model_agreements, diversity
            ),
        )

    def _calculate_diversity(
        self,
        prompt_outputs: list[list[ModelOutput]],
    ) -> float:
        """Calculate response diversity across prompts."""
        calculator = SimilarityCalculator()
        diversities = []

        for outputs in prompt_outputs:
            if len(outputs) < 2:
                continue

            sims = []
            for i, out1 in enumerate(outputs):
                for out2 in outputs[i + 1 :]:
                    sims.append(calculator.calculate(out1.response, out2.response))

            # Diversity = 1 - average similarity
            if sims:
                diversities.append(1 - statistics.mean(sims))

        return statistics.mean(diversities) if diversities else 0.0

    @staticmethod
    def _generate_recommendations(
        selection_rates: dict[str, float],
        agreements: dict[str, float],
        diversity: float,
    ) -> list[str]:
        """Generate recommendations based on evaluation."""
        recommendations = []

        # Check for dominant model
        if selection_rates:
            max_rate = max(selection_rates.values())
            if max_rate > 0.8:
                dominant = [m for m, r in selection_rates.items() if r > 0.8][0]
                recommendations.append(
                    f"Model '{dominant}' dominates selection; "
                    f"consider using it alone for efficiency"
                )

        # Check for low agreement
        if agreements:
            min_agreement = min(agreements.values())
            if min_agreement < 0.3:
                outlier = [m for m, a in agreements.items() if a < 0.3][0]
                recommendations.append(
                    f"Model '{outlier}' shows low agreement; "
                    f"verify its outputs or remove from ensemble"
                )

        # Check diversity
        if diversity < 0.1:
            recommendations.append("Low ensemble diversity; models produce similar outputs")
        elif diversity > 0.7:
            recommendations.append("High ensemble diversity; consider using consensus method")

        if not recommendations:
            recommendations.append("Ensemble appears well-balanced")

        return recommendations

    @staticmethod
    def _empty_report(method: AggregationMethod) -> EnsembleReport:
        """Create empty report."""
        return EnsembleReport(
            n_prompts=0,
            n_models=0,
            model_ids=[],
            aggregation_method=method,
            overall_agreement=0.0,
            per_model_selection_rate={},
            per_model_agreement={},
            best_performing_model="",
            most_agreeable_model="",
            ensemble_diversity=0.0,
            recommendations=["No data to evaluate"],
        )


class ModelEnsemble:
    """Manage and query an ensemble of models."""

    def __init__(
        self,
        models: dict[str, Callable[[str], str]],
        aggregator: ResponseAggregator | None = None,
        default_method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
    ):
        """Initialize ensemble.

        Args:
            models: Dictionary of model_id -> response function
            aggregator: Response aggregator
            default_method: Default aggregation method
        """
        self._models = models
        self._aggregator = aggregator or ResponseAggregator()
        self._default_method = default_method

    def query(
        self,
        prompt: str,
        method: AggregationMethod | None = None,
        scorer: Callable[[str], float] | None = None,
    ) -> AggregatedOutput:
        """Query the ensemble.

        Args:
            prompt: Prompt to send to models
            method: Aggregation method
            scorer: Optional scoring function

        Returns:
            AggregatedOutput object
        """
        method = method or self._default_method

        # Get outputs from all models
        outputs = []
        for model_id, model_fn in self._models.items():
            try:
                response = model_fn(prompt)
                outputs.append(
                    ModelOutput(
                        model_id=model_id,
                        response=response,
                    )
                )
            except Exception:
                # Skip failed models
                continue

        return self._aggregator.aggregate(outputs, method, scorer)

    def compare_methods(
        self,
        prompt: str,
        methods: list[AggregationMethod] | None = None,
    ) -> dict[AggregationMethod, AggregatedOutput]:
        """Compare different aggregation methods.

        Args:
            prompt: Prompt to test
            methods: Methods to compare

        Returns:
            Dictionary of method -> output
        """
        if methods is None:
            methods = list(AggregationMethod)

        # Get outputs once
        outputs = []
        for model_id, model_fn in self._models.items():
            try:
                response = model_fn(prompt)
                outputs.append(ModelOutput(model_id=model_id, response=response))
            except Exception:
                continue

        # Apply each method
        results = {}
        for method in methods:
            results[method] = self._aggregator.aggregate(outputs, method)

        return results


# Convenience functions


def create_ensemble(
    models: dict[str, Callable[[str], str]],
    method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
) -> ModelEnsemble:
    """Create a model ensemble.

    Args:
        models: Dictionary of model_id -> response function
        method: Default aggregation method

    Returns:
        ModelEnsemble object
    """
    return ModelEnsemble(models, default_method=method)


def aggregate_responses(
    outputs: list[ModelOutput],
    method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
) -> AggregatedOutput:
    """Aggregate model outputs.

    Args:
        outputs: List of model outputs
        method: Aggregation method

    Returns:
        AggregatedOutput object
    """
    aggregator = ResponseAggregator()
    return aggregator.aggregate(outputs, method)


def analyze_model_agreement(
    outputs: list[ModelOutput],
) -> dict[str, Any]:
    """Analyze agreement between models.

    Args:
        outputs: List of model outputs

    Returns:
        Dictionary with agreement analysis
    """
    analyzer = ModelAgreementAnalyzer()
    return analyzer.analyze(outputs)


def evaluate_ensemble(
    prompt_outputs: list[list[ModelOutput]],
    method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
) -> EnsembleReport:
    """Evaluate ensemble performance.

    Args:
        prompt_outputs: List of outputs per prompt
        method: Aggregation method

    Returns:
        EnsembleReport object
    """
    evaluator = EnsembleEvaluator()
    return evaluator.evaluate(prompt_outputs, method)


def quick_ensemble_check(
    models: dict[str, Callable[[str], str]],
    prompts: list[str],
) -> dict[str, Any]:
    """Quick ensemble performance check.

    Args:
        models: Dictionary of model_id -> response function
        prompts: List of prompts to test

    Returns:
        Dictionary with quick check results
    """
    ensemble = ModelEnsemble(models)

    results = []
    for prompt in prompts:
        output = ensemble.query(prompt)
        results.append(
            {
                "agreement": output.agreement_score,
                "selected": output.selected_model,
            }
        )

    agreements = [r["agreement"] for r in results]
    selections = Counter(r["selected"] for r in results if r["selected"])

    return {
        "n_prompts": len(prompts),
        "n_models": len(models),
        "avg_agreement": statistics.mean(agreements) if agreements else 0.0,
        "selection_distribution": dict(selections),
        "most_selected": selections.most_common(1)[0][0] if selections else None,
    }
