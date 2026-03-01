"""Multi-model ensemble evaluation for LLM comparison.

This module provides comprehensive tools for evaluating and combining outputs
from multiple language models to produce more reliable, consistent, and
diverse responses. It supports various aggregation strategies, agreement
analysis, and ensemble performance evaluation.

Key Features
------------
- **Response Aggregation**: Combine outputs using majority voting, weighted
  voting, consensus, or diversity-based selection methods.
- **Confidence-weighted Ensembles**: Weight model outputs by confidence scores
  for more intelligent aggregation.
- **Model Agreement Analysis**: Analyze how well models agree with each other
  and identify outliers or consensus responses.
- **Diversity-based Selection**: Select responses that maximize output diversity
  or find the most representative consensus.
- **Ensemble Performance Evaluation**: Generate comprehensive reports on
  ensemble behavior across multiple prompts.

Main Classes
------------
- :class:`ModelOutput`: Container for a single model's response.
- :class:`AggregatedOutput`: Result of aggregating multiple model outputs.
- :class:`ResponseAggregator`: Aggregates responses using various methods.
- :class:`ModelAgreementAnalyzer`: Analyzes agreement between model outputs.
- :class:`EnsembleEvaluator`: Evaluates ensemble performance across prompts.
- :class:`ModelEnsemble`: Manages and queries an ensemble of models.

Examples
--------
Basic ensemble creation and querying:

>>> from insideLLMs.contrib.ensemble import ModelEnsemble, AggregationMethod
>>> def model_a(prompt):
...     return "The answer is 42."
>>> def model_b(prompt):
...     return "The answer is 42."
>>> def model_c(prompt):
...     return "I believe the answer is forty-two."
>>> ensemble = ModelEnsemble(
...     models={"model_a": model_a, "model_b": model_b, "model_c": model_c},
...     default_method=AggregationMethod.MAJORITY_VOTE
... )
>>> result = ensemble.query("What is the answer?")
>>> print(result.agreement_level)
AgreementLevel.STRONG

Aggregating pre-collected responses:

>>> from insideLLMs.contrib.ensemble import ModelOutput, aggregate_responses
>>> outputs = [
...     ModelOutput(model_id="gpt-4", response="Paris", confidence=0.95),
...     ModelOutput(model_id="claude", response="Paris", confidence=0.92),
...     ModelOutput(model_id="llama", response="Paris, France", confidence=0.88),
... ]
>>> result = aggregate_responses(outputs, AggregationMethod.WEIGHTED_VOTE)
>>> print(f"Selected: {result.selected_model}, Agreement: {result.agreement_score:.2f}")
Selected: gpt-4, Agreement: 0.78

Analyzing model agreement:

>>> from insideLLMs.contrib.ensemble import analyze_model_agreement, ModelOutput
>>> outputs = [
...     ModelOutput(model_id="model_1", response="The sky is blue."),
...     ModelOutput(model_id="model_2", response="The sky appears blue."),
...     ModelOutput(model_id="model_3", response="Water is wet."),
... ]
>>> analysis = analyze_model_agreement(outputs)
>>> print(f"Overall agreement: {analysis['overall_agreement']:.2f}")
>>> print(f"Most agreeable: {analysis['most_agreeable']}")
Overall agreement: 0.45
Most agreeable: model_1

Quick ensemble health check:

>>> from insideLLMs.contrib.ensemble import quick_ensemble_check
>>> models = {"fast_model": lambda p: "quick response", "slow_model": lambda p: "detailed response"}
>>> prompts = ["Test 1", "Test 2", "Test 3"]
>>> check = quick_ensemble_check(models, prompts)
>>> print(f"Average agreement: {check['avg_agreement']:.2f}")
>>> print(f"Most selected: {check['most_selected']}")

See Also
--------
- :mod:`insideLLMs.trace` : For tracing individual model calls.
- :mod:`insideLLMs.metrics` : For computing quality metrics on outputs.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class AggregationMethod(Enum):
    """Methods for aggregating model outputs into a single response.

    This enum defines the available strategies for combining multiple model
    outputs into a final aggregated response. Each method has different
    characteristics suitable for different use cases.

    Attributes
    ----------
    MAJORITY_VOTE : str
        Select the response that appears most frequently (by similarity grouping).
        Best for classification tasks or when models should agree.
    WEIGHTED_VOTE : str
        Like majority vote, but weights each response by its confidence score.
        Best when confidence scores are meaningful and calibrated.
    BEST_OF_N : str
        Select the best response according to a custom scoring function.
        Best when you have a quality metric to optimize.
    CONSENSUS : str
        Select the response most similar to all others (centroid).
        Best for finding a representative middle-ground response.
    LONGEST : str
        Select the longest response by character count.
        Best when more detail is preferred.
    SHORTEST : str
        Select the shortest response by character count.
        Best when conciseness is preferred.
    MOST_CONFIDENT : str
        Select the response with the highest confidence score.
        Best when confidence scores reliably indicate quality.
    DIVERSE_SELECTION : str
        Select the response most different from others.
        Best when you want to maximize coverage or uniqueness.

    Examples
    --------
    Using different aggregation methods:

    >>> from insideLLMs.contrib.ensemble import AggregationMethod, ResponseAggregator, ModelOutput
    >>> outputs = [
    ...     ModelOutput(model_id="a", response="Yes", confidence=0.9),
    ...     ModelOutput(model_id="b", response="Yes", confidence=0.8),
    ...     ModelOutput(model_id="c", response="No", confidence=0.95),
    ... ]
    >>> aggregator = ResponseAggregator()

    Majority vote selects "Yes" (2 vs 1):

    >>> result = aggregator.aggregate(outputs, AggregationMethod.MAJORITY_VOTE)
    >>> result.final_response
    'Yes'

    Weighted vote considers confidence:

    >>> result = aggregator.aggregate(outputs, AggregationMethod.WEIGHTED_VOTE)
    >>> result.final_response  # "Yes" wins with 0.9 + 0.8 = 1.7 vs 0.95
    'Yes'

    Most confident selects "No" (0.95 confidence):

    >>> result = aggregator.aggregate(outputs, AggregationMethod.MOST_CONFIDENT)
    >>> result.final_response
    'No'

    Iterating over all methods:

    >>> for method in AggregationMethod:
    ...     print(f"{method.name}: {method.value}")
    MAJORITY_VOTE: majority_vote
    WEIGHTED_VOTE: weighted_vote
    ...
    """

    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    BEST_OF_N = "best_of_n"
    CONSENSUS = "consensus"
    LONGEST = "longest"
    SHORTEST = "shortest"
    MOST_CONFIDENT = "most_confident"
    DIVERSE_SELECTION = "diverse_selection"


class AgreementLevel(Enum):
    """Categorical levels of agreement between model outputs.

    This enum provides human-readable labels for agreement scores, making it
    easier to interpret ensemble behavior at a glance. Agreement levels are
    determined by thresholding the continuous agreement score.

    Attributes
    ----------
    UNANIMOUS : str
        All models produced essentially identical responses (score >= 0.95).
        Indicates very high confidence in the aggregated result.
    STRONG : str
        Most models agree with minor variations (0.75 <= score < 0.95).
        Indicates good confidence; small wording differences exist.
    MODERATE : str
        Models show partial agreement (0.50 <= score < 0.75).
        Some models diverge; review may be warranted.
    WEAK : str
        Models show limited agreement (0.25 <= score < 0.50).
        Significant disagreement; aggregated result less reliable.
    NONE : str
        Models show no meaningful agreement (score < 0.25).
        High diversity or conflict; manual review recommended.

    Examples
    --------
    Checking agreement levels after aggregation:

    >>> from insideLLMs.contrib.ensemble import aggregate_responses, ModelOutput, AgreementLevel
    >>> outputs = [
    ...     ModelOutput(model_id="a", response="The capital is Paris."),
    ...     ModelOutput(model_id="b", response="The capital is Paris."),
    ...     ModelOutput(model_id="c", response="Paris is the capital."),
    ... ]
    >>> result = aggregate_responses(outputs)
    >>> result.agreement_level
    AgreementLevel.STRONG

    Using agreement level for decision making:

    >>> if result.agreement_level in (AgreementLevel.UNANIMOUS, AgreementLevel.STRONG):
    ...     print("High confidence result")
    ... elif result.agreement_level == AgreementLevel.MODERATE:
    ...     print("Consider additional verification")
    ... else:
    ...     print("Low confidence - manual review needed")
    High confidence result

    Converting to string for logging:

    >>> result.agreement_level.value
    'strong'

    Comparing agreement levels:

    >>> AgreementLevel.UNANIMOUS != AgreementLevel.NONE
    True
    """

    UNANIMOUS = "unanimous"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class EnsembleStrategy(Enum):
    """Strategies for selecting which models to include in an ensemble.

    This enum defines strategies for choosing a subset of available models
    to participate in ensemble queries. Different strategies optimize for
    different goals like performance, diversity, or cost.

    Attributes
    ----------
    ALL : str
        Include all available models in the ensemble.
        Maximizes information but may be slow or expensive.
    TOP_K : str
        Include only the top K performing models based on historical metrics.
        Balances quality with efficiency.
    THRESHOLD : str
        Include models whose performance exceeds a minimum threshold.
        Ensures minimum quality while adapting to available models.
    DIVERSE : str
        Select models that maximize output diversity.
        Reduces correlation and increases coverage of solution space.
    RANDOM_SUBSET : str
        Randomly select a subset of models for each query.
        Useful for load balancing or experimentation.

    Examples
    --------
    Defining ensemble selection behavior:

    >>> from insideLLMs.contrib.ensemble import EnsembleStrategy
    >>> strategy = EnsembleStrategy.TOP_K
    >>> print(f"Using strategy: {strategy.value}")
    Using strategy: top_k

    Strategy-based model filtering (conceptual):

    >>> strategy = EnsembleStrategy.THRESHOLD
    >>> min_accuracy = 0.85
    >>> available_models = {"gpt-4": 0.92, "claude": 0.89, "llama": 0.78}
    >>> if strategy == EnsembleStrategy.THRESHOLD:
    ...     selected = {k: v for k, v in available_models.items() if v >= min_accuracy}
    ...     print(f"Selected models: {list(selected.keys())}")
    Selected models: ['gpt-4', 'claude']

    Checking strategy type:

    >>> EnsembleStrategy.DIVERSE.name
    'DIVERSE'
    >>> EnsembleStrategy.ALL == EnsembleStrategy.ALL
    True
    """

    ALL = "all"
    TOP_K = "top_k"
    THRESHOLD = "threshold"
    DIVERSE = "diverse"
    RANDOM_SUBSET = "random_subset"


@dataclass
class ModelOutput:
    """Container for a single model's response to a prompt.

    This dataclass captures the essential information about a model's output,
    including the response text, confidence score, timing information, and
    any additional metadata. It serves as the input to aggregation functions.

    Attributes
    ----------
    model_id : str
        Unique identifier for the model that produced this output.
        Used to track which models contribute to ensemble decisions.
    response : str
        The text response generated by the model.
    confidence : float, optional
        Confidence score between 0 and 1 (default: 1.0).
        Used by weighted aggregation methods. Higher values indicate
        the model is more certain about its response.
    latency : float, optional
        Time in seconds the model took to generate the response (default: 0.0).
        Useful for performance analysis and timeout handling.
    metadata : dict[str, Any], optional
        Additional model-specific information (default: empty dict).
        Can store token counts, model version, temperature, etc.

    Examples
    --------
    Creating a basic model output:

    >>> from insideLLMs.contrib.ensemble import ModelOutput
    >>> output = ModelOutput(
    ...     model_id="gpt-4",
    ...     response="The answer is 42."
    ... )
    >>> output.model_id
    'gpt-4'
    >>> output.confidence  # default value
    1.0

    Creating output with all fields:

    >>> output = ModelOutput(
    ...     model_id="claude-3-opus",
    ...     response="Paris is the capital of France.",
    ...     confidence=0.95,
    ...     latency=1.23,
    ...     metadata={"tokens": 150, "temperature": 0.7}
    ... )
    >>> output.latency
    1.23
    >>> output.metadata["tokens"]
    150

    Converting to dictionary for serialization:

    >>> output_dict = output.to_dict()
    >>> print(output_dict["model_id"])
    claude-3-opus
    >>> "confidence" in output_dict
    True

    Using in a list for aggregation:

    >>> outputs = [
    ...     ModelOutput(model_id="a", response="Yes", confidence=0.9),
    ...     ModelOutput(model_id="b", response="Yes", confidence=0.8),
    ...     ModelOutput(model_id="c", response="No", confidence=0.7),
    ... ]
    >>> len(outputs)
    3

    See Also
    --------
    AggregatedOutput : The result of combining multiple ModelOutputs.
    ResponseAggregator : Combines ModelOutputs into aggregated results.
    """

    model_id: str
    response: str
    confidence: float = 1.0
    latency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the ModelOutput to a dictionary representation.

        Creates a serializable dictionary suitable for JSON output, logging,
        or storage. The response is truncated to 500 characters to prevent
        excessive output sizes.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all fields with the response truncated.

        Examples
        --------
        >>> output = ModelOutput(
        ...     model_id="test-model",
        ...     response="A short response.",
        ...     confidence=0.85,
        ...     latency=0.5,
        ...     metadata={"version": "1.0"}
        ... )
        >>> d = output.to_dict()
        >>> d["model_id"]
        'test-model'
        >>> d["confidence"]
        0.85

        Long responses are truncated:

        >>> long_output = ModelOutput(
        ...     model_id="verbose-model",
        ...     response="x" * 1000
        ... )
        >>> len(long_output.to_dict()["response"])
        500
        """
        return {
            "model_id": self.model_id,
            "response": self.response[:500],
            "confidence": self.confidence,
            "latency": self.latency,
            "metadata": self.metadata,
        }


@dataclass
class AggregatedOutput:
    """Result of aggregating multiple model outputs into a single response.

    This dataclass contains the final aggregated response along with metadata
    about the aggregation process, including which model was selected, the
    level of agreement among models, and the vote distribution.

    Attributes
    ----------
    final_response : str
        The selected or synthesized response after aggregation.
    method : AggregationMethod
        The aggregation method used to produce this result.
    source_outputs : list[ModelOutput]
        The original model outputs that were aggregated.
    agreement_level : AgreementLevel
        Categorical assessment of how well models agreed.
    agreement_score : float
        Numerical agreement score between 0 (no agreement) and 1 (unanimous).
        Based on pairwise similarity of all responses.
    selected_model : Optional[str]
        The model_id whose response was selected, or None if synthesized.
    vote_distribution : dict[str, int]
        Distribution of votes across response groups.
        Keys are group indices, values are counts.

    Examples
    --------
    Accessing aggregation results:

    >>> from insideLLMs.contrib.ensemble import aggregate_responses, ModelOutput, AggregationMethod
    >>> outputs = [
    ...     ModelOutput(model_id="model_a", response="Paris"),
    ...     ModelOutput(model_id="model_b", response="Paris"),
    ...     ModelOutput(model_id="model_c", response="London"),
    ... ]
    >>> result = aggregate_responses(outputs, AggregationMethod.MAJORITY_VOTE)
    >>> result.final_response
    'Paris'
    >>> result.selected_model
    'model_a'
    >>> result.agreement_level
    AgreementLevel.MODERATE

    Checking agreement quality:

    >>> if result.agreement_score > 0.8:
    ...     print("High agreement - result is reliable")
    ... else:
    ...     print(f"Agreement: {result.agreement_score:.2f} - consider verification")
    Agreement: 0.67 - consider verification

    Examining source outputs:

    >>> len(result.source_outputs)
    3
    >>> [o.model_id for o in result.source_outputs]
    ['model_a', 'model_b', 'model_c']

    Serializing for logging:

    >>> result_dict = result.to_dict()
    >>> result_dict["method"]
    'majority_vote'
    >>> result_dict["n_models"]
    3

    See Also
    --------
    ModelOutput : Individual model responses before aggregation.
    ResponseAggregator : Class that produces AggregatedOutput instances.
    """

    final_response: str
    method: AggregationMethod
    source_outputs: list[ModelOutput]
    agreement_level: AgreementLevel
    agreement_score: float
    selected_model: Optional[str]
    vote_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert the AggregatedOutput to a dictionary representation.

        Creates a serializable dictionary suitable for JSON output, logging,
        or storage. The response is truncated to 500 characters and enum
        values are converted to strings.

        Returns
        -------
        dict[str, Any]
            Dictionary containing aggregation results with:
            - final_response (truncated to 500 chars)
            - method (string value)
            - n_models (count of source outputs)
            - agreement_level (string value)
            - agreement_score
            - selected_model
            - vote_distribution

        Examples
        --------
        >>> from insideLLMs.contrib.ensemble import aggregate_responses, ModelOutput
        >>> outputs = [
        ...     ModelOutput(model_id="a", response="Yes"),
        ...     ModelOutput(model_id="b", response="Yes"),
        ... ]
        >>> result = aggregate_responses(outputs)
        >>> d = result.to_dict()
        >>> d["method"]
        'majority_vote'
        >>> d["agreement_level"]
        'unanimous'
        >>> d["n_models"]
        2
        """
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
    """Detailed comparison of model outputs on a single task.

    This dataclass provides a comprehensive comparison of how different models
    performed on the same prompt, including rankings, pairwise agreement scores,
    and diversity metrics. Useful for model selection and evaluation.

    Attributes
    ----------
    prompt : str
        The prompt that was sent to all models.
    outputs : list[ModelOutput]
        The responses from each model.
    best_model : str
        The model_id of the best performing model (by score or selection).
    worst_model : str
        The model_id of the worst performing model.
    ranking : list[tuple[str, float]]
        Ordered list of (model_id, score) tuples from best to worst.
    agreement_matrix : dict[tuple[str, str], float]
        Pairwise similarity scores between models.
        Keys are (model_id_1, model_id_2) tuples.
    diversity_score : float
        Overall diversity of responses (0 = identical, 1 = completely different).

    Examples
    --------
    Creating a model comparison:

    >>> from insideLLMs.contrib.ensemble import ModelComparison, ModelOutput
    >>> comparison = ModelComparison(
    ...     prompt="What is 2+2?",
    ...     outputs=[
    ...         ModelOutput(model_id="a", response="4"),
    ...         ModelOutput(model_id="b", response="4"),
    ...         ModelOutput(model_id="c", response="Four"),
    ...     ],
    ...     best_model="a",
    ...     worst_model="c",
    ...     ranking=[("a", 1.0), ("b", 0.95), ("c", 0.8)],
    ...     agreement_matrix={("a", "b"): 1.0, ("a", "c"): 0.5, ("b", "c"): 0.5},
    ...     diversity_score=0.33,
    ... )
    >>> comparison.best_model
    'a'

    Examining the ranking:

    >>> for model, score in comparison.ranking:
    ...     print(f"{model}: {score:.2f}")
    a: 1.00
    b: 0.95
    c: 0.80

    Checking pairwise agreement:

    >>> comparison.agreement_matrix[("a", "b")]
    1.0
    >>> comparison.agreement_matrix[("a", "c")]
    0.5

    Serializing for reporting:

    >>> d = comparison.to_dict()
    >>> d["diversity_score"]
    0.33
    >>> d["n_models"]
    3
    """

    prompt: str
    outputs: list[ModelOutput]
    best_model: str
    worst_model: str
    ranking: list[tuple[str, float]]
    agreement_matrix: dict[tuple[str, str], float]
    diversity_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the ModelComparison to a dictionary representation.

        Creates a serializable dictionary suitable for JSON output or storage.
        The prompt is truncated to 200 characters and the agreement matrix
        is excluded (use agreement_matrix attribute directly if needed).

        Returns
        -------
        dict[str, Any]
            Dictionary containing comparison summary with:
            - prompt (truncated to 200 chars)
            - n_models (count of outputs)
            - best_model
            - worst_model
            - ranking (list of [model_id, score] pairs)
            - diversity_score

        Examples
        --------
        >>> from insideLLMs.contrib.ensemble import ModelComparison, ModelOutput
        >>> comparison = ModelComparison(
        ...     prompt="A very long prompt..." * 50,
        ...     outputs=[ModelOutput(model_id="x", response="test")],
        ...     best_model="x",
        ...     worst_model="x",
        ...     ranking=[("x", 1.0)],
        ...     agreement_matrix={},
        ...     diversity_score=0.0,
        ... )
        >>> d = comparison.to_dict()
        >>> len(d["prompt"]) <= 200
        True
        >>> d["n_models"]
        1
        """
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
    """Comprehensive report on ensemble performance across multiple prompts.

    This dataclass summarizes ensemble behavior over a batch of prompts,
    providing insights into model selection patterns, agreement levels,
    diversity, and actionable recommendations for ensemble optimization.

    Attributes
    ----------
    n_prompts : int
        Number of prompts evaluated.
    n_models : int
        Number of models in the ensemble.
    model_ids : list[str]
        List of all model identifiers in the ensemble.
    aggregation_method : AggregationMethod
        The aggregation method used for evaluation.
    overall_agreement : float
        Average agreement score across all prompts (0-1).
    per_model_selection_rate : dict[str, float]
        Fraction of prompts where each model was selected.
        Values sum to 1.0 (or less if some had no selection).
    per_model_agreement : dict[str, float]
        Average agreement score for each model with others.
    best_performing_model : str
        Model with highest selection rate.
    most_agreeable_model : str
        Model that agrees most with other models on average.
    ensemble_diversity : float
        Overall diversity score (0 = all same, 1 = all different).
    recommendations : list[str]
        Actionable suggestions for improving ensemble performance.

    Examples
    --------
    Generating and examining an ensemble report:

    >>> from insideLLMs.contrib.ensemble import evaluate_ensemble, ModelOutput, AggregationMethod
    >>> # Simulate outputs across multiple prompts
    >>> prompt_outputs = [
    ...     [
    ...         ModelOutput(model_id="gpt-4", response="Paris"),
    ...         ModelOutput(model_id="claude", response="Paris"),
    ...     ],
    ...     [
    ...         ModelOutput(model_id="gpt-4", response="Blue"),
    ...         ModelOutput(model_id="claude", response="Azure"),
    ...     ],
    ... ]
    >>> report = evaluate_ensemble(prompt_outputs)
    >>> report.n_prompts
    2
    >>> report.n_models
    2

    Analyzing model selection patterns:

    >>> for model, rate in report.per_model_selection_rate.items():
    ...     print(f"{model}: selected {rate*100:.1f}% of the time")

    Reviewing recommendations:

    >>> for rec in report.recommendations:
    ...     print(f"- {rec}")

    Checking overall ensemble health:

    >>> if report.overall_agreement > 0.8:
    ...     print("High agreement - consider reducing ensemble size")
    >>> if report.ensemble_diversity < 0.2:
    ...     print("Low diversity - models produce similar outputs")

    Serializing for export:

    >>> d = report.to_dict()
    >>> d["aggregation_method"]
    'majority_vote'
    """

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
        """Convert the EnsembleReport to a dictionary representation.

        Creates a serializable dictionary suitable for JSON export, logging,
        or database storage. All enum values are converted to strings.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all report fields with:
            - Numerical metrics (n_prompts, n_models, scores)
            - Model lists and selection/agreement rates
            - Recommendations as list of strings
            - aggregation_method as string value

        Examples
        --------
        >>> from insideLLMs.contrib.ensemble import EnsembleReport, AggregationMethod
        >>> report = EnsembleReport(
        ...     n_prompts=100,
        ...     n_models=3,
        ...     model_ids=["a", "b", "c"],
        ...     aggregation_method=AggregationMethod.MAJORITY_VOTE,
        ...     overall_agreement=0.75,
        ...     per_model_selection_rate={"a": 0.5, "b": 0.3, "c": 0.2},
        ...     per_model_agreement={"a": 0.8, "b": 0.7, "c": 0.75},
        ...     best_performing_model="a",
        ...     most_agreeable_model="a",
        ...     ensemble_diversity=0.4,
        ...     recommendations=["Ensemble appears well-balanced"],
        ... )
        >>> d = report.to_dict()
        >>> d["n_prompts"]
        100
        >>> d["overall_agreement"]
        0.75
        >>> len(d["recommendations"])
        1
        """
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
    """Normalize text responses for consistent comparison.

    This class applies configurable text transformations to normalize
    responses before comparison. Normalization ensures that superficial
    differences (like capitalization or extra spaces) don't affect
    similarity calculations.

    Attributes
    ----------
    lowercase : bool
        Whether to convert text to lowercase.
    strip_whitespace : bool
        Whether to collapse multiple spaces to single space.
    remove_punctuation : bool
        Whether to remove punctuation characters.

    Examples
    --------
    Basic normalization with default settings:

    >>> from insideLLMs.contrib.ensemble import ResponseNormalizer
    >>> normalizer = ResponseNormalizer()
    >>> normalizer.normalize("  Hello   WORLD!  ")
    'hello world!'

    Case-sensitive comparison:

    >>> normalizer = ResponseNormalizer(lowercase=False)
    >>> normalizer.normalize("Hello World")
    'Hello World'

    Removing punctuation for strict comparison:

    >>> normalizer = ResponseNormalizer(remove_punctuation=True)
    >>> normalizer.normalize("Hello, World!")
    'hello world'

    Full normalization pipeline:

    >>> normalizer = ResponseNormalizer(
    ...     lowercase=True,
    ...     strip_whitespace=True,
    ...     remove_punctuation=True
    ... )
    >>> normalizer.normalize("  HELLO,   World!!!  ")
    'hello world'
    """

    def __init__(
        self,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        remove_punctuation: bool = False,
    ):
        """Initialize the response normalizer with configuration options.

        Args
        ----
        lowercase : bool, optional
            Convert text to lowercase (default: True).
            Enables case-insensitive comparison.
        strip_whitespace : bool, optional
            Collapse multiple whitespace characters to single space and
            strip leading/trailing whitespace (default: True).
        remove_punctuation : bool, optional
            Remove all punctuation characters (default: False).
            Use with caution as it may affect meaning.

        Examples
        --------
        >>> normalizer = ResponseNormalizer()
        >>> normalizer.lowercase
        True

        >>> normalizer = ResponseNormalizer(lowercase=False, remove_punctuation=True)
        >>> normalizer.remove_punctuation
        True

        >>> normalizer = ResponseNormalizer(
        ...     lowercase=True,
        ...     strip_whitespace=True,
        ...     remove_punctuation=True
        ... )
        """
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.remove_punctuation = remove_punctuation

    def normalize(self, response: str) -> str:
        """Normalize a response string by applying configured transformations.

        Applies the normalization steps in order: lowercase, whitespace
        stripping, then punctuation removal (if enabled).

        Args
        ----
        response : str
            The raw response text to normalize.

        Returns
        -------
        str
            The normalized response text.

        Examples
        --------
        >>> normalizer = ResponseNormalizer()
        >>> normalizer.normalize("HELLO")
        'hello'

        >>> normalizer.normalize("  multiple   spaces  ")
        'multiple spaces'

        >>> normalizer = ResponseNormalizer(remove_punctuation=True)
        >>> normalizer.normalize("Hello, World!")
        'hello world'

        Empty string handling:

        >>> normalizer = ResponseNormalizer()
        >>> normalizer.normalize("")
        ''
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


class TextSimilarityCalculator:
    """Calculate similarity between text responses using Jaccard similarity.

    This class computes text similarity using word-level Jaccard similarity
    after normalization. Jaccard similarity measures the overlap between
    two sets as |intersection| / |union|, ranging from 0 (no overlap) to
    1 (identical sets).

    Attributes
    ----------
    _normalizer : ResponseNormalizer
        The normalizer used to preprocess text before comparison.

    Notes
    -----
    The Jaccard similarity is computed on word sets, not word sequences.
    This means "A B C" and "C B A" have similarity 1.0 (same words).
    For order-sensitive comparison, consider other similarity metrics.

    Examples
    --------
    Basic similarity calculation:

    >>> from insideLLMs.contrib.ensemble import TextSimilarityCalculator
    >>> calc = TextSimilarityCalculator()
    >>> calc.calculate("hello world", "hello world")
    1.0
    >>> calc.calculate("hello world", "hello there")
    0.5

    Similarity is case-insensitive by default:

    >>> calc.calculate("HELLO", "hello")
    1.0

    Completely different responses:

    >>> calc.calculate("cats are great", "dogs are fun")
    0.2

    Using a custom normalizer:

    >>> from insideLLMs.contrib.ensemble import ResponseNormalizer
    >>> strict_normalizer = ResponseNormalizer(remove_punctuation=True)
    >>> calc = TextSimilarityCalculator(normalizer=strict_normalizer)
    >>> calc.calculate("Hello!", "Hello")
    1.0
    """

    def __init__(
        self,
        normalizer: Optional[ResponseNormalizer] = None,
    ):
        """Initialize the similarity calculator.

        Args
        ----
        normalizer : Optional[ResponseNormalizer], optional
            Custom normalizer for preprocessing text (default: None).
            If None, a default ResponseNormalizer is created with
            lowercase=True and strip_whitespace=True.

        Examples
        --------
        >>> calc = TextSimilarityCalculator()
        >>> isinstance(calc._normalizer, ResponseNormalizer)
        True

        >>> from insideLLMs.contrib.ensemble import ResponseNormalizer
        >>> custom = ResponseNormalizer(lowercase=False)
        >>> calc = TextSimilarityCalculator(normalizer=custom)
        >>> calc._normalizer.lowercase
        False
        """
        self._normalizer = normalizer or ResponseNormalizer()

    def calculate(self, response1: str, response2: str) -> float:
        """Calculate Jaccard similarity between two responses.

        Normalizes both responses and computes word-level Jaccard similarity.
        The similarity score indicates the proportion of shared words relative
        to the total unique words in both responses.

        Args
        ----
        response1 : str
            The first response text.
        response2 : str
            The second response text.

        Returns
        -------
        float
            Similarity score between 0.0 (completely different) and
            1.0 (identical after normalization).

        Examples
        --------
        Identical responses:

        >>> calc = TextSimilarityCalculator()
        >>> calc.calculate("The sky is blue", "The sky is blue")
        1.0

        Partial overlap:

        >>> calc.calculate("the cat sat", "the dog sat")  # 2 shared / 4 total
        0.5

        Case insensitivity:

        >>> calc.calculate("HELLO WORLD", "hello world")
        1.0

        No overlap:

        >>> calc.calculate("apple banana", "cherry date")
        0.0

        Empty responses (both empty = identical):

        >>> calc.calculate("", "")
        1.0

        One empty (no overlap):

        >>> calc.calculate("hello", "")
        0.0
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
    """Aggregate multiple model responses into a single output.

    This class implements various strategies for combining responses from
    multiple language models into a single aggregated result. It handles
    similarity-based grouping, voting, and various selection methods.

    Attributes
    ----------
    _calculator : TextSimilarityCalculator
        Calculator for computing response similarity.
    _threshold : float
        Similarity threshold for grouping responses together.

    Examples
    --------
    Basic majority vote aggregation:

    >>> from insideLLMs.contrib.ensemble import ResponseAggregator, ModelOutput, AggregationMethod
    >>> aggregator = ResponseAggregator()
    >>> outputs = [
    ...     ModelOutput(model_id="a", response="Paris"),
    ...     ModelOutput(model_id="b", response="Paris"),
    ...     ModelOutput(model_id="c", response="London"),
    ... ]
    >>> result = aggregator.aggregate(outputs, AggregationMethod.MAJORITY_VOTE)
    >>> result.final_response
    'Paris'

    Weighted voting with confidence scores:

    >>> outputs = [
    ...     ModelOutput(model_id="a", response="Yes", confidence=0.9),
    ...     ModelOutput(model_id="b", response="No", confidence=0.95),
    ...     ModelOutput(model_id="c", response="Yes", confidence=0.85),
    ... ]
    >>> result = aggregator.aggregate(outputs, AggregationMethod.WEIGHTED_VOTE)
    >>> result.final_response  # "Yes" wins (0.9 + 0.85 = 1.75 vs 0.95)
    'Yes'

    Using a custom scorer for best-of-n selection:

    >>> def length_scorer(response: str) -> float:
    ...     return len(response)
    >>> outputs = [
    ...     ModelOutput(model_id="a", response="Short"),
    ...     ModelOutput(model_id="b", response="A longer response"),
    ... ]
    >>> result = aggregator.aggregate(outputs, AggregationMethod.BEST_OF_N, scorer=length_scorer)
    >>> result.final_response
    'A longer response'

    Custom similarity threshold:

    >>> aggregator = ResponseAggregator(similarity_threshold=0.5)
    >>> # Responses with >= 50% word overlap are grouped together
    """

    def __init__(
        self,
        similarity_calculator: Optional[TextSimilarityCalculator] = None,
        similarity_threshold: float = 0.8,
    ):
        """Initialize the response aggregator.

        Args
        ----
        similarity_calculator : Optional[TextSimilarityCalculator], optional
            Custom calculator for response similarity (default: None).
            If None, a default TextSimilarityCalculator is created.
        similarity_threshold : float, optional
            Threshold for considering two responses as similar (default: 0.8).
            Responses with similarity >= threshold are grouped together
            for voting purposes. Range: 0.0 to 1.0.

        Examples
        --------
        >>> aggregator = ResponseAggregator()
        >>> aggregator._threshold
        0.8

        >>> from insideLLMs.contrib.ensemble import TextSimilarityCalculator
        >>> custom_calc = TextSimilarityCalculator()
        >>> aggregator = ResponseAggregator(
        ...     similarity_calculator=custom_calc,
        ...     similarity_threshold=0.6
        ... )
        >>> aggregator._threshold
        0.6
        """
        self._calculator = similarity_calculator or TextSimilarityCalculator()
        self._threshold = similarity_threshold

    def aggregate(
        self,
        outputs: list[ModelOutput],
        method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
        scorer: Optional[Callable[[str], float]] = None,
    ) -> AggregatedOutput:
        """Aggregate model outputs using the specified method.

        Combines multiple model outputs into a single aggregated result.
        First calculates agreement metrics and groups similar responses,
        then applies the specified aggregation method to select or
        synthesize the final response.

        Args
        ----
        outputs : list[ModelOutput]
            List of model outputs to aggregate. Can be empty.
        method : AggregationMethod, optional
            The aggregation strategy to use (default: MAJORITY_VOTE).
            See :class:`AggregationMethod` for available options.
        scorer : Optional[Callable[[str], float]], optional
            Custom scoring function for BEST_OF_N method (default: None).
            If None and BEST_OF_N is used, defaults to selecting longest response.
            The function should take a response string and return a score.

        Returns
        -------
        AggregatedOutput
            The aggregated result containing:
            - final_response: The selected/synthesized response
            - method: The aggregation method used
            - source_outputs: Original inputs
            - agreement_level: Categorical agreement assessment
            - agreement_score: Numerical agreement (0-1)
            - selected_model: Which model's response was chosen
            - vote_distribution: How responses were grouped

        Examples
        --------
        Basic aggregation:

        >>> from insideLLMs.contrib.ensemble import ResponseAggregator, ModelOutput, AggregationMethod
        >>> aggregator = ResponseAggregator()
        >>> outputs = [
        ...     ModelOutput(model_id="m1", response="The answer is 4"),
        ...     ModelOutput(model_id="m2", response="The answer is 4"),
        ...     ModelOutput(model_id="m3", response="I think it is 5"),
        ... ]
        >>> result = aggregator.aggregate(outputs)
        >>> result.selected_model in ("m1", "m2")
        True
        >>> result.agreement_score > 0
        True

        Empty input handling:

        >>> result = aggregator.aggregate([])
        >>> result.final_response
        ''
        >>> result.agreement_level
        AgreementLevel.NONE

        Consensus method:

        >>> outputs = [
        ...     ModelOutput(model_id="a", response="cats and dogs"),
        ...     ModelOutput(model_id="b", response="dogs and cats"),
        ...     ModelOutput(model_id="c", response="birds and fish"),
        ... ]
        >>> result = aggregator.aggregate(outputs, AggregationMethod.CONSENSUS)
        >>> result.selected_model in ("a", "b")  # most similar to others
        True
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
        """Calculate overall agreement score across all model outputs.

        Computes the mean pairwise similarity between all outputs.
        Agreement of 1.0 means all outputs are identical (after normalization),
        while 0.0 means no overlap between any pair.

        Args
        ----
        outputs : list[ModelOutput]
            List of model outputs to analyze.

        Returns
        -------
        float
            Mean pairwise similarity score between 0.0 and 1.0.
            Returns 1.0 for single or empty outputs.
        """
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
        """Group responses by similarity into clusters.

        Uses a greedy clustering algorithm: each response is added to the
        first group where it meets the similarity threshold with the group's
        first member. Creates a new group if no match is found.

        Args
        ----
        outputs : list[ModelOutput]
            List of model outputs to cluster.

        Returns
        -------
        list[list[ModelOutput]]
            List of groups, where each group contains similar outputs.
            Groups are ordered by creation time (first match found).
        """
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
        """Convert a numerical agreement score to a categorical level.

        Maps continuous scores to discrete AgreementLevel categories
        using fixed thresholds: unanimous (>=0.95), strong (>=0.75),
        moderate (>=0.5), weak (>=0.25), none (<0.25).

        Args
        ----
        score : float
            Agreement score between 0.0 and 1.0.

        Returns
        -------
        AgreementLevel
            The categorical agreement level.
        """
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
        """Select the response from the largest similarity group.

        Implements majority voting by choosing the response that has
        the most similar responses (largest cluster). Returns the
        first member of the largest group.

        Args
        ----
        groups : list[list[ModelOutput]]
            Pre-computed similarity groups.
        outputs : list[ModelOutput]
            Original outputs (unused but kept for interface consistency).

        Returns
        -------
        tuple[str, str]
            Tuple of (response_text, model_id) for the selected output.
            Returns ("", "") if groups is empty.
        """
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
        """Select response by confidence-weighted voting.

        Sums confidence scores within each group and selects from the
        group with the highest total weight. Within that group, selects
        the individual response with highest confidence.

        Args
        ----
        groups : list[list[ModelOutput]]
            Pre-computed similarity groups.
        outputs : list[ModelOutput]
            Original outputs (unused but kept for interface consistency).

        Returns
        -------
        tuple[str, str]
            Tuple of (response_text, model_id) for the selected output.
            Returns ("", "") if groups is empty.
        """
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
        scorer: Optional[Callable[[str], float]],
    ) -> tuple[str, str]:
        """Select the best response according to a scoring function.

        If no scorer is provided, defaults to selecting the longest
        response (by character count).

        Args
        ----
        outputs : list[ModelOutput]
            List of model outputs to score.
        scorer : Optional[Callable[[str], float]]
            Function that takes a response string and returns a score.
            Higher scores are better. If None, uses response length.

        Returns
        -------
        tuple[str, str]
            Tuple of (response_text, model_id) for the highest-scoring output.
            Returns ("", "") if outputs is empty.
        """
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
        """Find the consensus response most similar to all others.

        Selects the response with the highest average similarity to
        all other responses, effectively finding the "centroid" of
        the response distribution.

        Args
        ----
        groups : list[list[ModelOutput]]
            Pre-computed similarity groups (unused but kept for interface).
        outputs : list[ModelOutput]
            List of model outputs to analyze.

        Returns
        -------
        tuple[str, str]
            Tuple of (response_text, model_id) for the most central output.
            Returns ("", "") if outputs is empty.
        """
        if not outputs:
            return "", ""

        calculator = TextSimilarityCalculator()
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
        """Select the longest response by character count.

        Useful when more detailed responses are preferred.

        Args
        ----
        outputs : list[ModelOutput]
            List of model outputs to compare.

        Returns
        -------
        tuple[str, str]
            Tuple of (response_text, model_id) for the longest output.
            Returns ("", "") if outputs is empty.
        """
        if not outputs:
            return "", ""
        selected = max(outputs, key=lambda x: len(x.response))
        return selected.response, selected.model_id

    @staticmethod
    def _shortest(outputs: list[ModelOutput]) -> tuple[str, str]:
        """Select the shortest response by character count.

        Useful when conciseness is preferred.

        Args
        ----
        outputs : list[ModelOutput]
            List of model outputs to compare.

        Returns
        -------
        tuple[str, str]
            Tuple of (response_text, model_id) for the shortest output.
            Returns ("", "") if outputs is empty.
        """
        if not outputs:
            return "", ""
        selected = min(outputs, key=lambda x: len(x.response))
        return selected.response, selected.model_id

    @staticmethod
    def _most_confident(outputs: list[ModelOutput]) -> tuple[str, str]:
        """Select the response with highest confidence score.

        Useful when model confidence is a reliable quality indicator.

        Args
        ----
        outputs : list[ModelOutput]
            List of model outputs to compare.

        Returns
        -------
        tuple[str, str]
            Tuple of (response_text, model_id) for the most confident output.
            Returns ("", "") if outputs is empty.
        """
        if not outputs:
            return "", ""
        selected = max(outputs, key=lambda x: x.confidence)
        return selected.response, selected.model_id

    def _diverse_selection(
        self,
        outputs: list[ModelOutput],
    ) -> tuple[str, str]:
        """Select the most diverse response (least similar to others).

        Chooses the response that is most different from all other
        responses, maximizing diversity in the selection. Useful for
        exploring alternative viewpoints.

        Args
        ----
        outputs : list[ModelOutput]
            List of model outputs to analyze.

        Returns
        -------
        tuple[str, str]
            Tuple of (response_text, model_id) for the most diverse output.
            Returns ("", "") if outputs is empty.
        """
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
    """Analyze pairwise agreement between model outputs.

    This class provides detailed analysis of how models agree or disagree
    with each other, including pairwise similarity matrices, per-model
    agreement scores, and identification of outlier models.

    Attributes
    ----------
    _calculator : TextSimilarityCalculator
        Calculator for computing text similarity between responses.

    Examples
    --------
    Basic agreement analysis:

    >>> from insideLLMs.contrib.ensemble import ModelAgreementAnalyzer, ModelOutput
    >>> analyzer = ModelAgreementAnalyzer()
    >>> outputs = [
    ...     ModelOutput(model_id="gpt-4", response="The sky is blue."),
    ...     ModelOutput(model_id="claude", response="The sky is blue."),
    ...     ModelOutput(model_id="llama", response="Water is wet."),
    ... ]
    >>> analysis = analyzer.analyze(outputs)
    >>> analysis["overall_agreement"] > 0
    True
    >>> analysis["most_agreeable"]  # Most similar to others
    'gpt-4'

    Examining pairwise agreement:

    >>> "gpt-4-claude" in analysis["agreement_matrix"]
    True

    Single model (trivially perfect agreement):

    >>> analysis = analyzer.analyze([ModelOutput(model_id="only", response="test")])
    >>> analysis["overall_agreement"]
    1.0

    Finding disagreeing models:

    >>> print(f"Least agreeable: {analysis['least_agreeable']}")
    Least agreeable: llama
    """

    def __init__(
        self,
        similarity_calculator: Optional[TextSimilarityCalculator] = None,
    ):
        """Initialize the agreement analyzer.

        Args
        ----
        similarity_calculator : Optional[TextSimilarityCalculator], optional
            Custom calculator for computing response similarity (default: None).
            If None, a default TextSimilarityCalculator is created.

        Examples
        --------
        >>> analyzer = ModelAgreementAnalyzer()
        >>> isinstance(analyzer._calculator, TextSimilarityCalculator)
        True

        >>> from insideLLMs.contrib.ensemble import TextSimilarityCalculator, ResponseNormalizer
        >>> custom = TextSimilarityCalculator(
        ...     normalizer=ResponseNormalizer(remove_punctuation=True)
        ... )
        >>> analyzer = ModelAgreementAnalyzer(similarity_calculator=custom)
        """
        self._calculator = similarity_calculator or TextSimilarityCalculator()

    def analyze(
        self,
        outputs: list[ModelOutput],
    ) -> dict[str, Any]:
        """Analyze agreement between model outputs.

        Computes pairwise similarity scores between all models, overall
        agreement metrics, and identifies the most/least agreeable models.

        Args
        ----
        outputs : list[ModelOutput]
            List of model outputs to analyze. Should contain at least 2
            outputs for meaningful analysis.

        Returns
        -------
        dict[str, Any]
            Analysis results containing:
            - n_models: Number of models analyzed
            - overall_agreement: Mean pairwise similarity (0-1)
            - agreement_matrix: Dict mapping "model1-model2" to similarity
            - per_model_agreement: Dict mapping model_id to avg agreement
            - most_agreeable: model_id that agrees most with others
            - least_agreeable: model_id that agrees least with others

        Examples
        --------
        Analyzing multiple models:

        >>> from insideLLMs.contrib.ensemble import ModelAgreementAnalyzer, ModelOutput
        >>> analyzer = ModelAgreementAnalyzer()
        >>> outputs = [
        ...     ModelOutput(model_id="a", response="Yes"),
        ...     ModelOutput(model_id="b", response="Yes"),
        ...     ModelOutput(model_id="c", response="No"),
        ... ]
        >>> result = analyzer.analyze(outputs)
        >>> result["n_models"]
        3
        >>> result["most_agreeable"] in ("a", "b")
        True
        >>> result["least_agreeable"]
        'c'

        Single output edge case:

        >>> result = analyzer.analyze([ModelOutput(model_id="x", response="test")])
        >>> result["overall_agreement"]
        1.0
        >>> result["agreement_matrix"]
        {}

        Empty outputs:

        >>> result = analyzer.analyze([])
        >>> result["n_models"]
        0
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
    """Evaluate ensemble model performance across multiple prompts.

    This class provides comprehensive evaluation of ensemble behavior by
    analyzing aggregation results, agreement patterns, and model selection
    statistics across a batch of prompts. It generates detailed reports
    with actionable recommendations.

    Attributes
    ----------
    _aggregator : ResponseAggregator
        Aggregator used to combine model outputs.
    _analyzer : ModelAgreementAnalyzer
        Analyzer used for agreement metrics.

    Examples
    --------
    Basic ensemble evaluation:

    >>> from insideLLMs.contrib.ensemble import EnsembleEvaluator, ModelOutput
    >>> evaluator = EnsembleEvaluator()
    >>> prompt_outputs = [
    ...     [  # First prompt
    ...         ModelOutput(model_id="gpt-4", response="Yes"),
    ...         ModelOutput(model_id="claude", response="Yes"),
    ...     ],
    ...     [  # Second prompt
    ...         ModelOutput(model_id="gpt-4", response="Blue"),
    ...         ModelOutput(model_id="claude", response="Blue"),
    ...     ],
    ... ]
    >>> report = evaluator.evaluate(prompt_outputs)
    >>> report.n_prompts
    2
    >>> report.overall_agreement
    1.0

    Evaluating with different aggregation methods:

    >>> from insideLLMs.contrib.ensemble import AggregationMethod
    >>> report = evaluator.evaluate(prompt_outputs, method=AggregationMethod.WEIGHTED_VOTE)
    >>> report.aggregation_method
    AggregationMethod.WEIGHTED_VOTE

    Checking recommendations:

    >>> for rec in report.recommendations:
    ...     print(f"- {rec}")

    Using custom components:

    >>> from insideLLMs.contrib.ensemble import ResponseAggregator, ModelAgreementAnalyzer
    >>> custom_aggregator = ResponseAggregator(similarity_threshold=0.5)
    >>> evaluator = EnsembleEvaluator(aggregator=custom_aggregator)
    """

    def __init__(
        self,
        aggregator: Optional[ResponseAggregator] = None,
        analyzer: Optional[ModelAgreementAnalyzer] = None,
    ):
        """Initialize the ensemble evaluator.

        Args
        ----
        aggregator : Optional[ResponseAggregator], optional
            Custom response aggregator (default: None).
            If None, a default ResponseAggregator is created.
        analyzer : Optional[ModelAgreementAnalyzer], optional
            Custom agreement analyzer (default: None).
            If None, a default ModelAgreementAnalyzer is created.

        Examples
        --------
        >>> evaluator = EnsembleEvaluator()
        >>> isinstance(evaluator._aggregator, ResponseAggregator)
        True

        >>> from insideLLMs.contrib.ensemble import ResponseAggregator
        >>> custom = ResponseAggregator(similarity_threshold=0.9)
        >>> evaluator = EnsembleEvaluator(aggregator=custom)
        >>> evaluator._aggregator._threshold
        0.9
        """
        self._aggregator = aggregator or ResponseAggregator()
        self._analyzer = analyzer or ModelAgreementAnalyzer()

    def evaluate(
        self,
        prompt_outputs: list[list[ModelOutput]],
        method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
        scorer: Optional[Callable[[str], float]] = None,
    ) -> EnsembleReport:
        """Evaluate ensemble performance across multiple prompts.

        Processes each prompt's outputs through aggregation and agreement
        analysis, then computes aggregate statistics and generates a
        comprehensive report with recommendations.

        Args
        ----
        prompt_outputs : list[list[ModelOutput]]
            List of model outputs for each prompt. Each inner list contains
            outputs from different models for the same prompt.
        method : AggregationMethod, optional
            Aggregation method to use (default: MAJORITY_VOTE).
        scorer : Optional[Callable[[str], float]], optional
            Custom scoring function for BEST_OF_N method (default: None).

        Returns
        -------
        EnsembleReport
            Comprehensive report containing:
            - n_prompts, n_models: Dataset size
            - overall_agreement: Mean agreement across prompts
            - per_model_selection_rate: How often each model was selected
            - per_model_agreement: How well each model agrees with others
            - best_performing_model: Most frequently selected model
            - most_agreeable_model: Model with highest average agreement
            - ensemble_diversity: Response diversity score
            - recommendations: Actionable suggestions

        Examples
        --------
        Standard evaluation:

        >>> from insideLLMs.contrib.ensemble import EnsembleEvaluator, ModelOutput
        >>> evaluator = EnsembleEvaluator()
        >>> outputs = [
        ...     [
        ...         ModelOutput(model_id="a", response="X"),
        ...         ModelOutput(model_id="b", response="X"),
        ...     ]
        ... ]
        >>> report = evaluator.evaluate(outputs)
        >>> report.n_prompts
        1
        >>> report.overall_agreement
        1.0

        Empty input handling:

        >>> report = evaluator.evaluate([])
        >>> report.n_prompts
        0
        >>> report.recommendations
        ['No data to evaluate']

        With custom scorer:

        >>> def quality_scorer(response: str) -> float:
        ...     return len(response) / 100.0
        >>> report = evaluator.evaluate(outputs, scorer=quality_scorer)
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
        """Calculate average response diversity across all prompts.

        Computes diversity as 1 - average_similarity for each prompt,
        then averages across all prompts. Higher diversity indicates
        models produce more varied responses.

        Args
        ----
        prompt_outputs : list[list[ModelOutput]]
            List of model outputs for each prompt.

        Returns
        -------
        float
            Average diversity score between 0.0 (identical) and 1.0 (different).
            Returns 0.0 if no prompts have at least 2 outputs.
        """
        calculator = TextSimilarityCalculator()
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
        """Generate actionable recommendations based on evaluation metrics.

        Analyzes selection patterns, agreement levels, and diversity to
        produce specific suggestions for ensemble optimization.

        Args
        ----
        selection_rates : dict[str, float]
            Per-model selection rate (0-1).
        agreements : dict[str, float]
            Per-model average agreement with others (0-1).
        diversity : float
            Overall ensemble diversity (0-1).

        Returns
        -------
        list[str]
            List of recommendation strings. Always contains at least one
            recommendation (defaults to "Ensemble appears well-balanced").
        """
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
        """Create an empty report for when no data is available.

        Args
        ----
        method : AggregationMethod
            The aggregation method that was requested.

        Returns
        -------
        EnsembleReport
            Report with zero counts and a "No data to evaluate" recommendation.
        """
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
    """Manage and query an ensemble of language models.

    This class provides a high-level interface for working with multiple
    language models as an ensemble. It handles querying all models,
    aggregating their responses, and comparing different aggregation methods.

    Attributes
    ----------
    _models : dict[str, Callable[[str], str]]
        Dictionary mapping model IDs to their query functions.
    _aggregator : ResponseAggregator
        Aggregator used to combine model responses.
    _default_method : AggregationMethod
        Default aggregation method for queries.

    Examples
    --------
    Creating and querying an ensemble:

    >>> from insideLLMs.contrib.ensemble import ModelEnsemble, AggregationMethod
    >>> def mock_gpt(prompt):
    ...     return "GPT response"
    >>> def mock_claude(prompt):
    ...     return "Claude response"
    >>> ensemble = ModelEnsemble(
    ...     models={"gpt-4": mock_gpt, "claude": mock_claude},
    ...     default_method=AggregationMethod.MAJORITY_VOTE
    ... )
    >>> result = ensemble.query("What is AI?")
    >>> result.final_response in ("GPT response", "Claude response")
    True

    Using weighted voting with confidence:

    >>> ensemble = ModelEnsemble(
    ...     models={"a": lambda p: "Yes", "b": lambda p: "No"},
    ...     default_method=AggregationMethod.WEIGHTED_VOTE
    ... )
    >>> result = ensemble.query("Is this good?")

    Comparing aggregation methods:

    >>> methods_comparison = ensemble.compare_methods("Test prompt")
    >>> AggregationMethod.MAJORITY_VOTE in methods_comparison
    True
    >>> AggregationMethod.CONSENSUS in methods_comparison
    True

    Handling model failures gracefully:

    >>> def failing_model(prompt):
    ...     raise RuntimeError("API error")
    >>> ensemble = ModelEnsemble(
    ...     models={"good": lambda p: "OK", "bad": failing_model}
    ... )
    >>> result = ensemble.query("Test")  # Only uses successful models
    >>> len(result.source_outputs)  # Only 1 output (failed model skipped)
    1
    """

    def __init__(
        self,
        models: dict[str, Callable[[str], str]],
        aggregator: Optional[ResponseAggregator] = None,
        default_method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
    ):
        """Initialize the model ensemble.

        Args
        ----
        models : dict[str, Callable[[str], str]]
            Dictionary mapping model identifiers to their query functions.
            Each function should take a prompt string and return a response string.
        aggregator : Optional[ResponseAggregator], optional
            Custom response aggregator (default: None).
            If None, a default ResponseAggregator is created.
        default_method : AggregationMethod, optional
            Default aggregation method for queries (default: MAJORITY_VOTE).

        Examples
        --------
        >>> def model_a(prompt):
        ...     return f"A: {prompt}"
        >>> def model_b(prompt):
        ...     return f"B: {prompt}"
        >>> ensemble = ModelEnsemble(models={"a": model_a, "b": model_b})
        >>> len(ensemble._models)
        2

        >>> from insideLLMs.contrib.ensemble import AggregationMethod
        >>> ensemble = ModelEnsemble(
        ...     models={"x": lambda p: "test"},
        ...     default_method=AggregationMethod.CONSENSUS
        ... )
        >>> ensemble._default_method
        AggregationMethod.CONSENSUS
        """
        self._models = models
        self._aggregator = aggregator or ResponseAggregator()
        self._default_method = default_method

    def query(
        self,
        prompt: str,
        method: Optional[AggregationMethod] = None,
        scorer: Optional[Callable[[str], float]] = None,
    ) -> AggregatedOutput:
        """Query all models and aggregate their responses.

        Sends the prompt to each model in the ensemble, collects their
        responses, and aggregates them using the specified method. Models
        that raise exceptions are silently skipped.

        Args
        ----
        prompt : str
            The prompt to send to all models.
        method : Optional[AggregationMethod], optional
            Aggregation method to use (default: None uses default_method).
        scorer : Optional[Callable[[str], float]], optional
            Custom scoring function for BEST_OF_N method (default: None).

        Returns
        -------
        AggregatedOutput
            The aggregated result from all successful model responses.

        Examples
        --------
        Basic query:

        >>> ensemble = ModelEnsemble(models={"a": lambda p: "Yes", "b": lambda p: "Yes"})
        >>> result = ensemble.query("Agree?")
        >>> result.final_response
        'Yes'
        >>> result.agreement_level
        AgreementLevel.UNANIMOUS

        Override aggregation method:

        >>> from insideLLMs.contrib.ensemble import AggregationMethod
        >>> result = ensemble.query("Test", method=AggregationMethod.SHORTEST)

        With custom scorer:

        >>> def quality_score(response):
        ...     return len(response)  # Prefer longer responses
        >>> result = ensemble.query("Explain", scorer=quality_score)
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
        methods: Optional[list[AggregationMethod]] = None,
    ) -> dict[AggregationMethod, AggregatedOutput]:
        """Compare results across different aggregation methods.

        Queries all models once, then applies each specified aggregation
        method to the same set of outputs. Useful for understanding how
        different methods affect the final result.

        Args
        ----
        prompt : str
            The prompt to send to all models.
        methods : Optional[list[AggregationMethod]], optional
            Methods to compare (default: None uses all methods).

        Returns
        -------
        dict[AggregationMethod, AggregatedOutput]
            Dictionary mapping each method to its aggregated output.

        Examples
        --------
        Compare all methods:

        >>> ensemble = ModelEnsemble(models={"a": lambda p: "X", "b": lambda p: "Y"})
        >>> comparison = ensemble.compare_methods("Test")
        >>> len(comparison)  # All 8 aggregation methods
        8

        Compare specific methods:

        >>> from insideLLMs.contrib.ensemble import AggregationMethod
        >>> methods = [AggregationMethod.MAJORITY_VOTE, AggregationMethod.CONSENSUS]
        >>> comparison = ensemble.compare_methods("Test", methods=methods)
        >>> len(comparison)
        2

        Analyze method differences:

        >>> for method, result in comparison.items():
        ...     print(f"{method.name}: {result.final_response[:20]}")
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
    """Create a model ensemble with the specified models and aggregation method.

    This is a convenience function that creates a :class:`ModelEnsemble` instance
    with sensible defaults. For more control, instantiate ModelEnsemble directly.

    Args
    ----
    models : dict[str, Callable[[str], str]]
        Dictionary mapping model identifiers to their query functions.
        Each function should take a prompt string and return a response string.
    method : AggregationMethod, optional
        Default aggregation method for queries (default: MAJORITY_VOTE).

    Returns
    -------
    ModelEnsemble
        Configured ensemble ready for querying.

    Examples
    --------
    Create a simple ensemble:

    >>> from insideLLMs.contrib.ensemble import create_ensemble
    >>> def fast_model(prompt):
    ...     return "Quick answer"
    >>> def slow_model(prompt):
    ...     return "Detailed comprehensive answer"
    >>> ensemble = create_ensemble({"fast": fast_model, "slow": slow_model})
    >>> result = ensemble.query("What is AI?")

    Create with weighted voting:

    >>> from insideLLMs.contrib.ensemble import AggregationMethod
    >>> ensemble = create_ensemble(
    ...     models={"a": lambda p: "Yes", "b": lambda p: "No"},
    ...     method=AggregationMethod.WEIGHTED_VOTE
    ... )

    Use consensus method:

    >>> ensemble = create_ensemble(
    ...     models={"m1": lambda p: "A", "m2": lambda p: "B", "m3": lambda p: "A"},
    ...     method=AggregationMethod.CONSENSUS
    ... )

    Chain with query:

    >>> result = create_ensemble({"x": lambda p: "test"}).query("prompt")
    >>> result.final_response
    'test'
    """
    return ModelEnsemble(models, default_method=method)


def aggregate_responses(
    outputs: list[ModelOutput],
    method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
) -> AggregatedOutput:
    """Aggregate pre-collected model outputs into a single response.

    This is a convenience function for one-off aggregation of existing outputs.
    For repeated aggregations, create a :class:`ResponseAggregator` instance.

    Args
    ----
    outputs : list[ModelOutput]
        List of model outputs to aggregate. Can be empty.
    method : AggregationMethod, optional
        Aggregation method to use (default: MAJORITY_VOTE).

    Returns
    -------
    AggregatedOutput
        The aggregated result with final response, agreement metrics, and metadata.

    Examples
    --------
    Basic aggregation:

    >>> from insideLLMs.contrib.ensemble import aggregate_responses, ModelOutput
    >>> outputs = [
    ...     ModelOutput(model_id="gpt-4", response="Paris"),
    ...     ModelOutput(model_id="claude", response="Paris"),
    ...     ModelOutput(model_id="llama", response="London"),
    ... ]
    >>> result = aggregate_responses(outputs)
    >>> result.final_response
    'Paris'
    >>> result.agreement_level
    AgreementLevel.MODERATE

    Use different method:

    >>> from insideLLMs.contrib.ensemble import AggregationMethod
    >>> result = aggregate_responses(outputs, AggregationMethod.CONSENSUS)

    With confidence scores:

    >>> outputs = [
    ...     ModelOutput(model_id="a", response="X", confidence=0.9),
    ...     ModelOutput(model_id="b", response="Y", confidence=0.95),
    ... ]
    >>> result = aggregate_responses(outputs, AggregationMethod.MOST_CONFIDENT)
    >>> result.selected_model
    'b'

    Empty outputs:

    >>> result = aggregate_responses([])
    >>> result.final_response
    ''
    """
    aggregator = ResponseAggregator()
    return aggregator.aggregate(outputs, method)


def analyze_model_agreement(
    outputs: list[ModelOutput],
) -> dict[str, Any]:
    """Analyze pairwise agreement between model outputs.

    This is a convenience function for quick agreement analysis.
    For repeated analysis, create a :class:`ModelAgreementAnalyzer` instance.

    Args
    ----
    outputs : list[ModelOutput]
        List of model outputs to analyze. Needs at least 2 outputs
        for meaningful pairwise analysis.

    Returns
    -------
    dict[str, Any]
        Analysis results containing:
        - n_models: Number of models analyzed
        - overall_agreement: Mean pairwise similarity (0-1)
        - agreement_matrix: Pairwise similarity scores
        - per_model_agreement: Per-model average agreement
        - most_agreeable: Model ID with highest average agreement
        - least_agreeable: Model ID with lowest average agreement

    Examples
    --------
    Analyze agreement:

    >>> from insideLLMs.contrib.ensemble import analyze_model_agreement, ModelOutput
    >>> outputs = [
    ...     ModelOutput(model_id="a", response="The sky is blue."),
    ...     ModelOutput(model_id="b", response="The sky is blue."),
    ...     ModelOutput(model_id="c", response="Water is wet."),
    ... ]
    >>> analysis = analyze_model_agreement(outputs)
    >>> analysis["n_models"]
    3
    >>> analysis["most_agreeable"] in ("a", "b")
    True

    Find outlier models:

    >>> if analysis["overall_agreement"] < 0.5:
    ...     print(f"Low agreement. Outlier: {analysis['least_agreeable']}")

    Single model edge case:

    >>> analysis = analyze_model_agreement([ModelOutput(model_id="x", response="test")])
    >>> analysis["overall_agreement"]
    1.0

    Check pairwise agreement:

    >>> "a-b" in analysis["agreement_matrix"]
    True
    """
    analyzer = ModelAgreementAnalyzer()
    return analyzer.analyze(outputs)


def evaluate_ensemble(
    prompt_outputs: list[list[ModelOutput]],
    method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
) -> EnsembleReport:
    """Evaluate ensemble performance across multiple prompts.

    This is a convenience function for generating ensemble performance reports.
    For customized evaluation, create an :class:`EnsembleEvaluator` instance.

    Args
    ----
    prompt_outputs : list[list[ModelOutput]]
        List of model outputs for each prompt. Each inner list contains
        outputs from different models for the same prompt.
    method : AggregationMethod, optional
        Aggregation method to use (default: MAJORITY_VOTE).

    Returns
    -------
    EnsembleReport
        Comprehensive report with metrics and recommendations.

    Examples
    --------
    Evaluate an ensemble:

    >>> from insideLLMs.contrib.ensemble import evaluate_ensemble, ModelOutput
    >>> prompt_outputs = [
    ...     [ModelOutput(model_id="a", response="X"), ModelOutput(model_id="b", response="X")],
    ...     [ModelOutput(model_id="a", response="Y"), ModelOutput(model_id="b", response="Z")],
    ... ]
    >>> report = evaluate_ensemble(prompt_outputs)
    >>> report.n_prompts
    2
    >>> report.overall_agreement > 0
    True

    Check model performance:

    >>> report.best_performing_model
    'a'
    >>> report.per_model_selection_rate
    {'a': 1.0}

    Get recommendations:

    >>> for rec in report.recommendations:
    ...     print(f"- {rec}")

    Using different method:

    >>> from insideLLMs.contrib.ensemble import AggregationMethod
    >>> report = evaluate_ensemble(prompt_outputs, AggregationMethod.CONSENSUS)
    """
    evaluator = EnsembleEvaluator()
    return evaluator.evaluate(prompt_outputs, method)


def quick_ensemble_check(
    models: dict[str, Callable[[str], str]],
    prompts: list[str],
) -> dict[str, Any]:
    """Run a quick health check on an ensemble across multiple prompts.

    This is a convenience function for rapid ensemble diagnostics. It queries
    all models with each prompt and returns summary statistics about agreement
    and model selection patterns.

    Args
    ----
    models : dict[str, Callable[[str], str]]
        Dictionary mapping model identifiers to their query functions.
    prompts : list[str]
        List of prompts to test the ensemble with.

    Returns
    -------
    dict[str, Any]
        Quick check results containing:
        - n_prompts: Number of prompts tested
        - n_models: Number of models in ensemble
        - avg_agreement: Average agreement score across prompts
        - selection_distribution: Count of how often each model was selected
        - most_selected: Model ID selected most frequently (or None)

    Examples
    --------
    Quick ensemble health check:

    >>> from insideLLMs.contrib.ensemble import quick_ensemble_check
    >>> models = {
    ...     "fast": lambda p: "Quick answer",
    ...     "slow": lambda p: "Detailed answer",
    ... }
    >>> prompts = ["What is AI?", "Explain ML", "Define data"]
    >>> check = quick_ensemble_check(models, prompts)
    >>> check["n_prompts"]
    3
    >>> check["n_models"]
    2

    Analyze agreement:

    >>> if check["avg_agreement"] > 0.8:
    ...     print("High agreement - models are consistent")
    >>> if check["avg_agreement"] < 0.3:
    ...     print("Low agreement - significant model disagreement")

    Check selection patterns:

    >>> print(f"Most selected: {check['most_selected']}")
    >>> for model, count in check["selection_distribution"].items():
    ...     print(f"  {model}: {count} times")

    Empty prompts:

    >>> check = quick_ensemble_check(models, [])
    >>> check["avg_agreement"]
    0.0
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


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import SimilarityCalculator. The canonical name is
# TextSimilarityCalculator.
SimilarityCalculator = TextSimilarityCalculator
