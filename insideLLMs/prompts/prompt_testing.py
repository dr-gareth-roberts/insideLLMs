"""
Prompt testing and experimentation utilities for systematic prompt engineering.

This module provides a comprehensive framework for testing, evaluating, and
optimizing prompts for large language models. It supports systematic
experimentation with prompt variations, A/B testing, and quantitative
scoring of model responses.

Key Components
--------------
PromptStrategy : Enum
    Common prompt engineering strategies (zero-shot, few-shot, chain-of-thought, etc.)

PromptVariant : dataclass
    A single prompt variant with metadata for tracking.

PromptVariationGenerator : class
    Fluent builder for generating prompt variations with different prefixes,
    suffixes, strategies, and combinations.

PromptScorer : class
    Configurable scoring system for evaluating model responses based on
    length, keywords, format compliance, and similarity criteria.

PromptABTestRunner : class
    Runs A/B tests across multiple prompt variants and collects results.

PromptExperiment : class
    High-level orchestration for complete prompt testing experiments.

Examples
--------
Basic prompt variation and testing:

>>> from insideLLMs.prompts.prompt_testing import (
...     PromptVariationGenerator,
...     PromptExperiment,
... )
>>>
>>> # Generate variations of a base prompt
>>> generator = PromptVariationGenerator("Explain quantum computing")
>>> generator.add_prefix_variations()
>>> generator.add_strategy_variations()
>>> variants = generator.generate()
>>> print(f"Generated {len(variants)} variants")
Generated 8 variants

Running a complete experiment:

>>> # Create and configure an experiment
>>> experiment = PromptExperiment("quantum_explanation_test")
>>> experiment.add_variant(
...     "Explain quantum computing in simple terms.",
...     variant_id="simple"
... )
>>> experiment.add_variant(
...     "You are a physics professor. Explain quantum computing.",
...     variant_id="expert_role"
... )
>>> experiment.configure_scorer(
...     length_range=(100, 500),
...     required_keywords=["qubit", "superposition"]
... )
>>>
>>> # Run with a mock model function for demonstration
>>> def mock_model(prompt: str) -> str:
...     return "Quantum computing uses qubits in superposition..."
>>>
>>> results = experiment.run(mock_model, runs_per_variant=3)
>>> print(f"Best variant: {results.best_variant_id}")
Best variant: expert_role

Creating few-shot variants:

>>> from insideLLMs.prompts.prompt_testing import create_few_shot_variants
>>>
>>> examples = [
...     {"input": "2 + 2", "output": "4"},
...     {"input": "3 * 4", "output": "12"},
...     {"input": "10 / 2", "output": "5"},
... ]
>>> variants = create_few_shot_variants(
...     task="Solve the math problem.",
...     examples=examples,
...     query="7 + 8",
...     num_shots=[0, 1, 3]
... )
>>> print([v.id for v in variants])
['0_shot', '1_shot', '3_shot']

Notes
-----
- All scoring functions return values in the range [0.0, 1.0].
- The module is designed for use with any LLM API through callable functions.
- Thread-safety is not guaranteed; use separate instances for concurrent testing.
- Experiment results include timing data for latency analysis.

See Also
--------
insideLLMs.nlp.tokenization : Tokenization utilities used for similarity scoring.
"""

import hashlib
import itertools
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Optional,
)

from insideLLMs.nlp.tokenization import word_tokenize_regex


class PromptStrategy(Enum):
    """
    Common prompt engineering strategies for LLM interactions.

    This enumeration defines standard prompting techniques that can be
    applied to improve model responses. Each strategy represents a
    different approach to structuring prompts.

    Attributes
    ----------
    ZERO_SHOT : str
        Direct prompting without examples. The model relies solely on
        its training to generate responses.
    FEW_SHOT : str
        Prompting with a small number of input-output examples before
        the actual query, enabling in-context learning.
    CHAIN_OF_THOUGHT : str
        Prompting that encourages the model to show its reasoning
        process step by step before arriving at an answer.
    STEP_BY_STEP : str
        Similar to chain-of-thought but with explicit instruction to
        break down the problem into numbered steps.
    ROLE_PLAY : str
        Assigning a specific persona or role to the model (e.g.,
        "You are an expert physicist").
    SOCRATIC : str
        Prompting that encourages the model to ask clarifying questions
        or explore the problem through dialogue.
    TREE_OF_THOUGHT : str
        Advanced prompting that explores multiple reasoning paths
        simultaneously before selecting the best approach.

    Examples
    --------
    Using strategies with prompt variants:

    >>> from insideLLMs.prompts.prompt_testing import PromptStrategy, PromptVariant
    >>>
    >>> # Create a chain-of-thought variant
    >>> cot_variant = PromptVariant(
    ...     id="math_cot",
    ...     content="What is 25 * 17? Let's think step by step.",
    ...     strategy=PromptStrategy.CHAIN_OF_THOUGHT
    ... )
    >>> print(cot_variant.strategy.value)
    chain_of_thought

    Checking strategy type in experiments:

    >>> variant = PromptVariant(
    ...     id="test",
    ...     content="Explain gravity",
    ...     strategy=PromptStrategy.ZERO_SHOT
    ... )
    >>> if variant.strategy == PromptStrategy.ZERO_SHOT:
    ...     print("No examples provided")
    No examples provided

    See Also
    --------
    PromptVariant : Container for prompts with strategy metadata.
    PromptVariationGenerator.add_strategy_variations : Auto-generates strategy variants.
    """

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STEP_BY_STEP = "step_by_step"
    ROLE_PLAY = "role_play"
    SOCRATIC = "socratic"
    TREE_OF_THOUGHT = "tree_of_thought"


@dataclass
class PromptVariant:
    """
    A single prompt variant for testing and experimentation.

    This dataclass represents one version of a prompt that can be tested
    against other variants. It includes metadata for tracking, optional
    strategy classification, and timestamps for experiment logging.

    Parameters
    ----------
    id : str
        Unique identifier for this variant within an experiment.
        Used for tracking results and generating reports.
    content : str
        The actual prompt text to send to the language model.
    strategy : PromptStrategy, optional
        The prompting strategy used (e.g., zero-shot, few-shot).
        Default is None.
    metadata : dict[str, Any], optional
        Additional metadata for tracking (e.g., variation type, source).
        Default is an empty dict.
    created_at : datetime, optional
        Timestamp when this variant was created.
        Default is the current time.

    Attributes
    ----------
    id : str
        The unique identifier for this variant.
    content : str
        The prompt text content.
    strategy : PromptStrategy or None
        The associated prompting strategy.
    metadata : dict[str, Any]
        Custom metadata dictionary.
    created_at : datetime
        Creation timestamp.

    Examples
    --------
    Creating a basic variant:

    >>> from insideLLMs.prompts.prompt_testing import PromptVariant
    >>>
    >>> variant = PromptVariant(
    ...     id="baseline",
    ...     content="Summarize the following article:"
    ... )
    >>> print(variant.id)
    baseline
    >>> print(len(variant.content))
    35

    Creating a variant with strategy and metadata:

    >>> from insideLLMs.prompts.prompt_testing import PromptVariant, PromptStrategy
    >>>
    >>> variant = PromptVariant(
    ...     id="expert_cot",
    ...     content="You are a senior data scientist. Analyze this dataset step by step.",
    ...     strategy=PromptStrategy.CHAIN_OF_THOUGHT,
    ...     metadata={
    ...         "variation_type": "role_plus_cot",
    ...         "author": "data_team",
    ...         "version": "2.1"
    ...     }
    ... )
    >>> print(variant.strategy.value)
    chain_of_thought
    >>> print(variant.metadata["version"])
    2.1

    Using variants in a set (hashable):

    >>> v1 = PromptVariant(id="a", content="Hello")
    >>> v2 = PromptVariant(id="b", content="World")
    >>> v3 = PromptVariant(id="a", content="Different content")
    >>>
    >>> variant_set = {v1, v2}
    >>> print(len(variant_set))
    2
    >>> # Note: hash is based on id only
    >>> print(v1 in variant_set)
    True

    See Also
    --------
    PromptVariationGenerator : Generates multiple variants from a base prompt.
    PromptTestResult : Stores test results for a variant.
    """

    id: str
    content: str
    strategy: Optional[PromptStrategy] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        """
        Return hash based on variant ID for use in sets and dicts.

        Returns
        -------
        int
            Hash value derived from the variant's id.

        Notes
        -----
        Only the `id` field is used for hashing. Two variants with the
        same id but different content will have the same hash.
        """
        digest = hashlib.sha256(self.id.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big", signed=True)
        # Avoid returning -1, which CPython reserves as an internal error sentinel.
        return -2 if value == -1 else value


@dataclass
class PromptTestResult:
    """
    Result from testing a single prompt variant.

    This dataclass captures all relevant information from a single test
    run, including the model's response, scoring metrics, timing data,
    and any errors that occurred.

    Parameters
    ----------
    variant_id : str
        The ID of the prompt variant that was tested.
    response : str
        The model's response to the prompt.
    score : float
        The overall score for this response (0.0 to 1.0).
    latency_ms : float
        Time taken to get the response, in milliseconds.
    metadata : dict[str, Any], optional
        Additional metadata including score details and expected response.
        Default is an empty dict.
    error : str, optional
        Error message if the test failed. Default is None.
    timestamp : datetime, optional
        When this test was run. Default is current time.

    Attributes
    ----------
    variant_id : str
        ID of the tested variant.
    response : str
        Model response text.
    score : float
        Overall score value.
    latency_ms : float
        Response latency in milliseconds.
    metadata : dict[str, Any]
        Test metadata and score breakdown.
    error : str or None
        Error message if test failed.
    timestamp : datetime
        Test execution timestamp.

    Examples
    --------
    Creating a successful test result:

    >>> from insideLLMs.prompts.prompt_testing import PromptTestResult
    >>>
    >>> result = PromptTestResult(
    ...     variant_id="baseline_v1",
    ...     response="The capital of France is Paris.",
    ...     score=0.95,
    ...     latency_ms=234.5,
    ...     metadata={
    ...         "score_details": {"accuracy": 1.0, "length": 0.9},
    ...         "expected": "Paris is the capital of France."
    ...     }
    ... )
    >>> print(f"Score: {result.score:.2f}, Latency: {result.latency_ms:.1f}ms")
    Score: 0.95, Latency: 234.5ms

    Creating a failed test result:

    >>> error_result = PromptTestResult(
    ...     variant_id="complex_v2",
    ...     response="",
    ...     score=0.0,
    ...     latency_ms=5000.0,
    ...     error="API timeout after 5 seconds"
    ... )
    >>> if error_result.error:
    ...     print(f"Test failed: {error_result.error}")
    Test failed: API timeout after 5 seconds

    Analyzing score breakdown:

    >>> result = PromptTestResult(
    ...     variant_id="detailed_v1",
    ...     response="Detailed explanation here...",
    ...     score=0.85,
    ...     latency_ms=456.7,
    ...     metadata={
    ...         "score_details": {
    ...             "length": 1.0,
    ...             "keywords": 0.8,
    ...             "format": 0.75
    ...         }
    ...     }
    ... )
    >>> for criterion, score in result.metadata.get("score_details", {}).items():
    ...     print(f"  {criterion}: {score:.2f}")
      length: 1.00
      keywords: 0.80
      format: 0.75

    See Also
    --------
    PromptVariant : The variant that was tested.
    PromptExperimentResult : Aggregates results across variants.
    PromptScorer : Generates scores for responses.
    """

    variant_id: str
    response: str
    score: float
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PromptExperimentResult:
    """
    Results from a complete prompt testing experiment.

    This dataclass aggregates all results from testing multiple prompt
    variants, providing methods to analyze scores and determine the
    best-performing variant.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for this experiment.
    variants : list[PromptVariant]
        All prompt variants that were tested.
    results : list[PromptTestResult]
        All individual test results.
    best_variant_id : str, optional
        ID of the best-performing variant. Default is None.
    summary : dict[str, Any], optional
        Summary statistics for the experiment. Default is empty dict.
    started_at : datetime, optional
        When the experiment started. Default is current time.
    completed_at : datetime, optional
        When the experiment completed. Default is None.

    Attributes
    ----------
    experiment_id : str
        Unique experiment identifier.
    variants : list[PromptVariant]
        Tested variants list.
    results : list[PromptTestResult]
        All test results.
    best_variant_id : str or None
        Best variant's ID.
    summary : dict[str, Any]
        Experiment summary statistics.
    started_at : datetime
        Start timestamp.
    completed_at : datetime or None
        Completion timestamp.

    Examples
    --------
    Creating and analyzing experiment results:

    >>> from insideLLMs.prompts.prompt_testing import (
    ...     PromptExperimentResult,
    ...     PromptVariant,
    ...     PromptTestResult,
    ... )
    >>>
    >>> # Create variants
    >>> variants = [
    ...     PromptVariant(id="v1", content="Explain briefly"),
    ...     PromptVariant(id="v2", content="Explain in detail"),
    ... ]
    >>>
    >>> # Create test results
    >>> results = [
    ...     PromptTestResult(variant_id="v1", response="...", score=0.7, latency_ms=100),
    ...     PromptTestResult(variant_id="v1", response="...", score=0.8, latency_ms=110),
    ...     PromptTestResult(variant_id="v2", response="...", score=0.9, latency_ms=200),
    ...     PromptTestResult(variant_id="v2", response="...", score=0.85, latency_ms=190),
    ... ]
    >>>
    >>> experiment = PromptExperimentResult(
    ...     experiment_id="test_exp_001",
    ...     variants=variants,
    ...     results=results,
    ... )
    >>>
    >>> # Analyze results
    >>> avg_scores = experiment.get_average_scores()
    >>> print(f"v1 average: {avg_scores['v1']:.2f}")
    v1 average: 0.75
    >>> print(f"v2 average: {avg_scores['v2']:.2f}")
    v2 average: 0.88
    >>> print(f"Best: {experiment.get_best_variant()}")
    Best: v2

    Working with score distributions:

    >>> scores_by_variant = experiment.get_variant_scores()
    >>> for variant_id, scores in scores_by_variant.items():
    ...     print(f"{variant_id}: {scores}")
    v1: [0.7, 0.8]
    v2: [0.9, 0.85]

    See Also
    --------
    PromptTestResult : Individual test results.
    PromptABTestRunner.run_experiment : Creates experiment results.
    PromptExperiment : High-level experiment orchestration.
    """

    experiment_id: str
    variants: list[PromptVariant]
    results: list[PromptTestResult]
    best_variant_id: Optional[str] = None
    summary: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def get_variant_scores(self) -> dict[str, list[float]]:
        """
        Get all scores grouped by variant ID.

        Collects and groups all individual test scores by their
        corresponding variant ID, allowing for distribution analysis.

        Returns
        -------
        dict[str, list[float]]
            Dictionary mapping variant IDs to lists of scores.
            Each score is in the range [0.0, 1.0].

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import (
        ...     PromptExperimentResult,
        ...     PromptTestResult,
        ... )
        >>>
        >>> results = [
        ...     PromptTestResult(variant_id="a", response="", score=0.8, latency_ms=100),
        ...     PromptTestResult(variant_id="a", response="", score=0.9, latency_ms=100),
        ...     PromptTestResult(variant_id="b", response="", score=0.7, latency_ms=100),
        ... ]
        >>>
        >>> experiment = PromptExperimentResult(
        ...     experiment_id="test",
        ...     variants=[],
        ...     results=results,
        ... )
        >>>
        >>> scores = experiment.get_variant_scores()
        >>> print(scores["a"])
        [0.8, 0.9]
        >>> print(scores["b"])
        [0.7]

        Computing score statistics:

        >>> import statistics
        >>> for vid, scores in experiment.get_variant_scores().items():
        ...     if len(scores) > 1:
        ...         print(f"{vid}: mean={statistics.mean(scores):.2f}, stdev={statistics.stdev(scores):.2f}")
        a: mean=0.85, stdev=0.07
        """
        scores: dict[str, list[float]] = {}
        for result in self.results:
            if result.variant_id not in scores:
                scores[result.variant_id] = []
            scores[result.variant_id].append(result.score)
        return scores

    def get_average_scores(self) -> dict[str, float]:
        """
        Get the average score for each variant.

        Computes the arithmetic mean of all test scores for each
        variant. Useful for comparing overall variant performance.

        Returns
        -------
        dict[str, float]
            Dictionary mapping variant IDs to their average scores.
            Returns 0.0 for variants with no scores.

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import (
        ...     PromptExperimentResult,
        ...     PromptTestResult,
        ... )
        >>>
        >>> results = [
        ...     PromptTestResult(variant_id="v1", response="", score=0.6, latency_ms=100),
        ...     PromptTestResult(variant_id="v1", response="", score=0.8, latency_ms=100),
        ...     PromptTestResult(variant_id="v2", response="", score=0.9, latency_ms=100),
        ... ]
        >>>
        >>> experiment = PromptExperimentResult(
        ...     experiment_id="test",
        ...     variants=[],
        ...     results=results,
        ... )
        >>>
        >>> averages = experiment.get_average_scores()
        >>> print(f"v1: {averages['v1']:.2f}")
        v1: 0.70
        >>> print(f"v2: {averages['v2']:.2f}")
        v2: 0.90

        Ranking variants by average score:

        >>> sorted_variants = sorted(
        ...     averages.items(),
        ...     key=lambda x: x[1],
        ...     reverse=True
        ... )
        >>> for rank, (vid, score) in enumerate(sorted_variants, 1):
        ...     print(f"{rank}. {vid}: {score:.2f}")
        1. v2: 0.90
        2. v1: 0.70
        """
        scores = self.get_variant_scores()
        return {vid: sum(s) / len(s) if s else 0.0 for vid, s in scores.items()}

    def get_best_variant(self) -> Optional[str]:
        """
        Get the ID of the variant with the highest average score.

        Identifies the best-performing variant based on mean score
        across all test runs.

        Returns
        -------
        str or None
            The ID of the best variant, or None if no results exist.

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import (
        ...     PromptExperimentResult,
        ...     PromptTestResult,
        ... )
        >>>
        >>> results = [
        ...     PromptTestResult(variant_id="basic", response="", score=0.7, latency_ms=100),
        ...     PromptTestResult(variant_id="advanced", response="", score=0.9, latency_ms=100),
        ...     PromptTestResult(variant_id="expert", response="", score=0.85, latency_ms=100),
        ... ]
        >>>
        >>> experiment = PromptExperimentResult(
        ...     experiment_id="test",
        ...     variants=[],
        ...     results=results,
        ... )
        >>>
        >>> best = experiment.get_best_variant()
        >>> print(f"Best performing variant: {best}")
        Best performing variant: advanced

        Handling empty results:

        >>> empty_experiment = PromptExperimentResult(
        ...     experiment_id="empty",
        ...     variants=[],
        ...     results=[],
        ... )
        >>> print(empty_experiment.get_best_variant())
        None
        """
        averages = self.get_average_scores()
        if not averages:
            return None
        return max(averages, key=averages.get)  # type: ignore[arg-type]


class PromptVariationGenerator:
    """
    Generate variations of a prompt for systematic testing.

    This class provides a fluent interface for creating multiple prompt
    variations from a base prompt. It supports various modification
    strategies including prefixes, suffixes, instruction modifiers,
    and prompting strategy variations.

    Parameters
    ----------
    base_prompt : str
        The original prompt to generate variations from.

    Attributes
    ----------
    base_prompt : str
        The original prompt text.
    ROLE_PREFIXES : list[str]
        Default role-based prefix options.
    INSTRUCTION_MODIFIERS : list[str]
        Default instruction modifier options.
    OUTPUT_FORMATS : list[str]
        Default output format suffix options.

    Examples
    --------
    Basic usage with chaining:

    >>> from insideLLMs.prompts.prompt_testing import PromptVariationGenerator
    >>>
    >>> generator = PromptVariationGenerator("Explain machine learning")
    >>> variants = (
    ...     generator
    ...     .add_prefix_variations()
    ...     .add_suffix_variations()
    ...     .generate()
    ... )
    >>> print(f"Generated {len(variants)} variants")
    Generated 10 variants

    Using custom prefixes and suffixes:

    >>> generator = PromptVariationGenerator("Translate this to Spanish")
    >>> variants = (
    ...     generator
    ...     .add_prefix_variations(["As a native speaker,", "You are a translator."])
    ...     .add_suffix_variations(["Be formal.", "Use casual language."])
    ...     .generate()
    ... )
    >>> for v in variants:
    ...     print(f"{v.id}: {v.content[:50]}...")
    prefix_0: As a native speaker,...
    prefix_1: You are a translator....
    suffix_0: Translate this to Spanish...
    suffix_1: Translate this to Spanish...

    Generating strategy-based variations:

    >>> generator = PromptVariationGenerator("What is 25 * 17?")
    >>> variants = generator.add_strategy_variations().generate()
    >>> for v in variants:
    ...     print(f"{v.id}: {v.strategy.value if v.strategy else 'none'}")
    strategy_zero_shot: zero_shot
    strategy_cot: chain_of_thought
    strategy_step: step_by_step

    Generating all combinations:

    >>> generator = PromptVariationGenerator("Summarize this text")
    >>> variants = generator.generate_combinations(
    ...     prefixes=["", "You are an editor."],
    ...     suffixes=["", "Be concise.", "Include key points."],
    ...     max_combinations=10
    ... )
    >>> print(f"Generated {len(variants)} combinations")
    Generated 6 combinations

    See Also
    --------
    PromptVariant : The variant dataclass produced by this generator.
    PromptExperiment.add_variants_from_generator : Uses this generator.
    create_few_shot_variants : Specialized function for few-shot variants.
    """

    # Common system role prefixes
    ROLE_PREFIXES = [
        "You are an expert",
        "You are a helpful assistant",
        "You are a knowledgeable",
        "Act as an expert",
        "As a professional",
    ]

    # Instruction modifiers
    INSTRUCTION_MODIFIERS = [
        "Please",
        "Kindly",
        "Could you",
        "I need you to",
        "",  # No modifier
    ]

    # Output format instructions
    OUTPUT_FORMATS = [
        "Provide a clear and concise answer.",
        "Be thorough in your response.",
        "Keep your response brief.",
        "Explain step by step.",
        "Give a direct answer.",
    ]

    def __init__(self, base_prompt: str):
        """
        Initialize with a base prompt.

        Parameters
        ----------
        base_prompt : str
            The original prompt to generate variations from.

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import PromptVariationGenerator
        >>>
        >>> generator = PromptVariationGenerator("What causes rain?")
        >>> print(generator.base_prompt)
        What causes rain?
        """
        self.base_prompt = base_prompt
        self._variations: list[PromptVariant] = []

    def add_prefix_variations(
        self, prefixes: Optional[list[str]] = None
    ) -> "PromptVariationGenerator":
        """
        Add variations with different role/context prefixes.

        Creates prompt variants by prepending different prefixes to the
        base prompt, typically used for role-playing or context-setting.

        Parameters
        ----------
        prefixes : list[str], optional
            Custom prefix strings. If None, uses ROLE_PREFIXES defaults.

        Returns
        -------
        PromptVariationGenerator
            Self reference for method chaining.

        Examples
        --------
        Using default prefixes:

        >>> from insideLLMs.prompts.prompt_testing import PromptVariationGenerator
        >>>
        >>> generator = PromptVariationGenerator("Explain DNA")
        >>> generator.add_prefix_variations()
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>> variants = generator.generate()
        >>> print(variants[0].content[:30])
        You are an expert

        Using custom prefixes:

        >>> generator = PromptVariationGenerator("Debug this code")
        >>> generator.add_prefix_variations([
        ...     "As a senior developer,",
        ...     "You are a code reviewer.",
        ... ])
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>> variants = generator.generate()
        >>> for v in variants:
        ...     print(v.metadata.get("prefix", "")[:25])
        As a senior developer,
        You are a code reviewer.

        See Also
        --------
        add_suffix_variations : Add suffix variations.
        ROLE_PREFIXES : Default prefix options.
        """
        prefixes = prefixes or self.ROLE_PREFIXES
        for i, prefix in enumerate(prefixes):
            variant = PromptVariant(
                id=f"prefix_{i}",
                content=f"{prefix}\n\n{self.base_prompt}",
                metadata={"variation_type": "prefix", "prefix": prefix},
            )
            self._variations.append(variant)
        return self

    def add_suffix_variations(
        self, suffixes: Optional[list[str]] = None
    ) -> "PromptVariationGenerator":
        """
        Add variations with different output format suffixes.

        Creates prompt variants by appending different suffixes to the
        base prompt, typically used for specifying output format or style.

        Parameters
        ----------
        suffixes : list[str], optional
            Custom suffix strings. If None, uses OUTPUT_FORMATS defaults.

        Returns
        -------
        PromptVariationGenerator
            Self reference for method chaining.

        Examples
        --------
        Using default suffixes:

        >>> from insideLLMs.prompts.prompt_testing import PromptVariationGenerator
        >>>
        >>> generator = PromptVariationGenerator("List the planets")
        >>> generator.add_suffix_variations()
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>> variants = generator.generate()
        >>> # Check a suffix was added
        >>> print("concise" in variants[0].content)
        True

        Using custom suffixes:

        >>> generator = PromptVariationGenerator("Explain recursion")
        >>> generator.add_suffix_variations([
        ...     "Use an analogy.",
        ...     "Provide code examples.",
        ...     "Keep it under 100 words.",
        ... ])
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>> variants = generator.generate()
        >>> print(len(variants))
        3

        See Also
        --------
        add_prefix_variations : Add prefix variations.
        OUTPUT_FORMATS : Default suffix options.
        """
        suffixes = suffixes or self.OUTPUT_FORMATS
        for i, suffix in enumerate(suffixes):
            variant = PromptVariant(
                id=f"suffix_{i}",
                content=f"{self.base_prompt}\n\n{suffix}",
                metadata={"variation_type": "suffix", "suffix": suffix},
            )
            self._variations.append(variant)
        return self

    def add_instruction_variations(
        self, modifiers: Optional[list[str]] = None
    ) -> "PromptVariationGenerator":
        """
        Add variations with different instruction modifiers.

        Creates prompt variants by prepending politeness or directive
        modifiers to the base prompt (e.g., "Please", "Could you").

        Parameters
        ----------
        modifiers : list[str], optional
            Custom modifier strings. If None, uses INSTRUCTION_MODIFIERS.

        Returns
        -------
        PromptVariationGenerator
            Self reference for method chaining.

        Examples
        --------
        Using default modifiers:

        >>> from insideLLMs.prompts.prompt_testing import PromptVariationGenerator
        >>>
        >>> generator = PromptVariationGenerator("explain this concept")
        >>> generator.add_instruction_variations()
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>> variants = generator.generate()
        >>> print(variants[0].content)
        Please explain this concept

        Using custom modifiers:

        >>> generator = PromptVariationGenerator("summarize the article")
        >>> generator.add_instruction_variations([
        ...     "I would like you to",
        ...     "Your task is to",
        ...     "",  # No modifier
        ... ])
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>> variants = generator.generate()
        >>> print(variants[0].content)
        I would like you to summarize the article
        >>> print(variants[2].content)
        summarize the article

        See Also
        --------
        INSTRUCTION_MODIFIERS : Default modifier options.
        """
        modifiers = modifiers or self.INSTRUCTION_MODIFIERS
        for i, modifier in enumerate(modifiers):
            content = f"{modifier} {self.base_prompt}" if modifier else self.base_prompt
            variant = PromptVariant(
                id=f"modifier_{i}",
                content=content,
                metadata={"variation_type": "modifier", "modifier": modifier},
            )
            self._variations.append(variant)
        return self

    def add_strategy_variations(self) -> "PromptVariationGenerator":
        """
        Add variations using different prompting strategies.

        Creates prompt variants that apply chain-of-thought, step-by-step,
        and zero-shot strategies to the base prompt.

        Returns
        -------
        PromptVariationGenerator
            Self reference for method chaining.

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import PromptVariationGenerator
        >>>
        >>> generator = PromptVariationGenerator("What is 145 + 278?")
        >>> generator.add_strategy_variations()
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>> variants = generator.generate()
        >>>
        >>> for v in variants:
        ...     print(f"{v.id}: {v.strategy.value}")
        strategy_zero_shot: zero_shot
        strategy_cot: chain_of_thought
        strategy_step: step_by_step

        Checking CoT prompt content:

        >>> cot_variant = next(v for v in variants if v.id == "strategy_cot")
        >>> print("step by step" in cot_variant.content)
        True

        See Also
        --------
        PromptStrategy : Enumeration of available strategies.
        create_cot_variants : Dedicated function for CoT variants.
        """
        # Zero-shot (base)
        self._variations.append(
            PromptVariant(
                id="strategy_zero_shot",
                content=self.base_prompt,
                strategy=PromptStrategy.ZERO_SHOT,
                metadata={"variation_type": "strategy"},
            )
        )

        # Chain of thought
        cot_prompt = f"{self.base_prompt}\n\nLet's think through this step by step:"
        self._variations.append(
            PromptVariant(
                id="strategy_cot",
                content=cot_prompt,
                strategy=PromptStrategy.CHAIN_OF_THOUGHT,
                metadata={"variation_type": "strategy"},
            )
        )

        # Step by step
        step_prompt = f"{self.base_prompt}\n\nPlease solve this step by step, showing your work."
        self._variations.append(
            PromptVariant(
                id="strategy_step",
                content=step_prompt,
                strategy=PromptStrategy.STEP_BY_STEP,
                metadata={"variation_type": "strategy"},
            )
        )

        return self

    def add_custom_variation(
        self,
        variant_id: str,
        content: str,
        strategy: Optional[PromptStrategy] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "PromptVariationGenerator":
        """
        Add a custom variation with explicit content.

        Allows adding manually crafted prompt variants alongside
        auto-generated ones.

        Parameters
        ----------
        variant_id : str
            Unique identifier for this variant.
        content : str
            The complete prompt content.
        strategy : PromptStrategy, optional
            The prompting strategy used. Default is None.
        metadata : dict[str, Any], optional
            Additional metadata. Default is None.

        Returns
        -------
        PromptVariationGenerator
            Self reference for method chaining.

        Examples
        --------
        Adding a custom variant:

        >>> from insideLLMs.prompts.prompt_testing import (
        ...     PromptVariationGenerator,
        ...     PromptStrategy,
        ... )
        >>>
        >>> generator = PromptVariationGenerator("Explain gravity")
        >>> generator.add_custom_variation(
        ...     variant_id="eli5",
        ...     content="Explain gravity like I'm 5 years old.",
        ...     strategy=PromptStrategy.ROLE_PLAY,
        ...     metadata={"audience": "child", "complexity": "simple"}
        ... )
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>>
        >>> variants = generator.generate()
        >>> custom = next(v for v in variants if v.id == "eli5")
        >>> print(custom.metadata["audience"])
        child

        Mixing custom and generated variants:

        >>> generator = PromptVariationGenerator("Sort these numbers")
        >>> variants = (
        ...     generator
        ...     .add_strategy_variations()
        ...     .add_custom_variation(
        ...         "expert_cs",
        ...         "You are a computer scientist. Explain the most efficient algorithm to sort these numbers.",
        ...         strategy=PromptStrategy.ROLE_PLAY
        ...     )
        ...     .generate()
        ... )
        >>> print(len(variants))
        4

        See Also
        --------
        add_prefix_variations : Auto-generate prefix variants.
        add_strategy_variations : Auto-generate strategy variants.
        """
        self._variations.append(
            PromptVariant(
                id=variant_id,
                content=content,
                strategy=strategy,
                metadata=metadata or {},
            )
        )
        return self

    def add_temperature_hint_variations(self) -> "PromptVariationGenerator":
        """
        Add variations suggesting different creativity levels.

        Creates variants with wording that hints at different "temperature"
        settings - creative/exploratory vs precise/deterministic responses.

        Returns
        -------
        PromptVariationGenerator
            Self reference for method chaining.

        Notes
        -----
        These variations work by adjusting prompt wording rather than
        actual temperature parameters. Effectiveness depends on the
        model's interpretation of creativity/precision cues.

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import PromptVariationGenerator
        >>>
        >>> generator = PromptVariationGenerator("Write a poem about the ocean")
        >>> generator.add_temperature_hint_variations()
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>> variants = generator.generate()
        >>>
        >>> for v in variants:
        ...     print(f"{v.id}: hint={v.metadata.get('hint', 'none')}")
        temp_creative: hint=creative
        temp_precise: hint=precise

        Checking creative variant content:

        >>> creative = next(v for v in variants if v.id == "temp_creative")
        >>> print("creative" in creative.content.lower())
        True

        See Also
        --------
        add_strategy_variations : Strategy-based variations.
        """
        # Creative/high temperature hint
        creative = f"{self.base_prompt}\n\nBe creative and explore different possibilities."
        self._variations.append(
            PromptVariant(
                id="temp_creative",
                content=creative,
                metadata={"variation_type": "temperature_hint", "hint": "creative"},
            )
        )

        # Precise/low temperature hint
        precise = f"{self.base_prompt}\n\nBe precise and give the most accurate answer."
        self._variations.append(
            PromptVariant(
                id="temp_precise",
                content=precise,
                metadata={"variation_type": "temperature_hint", "hint": "precise"},
            )
        )

        return self

    def generate(self) -> list[PromptVariant]:
        """
        Generate all accumulated variations.

        Returns all prompt variants that have been added through the
        various add_* methods. If no variations were added, returns
        the base prompt as a single variant.

        Returns
        -------
        list[PromptVariant]
            List of all generated prompt variants.

        Examples
        --------
        Generating with no modifications:

        >>> from insideLLMs.prompts.prompt_testing import PromptVariationGenerator
        >>>
        >>> generator = PromptVariationGenerator("Hello world")
        >>> variants = generator.generate()
        >>> print(len(variants))
        1
        >>> print(variants[0].id)
        base
        >>> print(variants[0].content)
        Hello world

        Generating with multiple modifications:

        >>> generator = PromptVariationGenerator("Explain AI")
        >>> variants = (
        ...     generator
        ...     .add_prefix_variations(["You are an expert."])
        ...     .add_suffix_variations(["Be brief."])
        ...     .generate()
        ... )
        >>> print(len(variants))
        2

        See Also
        --------
        generate_combinations : Generate prefix/suffix combinations.
        """
        # Always include base prompt if no variations
        if not self._variations:
            return [PromptVariant(id="base", content=self.base_prompt)]
        return self._variations

    def generate_combinations(
        self,
        prefixes: Optional[list[str]] = None,
        suffixes: Optional[list[str]] = None,
        max_combinations: int = 20,
    ) -> list[PromptVariant]:
        """
        Generate combinatorial variations of prefixes and suffixes.

        Creates prompt variants by combining all prefixes with all
        suffixes (Cartesian product), with optional random sampling
        for large combination spaces.

        Parameters
        ----------
        prefixes : list[str], optional
            Prefix options. Default is ["", "You are an expert."].
        suffixes : list[str], optional
            Suffix options. Default is ["", "Be concise."].
        max_combinations : int, optional
            Maximum number of combinations to generate. If the total
            exceeds this, random sampling is applied. Default is 20.

        Returns
        -------
        list[PromptVariant]
            List of combined prompt variants.

        Examples
        --------
        Basic combinations:

        >>> from insideLLMs.prompts.prompt_testing import PromptVariationGenerator
        >>>
        >>> generator = PromptVariationGenerator("Explain photosynthesis")
        >>> variants = generator.generate_combinations(
        ...     prefixes=["", "You are a biologist."],
        ...     suffixes=["", "Use simple terms.", "Include a diagram description."]
        ... )
        >>> print(len(variants))
        6

        Limiting combinations with max_combinations:

        >>> variants = generator.generate_combinations(
        ...     prefixes=["A", "B", "C", "D", "E"],
        ...     suffixes=["1", "2", "3", "4", "5"],
        ...     max_combinations=10
        ... )
        >>> print(len(variants))
        10

        Examining combination metadata:

        >>> variants = generator.generate_combinations(
        ...     prefixes=["As an expert,"],
        ...     suffixes=["Be thorough."]
        ... )
        >>> print(variants[0].metadata["variation_type"])
        combination
        >>> print(variants[0].metadata["prefix"])
        As an expert,

        See Also
        --------
        generate : Generate all added variations.
        add_prefix_variations : Add prefixes separately.
        add_suffix_variations : Add suffixes separately.
        """
        prefixes = prefixes or ["", "You are an expert."]
        suffixes = suffixes or ["", "Be concise."]

        combinations = list(itertools.product(prefixes, suffixes))
        if len(combinations) > max_combinations:
            combinations = random.sample(combinations, max_combinations)

        variants = []
        for i, (prefix, suffix) in enumerate(combinations):
            parts = []
            if prefix:
                parts.append(prefix)
            parts.append(self.base_prompt)
            if suffix:
                parts.append(suffix)

            variants.append(
                PromptVariant(
                    id=f"combo_{i}",
                    content="\n\n".join(parts),
                    metadata={
                        "variation_type": "combination",
                        "prefix": prefix,
                        "suffix": suffix,
                    },
                )
            )

        return variants


@dataclass
class ScoringCriteria:
    """
    Configuration for a single scoring criterion.

    Defines one aspect of response evaluation including its name,
    weight, scoring function, and description.

    Parameters
    ----------
    name : str
        Name identifying this criterion (e.g., "length", "keywords").
    weight : float, optional
        Relative weight for combining with other criteria. Default is 1.0.
    scorer : callable, optional
        Function taking (prompt, response, expected) and returning a
        float score in [0.0, 1.0]. Default is None.
    description : str, optional
        Human-readable description of this criterion. Default is "".

    Attributes
    ----------
    name : str
        Criterion identifier.
    weight : float
        Relative importance weight.
    scorer : callable or None
        Scoring function.
    description : str
        Human-readable description.

    Examples
    --------
    Creating a custom scoring criterion:

    >>> from insideLLMs.prompts.prompt_testing import ScoringCriteria
    >>>
    >>> def politeness_scorer(prompt: str, response: str, expected: str) -> float:
    ...     polite_words = ["please", "thank you", "appreciate"]
    ...     response_lower = response.lower()
    ...     matches = sum(1 for w in polite_words if w in response_lower)
    ...     return min(1.0, matches / 2)
    >>>
    >>> criterion = ScoringCriteria(
    ...     name="politeness",
    ...     weight=0.5,
    ...     scorer=politeness_scorer,
    ...     description="Checks for polite language in response"
    ... )
    >>> print(criterion.name)
    politeness

    See Also
    --------
    PromptScorer : Uses ScoringCriteria for response evaluation.
    PromptScorer.add_criteria : Adds criteria to a scorer.
    """

    name: str
    weight: float = 1.0
    scorer: Optional[Callable[[str, str, str], float]] = None
    description: str = ""


class PromptScorer:
    """
    Score prompt responses based on configurable criteria.

    Provides a flexible system for evaluating model responses using
    multiple weighted criteria. Supports built-in criteria for length,
    keywords, format, and similarity, as well as custom scoring functions.

    Attributes
    ----------
    None (internal _criteria list is private)

    Examples
    --------
    Basic scorer with length and keyword criteria:

    >>> from insideLLMs.prompts.prompt_testing import PromptScorer
    >>>
    >>> scorer = PromptScorer()
    >>> scorer.add_length_criteria(min_length=50, max_length=200)
    <insideLLMs.prompt_testing.PromptScorer object at ...>
    >>> scorer.add_keyword_criteria(["python", "function", "return"])
    <insideLLMs.prompt_testing.PromptScorer object at ...>
    >>>
    >>> response = "In Python, a function uses the return statement to send back a value."
    >>> overall, details = scorer.score("Explain functions", response)
    >>> print(f"Overall: {overall:.2f}")
    Overall: 0.83
    >>> print(f"Length: {details['length']:.2f}, Keywords: {details['keywords']:.2f}")
    Length: 0.66, Keywords: 1.00

    Chaining criteria with custom weights:

    >>> scorer = (
    ...     PromptScorer()
    ...     .add_length_criteria(min_length=100, max_length=500, weight=1.0)
    ...     .add_keyword_criteria(["algorithm", "complexity"], weight=2.0)
    ...     .add_similarity_criteria(weight=1.5)
    ... )

    Adding a custom criterion:

    >>> def starts_with_answer(prompt: str, response: str, expected: str) -> float:
    ...     return 1.0 if response.lower().startswith("the answer") else 0.0
    >>>
    >>> scorer = PromptScorer()
    >>> scorer.add_criteria(
    ...     name="answer_format",
    ...     scorer=starts_with_answer,
    ...     weight=1.0,
    ...     description="Response should start with 'The answer'"
    ... )
    <insideLLMs.prompt_testing.PromptScorer object at ...>

    See Also
    --------
    ScoringCriteria : Individual criterion configuration.
    PromptABTestRunner : Uses scorer for experiment evaluation.
    PromptExperiment.configure_scorer : High-level scorer configuration.
    """

    def __init__(self):
        """
        Initialize with empty scoring criteria.

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import PromptScorer
        >>>
        >>> scorer = PromptScorer()
        >>> overall, details = scorer.score("prompt", "response")
        >>> print(overall)
        1.0
        >>> print(details)
        {}
        """
        self._criteria: list[ScoringCriteria] = []

    def add_criteria(
        self,
        name: str,
        scorer: Callable[[str, str, str], float],
        weight: float = 1.0,
        description: str = "",
    ) -> "PromptScorer":
        """
        Add a custom scoring criterion.

        The scorer function receives (prompt, response, expected) and
        should return a score between 0.0 and 1.0.

        Parameters
        ----------
        name : str
            Name of the criterion for identification in results.
        scorer : callable
            Function with signature (prompt, response, expected) -> float.
            Should return a value in [0.0, 1.0].
        weight : float, optional
            Weight for combining with other criteria. Default is 1.0.
        description : str, optional
            Human-readable description. Default is "".

        Returns
        -------
        PromptScorer
            Self reference for method chaining.

        Raises
        ------
        None (invalid scorers may cause runtime errors during scoring)

        Examples
        --------
        Adding a custom confidence scorer:

        >>> from insideLLMs.prompts.prompt_testing import PromptScorer
        >>>
        >>> def confidence_scorer(prompt: str, response: str, expected: str) -> float:
        ...     uncertain_words = ["maybe", "perhaps", "might", "possibly"]
        ...     response_lower = response.lower()
        ...     uncertain_count = sum(1 for w in uncertain_words if w in response_lower)
        ...     return max(0.0, 1.0 - (uncertain_count * 0.25))
        >>>
        >>> scorer = PromptScorer()
        >>> scorer.add_criteria(
        ...     name="confidence",
        ...     scorer=confidence_scorer,
        ...     weight=1.5,
        ...     description="Penalizes uncertain language"
        ... )
        <insideLLMs.prompt_testing.PromptScorer object at ...>
        >>>
        >>> # Test with uncertain response
        >>> score, details = scorer.score(
        ...     "What is 2+2?",
        ...     "The answer might possibly be 4, maybe."
        ... )
        >>> print(f"Confidence: {details['confidence']:.2f}")
        Confidence: 0.25

        See Also
        --------
        add_length_criteria : Built-in length criterion.
        add_keyword_criteria : Built-in keyword criterion.
        add_similarity_criteria : Built-in similarity criterion.
        """
        self._criteria.append(
            ScoringCriteria(
                name=name,
                weight=weight,
                scorer=scorer,
                description=description,
            )
        )
        return self

    def add_length_criteria(
        self,
        min_length: int = 10,
        max_length: int = 1000,
        weight: float = 1.0,
    ) -> "PromptScorer":
        """
        Add length-based scoring criterion.

        Scores responses based on character length, with full score
        for responses within the acceptable range and reduced scores
        for responses outside the range.

        Parameters
        ----------
        min_length : int, optional
            Minimum acceptable character length. Default is 10.
        max_length : int, optional
            Maximum acceptable character length. Default is 1000.
        weight : float, optional
            Weight for this criterion. Default is 1.0.

        Returns
        -------
        PromptScorer
            Self reference for method chaining.

        Notes
        -----
        - Responses shorter than min_length: score = length / min_length
        - Responses longer than max_length: score = 1 - (excess / max_length)
        - Responses within range: score = 1.0

        Examples
        --------
        Testing length scoring:

        >>> from insideLLMs.prompts.prompt_testing import PromptScorer
        >>>
        >>> scorer = PromptScorer()
        >>> scorer.add_length_criteria(min_length=50, max_length=200)
        <insideLLMs.prompt_testing.PromptScorer object at ...>
        >>>
        >>> # Too short
        >>> score, details = scorer.score("", "Short")  # 5 chars
        >>> print(f"Score: {details['length']:.2f}")
        Score: 0.10
        >>>
        >>> # Just right
        >>> score, details = scorer.score("", "x" * 100)
        >>> print(f"Score: {details['length']:.2f}")
        Score: 1.00
        >>>
        >>> # Too long
        >>> score, details = scorer.score("", "x" * 300)
        >>> print(f"Score: {details['length']:.2f}")
        Score: 0.50

        See Also
        --------
        add_keyword_criteria : Keyword-based scoring.
        add_format_criteria : Format compliance scoring.
        """

        def length_scorer(prompt: str, response: str, expected: str) -> float:
            length = len(response)
            if length < min_length:
                return length / min_length
            elif length > max_length:
                return max(0, 1 - (length - max_length) / max_length)
            return 1.0

        return self.add_criteria(
            "length",
            length_scorer,
            weight,
            f"Response length between {min_length} and {max_length}",
        )

    def add_keyword_criteria(
        self,
        required_keywords: list[str],
        weight: float = 1.0,
    ) -> "PromptScorer":
        """
        Add keyword presence scoring criterion.

        Scores responses based on the proportion of required keywords
        that appear in the response text.

        Parameters
        ----------
        required_keywords : list[str]
            Keywords that should appear in the response.
        weight : float, optional
            Weight for this criterion. Default is 1.0.

        Returns
        -------
        PromptScorer
            Self reference for method chaining.

        Notes
        -----
        - Matching is case-insensitive
        - Score = (matched keywords) / (total required keywords)
        - Empty keyword list results in score of 1.0

        Examples
        --------
        Testing keyword scoring:

        >>> from insideLLMs.prompts.prompt_testing import PromptScorer
        >>>
        >>> scorer = PromptScorer()
        >>> scorer.add_keyword_criteria(["neural", "network", "training", "loss"])
        <insideLLMs.prompt_testing.PromptScorer object at ...>
        >>>
        >>> response = "A neural network learns through training to minimize loss."
        >>> score, details = scorer.score("", response)
        >>> print(f"Keywords: {details['keywords']:.2f}")
        Keywords: 1.00
        >>>
        >>> partial_response = "A neural network is a type of model."
        >>> score, details = scorer.score("", partial_response)
        >>> print(f"Keywords: {details['keywords']:.2f}")
        Keywords: 0.50

        See Also
        --------
        add_length_criteria : Length-based scoring.
        add_similarity_criteria : Similarity-based scoring.
        """

        def keyword_scorer(prompt: str, response: str, expected: str) -> float:
            response_lower = response.lower()
            matches = sum(1 for kw in required_keywords if kw.lower() in response_lower)
            return matches / len(required_keywords) if required_keywords else 1.0

        return self.add_criteria(
            "keywords",
            keyword_scorer,
            weight,
            f"Contains keywords: {', '.join(required_keywords)}",
        )

    def add_similarity_criteria(
        self,
        weight: float = 1.0,
    ) -> "PromptScorer":
        """
        Add similarity to expected response scoring criterion.

        Scores responses based on word overlap with the expected
        response, using a simple set-based similarity metric.

        Parameters
        ----------
        weight : float, optional
            Weight for this criterion. Default is 1.0.

        Returns
        -------
        PromptScorer
            Self reference for method chaining.

        Notes
        -----
        - Uses word tokenization to extract words from both texts
        - Score = |response_words AND expected_words| / |expected_words|
        - Returns 1.0 if expected response is empty
        - This is a simple bag-of-words similarity, not semantic similarity

        Examples
        --------
        Testing similarity scoring:

        >>> from insideLLMs.prompts.prompt_testing import PromptScorer
        >>>
        >>> scorer = PromptScorer()
        >>> scorer.add_similarity_criteria()
        <insideLLMs.prompt_testing.PromptScorer object at ...>
        >>>
        >>> expected = "The capital of France is Paris."
        >>>
        >>> # High similarity
        >>> response = "Paris is the capital city of France."
        >>> score, details = scorer.score("", response, expected)
        >>> print(f"Similarity: {details['similarity']:.2f}")
        Similarity: 0.83
        >>>
        >>> # Lower similarity
        >>> response = "London is a major European city."
        >>> score, details = scorer.score("", response, expected)
        >>> print(f"Similarity: {details['similarity']:.2f}")
        Similarity: 0.17
        >>>
        >>> # No expected response
        >>> score, details = scorer.score("", response, "")
        >>> print(f"Similarity: {details['similarity']:.2f}")
        Similarity: 1.00

        See Also
        --------
        add_keyword_criteria : Keyword-based scoring.
        insideLLMs.nlp.tokenization.word_tokenize_regex : Tokenization function used.
        """

        def similarity_scorer(prompt: str, response: str, expected: str) -> float:
            if not expected:
                return 1.0  # No expected response to compare

            # Simple word overlap similarity
            response_words = set(word_tokenize_regex(response))
            expected_words = set(word_tokenize_regex(expected))

            if not expected_words:
                return 1.0

            overlap = len(response_words & expected_words)
            return overlap / len(expected_words)

        return self.add_criteria(
            "similarity",
            similarity_scorer,
            weight,
            "Similarity to expected response",
        )

    def add_format_criteria(
        self,
        expected_format: str,
        weight: float = 1.0,
    ) -> "PromptScorer":
        """
        Add format compliance scoring criterion.

        Scores responses based on whether they conform to an expected
        output format (JSON, Markdown, code, or bullet points).

        Parameters
        ----------
        expected_format : str
            Expected format type. Supported values:
            - "json": Valid JSON structure
            - "markdown": Markdown formatting indicators
            - "code": Code block or function definition
            - "bullets": Bullet point or numbered list format
        weight : float, optional
            Weight for this criterion. Default is 1.0.

        Returns
        -------
        PromptScorer
            Self reference for method chaining.

        Notes
        -----
        Format detection is heuristic-based:
        - JSON: Attempts to parse; partial credit for JSON-like structure
        - Markdown: Checks for #, ```, **, *, -, 1. indicators
        - Code: Checks for ``` or "def " at start
        - Bullets: Regex check for bullet or numbered list patterns

        Examples
        --------
        Testing JSON format scoring:

        >>> from insideLLMs.prompts.prompt_testing import PromptScorer
        >>>
        >>> scorer = PromptScorer()
        >>> scorer.add_format_criteria("json")
        <insideLLMs.prompt_testing.PromptScorer object at ...>
        >>>
        >>> # Valid JSON
        >>> score, details = scorer.score("", '{"name": "test", "value": 42}')
        >>> print(f"Format: {details['format']:.2f}")
        Format: 1.00
        >>>
        >>> # JSON-like but invalid
        >>> score, details = scorer.score("", "The result is {name: test}")
        >>> print(f"Format: {details['format']:.2f}")
        Format: 0.50
        >>>
        >>> # Not JSON
        >>> score, details = scorer.score("", "Just plain text")
        >>> print(f"Format: {details['format']:.2f}")
        Format: 0.00

        Testing Markdown format scoring:

        >>> scorer = PromptScorer()
        >>> scorer.add_format_criteria("markdown")
        <insideLLMs.prompt_testing.PromptScorer object at ...>
        >>>
        >>> md_response = "# Title\\n\\n**Bold text** and *italic*"
        >>> score, details = scorer.score("", md_response)
        >>> print(f"Format: {details['format']:.2f}")
        Format: 1.00

        Testing bullet format scoring:

        >>> scorer = PromptScorer()
        >>> scorer.add_format_criteria("bullets")
        <insideLLMs.prompt_testing.PromptScorer object at ...>
        >>>
        >>> bullets = "- First item\\n- Second item\\n- Third item"
        >>> score, details = scorer.score("", bullets)
        >>> print(f"Format: {details['format']:.2f}")
        Format: 1.00

        See Also
        --------
        add_length_criteria : Length-based scoring.
        add_keyword_criteria : Keyword-based scoring.
        """

        def format_scorer(prompt: str, response: str, expected: str) -> float:
            if expected_format == "json":
                try:
                    import json

                    json.loads(response)
                    return 1.0
                except json.JSONDecodeError:
                    # Check for JSON-like structure
                    if "{" in response and "}" in response:
                        return 0.5
                    return 0.0
            elif expected_format == "markdown":
                # Check for markdown indicators
                indicators = ["#", "```", "**", "*", "-", "1."]
                score = sum(1 for ind in indicators if ind in response)
                return min(1.0, score / 2)
            elif expected_format == "code":
                if "```" in response or response.strip().startswith("def "):
                    return 1.0
                return 0.3
            elif expected_format == "bullets":
                if re.search(r"^[\-\*\•]\s", response, re.MULTILINE):
                    return 1.0
                if re.search(r"^\d+\.\s", response, re.MULTILINE):
                    return 0.8
                return 0.0
            return 1.0

        return self.add_criteria(
            "format",
            format_scorer,
            weight,
            f"Response format: {expected_format}",
        )

    def score(
        self,
        prompt: str,
        response: str,
        expected: str = "",
    ) -> tuple[float, dict[str, float]]:
        """
        Score a response against all configured criteria.

        Evaluates the response using all added scoring criteria and
        computes a weighted average overall score.

        Parameters
        ----------
        prompt : str
            The prompt that was used to generate the response.
        response : str
            The model's response to score.
        expected : str, optional
            Expected response for comparison (used by similarity scorer).
            Default is "".

        Returns
        -------
        tuple[float, dict[str, float]]
            A tuple containing:
            - overall_score: Weighted average of all criteria (0.0 to 1.0)
            - individual_scores: Dict mapping criterion names to scores

        Examples
        --------
        Scoring with multiple criteria:

        >>> from insideLLMs.prompts.prompt_testing import PromptScorer
        >>>
        >>> scorer = (
        ...     PromptScorer()
        ...     .add_length_criteria(min_length=20, max_length=100)
        ...     .add_keyword_criteria(["python", "code"])
        ... )
        >>>
        >>> response = "Here is a simple Python code example for you."
        >>> overall, details = scorer.score(
        ...     prompt="Write Python code",
        ...     response=response
        ... )
        >>> print(f"Overall: {overall:.2f}")
        Overall: 1.00
        >>> print(f"Length: {details['length']:.2f}")
        Length: 1.00
        >>> print(f"Keywords: {details['keywords']:.2f}")
        Keywords: 1.00

        Scoring with weighted criteria:

        >>> scorer = (
        ...     PromptScorer()
        ...     .add_length_criteria(weight=1.0)
        ...     .add_keyword_criteria(["important"], weight=3.0)
        ... )
        >>>
        >>> response = "This is a response without the keyword."
        >>> overall, details = scorer.score("", response)
        >>> # keywords weight is 3x, so 0*3 + 1*1 = 1 / (3+1) = 0.25
        >>> print(f"Overall: {overall:.2f}")
        Overall: 0.25

        Empty scorer returns perfect score:

        >>> scorer = PromptScorer()
        >>> overall, details = scorer.score("", "Any response")
        >>> print(overall)
        1.0
        >>> print(details)
        {}

        See Also
        --------
        add_criteria : Add custom scoring criteria.
        PromptABTestRunner.test_variant : Uses this method.
        """
        if not self._criteria:
            return 1.0, {}

        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for criterion in self._criteria:
            if criterion.scorer:
                score = criterion.scorer(prompt, response, expected)
                scores[criterion.name] = score
                weighted_sum += score * criterion.weight
                total_weight += criterion.weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        return overall, scores


class PromptABTestRunner:
    """
    Run A/B tests on prompt variants.

    Executes systematic testing of multiple prompt variants against a
    model function, collecting scores and timing data for analysis.

    Parameters
    ----------
    scorer : PromptScorer, optional
        Scorer for evaluating responses. Default creates a new PromptScorer.

    Attributes
    ----------
    scorer : PromptScorer
        The scorer used for evaluation.

    Examples
    --------
    Basic A/B test:

    >>> from insideLLMs.prompts.prompt_testing import (
    ...     PromptABTestRunner,
    ...     PromptScorer,
    ...     PromptVariant,
    ... )
    >>>
    >>> # Create scorer
    >>> scorer = PromptScorer()
    >>> scorer.add_length_criteria(min_length=10, max_length=100)
    <insideLLMs.prompt_testing.PromptScorer object at ...>
    >>>
    >>> # Create runner
    >>> runner = PromptABTestRunner(scorer=scorer)
    >>>
    >>> # Define a mock model
    >>> def mock_model(prompt: str) -> str:
    ...     return f"Response to: {prompt[:20]}..."
    >>>
    >>> # Create variant
    >>> variant = PromptVariant(id="test", content="Explain machine learning")
    >>>
    >>> # Run test
    >>> results = runner.test_variant(variant, mock_model, num_runs=3)
    >>> print(f"Ran {len(results)} tests")
    Ran 3 tests

    Running a complete experiment:

    >>> variants = [
    ...     PromptVariant(id="short", content="Explain ML"),
    ...     PromptVariant(id="long", content="Please explain machine learning in detail"),
    ... ]
    >>>
    >>> experiment = runner.run_experiment(
    ...     variants=variants,
    ...     model_fn=mock_model,
    ...     runs_per_variant=2,
    ...     experiment_id="ml_explanation_test"
    ... )
    >>>
    >>> print(f"Best variant: {experiment.best_variant_id}")
    Best variant: long

    See Also
    --------
    PromptScorer : Configures response scoring.
    PromptExperimentResult : Structure for experiment results.
    PromptExperiment : High-level experiment orchestration.
    """

    def __init__(
        self,
        scorer: Optional[PromptScorer] = None,
    ):
        """
        Initialize the A/B test runner.

        Parameters
        ----------
        scorer : PromptScorer, optional
            Custom scorer for evaluating responses. If None, creates
            a new PromptScorer with no criteria (all responses score 1.0).

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import PromptABTestRunner, PromptScorer
        >>>
        >>> # With default scorer
        >>> runner = PromptABTestRunner()
        >>>
        >>> # With custom scorer
        >>> scorer = PromptScorer()
        >>> scorer.add_length_criteria(min_length=50)
        <insideLLMs.prompt_testing.PromptScorer object at ...>
        >>> runner = PromptABTestRunner(scorer=scorer)
        """
        self.scorer = scorer or PromptScorer()
        self._results: list[PromptTestResult] = []

    def test_variant(
        self,
        variant: PromptVariant,
        model_fn: Callable[[str], str],
        expected: str = "",
        num_runs: int = 1,
    ) -> list[PromptTestResult]:
        """
        Test a single variant multiple times.

        Executes the model function with the variant's prompt content,
        measures latency, and scores the response.

        Parameters
        ----------
        variant : PromptVariant
            The prompt variant to test.
        model_fn : callable
            Function that takes a prompt string and returns a response string.
        expected : str, optional
            Expected response for similarity scoring. Default is "".
        num_runs : int, optional
            Number of times to run the test. Default is 1.

        Returns
        -------
        list[PromptTestResult]
            List of test results, one per run.

        Notes
        -----
        - Errors during model execution are caught and stored in the result
        - Failed tests receive a score of 0.0
        - Latency is measured in milliseconds
        - Results are also accumulated in the runner's internal state

        Examples
        --------
        Single test run:

        >>> from insideLLMs.prompts.prompt_testing import (
        ...     PromptABTestRunner,
        ...     PromptVariant,
        ... )
        >>>
        >>> def simple_model(prompt: str) -> str:
        ...     return "This is a test response."
        >>>
        >>> runner = PromptABTestRunner()
        >>> variant = PromptVariant(id="v1", content="Test prompt")
        >>> results = runner.test_variant(variant, simple_model)
        >>>
        >>> print(f"Score: {results[0].score:.2f}")
        Score: 1.00
        >>> print(f"Latency: {results[0].latency_ms:.1f}ms")
        Latency: ...

        Multiple runs for statistical analysis:

        >>> results = runner.test_variant(variant, simple_model, num_runs=5)
        >>> print(f"Ran {len(results)} tests")
        Ran 5 tests
        >>> scores = [r.score for r in results]
        >>> print(f"Scores: {scores}")
        Scores: [1.0, 1.0, 1.0, 1.0, 1.0]

        Handling model errors:

        >>> def failing_model(prompt: str) -> str:
        ...     raise ValueError("API error")
        >>>
        >>> runner = PromptABTestRunner()
        >>> results = runner.test_variant(variant, failing_model)
        >>> print(f"Error: {results[0].error}")
        Error: API error
        >>> print(f"Score: {results[0].score}")
        Score: 0.0

        See Also
        --------
        run_experiment : Test multiple variants in one call.
        PromptTestResult : Structure for test results.
        """
        results = []
        for _ in range(num_runs):
            start_time = datetime.now()
            error = None

            try:
                response = model_fn(variant.content)
            except Exception as e:
                response = ""
                error = str(e)

            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            if error:
                score = 0.0
                score_details = {}
            else:
                score, score_details = self.scorer.score(variant.content, response, expected)

            result = PromptTestResult(
                variant_id=variant.id,
                response=response,
                score=score,
                latency_ms=latency_ms,
                metadata={
                    "score_details": score_details,
                    "expected": expected,
                },
                error=error,
            )
            results.append(result)
            self._results.append(result)

        return results

    def run_experiment(
        self,
        variants: list[PromptVariant],
        model_fn: Callable[[str], str],
        expected: str = "",
        runs_per_variant: int = 1,
        experiment_id: Optional[str] = None,
    ) -> PromptExperimentResult:
        """
        Run a complete experiment with multiple variants.

        Tests all provided variants, aggregates results, and computes
        summary statistics including the best-performing variant.

        Parameters
        ----------
        variants : list[PromptVariant]
            List of prompt variants to test.
        model_fn : callable
            Function that takes a prompt string and returns a response string.
        expected : str, optional
            Expected response for similarity scoring. Default is "".
        runs_per_variant : int, optional
            Number of test runs per variant. Default is 1.
        experiment_id : str, optional
            Unique identifier for the experiment. If None, auto-generated
            from timestamp.

        Returns
        -------
        PromptExperimentResult
            Complete experiment results with summary statistics.

        Examples
        --------
        Running a basic experiment:

        >>> from insideLLMs.prompts.prompt_testing import (
        ...     PromptABTestRunner,
        ...     PromptVariant,
        ...     PromptScorer,
        ... )
        >>>
        >>> # Setup
        >>> scorer = PromptScorer()
        >>> scorer.add_length_criteria(min_length=10, max_length=50)
        <insideLLMs.prompt_testing.PromptScorer object at ...>
        >>> runner = PromptABTestRunner(scorer=scorer)
        >>>
        >>> variants = [
        ...     PromptVariant(id="concise", content="Briefly explain X"),
        ...     PromptVariant(id="detailed", content="Explain X in full detail"),
        ... ]
        >>>
        >>> def model(prompt: str) -> str:
        ...     if "Briefly" in prompt:
        ...         return "X is a thing."  # 14 chars
        ...     return "X is a very complex concept that requires detailed explanation."  # 64 chars
        >>>
        >>> result = runner.run_experiment(
        ...     variants=variants,
        ...     model_fn=model,
        ...     runs_per_variant=2,
        ...     experiment_id="x_explanation"
        ... )
        >>>
        >>> print(f"Experiment: {result.experiment_id}")
        Experiment: x_explanation
        >>> print(f"Best: {result.best_variant_id}")
        Best: concise
        >>> print(f"Total runs: {result.summary['total_runs']}")
        Total runs: 4

        Analyzing experiment summary:

        >>> avg_scores = result.summary['average_scores']
        >>> for vid, score in sorted(avg_scores.items(), key=lambda x: -x[1]):
        ...     print(f"{vid}: {score:.2f}")
        concise: 1.00
        detailed: ...

        See Also
        --------
        test_variant : Test a single variant.
        PromptExperimentResult : Structure for experiment results.
        PromptExperiment : High-level experiment orchestration.
        """
        experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_results = []

        for variant in variants:
            results = self.test_variant(variant, model_fn, expected, runs_per_variant)
            all_results.extend(results)

        experiment = PromptExperimentResult(
            experiment_id=experiment_id,
            variants=variants,
            results=all_results,
            completed_at=datetime.now(),
        )

        # Compute summary
        avg_scores = experiment.get_average_scores()
        experiment.best_variant_id = experiment.get_best_variant()
        experiment.summary = {
            "average_scores": avg_scores,
            "best_variant": experiment.best_variant_id,
            "total_runs": len(all_results),
            "variants_tested": len(variants),
        }

        return experiment


@dataclass
class ExpandablePromptTemplate:
    """
    Template for generating prompts with variable substitution.

    Supports defining a prompt template with placeholders and a set of
    possible values for each variable, then expanding to all combinations.

    Parameters
    ----------
    template : str
        Prompt template with {variable} placeholders.
    variables : dict[str, list[Any]], optional
        Mapping of variable names to lists of possible values.
        Default is empty dict.

    Attributes
    ----------
    template : str
        The template string with placeholders.
    variables : dict[str, list[Any]]
        Variable name to value list mapping.

    Examples
    --------
    Basic template expansion:

    >>> from insideLLMs.prompts.prompt_testing import ExpandablePromptTemplate
    >>>
    >>> template = ExpandablePromptTemplate(
    ...     template="Explain {topic} to a {audience}.",
    ...     variables={
    ...         "topic": ["quantum physics", "machine learning"],
    ...         "audience": ["child", "expert"]
    ...     }
    ... )
    >>>
    >>> expansions = list(template.expand())
    >>> print(f"Generated {len(expansions)} prompts")
    Generated 4 prompts
    >>>
    >>> for prompt, vars in expansions[:2]:
    ...     print(f"Prompt: {prompt}")
    ...     print(f"Variables: {vars}")
    Prompt: Explain quantum physics to a child.
    Variables: {'topic': 'quantum physics', 'audience': 'child'}
    Prompt: Explain quantum physics to a expert.
    Variables: {'topic': 'quantum physics', 'audience': 'expert'}

    Template with no variables:

    >>> template = ExpandablePromptTemplate(
    ...     template="What is the meaning of life?"
    ... )
    >>> expansions = list(template.expand())
    >>> print(len(expansions))
    1
    >>> print(expansions[0][0])
    What is the meaning of life?

    Using with prompt experiments:

    >>> template = ExpandablePromptTemplate(
    ...     template="Write a {style} {format} about {topic}.",
    ...     variables={
    ...         "style": ["formal", "casual"],
    ...         "format": ["paragraph", "bullet list"],
    ...         "topic": ["AI ethics"]
    ...     }
    ... )
    >>>
    >>> from insideLLMs.prompts.prompt_testing import PromptVariant
    >>> variants = [
    ...     PromptVariant(id=f"v_{i}", content=prompt, metadata=vars)
    ...     for i, (prompt, vars) in enumerate(template.expand())
    ... ]
    >>> print(f"Created {len(variants)} variants")
    Created 4 variants

    See Also
    --------
    PromptVariationGenerator : Alternative approach to generating variants.
    PromptExperiment.add_variant : Add expanded variants to experiments.
    """

    template: str
    variables: dict[str, list[Any]] = field(default_factory=dict)

    def expand(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """
        Expand template with all variable combinations.

        Generates all possible prompts by substituting every combination
        of variable values into the template.

        Yields
        ------
        tuple[str, dict[str, Any]]
            Each yielded item is a tuple containing:
            - rendered_prompt: The template with variables substituted
            - variables_used: Dict of variable names to values used

        Notes
        -----
        - Uses Python's str.format() for substitution
        - Invalid variable references (KeyError) are silently skipped
        - Empty variables dict yields the original template once

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import ExpandablePromptTemplate
        >>>
        >>> template = ExpandablePromptTemplate(
        ...     template="Translate '{text}' to {language}.",
        ...     variables={
        ...         "text": ["Hello", "Goodbye"],
        ...         "language": ["Spanish", "French"]
        ...     }
        ... )
        >>>
        >>> for prompt, vars in template.expand():
        ...     print(f"{vars['text']} -> {vars['language']}: {prompt}")
        Hello -> Spanish: Translate 'Hello' to Spanish.
        Hello -> French: Translate 'Hello' to French.
        Goodbye -> Spanish: Translate 'Goodbye' to Spanish.
        Goodbye -> French: Translate 'Goodbye' to French.

        Handling missing variables gracefully:

        >>> template = ExpandablePromptTemplate(
        ...     template="Hello {name}, your {missing} is ready.",
        ...     variables={"name": ["Alice"]}
        ... )
        >>> # This will skip due to missing 'missing' variable
        >>> expansions = list(template.expand())
        >>> print(len(expansions))
        0
        """
        if not self.variables:
            yield self.template, {}
            return

        keys = list(self.variables.keys())
        values = [self.variables[k] for k in keys]

        for combo in itertools.product(*values):
            var_dict = dict(zip(keys, combo))
            try:
                rendered = self.template.format(**var_dict)
                yield rendered, var_dict
            except KeyError:
                continue


class PromptExperiment:
    """
    High-level experiment orchestration for prompt testing.

    Provides a convenient interface for setting up, running, and
    analyzing prompt testing experiments with minimal boilerplate.

    Parameters
    ----------
    name : str
        Name for this experiment, used as experiment ID.

    Attributes
    ----------
    name : str
        Experiment name.
    variants : list[PromptVariant]
        Configured prompt variants.
    scorer : PromptScorer
        Configured scorer instance.
    results : PromptExperimentResult or None
        Results after running the experiment.

    Examples
    --------
    Complete experiment workflow:

    >>> from insideLLMs.prompts.prompt_testing import PromptExperiment
    >>>
    >>> # Create experiment
    >>> experiment = PromptExperiment("summarization_test")
    >>>
    >>> # Add variants
    >>> experiment.add_variant(
    ...     "Summarize the following text:",
    ...     variant_id="basic"
    ... )
    <insideLLMs.prompt_testing.PromptExperiment object at ...>
    >>> experiment.add_variant(
    ...     "You are an expert editor. Provide a concise summary:",
    ...     variant_id="expert_role"
    ... )
    <insideLLMs.prompt_testing.PromptExperiment object at ...>
    >>>
    >>> # Configure scoring
    >>> experiment.configure_scorer(
    ...     length_range=(50, 200),
    ...     required_keywords=["summary", "key points"]
    ... )
    <insideLLMs.prompt_testing.PromptExperiment object at ...>
    >>>
    >>> # Run experiment
    >>> def mock_model(prompt: str) -> str:
    ...     return "Here is a summary with key points about the topic."
    >>>
    >>> results = experiment.run(mock_model, runs_per_variant=2)
    >>> print(f"Best: {experiment.get_best_prompt()[:30]}...")
    Best: ...

    Using with a PromptVariationGenerator:

    >>> from insideLLMs.prompts.prompt_testing import (
    ...     PromptExperiment,
    ...     PromptVariationGenerator,
    ... )
    >>>
    >>> generator = PromptVariationGenerator("Explain quantum entanglement")
    >>> generator.add_strategy_variations()
    <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
    >>>
    >>> experiment = PromptExperiment("quantum_test")
    >>> experiment.add_variants_from_generator(generator)
    <insideLLMs.prompt_testing.PromptExperiment object at ...>
    >>>
    >>> print(f"Added {len(experiment.variants)} variants")
    Added 3 variants

    Generating a report:

    >>> experiment = PromptExperiment("demo")
    >>> experiment.add_variant("Test prompt A", variant_id="a")
    <insideLLMs.prompt_testing.PromptExperiment object at ...>
    >>> experiment.add_variant("Test prompt B", variant_id="b")
    <insideLLMs.prompt_testing.PromptExperiment object at ...>
    >>>
    >>> results = experiment.run(lambda p: "Response", runs_per_variant=1)
    >>> report = experiment.generate_report()
    >>> print("# Prompt Experiment:" in report)
    True

    See Also
    --------
    PromptABTestRunner : Lower-level testing interface.
    PromptVariationGenerator : Generate variants programmatically.
    PromptScorer : Configure response scoring.
    """

    def __init__(self, name: str):
        """
        Initialize an experiment.

        Parameters
        ----------
        name : str
            Name for this experiment. Used as experiment ID in results.

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import PromptExperiment
        >>>
        >>> experiment = PromptExperiment("my_first_experiment")
        >>> print(experiment.name)
        my_first_experiment
        >>> print(len(experiment.variants))
        0
        """
        self.name = name
        self.variants: list[PromptVariant] = []
        self.scorer = PromptScorer()
        self.results: Optional[PromptExperimentResult] = None

    def add_variant(
        self,
        content: str,
        variant_id: Optional[str] = None,
        strategy: Optional[PromptStrategy] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "PromptExperiment":
        """
        Add a prompt variant to the experiment.

        Parameters
        ----------
        content : str
            The prompt text content.
        variant_id : str, optional
            Unique ID for this variant. Auto-generated if not provided.
        strategy : PromptStrategy, optional
            The prompting strategy used. Default is None.
        metadata : dict[str, Any], optional
            Additional metadata. Default is None.

        Returns
        -------
        PromptExperiment
            Self reference for method chaining.

        Examples
        --------
        Adding variants with chaining:

        >>> from insideLLMs.prompts.prompt_testing import PromptExperiment, PromptStrategy
        >>>
        >>> experiment = (
        ...     PromptExperiment("test")
        ...     .add_variant("Basic prompt", variant_id="basic")
        ...     .add_variant(
        ...         "Let's think step by step. Basic prompt",
        ...         variant_id="cot",
        ...         strategy=PromptStrategy.CHAIN_OF_THOUGHT
        ...     )
        ...     .add_variant(
        ...         "Expert prompt",
        ...         metadata={"author": "team_a"}
        ...     )
        ... )
        >>>
        >>> print(len(experiment.variants))
        3
        >>> print(experiment.variants[2].id)
        variant_2

        See Also
        --------
        add_variants_from_generator : Add multiple variants at once.
        PromptVariant : Variant dataclass structure.
        """
        variant_id = variant_id or f"variant_{len(self.variants)}"
        self.variants.append(
            PromptVariant(
                id=variant_id,
                content=content,
                strategy=strategy,
                metadata=metadata or {},
            )
        )
        return self

    def add_variants_from_generator(
        self, generator: PromptVariationGenerator
    ) -> "PromptExperiment":
        """
        Add all variants from a PromptVariationGenerator.

        Parameters
        ----------
        generator : PromptVariationGenerator
            Generator with configured variations.

        Returns
        -------
        PromptExperiment
            Self reference for method chaining.

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import (
        ...     PromptExperiment,
        ...     PromptVariationGenerator,
        ... )
        >>>
        >>> generator = PromptVariationGenerator("Explain AI")
        >>> generator.add_prefix_variations(["You are an expert."])
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>> generator.add_suffix_variations(["Be concise."])
        <insideLLMs.prompt_testing.PromptVariationGenerator object at ...>
        >>>
        >>> experiment = PromptExperiment("ai_test")
        >>> experiment.add_variants_from_generator(generator)
        <insideLLMs.prompt_testing.PromptExperiment object at ...>
        >>>
        >>> print(len(experiment.variants))
        2

        See Also
        --------
        add_variant : Add individual variants.
        PromptVariationGenerator : Generate variants programmatically.
        """
        self.variants.extend(generator.generate())
        return self

    def configure_scorer(
        self,
        length_range: Optional[tuple[int, int]] = None,
        required_keywords: Optional[list[str]] = None,
        expected_format: Optional[str] = None,
        check_similarity: bool = False,
    ) -> "PromptExperiment":
        """
        Configure the scorer with common criteria.

        Provides a convenient interface for adding standard scoring
        criteria without directly manipulating the PromptScorer.

        Parameters
        ----------
        length_range : tuple[int, int], optional
            (min_length, max_length) for length scoring. Default is None.
        required_keywords : list[str], optional
            Keywords that should appear in responses. Default is None.
        expected_format : str, optional
            Expected format ("json", "markdown", "code", "bullets").
            Default is None.
        check_similarity : bool, optional
            Whether to enable similarity scoring. Default is False.

        Returns
        -------
        PromptExperiment
            Self reference for method chaining.

        Examples
        --------
        Configuring all criteria:

        >>> from insideLLMs.prompts.prompt_testing import PromptExperiment
        >>>
        >>> experiment = (
        ...     PromptExperiment("test")
        ...     .configure_scorer(
        ...         length_range=(100, 500),
        ...         required_keywords=["algorithm", "complexity"],
        ...         expected_format="markdown",
        ...         check_similarity=True
        ...     )
        ... )

        Minimal configuration:

        >>> experiment = (
        ...     PromptExperiment("quick_test")
        ...     .configure_scorer(length_range=(50, 200))
        ... )

        See Also
        --------
        PromptScorer : Underlying scorer with more options.
        PromptScorer.add_criteria : Add custom criteria.
        """
        if length_range:
            self.scorer.add_length_criteria(length_range[0], length_range[1])
        if required_keywords:
            self.scorer.add_keyword_criteria(required_keywords)
        if expected_format:
            self.scorer.add_format_criteria(expected_format)
        if check_similarity:
            self.scorer.add_similarity_criteria()
        return self

    def run(
        self,
        model_fn: Callable[[str], str],
        expected: str = "",
        runs_per_variant: int = 1,
    ) -> PromptExperimentResult:
        """
        Run the experiment with all configured variants.

        Executes all variants against the model function and collects
        results. After running, results are stored in self.results.

        Parameters
        ----------
        model_fn : callable
            Function that takes a prompt string and returns a response.
        expected : str, optional
            Expected response for similarity scoring. Default is "".
        runs_per_variant : int, optional
            Number of times to test each variant. Default is 1.

        Returns
        -------
        PromptExperimentResult
            Complete experiment results with summary statistics.

        Examples
        --------
        Running an experiment:

        >>> from insideLLMs.prompts.prompt_testing import PromptExperiment
        >>>
        >>> experiment = (
        ...     PromptExperiment("test")
        ...     .add_variant("Prompt A", variant_id="a")
        ...     .add_variant("Prompt B", variant_id="b")
        ...     .configure_scorer(length_range=(10, 100))
        ... )
        >>>
        >>> def model(prompt: str) -> str:
        ...     return f"Response to {prompt}"
        >>>
        >>> results = experiment.run(model, runs_per_variant=3)
        >>>
        >>> print(f"Total runs: {len(results.results)}")
        Total runs: 6
        >>> print(f"Best: {results.best_variant_id}")
        Best: ...

        Accessing results after running:

        >>> print(experiment.results is not None)
        True
        >>> print(experiment.results.experiment_id)
        test

        See Also
        --------
        get_best_prompt : Get the winning prompt content.
        generate_report : Create a text report of results.
        PromptABTestRunner.run_experiment : Lower-level method used.
        """
        runner = PromptABTestRunner(scorer=self.scorer)
        self.results = runner.run_experiment(
            variants=self.variants,
            model_fn=model_fn,
            expected=expected,
            runs_per_variant=runs_per_variant,
            experiment_id=self.name,
        )
        return self.results

    def get_best_prompt(self) -> Optional[str]:
        """
        Get the best performing prompt content.

        Returns the content of the variant with the highest average
        score after the experiment has been run.

        Returns
        -------
        str or None
            The prompt content of the best variant, or None if the
            experiment hasn't been run or no best variant exists.

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import PromptExperiment
        >>>
        >>> experiment = (
        ...     PromptExperiment("test")
        ...     .add_variant("Short prompt", variant_id="short")
        ...     .add_variant("This is a longer prompt with more detail", variant_id="long")
        ...     .configure_scorer(length_range=(20, 100))
        ... )
        >>>
        >>> # Before running
        >>> print(experiment.get_best_prompt())
        None
        >>>
        >>> # After running
        >>> results = experiment.run(lambda p: p * 2)
        >>> best = experiment.get_best_prompt()
        >>> print(best is not None)
        True

        See Also
        --------
        run : Execute the experiment first.
        PromptExperimentResult.get_best_variant : Get the variant ID.
        """
        if not self.results or not self.results.best_variant_id:
            return None

        for variant in self.variants:
            if variant.id == self.results.best_variant_id:
                return variant.content
        return None

    def generate_report(self) -> str:
        """
        Generate a text report of experiment results.

        Creates a formatted Markdown report showing rankings,
        scores, and the best-performing prompt.

        Returns
        -------
        str
            Formatted report string in Markdown format.

        Examples
        --------
        >>> from insideLLMs.prompts.prompt_testing import PromptExperiment
        >>>
        >>> experiment = (
        ...     PromptExperiment("demo")
        ...     .add_variant("Prompt A", variant_id="a")
        ...     .add_variant("Prompt B", variant_id="b")
        ... )
        >>>
        >>> # Before running
        >>> print(experiment.generate_report())
        Experiment has not been run yet.
        >>>
        >>> # After running
        >>> results = experiment.run(lambda p: "Response")
        >>> report = experiment.generate_report()
        >>> print("# Prompt Experiment: demo" in report)
        True
        >>> print("## Results by Variant" in report)
        True
        >>> print("## Best Prompt" in report)
        True

        See Also
        --------
        run : Execute the experiment first.
        PromptExperimentResult : Contains the data for reporting.
        """
        if not self.results:
            return "Experiment has not been run yet."

        lines = [
            f"# Prompt Experiment: {self.name}",
            "",
            f"**Total Variants:** {len(self.variants)}",
            f"**Total Runs:** {len(self.results.results)}",
            "",
            "## Results by Variant",
            "",
        ]

        avg_scores = self.results.get_average_scores()
        sorted_variants = sorted(
            avg_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for rank, (variant_id, avg_score) in enumerate(sorted_variants, 1):
            indicator = " *" if variant_id == self.results.best_variant_id else ""
            lines.append(f"{rank}. **{variant_id}**: {avg_score:.3f}{indicator}")

        lines.extend(
            [
                "",
                "## Best Prompt",
                "",
                "```",
                self.get_best_prompt() or "N/A",
                "```",
            ]
        )

        return "\n".join(lines)


def create_few_shot_variants(
    task: str,
    examples: list[dict[str, str]],
    query: str,
    num_shots: list[int] | None = None,
) -> list[PromptVariant]:
    """
    Create prompt variants with different numbers of few-shot examples.

    Generates a set of prompts ranging from zero-shot to n-shot,
    allowing comparison of how example count affects model performance.

    Parameters
    ----------
    task : str
        The task description or instruction.
    examples : list[dict[str, str]]
        Pool of examples to draw from. Each example should be a dict
        with keys like "input"/"output" or "question"/"answer".
    query : str
        The actual query or input to evaluate.
    num_shots : list[int], optional
        List of example counts to try. Default is [0, 1, 3, 5].

    Returns
    -------
    list[PromptVariant]
        List of variants with different example counts.

    Examples
    --------
    Creating few-shot variants for math problems:

    >>> from insideLLMs.prompts.prompt_testing import create_few_shot_variants
    >>>
    >>> examples = [
    ...     {"input": "2 + 3", "output": "5"},
    ...     {"input": "10 - 4", "output": "6"},
    ...     {"input": "3 * 7", "output": "21"},
    ...     {"input": "15 / 3", "output": "5"},
    ...     {"input": "8 + 9", "output": "17"},
    ... ]
    >>>
    >>> variants = create_few_shot_variants(
    ...     task="Calculate the result of the math expression.",
    ...     examples=examples,
    ...     query="12 + 15",
    ...     num_shots=[0, 1, 3]
    ... )
    >>>
    >>> print(len(variants))
    3
    >>> print([v.id for v in variants])
    ['0_shot', '1_shot', '3_shot']

    Examining variant content:

    >>> zero_shot = variants[0]
    >>> print(zero_shot.strategy.value)
    zero_shot
    >>> print("Examples:" not in zero_shot.content)
    True
    >>>
    >>> three_shot = variants[2]
    >>> print(three_shot.strategy.value)
    few_shot
    >>> print("Examples:" in three_shot.content)
    True
    >>> print(three_shot.metadata["num_examples"])
    3

    Using with question/answer format:

    >>> qa_examples = [
    ...     {"question": "What is the capital of France?", "answer": "Paris"},
    ...     {"question": "What is 2+2?", "answer": "4"},
    ... ]
    >>>
    >>> variants = create_few_shot_variants(
    ...     task="Answer the following question.",
    ...     examples=qa_examples,
    ...     query="What is the largest planet?",
    ...     num_shots=[0, 2]
    ... )
    >>> print(len(variants))
    2

    See Also
    --------
    create_cot_variants : Create chain-of-thought variants.
    PromptVariationGenerator : General-purpose variant generation.
    PromptStrategy.FEW_SHOT : The strategy used for n>0 variants.
    """
    if num_shots is None:
        num_shots = [0, 1, 3, 5]
    variants = []

    for n in num_shots:
        if n == 0:
            # Zero-shot
            content = f"{task}\n\nQuery: {query}"
            strategy = PromptStrategy.ZERO_SHOT
        else:
            # Few-shot
            selected_examples = examples[:n]
            example_text = "\n\n".join(
                f"Input: {ex.get('input', ex.get('question', ''))}\n"
                f"Output: {ex.get('output', ex.get('answer', ''))}"
                for ex in selected_examples
            )
            content = f"{task}\n\nExamples:\n{example_text}\n\nQuery: {query}"
            strategy = PromptStrategy.FEW_SHOT

        variants.append(
            PromptVariant(
                id=f"{n}_shot",
                content=content,
                strategy=strategy,
                metadata={"num_examples": n},
            )
        )

    return variants


def create_cot_variants(
    prompt: str,
) -> list[PromptVariant]:
    """
    Create chain-of-thought prompt variants.

    Generates multiple versions of a prompt with different
    chain-of-thought triggers, from no CoT to explicit reasoning cues.

    Parameters
    ----------
    prompt : str
        The base prompt to create variants from.

    Returns
    -------
    list[PromptVariant]
        List of four variants:
        - no_cot: Original prompt without CoT
        - cot_basic: "Let's think step by step."
        - cot_detailed: "Let's work through this step by step: 1."
        - cot_reasoning: "Before answering, let me reason through this carefully:"

    Examples
    --------
    Creating CoT variants for a math problem:

    >>> from insideLLMs.prompts.prompt_testing import create_cot_variants
    >>>
    >>> variants = create_cot_variants("What is 145 * 23?")
    >>>
    >>> print(len(variants))
    4
    >>> print([v.id for v in variants])
    ['no_cot', 'cot_basic', 'cot_detailed', 'cot_reasoning']

    Examining variant strategies:

    >>> for v in variants:
    ...     print(f"{v.id}: {v.strategy.value}")
    no_cot: zero_shot
    cot_basic: chain_of_thought
    cot_detailed: chain_of_thought
    cot_reasoning: chain_of_thought

    Checking variant content:

    >>> cot_basic = next(v for v in variants if v.id == "cot_basic")
    >>> print("step by step" in cot_basic.content)
    True
    >>>
    >>> cot_detailed = next(v for v in variants if v.id == "cot_detailed")
    >>> print("1." in cot_detailed.content)
    True

    Using in an experiment:

    >>> from insideLLMs.prompts.prompt_testing import PromptExperiment
    >>>
    >>> experiment = PromptExperiment("cot_comparison")
    >>> for variant in create_cot_variants("Solve: 3x + 5 = 20"):
    ...     experiment.add_variant(
    ...         variant.content,
    ...         variant_id=variant.id,
    ...         strategy=variant.strategy
    ...     )
    <insideLLMs.prompt_testing.PromptExperiment object at ...>
    >>>
    >>> print(len(experiment.variants))
    4

    See Also
    --------
    create_few_shot_variants : Create few-shot variants.
    PromptVariationGenerator.add_strategy_variations : Alternative approach.
    PromptStrategy.CHAIN_OF_THOUGHT : The strategy used for CoT variants.
    """
    return [
        PromptVariant(
            id="no_cot",
            content=prompt,
            strategy=PromptStrategy.ZERO_SHOT,
        ),
        PromptVariant(
            id="cot_basic",
            content=f"{prompt}\n\nLet's think step by step.",
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        ),
        PromptVariant(
            id="cot_detailed",
            content=f"{prompt}\n\nLet's work through this step by step:\n1.",
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        ),
        PromptVariant(
            id="cot_reasoning",
            content=f"{prompt}\n\nBefore answering, let me reason through this carefully:",
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        ),
    ]


# Backwards-compatible alias (deprecated)
TestResult = PromptTestResult

# Older code and tests may import ExperimentResult. The canonical name is
# PromptExperimentResult.
ExperimentResult = PromptExperimentResult

# Older code and tests may import PromptTemplate. The canonical name is
# ExpandablePromptTemplate.
PromptTemplate = ExpandablePromptTemplate

# Older code and tests may import ABTestRunner. The canonical name is
# PromptABTestRunner.
ABTestRunner = PromptABTestRunner
