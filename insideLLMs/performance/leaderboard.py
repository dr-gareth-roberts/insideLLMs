"""Benchmark leaderboard generation and management for LLM evaluation.

This module provides a comprehensive toolkit for creating, managing, and analyzing
benchmark leaderboards from Large Language Model (LLM) evaluation results. It supports
multiple ranking methodologies, score aggregation strategies, historical tracking,
and various export formats.

Overview
--------
The leaderboard system is designed around a pipeline architecture:

1. **Score Collection**: Gather `ModelScore` objects from evaluation runs
2. **Aggregation**: Combine scores across benchmarks using `ScoreAggregator`
3. **Ranking**: Rank models using `ModelRanker` with configurable methods
4. **Building**: Construct `Leaderboard` objects via `LeaderboardBuilder`
5. **Formatting**: Export to various formats using `LeaderboardFormatter`
6. **Analysis**: Compare and analyze trends with `LeaderboardAnalyzer`

Key Components
--------------
Enumerations
    - `RankingMethod`: Defines ranking strategies (score, wins, ELO, percentile)
    - `ScoreAggregation`: Defines aggregation methods (mean, median, weighted, etc.)
    - `TrendDirection`: Indicates performance trend (up, down, stable, new)

Data Classes
    - `ModelScore`: Individual benchmark score for a model
    - `LeaderboardEntry`: Single entry in a leaderboard with rank and trend
    - `Leaderboard`: Complete leaderboard with entries and metadata
    - `LeaderboardComparison`: Result of comparing two leaderboards

Core Classes
    - `ScoreAggregator`: Aggregates multiple scores into a single value
    - `ModelRanker`: Ranks models based on aggregated scores
    - `LeaderboardBuilder`: Builds leaderboards from collected scores
    - `LeaderboardFormatter`: Formats leaderboards for export
    - `LeaderboardAnalyzer`: Analyzes and compares leaderboards

Convenience Functions
    - `create_leaderboard()`: Quick leaderboard creation from scores
    - `format_leaderboard()`: Format leaderboard to string
    - `compare_leaderboards()`: Compare two leaderboards
    - `quick_leaderboard()`: Create leaderboard from dictionary

Examples
--------
Basic leaderboard creation:

>>> from insideLLMs.performance.leaderboard import ModelScore, create_leaderboard
>>> scores = [
...     ModelScore("gpt-4", "mmlu", 0.89),
...     ModelScore("gpt-4", "hellaswag", 0.95),
...     ModelScore("claude-3", "mmlu", 0.91),
...     ModelScore("claude-3", "hellaswag", 0.93),
...     ModelScore("llama-3", "mmlu", 0.78),
...     ModelScore("llama-3", "hellaswag", 0.82),
... ]
>>> lb = create_leaderboard(scores, "LLM Benchmark", "Q1 2024 Results")
>>> print(lb.top_model)
'claude-3'

Quick leaderboard from dictionary:

>>> from insideLLMs.performance.leaderboard import quick_leaderboard
>>> results = {
...     "gpt-4": {"coding": 0.92, "reasoning": 0.88},
...     "claude-3": {"coding": 0.90, "reasoning": 0.91},
...     "gemini": {"coding": 0.85, "reasoning": 0.87},
... }
>>> lb = quick_leaderboard(results, "Capability Comparison")
>>> for entry in lb.entries:
...     print(f"{entry.rank}. {entry.model_id}: {entry.score:.3f}")
1. claude-3: 0.905
2. gpt-4: 0.900
3. gemini: 0.860

Weighted scoring:

>>> from insideLLMs.performance.leaderboard import (
...     ModelScore, ScoreAggregator, ScoreAggregation,
...     ModelRanker, LeaderboardBuilder
... )
>>> weights = {"safety": 2.0, "capability": 1.0, "speed": 0.5}
>>> aggregator = ScoreAggregator(ScoreAggregation.WEIGHTED, weights)
>>> ranker = ModelRanker(aggregator=aggregator)
>>> builder = LeaderboardBuilder(ranker=ranker)
>>> builder.add_scores([
...     ModelScore("model-a", "safety", 0.95),
...     ModelScore("model-a", "capability", 0.80),
...     ModelScore("model-a", "speed", 0.70),
...     ModelScore("model-b", "safety", 0.85),
...     ModelScore("model-b", "capability", 0.92),
...     ModelScore("model-b", "speed", 0.90),
... ])
>>> lb = builder.build("Weighted Leaderboard", metric_weights=weights)

Exporting to multiple formats:

>>> from insideLLMs.performance.leaderboard import format_leaderboard
>>> markdown = format_leaderboard(lb, "markdown")
>>> csv_data = format_leaderboard(lb, "csv")
>>> json_data = format_leaderboard(lb, "json")
>>> html_page = format_leaderboard(lb, "html")

Comparing leaderboards:

>>> from insideLLMs.performance.leaderboard import compare_leaderboards
>>> comparison = compare_leaderboards(lb_january, lb_february)
>>> print(f"Rank correlation: {comparison.rank_correlations:.2f}")
>>> print(f"Divergent models: {len(comparison.divergent_rankings)}")

Notes
-----
- Scores are typically normalized to [0, 1] range for consistency
- The ELO ranking method requires pairwise comparison data
- Historical tracking enables trend analysis across time
- All timestamps use ISO 8601 format
- Thread-safe for read operations; write operations should be synchronized

See Also
--------
insideLLMs.evaluation : Evaluation framework for generating scores
insideLLMs.metrics : Metric definitions and calculations
insideLLMs.visualization : Visualization tools for leaderboards
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class RankingMethod(Enum):
    """Methods for ranking models in a leaderboard.

    This enumeration defines the available strategies for ordering models
    based on their performance scores. Each method has different characteristics
    suitable for various evaluation scenarios.

    Attributes
    ----------
    SCORE : str
        Direct score-based ranking. Models are ordered by their aggregated
        score in descending order. Simple and transparent but sensitive to
        score distribution.
    WINS : str
        Win-count based ranking. Models are ranked by the number of benchmarks
        where they achieved the highest score. Good for identifying consistent
        top performers across diverse tasks.
    ELO : str
        ELO rating system ranking. Borrowed from chess, this method uses
        pairwise comparisons to compute ratings. Handles transitivity well
        and works with incomplete comparison data.
    PERCENTILE : str
        Percentile-based ranking. Models are ranked by their average percentile
        position across benchmarks. Normalizes for different score scales
        across benchmarks.
    WEIGHTED_AVERAGE : str
        Weighted average ranking. Combines scores using configurable weights
        for each benchmark. Allows prioritizing certain benchmarks over others.

    Examples
    --------
    Using score-based ranking (default):

    >>> from insideLLMs.performance.leaderboard import ModelRanker, RankingMethod
    >>> ranker = ModelRanker(method=RankingMethod.SCORE)
    >>> scores = {
    ...     "gpt-4": {"mmlu": 0.89, "arc": 0.92},
    ...     "claude": {"mmlu": 0.91, "arc": 0.88},
    ... }
    >>> entries = ranker.rank(scores)
    >>> print(entries[0].model_id)
    'gpt-4'

    Using weighted average for custom prioritization:

    >>> from insideLLMs.performance.leaderboard import (
    ...     ModelRanker, RankingMethod, ScoreAggregator, ScoreAggregation
    ... )
    >>> weights = {"safety": 3.0, "capability": 1.0}
    >>> aggregator = ScoreAggregator(ScoreAggregation.WEIGHTED, weights)
    >>> ranker = ModelRanker(
    ...     method=RankingMethod.WEIGHTED_AVERAGE,
    ...     aggregator=aggregator
    ... )

    Selecting ranking method based on use case:

    >>> method = RankingMethod.ELO  # For competitive rankings
    >>> method = RankingMethod.PERCENTILE  # For normalized comparisons
    >>> method = RankingMethod.WINS  # For counting benchmark victories

    See Also
    --------
    ModelRanker : Class that uses RankingMethod for ranking
    ScoreAggregation : Complementary enum for score aggregation methods
    """

    SCORE = "score"
    WINS = "wins"
    ELO = "elo"
    PERCENTILE = "percentile"
    WEIGHTED_AVERAGE = "weighted_average"


class ScoreAggregation(Enum):
    """Methods for aggregating multiple benchmark scores into a single value.

    This enumeration defines strategies for combining scores from multiple
    benchmarks into a single aggregate score for ranking purposes. The choice
    of aggregation method can significantly impact leaderboard rankings.

    Attributes
    ----------
    MEAN : str
        Arithmetic mean of all scores. Simple and intuitive but sensitive to
        outliers. Best when all benchmarks are equally important and scores
        are on similar scales.
    MEDIAN : str
        Median score across benchmarks. Robust to outliers and extreme values.
        Good for comparing "typical" performance when score distributions vary.
    MAX : str
        Maximum score across all benchmarks. Highlights peak performance.
        Useful when you want to identify models that excel in at least one area.
    MIN : str
        Minimum score across all benchmarks. Identifies worst-case performance.
        Useful for safety-critical applications where minimum capability matters.
    WEIGHTED : str
        Weighted average using custom benchmark weights. Allows prioritizing
        certain benchmarks over others. Requires weight configuration.
    GEOMETRIC : str
        Geometric mean of scores. Appropriate when scores represent ratios or
        percentages. Penalizes models with zero or near-zero scores more heavily.

    Examples
    --------
    Basic mean aggregation:

    >>> from insideLLMs.performance.leaderboard import ScoreAggregator, ScoreAggregation
    >>> aggregator = ScoreAggregator(ScoreAggregation.MEAN)
    >>> scores = {"mmlu": 0.85, "arc": 0.90, "hellaswag": 0.88}
    >>> result = aggregator.aggregate(scores)
    >>> print(f"{result:.3f}")
    0.877

    Using median for outlier robustness:

    >>> aggregator = ScoreAggregator(ScoreAggregation.MEDIAN)
    >>> scores = {"task1": 0.95, "task2": 0.92, "task3": 0.10}  # outlier
    >>> result = aggregator.aggregate(scores)
    >>> print(f"{result:.2f}")
    0.92

    Weighted aggregation for prioritized benchmarks:

    >>> weights = {"safety": 2.0, "capability": 1.0, "speed": 0.5}
    >>> aggregator = ScoreAggregator(ScoreAggregation.WEIGHTED, weights)
    >>> scores = {"safety": 0.95, "capability": 0.80, "speed": 0.70}
    >>> result = aggregator.aggregate(scores)
    >>> # (0.95*2 + 0.80*1 + 0.70*0.5) / (2+1+0.5) = 0.871
    >>> print(f"{result:.3f}")
    0.871

    See Also
    --------
    ScoreAggregator : Class that implements these aggregation methods
    RankingMethod : Enum for ranking strategies
    """

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    WEIGHTED = "weighted"
    GEOMETRIC = "geometric"


class TrendDirection(Enum):
    """Direction of a model's performance trend compared to previous rankings.

    This enumeration indicates how a model's ranking has changed since the
    previous leaderboard snapshot. It enables tracking model performance
    over time and identifying improving or declining models.

    Attributes
    ----------
    UP : str
        Model has improved in rank (moved up the leaderboard). The current
        rank is lower (better) than the previous rank.
    DOWN : str
        Model has declined in rank (moved down the leaderboard). The current
        rank is higher (worse) than the previous rank.
    STABLE : str
        Model's rank has not changed since the previous leaderboard.
        Indicates consistent relative performance.
    NEW : str
        Model is new to the leaderboard and has no previous ranking history.
        Used for first appearances.

    Examples
    --------
    Checking trend in leaderboard entries:

    >>> from insideLLMs.performance.leaderboard import quick_leaderboard, TrendDirection
    >>> lb = quick_leaderboard({"gpt-4": {"mmlu": 0.9}}, "Test")
    >>> entry = lb.entries[0]
    >>> if entry.trend == TrendDirection.NEW:
    ...     print(f"{entry.model_id} is new to the leaderboard")
    gpt-4 is new to the leaderboard

    Filtering by trend direction:

    >>> from insideLLMs.performance.leaderboard import TrendDirection
    >>> improving_models = [
    ...     e for e in leaderboard.entries
    ...     if e.trend == TrendDirection.UP
    ... ]
    >>> declining_models = [
    ...     e for e in leaderboard.entries
    ...     if e.trend == TrendDirection.DOWN
    ... ]

    Using trend for reporting:

    >>> def trend_summary(entries):
    ...     trends = {t: 0 for t in TrendDirection}
    ...     for entry in entries:
    ...         trends[entry.trend] += 1
    ...     return trends
    >>> summary = trend_summary(leaderboard.entries)
    >>> print(f"Improving: {summary[TrendDirection.UP]}")

    See Also
    --------
    LeaderboardEntry : Contains trend information for each model
    ModelRanker : Calculates trends during ranking
    """

    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    NEW = "new"


@dataclass
class ModelScore:
    """Score achieved by a model on a specific benchmark evaluation.

    This dataclass represents a single evaluation result, capturing the score
    a model achieved on a particular benchmark along with associated metadata.
    Multiple ModelScore objects are aggregated to build leaderboards.

    Parameters
    ----------
    model_id : str
        Unique identifier for the model being evaluated. Should be consistent
        across all evaluations (e.g., "gpt-4-turbo", "claude-3-opus").
    benchmark_id : str
        Unique identifier for the benchmark (e.g., "mmlu", "hellaswag", "arc").
    score : float
        The score achieved, typically normalized to [0, 1] range where higher
        is better. For benchmarks with different scales, normalize before
        creating ModelScore objects.
    metadata : dict[str, Any], optional
        Additional metadata about the evaluation run. Can include configuration
        parameters, evaluation settings, or any relevant context. Default is
        an empty dictionary.
    timestamp : str, optional
        ISO 8601 formatted timestamp of when the evaluation was performed.
        Useful for tracking historical performance. Default is None.
    run_id : str, optional
        Unique identifier for the evaluation run. Enables tracing back to
        specific experiment logs or configurations. Default is None.

    Attributes
    ----------
    model_id : str
        The model identifier.
    benchmark_id : str
        The benchmark identifier.
    score : float
        The achieved score.
    metadata : dict[str, Any]
        Additional metadata dictionary.
    timestamp : str or None
        Evaluation timestamp.
    run_id : str or None
        Evaluation run identifier.

    Examples
    --------
    Basic score creation:

    >>> from insideLLMs.performance.leaderboard import ModelScore
    >>> score = ModelScore(
    ...     model_id="gpt-4",
    ...     benchmark_id="mmlu",
    ...     score=0.89
    ... )
    >>> print(f"{score.model_id} scored {score.score} on {score.benchmark_id}")
    gpt-4 scored 0.89 on mmlu

    Score with full metadata:

    >>> from datetime import datetime
    >>> score = ModelScore(
    ...     model_id="claude-3-opus",
    ...     benchmark_id="humaneval",
    ...     score=0.84,
    ...     metadata={
    ...         "temperature": 0.0,
    ...         "max_tokens": 2048,
    ...         "few_shot": 0,
    ...         "evaluator_version": "1.2.0"
    ...     },
    ...     timestamp=datetime.now().isoformat(),
    ...     run_id="eval-2024-01-15-001"
    ... )

    Converting to dictionary for serialization:

    >>> score_dict = score.to_dict()
    >>> import json
    >>> json_str = json.dumps(score_dict)
    >>> print(json.loads(json_str)["model_id"])
    claude-3-opus

    Batch creation for multiple benchmarks:

    >>> models = ["gpt-4", "claude-3", "gemini"]
    >>> benchmarks = {"mmlu": [0.89, 0.91, 0.85], "arc": [0.92, 0.88, 0.86]}
    >>> scores = [
    ...     ModelScore(model, bench, scores[i])
    ...     for bench, scores in benchmarks.items()
    ...     for i, model in enumerate(models)
    ... ]
    >>> len(scores)
    6

    See Also
    --------
    LeaderboardBuilder : Uses ModelScore objects to build leaderboards
    create_leaderboard : Convenience function that accepts ModelScore lists
    """

    model_id: str
    benchmark_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the ModelScore to a dictionary representation.

        Creates a serializable dictionary containing all score information,
        suitable for JSON serialization or database storage.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all ModelScore fields:
            - "model_id": Model identifier string
            - "benchmark_id": Benchmark identifier string
            - "score": Numeric score value
            - "metadata": Metadata dictionary
            - "timestamp": Timestamp string or None
            - "run_id": Run identifier string or None

        Examples
        --------
        Basic conversion:

        >>> score = ModelScore("gpt-4", "mmlu", 0.89)
        >>> d = score.to_dict()
        >>> d["model_id"]
        'gpt-4'
        >>> d["score"]
        0.89

        Serializing to JSON:

        >>> import json
        >>> score = ModelScore(
        ...     "claude-3", "arc", 0.88,
        ...     metadata={"shots": 5}
        ... )
        >>> json_str = json.dumps(score.to_dict())
        >>> restored = json.loads(json_str)
        >>> restored["metadata"]["shots"]
        5

        Storing multiple scores:

        >>> scores = [
        ...     ModelScore("m1", "b1", 0.9),
        ...     ModelScore("m1", "b2", 0.85),
        ... ]
        >>> records = [s.to_dict() for s in scores]
        >>> len(records)
        2
        """
        return {
            "model_id": self.model_id,
            "benchmark_id": self.benchmark_id,
            "score": self.score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


@dataclass
class LeaderboardEntry:
    """Single entry representing a model's position in a leaderboard.

    This dataclass contains all information about a model's ranking including
    its current position, aggregated score, performance trend compared to
    previous rankings, and detailed score breakdown by benchmark.

    Parameters
    ----------
    rank : int
        Current ranking position (1 = first place, highest score).
    model_id : str
        Unique identifier for the model.
    score : float
        Aggregated score used for ranking. The aggregation method depends
        on the ScoreAggregator configuration.
    previous_rank : int or None
        Rank from the previous leaderboard snapshot. None if the model
        is new to the leaderboard.
    trend : TrendDirection
        Direction of rank change (UP, DOWN, STABLE, or NEW).
    score_breakdown : dict[str, float]
        Individual scores for each benchmark, keyed by benchmark_id.
        Useful for detailed analysis of strengths and weaknesses.
    metadata : dict[str, Any], optional
        Additional metadata about the entry. Default is an empty dictionary.

    Attributes
    ----------
    rank : int
        Current ranking position.
    model_id : str
        Model identifier.
    score : float
        Aggregated score.
    previous_rank : int or None
        Previous ranking position.
    trend : TrendDirection
        Performance trend direction.
    score_breakdown : dict[str, float]
        Per-benchmark score breakdown.
    metadata : dict[str, Any]
        Additional metadata.
    rank_change : int or None
        Calculated change in rank (property).

    Examples
    --------
    Creating a leaderboard entry:

    >>> from insideLLMs.performance.leaderboard import LeaderboardEntry, TrendDirection
    >>> entry = LeaderboardEntry(
    ...     rank=1,
    ...     model_id="claude-3-opus",
    ...     score=0.912,
    ...     previous_rank=2,
    ...     trend=TrendDirection.UP,
    ...     score_breakdown={"mmlu": 0.91, "arc": 0.92, "hellaswag": 0.90}
    ... )
    >>> print(f"#{entry.rank}: {entry.model_id} ({entry.score:.3f})")
    #1: claude-3-opus (0.912)

    Checking rank improvement:

    >>> entry = LeaderboardEntry(
    ...     rank=3,
    ...     model_id="gpt-4",
    ...     score=0.895,
    ...     previous_rank=5,
    ...     trend=TrendDirection.UP,
    ...     score_breakdown={"mmlu": 0.89, "arc": 0.90}
    ... )
    >>> change = entry.rank_change
    >>> print(f"Moved up {change} positions" if change > 0 else "Declined")
    Moved up 2 positions

    Analyzing score breakdown:

    >>> entry = LeaderboardEntry(
    ...     rank=2,
    ...     model_id="gemini-ultra",
    ...     score=0.88,
    ...     previous_rank=2,
    ...     trend=TrendDirection.STABLE,
    ...     score_breakdown={
    ...         "coding": 0.95,
    ...         "reasoning": 0.85,
    ...         "knowledge": 0.84
    ...     }
    ... )
    >>> best_bench = max(entry.score_breakdown, key=entry.score_breakdown.get)
    >>> print(f"Best at: {best_bench}")
    Best at: coding

    See Also
    --------
    Leaderboard : Contains a list of LeaderboardEntry objects
    ModelRanker : Creates LeaderboardEntry objects during ranking
    TrendDirection : Enum for trend values
    """

    rank: int
    model_id: str
    score: float
    previous_rank: Optional[int]
    trend: TrendDirection
    score_breakdown: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rank_change(self) -> Optional[int]:
        """Calculate the change in rank from the previous leaderboard.

        Computes how many positions the model has moved since the last
        leaderboard snapshot. A positive value indicates improvement
        (moving up), negative indicates decline.

        Returns
        -------
        int or None
            Number of positions changed. Positive values mean the model
            improved (e.g., moved from rank 5 to rank 3 = +2). Negative
            values mean decline. Returns None if there is no previous
            rank (new model).

        Examples
        --------
        Model that improved:

        >>> entry = LeaderboardEntry(
        ...     rank=2, model_id="gpt-4", score=0.9,
        ...     previous_rank=4, trend=TrendDirection.UP,
        ...     score_breakdown={}
        ... )
        >>> entry.rank_change
        2

        Model that declined:

        >>> entry = LeaderboardEntry(
        ...     rank=5, model_id="llama", score=0.8,
        ...     previous_rank=3, trend=TrendDirection.DOWN,
        ...     score_breakdown={}
        ... )
        >>> entry.rank_change
        -2

        New model (no previous rank):

        >>> entry = LeaderboardEntry(
        ...     rank=1, model_id="new-model", score=0.95,
        ...     previous_rank=None, trend=TrendDirection.NEW,
        ...     score_breakdown={}
        ... )
        >>> entry.rank_change is None
        True
        """
        if self.previous_rank is None:
            return None
        return self.previous_rank - self.rank  # Positive = improved

    def to_dict(self) -> dict[str, Any]:
        """Convert the LeaderboardEntry to a dictionary representation.

        Creates a serializable dictionary containing all entry information,
        including the computed rank_change property.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "rank": Current rank (int)
            - "model_id": Model identifier (str)
            - "score": Aggregated score (float)
            - "previous_rank": Previous rank or None
            - "rank_change": Computed rank change or None
            - "trend": Trend direction as string value
            - "score_breakdown": Per-benchmark scores (dict)
            - "metadata": Additional metadata (dict)

        Examples
        --------
        Basic conversion:

        >>> entry = LeaderboardEntry(
        ...     rank=1, model_id="gpt-4", score=0.92,
        ...     previous_rank=2, trend=TrendDirection.UP,
        ...     score_breakdown={"mmlu": 0.91, "arc": 0.93}
        ... )
        >>> d = entry.to_dict()
        >>> d["rank"]
        1
        >>> d["trend"]
        'up'
        >>> d["rank_change"]
        1

        Serializing for API response:

        >>> import json
        >>> entry = LeaderboardEntry(
        ...     rank=3, model_id="claude", score=0.88,
        ...     previous_rank=None, trend=TrendDirection.NEW,
        ...     score_breakdown={"test": 0.88},
        ...     metadata={"version": "3.0"}
        ... )
        >>> json_str = json.dumps(entry.to_dict())
        >>> data = json.loads(json_str)
        >>> data["metadata"]["version"]
        '3.0'

        Building leaderboard response:

        >>> entries = [entry1, entry2, entry3]  # LeaderboardEntry objects
        >>> response = {"entries": [e.to_dict() for e in entries]}
        """
        return {
            "rank": self.rank,
            "model_id": self.model_id,
            "score": self.score,
            "previous_rank": self.previous_rank,
            "rank_change": self.rank_change,
            "trend": self.trend.value,
            "score_breakdown": self.score_breakdown,
            "metadata": self.metadata,
        }


@dataclass
class Leaderboard:
    """Complete leaderboard containing ranked model entries and metadata.

    This dataclass represents a fully constructed leaderboard with all model
    rankings, configuration details, and metadata. It serves as the primary
    output of the LeaderboardBuilder and input for LeaderboardFormatter.

    Parameters
    ----------
    name : str
        Human-readable name for the leaderboard (e.g., "Q1 2024 LLM Benchmark").
    description : str
        Detailed description of the leaderboard's purpose and methodology.
    entries : list[LeaderboardEntry]
        List of model entries sorted by rank (position 0 = rank 1).
    benchmark_ids : list[str]
        List of benchmark identifiers included in this leaderboard.
    metric_weights : dict[str, float]
        Weights used for each benchmark in score aggregation. Empty dict
        if uniform weighting was used.
    ranking_method : RankingMethod
        The ranking method used to order models.
    updated_at : str
        ISO 8601 timestamp of when the leaderboard was generated.
    version : str, optional
        Version string for the leaderboard schema. Default is "1.0".

    Attributes
    ----------
    name : str
        Leaderboard name.
    description : str
        Leaderboard description.
    entries : list[LeaderboardEntry]
        Ranked model entries.
    benchmark_ids : list[str]
        Included benchmark identifiers.
    metric_weights : dict[str, float]
        Benchmark weights.
    ranking_method : RankingMethod
        Ranking method used.
    updated_at : str
        Generation timestamp.
    version : str
        Schema version.
    top_model : str or None
        Model ID of the top-ranked model (property).
    n_models : int
        Number of models in the leaderboard (property).

    Examples
    --------
    Creating a leaderboard manually:

    >>> from insideLLMs.performance.leaderboard import (
    ...     Leaderboard, LeaderboardEntry, RankingMethod, TrendDirection
    ... )
    >>> entries = [
    ...     LeaderboardEntry(1, "claude-3", 0.92, None, TrendDirection.NEW, {}),
    ...     LeaderboardEntry(2, "gpt-4", 0.90, None, TrendDirection.NEW, {}),
    ... ]
    >>> lb = Leaderboard(
    ...     name="AI Benchmark 2024",
    ...     description="Comprehensive LLM evaluation",
    ...     entries=entries,
    ...     benchmark_ids=["mmlu", "arc", "hellaswag"],
    ...     metric_weights={},
    ...     ranking_method=RankingMethod.SCORE,
    ...     updated_at="2024-01-15T10:30:00"
    ... )
    >>> print(f"Top model: {lb.top_model}")
    Top model: claude-3

    Using the builder (recommended):

    >>> from insideLLMs.performance.leaderboard import quick_leaderboard
    >>> scores = {
    ...     "gpt-4": {"mmlu": 0.89, "arc": 0.92},
    ...     "claude-3": {"mmlu": 0.91, "arc": 0.90},
    ... }
    >>> lb = quick_leaderboard(scores, "Quick Test")
    >>> print(f"Models: {lb.n_models}, Top: {lb.top_model}")
    Models: 2, Top: claude-3

    Looking up a specific model:

    >>> lb = quick_leaderboard(
    ...     {"a": {"t": 0.9}, "b": {"t": 0.8}, "c": {"t": 0.7}},
    ...     "Test"
    ... )
    >>> entry = lb.get_entry("b")
    >>> if entry:
    ...     print(f"Model 'b' is ranked #{entry.rank}")
    Model 'b' is ranked #2

    See Also
    --------
    LeaderboardBuilder : Builds Leaderboard objects from scores
    LeaderboardFormatter : Formats Leaderboard for export
    LeaderboardAnalyzer : Analyzes and compares Leaderboard objects
    """

    name: str
    description: str
    entries: list[LeaderboardEntry]
    benchmark_ids: list[str]
    metric_weights: dict[str, float]
    ranking_method: RankingMethod
    updated_at: str
    version: str = "1.0"

    @property
    def top_model(self) -> Optional[str]:
        """Get the identifier of the top-ranked model.

        Returns the model_id of the model in first place (rank 1).
        Returns None if the leaderboard has no entries.

        Returns
        -------
        str or None
            Model ID of the top-ranked model, or None if the leaderboard
            is empty.

        Examples
        --------
        Getting the top model:

        >>> lb = quick_leaderboard(
        ...     {"gpt-4": {"test": 0.9}, "claude": {"test": 0.95}},
        ...     "Test"
        ... )
        >>> lb.top_model
        'claude'

        Handling empty leaderboard:

        >>> from insideLLMs.performance.leaderboard import (
        ...     Leaderboard, RankingMethod
        ... )
        >>> empty_lb = Leaderboard(
        ...     name="Empty", description="", entries=[],
        ...     benchmark_ids=[], metric_weights={},
        ...     ranking_method=RankingMethod.SCORE,
        ...     updated_at="2024-01-01"
        ... )
        >>> empty_lb.top_model is None
        True

        Using in conditional logic:

        >>> if lb.top_model:
        ...     print(f"Winner: {lb.top_model}")
        ... else:
        ...     print("No models ranked")
        """
        return self.entries[0].model_id if self.entries else None

    @property
    def n_models(self) -> int:
        """Get the total number of models in the leaderboard.

        Returns
        -------
        int
            Count of models (entries) in the leaderboard.

        Examples
        --------
        Counting models:

        >>> lb = quick_leaderboard(
        ...     {"a": {"t": 0.9}, "b": {"t": 0.8}, "c": {"t": 0.7}},
        ...     "Test"
        ... )
        >>> lb.n_models
        3

        Using for pagination:

        >>> page_size = 10
        >>> total_pages = (lb.n_models + page_size - 1) // page_size
        >>> print(f"Showing page 1 of {total_pages}")

        Checking if leaderboard has entries:

        >>> if lb.n_models > 0:
        ...     print(f"Leaderboard has {lb.n_models} models")
        """
        return len(self.entries)

    def get_entry(self, model_id: str) -> Optional[LeaderboardEntry]:
        """Retrieve the leaderboard entry for a specific model.

        Searches the leaderboard entries for the specified model and returns
        its entry if found. Useful for looking up a model's rank and scores.

        Parameters
        ----------
        model_id : str
            The unique identifier of the model to look up.

        Returns
        -------
        LeaderboardEntry or None
            The entry for the specified model, or None if the model is not
            found in the leaderboard.

        Examples
        --------
        Looking up a model's entry:

        >>> lb = quick_leaderboard(
        ...     {
        ...         "gpt-4": {"mmlu": 0.89, "arc": 0.92},
        ...         "claude-3": {"mmlu": 0.91, "arc": 0.90},
        ...         "llama-3": {"mmlu": 0.78, "arc": 0.82},
        ...     },
        ...     "Test"
        ... )
        >>> entry = lb.get_entry("claude-3")
        >>> if entry:
        ...     print(f"claude-3 is ranked #{entry.rank} with score {entry.score:.3f}")
        claude-3 is ranked #1 with score 0.905

        Handling missing models:

        >>> entry = lb.get_entry("nonexistent-model")
        >>> if entry is None:
        ...     print("Model not found in leaderboard")
        Model not found in leaderboard

        Comparing two models:

        >>> entry_a = lb.get_entry("gpt-4")
        >>> entry_b = lb.get_entry("llama-3")
        >>> if entry_a and entry_b:
        ...     diff = entry_a.score - entry_b.score
        ...     print(f"gpt-4 scores {diff:.3f} higher than llama-3")
        gpt-4 scores 0.105 higher than llama-3
        """
        for entry in self.entries:
            if entry.model_id == model_id:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert the Leaderboard to a dictionary representation.

        Creates a fully serializable dictionary containing all leaderboard
        data including entries and computed properties. Suitable for JSON
        serialization, API responses, or database storage.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "name": Leaderboard name (str)
            - "description": Description (str)
            - "entries": List of entry dictionaries
            - "benchmark_ids": List of benchmark IDs
            - "metric_weights": Weight dictionary
            - "ranking_method": Method as string value
            - "updated_at": Timestamp string
            - "version": Schema version
            - "top_model": Top model ID or None
            - "n_models": Number of models

        Examples
        --------
        Basic conversion:

        >>> lb = quick_leaderboard(
        ...     {"gpt-4": {"test": 0.9}, "claude": {"test": 0.85}},
        ...     "API Response"
        ... )
        >>> d = lb.to_dict()
        >>> d["name"]
        'API Response'
        >>> d["n_models"]
        2

        JSON serialization for API:

        >>> import json
        >>> lb = quick_leaderboard(
        ...     {"model1": {"bench": 0.9}},
        ...     "Test Leaderboard"
        ... )
        >>> response = json.dumps(lb.to_dict(), indent=2)
        >>> print(response[:50])
        {
          "name": "Test Leaderboard",
          "descri

        Storing leaderboard snapshot:

        >>> import json
        >>> from pathlib import Path
        >>> lb = quick_leaderboard({"a": {"t": 0.9}}, "Snapshot")
        >>> # Path("leaderboard.json").write_text(json.dumps(lb.to_dict()))
        """
        return {
            "name": self.name,
            "description": self.description,
            "entries": [e.to_dict() for e in self.entries],
            "benchmark_ids": self.benchmark_ids,
            "metric_weights": self.metric_weights,
            "ranking_method": self.ranking_method.value,
            "updated_at": self.updated_at,
            "version": self.version,
            "top_model": self.top_model,
            "n_models": self.n_models,
        }


@dataclass
class LeaderboardComparison:
    """Comparison between leaderboards."""

    leaderboard_a: str
    leaderboard_b: str
    common_models: list[str]
    rank_correlations: float
    score_correlations: float
    divergent_rankings: list[tuple[str, int, int]]  # model, rank_a, rank_b

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "leaderboard_a": self.leaderboard_a,
            "leaderboard_b": self.leaderboard_b,
            "common_models": self.common_models,
            "rank_correlations": self.rank_correlations,
            "score_correlations": self.score_correlations,
            "divergent_rankings": [
                {"model": m, "rank_a": ra, "rank_b": rb} for m, ra, rb in self.divergent_rankings
            ],
        }


class ScoreAggregator:
    """Aggregate scores across benchmarks."""

    def __init__(
        self,
        method: ScoreAggregation = ScoreAggregation.MEAN,
        weights: Optional[dict[str, float]] = None,
    ):
        """Initialize aggregator.

        Args:
            method: Aggregation method
            weights: Optional weights for benchmarks
        """
        self._method = method
        self._weights = weights or {}

    def aggregate(
        self,
        scores: dict[str, float],
    ) -> float:
        """Aggregate multiple scores.

        Args:
            scores: Dictionary of benchmark_id -> score

        Returns:
            Aggregated score
        """
        if not scores:
            return 0.0

        values = list(scores.values())

        if self._method == ScoreAggregation.MEAN:
            return statistics.mean(values)
        elif self._method == ScoreAggregation.MEDIAN:
            return statistics.median(values)
        elif self._method == ScoreAggregation.MAX:
            return max(values)
        elif self._method == ScoreAggregation.MIN:
            return min(values)
        elif self._method == ScoreAggregation.GEOMETRIC:
            # Geometric mean
            product = 1.0
            for v in values:
                product *= max(v, 0.001)  # Avoid zero
            return product ** (1.0 / len(values))
        elif self._method == ScoreAggregation.WEIGHTED:
            return self._weighted_aggregate(scores)

        return statistics.mean(values)

    def _weighted_aggregate(self, scores: dict[str, float]) -> float:
        """Calculate weighted aggregate."""
        if not self._weights:
            return statistics.mean(scores.values())

        total_weight = 0.0
        weighted_sum = 0.0

        for benchmark_id, score in scores.items():
            weight = self._weights.get(benchmark_id, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0


class ModelRanker:
    """Rank models based on scores."""

    def __init__(
        self,
        method: RankingMethod = RankingMethod.SCORE,
        aggregator: Optional[ScoreAggregator] = None,
    ):
        """Initialize ranker.

        Args:
            method: Ranking method
            aggregator: Score aggregator
        """
        self._method = method
        self._aggregator = aggregator or ScoreAggregator()

    def rank(
        self,
        model_scores: dict[str, dict[str, float]],
        previous_rankings: Optional[dict[str, int]] = None,
    ) -> list[LeaderboardEntry]:
        """Rank models.

        Args:
            model_scores: Dict of model_id -> {benchmark_id: score}
            previous_rankings: Optional previous rankings for trend

        Returns:
            List of LeaderboardEntry objects, sorted by rank
        """
        if not model_scores:
            return []

        # Calculate aggregated scores
        aggregated = {
            model_id: self._aggregator.aggregate(scores)
            for model_id, scores in model_scores.items()
        }

        # Sort by score (descending)
        sorted_models = sorted(
            aggregated.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Create entries
        entries = []
        for rank, (model_id, score) in enumerate(sorted_models, 1):
            prev_rank = previous_rankings.get(model_id) if previous_rankings else None
            trend = self._calculate_trend(rank, prev_rank)

            entries.append(
                LeaderboardEntry(
                    rank=rank,
                    model_id=model_id,
                    score=score,
                    previous_rank=prev_rank,
                    trend=trend,
                    score_breakdown=model_scores[model_id],
                )
            )

        return entries

    @staticmethod
    def _calculate_trend(
        current_rank: int,
        previous_rank: Optional[int],
    ) -> TrendDirection:
        """Calculate trend direction."""
        if previous_rank is None:
            return TrendDirection.NEW

        if current_rank < previous_rank:
            return TrendDirection.UP
        elif current_rank > previous_rank:
            return TrendDirection.DOWN
        return TrendDirection.STABLE


class LeaderboardBuilder:
    """Build leaderboards from scores."""

    def __init__(
        self,
        ranker: Optional[ModelRanker] = None,
    ):
        """Initialize builder.

        Args:
            ranker: Model ranker
        """
        self._ranker = ranker or ModelRanker()
        self._scores: list[ModelScore] = []
        self._history: dict[str, list[Leaderboard]] = defaultdict(list)

    def add_score(self, score: ModelScore) -> None:
        """Add a score.

        Args:
            score: ModelScore to add
        """
        self._scores.append(score)

    def add_scores(self, scores: list[ModelScore]) -> None:
        """Add multiple scores.

        Args:
            scores: List of ModelScore objects
        """
        self._scores.extend(scores)

    def build(
        self,
        name: str,
        description: str = "",
        benchmark_filter: Optional[list[str]] = None,
        model_filter: Optional[list[str]] = None,
        metric_weights: Optional[dict[str, float]] = None,
    ) -> Leaderboard:
        """Build a leaderboard.

        Args:
            name: Leaderboard name
            description: Leaderboard description
            benchmark_filter: Optional list of benchmark IDs to include
            model_filter: Optional list of model IDs to include
            metric_weights: Optional weights for benchmarks

        Returns:
            Leaderboard object
        """
        # Filter scores
        filtered_scores = self._scores
        if benchmark_filter:
            filtered_scores = [s for s in filtered_scores if s.benchmark_id in benchmark_filter]
        if model_filter:
            filtered_scores = [s for s in filtered_scores if s.model_id in model_filter]

        # Organize by model
        model_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for score in filtered_scores:
            model_scores[score.model_id][score.benchmark_id] = score.score

        # Get previous rankings
        previous_rankings = self._get_previous_rankings(name)

        # Rank models
        entries = self._ranker.rank(dict(model_scores), previous_rankings)

        # Build leaderboard
        benchmark_ids = list({s.benchmark_id for s in filtered_scores})
        leaderboard = Leaderboard(
            name=name,
            description=description,
            entries=entries,
            benchmark_ids=benchmark_ids,
            metric_weights=metric_weights or {},
            ranking_method=self._ranker._method,
            updated_at=datetime.now().isoformat(),
        )

        # Save to history
        self._history[name].append(leaderboard)

        return leaderboard

    def _get_previous_rankings(self, name: str) -> dict[str, int]:
        """Get previous rankings for a leaderboard."""
        history = self._history.get(name, [])
        if not history:
            return {}

        previous = history[-1]
        return {e.model_id: e.rank for e in previous.entries}

    def clear(self) -> None:
        """Clear all scores."""
        self._scores = []

    def get_history(self, name: str) -> list[Leaderboard]:
        """Get leaderboard history."""
        return self._history.get(name, [])


class LeaderboardFormatter:
    """Format leaderboards for display/export."""

    def to_markdown(
        self,
        leaderboard: Leaderboard,
        show_breakdown: bool = False,
    ) -> str:
        """Format leaderboard as Markdown.

        Args:
            leaderboard: Leaderboard to format
            show_breakdown: Include score breakdown

        Returns:
            Markdown string
        """
        lines = [
            f"# {leaderboard.name}",
            "",
            f"_{leaderboard.description}_" if leaderboard.description else "",
            "",
            f"**Updated:** {leaderboard.updated_at}",
            f"**Models:** {leaderboard.n_models}",
            "",
        ]

        # Header
        if show_breakdown and leaderboard.benchmark_ids:
            headers = ["Rank", "Model", "Score"] + leaderboard.benchmark_ids + ["Trend"]
        else:
            headers = ["Rank", "Model", "Score", "Change", "Trend"]

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Rows
        for entry in leaderboard.entries:
            trend_emoji = self._trend_emoji(entry.trend)
            change = entry.rank_change
            change_str = f"+{change}" if change and change > 0 else str(change) if change else "-"

            if show_breakdown and leaderboard.benchmark_ids:
                breakdown_vals = [
                    f"{entry.score_breakdown.get(b, 0):.3f}" for b in leaderboard.benchmark_ids
                ]
                row = (
                    [
                        str(entry.rank),
                        entry.model_id,
                        f"{entry.score:.3f}",
                    ]
                    + breakdown_vals
                    + [trend_emoji]
                )
            else:
                row = [
                    str(entry.rank),
                    entry.model_id,
                    f"{entry.score:.3f}",
                    change_str,
                    trend_emoji,
                ]

            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def to_csv(self, leaderboard: Leaderboard) -> str:
        """Format leaderboard as CSV.

        Args:
            leaderboard: Leaderboard to format

        Returns:
            CSV string
        """
        lines = []

        # Header
        headers = ["rank", "model_id", "score", "previous_rank", "trend"]
        headers.extend(leaderboard.benchmark_ids)
        lines.append(",".join(headers))

        # Rows
        for entry in leaderboard.entries:
            row = [
                str(entry.rank),
                entry.model_id,
                f"{entry.score:.4f}",
                str(entry.previous_rank or ""),
                entry.trend.value,
            ]
            for benchmark in leaderboard.benchmark_ids:
                row.append(f"{entry.score_breakdown.get(benchmark, 0):.4f}")
            lines.append(",".join(row))

        return "\n".join(lines)

    def to_json(self, leaderboard: Leaderboard) -> str:
        """Format leaderboard as JSON.

        Args:
            leaderboard: Leaderboard to format

        Returns:
            JSON string
        """
        return json.dumps(leaderboard.to_dict(), indent=2)

    def to_html(self, leaderboard: Leaderboard) -> str:
        """Format leaderboard as HTML.

        Args:
            leaderboard: Leaderboard to format

        Returns:
            HTML string
        """
        rows = []
        for entry in leaderboard.entries:
            trend_class = f"trend-{entry.trend.value}"
            rows.append(f"""
            <tr class="{trend_class}">
                <td>{entry.rank}</td>
                <td>{entry.model_id}</td>
                <td>{entry.score:.3f}</td>
                <td>{entry.rank_change or "-"}</td>
                <td>{self._trend_emoji(entry.trend)}</td>
            </tr>
            """)

        return f"""
        <div class="leaderboard">
            <h2>{leaderboard.name}</h2>
            <p>{leaderboard.description}</p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Score</th>
                        <th>Change</th>
                        <th>Trend</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
            <p class="meta">Updated: {leaderboard.updated_at}</p>
        </div>
        """

    @staticmethod
    def _trend_emoji(trend: TrendDirection) -> str:
        """Get emoji for trend."""
        return {
            TrendDirection.UP: "^",
            TrendDirection.DOWN: "v",
            TrendDirection.STABLE: "-",
            TrendDirection.NEW: "*",
        }.get(trend, "?")


class LeaderboardAnalyzer:
    """Analyze leaderboards."""

    def compare_leaderboards(
        self,
        lb_a: Leaderboard,
        lb_b: Leaderboard,
    ) -> LeaderboardComparison:
        """Compare two leaderboards.

        Args:
            lb_a: First leaderboard
            lb_b: Second leaderboard

        Returns:
            LeaderboardComparison object
        """
        # Find common models
        models_a = {e.model_id for e in lb_a.entries}
        models_b = {e.model_id for e in lb_b.entries}
        common = sorted(models_a & models_b)

        if not common:
            return LeaderboardComparison(
                leaderboard_a=lb_a.name,
                leaderboard_b=lb_b.name,
                common_models=[],
                rank_correlations=0.0,
                score_correlations=0.0,
                divergent_rankings=[],
            )

        # Get rankings for common models
        ranks_a = {e.model_id: e.rank for e in lb_a.entries}
        ranks_b = {e.model_id: e.rank for e in lb_b.entries}
        scores_a = {e.model_id: e.score for e in lb_a.entries}
        scores_b = {e.model_id: e.score for e in lb_b.entries}

        # Calculate correlations
        rank_corr = self._spearman_correlation(
            [ranks_a[m] for m in common],
            [ranks_b[m] for m in common],
        )
        score_corr = self._pearson_correlation(
            [scores_a[m] for m in common],
            [scores_b[m] for m in common],
        )

        # Find divergent rankings
        divergent = []
        for model in common:
            rank_diff = abs(ranks_a[model] - ranks_b[model])
            if rank_diff >= 3:  # Significant difference
                divergent.append((model, ranks_a[model], ranks_b[model]))

        return LeaderboardComparison(
            leaderboard_a=lb_a.name,
            leaderboard_b=lb_b.name,
            common_models=common,
            rank_correlations=rank_corr,
            score_correlations=score_corr,
            divergent_rankings=divergent,
        )

    def analyze_trends(
        self,
        history: list[Leaderboard],
        model_id: str,
    ) -> dict[str, Any]:
        """Analyze trends for a model.

        Args:
            history: List of historical leaderboards
            model_id: Model to analyze

        Returns:
            Dictionary with trend analysis
        """
        ranks = []
        scores = []

        for lb in history:
            entry = lb.get_entry(model_id)
            if entry:
                ranks.append(entry.rank)
                scores.append(entry.score)

        if not ranks:
            return {
                "model_id": model_id,
                "found": False,
            }

        return {
            "model_id": model_id,
            "found": True,
            "n_appearances": len(ranks),
            "best_rank": min(ranks),
            "worst_rank": max(ranks),
            "avg_rank": statistics.mean(ranks),
            "best_score": max(scores),
            "worst_score": min(scores),
            "avg_score": statistics.mean(scores),
            "rank_trend": self._calculate_linear_trend(ranks),  # type: ignore[arg-type]
            "score_trend": self._calculate_linear_trend(scores),
        }

    @staticmethod
    def _spearman_correlation(x: list[int], y: list[int]) -> float:
        """Calculate Spearman rank correlation."""
        n = len(x)
        if n < 2:
            return 0.0

        # Calculate rank differences
        d_squared_sum = sum((xi - yi) ** 2 for xi, yi in zip(x, y))

        # Spearman formula
        rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        return rho

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

        if denom_x == 0 or denom_y == 0:
            return 0.0

        return numerator / (denom_x * denom_y)

    @staticmethod
    def _calculate_linear_trend(values: list[float]) -> str:
        """Calculate linear trend direction."""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.1:
            return "improving" if "rank" not in str(values) else "declining"
        elif slope < -0.1:
            return "declining" if "rank" not in str(values) else "improving"
        return "stable"


# Convenience functions


def create_leaderboard(
    scores: list[ModelScore],
    name: str,
    description: str = "",
    weights: Optional[dict[str, float]] = None,
) -> Leaderboard:
    """Create a leaderboard from scores.

    Args:
        scores: List of ModelScore objects
        name: Leaderboard name
        description: Leaderboard description
        weights: Optional benchmark weights

    Returns:
        Leaderboard object
    """
    aggregator = ScoreAggregator(
        method=ScoreAggregation.WEIGHTED if weights else ScoreAggregation.MEAN,
        weights=weights,
    )
    ranker = ModelRanker(aggregator=aggregator)
    builder = LeaderboardBuilder(ranker=ranker)
    builder.add_scores(scores)
    return builder.build(name, description, metric_weights=weights)


def format_leaderboard(
    leaderboard: Leaderboard,
    format_type: str = "markdown",
) -> str:
    """Format leaderboard for display.

    Args:
        leaderboard: Leaderboard to format
        format_type: Output format (markdown, csv, json, html)

    Returns:
        Formatted string
    """
    formatter = LeaderboardFormatter()

    if format_type == "markdown":
        return formatter.to_markdown(leaderboard)
    elif format_type == "csv":
        return formatter.to_csv(leaderboard)
    elif format_type == "json":
        return formatter.to_json(leaderboard)
    elif format_type == "html":
        return formatter.to_html(leaderboard)

    return formatter.to_markdown(leaderboard)


def compare_leaderboards(
    lb_a: Leaderboard,
    lb_b: Leaderboard,
) -> LeaderboardComparison:
    """Compare two leaderboards.

    Args:
        lb_a: First leaderboard
        lb_b: Second leaderboard

    Returns:
        LeaderboardComparison object
    """
    analyzer = LeaderboardAnalyzer()
    return analyzer.compare_leaderboards(lb_a, lb_b)


def quick_leaderboard(
    model_scores: dict[str, dict[str, float]],
    name: str = "Quick Leaderboard",
) -> Leaderboard:
    """Create a quick leaderboard from model scores.

    Args:
        model_scores: Dict of model_id -> {benchmark: score}
        name: Leaderboard name

    Returns:
        Leaderboard object
    """
    scores = []
    for model_id, benchmarks in model_scores.items():
        for benchmark_id, score in benchmarks.items():
            scores.append(
                ModelScore(
                    model_id=model_id,
                    benchmark_id=benchmark_id,
                    score=score,
                )
            )

    return create_leaderboard(scores, name)
