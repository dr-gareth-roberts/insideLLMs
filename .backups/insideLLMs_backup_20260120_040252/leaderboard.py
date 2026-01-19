"""Benchmark leaderboard generation for LLM evaluation.

This module provides tools for creating and managing benchmark
leaderboards from evaluation results:

- Score aggregation and ranking
- Multi-metric leaderboards
- Historical tracking
- Leaderboard export formats
- Comparative analysis
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RankingMethod(Enum):
    """Methods for ranking models."""

    SCORE = "score"
    WINS = "wins"
    ELO = "elo"
    PERCENTILE = "percentile"
    WEIGHTED_AVERAGE = "weighted_average"


class ScoreAggregation(Enum):
    """Methods for aggregating scores."""

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    WEIGHTED = "weighted"
    GEOMETRIC = "geometric"


class TrendDirection(Enum):
    """Direction of performance trend."""

    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    NEW = "new"


@dataclass
class ModelScore:
    """Score for a model on a benchmark."""

    model_id: str
    benchmark_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Entry in a leaderboard."""

    rank: int
    model_id: str
    score: float
    previous_rank: int | None
    trend: TrendDirection
    score_breakdown: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rank_change(self) -> int | None:
        """Calculate rank change from previous."""
        if self.previous_rank is None:
            return None
        return self.previous_rank - self.rank  # Positive = improved

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Complete leaderboard."""

    name: str
    description: str
    entries: list[LeaderboardEntry]
    benchmark_ids: list[str]
    metric_weights: dict[str, float]
    ranking_method: RankingMethod
    updated_at: str
    version: str = "1.0"

    @property
    def top_model(self) -> str | None:
        """Get top-ranked model."""
        return self.entries[0].model_id if self.entries else None

    @property
    def n_models(self) -> int:
        """Get number of models."""
        return len(self.entries)

    def get_entry(self, model_id: str) -> LeaderboardEntry | None:
        """Get entry for a model."""
        for entry in self.entries:
            if entry.model_id == model_id:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
        weights: dict[str, float] | None = None,
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
        aggregator: ScoreAggregator | None = None,
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
        previous_rankings: dict[str, int] | None = None,
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
        previous_rank: int | None,
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
        ranker: ModelRanker | None = None,
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
        benchmark_filter: list[str] | None = None,
        model_filter: list[str] | None = None,
        metric_weights: dict[str, float] | None = None,
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
            "rank_trend": self._calculate_linear_trend(ranks),
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
    weights: dict[str, float] | None = None,
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
