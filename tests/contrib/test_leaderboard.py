"""Tests for benchmark leaderboard generation module."""

import json

import pytest

from insideLLMs.contrib.leaderboard import (
    Leaderboard,
    LeaderboardAnalyzer,
    LeaderboardBuilder,
    LeaderboardComparison,
    LeaderboardEntry,
    LeaderboardFormatter,
    ModelRanker,
    # Dataclasses
    ModelScore,
    # Enums
    RankingMethod,
    ScoreAggregation,
    # Classes
    ScoreAggregator,
    TrendDirection,
    compare_leaderboards,
    # Functions
    create_leaderboard,
    format_leaderboard,
    quick_leaderboard,
)

# ============================================================================
# Enum Tests
# ============================================================================


class TestRankingMethod:
    """Tests for RankingMethod enum."""

    def test_all_methods_defined(self):
        """Test all ranking methods are defined."""
        assert RankingMethod.SCORE.value == "score"
        assert RankingMethod.WINS.value == "wins"
        assert RankingMethod.ELO.value == "elo"
        assert RankingMethod.PERCENTILE.value == "percentile"
        assert RankingMethod.WEIGHTED_AVERAGE.value == "weighted_average"


class TestScoreAggregation:
    """Tests for ScoreAggregation enum."""

    def test_all_methods_defined(self):
        """Test all aggregation methods are defined."""
        assert ScoreAggregation.MEAN.value == "mean"
        assert ScoreAggregation.MEDIAN.value == "median"
        assert ScoreAggregation.MAX.value == "max"
        assert ScoreAggregation.MIN.value == "min"
        assert ScoreAggregation.WEIGHTED.value == "weighted"
        assert ScoreAggregation.GEOMETRIC.value == "geometric"


class TestTrendDirection:
    """Tests for TrendDirection enum."""

    def test_all_directions_defined(self):
        """Test all trend directions are defined."""
        assert TrendDirection.UP.value == "up"
        assert TrendDirection.DOWN.value == "down"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.NEW.value == "new"


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestModelScore:
    """Tests for ModelScore dataclass."""

    def test_creation(self):
        """Test creating a model score."""
        score = ModelScore(
            model_id="gpt-4",
            benchmark_id="mmlu",
            score=0.85,
        )
        assert score.model_id == "gpt-4"
        assert score.benchmark_id == "mmlu"
        assert score.score == 0.85

    def test_default_values(self):
        """Test default values."""
        score = ModelScore(model_id="m", benchmark_id="b", score=0.5)
        assert score.metadata == {}
        assert score.timestamp is None

    def test_to_dict(self):
        """Test dictionary conversion."""
        score = ModelScore(model_id="test", benchmark_id="bench", score=0.75)
        d = score.to_dict()
        assert d["model_id"] == "test"
        assert d["score"] == 0.75


class TestLeaderboardEntry:
    """Tests for LeaderboardEntry dataclass."""

    def test_creation(self):
        """Test creating a leaderboard entry."""
        entry = LeaderboardEntry(
            rank=1,
            model_id="gpt-4",
            score=0.9,
            previous_rank=2,
            trend=TrendDirection.UP,
            score_breakdown={"mmlu": 0.9, "hellaswag": 0.85},
        )
        assert entry.rank == 1
        assert entry.model_id == "gpt-4"
        assert entry.score == 0.9

    def test_rank_change(self):
        """Test rank change calculation."""
        entry = LeaderboardEntry(
            rank=1,
            model_id="m1",
            score=0.9,
            previous_rank=3,
            trend=TrendDirection.UP,
            score_breakdown={},
        )
        assert entry.rank_change == 2  # Improved by 2

    def test_rank_change_none(self):
        """Test rank change when no previous rank."""
        entry = LeaderboardEntry(
            rank=1,
            model_id="m1",
            score=0.9,
            previous_rank=None,
            trend=TrendDirection.NEW,
            score_breakdown={},
        )
        assert entry.rank_change is None

    def test_to_dict(self):
        """Test dictionary conversion."""
        entry = LeaderboardEntry(
            rank=2,
            model_id="test",
            score=0.8,
            previous_rank=1,
            trend=TrendDirection.DOWN,
            score_breakdown={"b1": 0.8},
        )
        d = entry.to_dict()
        assert d["rank"] == 2
        assert d["trend"] == "down"
        assert d["rank_change"] == -1


class TestLeaderboard:
    """Tests for Leaderboard dataclass."""

    def test_creation(self):
        """Test creating a leaderboard."""
        entries = [
            LeaderboardEntry(
                rank=1,
                model_id="m1",
                score=0.9,
                previous_rank=None,
                trend=TrendDirection.NEW,
                score_breakdown={},
            ),
        ]
        lb = Leaderboard(
            name="Test Leaderboard",
            description="Test description",
            entries=entries,
            benchmark_ids=["b1"],
            metric_weights={},
            ranking_method=RankingMethod.SCORE,
            updated_at="2024-01-01T00:00:00",
        )
        assert lb.name == "Test Leaderboard"
        assert lb.n_models == 1

    def test_top_model(self):
        """Test top model property."""
        entries = [
            LeaderboardEntry(
                rank=1,
                model_id="winner",
                score=0.95,
                previous_rank=None,
                trend=TrendDirection.NEW,
                score_breakdown={},
            ),
            LeaderboardEntry(
                rank=2,
                model_id="second",
                score=0.85,
                previous_rank=None,
                trend=TrendDirection.NEW,
                score_breakdown={},
            ),
        ]
        lb = Leaderboard(
            name="Test",
            description="",
            entries=entries,
            benchmark_ids=[],
            metric_weights={},
            ranking_method=RankingMethod.SCORE,
            updated_at="2024-01-01",
        )
        assert lb.top_model == "winner"

    def test_get_entry(self):
        """Test getting entry by model ID."""
        entries = [
            LeaderboardEntry(
                rank=1,
                model_id="m1",
                score=0.9,
                previous_rank=None,
                trend=TrendDirection.NEW,
                score_breakdown={},
            ),
        ]
        lb = Leaderboard(
            name="Test",
            description="",
            entries=entries,
            benchmark_ids=[],
            metric_weights={},
            ranking_method=RankingMethod.SCORE,
            updated_at="2024-01-01",
        )
        assert lb.get_entry("m1") is not None
        assert lb.get_entry("nonexistent") is None


# ============================================================================
# ScoreAggregator Tests
# ============================================================================


class TestScoreAggregator:
    """Tests for ScoreAggregator class."""

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        aggregator = ScoreAggregator(method=ScoreAggregation.MEAN)
        scores = {"b1": 0.8, "b2": 0.9, "b3": 0.7}
        assert aggregator.aggregate(scores) == 0.8

    def test_median_aggregation(self):
        """Test median aggregation."""
        aggregator = ScoreAggregator(method=ScoreAggregation.MEDIAN)
        scores = {"b1": 0.8, "b2": 0.9, "b3": 0.7}
        assert aggregator.aggregate(scores) == 0.8

    def test_max_aggregation(self):
        """Test max aggregation."""
        aggregator = ScoreAggregator(method=ScoreAggregation.MAX)
        scores = {"b1": 0.8, "b2": 0.9, "b3": 0.7}
        assert aggregator.aggregate(scores) == 0.9

    def test_min_aggregation(self):
        """Test min aggregation."""
        aggregator = ScoreAggregator(method=ScoreAggregation.MIN)
        scores = {"b1": 0.8, "b2": 0.9, "b3": 0.7}
        assert aggregator.aggregate(scores) == 0.7

    def test_weighted_aggregation(self):
        """Test weighted aggregation."""
        weights = {"b1": 2.0, "b2": 1.0}
        aggregator = ScoreAggregator(
            method=ScoreAggregation.WEIGHTED,
            weights=weights,
        )
        scores = {"b1": 0.9, "b2": 0.6}
        # (0.9 * 2 + 0.6 * 1) / 3 = 0.8
        assert aggregator.aggregate(scores) == pytest.approx(0.8)

    def test_geometric_aggregation(self):
        """Test geometric mean aggregation."""
        aggregator = ScoreAggregator(method=ScoreAggregation.GEOMETRIC)
        scores = {"b1": 0.8, "b2": 0.8, "b3": 0.8}
        result = aggregator.aggregate(scores)
        assert abs(result - 0.8) < 0.01

    def test_empty_scores(self):
        """Test with empty scores."""
        aggregator = ScoreAggregator()
        assert aggregator.aggregate({}) == 0.0


# ============================================================================
# ModelRanker Tests
# ============================================================================


class TestModelRanker:
    """Tests for ModelRanker class."""

    def test_basic_ranking(self):
        """Test basic model ranking."""
        ranker = ModelRanker()
        model_scores = {
            "m1": {"b1": 0.9},
            "m2": {"b1": 0.8},
            "m3": {"b1": 0.7},
        }
        entries = ranker.rank(model_scores)

        assert len(entries) == 3
        assert entries[0].model_id == "m1"
        assert entries[0].rank == 1
        assert entries[1].model_id == "m2"
        assert entries[2].model_id == "m3"

    def test_ranking_with_previous(self):
        """Test ranking with previous rankings."""
        ranker = ModelRanker()
        model_scores = {
            "m1": {"b1": 0.9},
            "m2": {"b1": 0.8},
        }
        previous = {"m1": 2, "m2": 1}
        entries = ranker.rank(model_scores, previous)

        assert entries[0].model_id == "m1"
        assert entries[0].previous_rank == 2
        assert entries[0].trend == TrendDirection.UP

        assert entries[1].model_id == "m2"
        assert entries[1].previous_rank == 1
        assert entries[1].trend == TrendDirection.DOWN

    def test_new_model_trend(self):
        """Test trend for new models."""
        ranker = ModelRanker()
        model_scores = {"new_model": {"b1": 0.9}}
        entries = ranker.rank(model_scores, {})

        assert entries[0].trend == TrendDirection.NEW

    def test_stable_trend(self):
        """Test stable trend."""
        ranker = ModelRanker()
        model_scores = {"m1": {"b1": 0.9}}
        previous = {"m1": 1}
        entries = ranker.rank(model_scores, previous)

        assert entries[0].trend == TrendDirection.STABLE

    def test_empty_scores(self):
        """Test with empty scores."""
        ranker = ModelRanker()
        assert ranker.rank({}) == []


# ============================================================================
# LeaderboardBuilder Tests
# ============================================================================


class TestLeaderboardBuilder:
    """Tests for LeaderboardBuilder class."""

    def test_add_score(self):
        """Test adding a score."""
        builder = LeaderboardBuilder()
        score = ModelScore(model_id="m1", benchmark_id="b1", score=0.9)
        builder.add_score(score)
        lb = builder.build("Test")
        assert lb.n_models == 1

    def test_add_multiple_scores(self):
        """Test adding multiple scores."""
        builder = LeaderboardBuilder()
        scores = [
            ModelScore(model_id="m1", benchmark_id="b1", score=0.9),
            ModelScore(model_id="m1", benchmark_id="b2", score=0.8),
            ModelScore(model_id="m2", benchmark_id="b1", score=0.7),
        ]
        builder.add_scores(scores)
        lb = builder.build("Test")

        assert lb.n_models == 2
        assert len(lb.benchmark_ids) == 2

    def test_benchmark_filter(self):
        """Test filtering by benchmark."""
        builder = LeaderboardBuilder()
        builder.add_scores(
            [
                ModelScore(model_id="m1", benchmark_id="include", score=0.9),
                ModelScore(model_id="m1", benchmark_id="exclude", score=0.5),
            ]
        )
        lb = builder.build("Test", benchmark_filter=["include"])

        assert "include" in lb.benchmark_ids
        assert "exclude" not in lb.benchmark_ids

    def test_model_filter(self):
        """Test filtering by model."""
        builder = LeaderboardBuilder()
        builder.add_scores(
            [
                ModelScore(model_id="include", benchmark_id="b1", score=0.9),
                ModelScore(model_id="exclude", benchmark_id="b1", score=0.8),
            ]
        )
        lb = builder.build("Test", model_filter=["include"])

        assert lb.n_models == 1
        assert lb.top_model == "include"

    def test_history_tracking(self):
        """Test leaderboard history."""
        builder = LeaderboardBuilder()
        builder.add_scores(
            [
                ModelScore(model_id="m1", benchmark_id="b1", score=0.9),
            ]
        )

        builder.build("History Test")
        builder.build("History Test")  # Same name

        history = builder.get_history("History Test")
        assert len(history) == 2

    def test_clear(self):
        """Test clearing scores."""
        builder = LeaderboardBuilder()
        builder.add_score(ModelScore(model_id="m1", benchmark_id="b1", score=0.9))
        builder.clear()
        lb = builder.build("Test")
        assert lb.n_models == 0


# ============================================================================
# LeaderboardFormatter Tests
# ============================================================================


class TestLeaderboardFormatter:
    """Tests for LeaderboardFormatter class."""

    def _create_sample_leaderboard(self) -> Leaderboard:
        """Create sample leaderboard for testing."""
        entries = [
            LeaderboardEntry(
                rank=1,
                model_id="gpt-4",
                score=0.92,
                previous_rank=2,
                trend=TrendDirection.UP,
                score_breakdown={"mmlu": 0.95, "hellaswag": 0.89},
            ),
            LeaderboardEntry(
                rank=2,
                model_id="claude-3",
                score=0.90,
                previous_rank=1,
                trend=TrendDirection.DOWN,
                score_breakdown={"mmlu": 0.91, "hellaswag": 0.89},
            ),
        ]
        return Leaderboard(
            name="Test Leaderboard",
            description="A test leaderboard",
            entries=entries,
            benchmark_ids=["mmlu", "hellaswag"],
            metric_weights={},
            ranking_method=RankingMethod.SCORE,
            updated_at="2024-01-01T00:00:00",
        )

    def test_to_markdown(self):
        """Test Markdown formatting."""
        formatter = LeaderboardFormatter()
        lb = self._create_sample_leaderboard()
        md = formatter.to_markdown(lb)

        assert "# Test Leaderboard" in md
        assert "gpt-4" in md
        assert "claude-3" in md
        assert "|" in md

    def test_to_csv(self):
        """Test CSV formatting."""
        formatter = LeaderboardFormatter()
        lb = self._create_sample_leaderboard()
        csv = formatter.to_csv(lb)

        lines = csv.split("\n")
        assert "rank" in lines[0]
        assert "model_id" in lines[0]
        assert "gpt-4" in lines[1]

    def test_to_json(self):
        """Test JSON formatting."""
        formatter = LeaderboardFormatter()
        lb = self._create_sample_leaderboard()
        json_str = formatter.to_json(lb)

        data = json.loads(json_str)
        assert data["name"] == "Test Leaderboard"
        assert len(data["entries"]) == 2

    def test_to_html(self):
        """Test HTML formatting."""
        formatter = LeaderboardFormatter()
        lb = self._create_sample_leaderboard()
        html = formatter.to_html(lb)

        assert "<table>" in html
        assert "gpt-4" in html
        assert "Test Leaderboard" in html


# ============================================================================
# LeaderboardAnalyzer Tests
# ============================================================================


class TestLeaderboardAnalyzer:
    """Tests for LeaderboardAnalyzer class."""

    def _create_leaderboard(
        self,
        name: str,
        rankings: dict[str, int],
    ) -> Leaderboard:
        """Create leaderboard with given rankings."""
        entries = [
            LeaderboardEntry(
                rank=rank,
                model_id=model,
                score=1.0 - rank * 0.1,
                previous_rank=None,
                trend=TrendDirection.NEW,
                score_breakdown={},
            )
            for model, rank in sorted(rankings.items(), key=lambda x: x[1])
        ]
        return Leaderboard(
            name=name,
            description="",
            entries=entries,
            benchmark_ids=[],
            metric_weights={},
            ranking_method=RankingMethod.SCORE,
            updated_at="2024-01-01",
        )

    def test_compare_identical_leaderboards(self):
        """Test comparing identical leaderboards."""
        lb = self._create_leaderboard("Test", {"m1": 1, "m2": 2, "m3": 3})

        analyzer = LeaderboardAnalyzer()
        comparison = analyzer.compare_leaderboards(lb, lb)

        assert comparison.rank_correlations == 1.0
        assert len(comparison.divergent_rankings) == 0

    def test_compare_different_leaderboards(self):
        """Test comparing different leaderboards."""
        lb_a = self._create_leaderboard("A", {"m1": 1, "m2": 2, "m3": 3})
        lb_b = self._create_leaderboard("B", {"m1": 3, "m2": 2, "m3": 1})

        analyzer = LeaderboardAnalyzer()
        comparison = analyzer.compare_leaderboards(lb_a, lb_b)

        assert comparison.rank_correlations < 0  # Negative correlation
        assert len(comparison.common_models) == 3

    def test_compare_no_common_models(self):
        """Test comparing leaderboards with no common models."""
        lb_a = self._create_leaderboard("A", {"m1": 1})
        lb_b = self._create_leaderboard("B", {"m2": 1})

        analyzer = LeaderboardAnalyzer()
        comparison = analyzer.compare_leaderboards(lb_a, lb_b)

        assert len(comparison.common_models) == 0
        assert comparison.rank_correlations == 0.0

    def test_analyze_trends(self):
        """Test trend analysis."""
        # Create history with improving model
        history = [
            self._create_leaderboard("H1", {"m1": 3}),
            self._create_leaderboard("H2", {"m1": 2}),
            self._create_leaderboard("H3", {"m1": 1}),
        ]

        analyzer = LeaderboardAnalyzer()
        trends = analyzer.analyze_trends(history, "m1")

        assert trends["found"] is True
        assert trends["n_appearances"] == 3
        assert trends["best_rank"] == 1
        assert trends["worst_rank"] == 3

    def test_analyze_trends_missing_model(self):
        """Test trend analysis for missing model."""
        history = [self._create_leaderboard("H1", {"m1": 1})]

        analyzer = LeaderboardAnalyzer()
        trends = analyzer.analyze_trends(history, "nonexistent")

        assert trends["found"] is False


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_leaderboard(self):
        """Test create_leaderboard function."""
        scores = [
            ModelScore(model_id="m1", benchmark_id="b1", score=0.9),
            ModelScore(model_id="m2", benchmark_id="b1", score=0.8),
        ]
        lb = create_leaderboard(scores, "Test")

        assert isinstance(lb, Leaderboard)
        assert lb.name == "Test"
        assert lb.n_models == 2
        assert lb.top_model == "m1"

    def test_format_leaderboard(self):
        """Test format_leaderboard function."""
        scores = [ModelScore(model_id="m1", benchmark_id="b1", score=0.9)]
        lb = create_leaderboard(scores, "Test")

        md = format_leaderboard(lb, "markdown")
        assert "# Test" in md

        csv = format_leaderboard(lb, "csv")
        assert "rank" in csv

        json_str = format_leaderboard(lb, "json")
        assert json.loads(json_str)["name"] == "Test"

    def test_compare_leaderboards_function(self):
        """Test compare_leaderboards function."""
        scores_a = [ModelScore(model_id="m1", benchmark_id="b1", score=0.9)]
        scores_b = [ModelScore(model_id="m1", benchmark_id="b1", score=0.8)]

        lb_a = create_leaderboard(scores_a, "A")
        lb_b = create_leaderboard(scores_b, "B")

        comparison = compare_leaderboards(lb_a, lb_b)

        assert isinstance(comparison, LeaderboardComparison)
        assert comparison.leaderboard_a == "A"

    def test_quick_leaderboard(self):
        """Test quick_leaderboard function."""
        model_scores = {
            "gpt-4": {"mmlu": 0.9, "hellaswag": 0.85},
            "claude": {"mmlu": 0.88, "hellaswag": 0.87},
        }
        lb = quick_leaderboard(model_scores, "Quick Test")

        assert isinstance(lb, Leaderboard)
        assert lb.n_models == 2


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_model(self):
        """Test with single model."""
        scores = [ModelScore(model_id="only", benchmark_id="b1", score=0.5)]
        lb = create_leaderboard(scores, "Single")

        assert lb.n_models == 1
        assert lb.top_model == "only"

    def test_single_benchmark(self):
        """Test with single benchmark."""
        scores = [
            ModelScore(model_id="m1", benchmark_id="only", score=0.9),
            ModelScore(model_id="m2", benchmark_id="only", score=0.8),
        ]
        lb = create_leaderboard(scores, "Single Bench")

        assert len(lb.benchmark_ids) == 1

    def test_tied_scores(self):
        """Test with tied scores."""
        scores = [
            ModelScore(model_id="m1", benchmark_id="b1", score=0.9),
            ModelScore(model_id="m2", benchmark_id="b1", score=0.9),
        ]
        lb = create_leaderboard(scores, "Tied")

        # Both should be ranked
        assert lb.n_models == 2

    def test_zero_scores(self):
        """Test with zero scores."""
        scores = [
            ModelScore(model_id="m1", benchmark_id="b1", score=0.0),
        ]
        lb = create_leaderboard(scores, "Zero")

        assert lb.entries[0].score == 0.0

    def test_many_models(self):
        """Test with many models."""
        scores = [
            ModelScore(model_id=f"m{i}", benchmark_id="b1", score=i / 100) for i in range(100)
        ]
        lb = create_leaderboard(scores, "Many")

        assert lb.n_models == 100
        assert lb.entries[0].rank == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for leaderboard module."""

    def test_full_workflow(self):
        """Test complete leaderboard workflow."""
        # Create scores from multiple runs
        scores = [
            # Model A
            ModelScore(model_id="model-a", benchmark_id="accuracy", score=0.92),
            ModelScore(model_id="model-a", benchmark_id="latency", score=0.85),
            ModelScore(model_id="model-a", benchmark_id="cost", score=0.70),
            # Model B
            ModelScore(model_id="model-b", benchmark_id="accuracy", score=0.88),
            ModelScore(model_id="model-b", benchmark_id="latency", score=0.95),
            ModelScore(model_id="model-b", benchmark_id="cost", score=0.90),
            # Model C
            ModelScore(model_id="model-c", benchmark_id="accuracy", score=0.90),
            ModelScore(model_id="model-c", benchmark_id="latency", score=0.80),
            ModelScore(model_id="model-c", benchmark_id="cost", score=0.85),
        ]

        # Create weighted leaderboard
        weights = {"accuracy": 2.0, "latency": 1.0, "cost": 1.0}
        lb = create_leaderboard(
            scores,
            name="Model Comparison",
            description="Comparing 3 models",
            weights=weights,
        )

        # Verify structure
        assert lb.n_models == 3
        assert len(lb.benchmark_ids) == 3

        # Verify ranking
        assert lb.entries[0].rank == 1
        assert lb.entries[1].rank == 2
        assert lb.entries[2].rank == 3

        # Format and verify
        md = format_leaderboard(lb, "markdown")
        assert "Model Comparison" in md
        assert "model-a" in md

    def test_historical_tracking(self):
        """Test historical leaderboard tracking."""
        builder = LeaderboardBuilder()

        # Initial scores
        builder.add_scores(
            [
                ModelScore(model_id="rising", benchmark_id="b1", score=0.5),
                ModelScore(model_id="falling", benchmark_id="b1", score=0.9),
            ]
        )
        builder.build("History")

        # Clear and add new scores
        builder.clear()
        builder.add_scores(
            [
                ModelScore(model_id="rising", benchmark_id="b1", score=0.95),
                ModelScore(model_id="falling", benchmark_id="b1", score=0.4),
            ]
        )
        lb2 = builder.build("History")

        # Check trends
        rising_entry = lb2.get_entry("rising")
        falling_entry = lb2.get_entry("falling")

        assert rising_entry.trend == TrendDirection.UP
        assert falling_entry.trend == TrendDirection.DOWN

    def test_serialization_roundtrip(self):
        """Test that results can be serialized."""
        scores = [
            ModelScore(model_id="m1", benchmark_id="b1", score=0.9),
        ]
        lb = create_leaderboard(scores, "Test")

        # Convert to JSON
        json_str = format_leaderboard(lb, "json")

        # Parse and verify
        data = json.loads(json_str)
        assert data["name"] == "Test"
        assert len(data["entries"]) == 1
