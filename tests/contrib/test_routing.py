"""Tests for Semantic Model Routing module."""

from unittest.mock import MagicMock

from insideLLMs.contrib.routing import (
    IntentClassifier,
    ModelPool,
    Route,
    RouteMatch,
    RouterConfig,
    RouteStats,
    RouteStatus,
    RoutingStrategy,
    SemanticMatcher,
    SemanticRouter,
    create_router,
    quick_route,
)
from insideLLMs.models import DummyModel

# =============================================================================
# Test Configuration
# =============================================================================


class TestRouterConfig:
    """Tests for RouterConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = RouterConfig()

        assert config.strategy == RoutingStrategy.BEST_MATCH
        assert config.similarity_threshold == 0.3
        assert config.use_fallback is True
        assert config.max_retries == 2

    def test_custom_values(self):
        """Test custom configuration."""
        config = RouterConfig(
            strategy=RoutingStrategy.ROUND_ROBIN,
            similarity_threshold=0.5,
            max_retries=5,
        )

        assert config.strategy == RoutingStrategy.ROUND_ROBIN
        assert config.similarity_threshold == 0.5

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = RouterConfig()
        data = config.to_dict()

        assert "strategy" in data
        assert "similarity_threshold" in data


class TestRoutingStrategy:
    """Tests for RoutingStrategy enum."""

    def test_strategy_values(self):
        """Test strategy values."""
        assert RoutingStrategy.FIRST_MATCH.value == "first_match"
        assert RoutingStrategy.BEST_MATCH.value == "best_match"
        assert RoutingStrategy.ROUND_ROBIN.value == "round_robin"
        assert RoutingStrategy.LOAD_BALANCED.value == "load_balanced"


# =============================================================================
# Test Route
# =============================================================================


class TestRoute:
    """Tests for Route class."""

    def test_basic_route(self):
        """Test basic route creation."""
        model = DummyModel()
        route = Route(
            name="test",
            model=model,
            patterns=["hello", "hi"],
            description="Test route",
        )

        assert route.name == "test"
        assert route.model is model
        assert len(route.patterns) == 2

    def test_route_matches_pattern(self):
        """Test pattern matching."""
        route = Route(
            name="code",
            model=DummyModel(),
            patterns=["code", "programming", "function"],
        )

        matches, score, patterns = route.matches("Write some code for me")

        assert matches is True
        assert score > 0
        assert "code" in patterns

    def test_route_no_match(self):
        """Test non-matching query."""
        route = Route(
            name="code",
            model=DummyModel(),
            patterns=["code", "programming"],
        )

        matches, score, patterns = route.matches("What is the weather?")

        assert matches is False
        assert score == 0

    def test_route_regex_pattern(self):
        """Test regex pattern matching."""
        route = Route(
            name="email",
            model=DummyModel(),
            regex_patterns=[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
        )

        matches, score, _ = route.matches("Send to test@example.com")

        assert matches is True

    def test_route_priority(self):
        """Test route priority."""
        route = Route(
            name="high",
            model=DummyModel(),
            patterns=["test"],
            priority=10,
        )

        assert route.priority == 10

    def test_route_stats(self):
        """Test route statistics."""
        route = Route(name="test", model=DummyModel())

        assert route.stats.total_requests == 0
        assert route.stats.route_name == "test"

    def test_route_to_dict(self):
        """Test dictionary conversion."""
        route = Route(
            name="test",
            model=DummyModel(),
            patterns=["test"],
            description="Test route",
        )

        data = route.to_dict()

        assert data["name"] == "test"
        assert data["patterns"] == ["test"]


# =============================================================================
# Test RouteStats
# =============================================================================


class TestRouteStats:
    """Tests for RouteStats."""

    def test_initial_stats(self):
        """Test initial statistics."""
        stats = RouteStats(route_name="test")

        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.avg_latency_ms == 0

    def test_update_success(self):
        """Test updating with success."""
        stats = RouteStats(route_name="test")
        stats.update(success=True, latency_ms=100)

        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.failed_requests == 0
        assert stats.avg_latency_ms == 100

    def test_update_failure(self):
        """Test updating with failure."""
        stats = RouteStats(route_name="test")
        stats.update(success=False, latency_ms=50)

        assert stats.total_requests == 1
        assert stats.failed_requests == 1
        assert stats.error_rate == 1.0

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        stats = RouteStats(route_name="test")
        stats.update(success=True, latency_ms=100)
        stats.update(success=True, latency_ms=100)
        stats.update(success=False, latency_ms=100)
        stats.update(success=False, latency_ms=100)

        assert stats.error_rate == 0.5


# =============================================================================
# Test SemanticMatcher
# =============================================================================


class TestSemanticMatcher:
    """Tests for SemanticMatcher."""

    def test_word_similarity_identical(self):
        """Test word similarity with identical texts."""
        matcher = SemanticMatcher()
        sim = matcher.compute_similarity("hello world", "hello world")

        assert sim == 1.0

    def test_word_similarity_partial(self):
        """Test word similarity with partial overlap."""
        matcher = SemanticMatcher()
        sim = matcher.compute_similarity("hello world", "hello there")

        assert 0 < sim < 1

    def test_word_similarity_no_overlap(self):
        """Test word similarity with no overlap."""
        matcher = SemanticMatcher()
        sim = matcher.compute_similarity("apple banana", "car house")

        assert sim == 0.0

    def test_word_similarity_empty(self):
        """Test word similarity with empty string."""
        matcher = SemanticMatcher()
        sim = matcher.compute_similarity("", "hello")

        assert sim == 0.0

    def test_with_custom_embedder(self):
        """Test with custom embedder."""

        def mock_embedder(text: str) -> list:
            return [1.0, 0.0, 0.0]

        matcher = SemanticMatcher(embedder=mock_embedder)
        sim = matcher.compute_similarity("test1", "test2")

        assert sim == 1.0  # Same mock embedding


# =============================================================================
# Test SemanticRouter
# =============================================================================


class TestSemanticRouter:
    """Tests for SemanticRouter."""

    def test_initialization(self):
        """Test router initialization."""
        router = SemanticRouter()

        assert router.config is not None
        assert router.fallback_model is None

    def test_add_route(self):
        """Test adding routes."""
        router = SemanticRouter()
        route = Route(name="test", model=DummyModel(), patterns=["test"])

        router.add_route(route)

        assert router.get_route("test") is route

    def test_remove_route(self):
        """Test removing routes."""
        router = SemanticRouter()
        router.add_route(Route(name="test", model=DummyModel()))

        result = router.remove_route("test")

        assert result is True
        assert router.get_route("test") is None

    def test_list_routes(self):
        """Test listing routes."""
        router = SemanticRouter()
        router.add_route(Route(name="r1", model=DummyModel()))
        router.add_route(Route(name="r2", model=DummyModel()))

        routes = router.list_routes()

        assert len(routes) == 2

    def test_match_single_route(self):
        """Test matching with single route."""
        router = SemanticRouter()
        router.add_route(
            Route(
                name="code",
                model=DummyModel(),
                patterns=["code", "program"],
            )
        )

        matches = router.match("Write some code")

        assert len(matches) >= 1
        assert matches[0].route_name == "code"

    def test_match_multiple_routes(self):
        """Test matching with multiple routes."""
        router = SemanticRouter()
        router.add_route(
            Route(
                name="code",
                model=DummyModel(),
                patterns=["code", "program"],
            )
        )
        router.add_route(
            Route(
                name="python",
                model=DummyModel(),
                patterns=["python", "code"],
            )
        )

        matches = router.match("Write Python code")

        assert len(matches) >= 1

    def test_select_route_best_match(self):
        """Test selecting best match."""
        config = RouterConfig(strategy=RoutingStrategy.BEST_MATCH)
        router = SemanticRouter(config)

        matches = [
            RouteMatch("low", 0.3, [], DummyModel()),
            RouteMatch("high", 0.9, [], DummyModel()),
            RouteMatch("medium", 0.5, [], DummyModel()),
        ]

        # Sort by score for BEST_MATCH
        matches.sort(key=lambda m: m.score, reverse=True)
        selected = router.select_route(matches)

        assert selected.route_name == "high"

    def test_select_route_round_robin(self):
        """Test round robin selection."""
        config = RouterConfig(strategy=RoutingStrategy.ROUND_ROBIN)
        router = SemanticRouter(config)

        matches = [
            RouteMatch("a", 0.5, [], DummyModel()),
            RouteMatch("b", 0.5, [], DummyModel()),
            RouteMatch("c", 0.5, [], DummyModel()),
        ]

        selected1 = router.select_route(matches)
        selected2 = router.select_route(matches)
        selected3 = router.select_route(matches)

        # Should cycle through
        names = [selected1.route_name, selected2.route_name, selected3.route_name]
        assert len(set(names)) == 3

    def test_route_with_fallback(self):
        """Test routing with fallback."""
        fallback = DummyModel(name="Fallback")
        router = SemanticRouter(fallback_model=fallback)

        result = router.route("Unmatched query")

        assert result.fallback_used is True

    def test_route_success(self):
        """Test successful routing."""
        model = MagicMock()
        model.generate.return_value = "Response"

        router = SemanticRouter()
        router.add_route(
            Route(
                name="test",
                model=model,
                patterns=["test"],
            )
        )

        result = router.route("Test query")

        assert result.route_name == "test"
        assert result.response == "Response"

    def test_route_no_match_no_fallback(self):
        """Test routing with no match and no fallback."""
        config = RouterConfig(use_fallback=False)
        router = SemanticRouter(config)

        result = router.route("Unmatched query")

        assert result.route_name == "none"
        assert result.response is None

    def test_get_stats(self):
        """Test getting statistics."""
        router = SemanticRouter()
        router.add_route(Route(name="r1", model=DummyModel()))
        router.add_route(Route(name="r2", model=DummyModel()))

        stats = router.get_stats()

        assert "r1" in stats
        assert "r2" in stats


# =============================================================================
# Test ModelPool
# =============================================================================


class TestModelPool:
    """Tests for ModelPool."""

    def test_initialization(self):
        """Test pool initialization."""
        models = [DummyModel(), DummyModel()]
        pool = ModelPool(models)

        assert len(pool.models) == 2

    def test_round_robin_selection(self):
        """Test round robin selection."""
        models = [DummyModel(name=f"M{i}") for i in range(3)]
        pool = ModelPool(models, strategy=RoutingStrategy.ROUND_ROBIN)

        selections = [pool.select()[1] for _ in range(6)]

        # Should cycle: 0, 1, 2, 0, 1, 2
        assert selections == [0, 1, 2, 0, 1, 2]

    def test_random_selection(self):
        """Test random selection."""
        models = [DummyModel() for _ in range(3)]
        pool = ModelPool(models, strategy=RoutingStrategy.RANDOM)

        # Should select without error
        model, idx = pool.select()
        assert 0 <= idx < 3

    def test_generate(self):
        """Test generating through pool."""
        model = MagicMock()
        model.generate.return_value = "Response"

        pool = ModelPool([model])
        response = pool.generate("Test")

        assert response == "Response"
        assert model.generate.called

    def test_stats(self):
        """Test pool statistics."""
        models = [DummyModel() for _ in range(2)]
        pool = ModelPool(models)

        stats = pool.stats()

        assert 0 in stats
        assert 1 in stats


# =============================================================================
# Test IntentClassifier
# =============================================================================


class TestIntentClassifier:
    """Tests for IntentClassifier."""

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = IntentClassifier()
        assert len(classifier.intents) == 0

    def test_add_intent(self):
        """Test adding intents."""
        classifier = IntentClassifier()
        classifier.add_intent("code", "Code generation", ["code", "program"])

        assert "code" in classifier.intents

    def test_classify_with_patterns(self):
        """Test classification with patterns."""
        classifier = IntentClassifier()
        classifier.add_intent("code", "Code generation", ["code", "program", "function"])
        classifier.add_intent("chat", "General chat", ["hello", "hi", "chat"])

        intent, confidence = classifier.classify("Write a function")

        assert intent == "code"
        assert confidence > 0

    def test_classify_no_match(self):
        """Test classification with no match."""
        classifier = IntentClassifier()
        classifier.add_intent("code", "Code generation", ["code"])

        intent, confidence = classifier.classify("What is the weather?")

        # Should return unknown or low confidence
        assert confidence < 0.5 or intent == "unknown"

    def test_classify_with_model(self):
        """Test classification with model."""
        model = MagicMock()
        model.generate.return_value = "code"

        classifier = IntentClassifier(model=model)
        classifier.add_intent("code", "Code generation")
        classifier.add_intent("chat", "General chat")

        intent, confidence = classifier.classify("Generate some code")

        # Should use model for classification
        assert model.generate.called


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_router(self):
        """Test create_router function."""
        model = DummyModel()
        routes = [
            {"name": "test", "model": model, "patterns": ["test"]},
        ]

        router = create_router(routes)

        assert router.get_route("test") is not None

    def test_create_router_with_fallback(self):
        """Test create_router with fallback."""
        model = DummyModel()
        fallback = DummyModel(name="Fallback")

        router = create_router(
            routes=[{"name": "test", "model": model, "patterns": ["test"]}],
            fallback_model=fallback,
        )

        assert router.fallback_model is fallback

    def test_quick_route(self):
        """Test quick_route function."""
        model = MagicMock()
        model.generate.return_value = "Response"

        result = quick_route(
            "Test query",
            routes=[{"name": "test", "model": model, "patterns": ["test"]}],
        )

        assert result.route_name == "test"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_patterns(self):
        """Test route with no patterns."""
        route = Route(name="empty", model=DummyModel())

        matches, score, patterns = route.matches("any query")

        assert matches is False
        assert score == 0

    def test_case_sensitivity(self):
        """Test case-sensitive matching."""
        route = Route(name="test", model=DummyModel(), patterns=["Test"])

        # Case insensitive (default)
        matches1, _, _ = route.matches("test query", case_sensitive=False)
        assert matches1 is True

        # Case sensitive
        matches2, _, _ = route.matches("test query", case_sensitive=True)
        assert matches2 is False

    def test_disabled_route(self):
        """Test disabled route not matched."""
        router = SemanticRouter()
        route = Route(name="test", model=DummyModel(), patterns=["test"])
        route.status = RouteStatus.DISABLED

        router.add_route(route)
        matches = router.match("test query")

        assert len(matches) == 0

    def test_concurrent_routing(self):
        """Test thread safety."""
        import threading

        model = MagicMock()
        model.generate.return_value = "Response"

        router = SemanticRouter()
        router.add_route(Route(name="test", model=model, patterns=["test"]))

        errors = []

        def route_query():
            try:
                router.route("test query")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=route_query) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_special_characters_in_query(self):
        """Test special characters in query."""
        route = Route(name="test", model=DummyModel(), patterns=["test"])

        matches, _, _ = route.matches("Test with 'quotes' and \"doubles\"")

        assert matches is True

    def test_unicode_query(self):
        """Test unicode in query."""
        route = Route(name="test", model=DummyModel(), patterns=["测试"])

        matches, _, _ = route.matches("这是一个测试")

        assert matches is True
