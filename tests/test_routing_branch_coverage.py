import hashlib
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.models import DummyModel
from insideLLMs.routing import (
    IntentClassifier,
    ModelPool,
    Route,
    RouteMatch,
    RouterConfig,
    RouteStats,
    RoutingResult,
    RoutingStrategy,
    SemanticMatcher,
    SemanticRouter,
    create_intent_router,
    create_router,
)


def test_route_match_and_routing_result_to_dict_paths():
    match = RouteMatch(
        route_name="route-a",
        score=0.9,
        matched_patterns=["test"],
        model=DummyModel(),
        metadata={"priority": 1},
    )
    match_dict = match.to_dict()
    assert match_dict["route_name"] == "route-a"
    assert match_dict["model"] == "DummyModel"

    result = RoutingResult(
        query="hello",
        route_name="route-a",
        response="x" * 300,
        latency_ms=10.0,
        match_score=0.9,
    )
    result_dict = result.to_dict()
    assert result_dict["response"] == "x" * 200

    stats = RouteStats(route_name="route-a")
    stats.update(success=True, latency_ms=3.0)
    stats_dict = stats.to_dict()
    assert stats_dict["route_name"] == "route-a"
    assert stats_dict["total_requests"] == 1


def test_semantic_matcher_cosine_edge_cases_and_embedding_cache():
    matcher = SemanticMatcher()
    assert matcher._cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert matcher._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def embedder(text):
        return [float(len(text))]

    embed_matcher = SemanticMatcher(embedder=embedder)
    embed_matcher.cache_route_embedding("r1", "description")
    assert embed_matcher.get_cached_embedding("r1") == [11.0]
    assert embed_matcher.get_cached_embedding("missing") is None


def test_add_route_description_caches_embedding_and_remove_missing_returns_false():
    router = SemanticRouter()
    with patch.object(router._matcher, "cache_route_embedding") as cache_embedding:
        router.add_route(
            Route(
                name="support",
                model=DummyModel(),
                patterns=["help"],
                description="Customer support route",
            )
        )
    cache_embedding.assert_called_once_with("support", "Customer support route")
    assert router.remove_route("does-not-exist") is False


def test_router_match_semantic_threshold_without_pattern_match():
    def embedder(text: str) -> list[float]:
        return [1.0, 0.0] if "semantic" in text else [1.0, 0.0]

    config = RouterConfig(similarity_threshold=0.2)
    router = SemanticRouter(config=config, embedder=embedder)
    router.add_route(
        Route(
            name="semantic",
            model=DummyModel(),
            patterns=["different-pattern"],
            description="semantic route",
        )
    )

    matches = router.match("query with no explicit different-pattern token")
    assert matches
    assert matches[0].route_name == "semantic"


def test_select_route_strategy_branches():
    base_matches = [
        RouteMatch("a", 0.9, [], DummyModel(), metadata={"priority": 1}),
        RouteMatch("b", 0.7, [], DummyModel(), metadata={"priority": 2}),
    ]

    router_first = SemanticRouter(config=RouterConfig(strategy=RoutingStrategy.FIRST_MATCH))
    assert router_first.select_route(base_matches).route_name == "a"

    router_random = SemanticRouter(config=RouterConfig(strategy=RoutingStrategy.RANDOM))
    with patch("insideLLMs.routing.random.choice", return_value=base_matches[1]):
        assert router_random.select_route(base_matches).route_name == "b"

    router_load = SemanticRouter(config=RouterConfig(strategy=RoutingStrategy.LOAD_BALANCED))
    router_load.add_route(Route(name="a", model=DummyModel(), patterns=["a"]))
    router_load.add_route(Route(name="b", model=DummyModel(), patterns=["b"]))
    router_load._routes["a"]._stats.current_load = 5
    router_load._routes["b"]._stats.current_load = 1
    assert router_load.select_route(base_matches).route_name == "b"

    router_cost = SemanticRouter(config=RouterConfig(strategy=RoutingStrategy.COST_OPTIMIZED))
    router_cost.add_route(Route(name="a", model=DummyModel(), patterns=["a"], cost_per_token=0.5))
    router_cost.add_route(Route(name="b", model=DummyModel(), patterns=["b"], cost_per_token=0.1))
    assert router_cost.select_route(base_matches).route_name == "b"

    router_unknown = SemanticRouter()
    router_unknown.config.strategy = "unknown"  # type: ignore[assignment]
    assert router_unknown.select_route(base_matches).route_name == "a"


def test_router_cache_missing_route_branch_and_no_cache_branch():
    model = MagicMock()
    model.generate.return_value = "ok"

    router = SemanticRouter()
    router.add_route(Route(name="real", model=model, patterns=["hello"]))

    query = "hello world"
    key = hashlib.md5(query.encode()).hexdigest()
    router._route_cache[key] = ("missing-route", datetime.now())

    result = router.route(query)
    assert result.route_name == "real"

    no_cache_router = SemanticRouter(config=RouterConfig(cache_routes=False))
    no_cache_router.add_route(Route(name="real", model=model, patterns=["hello"]))
    result_no_cache = no_cache_router.route(query)
    assert result_no_cache.route_name == "real"


def test_router_retry_error_path_and_cache_expiry():
    model = MagicMock()
    model.generate.side_effect = RuntimeError("route-failure")

    config = RouterConfig(max_retries=1, retry_delay_ms=1)
    router = SemanticRouter(config=config)
    router.add_route(Route(name="fail", model=model, patterns=["fail"]))

    with patch("insideLLMs.routing.time.sleep", return_value=None):
        result = router.route("fail query")

    assert result.route_name == "fail"
    assert result.retries == 2
    assert str(result.response).startswith("Error: route-failure")

    cache_query = "expired-cache-query"
    cache_key = hashlib.md5(cache_query.encode()).hexdigest()
    router._route_cache[cache_key] = (
        "some-route",
        datetime.now() - timedelta(seconds=router.config.cache_ttl_seconds + 1),
    )
    assert router._get_cached_route(cache_query) is None
    assert cache_key not in router._route_cache


def test_model_pool_load_balanced_else_and_generate_exception_paths():
    models = [MagicMock(), MagicMock()]
    models[0].generate.return_value = "a"
    models[1].generate.return_value = "b"

    pool = ModelPool(models, strategy=RoutingStrategy.LOAD_BALANCED)
    pool._stats[0].current_load = 10
    pool._stats[1].current_load = 1
    _, idx = pool.select()
    assert idx == 1

    pool.strategy = "unknown"  # type: ignore[assignment]
    _, fallback_idx = pool.select()
    assert fallback_idx == 0

    failing_model = MagicMock()
    failing_model.generate.side_effect = ValueError("pool-fail")
    failing_pool = ModelPool([failing_model])
    with pytest.raises(ValueError, match="pool-fail"):
        failing_pool.generate("prompt")


def test_intent_classifier_pattern_model_unknown_and_error_paths():
    classifier = IntentClassifier()
    classifier.add_intent("urgent", "Urgent issues", ["urgent", "asap"])
    intent, confidence = classifier.classify("urgent asap response needed")
    assert intent == "urgent"
    assert confidence > 0.5

    model_classifier = IntentClassifier(model=MagicMock())
    model_classifier.add_intent("booking", "Travel booking")
    model_classifier.model.generate.return_value = "booking"
    assert model_classifier._classify_with_model("book a flight") == ("booking", 0.8)

    model_classifier.model.generate.return_value = "something else"
    assert model_classifier._classify_with_model("book a flight") == ("unknown", 0.3)

    model_classifier.model.generate.side_effect = RuntimeError("classifier-fail")
    assert model_classifier._classify_with_model("book a flight") == ("unknown", 0.0)


def test_create_router_without_routes_and_create_intent_router_paths():
    fallback = DummyModel(name="fallback")
    router = create_router(routes=None, fallback_model=fallback, strategy="best_match")
    result = router.route("unmatched query")
    assert result.fallback_used is True

    intent_router = create_intent_router(
        model=DummyModel(),
        intent_models={"support": DummyModel(), "billing": DummyModel()},
        fallback_model=fallback,
    )
    route_names = {route.name for route in intent_router.list_routes()}
    assert route_names == {"support", "billing"}
    assert intent_router.get_route("support").description == "Handle support queries"
