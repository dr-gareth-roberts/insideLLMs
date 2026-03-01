"""Semantic Model Routing Module for insideLLMs.

This module provides intelligent routing of queries to appropriate language models
based on semantic similarity, pattern matching, and configurable routing strategies.
It enables building sophisticated multi-model systems where different queries are
automatically directed to the most appropriate model.

Key Features:
    - Semantic route matching based on query content and embeddings
    - Dynamic model selection based on task type and query patterns
    - Multiple routing strategies (best match, round robin, load balanced, cost optimized)
    - Fallback and retry strategies for robustness
    - Cost-aware routing for budget optimization
    - Load balancing across multiple model instances
    - Route caching for improved performance
    - Comprehensive statistics tracking

Core Components:
    - SemanticRouter: Main router class for query-to-model routing
    - Route: Defines a mapping from patterns/semantics to a model
    - SemanticMatcher: Computes semantic similarity between queries and routes
    - ModelPool: Load balances across multiple model instances
    - IntentClassifier: Classifies query intent for routing decisions

Example - Basic Routing:
    >>> from insideLLMs.agents.routing import SemanticRouter, Route
    >>> from insideLLMs import DummyModel
    >>>
    >>> # Create a router with routes for different query types
    >>> router = SemanticRouter()
    >>> router.add_route(Route(
    ...     name="code",
    ...     patterns=["code", "programming", "function", "debug"],
    ...     model=DummyModel(name="CodeModel"),
    ...     description="Handles programming and coding questions",
    ... ))
    >>> router.add_route(Route(
    ...     name="general",
    ...     patterns=["what", "how", "explain", "describe"],
    ...     model=DummyModel(name="GeneralModel"),
    ...     description="Handles general knowledge questions",
    ... ))
    >>>
    >>> # Route a query - automatically selects the best matching model
    >>> result = router.route("Write a Python function to sort a list")
    >>> print(result.route_name)  # "code"

Example - Cost-Optimized Routing:
    >>> from insideLLMs.agents.routing import SemanticRouter, Route, RouterConfig, RoutingStrategy
    >>> from insideLLMs import DummyModel
    >>>
    >>> # Configure router for cost optimization
    >>> config = RouterConfig(
    ...     strategy=RoutingStrategy.COST_OPTIMIZED,
    ...     similarity_threshold=0.2,
    ... )
    >>> router = SemanticRouter(config=config)
    >>>
    >>> # Add routes with different costs
    >>> router.add_route(Route(
    ...     name="premium",
    ...     patterns=["complex", "analyze", "research"],
    ...     model=DummyModel(name="GPT-4"),
    ...     cost_per_token=0.03,
    ... ))
    >>> router.add_route(Route(
    ...     name="budget",
    ...     patterns=["simple", "basic", "quick"],
    ...     model=DummyModel(name="GPT-3.5"),
    ...     cost_per_token=0.002,
    ... ))

Example - Load Balanced Model Pool:
    >>> from insideLLMs.agents.routing import ModelPool, RoutingStrategy
    >>> from insideLLMs import DummyModel
    >>>
    >>> # Create a pool of models for load balancing
    >>> models = [DummyModel(name=f"Model_{i}") for i in range(3)]
    >>> pool = ModelPool(models, strategy=RoutingStrategy.LOAD_BALANCED)
    >>>
    >>> # Generate responses - automatically distributes load
    >>> response = pool.generate("Hello, world!")
    >>> stats = pool.stats()

Example - Intent-Based Routing:
    >>> from insideLLMs.agents.routing import IntentClassifier, create_intent_router
    >>> from insideLLMs import DummyModel
    >>>
    >>> # Create classifier with intent patterns
    >>> classifier = IntentClassifier()
    >>> classifier.add_intent("code_help", "Programming assistance", ["code", "function", "bug"])
    >>> classifier.add_intent("math_help", "Mathematical problems", ["calculate", "solve", "equation"])
    >>>
    >>> # Classify a query
    >>> intent, confidence = classifier.classify("Help me fix this bug in my code")
    >>> print(intent)  # "code_help"
"""

import hashlib
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Optional,
)

from insideLLMs.nlp.similarity import word_overlap_similarity

# =============================================================================
# Configuration and Types
# =============================================================================


class RoutingStrategy(Enum):
    """Strategy for selecting among matching routes when multiple routes match a query.

    The routing strategy determines how the router chooses between multiple candidate
    routes that all match a given query. Different strategies optimize for different
    goals such as performance, cost, or load distribution.

    Attributes:
        FIRST_MATCH: Use the first matching route found (fastest, deterministic).
        BEST_MATCH: Use the route with the highest similarity score (most accurate).
        ROUND_ROBIN: Rotate sequentially among matching routes (even distribution).
        RANDOM: Randomly select from matching routes (statistical distribution).
        LOAD_BALANCED: Select route with lowest current load (optimal throughput).
        COST_OPTIMIZED: Prefer routes with lower cost_per_token (budget friendly).

    Example - Using BEST_MATCH strategy:
        >>> from insideLLMs.agents.routing import RouterConfig, RoutingStrategy
        >>> config = RouterConfig(strategy=RoutingStrategy.BEST_MATCH)
        >>> # Router will select the route with highest similarity score

    Example - Using LOAD_BALANCED strategy:
        >>> from insideLLMs.agents.routing import RouterConfig, RoutingStrategy
        >>> config = RouterConfig(strategy=RoutingStrategy.LOAD_BALANCED)
        >>> # Router will select the route with fewest active requests

    Example - Using COST_OPTIMIZED strategy:
        >>> from insideLLMs.agents.routing import RouterConfig, RoutingStrategy
        >>> config = RouterConfig(strategy=RoutingStrategy.COST_OPTIMIZED)
        >>> # Router will prefer cheaper models when multiple routes match

    Example - Using ROUND_ROBIN for even distribution:
        >>> from insideLLMs.agents.routing import RouterConfig, RoutingStrategy
        >>> config = RouterConfig(strategy=RoutingStrategy.ROUND_ROBIN)
        >>> # Requests will be distributed evenly across matching routes
    """

    FIRST_MATCH = "first_match"  # Use first matching route
    BEST_MATCH = "best_match"  # Use highest scoring route
    ROUND_ROBIN = "round_robin"  # Rotate among matching routes
    RANDOM = "random"  # Random selection among matches
    LOAD_BALANCED = "load_balanced"  # Based on current load
    COST_OPTIMIZED = "cost_optimized"  # Prefer cheaper models


class RouteStatus(Enum):
    """Status of a route indicating its availability for routing.

    Routes can be in different states that affect whether they are considered
    during route matching. Only ACTIVE routes participate in query matching.

    Attributes:
        ACTIVE: Route is available and will be considered for matching queries.
        DISABLED: Route has been manually disabled and will not match any queries.
        RATE_LIMITED: Route is temporarily unavailable due to rate limiting.
        ERROR: Route encountered an error and is temporarily unavailable.

    Example - Checking route status:
        >>> from insideLLMs.agents.routing import Route, RouteStatus
        >>> from insideLLMs import DummyModel
        >>> route = Route(name="test", model=DummyModel())
        >>> print(route.status)  # RouteStatus.ACTIVE

    Example - Disabling a route:
        >>> from insideLLMs.agents.routing import Route, RouteStatus
        >>> from insideLLMs import DummyModel
        >>> route = Route(name="test", model=DummyModel())
        >>> route.status = RouteStatus.DISABLED
        >>> # Route will no longer match any queries

    Example - Handling rate-limited routes:
        >>> from insideLLMs.agents.routing import Route, RouteStatus
        >>> from insideLLMs import DummyModel
        >>> route = Route(name="test", model=DummyModel())
        >>> if route.status == RouteStatus.RATE_LIMITED:
        ...     print("Route is rate limited, using fallback")

    Example - Checking for error state:
        >>> from insideLLMs.agents.routing import Route, RouteStatus
        >>> from insideLLMs import DummyModel
        >>> route = Route(name="test", model=DummyModel())
        >>> if route.status == RouteStatus.ERROR:
        ...     print(f"Route {route.name} has errors")
    """

    ACTIVE = "active"
    DISABLED = "disabled"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


@dataclass
class RouterConfig:
    """Configuration settings for the SemanticRouter.

    This dataclass holds all configuration options that control how the router
    matches queries to routes, handles failures, and optimizes performance.

    Attributes:
        strategy: The routing strategy to use when multiple routes match.
            Defaults to BEST_MATCH for highest accuracy.
        similarity_threshold: Minimum similarity score (0.0-1.0) required for
            a route to be considered a match. Lower values are more permissive.
        case_sensitive: Whether pattern matching should be case-sensitive.
            Defaults to False for more flexible matching.
        use_fallback: Whether to use the fallback model when no routes match.
            Defaults to True for robustness.
        fallback_threshold: Minimum score below which fallback is triggered.
            Only used when use_fallback is True.
        max_retries: Maximum number of retry attempts on failure.
            Defaults to 2 for resilience without excessive delays.
        retry_delay_ms: Delay in milliseconds between retry attempts.
            Defaults to 100ms for quick recovery.
        cache_routes: Whether to cache route decisions for repeated queries.
            Defaults to True for improved performance.
        cache_ttl_seconds: Time-to-live for cached route decisions in seconds.
            Defaults to 300 (5 minutes).
        max_concurrent_per_route: Maximum concurrent requests per route.
            Used for load balancing decisions.

    Example - Default configuration:
        >>> from insideLLMs.agents.routing import RouterConfig
        >>> config = RouterConfig()
        >>> print(config.strategy)  # RoutingStrategy.BEST_MATCH
        >>> print(config.similarity_threshold)  # 0.3

    Example - Custom configuration for high-throughput:
        >>> from insideLLMs.agents.routing import RouterConfig, RoutingStrategy
        >>> config = RouterConfig(
        ...     strategy=RoutingStrategy.LOAD_BALANCED,
        ...     max_retries=3,
        ...     cache_routes=True,
        ...     cache_ttl_seconds=600,
        ...     max_concurrent_per_route=200,
        ... )

    Example - Strict matching configuration:
        >>> from insideLLMs.agents.routing import RouterConfig, RoutingStrategy
        >>> config = RouterConfig(
        ...     strategy=RoutingStrategy.BEST_MATCH,
        ...     similarity_threshold=0.7,  # Require high similarity
        ...     case_sensitive=True,  # Exact case matching
        ...     use_fallback=False,  # No fallback, fail if no match
        ... )

    Example - Converting to dictionary for serialization:
        >>> from insideLLMs.agents.routing import RouterConfig
        >>> config = RouterConfig()
        >>> config_dict = config.to_dict()
        >>> print(config_dict["strategy"])  # "best_match"
    """

    # Matching settings
    strategy: RoutingStrategy = RoutingStrategy.BEST_MATCH
    similarity_threshold: float = 0.3
    case_sensitive: bool = False

    # Fallback settings
    use_fallback: bool = True
    fallback_threshold: float = 0.1

    # Retry settings
    max_retries: int = 2
    retry_delay_ms: int = 100

    # Caching
    cache_routes: bool = True
    cache_ttl_seconds: int = 300

    # Load balancing
    max_concurrent_per_route: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary for serialization.

        Creates a dictionary representation of the configuration that can be
        used for logging, serialization, or reconstructing the configuration.

        Returns:
            dict[str, Any]: Dictionary containing configuration values with
                string keys. The strategy is converted to its string value.

        Example:
            >>> from insideLLMs.agents.routing import RouterConfig, RoutingStrategy
            >>> config = RouterConfig(strategy=RoutingStrategy.COST_OPTIMIZED)
            >>> d = config.to_dict()
            >>> print(d["strategy"])  # "cost_optimized"
            >>> print(d["similarity_threshold"])  # 0.3
        """
        return {
            "strategy": self.strategy.value,
            "similarity_threshold": self.similarity_threshold,
            "case_sensitive": self.case_sensitive,
            "use_fallback": self.use_fallback,
            "max_retries": self.max_retries,
        }


@dataclass
class RouteMatch:
    """Result of matching a query to a route.

    Contains information about how well a query matched a specific route,
    including the similarity score, which patterns matched, and the model
    that would handle the query if this route is selected.

    Attributes:
        route_name: The name of the matched route.
        score: Similarity score between 0.0 and 1.0 indicating match quality.
            Higher scores indicate better matches.
        matched_patterns: List of pattern strings that matched the query.
            May be empty if matching was purely semantic.
        model: The model instance associated with this route.
        metadata: Additional metadata about the match (e.g., priority).

    Example - Inspecting a route match:
        >>> from insideLLMs.agents.routing import SemanticRouter, Route
        >>> from insideLLMs import DummyModel
        >>> router = SemanticRouter()
        >>> router.add_route(Route(
        ...     name="code",
        ...     patterns=["python", "function"],
        ...     model=DummyModel(),
        ... ))
        >>> matches = router.match("Write a python function")
        >>> if matches:
        ...     print(f"Best match: {matches[0].route_name}")
        ...     print(f"Score: {matches[0].score:.2f}")
        ...     print(f"Patterns: {matches[0].matched_patterns}")

    Example - Comparing multiple matches:
        >>> from insideLLMs.agents.routing import SemanticRouter, Route
        >>> from insideLLMs import DummyModel
        >>> router = SemanticRouter()
        >>> router.add_route(Route(name="a", patterns=["hello"], model=DummyModel()))
        >>> router.add_route(Route(name="b", patterns=["hello", "world"], model=DummyModel()))
        >>> matches = router.match("hello world")
        >>> for m in matches:
        ...     print(f"{m.route_name}: score={m.score:.2f}")

    Example - Converting to dictionary for logging:
        >>> from insideLLMs.agents.routing import RouteMatch
        >>> from insideLLMs import DummyModel
        >>> match = RouteMatch(
        ...     route_name="test",
        ...     score=0.85,
        ...     matched_patterns=["code"],
        ...     model=DummyModel(),
        ... )
        >>> print(match.to_dict())

    Example - Accessing match metadata:
        >>> from insideLLMs.agents.routing import RouteMatch
        >>> from insideLLMs import DummyModel
        >>> match = RouteMatch(
        ...     route_name="premium",
        ...     score=0.9,
        ...     matched_patterns=["analyze"],
        ...     model=DummyModel(),
        ...     metadata={"priority": 10, "cost": 0.03},
        ... )
        >>> print(match.metadata["priority"])  # 10
    """

    route_name: str
    score: float
    matched_patterns: list[str]
    model: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the route match to a dictionary for serialization.

        Creates a JSON-serializable dictionary representation of the match
        result, useful for logging, debugging, or API responses.

        Returns:
            dict[str, Any]: Dictionary containing match information. The model
                is represented by its class name rather than the full object.

        Example:
            >>> from insideLLMs.agents.routing import RouteMatch
            >>> from insideLLMs import DummyModel
            >>> match = RouteMatch(
            ...     route_name="code",
            ...     score=0.75,
            ...     matched_patterns=["python", "function"],
            ...     model=DummyModel(name="CodeAssist"),
            ... )
            >>> d = match.to_dict()
            >>> print(d["route_name"])  # "code"
            >>> print(d["model"])  # "DummyModel"
        """
        return {
            "route_name": self.route_name,
            "score": self.score,
            "matched_patterns": self.matched_patterns,
            "model": str(type(self.model).__name__),
            "metadata": self.metadata,
        }


@dataclass
class RoutingResult:
    """Complete result of routing and executing a query.

    Contains all information about a routed query including the original query,
    which route handled it, the model's response, performance metrics, and
    any retry or fallback behavior that occurred.

    Attributes:
        query: The original query string that was routed.
        route_name: Name of the route that handled the query, or "none" if
            no route matched and no fallback was available.
        response: The response from the model, or None if routing failed.
        latency_ms: Total time in milliseconds from routing start to completion,
            including any retries.
        match_score: The similarity score of the selected route (0.0-1.0).
        retries: Number of retry attempts that were made (0 if first attempt
            succeeded).
        fallback_used: True if the fallback model was used because no routes
            matched the query.
        metadata: Additional metadata such as error information.

    Example - Basic routing result:
        >>> from insideLLMs.agents.routing import SemanticRouter, Route
        >>> from insideLLMs import DummyModel
        >>> router = SemanticRouter()
        >>> router.add_route(Route(
        ...     name="code",
        ...     patterns=["python"],
        ...     model=DummyModel(),
        ... ))
        >>> result = router.route("Write python code")
        >>> print(f"Route: {result.route_name}")
        >>> print(f"Latency: {result.latency_ms:.2f}ms")
        >>> print(f"Score: {result.match_score:.2f}")

    Example - Checking for fallback usage:
        >>> from insideLLMs.agents.routing import SemanticRouter, Route
        >>> from insideLLMs import DummyModel
        >>> router = SemanticRouter(fallback_model=DummyModel(name="Fallback"))
        >>> result = router.route("Unknown query type")
        >>> if result.fallback_used:
        ...     print("Fallback model was used")

    Example - Monitoring retries:
        >>> from insideLLMs.agents.routing import SemanticRouter, Route, RouterConfig
        >>> from insideLLMs import DummyModel
        >>> config = RouterConfig(max_retries=3)
        >>> router = SemanticRouter(config=config)
        >>> router.add_route(Route(name="test", patterns=["test"], model=DummyModel()))
        >>> result = router.route("test query")
        >>> if result.retries > 0:
        ...     print(f"Query required {result.retries} retries")

    Example - Error handling:
        >>> from insideLLMs.agents.routing import SemanticRouter
        >>> router = SemanticRouter()
        >>> result = router.route("No routes configured")
        >>> if result.route_name == "none":
        ...     print(f"Error: {result.metadata.get('error')}")
    """

    query: str
    route_name: str
    response: Any
    latency_ms: float
    match_score: float
    retries: int = 0
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the routing result to a dictionary for serialization.

        Creates a JSON-serializable dictionary representation of the routing
        result, suitable for logging, API responses, or analytics.

        Returns:
            dict[str, Any]: Dictionary containing all routing result fields.
                The response is truncated to 200 characters to prevent
                excessive log sizes.

        Example:
            >>> from insideLLMs.agents.routing import RoutingResult
            >>> result = RoutingResult(
            ...     query="Hello world",
            ...     route_name="greeting",
            ...     response="Hi there!",
            ...     latency_ms=45.2,
            ...     match_score=0.85,
            ... )
            >>> d = result.to_dict()
            >>> print(d["route_name"])  # "greeting"
            >>> print(d["latency_ms"])  # 45.2
        """
        return {
            "query": self.query,
            "route_name": self.route_name,
            "response": str(self.response)[:200] if self.response else None,
            "latency_ms": self.latency_ms,
            "match_score": self.match_score,
            "retries": self.retries,
            "fallback_used": self.fallback_used,
            "metadata": self.metadata,
        }


@dataclass
class RouteStats:
    """Statistics tracking for a route's performance and usage.

    Tracks comprehensive metrics for monitoring route health, performance,
    and utilization. Used by the router for load balancing decisions and
    by operators for observability.

    Attributes:
        route_name: The name of the route these statistics belong to.
        total_requests: Total number of requests routed to this route.
        successful_requests: Number of requests that completed successfully.
        failed_requests: Number of requests that failed (including retries).
        total_latency_ms: Cumulative latency of all requests in milliseconds.
        avg_latency_ms: Running average latency in milliseconds.
        current_load: Number of requests currently being processed by this route.
            Used for load balancing decisions.
        last_used: Timestamp of the last request to this route.
        error_rate: Proportion of failed requests (0.0-1.0).

    Example - Viewing route statistics:
        >>> from insideLLMs.agents.routing import SemanticRouter, Route
        >>> from insideLLMs import DummyModel
        >>> router = SemanticRouter()
        >>> router.add_route(Route(name="test", patterns=["test"], model=DummyModel()))
        >>> _ = router.route("test query")
        >>> stats = router.get_stats()
        >>> print(f"Total requests: {stats['test'].total_requests}")
        >>> print(f"Avg latency: {stats['test'].avg_latency_ms:.2f}ms")

    Example - Monitoring error rates:
        >>> from insideLLMs.agents.routing import Route, RouteStats
        >>> from insideLLMs import DummyModel
        >>> route = Route(name="api", patterns=["api"], model=DummyModel())
        >>> # After some requests...
        >>> if route.stats.error_rate > 0.1:
        ...     print(f"High error rate: {route.stats.error_rate:.1%}")

    Example - Load balancing metrics:
        >>> from insideLLMs.agents.routing import Route, RouteStats
        >>> from insideLLMs import DummyModel
        >>> route = Route(name="worker", patterns=["work"], model=DummyModel())
        >>> print(f"Current load: {route.stats.current_load}")
        >>> print(f"Last used: {route.stats.last_used}")

    Example - Converting to dictionary for reporting:
        >>> from insideLLMs.agents.routing import RouteStats
        >>> stats = RouteStats(route_name="test")
        >>> stats.update(success=True, latency_ms=50.0)
        >>> stats.update(success=True, latency_ms=60.0)
        >>> d = stats.to_dict()
        >>> print(f"Avg latency: {d['avg_latency_ms']:.1f}ms")
    """

    route_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    current_load: int = 0
    last_used: Optional[datetime] = None
    error_rate: float = 0.0

    def update(self, success: bool, latency_ms: float) -> None:
        """Update statistics after a request completes.

        Should be called after each request to this route completes,
        regardless of success or failure. Updates all derived metrics
        like average latency and error rate.

        Args:
            success: True if the request completed successfully, False if
                it failed (after all retries).
            latency_ms: Total latency of the request in milliseconds,
                including any retry time.

        Example - Updating after successful request:
            >>> from insideLLMs.agents.routing import RouteStats
            >>> stats = RouteStats(route_name="test")
            >>> stats.update(success=True, latency_ms=45.0)
            >>> print(stats.successful_requests)  # 1
            >>> print(stats.avg_latency_ms)  # 45.0

        Example - Updating after failed request:
            >>> from insideLLMs.agents.routing import RouteStats
            >>> stats = RouteStats(route_name="test")
            >>> stats.update(success=False, latency_ms=100.0)
            >>> print(stats.failed_requests)  # 1
            >>> print(stats.error_rate)  # 1.0

        Example - Tracking multiple requests:
            >>> from insideLLMs.agents.routing import RouteStats
            >>> stats = RouteStats(route_name="test")
            >>> stats.update(success=True, latency_ms=40.0)
            >>> stats.update(success=True, latency_ms=60.0)
            >>> stats.update(success=False, latency_ms=100.0)
            >>> print(f"Total: {stats.total_requests}")  # 3
            >>> print(f"Error rate: {stats.error_rate:.2f}")  # 0.33
        """
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.total_requests
        self.last_used = datetime.now()

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.error_rate = (
            self.failed_requests / self.total_requests if self.total_requests > 0 else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to a dictionary for serialization.

        Creates a JSON-serializable dictionary representation of the
        statistics, suitable for logging, monitoring dashboards, or APIs.

        Returns:
            dict[str, Any]: Dictionary containing all statistics fields.
                The last_used timestamp is converted to ISO format string.

        Example:
            >>> from insideLLMs.agents.routing import RouteStats
            >>> stats = RouteStats(route_name="api")
            >>> stats.update(success=True, latency_ms=50.0)
            >>> d = stats.to_dict()
            >>> print(d["route_name"])  # "api"
            >>> print(d["total_requests"])  # 1
            >>> print(type(d["last_used"]))  # <class 'str'>
        """
        return {
            "route_name": self.route_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "current_load": self.current_load,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "error_rate": self.error_rate,
        }


# =============================================================================
# Route Definition
# =============================================================================


class Route:
    """A route that maps queries to a specific model based on patterns and semantics.

    Routes are the fundamental building blocks of the semantic router. Each route
    defines a set of patterns (string or regex) that determine which queries it
    should handle, along with the model that will process matched queries.

    Routes support both simple string pattern matching and regex patterns for
    more complex matching needs. They also track their own statistics for
    monitoring and load balancing purposes.

    Args:
        name: Unique identifier for this route. Used in routing results and stats.
        model: The model instance to use for queries matching this route. Must
            have a generate(prompt, **kwargs) method.
        patterns: List of substring patterns to match against queries. A query
            matches if it contains any pattern as a substring.
        regex_patterns: List of regex pattern strings for advanced matching.
            Patterns are compiled with re.IGNORECASE flag.
        description: Human-readable description of what this route handles.
            Used for semantic matching when embeddings are available.
        priority: Route priority for tie-breaking (higher = preferred).
        cost_per_token: Estimated cost per token for this route's model.
            Used by COST_OPTIMIZED routing strategy.
        max_tokens: Maximum tokens allowed for this route (for validation).
        metadata: Additional custom metadata for the route.

    Attributes:
        status: Current status of the route (RouteStatus enum).
        stats: RouteStats instance tracking this route's performance.

    Example - Creating a simple route:
        >>> from insideLLMs.agents.routing import Route
        >>> from insideLLMs import DummyModel
        >>> route = Route(
        ...     name="greeting",
        ...     model=DummyModel(name="Greeter"),
        ...     patterns=["hello", "hi", "hey"],
        ...     description="Handles greeting queries",
        ... )
        >>> is_match, score, patterns = route.matches("hello there")
        >>> print(f"Matched: {is_match}, Score: {score:.2f}")

    Example - Route with regex patterns:
        >>> from insideLLMs.agents.routing import Route
        >>> from insideLLMs import DummyModel
        >>> route = Route(
        ...     name="email",
        ...     model=DummyModel(name="EmailParser"),
        ...     regex_patterns=[r"\\b[\\w.-]+@[\\w.-]+\\.\\w+\\b"],
        ...     description="Handles queries containing email addresses",
        ... )
        >>> is_match, score, patterns = route.matches("Contact me at user@example.com")
        >>> print(f"Matched: {is_match}")  # True

    Example - High-priority route with cost:
        >>> from insideLLMs.agents.routing import Route
        >>> from insideLLMs import DummyModel
        >>> premium_route = Route(
        ...     name="complex_analysis",
        ...     model=DummyModel(name="GPT-4"),
        ...     patterns=["analyze", "research", "investigate"],
        ...     priority=10,
        ...     cost_per_token=0.03,
        ...     description="Complex analysis requiring advanced reasoning",
        ... )

    Example - Converting route to dictionary:
        >>> from insideLLMs.agents.routing import Route
        >>> from insideLLMs import DummyModel
        >>> route = Route(
        ...     name="test",
        ...     model=DummyModel(),
        ...     patterns=["test"],
        ...     metadata={"version": "1.0"},
        ... )
        >>> d = route.to_dict()
        >>> print(d["name"])  # "test"
        >>> print(d["metadata"])  # {"version": "1.0"}
    """

    def __init__(
        self,
        name: str,
        model: Any,
        patterns: Optional[list[str]] = None,
        regex_patterns: Optional[list[str]] = None,
        description: str = "",
        priority: int = 0,
        cost_per_token: float = 0.0,
        max_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.model = model
        self.patterns = patterns or []
        self.regex_patterns = [re.compile(p, re.IGNORECASE) for p in (regex_patterns or [])]
        self.description = description
        self.priority = priority
        self.cost_per_token = cost_per_token
        self.max_tokens = max_tokens
        self.metadata = metadata or {}
        self.status = RouteStatus.ACTIVE
        self._stats = RouteStats(route_name=name)
        self._lock = threading.Lock()

    def matches(
        self,
        query: str,
        case_sensitive: bool = False,
    ) -> tuple[bool, float, list[str]]:
        """Check if a query matches this route's patterns.

        Evaluates the query against both string patterns and regex patterns.
        The match score is calculated based on how much of the query is covered
        by the matched patterns.

        Args:
            query: The query string to match against this route's patterns.
            case_sensitive: If True, perform case-sensitive matching for string
                patterns. Regex patterns always use IGNORECASE. Defaults to False.

        Returns:
            tuple[bool, float, list[str]]: A tuple containing:
                - bool: True if at least one pattern matched, False otherwise.
                - float: Match score between 0.0 and 1.0 based on pattern coverage.
                - list[str]: List of pattern strings that matched the query.

        Example - Basic pattern matching:
            >>> from insideLLMs.agents.routing import Route
            >>> from insideLLMs import DummyModel
            >>> route = Route(
            ...     name="code",
            ...     model=DummyModel(),
            ...     patterns=["python", "function", "code"],
            ... )
            >>> is_match, score, matched = route.matches("Write a python function")
            >>> print(f"Matched: {is_match}")  # True
            >>> print(f"Patterns: {matched}")  # ["python", "function"]

        Example - Case-sensitive matching:
            >>> from insideLLMs.agents.routing import Route
            >>> from insideLLMs import DummyModel
            >>> route = Route(name="test", model=DummyModel(), patterns=["Python"])
            >>> is_match, _, _ = route.matches("python code", case_sensitive=True)
            >>> print(is_match)  # False
            >>> is_match, _, _ = route.matches("Python code", case_sensitive=True)
            >>> print(is_match)  # True

        Example - Regex pattern matching:
            >>> from insideLLMs.agents.routing import Route
            >>> from insideLLMs import DummyModel
            >>> route = Route(
            ...     name="version",
            ...     model=DummyModel(),
            ...     regex_patterns=[r"v\\d+\\.\\d+"],
            ... )
            >>> is_match, score, matched = route.matches("Upgrade to v2.0")
            >>> print(f"Matched: {is_match}")  # True

        Example - No match scenario:
            >>> from insideLLMs.agents.routing import Route
            >>> from insideLLMs import DummyModel
            >>> route = Route(name="math", model=DummyModel(), patterns=["calculate", "solve"])
            >>> is_match, score, matched = route.matches("Tell me a story")
            >>> print(f"Matched: {is_match}, Score: {score}")  # False, 0.0
        """
        query_lower = query.lower() if not case_sensitive else query

        matched = []
        total_score = 0.0

        # Check string patterns
        for pattern in self.patterns:
            if not case_sensitive:
                pattern = pattern.lower()

            if pattern in query_lower:
                matched.append(pattern)
                # Score based on pattern length relative to query
                total_score += len(pattern) / max(len(query), 1)

        # Check regex patterns
        for regex in self.regex_patterns:
            match = regex.search(query)
            if match:
                matched.append(regex.pattern)
                total_score += 0.5  # Fixed score for regex matches

        # Normalize score
        score = min(total_score / len(matched) if matched else 0, 1.0) if matched else 0.0

        return len(matched) > 0, score, matched

    @property
    def stats(self) -> RouteStats:
        """Get the statistics for this route.

        Returns:
            RouteStats: Statistics object tracking requests, latency, and errors.

        Example:
            >>> from insideLLMs.agents.routing import Route
            >>> from insideLLMs import DummyModel
            >>> route = Route(name="test", model=DummyModel(), patterns=["test"])
            >>> print(route.stats.total_requests)  # 0
            >>> print(route.stats.route_name)  # "test"
        """
        return self._stats

    def to_dict(self) -> dict[str, Any]:
        """Convert the route to a dictionary for serialization.

        Creates a JSON-serializable dictionary representation of the route
        configuration, useful for saving, logging, or API responses.

        Returns:
            dict[str, Any]: Dictionary containing route configuration. The model
                is not included (only configuration is serialized).

        Example:
            >>> from insideLLMs.agents.routing import Route, RouteStatus
            >>> from insideLLMs import DummyModel
            >>> route = Route(
            ...     name="api",
            ...     model=DummyModel(),
            ...     patterns=["api", "endpoint"],
            ...     regex_patterns=[r"/api/v\\d+"],
            ...     description="API endpoint queries",
            ...     priority=5,
            ...     cost_per_token=0.01,
            ... )
            >>> d = route.to_dict()
            >>> print(d["name"])  # "api"
            >>> print(d["priority"])  # 5
            >>> print(d["status"])  # "active"
        """
        return {
            "name": self.name,
            "patterns": self.patterns,
            "regex_patterns": [p.pattern for p in self.regex_patterns],
            "description": self.description,
            "priority": self.priority,
            "cost_per_token": self.cost_per_token,
            "status": self.status.value,
            "metadata": self.metadata,
        }


# =============================================================================
# Semantic Matching
# =============================================================================


class SemanticMatcher:
    """Semantic similarity matcher for comparing queries to route descriptions.

    Provides semantic matching capabilities for the router by computing similarity
    scores between query text and route descriptions. Supports two modes:
    - Embedding-based matching using cosine similarity (when embedder provided)
    - Word-overlap-based matching using Jaccard similarity (fallback)

    The matcher can cache route embeddings for performance optimization when
    using embedding-based matching.

    Args:
        embedder: Optional callable that converts text to embedding vectors.
            Should accept a string and return a list of floats. When provided,
            enables high-quality semantic matching. When None, falls back to
            word-based similarity.

    Example - Using with embeddings:
        >>> from insideLLMs.agents.routing import SemanticMatcher
        >>> # Using a hypothetical embedding function
        >>> def embed(text):
        ...     # In practice, use sentence-transformers or similar
        ...     return [0.1] * 384  # Placeholder
        >>> matcher = SemanticMatcher(embedder=embed)
        >>> score = matcher.compute_similarity("Hello", "Hi there")
        >>> print(f"Similarity: {score:.2f}")

    Example - Using word-based fallback:
        >>> from insideLLMs.agents.routing import SemanticMatcher
        >>> matcher = SemanticMatcher()  # No embedder
        >>> score = matcher.compute_similarity(
        ...     "python programming language",
        ...     "python code development"
        ... )
        >>> print(f"Word overlap: {score:.2f}")

    Example - Caching route embeddings:
        >>> from insideLLMs.agents.routing import SemanticMatcher
        >>> def embed(text):
        ...     return [hash(word) % 100 / 100 for word in text.split()[:10]]
        >>> matcher = SemanticMatcher(embedder=embed)
        >>> matcher.cache_route_embedding("code", "programming and development")
        >>> cached = matcher.get_cached_embedding("code")
        >>> print(f"Cached: {cached is not None}")  # True

    Example - Computing cosine similarity directly:
        >>> from insideLLMs.agents.routing import SemanticMatcher
        >>> matcher = SemanticMatcher()
        >>> vec1 = [1.0, 0.0, 1.0]
        >>> vec2 = [1.0, 1.0, 0.0]
        >>> sim = matcher._cosine_similarity(vec1, vec2)
        >>> print(f"Cosine similarity: {sim:.2f}")  # 0.5
    """

    def __init__(
        self,
        embedder: Optional[Callable[[str], list[float]]] = None,
    ):
        self._embedder = embedder
        self._route_embeddings: dict[str, list[float]] = {}

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text strings.

        Uses embedding-based cosine similarity if an embedder was provided,
        otherwise falls back to word-overlap Jaccard similarity.

        Args:
            text1: First text string to compare.
            text2: Second text string to compare.

        Returns:
            float: Similarity score between 0.0 (no similarity) and 1.0
                (identical meaning/content).

        Example - Embedding-based similarity:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> def simple_embed(text):
            ...     return [len(text), sum(ord(c) for c in text) / 1000]
            >>> matcher = SemanticMatcher(embedder=simple_embed)
            >>> score = matcher.compute_similarity("hello world", "hello there")
            >>> print(f"Score: {score:.2f}")

        Example - Word-based similarity:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> matcher = SemanticMatcher()  # No embedder
            >>> score = matcher.compute_similarity(
            ...     "the quick brown fox",
            ...     "the lazy brown dog"
            ... )
            >>> print(f"Overlap score: {score:.2f}")

        Example - High similarity:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> matcher = SemanticMatcher()
            >>> score = matcher.compute_similarity("python code", "python code")
            >>> print(f"Identical: {score}")  # 1.0

        Example - Low similarity:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> matcher = SemanticMatcher()
            >>> score = matcher.compute_similarity("apple orange", "car truck")
            >>> print(f"No overlap: {score}")  # 0.0
        """
        if self._embedder:
            emb1 = self._embedder(text1)
            emb2 = self._embedder(text2)
            return self._cosine_similarity(emb1, emb2)
        else:
            return self._word_similarity(text1, text2)

    def _word_similarity(self, text1: str, text2: str) -> float:
        """Compute word-based Jaccard similarity between texts.

        Calculates the overlap between the word sets of two texts using
        the Jaccard index: |intersection| / |union|.

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            float: Jaccard similarity coefficient (0.0 to 1.0).

        Example:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> matcher = SemanticMatcher()
            >>> score = matcher._word_similarity("hello world", "hello there")
            >>> print(f"Jaccard: {score:.2f}")  # 0.33 (1 common / 3 total)
        """
        return word_overlap_similarity(text1, text2)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Calculates the cosine of the angle between two vectors, which measures
        their directional similarity independent of magnitude.

        Args:
            vec1: First embedding vector.
            vec2: Second embedding vector.

        Returns:
            float: Cosine similarity (-1.0 to 1.0, typically 0.0 to 1.0 for
                normalized embeddings). Returns 0.0 if vectors have different
                lengths or if either has zero magnitude.

        Example - Similar vectors:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> matcher = SemanticMatcher()
            >>> v1 = [1.0, 0.5, 0.3]
            >>> v2 = [0.9, 0.6, 0.2]
            >>> sim = matcher._cosine_similarity(v1, v2)
            >>> print(f"High similarity: {sim:.2f}")

        Example - Orthogonal vectors:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> matcher = SemanticMatcher()
            >>> v1 = [1.0, 0.0]
            >>> v2 = [0.0, 1.0]
            >>> sim = matcher._cosine_similarity(v1, v2)
            >>> print(f"Orthogonal: {sim}")  # 0.0

        Example - Different length vectors:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> matcher = SemanticMatcher()
            >>> sim = matcher._cosine_similarity([1, 2], [1, 2, 3])
            >>> print(f"Invalid: {sim}")  # 0.0

        Example - Zero vector handling:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> matcher = SemanticMatcher()
            >>> sim = matcher._cosine_similarity([0, 0, 0], [1, 2, 3])
            >>> print(f"Zero vector: {sim}")  # 0.0
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def cache_route_embedding(self, route_name: str, text: str) -> None:
        """Cache the embedding for a route's description.

        Pre-computes and stores the embedding for a route's description text,
        avoiding repeated embedding computation during query matching.

        Only caches if an embedder function was provided. Does nothing if
        using word-based similarity.

        Args:
            route_name: The name of the route to cache embedding for.
            text: The description text to embed and cache.

        Example:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> def embed(text):
            ...     return [float(ord(c)) for c in text[:5]]
            >>> matcher = SemanticMatcher(embedder=embed)
            >>> matcher.cache_route_embedding("code", "programming assistance")
            >>> embedding = matcher.get_cached_embedding("code")
            >>> print(f"Cached {len(embedding)} dimensions")

        Example - No caching without embedder:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> matcher = SemanticMatcher()  # No embedder
            >>> matcher.cache_route_embedding("test", "some text")
            >>> print(matcher.get_cached_embedding("test"))  # None
        """
        if self._embedder:
            self._route_embeddings[route_name] = self._embedder(text)

    def get_cached_embedding(self, route_name: str) -> Optional[list[float]]:
        """Retrieve a cached embedding for a route.

        Gets the pre-computed embedding for a route if it was previously
        cached using cache_route_embedding().

        Args:
            route_name: The name of the route to get the embedding for.

        Returns:
            Optional[list[float]]: The cached embedding vector, or None if
                no embedding was cached for this route.

        Example:
            >>> from insideLLMs.agents.routing import SemanticMatcher
            >>> def embed(text):
            ...     return [1.0, 2.0, 3.0]
            >>> matcher = SemanticMatcher(embedder=embed)
            >>> matcher.cache_route_embedding("route1", "description")
            >>> emb = matcher.get_cached_embedding("route1")
            >>> print(emb)  # [1.0, 2.0, 3.0]
            >>> print(matcher.get_cached_embedding("unknown"))  # None
        """
        return self._route_embeddings.get(route_name)


# =============================================================================
# Semantic Router
# =============================================================================


class SemanticRouter:
    """Main semantic router for intelligent model selection and query routing.

    The SemanticRouter is the central component for routing queries to appropriate
    models based on pattern matching, semantic similarity, and configurable
    routing strategies. It supports multiple selection strategies, automatic
    fallback, retry logic, and route caching.

    The router matches queries against registered routes using both pattern-based
    matching (string and regex patterns) and semantic similarity (when route
    descriptions are provided). The best matching route is selected based on
    the configured strategy.

    Args:
        config: RouterConfig instance controlling matching behavior, caching,
            and retry settings. Defaults to RouterConfig() with sensible defaults.
        fallback_model: Model to use when no routes match the query. If None
            and no routes match, routing returns an error result.
        embedder: Optional callable for computing text embeddings. When provided,
            enables semantic similarity matching based on route descriptions.

    Attributes:
        config: The router configuration.
        fallback_model: The fallback model (may be None).

    Example - Basic router setup:
        >>> from insideLLMs.agents.routing import SemanticRouter, Route
        >>> from insideLLMs import DummyModel
        >>> router = SemanticRouter()
        >>> router.add_route(Route(
        ...     name="code",
        ...     model=DummyModel(name="CodeAssist"),
        ...     patterns=["python", "javascript", "code"],
        ...     description="Programming and coding assistance",
        ... ))
        >>> router.add_route(Route(
        ...     name="math",
        ...     model=DummyModel(name="MathSolver"),
        ...     patterns=["calculate", "solve", "equation"],
        ...     description="Mathematical problem solving",
        ... ))
        >>> result = router.route("Write a python function to sort a list")
        >>> print(f"Routed to: {result.route_name}")  # "code"

    Example - Router with fallback:
        >>> from insideLLMs.agents.routing import SemanticRouter, Route
        >>> from insideLLMs import DummyModel
        >>> fallback = DummyModel(name="GeneralAssistant")
        >>> router = SemanticRouter(fallback_model=fallback)
        >>> router.add_route(Route(
        ...     name="weather",
        ...     model=DummyModel(name="WeatherBot"),
        ...     patterns=["weather", "temperature", "forecast"],
        ... ))
        >>> result = router.route("What is the meaning of life?")
        >>> print(f"Used fallback: {result.fallback_used}")  # True

    Example - Cost-optimized routing:
        >>> from insideLLMs.agents.routing import SemanticRouter, Route, RouterConfig, RoutingStrategy
        >>> from insideLLMs import DummyModel
        >>> config = RouterConfig(strategy=RoutingStrategy.COST_OPTIMIZED)
        >>> router = SemanticRouter(config=config)
        >>> router.add_route(Route(
        ...     name="premium",
        ...     model=DummyModel(name="GPT-4"),
        ...     patterns=["analyze", "research"],
        ...     cost_per_token=0.03,
        ... ))
        >>> router.add_route(Route(
        ...     name="budget",
        ...     model=DummyModel(name="GPT-3.5"),
        ...     patterns=["analyze", "quick"],
        ...     cost_per_token=0.002,
        ... ))
        >>> result = router.route("Analyze this data quickly")
        >>> # Selects "budget" route due to lower cost

    Example - Viewing routing statistics:
        >>> from insideLLMs.agents.routing import SemanticRouter, Route
        >>> from insideLLMs import DummyModel
        >>> router = SemanticRouter()
        >>> router.add_route(Route(name="test", model=DummyModel(), patterns=["test"]))
        >>> _ = router.route("test query")
        >>> _ = router.route("another test")
        >>> stats = router.get_stats()
        >>> print(f"Total requests: {stats['test'].total_requests}")  # 2
    """

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        fallback_model: Optional[Any] = None,
        embedder: Optional[Callable[[str], list[float]]] = None,
    ):
        self.config = config or RouterConfig()
        self.fallback_model = fallback_model
        self._routes: dict[str, Route] = {}
        self._matcher = SemanticMatcher(embedder)
        self._route_cache: dict[str, tuple[str, datetime]] = {}
        self._lock = threading.RLock()
        self._round_robin_index = 0

    def add_route(self, route: Route) -> None:
        """Add a route to the router.

        Registers a route for query matching. If the route has a description,
        its embedding is cached for semantic matching.

        Args:
            route: The Route instance to add. If a route with the same name
                already exists, it will be replaced.

        Example - Adding a simple route:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(
            ...     name="greeting",
            ...     model=DummyModel(),
            ...     patterns=["hello", "hi"],
            ... ))
            >>> print(len(router.list_routes()))  # 1

        Example - Adding route with semantic description:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(
            ...     name="support",
            ...     model=DummyModel(),
            ...     patterns=["help", "issue"],
            ...     description="Customer support and troubleshooting",
            ... ))

        Example - Replacing an existing route:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(name="v1", model=DummyModel(), patterns=["old"]))
            >>> router.add_route(Route(name="v1", model=DummyModel(), patterns=["new"]))
            >>> route = router.get_route("v1")
            >>> print(route.patterns)  # ["new"]
        """
        with self._lock:
            self._routes[route.name] = route

            # Cache embedding for semantic matching
            if route.description:
                self._matcher.cache_route_embedding(route.name, route.description)

    def remove_route(self, name: str) -> bool:
        """Remove a route from the router by name.

        Args:
            name: The name of the route to remove.

        Returns:
            bool: True if the route was found and removed, False if no route
                with that name existed.

        Example - Removing an existing route:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(name="temp", model=DummyModel(), patterns=["test"]))
            >>> removed = router.remove_route("temp")
            >>> print(removed)  # True
            >>> print(router.get_route("temp"))  # None

        Example - Attempting to remove non-existent route:
            >>> from insideLLMs.agents.routing import SemanticRouter
            >>> router = SemanticRouter()
            >>> removed = router.remove_route("nonexistent")
            >>> print(removed)  # False

        Example - Dynamic route management:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(name="a", model=DummyModel(), patterns=["a"]))
            >>> router.add_route(Route(name="b", model=DummyModel(), patterns=["b"]))
            >>> print(len(router.list_routes()))  # 2
            >>> router.remove_route("a")
            >>> print(len(router.list_routes()))  # 1
        """
        with self._lock:
            if name in self._routes:
                del self._routes[name]
                return True
            return False

    def get_route(self, name: str) -> Optional[Route]:
        """Get a route by its name.

        Args:
            name: The name of the route to retrieve.

        Returns:
            Optional[Route]: The Route instance if found, None otherwise.

        Example:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(name="code", model=DummyModel(), patterns=["code"]))
            >>> route = router.get_route("code")
            >>> print(route.name)  # "code"
            >>> print(router.get_route("unknown"))  # None
        """
        return self._routes.get(name)

    def list_routes(self) -> list[Route]:
        """List all registered routes.

        Returns:
            list[Route]: List of all Route instances registered with the router.

        Example:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(name="a", model=DummyModel(), patterns=["a"]))
            >>> router.add_route(Route(name="b", model=DummyModel(), patterns=["b"]))
            >>> routes = router.list_routes()
            >>> print([r.name for r in routes])  # ["a", "b"]
        """
        return list(self._routes.values())

    def match(self, query: str) -> list[RouteMatch]:
        """Match a query against all active routes.

        Evaluates the query against all registered active routes using both
        pattern matching and semantic similarity. Returns matches sorted by
        score (highest first).

        Args:
            query: The query string to match against routes.

        Returns:
            list[RouteMatch]: List of RouteMatch objects for all matching routes,
                sorted by score in descending order. Empty if no routes match.

        Example - Basic matching:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(name="code", model=DummyModel(), patterns=["python"]))
            >>> router.add_route(Route(name="data", model=DummyModel(), patterns=["data"]))
            >>> matches = router.match("python programming")
            >>> print(len(matches))  # 1
            >>> print(matches[0].route_name)  # "code"

        Example - Multiple matches:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(name="a", model=DummyModel(), patterns=["test"]))
            >>> router.add_route(Route(name="b", model=DummyModel(), patterns=["test", "more"]))
            >>> matches = router.match("test more content")
            >>> for m in matches:
            ...     print(f"{m.route_name}: {m.score:.2f}")

        Example - No matches:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(name="code", model=DummyModel(), patterns=["python"]))
            >>> matches = router.match("cooking recipes")
            >>> print(len(matches))  # 0

        Example - Semantic matching with descriptions:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(
            ...     name="support",
            ...     model=DummyModel(),
            ...     patterns=[],
            ...     description="help troubleshoot issues problems",
            ... ))
            >>> matches = router.match("I need help with a problem")
            >>> # May match based on semantic similarity to description
        """
        matches = []

        for route in self._routes.values():
            if route.status != RouteStatus.ACTIVE:
                continue

            # Pattern matching
            is_match, score, matched_patterns = route.matches(query, self.config.case_sensitive)

            # Add semantic similarity
            if route.description:
                semantic_score = self._matcher.compute_similarity(query, route.description)
                score = max(score, semantic_score)

            if is_match or score >= self.config.similarity_threshold:
                matches.append(
                    RouteMatch(
                        route_name=route.name,
                        score=score,
                        matched_patterns=matched_patterns,
                        model=route.model,
                        metadata={"priority": route.priority},
                    )
                )

        # Sort by score (and priority for ties)
        matches.sort(key=lambda m: (m.score, m.metadata.get("priority", 0)), reverse=True)

        return matches

    def select_route(self, matches: list[RouteMatch]) -> Optional[RouteMatch]:
        """Select a route from matches based on the configured strategy.

        Applies the routing strategy to choose one route from a list of
        matching routes. Different strategies optimize for different goals.

        Args:
            matches: List of RouteMatch objects to select from.

        Returns:
            Optional[RouteMatch]: The selected RouteMatch, or None if matches
                is empty.

        Example - Best match selection:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route, RouteMatch
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()  # Default: BEST_MATCH strategy
            >>> matches = [
            ...     RouteMatch("low", 0.5, [], DummyModel(), {}),
            ...     RouteMatch("high", 0.9, [], DummyModel(), {}),
            ... ]
            >>> selected = router.select_route(matches)
            >>> print(selected.route_name)  # "high" (highest score)

        Example - Round robin selection:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route, RouterConfig, RoutingStrategy, RouteMatch
            >>> from insideLLMs import DummyModel
            >>> config = RouterConfig(strategy=RoutingStrategy.ROUND_ROBIN)
            >>> router = SemanticRouter(config=config)
            >>> matches = [
            ...     RouteMatch("a", 0.5, [], DummyModel(), {}),
            ...     RouteMatch("b", 0.5, [], DummyModel(), {}),
            ... ]
            >>> # First call returns "a", second returns "b", third returns "a", etc.

        Example - Empty matches:
            >>> from insideLLMs.agents.routing import SemanticRouter
            >>> router = SemanticRouter()
            >>> selected = router.select_route([])
            >>> print(selected)  # None

        Example - Load balanced selection:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route, RouterConfig, RoutingStrategy
            >>> from insideLLMs import DummyModel
            >>> config = RouterConfig(strategy=RoutingStrategy.LOAD_BALANCED)
            >>> router = SemanticRouter(config=config)
            >>> router.add_route(Route(name="a", model=DummyModel(), patterns=["test"]))
            >>> router.add_route(Route(name="b", model=DummyModel(), patterns=["test"]))
            >>> # Selects route with lowest current_load
        """
        if not matches:
            return None

        strategy = self.config.strategy

        if strategy == RoutingStrategy.FIRST_MATCH:
            return matches[0]

        elif strategy == RoutingStrategy.BEST_MATCH:
            return matches[0]  # Already sorted by score

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            with self._lock:
                selected = matches[self._round_robin_index % len(matches)]
                self._round_robin_index += 1
                return selected

        elif strategy == RoutingStrategy.RANDOM:
            return random.choice(matches)

        elif strategy == RoutingStrategy.LOAD_BALANCED:
            # Select route with lowest current load
            best = min(
                matches,
                key=lambda m: self._routes[m.route_name].stats.current_load,
            )
            return best

        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            # Select cheapest route
            best = min(
                matches,
                key=lambda m: self._routes[m.route_name].cost_per_token,
            )
            return best

        return matches[0]

    def route(
        self,
        query: str,
        **kwargs: Any,
    ) -> RoutingResult:
        """Route a query to the best matching model and get the response.

        The main routing method that:
        1. Checks the route cache for previously routed identical queries
        2. Matches the query against all active routes
        3. Selects the best route based on the configured strategy
        4. Falls back to the fallback model if no routes match
        5. Executes the query with retry logic on failure
        6. Updates route statistics and caches the route decision

        Args:
            query: The query string to route and process.
            **kwargs: Additional keyword arguments passed to the model's
                generate() method.

        Returns:
            RoutingResult: Complete result including the response, latency,
                match score, retry count, and whether fallback was used.

        Example - Basic routing:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(
            ...     name="code",
            ...     model=DummyModel(),
            ...     patterns=["python", "code"],
            ... ))
            >>> result = router.route("Write python code")
            >>> print(f"Route: {result.route_name}")
            >>> print(f"Response: {result.response}")
            >>> print(f"Latency: {result.latency_ms:.2f}ms")

        Example - Routing with model kwargs:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(name="test", model=DummyModel(), patterns=["test"]))
            >>> result = router.route("test query", temperature=0.7, max_tokens=100)

        Example - Fallback behavior:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter(fallback_model=DummyModel(name="Fallback"))
            >>> router.add_route(Route(name="specific", model=DummyModel(), patterns=["xyz"]))
            >>> result = router.route("unrelated query")
            >>> print(result.fallback_used)  # True
            >>> print(result.route_name)  # "fallback"

        Example - No match and no fallback:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route, RouterConfig
            >>> from insideLLMs import DummyModel
            >>> config = RouterConfig(use_fallback=False)
            >>> router = SemanticRouter(config=config)
            >>> router.add_route(Route(name="specific", model=DummyModel(), patterns=["xyz"]))
            >>> result = router.route("unrelated query")
            >>> print(result.route_name)  # "none"
            >>> print(result.response)  # None
        """
        start_time = time.time()

        # Check cache
        if self.config.cache_routes:
            cached = self._get_cached_route(query)
            if cached:
                route = self._routes.get(cached)
                if route:
                    matches = [
                        RouteMatch(
                            route_name=route.name,
                            score=1.0,
                            matched_patterns=[],
                            model=route.model,
                        )
                    ]
                else:
                    matches = self.match(query)
            else:
                matches = self.match(query)
        else:
            matches = self.match(query)

        selected = self.select_route(matches)

        # Use fallback if no match
        fallback_used = False
        if selected is None and self.config.use_fallback and self.fallback_model:
            selected = RouteMatch(
                route_name="fallback",
                score=0.0,
                matched_patterns=[],
                model=self.fallback_model,
            )
            fallback_used = True

        if selected is None:
            return RoutingResult(
                query=query,
                route_name="none",
                response=None,
                latency_ms=(time.time() - start_time) * 1000,
                match_score=0.0,
                metadata={"error": "No matching route found"},
            )

        # Execute with retries
        route = self._routes.get(selected.route_name)
        response = None
        retries = 0

        for attempt in range(self.config.max_retries + 1):
            try:
                if route:
                    with route._lock:
                        route._stats.current_load += 1

                response = selected.model.generate(query, **kwargs)

                if route:
                    with route._lock:
                        route._stats.current_load -= 1
                        route._stats.update(True, (time.time() - start_time) * 1000)

                break

            except Exception as e:
                retries = attempt + 1
                if route:
                    with route._lock:
                        route._stats.current_load -= 1
                        route._stats.update(False, (time.time() - start_time) * 1000)

                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_ms / 1000)
                else:
                    response = f"Error: {str(e)}"

        # Cache successful route
        if self.config.cache_routes and not fallback_used:
            self._cache_route(query, selected.route_name)

        return RoutingResult(
            query=query,
            route_name=selected.route_name,
            response=response,
            latency_ms=(time.time() - start_time) * 1000,
            match_score=selected.score,
            retries=retries,
            fallback_used=fallback_used,
        )

    def _get_cached_route(self, query: str) -> Optional[str]:
        """Get the cached route name for a query if it exists and is not expired.

        Args:
            query: The query string to look up in the cache.

        Returns:
            Optional[str]: The cached route name, or None if not cached or expired.

        Example:
            >>> from insideLLMs.agents.routing import SemanticRouter
            >>> router = SemanticRouter()
            >>> # Internal method - used automatically by route()
            >>> cached = router._get_cached_route("some query")
            >>> print(cached)  # None (nothing cached yet)
        """
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self._route_cache:
            route_name, cached_at = self._route_cache[key]
            if datetime.now() - cached_at < timedelta(seconds=self.config.cache_ttl_seconds):
                return route_name
            else:
                del self._route_cache[key]
        return None

    def _cache_route(self, query: str, route_name: str) -> None:
        """Cache a route decision for a query.

        Args:
            query: The query string to cache.
            route_name: The route name to associate with the query.

        Example:
            >>> from insideLLMs.agents.routing import SemanticRouter
            >>> router = SemanticRouter()
            >>> # Internal method - used automatically by route()
            >>> router._cache_route("my query", "code_route")
            >>> cached = router._get_cached_route("my query")
            >>> print(cached)  # "code_route"
        """
        key = hashlib.md5(query.encode()).hexdigest()
        self._route_cache[key] = (route_name, datetime.now())

    def get_stats(self) -> dict[str, RouteStats]:
        """Get statistics for all registered routes.

        Returns:
            dict[str, RouteStats]: Dictionary mapping route names to their
                RouteStats objects containing request counts, latencies, and
                error rates.

        Example:
            >>> from insideLLMs.agents.routing import SemanticRouter, Route
            >>> from insideLLMs import DummyModel
            >>> router = SemanticRouter()
            >>> router.add_route(Route(name="a", model=DummyModel(), patterns=["a"]))
            >>> router.add_route(Route(name="b", model=DummyModel(), patterns=["b"]))
            >>> _ = router.route("a query")
            >>> _ = router.route("a again")
            >>> _ = router.route("b query")
            >>> stats = router.get_stats()
            >>> print(stats["a"].total_requests)  # 2
            >>> print(stats["b"].total_requests)  # 1
        """
        return {name: route.stats for name, route in self._routes.items()}


# =============================================================================
# Model Pool
# =============================================================================


class ModelPool:
    """Pool of models for load balancing and redundancy.

    Manages multiple model instances and distributes requests across them
    using configurable strategies. Useful for:
    - Horizontal scaling across multiple model instances
    - Load balancing to prevent overloading individual models
    - Redundancy in case one model instance fails

    The pool tracks statistics for each model instance, enabling informed
    load balancing decisions and performance monitoring.

    Args:
        models: List of model instances to include in the pool. All models
            should have a compatible generate(prompt, **kwargs) interface.
        strategy: Selection strategy for choosing which model handles each
            request. Defaults to ROUND_ROBIN for even distribution.

    Attributes:
        models: The list of model instances in the pool.
        strategy: The current selection strategy.

    Example - Basic round-robin pool:
        >>> from insideLLMs.agents.routing import ModelPool
        >>> from insideLLMs import DummyModel
        >>> models = [DummyModel(name=f"Worker_{i}") for i in range(3)]
        >>> pool = ModelPool(models)
        >>> response1 = pool.generate("First request")  # Uses Worker_0
        >>> response2 = pool.generate("Second request")  # Uses Worker_1
        >>> response3 = pool.generate("Third request")  # Uses Worker_2
        >>> response4 = pool.generate("Fourth request")  # Uses Worker_0 again

    Example - Load-balanced pool:
        >>> from insideLLMs.agents.routing import ModelPool, RoutingStrategy
        >>> from insideLLMs import DummyModel
        >>> models = [DummyModel(name=f"Server_{i}") for i in range(3)]
        >>> pool = ModelPool(models, strategy=RoutingStrategy.LOAD_BALANCED)
        >>> # Automatically selects the model with lowest current load
        >>> response = pool.generate("Query")

    Example - Monitoring pool statistics:
        >>> from insideLLMs.agents.routing import ModelPool
        >>> from insideLLMs import DummyModel
        >>> pool = ModelPool([DummyModel(), DummyModel()])
        >>> _ = pool.generate("Test 1")
        >>> _ = pool.generate("Test 2")
        >>> _ = pool.generate("Test 3")
        >>> stats = pool.stats()
        >>> for idx, stat in stats.items():
        ...     print(f"Model {idx}: {stat.total_requests} requests")

    Example - Using pool as a route model:
        >>> from insideLLMs.agents.routing import ModelPool, SemanticRouter, Route
        >>> from insideLLMs import DummyModel
        >>> # Create a pool of code models
        >>> code_pool = ModelPool([
        ...     DummyModel(name="CodeLLM_1"),
        ...     DummyModel(name="CodeLLM_2"),
        ... ])
        >>> # Use the pool as the model for a route
        >>> router = SemanticRouter()
        >>> router.add_route(Route(
        ...     name="code",
        ...     model=code_pool,  # Pool handles load balancing
        ...     patterns=["code", "function"],
        ... ))
    """

    def __init__(
        self,
        models: list[Any],
        strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN,
    ):
        self.models = models
        self.strategy = strategy
        self._index = 0
        self._stats: dict[int, RouteStats] = {
            i: RouteStats(route_name=f"model_{i}") for i in range(len(models))
        }
        self._lock = threading.Lock()

    def select(self) -> tuple[Any, int]:
        """Select a model from the pool based on the configured strategy.

        Thread-safe method that chooses the next model to handle a request
        according to the pool's selection strategy.

        Returns:
            tuple[Any, int]: A tuple containing:
                - The selected model instance
                - The index of the model in the pool (0-based)

        Example - Round-robin selection:
            >>> from insideLLMs.agents.routing import ModelPool
            >>> from insideLLMs import DummyModel
            >>> pool = ModelPool([DummyModel(), DummyModel()])
            >>> model1, idx1 = pool.select()
            >>> print(f"Selected model at index {idx1}")  # 0
            >>> model2, idx2 = pool.select()
            >>> print(f"Selected model at index {idx2}")  # 1
            >>> model3, idx3 = pool.select()
            >>> print(f"Selected model at index {idx3}")  # 0

        Example - Random selection:
            >>> from insideLLMs.agents.routing import ModelPool, RoutingStrategy
            >>> from insideLLMs import DummyModel
            >>> pool = ModelPool(
            ...     [DummyModel(), DummyModel(), DummyModel()],
            ...     strategy=RoutingStrategy.RANDOM,
            ... )
            >>> model, idx = pool.select()
            >>> print(f"Randomly selected index: {idx}")  # 0, 1, or 2

        Example - Load-balanced selection:
            >>> from insideLLMs.agents.routing import ModelPool, RoutingStrategy
            >>> from insideLLMs import DummyModel
            >>> pool = ModelPool(
            ...     [DummyModel(), DummyModel()],
            ...     strategy=RoutingStrategy.LOAD_BALANCED,
            ... )
            >>> # Selects model with lowest current_load
            >>> model, idx = pool.select()
        """
        with self._lock:
            if self.strategy == RoutingStrategy.ROUND_ROBIN:
                idx = self._index % len(self.models)
                self._index += 1

            elif self.strategy == RoutingStrategy.RANDOM:
                idx = random.randint(0, len(self.models) - 1)

            elif self.strategy == RoutingStrategy.LOAD_BALANCED:
                idx = min(
                    range(len(self.models)),
                    key=lambda i: self._stats[i].current_load,
                )

            else:
                idx = 0

            return self.models[idx], idx

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate a response using a model selected from the pool.

        Selects a model based on the configured strategy, executes the prompt,
        and updates statistics for the selected model. Thread-safe and handles
        load tracking automatically.

        Args:
            prompt: The prompt text to send to the model.
            **kwargs: Additional keyword arguments passed to the model's
                generate() method (e.g., temperature, max_tokens).

        Returns:
            Any: The response from the selected model.

        Raises:
            Exception: Re-raises any exception from the underlying model after
                updating statistics.

        Example - Basic generation:
            >>> from insideLLMs.agents.routing import ModelPool
            >>> from insideLLMs import DummyModel
            >>> pool = ModelPool([DummyModel(), DummyModel()])
            >>> response = pool.generate("Hello, world!")
            >>> print(response)

        Example - Generation with parameters:
            >>> from insideLLMs.agents.routing import ModelPool
            >>> from insideLLMs import DummyModel
            >>> pool = ModelPool([DummyModel()])
            >>> response = pool.generate(
            ...     "Write a story",
            ...     temperature=0.8,
            ...     max_tokens=500,
            ... )

        Example - Parallel usage:
            >>> from insideLLMs.agents.routing import ModelPool
            >>> from insideLLMs import DummyModel
            >>> import concurrent.futures
            >>> pool = ModelPool([DummyModel() for _ in range(4)])
            >>> prompts = ["Query 1", "Query 2", "Query 3", "Query 4"]
            >>> with concurrent.futures.ThreadPoolExecutor() as executor:
            ...     results = list(executor.map(pool.generate, prompts))

        Example - Checking statistics after generation:
            >>> from insideLLMs.agents.routing import ModelPool
            >>> from insideLLMs import DummyModel
            >>> pool = ModelPool([DummyModel()])
            >>> _ = pool.generate("Test prompt")
            >>> stats = pool.stats()
            >>> print(f"Requests: {stats[0].total_requests}")  # 1
        """
        model, idx = self.select()

        with self._lock:
            self._stats[idx].current_load += 1

        start_time = time.time()

        try:
            response = model.generate(prompt, **kwargs)
            with self._lock:
                self._stats[idx].current_load -= 1
                self._stats[idx].update(True, (time.time() - start_time) * 1000)
            return response

        except Exception:
            with self._lock:
                self._stats[idx].current_load -= 1
                self._stats[idx].update(False, (time.time() - start_time) * 1000)
            raise

    def stats(self) -> dict[int, RouteStats]:
        """Get statistics for all models in the pool.

        Returns a copy of the statistics dictionary to prevent external
        modification of internal state.

        Returns:
            dict[int, RouteStats]: Dictionary mapping model indices (0-based)
                to their RouteStats objects containing request counts,
                latencies, and error rates.

        Example:
            >>> from insideLLMs.agents.routing import ModelPool
            >>> from insideLLMs import DummyModel
            >>> pool = ModelPool([DummyModel(), DummyModel()])
            >>> _ = pool.generate("Query 1")
            >>> _ = pool.generate("Query 2")
            >>> _ = pool.generate("Query 3")
            >>> stats = pool.stats()
            >>> for idx, stat in stats.items():
            ...     print(f"Model {idx}:")
            ...     print(f"  Total requests: {stat.total_requests}")
            ...     print(f"  Avg latency: {stat.avg_latency_ms:.2f}ms")
            ...     print(f"  Error rate: {stat.error_rate:.1%}")
        """
        return self._stats.copy()


# =============================================================================
# Intent Classifier
# =============================================================================


class IntentClassifier:
    """Classify query intent for intelligent routing decisions.

    The IntentClassifier determines the intent or purpose of a query, which
    can then be used to route the query to an appropriate model. Supports
    two classification methods:
    - Pattern-based: Fast matching using keyword patterns (no model required)
    - Model-based: Uses an LLM to classify queries into defined intents

    Pattern matching is attempted first for speed. If no strong pattern match
    is found and a model is configured, model-based classification is used.

    Args:
        model: Optional model instance for LLM-based classification. Must
            have a generate(prompt) method. If None, only pattern-based
            classification is available.
        intents: Optional dictionary mapping intent names to descriptions.
            Descriptions are used in the model prompt for classification.

    Attributes:
        model: The classification model (may be None).
        intents: Dictionary of intent names to descriptions.

    Example - Pattern-based classification:
        >>> from insideLLMs.agents.routing import IntentClassifier
        >>> classifier = IntentClassifier()
        >>> classifier.add_intent(
        ...     name="code_help",
        ...     description="Programming assistance",
        ...     patterns=["code", "function", "bug", "error"],
        ... )
        >>> classifier.add_intent(
        ...     name="general_qa",
        ...     description="General questions",
        ...     patterns=["what", "why", "how", "explain"],
        ... )
        >>> intent, confidence = classifier.classify("Fix this bug in my code")
        >>> print(f"Intent: {intent}, Confidence: {confidence:.2f}")

    Example - Model-based classification:
        >>> from insideLLMs.agents.routing import IntentClassifier
        >>> from insideLLMs import DummyModel
        >>> classifier = IntentClassifier(model=DummyModel())
        >>> classifier.add_intent("booking", "Travel and hotel reservations")
        >>> classifier.add_intent("support", "Technical support requests")
        >>> intent, confidence = classifier.classify("I need help with my account")

    Example - Using with router:
        >>> from insideLLMs.agents.routing import IntentClassifier, SemanticRouter, Route
        >>> from insideLLMs import DummyModel
        >>> # Create classifier
        >>> classifier = IntentClassifier()
        >>> classifier.add_intent("code", "Programming", ["code", "python"])
        >>> classifier.add_intent("math", "Mathematics", ["calculate", "solve"])
        >>> # Classify and route manually
        >>> intent, _ = classifier.classify("Write python code")
        >>> # Could use intent to select appropriate route

    Example - Combining patterns and model:
        >>> from insideLLMs.agents.routing import IntentClassifier
        >>> from insideLLMs import DummyModel
        >>> # Model is used as fallback when patterns don't match well
        >>> classifier = IntentClassifier(model=DummyModel())
        >>> classifier.add_intent(
        ...     name="urgent",
        ...     description="Urgent requests needing immediate attention",
        ...     patterns=["urgent", "emergency", "asap"],
        ... )
        >>> # Clear pattern match - uses patterns
        >>> intent1, conf1 = classifier.classify("This is urgent!")
        >>> # Unclear - falls back to model
        >>> intent2, conf2 = classifier.classify("Need help soon")
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        intents: Optional[dict[str, str]] = None,
    ):
        self.model = model
        self.intents = intents or {}
        self._intent_patterns: dict[str, list[str]] = {}

    def add_intent(
        self,
        name: str,
        description: str,
        patterns: Optional[list[str]] = None,
    ) -> None:
        """Add an intent to the classifier.

        Registers an intent with its description and optional keyword patterns.
        Patterns enable fast keyword-based classification without model calls.

        Args:
            name: Unique identifier for the intent (e.g., "code_help", "booking").
            description: Human-readable description of what this intent covers.
                Used in the model prompt for LLM-based classification.
            patterns: Optional list of keywords that indicate this intent.
                If any pattern appears in the query, it contributes to the
                match score for this intent.

        Example - Adding intent with patterns:
            >>> from insideLLMs.agents.routing import IntentClassifier
            >>> classifier = IntentClassifier()
            >>> classifier.add_intent(
            ...     name="weather",
            ...     description="Weather forecasts and conditions",
            ...     patterns=["weather", "temperature", "rain", "sunny"],
            ... )
            >>> intent, _ = classifier.classify("What's the weather like?")
            >>> print(intent)  # "weather"

        Example - Adding intent without patterns:
            >>> from insideLLMs.agents.routing import IntentClassifier
            >>> classifier = IntentClassifier()
            >>> classifier.add_intent(
            ...     name="philosophy",
            ...     description="Philosophical discussions and questions",
            ... )
            >>> # This intent can only match via model-based classification

        Example - Building a multi-intent classifier:
            >>> from insideLLMs.agents.routing import IntentClassifier
            >>> classifier = IntentClassifier()
            >>> classifier.add_intent("greeting", "Greetings", ["hello", "hi", "hey"])
            >>> classifier.add_intent("farewell", "Goodbyes", ["bye", "goodbye", "later"])
            >>> classifier.add_intent("thanks", "Gratitude", ["thank", "thanks", "appreciate"])
            >>> print(len(classifier.intents))  # 3

        Example - Updating an existing intent:
            >>> from insideLLMs.agents.routing import IntentClassifier
            >>> classifier = IntentClassifier()
            >>> classifier.add_intent("test", "First version", ["a"])
            >>> classifier.add_intent("test", "Updated version", ["a", "b"])
            >>> print(classifier.intents["test"])  # "Updated version"
        """
        self.intents[name] = description
        if patterns:
            self._intent_patterns[name] = patterns

    def classify(self, query: str) -> tuple[str, float]:
        """Classify a query into one of the registered intents.

        Uses a two-phase classification approach:
        1. Pattern matching: Checks for keyword patterns in the query
        2. Model-based: Falls back to LLM classification if patterns inconclusive

        Args:
            query: The query string to classify.

        Returns:
            tuple[str, float]: A tuple containing:
                - The classified intent name (or "unknown" if unclassifiable)
                - Confidence score between 0.0 and 1.0

        Example - High-confidence pattern match:
            >>> from insideLLMs.agents.routing import IntentClassifier
            >>> classifier = IntentClassifier()
            >>> classifier.add_intent("code", "Programming", ["python", "code", "function"])
            >>> intent, confidence = classifier.classify("Write a python function")
            >>> print(f"{intent}: {confidence:.2f}")  # code: 0.67

        Example - Low-confidence result:
            >>> from insideLLMs.agents.routing import IntentClassifier
            >>> classifier = IntentClassifier()
            >>> classifier.add_intent("code", "Programming", ["python"])
            >>> intent, confidence = classifier.classify("Hello world")
            >>> print(f"{intent}: {confidence:.2f}")  # unknown: 0.00

        Example - Model-based fallback:
            >>> from insideLLMs.agents.routing import IntentClassifier
            >>> from insideLLMs import DummyModel
            >>> classifier = IntentClassifier(model=DummyModel())
            >>> classifier.add_intent("support", "Technical support")
            >>> # No patterns defined, so model is used
            >>> intent, confidence = classifier.classify("I have a problem")

        Example - Using classification results:
            >>> from insideLLMs.agents.routing import IntentClassifier
            >>> classifier = IntentClassifier()
            >>> classifier.add_intent("urgent", "Urgent", ["urgent", "asap", "emergency"])
            >>> classifier.add_intent("normal", "Normal", ["help", "question"])
            >>> intent, conf = classifier.classify("Urgent help needed")
            >>> if intent == "urgent" and conf > 0.5:
            ...     print("Routing to priority queue")
        """
        query_lower = query.lower()

        # Try pattern matching first
        best_intent = None
        best_score = 0.0

        for intent, patterns in self._intent_patterns.items():
            score = sum(1 for p in patterns if p.lower() in query_lower)
            score = score / len(patterns) if patterns else 0
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score > 0.5:
            return best_intent, best_score

        # Use model if available and patterns didn't match well
        if self.model and self.intents:
            return self._classify_with_model(query)

        # Fallback to best pattern match or unknown
        return best_intent or "unknown", best_score

    def _classify_with_model(self, query: str) -> tuple[str, float]:
        """Classify a query using the configured LLM model.

        Constructs a prompt listing all registered intents and asks the model
        to classify the query into one of them.

        Args:
            query: The query string to classify.

        Returns:
            tuple[str, float]: A tuple containing:
                - The classified intent name (or "unknown" if classification failed)
                - Confidence score (0.8 for matched intent, 0.3 for unknown, 0.0 on error)

        Example:
            >>> from insideLLMs.agents.routing import IntentClassifier
            >>> from insideLLMs import DummyModel
            >>> classifier = IntentClassifier(model=DummyModel())
            >>> classifier.add_intent("booking", "Travel reservations")
            >>> classifier.add_intent("support", "Technical support")
            >>> # Internal method - called automatically by classify()
            >>> intent, conf = classifier._classify_with_model("Book a flight")
        """
        intent_list = "\n".join(f"- {name}: {desc}" for name, desc in self.intents.items())

        prompt = f"""Classify the following query into one of these intents:

{intent_list}

Query: {query}

Respond with just the intent name."""

        try:
            response = self.model.generate(prompt)
            intent = response.strip().lower()

            # Find best matching intent
            for name in self.intents:
                if name.lower() in intent or intent in name.lower():
                    return name, 0.8

            return "unknown", 0.3
        except Exception:
            return "unknown", 0.0


# =============================================================================
# Convenience Functions
# =============================================================================


def create_router(
    routes: Optional[list[dict[str, Any]]] = None,
    fallback_model: Optional[Any] = None,
    strategy: str = "best_match",
) -> SemanticRouter:
    """Create a semantic router from a list of route configurations.

    Convenience function for quickly setting up a router with multiple routes
    using simple dictionary configurations instead of constructing Route objects.

    Args:
        routes: List of route configuration dictionaries. Each dict should have:
            - name (str, required): Unique route identifier
            - model (Any, required): Model instance for this route
            - patterns (list[str], optional): Pattern keywords to match
            - description (str, optional): Route description for semantic matching
            - priority (int, optional): Route priority (default 0)
        fallback_model: Optional model to use when no routes match.
        strategy: Routing strategy name as string. One of:
            - "first_match": Use first matching route
            - "best_match": Use highest scoring route (default)
            - "round_robin": Rotate among matches
            - "random": Random selection
            - "load_balanced": Based on current load
            - "cost_optimized": Prefer cheaper models

    Returns:
        SemanticRouter: Configured router instance with all routes added.

    Example - Basic router creation:
        >>> from insideLLMs.agents.routing import create_router
        >>> from insideLLMs import DummyModel
        >>> router = create_router(
        ...     routes=[
        ...         {"name": "code", "model": DummyModel(), "patterns": ["python", "code"]},
        ...         {"name": "math", "model": DummyModel(), "patterns": ["calculate"]},
        ...     ]
        ... )
        >>> result = router.route("Write python code")
        >>> print(result.route_name)  # "code"

    Example - With fallback and custom strategy:
        >>> from insideLLMs.agents.routing import create_router
        >>> from insideLLMs import DummyModel
        >>> router = create_router(
        ...     routes=[
        ...         {"name": "specialized", "model": DummyModel(), "patterns": ["specific"]},
        ...     ],
        ...     fallback_model=DummyModel(name="General"),
        ...     strategy="cost_optimized",
        ... )

    Example - Routes with descriptions and priorities:
        >>> from insideLLMs.agents.routing import create_router
        >>> from insideLLMs import DummyModel
        >>> router = create_router(
        ...     routes=[
        ...         {
        ...             "name": "premium",
        ...             "model": DummyModel(name="GPT-4"),
        ...             "patterns": ["analyze"],
        ...             "description": "Complex analysis tasks",
        ...             "priority": 10,
        ...         },
        ...         {
        ...             "name": "basic",
        ...             "model": DummyModel(name="GPT-3.5"),
        ...             "patterns": ["simple"],
        ...             "description": "Simple questions",
        ...             "priority": 1,
        ...         },
        ...     ]
        ... )

    Example - Empty routes with only fallback:
        >>> from insideLLMs.agents.routing import create_router
        >>> from insideLLMs import DummyModel
        >>> router = create_router(fallback_model=DummyModel())
        >>> result = router.route("Any query")
        >>> print(result.fallback_used)  # True
    """
    config = RouterConfig(strategy=RoutingStrategy(strategy))
    router = SemanticRouter(config, fallback_model)

    if routes:
        for r in routes:
            route = Route(
                name=r["name"],
                model=r["model"],
                patterns=r.get("patterns", []),
                description=r.get("description", ""),
                priority=r.get("priority", 0),
            )
            router.add_route(route)

    return router


def quick_route(
    query: str,
    routes: list[dict[str, Any]],
    fallback: Optional[Any] = None,
) -> RoutingResult:
    """Quick one-liner for routing a single query.

    Creates a temporary router, routes the query, and returns the result.
    Useful for simple one-off routing needs without persistent router setup.

    Note: For multiple queries, use create_router() instead to avoid
    recreating the router for each query.

    Args:
        query: The query string to route and process.
        routes: List of route configuration dictionaries. Each dict should have:
            - name (str, required): Unique route identifier
            - model (Any, required): Model instance for this route
            - patterns (list[str], optional): Pattern keywords to match
            - description (str, optional): Route description
            - priority (int, optional): Route priority
        fallback: Optional fallback model for unmatched queries.

    Returns:
        RoutingResult: Complete result including response, latency, and match info.

    Example - Simple quick routing:
        >>> from insideLLMs.agents.routing import quick_route
        >>> from insideLLMs import DummyModel
        >>> result = quick_route(
        ...     query="Write python code",
        ...     routes=[
        ...         {"name": "code", "model": DummyModel(), "patterns": ["python", "code"]},
        ...     ]
        ... )
        >>> print(result.route_name)  # "code"
        >>> print(result.response)

    Example - With fallback:
        >>> from insideLLMs.agents.routing import quick_route
        >>> from insideLLMs import DummyModel
        >>> result = quick_route(
        ...     query="Random question",
        ...     routes=[{"name": "specific", "model": DummyModel(), "patterns": ["xyz"]}],
        ...     fallback=DummyModel(name="Fallback"),
        ... )
        >>> print(result.fallback_used)  # True

    Example - Multiple routes:
        >>> from insideLLMs.agents.routing import quick_route
        >>> from insideLLMs import DummyModel
        >>> result = quick_route(
        ...     query="Calculate 2+2",
        ...     routes=[
        ...         {"name": "code", "model": DummyModel(), "patterns": ["code"]},
        ...         {"name": "math", "model": DummyModel(), "patterns": ["calculate", "math"]},
        ...     ]
        ... )
        >>> print(result.route_name)  # "math"

    Example - Inspecting the full result:
        >>> from insideLLMs.agents.routing import quick_route
        >>> from insideLLMs import DummyModel
        >>> result = quick_route(
        ...     query="Hello",
        ...     routes=[{"name": "greeting", "model": DummyModel(), "patterns": ["hello"]}],
        ... )
        >>> print(f"Route: {result.route_name}")
        >>> print(f"Score: {result.match_score:.2f}")
        >>> print(f"Latency: {result.latency_ms:.2f}ms")
    """
    router = create_router(routes, fallback)
    return router.route(query)


def create_intent_router(
    model: Any,
    intent_models: dict[str, Any],
    fallback_model: Optional[Any] = None,
) -> SemanticRouter:
    """Create a router that routes based on query intent names.

    Creates a simple router where each intent maps directly to a model.
    The intent name itself is used as a pattern keyword, making this
    suitable for cases where queries explicitly mention the intent type.

    For more sophisticated intent-based routing, combine IntentClassifier
    with a SemanticRouter manually.

    Args:
        model: The model used for intent classification (currently unused but
            reserved for future intent classification integration).
        intent_models: Dictionary mapping intent names to model instances.
            Each intent becomes a route with the intent name as a pattern.
        fallback_model: Optional model for queries that don't match any intent.

    Returns:
        SemanticRouter: Router configured with intent-based routes.

    Example - Basic intent router:
        >>> from insideLLMs.agents.routing import create_intent_router
        >>> from insideLLMs import DummyModel
        >>> router = create_intent_router(
        ...     model=DummyModel(),  # Classification model
        ...     intent_models={
        ...         "code": DummyModel(name="CodeHelper"),
        ...         "math": DummyModel(name="MathSolver"),
        ...         "writing": DummyModel(name="Writer"),
        ...     }
        ... )
        >>> result = router.route("Help me with code")
        >>> print(result.route_name)  # "code"

    Example - With fallback:
        >>> from insideLLMs.agents.routing import create_intent_router
        >>> from insideLLMs import DummyModel
        >>> router = create_intent_router(
        ...     model=DummyModel(),
        ...     intent_models={"specific": DummyModel()},
        ...     fallback_model=DummyModel(name="General"),
        ... )
        >>> result = router.route("Random unrelated query")
        >>> print(result.fallback_used)  # True

    Example - Checking available routes:
        >>> from insideLLMs.agents.routing import create_intent_router
        >>> from insideLLMs import DummyModel
        >>> router = create_intent_router(
        ...     model=DummyModel(),
        ...     intent_models={
        ...         "booking": DummyModel(),
        ...         "support": DummyModel(),
        ...         "feedback": DummyModel(),
        ...     }
        ... )
        >>> routes = router.list_routes()
        >>> print([r.name for r in routes])  # ["booking", "support", "feedback"]

    Example - Semantic matching with descriptions:
        >>> from insideLLMs.agents.routing import create_intent_router
        >>> from insideLLMs import DummyModel
        >>> router = create_intent_router(
        ...     model=DummyModel(),
        ...     intent_models={"support": DummyModel()},
        ... )
        >>> # Routes are created with auto-generated descriptions
        >>> route = router.get_route("support")
        >>> print(route.description)  # "Handle support queries"
    """
    router = SemanticRouter(fallback_model=fallback_model)

    for intent, intent_model in intent_models.items():
        router.add_route(
            Route(
                name=intent,
                model=intent_model,
                patterns=[intent],
                description=f"Handle {intent} queries",
            )
        )

    return router


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Configuration
    "RouterConfig",
    "RoutingStrategy",
    "RouteStatus",
    # Data classes
    "RouteMatch",
    "RoutingResult",
    "RouteStats",
    # Routes
    "Route",
    # Router
    "SemanticRouter",
    "SemanticMatcher",
    # Utilities
    "ModelPool",
    "IntentClassifier",
    # Convenience functions
    "create_router",
    "quick_route",
    "create_intent_router",
]
