"""Semantic Model Routing Module for insideLLMs.

This module provides intelligent routing of queries to appropriate models:
- Semantic route matching based on query content
- Dynamic model selection based on task type
- Fallback and retry strategies
- Cost-aware routing
- Load balancing across models

Example:
    >>> from insideLLMs.routing import SemanticRouter, Route
    >>> from insideLLMs import DummyModel
    >>>
    >>> # Create routes
    >>> router = SemanticRouter()
    >>> router.add_route(Route(
    ...     name="code",
    ...     patterns=["code", "programming", "function"],
    ...     model=DummyModel(name="CodeModel"),
    ... ))
    >>> router.add_route(Route(
    ...     name="general",
    ...     patterns=["what", "how", "explain"],
    ...     model=DummyModel(name="GeneralModel"),
    ... ))
    >>>
    >>> # Route query
    >>> result = router.route("Write a Python function")
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

# =============================================================================
# Configuration and Types
# =============================================================================


class RoutingStrategy(Enum):
    """Strategy for selecting among matching routes."""

    FIRST_MATCH = "first_match"  # Use first matching route
    BEST_MATCH = "best_match"  # Use highest scoring route
    ROUND_ROBIN = "round_robin"  # Rotate among matching routes
    RANDOM = "random"  # Random selection among matches
    LOAD_BALANCED = "load_balanced"  # Based on current load
    COST_OPTIMIZED = "cost_optimized"  # Prefer cheaper models


class RouteStatus(Enum):
    """Status of a route."""

    ACTIVE = "active"
    DISABLED = "disabled"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


@dataclass
class RouterConfig:
    """Configuration for the router."""

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
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "similarity_threshold": self.similarity_threshold,
            "case_sensitive": self.case_sensitive,
            "use_fallback": self.use_fallback,
            "max_retries": self.max_retries,
        }


@dataclass
class RouteMatch:
    """Result of matching a query to a route."""

    route_name: str
    score: float
    matched_patterns: list[str]
    model: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "route_name": self.route_name,
            "score": self.score,
            "matched_patterns": self.matched_patterns,
            "model": str(type(self.model).__name__),
            "metadata": self.metadata,
        }


@dataclass
class RoutingResult:
    """Result of routing a query."""

    query: str
    route_name: str
    response: Any
    latency_ms: float
    match_score: float
    retries: int = 0
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Statistics for a route."""

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
        """Update stats after a request."""
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
        """Convert to dictionary."""
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
    """A route that maps queries to a model.

    Args:
        name: Route name
        model: Model to use for this route
        patterns: List of pattern strings to match
        regex_patterns: List of regex patterns
        description: Human-readable description
        priority: Route priority (higher = checked first)
        cost_per_token: Cost estimate for billing
        max_tokens: Max tokens for this route
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
        """Check if query matches this route.

        Args:
            query: The query to match
            case_sensitive: Whether to match case-sensitively

        Returns:
            Tuple of (matches, score, matched_patterns)
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
        """Get route statistics."""
        return self._stats

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Semantic matcher for routes using embeddings.

    Uses simple word-based similarity when no embedder is provided.
    """

    def __init__(
        self,
        embedder: Optional[Callable[[str], list[float]]] = None,
    ):
        self._embedder = embedder
        self._route_embeddings: dict[str, list[float]] = {}

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        if self._embedder:
            emb1 = self._embedder(text1)
            emb2 = self._embedder(text2)
            return self._cosine_similarity(emb1, emb2)
        else:
            return self._word_similarity(text1, text2)

    def _word_similarity(self, text1: str, text2: str) -> float:
        """Simple word-based similarity (Jaccard)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def cache_route_embedding(self, route_name: str, text: str) -> None:
        """Cache embedding for a route."""
        if self._embedder:
            self._route_embeddings[route_name] = self._embedder(text)

    def get_cached_embedding(self, route_name: str) -> Optional[list[float]]:
        """Get cached embedding for a route."""
        return self._route_embeddings.get(route_name)


# =============================================================================
# Semantic Router
# =============================================================================


class SemanticRouter:
    """Semantic router for intelligent model selection.

    Routes queries to appropriate models based on:
    - Pattern matching
    - Semantic similarity
    - Priority and cost optimization
    - Load balancing

    Args:
        config: Router configuration
        fallback_model: Model to use when no route matches
        embedder: Optional embedding function for semantic matching
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

        Args:
            route: Route to add
        """
        with self._lock:
            self._routes[route.name] = route

            # Cache embedding for semantic matching
            if route.description:
                self._matcher.cache_route_embedding(route.name, route.description)

    def remove_route(self, name: str) -> bool:
        """Remove a route by name.

        Args:
            name: Route name

        Returns:
            True if removed
        """
        with self._lock:
            if name in self._routes:
                del self._routes[name]
                return True
            return False

    def get_route(self, name: str) -> Optional[Route]:
        """Get a route by name."""
        return self._routes.get(name)

    def list_routes(self) -> list[Route]:
        """List all routes."""
        return list(self._routes.values())

    def match(self, query: str) -> list[RouteMatch]:
        """Match a query to routes.

        Args:
            query: The query to match

        Returns:
            List of RouteMatch sorted by score
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
        """Select a route from matches based on strategy.

        Args:
            matches: List of matching routes

        Returns:
            Selected RouteMatch or None
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
        """Route a query and get response.

        Args:
            query: The query to route
            **kwargs: Additional arguments for the model

        Returns:
            RoutingResult with response
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
        """Get cached route for query."""
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self._route_cache:
            route_name, cached_at = self._route_cache[key]
            if datetime.now() - cached_at < timedelta(seconds=self.config.cache_ttl_seconds):
                return route_name
            else:
                del self._route_cache[key]
        return None

    def _cache_route(self, query: str, route_name: str) -> None:
        """Cache route for query."""
        key = hashlib.md5(query.encode()).hexdigest()
        self._route_cache[key] = (route_name, datetime.now())

    def get_stats(self) -> dict[str, RouteStats]:
        """Get statistics for all routes."""
        return {name: route.stats for name, route in self._routes.items()}


# =============================================================================
# Model Pool
# =============================================================================


class ModelPool:
    """Pool of models for load balancing.

    Args:
        models: List of models
        strategy: Selection strategy
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
        """Select a model from the pool.

        Returns:
            Tuple of (model, index)
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
        """Generate using a model from the pool."""
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
        """Get pool statistics."""
        return self._stats.copy()


# =============================================================================
# Intent Classifier
# =============================================================================


class IntentClassifier:
    """Classify query intent for routing.

    Args:
        model: Model to use for classification
        intents: Dict mapping intent name to description
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
        """Add an intent.

        Args:
            name: Intent name
            description: Intent description
            patterns: Optional pattern keywords
        """
        self.intents[name] = description
        if patterns:
            self._intent_patterns[name] = patterns

    def classify(self, query: str) -> tuple[str, float]:
        """Classify query intent.

        Args:
            query: The query to classify

        Returns:
            Tuple of (intent_name, confidence)
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
        """Classify using the model."""
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
    """Create a semantic router.

    Args:
        routes: List of route configurations
        fallback_model: Fallback model
        strategy: Routing strategy

    Returns:
        Configured SemanticRouter
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
    """Quick helper for routing a query.

    Args:
        query: The query to route
        routes: Route configurations
        fallback: Fallback model

    Returns:
        RoutingResult
    """
    router = create_router(routes, fallback)
    return router.route(query)


def create_intent_router(
    model: Any,
    intent_models: dict[str, Any],
    fallback_model: Optional[Any] = None,
) -> SemanticRouter:
    """Create a router based on intent classification.

    Args:
        model: Model for intent classification
        intent_models: Dict mapping intent to model
        fallback_model: Fallback model

    Returns:
        Configured SemanticRouter
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
