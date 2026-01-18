"""Deployment wrapper for serving insideLLMs via FastAPI.

This module provides tools for deploying LLM services as REST APIs:
- FastAPI application factory
- Model and probe endpoints
- Request/response schemas
- Health checks and monitoring
- Async support
- Rate limiting and authentication

Example:
    >>> from insideLLMs.deployment import create_app, ModelEndpoint
    >>> from insideLLMs import DummyModel
    >>>
    >>> model = DummyModel()
    >>> app = create_app(model)
    >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000

Note:
    FastAPI and uvicorn are optional dependencies. Install with:
    pip install fastapi uvicorn
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
)
import asyncio
import functools
import hashlib
import inspect
import json
import logging
import threading
import time
import uuid

__all__ = [
    # Application
    "create_app",
    "DeploymentApp",
    "AppConfig",
    # Endpoints
    "ModelEndpoint",
    "ProbeEndpoint",
    "BatchEndpoint",
    # Schemas
    "GenerateRequest",
    "GenerateResponse",
    "ProbeRequest",
    "ProbeResponse",
    "HealthResponse",
    "ErrorResponse",
    # Middleware
    "RateLimiter",
    "APIKeyAuth",
    "RequestLogger",
    # Monitoring
    "MetricsCollector",
    "HealthChecker",
    # Configuration
    "DeploymentConfig",
    "EndpointConfig",
    # Convenience
    "quick_deploy",
    "create_model_endpoint",
    "create_probe_endpoint",
]

# Check for FastAPI availability
try:
    from fastapi import FastAPI, HTTPException, Depends, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Dummy classes for type hints
    class FastAPI:  # type: ignore
        pass
    class BaseModel:  # type: ignore
        pass
    class HTTPException(Exception):  # type: ignore
        pass
    def Field(*args, **kwargs):  # type: ignore
        return None
    def Depends(*args, **kwargs):  # type: ignore
        return None

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Schemas
# =============================================================================


if FASTAPI_AVAILABLE:
    class GenerateRequest(BaseModel):
        """Request schema for generation endpoint."""
        prompt: str = Field(..., description="Input prompt for generation")
        temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
        max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
        stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
        stream: bool = Field(False, description="Enable streaming response")
        metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

        class Config:
            json_schema_extra = {
                "example": {
                    "prompt": "What is the capital of France?",
                    "temperature": 0.7,
                    "max_tokens": 100,
                }
            }

    class GenerateResponse(BaseModel):
        """Response schema for generation endpoint."""
        response: str = Field(..., description="Generated text")
        model_id: Optional[str] = Field(None, description="Model identifier")
        prompt_tokens: Optional[int] = Field(None, description="Input tokens used")
        completion_tokens: Optional[int] = Field(None, description="Output tokens generated")
        latency_ms: float = Field(..., description="Processing latency in milliseconds")
        request_id: str = Field(..., description="Unique request identifier")
        metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class ProbeRequest(BaseModel):
        """Request schema for probe endpoint."""
        prompts: List[str] = Field(..., description="List of prompts to probe")
        probe_type: Optional[str] = Field(None, description="Type of probe to run")
        metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class ProbeResponse(BaseModel):
        """Response schema for probe endpoint."""
        results: List[Dict[str, Any]] = Field(..., description="Probe results")
        summary: Optional[Dict[str, Any]] = Field(None, description="Results summary")
        latency_ms: float = Field(..., description="Processing latency")
        request_id: str = Field(..., description="Unique request identifier")

    class BatchRequest(BaseModel):
        """Request schema for batch generation."""
        prompts: List[str] = Field(..., description="List of prompts")
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        max_tokens: Optional[int] = Field(None)

    class BatchResponse(BaseModel):
        """Response schema for batch generation."""
        responses: List[str] = Field(..., description="Generated responses")
        total_tokens: int = Field(..., description="Total tokens used")
        latency_ms: float = Field(..., description="Total processing latency")
        request_id: str = Field(..., description="Unique request identifier")

    class HealthResponse(BaseModel):
        """Response schema for health check."""
        status: str = Field(..., description="Health status")
        version: str = Field(..., description="API version")
        model_id: Optional[str] = Field(None, description="Model identifier")
        uptime_seconds: float = Field(..., description="Server uptime")
        checks: Optional[Dict[str, bool]] = Field(None, description="Individual health checks")

    class ErrorResponse(BaseModel):
        """Response schema for errors."""
        error: str = Field(..., description="Error message")
        error_code: str = Field(..., description="Error code")
        request_id: str = Field(..., description="Request identifier")
        details: Optional[Dict[str, Any]] = Field(None, description="Additional details")

else:
    # Dummy classes when FastAPI not available
    class GenerateRequest:  # type: ignore
        pass
    class GenerateResponse:  # type: ignore
        pass
    class ProbeRequest:  # type: ignore
        pass
    class ProbeResponse:  # type: ignore
        pass
    class BatchRequest:  # type: ignore
        pass
    class BatchResponse:  # type: ignore
        pass
    class HealthResponse:  # type: ignore
        pass
    class ErrorResponse:  # type: ignore
        pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EndpointConfig:
    """Configuration for an endpoint."""
    path: str = "/generate"
    method: str = "POST"
    enabled: bool = True
    rate_limit: Optional[int] = None  # Requests per minute
    timeout_seconds: float = 30.0
    require_auth: bool = False
    tags: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "method": self.method,
            "enabled": self.enabled,
            "rate_limit": self.rate_limit,
            "timeout_seconds": self.timeout_seconds,
            "require_auth": self.require_auth,
            "tags": self.tags,
            "description": self.description,
        }


@dataclass
class DeploymentConfig:
    """Configuration for the deployment."""
    title: str = "insideLLMs API"
    description: str = "API for LLM probing and evaluation"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_docs: bool = True
    enable_metrics: bool = True
    enable_health: bool = True
    api_key: Optional[str] = None
    log_requests: bool = True
    max_batch_size: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "version": self.version,
            "host": self.host,
            "port": self.port,
            "cors_origins": self.cors_origins,
            "enable_docs": self.enable_docs,
            "enable_metrics": self.enable_metrics,
            "enable_health": self.enable_health,
            "log_requests": self.log_requests,
            "max_batch_size": self.max_batch_size,
        }


@dataclass
class AppConfig:
    """Combined application configuration."""
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    generate_endpoint: EndpointConfig = field(default_factory=lambda: EndpointConfig(
        path="/generate",
        description="Generate text from prompt",
        tags=["generation"],
    ))
    batch_endpoint: EndpointConfig = field(default_factory=lambda: EndpointConfig(
        path="/batch",
        description="Batch generation",
        tags=["generation"],
    ))
    probe_endpoint: EndpointConfig = field(default_factory=lambda: EndpointConfig(
        path="/probe",
        description="Run probes on model",
        tags=["probing"],
    ))


# =============================================================================
# Middleware Components
# =============================================================================


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (defaults to requests_per_minute)
        """
        self.rate = requests_per_minute / 60.0  # Tokens per second
        self.burst_size = burst_size or requests_per_minute
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, key: str) -> Dict[str, float]:
        """Get or create bucket for key."""
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": float(self.burst_size),
                    "last_update": time.time(),
                }
            return self._buckets[key]

    def is_allowed(self, key: str = "default") -> bool:
        """Check if request is allowed.

        Args:
            key: Rate limit key (e.g., API key, IP address)

        Returns:
            True if allowed, False if rate limited
        """
        bucket = self._get_bucket(key)

        with self._lock:
            now = time.time()
            elapsed = now - bucket["last_update"]
            bucket["last_update"] = now

            # Add tokens based on elapsed time
            bucket["tokens"] = min(
                self.burst_size,
                bucket["tokens"] + elapsed * self.rate
            )

            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False

    def get_wait_time(self, key: str = "default") -> float:
        """Get time to wait before next request.

        Args:
            key: Rate limit key

        Returns:
            Seconds to wait (0 if no wait needed)
        """
        bucket = self._get_bucket(key)

        with self._lock:
            if bucket["tokens"] >= 1:
                return 0.0
            return (1 - bucket["tokens"]) / self.rate


class APIKeyAuth:
    """API key authentication handler."""

    def __init__(
        self,
        valid_keys: Optional[List[str]] = None,
        header_name: str = "X-API-Key",
    ):
        """Initialize API key auth.

        Args:
            valid_keys: List of valid API keys
            header_name: Header name for API key
        """
        self.valid_keys = set(valid_keys) if valid_keys else set()
        self.header_name = header_name
        self._key_usage: Dict[str, int] = {}

    def add_key(self, key: str) -> None:
        """Add a valid API key."""
        self.valid_keys.add(key)

    def remove_key(self, key: str) -> None:
        """Remove an API key."""
        self.valid_keys.discard(key)

    def validate(self, key: Optional[str]) -> bool:
        """Validate an API key.

        Args:
            key: API key to validate

        Returns:
            True if valid, False otherwise
        """
        if not self.valid_keys:
            return True  # No keys configured = open access

        if key and key in self.valid_keys:
            self._key_usage[key] = self._key_usage.get(key, 0) + 1
            return True
        return False

    def get_usage(self, key: str) -> int:
        """Get usage count for a key."""
        return self._key_usage.get(key, 0)


class RequestLogger:
    """Request logging handler."""

    def __init__(
        self,
        log_func: Optional[Callable[[Dict[str, Any]], None]] = None,
        include_body: bool = False,
        include_response: bool = False,
    ):
        """Initialize request logger.

        Args:
            log_func: Custom logging function
            include_body: Include request body in logs
            include_response: Include response in logs
        """
        self.log_func = log_func or self._default_log
        self.include_body = include_body
        self.include_response = include_response
        self._logs: List[Dict[str, Any]] = []

    def _default_log(self, entry: Dict[str, Any]) -> None:
        """Default logging to list."""
        self._logs.append(entry)
        logger.info(f"Request: {entry.get('method')} {entry.get('path')} - {entry.get('status_code')}")

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        request_id: str,
        body: Optional[Dict[str, Any]] = None,
        response: Optional[Dict[str, Any]] = None,
        client_ip: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Log a request.

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            latency_ms: Request latency
            request_id: Request identifier
            body: Request body (if include_body)
            response: Response body (if include_response)
            client_ip: Client IP address
            api_key: API key used (masked)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "request_id": request_id,
            "client_ip": client_ip,
        }

        if api_key:
            entry["api_key"] = f"***{api_key[-4:]}" if len(api_key) > 4 else "***"

        if self.include_body and body:
            entry["body"] = body

        if self.include_response and response:
            entry["response"] = response

        self.log_func(entry)

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs."""
        return self._logs[-limit:]


# =============================================================================
# Monitoring Components
# =============================================================================


class MetricsCollector:
    """Collects and exposes metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value

    def record(self, name: str, value: float) -> None:
        """Record a histogram value."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)
            # Keep last 1000 values
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            histogram_stats = {}
            for name, values in self._histograms.items():
                if values:
                    sorted_values = sorted(values)
                    histogram_stats[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "p50": sorted_values[len(values) // 2],
                        "p95": sorted_values[int(len(values) * 0.95)],
                        "p99": sorted_values[int(len(values) * 0.99)],
                    }

            return {
                "uptime_seconds": time.time() - self._start_time,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": histogram_stats,
            }


class HealthChecker:
    """Health check manager."""

    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable[[], bool]] = {}
        self._start_time = time.time()

    def add_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Add a health check.

        Args:
            name: Check name
            check_func: Function that returns True if healthy
        """
        self._checks[name] = check_func

    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks.

        Returns:
            Health status with individual check results
        """
        results = {}
        all_healthy = True

        for name, check_func in self._checks.items():
            try:
                results[name] = check_func()
                if not results[name]:
                    all_healthy = False
            except Exception as e:
                results[name] = False
                all_healthy = False
                logger.error(f"Health check '{name}' failed: {e}")

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": results,
            "uptime_seconds": time.time() - self._start_time,
        }


# =============================================================================
# Endpoint Classes
# =============================================================================


class ModelEndpoint:
    """Wraps a model for API serving."""

    def __init__(
        self,
        model: Any,
        config: Optional[EndpointConfig] = None,
    ):
        """Initialize model endpoint.

        Args:
            model: Model to serve
            config: Endpoint configuration
        """
        self.model = model
        self.config = config or EndpointConfig()
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate response from model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional model arguments

        Returns:
            Generation result dictionary
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Check if model has async generate
            if hasattr(self.model, 'agenerate'):
                response = await self.model.agenerate(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            elif asyncio.iscoroutinefunction(getattr(self.model, 'generate', None)):
                response = await self.model.generate(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            else:
                # Run sync model in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate(
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens if max_tokens else None,
                        **kwargs
                    )
                )

            latency_ms = (time.time() - start_time) * 1000
            self._request_count += 1
            self._total_latency += latency_ms

            return {
                "response": response,
                "model_id": getattr(self.model, 'model_id', None),
                "latency_ms": latency_ms,
                "request_id": request_id,
            }

        except Exception as e:
            self._error_count += 1
            raise RuntimeError(f"Generation failed: {str(e)}") from e

    def get_stats(self) -> Dict[str, Any]:
        """Get endpoint statistics."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "average_latency_ms": (
                self._total_latency / self._request_count
                if self._request_count > 0 else 0
            ),
        }


class ProbeEndpoint:
    """Wraps probes for API serving."""

    def __init__(
        self,
        model: Any,
        probes: Optional[Dict[str, Any]] = None,
        config: Optional[EndpointConfig] = None,
    ):
        """Initialize probe endpoint.

        Args:
            model: Model to probe
            probes: Dictionary of probe name -> probe instance
            config: Endpoint configuration
        """
        self.model = model
        self.probes = probes or {}
        self.config = config or EndpointConfig(path="/probe")

    def add_probe(self, name: str, probe: Any) -> None:
        """Add a probe.

        Args:
            name: Probe name
            probe: Probe instance
        """
        self.probes[name] = probe

    async def run_probe(
        self,
        prompts: List[str],
        probe_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run probes on prompts.

        Args:
            prompts: List of prompts
            probe_type: Specific probe to run (or all if None)
            **kwargs: Additional probe arguments

        Returns:
            Probe results dictionary
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        results = []
        probes_to_run = (
            {probe_type: self.probes[probe_type]}
            if probe_type and probe_type in self.probes
            else self.probes
        )

        for probe_name, probe in probes_to_run.items():
            try:
                probe_results = []
                for prompt in prompts:
                    # Run probe evaluation
                    if hasattr(probe, 'evaluate'):
                        response = self.model.generate(prompt)
                        result = probe.evaluate(prompt, response)
                        probe_results.append({
                            "prompt": prompt,
                            "response": response,
                            "result": result,
                        })
                    else:
                        probe_results.append({
                            "prompt": prompt,
                            "error": "Probe does not support evaluation",
                        })

                results.append({
                    "probe": probe_name,
                    "results": probe_results,
                })
            except Exception as e:
                results.append({
                    "probe": probe_name,
                    "error": str(e),
                })

        latency_ms = (time.time() - start_time) * 1000

        return {
            "results": results,
            "latency_ms": latency_ms,
            "request_id": request_id,
        }


class BatchEndpoint:
    """Handles batch generation requests."""

    def __init__(
        self,
        model: Any,
        max_batch_size: int = 100,
        config: Optional[EndpointConfig] = None,
    ):
        """Initialize batch endpoint.

        Args:
            model: Model to use
            max_batch_size: Maximum batch size
            config: Endpoint configuration
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.config = config or EndpointConfig(path="/batch")

    async def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate responses for batch of prompts.

        Args:
            prompts: List of prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            **kwargs: Additional arguments

        Returns:
            Batch generation results
        """
        if len(prompts) > self.max_batch_size:
            raise ValueError(f"Batch size exceeds maximum of {self.max_batch_size}")

        request_id = str(uuid.uuid4())
        start_time = time.time()

        responses = []
        total_tokens = 0

        for prompt in prompts:
            try:
                if hasattr(self.model, 'agenerate'):
                    response = await self.model.agenerate(prompt, temperature=temperature)
                else:
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda p=prompt: self.model.generate(p, temperature=temperature)
                    )
                responses.append(response)
                # Estimate tokens (rough approximation)
                total_tokens += len(prompt.split()) + len(response.split())
            except Exception as e:
                responses.append(f"Error: {str(e)}")

        latency_ms = (time.time() - start_time) * 1000

        return {
            "responses": responses,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "request_id": request_id,
        }


# =============================================================================
# Application Factory
# =============================================================================


class DeploymentApp:
    """Wrapper for FastAPI deployment."""

    def __init__(
        self,
        model: Any,
        config: Optional[AppConfig] = None,
    ):
        """Initialize deployment app.

        Args:
            model: Model to serve
            config: Application configuration
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for deployment. "
                "Install with: pip install fastapi uvicorn"
            )

        self.model = model
        self.config = config or AppConfig()
        self._app: Optional[FastAPI] = None
        self._model_endpoint = ModelEndpoint(model, self.config.generate_endpoint)
        self._batch_endpoint = BatchEndpoint(model, self.config.deployment.max_batch_size)
        self._probe_endpoint = ProbeEndpoint(model, config=self.config.probe_endpoint)
        self._metrics = MetricsCollector()
        self._health_checker = HealthChecker()
        self._rate_limiter = RateLimiter()
        self._auth = APIKeyAuth(
            [self.config.deployment.api_key] if self.config.deployment.api_key else None
        )
        self._request_logger = RequestLogger()

    def build_app(self) -> "FastAPI":
        """Build and configure FastAPI application.

        Returns:
            Configured FastAPI application
        """
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse

        app = FastAPI(
            title=self.config.deployment.title,
            description=self.config.deployment.description,
            version=self.config.deployment.version,
            docs_url="/docs" if self.config.deployment.enable_docs else None,
            redoc_url="/redoc" if self.config.deployment.enable_docs else None,
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.deployment.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add request ID middleware
        @app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # Register endpoints
        self._register_endpoints(app)

        self._app = app
        return app

    def _register_endpoints(self, app: "FastAPI") -> None:
        """Register all endpoints on the app."""
        from fastapi import HTTPException, Request

        # Generate endpoint
        if self.config.generate_endpoint.enabled:
            @app.post(
                self.config.generate_endpoint.path,
                response_model=GenerateResponse,
                tags=self.config.generate_endpoint.tags,
                description=self.config.generate_endpoint.description,
            )
            async def generate(request: GenerateRequest, req: Request):
                # Rate limiting
                if not self._rate_limiter.is_allowed(getattr(req.state, 'request_id', 'default')):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

                self._metrics.increment("requests_total")
                start_time = time.time()

                try:
                    result = await self._model_endpoint.generate(
                        prompt=request.prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    )

                    latency_ms = (time.time() - start_time) * 1000
                    self._metrics.record("latency_ms", latency_ms)
                    self._metrics.increment("requests_success")

                    return GenerateResponse(
                        response=result["response"],
                        model_id=result.get("model_id"),
                        latency_ms=result["latency_ms"],
                        request_id=result["request_id"],
                    )

                except Exception as e:
                    self._metrics.increment("requests_error")
                    raise HTTPException(status_code=500, detail=str(e))

        # Batch endpoint
        if self.config.batch_endpoint.enabled:
            @app.post(
                self.config.batch_endpoint.path,
                response_model=BatchResponse,
                tags=self.config.batch_endpoint.tags,
                description=self.config.batch_endpoint.description,
            )
            async def batch_generate(request: BatchRequest):
                self._metrics.increment("batch_requests_total")

                try:
                    result = await self._batch_endpoint.generate_batch(
                        prompts=request.prompts,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    )

                    return BatchResponse(
                        responses=result["responses"],
                        total_tokens=result["total_tokens"],
                        latency_ms=result["latency_ms"],
                        request_id=result["request_id"],
                    )

                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

        # Probe endpoint
        if self.config.probe_endpoint.enabled:
            @app.post(
                self.config.probe_endpoint.path,
                response_model=ProbeResponse,
                tags=self.config.probe_endpoint.tags,
                description=self.config.probe_endpoint.description,
            )
            async def run_probe(request: ProbeRequest):
                self._metrics.increment("probe_requests_total")

                try:
                    result = await self._probe_endpoint.run_probe(
                        prompts=request.prompts,
                        probe_type=request.probe_type,
                    )

                    return ProbeResponse(
                        results=result["results"],
                        latency_ms=result["latency_ms"],
                        request_id=result["request_id"],
                    )

                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

        # Health endpoint
        if self.config.deployment.enable_health:
            @app.get("/health", response_model=HealthResponse, tags=["monitoring"])
            async def health():
                health_status = self._health_checker.run_checks()
                return HealthResponse(
                    status=health_status["status"],
                    version=self.config.deployment.version,
                    model_id=getattr(self.model, 'model_id', None),
                    uptime_seconds=health_status["uptime_seconds"],
                    checks=health_status["checks"],
                )

        # Metrics endpoint
        if self.config.deployment.enable_metrics:
            @app.get("/metrics", tags=["monitoring"])
            async def metrics():
                return self._metrics.get_metrics()

        # Root endpoint
        @app.get("/", tags=["info"])
        async def root():
            return {
                "name": self.config.deployment.title,
                "version": self.config.deployment.version,
                "model_id": getattr(self.model, 'model_id', None),
            }

    def add_probe(self, name: str, probe: Any) -> None:
        """Add a probe to the endpoint.

        Args:
            name: Probe name
            probe: Probe instance
        """
        self._probe_endpoint.add_probe(name, probe)

    def add_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Add a health check.

        Args:
            name: Check name
            check_func: Health check function
        """
        self._health_checker.add_check(name, check_func)

    @property
    def app(self) -> "FastAPI":
        """Get or build FastAPI application."""
        if self._app is None:
            self.build_app()
        return self._app  # type: ignore


def create_app(
    model: Any,
    title: str = "insideLLMs API",
    description: str = "API for LLM probing and evaluation",
    version: str = "1.0.0",
    cors_origins: Optional[List[str]] = None,
    enable_docs: bool = True,
    enable_metrics: bool = True,
    api_key: Optional[str] = None,
) -> "FastAPI":
    """Create a FastAPI application for serving a model.

    Args:
        model: Model to serve
        title: API title
        description: API description
        version: API version
        cors_origins: CORS allowed origins
        enable_docs: Enable OpenAPI docs
        enable_metrics: Enable metrics endpoint
        api_key: Optional API key for authentication

    Returns:
        Configured FastAPI application

    Example:
        >>> from insideLLMs import DummyModel
        >>> from insideLLMs.deployment import create_app
        >>>
        >>> model = DummyModel()
        >>> app = create_app(model, title="My LLM API")
        >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000
    """
    config = AppConfig(
        deployment=DeploymentConfig(
            title=title,
            description=description,
            version=version,
            cors_origins=cors_origins or ["*"],
            enable_docs=enable_docs,
            enable_metrics=enable_metrics,
            api_key=api_key,
        )
    )

    deployment = DeploymentApp(model, config)
    return deployment.build_app()


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_deploy(
    model: Any,
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs: Any,
) -> None:
    """Quickly deploy a model as an API.

    Args:
        model: Model to serve
        host: Host to bind to
        port: Port to bind to
        **kwargs: Additional arguments for create_app

    Example:
        >>> from insideLLMs import DummyModel
        >>> from insideLLMs.deployment import quick_deploy
        >>>
        >>> model = DummyModel()
        >>> quick_deploy(model, port=8080)  # Blocking call
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required. Install with: pip install fastapi uvicorn")

    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn required. Install with: pip install uvicorn")

    app = create_app(model, **kwargs)
    uvicorn.run(app, host=host, port=port)


def create_model_endpoint(
    model: Any,
    path: str = "/generate",
    rate_limit: Optional[int] = None,
) -> ModelEndpoint:
    """Create a model endpoint.

    Args:
        model: Model to wrap
        path: Endpoint path
        rate_limit: Rate limit (requests per minute)

    Returns:
        Configured ModelEndpoint
    """
    config = EndpointConfig(
        path=path,
        rate_limit=rate_limit,
    )
    return ModelEndpoint(model, config)


def create_probe_endpoint(
    model: Any,
    probes: Optional[Dict[str, Any]] = None,
    path: str = "/probe",
) -> ProbeEndpoint:
    """Create a probe endpoint.

    Args:
        model: Model to probe
        probes: Dictionary of probe name -> probe instance
        path: Endpoint path

    Returns:
        Configured ProbeEndpoint
    """
    config = EndpointConfig(path=path)
    return ProbeEndpoint(model, probes, config)
