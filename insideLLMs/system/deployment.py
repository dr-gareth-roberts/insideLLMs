"""Deployment wrapper for serving insideLLMs via FastAPI.

This module provides a complete toolkit for deploying LLM services as REST APIs,
including FastAPI application factories, model and probe endpoints, request/response
schemas, health checks, monitoring, async support, rate limiting, and authentication.

Overview
--------
The deployment module enables you to expose your LLM models and probes as production-ready
REST APIs with minimal configuration. It supports:

- **FastAPI Application Factory**: Create configured FastAPI apps with a single call
- **Model Endpoints**: Wrap any model for text generation via HTTP
- **Probe Endpoints**: Run evaluation probes through API calls
- **Batch Processing**: Handle multiple prompts in a single request
- **Rate Limiting**: Token bucket algorithm with per-key limits
- **Authentication**: API key-based access control
- **Monitoring**: Health checks, metrics collection, and request logging
- **Async Support**: Full async/await support for non-blocking operations

Examples
--------
Basic deployment with default configuration:

    >>> from insideLLMs.system.deployment import create_app
    >>> from insideLLMs import DummyModel
    >>>
    >>> model = DummyModel()
    >>> app = create_app(model)
    >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000

Quick deployment for development and testing:

    >>> from insideLLMs.system.deployment import quick_deploy
    >>> from insideLLMs import DummyModel
    >>>
    >>> model = DummyModel()
    >>> quick_deploy(model, host="127.0.0.1", port=8080)  # Blocking call

Custom configuration with authentication and rate limiting:

    >>> from insideLLMs.system.deployment import create_app, DeploymentConfig, AppConfig
    >>> from insideLLMs import DummyModel
    >>>
    >>> config = AppConfig(
    ...     deployment=DeploymentConfig(
    ...         title="My Secure LLM API",
    ...         api_key="secret-key-123",
    ...         enable_metrics=True,
    ...     )
    ... )
    >>> model = DummyModel()
    >>> deployment = DeploymentApp(model, config)
    >>> app = deployment.build_app()

Adding probes to the deployment:

    >>> from insideLLMs.system.deployment import DeploymentApp, AppConfig
    >>> from insideLLMs import DummyModel
    >>> from insideLLMs.probes import TruthfulnessProbe
    >>>
    >>> model = DummyModel()
    >>> deployment = DeploymentApp(model)
    >>> deployment.add_probe("truthfulness", TruthfulnessProbe())
    >>> app = deployment.app

Note
----
FastAPI and uvicorn are optional dependencies. Install with::

    pip install fastapi uvicorn

See Also
--------
- FastAPI documentation: https://fastapi.tiangolo.com/
- Uvicorn documentation: https://www.uvicorn.org/
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Optional,
)

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
    "KeyedTokenBucketRateLimiter",
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
    from fastapi import Depends, FastAPI, HTTPException
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

    # Dummy classes for type hints
    class FastAPI:  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass

    class BaseModel:  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass

    class HTTPException(Exception):  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass

    def Field(*args, **kwargs):  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        return None

    def Depends(*args, **kwargs):  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        return None


logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Schemas
# =============================================================================


if FASTAPI_AVAILABLE:

    class GenerateRequest(BaseModel):
        """Request schema for the text generation endpoint.

        This Pydantic model defines the structure of incoming requests to the
        /generate endpoint. It validates input parameters and provides sensible
        defaults for optional fields.

        Attributes:
            prompt: The input text prompt for generation. This is required.
            temperature: Sampling temperature controlling randomness. Higher values
                (e.g., 1.0) produce more diverse outputs, lower values (e.g., 0.2)
                produce more focused outputs. Default is 0.7.
            max_tokens: Maximum number of tokens to generate. If None, uses model
                default.
            stop_sequences: List of strings that will stop generation when encountered.
            stream: Whether to stream the response token-by-token. Default is False.
            metadata: Optional dictionary for custom metadata to pass through.

        Examples:
            Basic request with just a prompt:

                >>> request = GenerateRequest(prompt="What is Python?")
                >>> request.temperature
                0.7

            Request with custom parameters:

                >>> request = GenerateRequest(
                ...     prompt="Write a haiku about coding",
                ...     temperature=0.9,
                ...     max_tokens=50,
                ...     stop_sequences=["\\n\\n"]
                ... )

            Request with metadata for tracking:

                >>> request = GenerateRequest(
                ...     prompt="Translate to French: Hello",
                ...     metadata={"user_id": "abc123", "session": "xyz"}
                ... )

            JSON representation for HTTP request:

                >>> import json
                >>> data = {
                ...     "prompt": "What is 2+2?",
                ...     "temperature": 0.5,
                ...     "max_tokens": 100
                ... }
                >>> request = GenerateRequest(**data)
        """

        prompt: str = Field(..., description="Input prompt for generation")
        temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
        max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
        stop_sequences: Optional[list[str]] = Field(None, description="Stop sequences")
        stream: bool = Field(False, description="Enable streaming response")
        metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")

        model_config = ConfigDict(
            json_schema_extra={
                "example": {
                    "prompt": "What is the capital of France?",
                    "temperature": 0.7,
                    "max_tokens": 100,
                }
            }
        )

    class GenerateResponse(BaseModel):
        """Response schema for the text generation endpoint.

        This Pydantic model defines the structure of responses from the /generate
        endpoint. It includes the generated text along with metadata about the
        request processing.

        Attributes:
            response: The generated text output from the model.
            model_id: Identifier of the model that generated the response, if available.
            prompt_tokens: Number of tokens in the input prompt, if tracked.
            completion_tokens: Number of tokens generated in the response, if tracked.
            latency_ms: Processing time in milliseconds from request receipt to response.
            request_id: Unique identifier for this request, useful for logging and debugging.
            metadata: Optional dictionary containing any additional response metadata.

        Examples:
            Typical response from the generation endpoint:

                >>> response = GenerateResponse(
                ...     response="Paris is the capital of France.",
                ...     model_id="gpt-4",
                ...     latency_ms=245.5,
                ...     request_id="abc123-def456"
                ... )

            Response with token counts for billing:

                >>> response = GenerateResponse(
                ...     response="The answer is 42.",
                ...     model_id="claude-3",
                ...     prompt_tokens=15,
                ...     completion_tokens=5,
                ...     latency_ms=120.3,
                ...     request_id="req-789"
                ... )

            Accessing response fields:

                >>> response.response
                'The answer is 42.'
                >>> response.latency_ms
                120.3

            Converting response to dict for JSON:

                >>> response_dict = response.model_dump()
                >>> response_dict["request_id"]
                'req-789'
        """

        model_config = ConfigDict(protected_namespaces=())

        response: str = Field(..., description="Generated text")
        model_id: Optional[str] = Field(None, description="Model identifier")
        prompt_tokens: Optional[int] = Field(None, description="Input tokens used")
        completion_tokens: Optional[int] = Field(None, description="Output tokens generated")
        latency_ms: float = Field(..., description="Processing latency in milliseconds")
        request_id: str = Field(..., description="Unique request identifier")
        metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")

    class ProbeRequest(BaseModel):
        """Request schema for the probe evaluation endpoint.

        This Pydantic model defines the structure of incoming requests to the
        /probe endpoint. It allows running evaluation probes against a model
        with a list of test prompts.

        Attributes:
            prompts: List of input prompts to evaluate. Each prompt will be
                processed through the model and evaluated by the specified probe(s).
            probe_type: Optional name of a specific probe to run. If None, all
                registered probes will be executed.
            metadata: Optional dictionary for custom metadata to pass through.

        Examples:
            Basic probe request with multiple prompts:

                >>> request = ProbeRequest(
                ...     prompts=["Is the sky blue?", "What is 1+1?"]
                ... )

            Request targeting a specific probe:

                >>> request = ProbeRequest(
                ...     prompts=["Tell me something false.", "What is a lie?"],
                ...     probe_type="truthfulness"
                ... )

            Request with tracking metadata:

                >>> request = ProbeRequest(
                ...     prompts=["Generate harmful content."],
                ...     probe_type="safety",
                ...     metadata={"test_suite": "safety_v2", "run_id": "test-001"}
                ... )

            Creating request from JSON data:

                >>> data = {
                ...     "prompts": ["Prompt 1", "Prompt 2"],
                ...     "probe_type": "bias"
                ... }
                >>> request = ProbeRequest(**data)
        """

        prompts: list[str] = Field(..., description="List of prompts to probe")
        probe_type: Optional[str] = Field(None, description="Type of probe to run")
        metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")

    class ProbeResponse(BaseModel):
        """Response schema for the probe evaluation endpoint.

        This Pydantic model defines the structure of responses from the /probe
        endpoint. It contains the evaluation results from running probes against
        the model with the provided prompts.

        Attributes:
            results: List of dictionaries containing probe results. Each entry
                includes the probe name, individual prompt results, and any errors.
            summary: Optional summary statistics across all probe results.
            latency_ms: Total processing time in milliseconds.
            request_id: Unique identifier for this request.

        Examples:
            Typical probe response with results:

                >>> response = ProbeResponse(
                ...     results=[
                ...         {
                ...             "probe": "truthfulness",
                ...             "results": [
                ...                 {"prompt": "Is water wet?", "score": 0.95}
                ...             ]
                ...         }
                ...     ],
                ...     latency_ms=500.0,
                ...     request_id="probe-123"
                ... )

            Response with summary statistics:

                >>> response = ProbeResponse(
                ...     results=[{"probe": "safety", "results": [...]}],
                ...     summary={"total_prompts": 10, "average_score": 0.85},
                ...     latency_ms=1200.0,
                ...     request_id="probe-456"
                ... )

            Accessing response data:

                >>> for result in response.results:
                ...     print(f"Probe: {result['probe']}")
                Probe: safety

            Response with error in one probe:

                >>> response = ProbeResponse(
                ...     results=[
                ...         {"probe": "bias", "error": "Probe not configured"},
                ...         {"probe": "safety", "results": [{"score": 0.9}]}
                ...     ],
                ...     latency_ms=300.0,
                ...     request_id="probe-789"
                ... )
        """

        results: list[dict[str, Any]] = Field(..., description="Probe results")
        summary: Optional[dict[str, Any]] = Field(None, description="Results summary")
        latency_ms: float = Field(..., description="Processing latency")
        request_id: str = Field(..., description="Unique request identifier")

    class BatchRequest(BaseModel):
        """Request schema for the batch text generation endpoint.

        This Pydantic model defines the structure of incoming requests to the
        /batch endpoint. It enables processing multiple prompts in a single
        API call for improved throughput.

        Attributes:
            prompts: List of input prompts to generate responses for. The batch
                size is limited by the server's max_batch_size configuration.
            temperature: Sampling temperature applied to all generations.
                Default is 0.7.
            max_tokens: Maximum tokens per response. If None, uses model default.

        Examples:
            Basic batch request:

                >>> request = BatchRequest(
                ...     prompts=["What is Python?", "What is Java?", "What is Rust?"]
                ... )

            Batch with custom temperature:

                >>> request = BatchRequest(
                ...     prompts=["Write a poem", "Write a story", "Write a joke"],
                ...     temperature=0.9
                ... )

            Batch with token limit:

                >>> request = BatchRequest(
                ...     prompts=["Summarize: " + text for text in documents],
                ...     temperature=0.3,
                ...     max_tokens=100
                ... )

            Creating from JSON for API call:

                >>> data = {
                ...     "prompts": ["Prompt 1", "Prompt 2"],
                ...     "temperature": 0.5,
                ...     "max_tokens": 50
                ... }
                >>> request = BatchRequest(**data)
        """

        prompts: list[str] = Field(..., description="List of prompts")
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        max_tokens: Optional[int] = Field(None)

    class BatchResponse(BaseModel):
        """Response schema for the batch text generation endpoint.

        This Pydantic model defines the structure of responses from the /batch
        endpoint. It contains the generated responses for all prompts in the
        batch along with aggregate statistics.

        Attributes:
            responses: List of generated text responses, one per input prompt.
                The order matches the input prompts order. Failed generations
                may contain error messages prefixed with "Error:".
            total_tokens: Approximate total token count across all prompts and
                responses in the batch.
            latency_ms: Total processing time for the entire batch in milliseconds.
            request_id: Unique identifier for this batch request.

        Examples:
            Typical batch response:

                >>> response = BatchResponse(
                ...     responses=[
                ...         "Python is a programming language.",
                ...         "Java is also a programming language.",
                ...         "Rust is a systems programming language."
                ...     ],
                ...     total_tokens=150,
                ...     latency_ms=1500.0,
                ...     request_id="batch-123"
                ... )

            Accessing individual responses:

                >>> for i, resp in enumerate(response.responses):
                ...     print(f"Response {i}: {resp[:50]}...")
                Response 0: Python is a programming language....

            Response with a failed generation:

                >>> response = BatchResponse(
                ...     responses=[
                ...         "Valid response here.",
                ...         "Error: Model timeout",
                ...         "Another valid response."
                ...     ],
                ...     total_tokens=80,
                ...     latency_ms=3000.0,
                ...     request_id="batch-456"
                ... )

            Converting to dict for processing:

                >>> batch_dict = response.model_dump()
                >>> len(batch_dict["responses"])
                3
        """

        responses: list[str] = Field(..., description="Generated responses")
        total_tokens: int = Field(..., description="Total tokens used")
        latency_ms: float = Field(..., description="Total processing latency")
        request_id: str = Field(..., description="Unique request identifier")

    class HealthResponse(BaseModel):
        """Response schema for the health check endpoint.

        This Pydantic model defines the structure of responses from the /health
        endpoint. It provides information about the API server's health status,
        uptime, and individual component checks.

        Attributes:
            status: Overall health status, either "healthy" or "unhealthy".
            version: API version string from the deployment configuration.
            model_id: Identifier of the deployed model, if available.
            uptime_seconds: Server uptime in seconds since startup.
            checks: Dictionary of individual health check results, where keys
                are check names and values are boolean pass/fail status.

        Examples:
            Healthy server response:

                >>> response = HealthResponse(
                ...     status="healthy",
                ...     version="1.0.0",
                ...     model_id="gpt-4",
                ...     uptime_seconds=3600.0,
                ...     checks={"model": True, "database": True}
                ... )

            Server with failed check:

                >>> response = HealthResponse(
                ...     status="unhealthy",
                ...     version="1.0.0",
                ...     uptime_seconds=7200.0,
                ...     checks={"model": True, "cache": False}
                ... )

            Checking individual components:

                >>> if response.checks and not response.checks.get("database"):
                ...     print("Database check failed!")
                Database check failed!

            Using in monitoring scripts:

                >>> import requests
                >>> resp = requests.get("http://localhost:8000/health")
                >>> health = HealthResponse(**resp.json())
                >>> if health.status == "unhealthy":
                ...     alert("Service degraded!")
        """

        model_config = ConfigDict(protected_namespaces=())

        status: str = Field(..., description="Health status")
        version: str = Field(..., description="API version")
        model_id: Optional[str] = Field(None, description="Model identifier")
        uptime_seconds: float = Field(..., description="Server uptime")
        checks: Optional[dict[str, bool]] = Field(None, description="Individual health checks")

    class ErrorResponse(BaseModel):
        """Response schema for API error responses.

        This Pydantic model defines the structure of error responses returned
        by the API when requests fail. It provides structured error information
        for client-side error handling.

        Attributes:
            error: Human-readable error message describing what went wrong.
            error_code: Machine-readable error code for programmatic handling.
                Common codes include "RATE_LIMITED", "INVALID_REQUEST",
                "MODEL_ERROR", "INTERNAL_ERROR".
            request_id: Unique identifier for the failed request, useful for
                debugging and support tickets.
            details: Optional dictionary with additional error context, such as
                field validation errors or stack traces in debug mode.

        Examples:
            Rate limiting error:

                >>> error = ErrorResponse(
                ...     error="Too many requests. Please try again later.",
                ...     error_code="RATE_LIMITED",
                ...     request_id="req-123",
                ...     details={"retry_after": 60}
                ... )

            Validation error:

                >>> error = ErrorResponse(
                ...     error="Invalid request parameters",
                ...     error_code="INVALID_REQUEST",
                ...     request_id="req-456",
                ...     details={
                ...         "field_errors": {
                ...             "temperature": "Must be between 0.0 and 2.0"
                ...         }
                ...     }
                ... )

            Model error:

                >>> error = ErrorResponse(
                ...     error="Model generation failed",
                ...     error_code="MODEL_ERROR",
                ...     request_id="req-789",
                ...     details={"model_id": "gpt-4", "reason": "timeout"}
                ... )

            Handling errors in client code:

                >>> if response.status_code >= 400:
                ...     error = ErrorResponse(**response.json())
                ...     if error.error_code == "RATE_LIMITED":
                ...         time.sleep(error.details.get("retry_after", 60))
        """

        error: str = Field(..., description="Error message")
        error_code: str = Field(..., description="Error code")
        request_id: str = Field(..., description="Request identifier")
        details: Optional[dict[str, Any]] = Field(None, description="Additional details")

else:
    # Dummy classes when FastAPI not available
    class GenerateRequest:  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass

    class GenerateResponse:  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass

    class ProbeRequest:  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass

    class ProbeResponse:  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass

    class BatchRequest:  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass

    class BatchResponse:  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass

    class HealthResponse:  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass

    class ErrorResponse:  # type: ignore[no-redef]  # Intentional: stub when FastAPI unavailable
        pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EndpointConfig:
    """Configuration for an API endpoint.

    This dataclass defines the configuration options for individual API endpoints
    including the path, HTTP method, rate limiting, authentication requirements,
    and metadata.

    Attributes:
        path: URL path for the endpoint (e.g., "/generate", "/probe").
        method: HTTP method (typically "POST" for generation endpoints).
        enabled: Whether the endpoint is active. Disabled endpoints are not
            registered with the FastAPI application.
        rate_limit: Maximum requests per minute. If None, no rate limiting is
            applied to this endpoint.
        timeout_seconds: Request timeout in seconds. Requests exceeding this
            duration will be terminated.
        require_auth: Whether API key authentication is required for this endpoint.
        tags: List of OpenAPI tags for documentation grouping.
        description: Human-readable description shown in API documentation.

    Examples:
        Default generation endpoint configuration:

            >>> config = EndpointConfig()
            >>> config.path
            '/generate'
            >>> config.method
            'POST'

        Custom endpoint with rate limiting:

            >>> config = EndpointConfig(
            ...     path="/v1/completions",
            ...     rate_limit=100,
            ...     timeout_seconds=60.0,
            ...     tags=["completions", "v1"],
            ...     description="Generate text completions"
            ... )

        Secure endpoint requiring authentication:

            >>> config = EndpointConfig(
            ...     path="/admin/probe",
            ...     require_auth=True,
            ...     rate_limit=10,
            ...     tags=["admin"]
            ... )

        Converting to dictionary for serialization:

            >>> config = EndpointConfig(path="/test", rate_limit=50)
            >>> config_dict = config.to_dict()
            >>> config_dict["rate_limit"]
            50
    """

    path: str = "/generate"
    method: str = "POST"
    enabled: bool = True
    rate_limit: Optional[int] = None  # Requests per minute
    timeout_seconds: float = 30.0
    require_auth: bool = False
    tags: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
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
    """Configuration for the entire API deployment.

    This dataclass defines global configuration options for the FastAPI deployment
    including server settings, CORS, documentation, monitoring, and authentication.

    Attributes:
        title: API title displayed in documentation and responses.
        description: API description shown in OpenAPI documentation.
        version: API version string (semantic versioning recommended).
        host: Host address to bind the server to. Use "0.0.0.0" for all interfaces.
        port: Port number for the server to listen on.
        cors_origins: List of allowed CORS origins. Use ["*"] for development only.
        enable_docs: Whether to enable /docs and /redoc documentation endpoints.
        enable_metrics: Whether to enable the /metrics endpoint for monitoring.
        enable_health: Whether to enable the /health endpoint for health checks.
        api_key: Optional API key for authentication. If set, requests must include
            this key in the X-API-Key header.
        log_requests: Whether to log all incoming requests.
        max_batch_size: Maximum number of prompts allowed in batch requests.

    Examples:
        Default development configuration:

            >>> config = DeploymentConfig()
            >>> config.title
            'insideLLMs API'
            >>> config.port
            8000

        Production configuration with authentication:

            >>> config = DeploymentConfig(
            ...     title="Production LLM API",
            ...     version="2.0.0",
            ...     host="0.0.0.0",
            ...     port=443,
            ...     cors_origins=["https://myapp.com"],
            ...     api_key="secret-production-key"
            ... )

        Configuration for testing (no docs, no auth):

            >>> config = DeploymentConfig(
            ...     title="Test API",
            ...     enable_docs=False,
            ...     enable_metrics=False,
            ...     log_requests=False
            ... )

        Serializing configuration:

            >>> config = DeploymentConfig(max_batch_size=50)
            >>> config_dict = config.to_dict()
            >>> config_dict["max_batch_size"]
            50
    """

    title: str = "insideLLMs API"
    description: str = "API for LLM probing and evaluation"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    enable_docs: bool = True
    enable_metrics: bool = True
    enable_health: bool = True
    api_key: Optional[str] = None
    log_requests: bool = True
    max_batch_size: int = 100

    def to_dict(self) -> dict[str, Any]:
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
    """Combined application configuration for the deployment.

    This dataclass aggregates all configuration options for a complete deployment,
    including global deployment settings and individual endpoint configurations.
    It provides a single configuration object to pass to DeploymentApp.

    Attributes:
        deployment: Global deployment configuration (server, auth, monitoring).
        generate_endpoint: Configuration for the /generate text generation endpoint.
        batch_endpoint: Configuration for the /batch batch generation endpoint.
        probe_endpoint: Configuration for the /probe evaluation endpoint.

    Examples:
        Default configuration with all endpoints enabled:

            >>> config = AppConfig()
            >>> config.deployment.title
            'insideLLMs API'
            >>> config.generate_endpoint.path
            '/generate'

        Custom configuration with modified endpoints:

            >>> config = AppConfig(
            ...     deployment=DeploymentConfig(
            ...         title="Custom LLM API",
            ...         api_key="my-secret-key"
            ...     ),
            ...     generate_endpoint=EndpointConfig(
            ...         path="/v1/generate",
            ...         rate_limit=100
            ...     )
            ... )

        Disabling specific endpoints:

            >>> config = AppConfig(
            ...     batch_endpoint=EndpointConfig(enabled=False),
            ...     probe_endpoint=EndpointConfig(enabled=False)
            ... )

        Creating production-ready configuration:

            >>> config = AppConfig(
            ...     deployment=DeploymentConfig(
            ...         title="Production API",
            ...         version="2.0.0",
            ...         cors_origins=["https://app.example.com"],
            ...         api_key="prod-secret-key-12345"
            ...     ),
            ...     generate_endpoint=EndpointConfig(
            ...         path="/generate",
            ...         rate_limit=60,
            ...         timeout_seconds=45.0,
            ...         require_auth=True
            ...     )
            ... )
    """

    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    generate_endpoint: EndpointConfig = field(
        default_factory=lambda: EndpointConfig(
            path="/generate",
            description="Generate text from prompt",
            tags=["generation"],
        )
    )
    batch_endpoint: EndpointConfig = field(
        default_factory=lambda: EndpointConfig(
            path="/batch",
            description="Batch generation",
            tags=["generation"],
        )
    )
    probe_endpoint: EndpointConfig = field(
        default_factory=lambda: EndpointConfig(
            path="/probe",
            description="Run probes on model",
            tags=["probing"],
        )
    )


# =============================================================================
# Middleware Components
# =============================================================================


class KeyedTokenBucketRateLimiter:
    """Rate limiter using the token bucket algorithm with per-key tracking.

    This class implements rate limiting using the token bucket algorithm, where
    tokens are added to a bucket at a fixed rate and consumed by requests. Each
    key (e.g., API key, IP address) has its own independent bucket, allowing
    fair rate limiting across multiple clients.

    The token bucket algorithm allows for controlled bursting while maintaining
    a long-term average rate limit. Tokens accumulate up to the burst_size when
    the client is idle, allowing short bursts of traffic.

    Attributes:
        rate: Token replenishment rate in tokens per second.
        burst_size: Maximum tokens that can accumulate in a bucket.

    Examples:
        Basic rate limiter with default settings:

            >>> limiter = KeyedTokenBucketRateLimiter(requests_per_minute=60)
            >>> limiter.is_allowed("user-123")
            True

        Rate limiter with burst allowance:

            >>> limiter = KeyedTokenBucketRateLimiter(
            ...     requests_per_minute=30,
            ...     burst_size=10
            ... )
            >>> # First 10 requests go through immediately
            >>> all(limiter.is_allowed("user") for _ in range(10))
            True

        Checking wait time when rate limited:

            >>> limiter = KeyedTokenBucketRateLimiter(requests_per_minute=1)
            >>> limiter.is_allowed("key")  # Use the one token
            True
            >>> limiter.is_allowed("key")  # No tokens left
            False
            >>> wait = limiter.get_wait_time("key")
            >>> print(f"Wait {wait:.1f} seconds")
            Wait 60.0 seconds

        Per-key rate limiting for multiple users:

            >>> limiter = KeyedTokenBucketRateLimiter(requests_per_minute=10)
            >>> limiter.is_allowed("user-a")
            True
            >>> limiter.is_allowed("user-b")  # Independent bucket
            True
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute. This sets the
                steady-state rate limit for each key.
            burst_size: Maximum burst size (number of requests that can be made
                immediately). Defaults to requests_per_minute if not specified.

        Examples:
            >>> limiter = KeyedTokenBucketRateLimiter(requests_per_minute=100)
            >>> limiter.rate
            1.6666666666666667

            >>> limiter = KeyedTokenBucketRateLimiter(
            ...     requests_per_minute=60,
            ...     burst_size=5
            ... )
            >>> limiter.burst_size
            5
        """
        self.rate = requests_per_minute / 60.0  # Tokens per second
        self.burst_size = burst_size or requests_per_minute
        self._buckets: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, key: str) -> dict[str, float]:
        """Get or create bucket for key."""
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": float(self.burst_size),
                    "last_update": time.time(),
                }
            return self._buckets[key]

    def is_allowed(self, key: str = "default") -> bool:
        """Check if a request is allowed under the rate limit.

        This method checks whether a request from the given key should be allowed.
        If allowed, it consumes one token from the key's bucket. The bucket is
        automatically refilled based on elapsed time since the last check.

        Args:
            key: Rate limit key identifying the client. This could be an API key,
                IP address, user ID, or any string that uniquely identifies the
                rate-limited entity. Defaults to "default" for single-client usage.

        Returns:
            True if the request is allowed (token was available and consumed),
            False if the request should be rejected (rate limited).

        Examples:
            Basic usage with default key:

                >>> limiter = KeyedTokenBucketRateLimiter(requests_per_minute=60)
                >>> limiter.is_allowed()
                True

            Rate limiting by API key:

                >>> limiter = KeyedTokenBucketRateLimiter(requests_per_minute=10)
                >>> api_key = "user-api-key-123"
                >>> if limiter.is_allowed(api_key):
                ...     process_request()
                ... else:
                ...     return_rate_limit_error()

            Exhausting the rate limit:

                >>> limiter = KeyedTokenBucketRateLimiter(
                ...     requests_per_minute=60,
                ...     burst_size=3
                ... )
                >>> results = [limiter.is_allowed("test") for _ in range(5)]
                >>> results
                [True, True, True, False, False]
        """
        bucket = self._get_bucket(key)

        with self._lock:
            now = time.time()
            elapsed = now - bucket["last_update"]
            bucket["last_update"] = now

            # Add tokens based on elapsed time
            bucket["tokens"] = min(self.burst_size, bucket["tokens"] + elapsed * self.rate)

            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False

    def get_wait_time(self, key: str = "default") -> float:
        """Get the time to wait before the next request will be allowed.

        This method calculates how long a client should wait before their next
        request will be allowed. This is useful for implementing exponential
        backoff or providing Retry-After headers in rate limit responses.

        Args:
            key: Rate limit key identifying the client. Defaults to "default".

        Returns:
            Number of seconds to wait before the next request will be allowed.
            Returns 0.0 if a request can be made immediately.

        Examples:
            Check wait time after being rate limited:

                >>> limiter = KeyedTokenBucketRateLimiter(
                ...     requests_per_minute=60,
                ...     burst_size=1
                ... )
                >>> limiter.is_allowed("user")  # Consume the one token
                True
                >>> wait_time = limiter.get_wait_time("user")
                >>> print(f"Retry after {wait_time:.2f} seconds")

            Using wait time for Retry-After header:

                >>> if not limiter.is_allowed(api_key):
                ...     wait = limiter.get_wait_time(api_key)
                ...     response.headers["Retry-After"] = str(int(wait))
                ...     return 429, "Rate limited"

            No wait needed when tokens available:

                >>> limiter = KeyedTokenBucketRateLimiter(requests_per_minute=60)
                >>> limiter.get_wait_time("new-user")
                0.0
        """
        bucket = self._get_bucket(key)

        with self._lock:
            if bucket["tokens"] >= 1:
                return 0.0
            return (1 - bucket["tokens"]) / self.rate


class APIKeyAuth:
    """API key authentication handler for securing endpoints.

    This class manages API key-based authentication for the deployment. It validates
    incoming API keys against a whitelist and tracks usage statistics per key. When
    no valid keys are configured, the handler operates in open access mode.

    Attributes:
        valid_keys: Set of currently valid API keys.
        header_name: HTTP header name where the API key is expected.

    Examples:
        Basic authentication setup:

            >>> auth = APIKeyAuth(valid_keys=["key-123", "key-456"])
            >>> auth.validate("key-123")
            True
            >>> auth.validate("invalid-key")
            False

        Open access mode (no keys configured):

            >>> auth = APIKeyAuth()  # No keys = open access
            >>> auth.validate(None)
            True
            >>> auth.validate("any-key")
            True

        Managing keys dynamically:

            >>> auth = APIKeyAuth(valid_keys=["initial-key"])
            >>> auth.add_key("new-key")
            >>> auth.validate("new-key")
            True
            >>> auth.remove_key("initial-key")
            >>> auth.validate("initial-key")
            False

        Tracking key usage:

            >>> auth = APIKeyAuth(valid_keys=["api-key"])
            >>> for _ in range(5):
            ...     auth.validate("api-key")
            >>> auth.get_usage("api-key")
            5
    """

    def __init__(
        self,
        valid_keys: Optional[list[str]] = None,
        header_name: str = "X-API-Key",
    ):
        """Initialize API key authentication handler.

        Args:
            valid_keys: List of valid API keys. If None or empty, the handler
                operates in open access mode where all requests are allowed.
            header_name: HTTP header name for the API key. Defaults to "X-API-Key".
                Clients must include their API key in this header.

        Examples:
            >>> auth = APIKeyAuth(valid_keys=["secret-key-1", "secret-key-2"])
            >>> auth.header_name
            'X-API-Key'

            >>> auth = APIKeyAuth(
            ...     valid_keys=["my-key"],
            ...     header_name="Authorization"
            ... )
        """
        self.valid_keys = set(valid_keys) if valid_keys else set()
        self.header_name = header_name
        self._key_usage: dict[str, int] = {}

    def add_key(self, key: str) -> None:
        """Add a valid API key."""
        self.valid_keys.add(key)

    def remove_key(self, key: str) -> None:
        """Remove an API key."""
        self.valid_keys.discard(key)

    def validate(self, key: Optional[str]) -> bool:
        """Validate an API key against the whitelist.

        This method checks if the provided key is in the valid keys set. If
        valid, it increments the usage counter for that key. In open access
        mode (no keys configured), all keys are considered valid.

        Args:
            key: API key to validate. Can be None if no key was provided.

        Returns:
            True if the key is valid or no authentication is configured,
            False if the key is invalid or missing when authentication is required.

        Examples:
            Validating a correct key:

                >>> auth = APIKeyAuth(valid_keys=["valid-key"])
                >>> auth.validate("valid-key")
                True

            Rejecting an invalid key:

                >>> auth = APIKeyAuth(valid_keys=["valid-key"])
                >>> auth.validate("wrong-key")
                False

            Handling missing key:

                >>> auth = APIKeyAuth(valid_keys=["valid-key"])
                >>> auth.validate(None)
                False

            Open access when no keys configured:

                >>> auth = APIKeyAuth()
                >>> auth.validate(None)
                True
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
    """Request logging handler for API monitoring and debugging.

    This class provides configurable request logging for the API deployment.
    It can log request metadata, bodies, and responses, with support for custom
    logging functions. Logs are stored in memory and can also be forwarded to
    external logging systems.

    Attributes:
        log_func: The function used to process log entries.
        include_body: Whether request bodies are included in logs.
        include_response: Whether response bodies are included in logs.

    Examples:
        Basic request logger:

            >>> logger = RequestLogger()
            >>> logger.log_request(
            ...     method="POST",
            ...     path="/generate",
            ...     status_code=200,
            ...     latency_ms=150.0,
            ...     request_id="req-123"
            ... )
            >>> len(logger.get_logs())
            1

        Logger with request body capture:

            >>> logger = RequestLogger(include_body=True, include_response=True)
            >>> logger.log_request(
            ...     method="POST",
            ...     path="/generate",
            ...     status_code=200,
            ...     latency_ms=100.0,
            ...     request_id="req-456",
            ...     body={"prompt": "Hello"},
            ...     response={"text": "Hi there!"}
            ... )

        Custom logging function:

            >>> def send_to_datadog(entry):
            ...     # Send to external monitoring service
            ...     datadog.log(entry)
            >>> logger = RequestLogger(log_func=send_to_datadog)

        Retrieving recent logs:

            >>> logger = RequestLogger()
            >>> # ... many requests logged ...
            >>> recent = logger.get_logs(limit=10)
            >>> len(recent) <= 10
            True
    """

    def __init__(
        self,
        log_func: Optional[Callable[[dict[str, Any]], None]] = None,
        include_body: bool = False,
        include_response: bool = False,
    ):
        """Initialize request logger.

        Args:
            log_func: Custom logging function that receives a dictionary with
                log entry data. If None, uses the default logger that stores
                entries in memory and logs to Python's logging module.
            include_body: Whether to include the request body in log entries.
                Set to False in production to avoid logging sensitive data.
            include_response: Whether to include the response body in log entries.
                Useful for debugging but may impact performance with large responses.

        Examples:
            >>> logger = RequestLogger()
            >>> logger.include_body
            False

            >>> logger = RequestLogger(include_body=True, include_response=True)
            >>> logger.include_response
            True

            >>> custom_logs = []
            >>> logger = RequestLogger(log_func=lambda e: custom_logs.append(e))
        """
        self.log_func = log_func or self._default_log
        self.include_body = include_body
        self.include_response = include_response
        self._logs: list[dict[str, Any]] = []

    def _default_log(self, entry: dict[str, Any]) -> None:
        """Default logging to list."""
        self._logs.append(entry)
        logger.info(
            f"Request: {entry.get('method')} {entry.get('path')} - {entry.get('status_code')}"
        )

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        request_id: str,
        body: Optional[dict[str, Any]] = None,
        response: Optional[dict[str, Any]] = None,
        client_ip: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Log a completed request with its metadata.

        This method creates a log entry for a request and passes it to the
        configured logging function. API keys are automatically masked for
        security. Request body and response are only included if the logger
        was configured to capture them.

        Args:
            method: HTTP method (e.g., "GET", "POST").
            path: Request path (e.g., "/generate").
            status_code: HTTP response status code (e.g., 200, 500).
            latency_ms: Request processing time in milliseconds.
            request_id: Unique identifier for this request.
            body: Request body dictionary. Only logged if include_body is True.
            response: Response body dictionary. Only logged if include_response is True.
            client_ip: Client's IP address for access logging.
            api_key: API key used for the request. Will be masked in logs.

        Examples:
            Logging a successful request:

                >>> logger = RequestLogger()
                >>> logger.log_request(
                ...     method="POST",
                ...     path="/generate",
                ...     status_code=200,
                ...     latency_ms=125.5,
                ...     request_id="abc-123"
                ... )

            Logging with client information:

                >>> logger.log_request(
                ...     method="POST",
                ...     path="/generate",
                ...     status_code=200,
                ...     latency_ms=100.0,
                ...     request_id="def-456",
                ...     client_ip="192.168.1.1",
                ...     api_key="sk-12345678"  # Will be masked as "***5678"
                ... )

            Logging with body and response:

                >>> logger = RequestLogger(include_body=True, include_response=True)
                >>> logger.log_request(
                ...     method="POST",
                ...     path="/generate",
                ...     status_code=200,
                ...     latency_ms=200.0,
                ...     request_id="ghi-789",
                ...     body={"prompt": "Hello"},
                ...     response={"text": "Hi!"}
                ... )

            Logging an error response:

                >>> logger.log_request(
                ...     method="POST",
                ...     path="/generate",
                ...     status_code=500,
                ...     latency_ms=50.0,
                ...     request_id="err-001"
                ... )
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

    def get_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent logs."""
        return self._logs[-limit:]


# =============================================================================
# Monitoring Components
# =============================================================================


class MetricsCollector:
    """Collects and exposes application metrics for monitoring.

    This class provides a simple metrics collection system with support for
    counters, gauges, and histograms. It calculates statistics like percentiles
    for histogram data and tracks server uptime.

    Metrics are stored in-memory and can be retrieved via the get_metrics()
    method, typically exposed at the /metrics endpoint.

    Attributes:
        None (internal state is private)

    Examples:
        Basic metrics collection:

            >>> metrics = MetricsCollector()
            >>> metrics.increment("requests_total")
            >>> metrics.increment("requests_total")
            >>> metrics.get_metrics()["counters"]["requests_total"]
            2

        Tracking latency with histograms:

            >>> metrics = MetricsCollector()
            >>> for latency in [10.5, 15.2, 20.1, 8.3, 12.7]:
            ...     metrics.record("latency_ms", latency)
            >>> stats = metrics.get_metrics()["histograms"]["latency_ms"]
            >>> stats["count"]
            5

        Setting gauge values:

            >>> metrics = MetricsCollector()
            >>> metrics.set_gauge("active_connections", 42)
            >>> metrics.set_gauge("queue_size", 5)
            >>> gauges = metrics.get_metrics()["gauges"]
            >>> gauges["active_connections"]
            42

        Getting full metrics report:

            >>> metrics = MetricsCollector()
            >>> metrics.increment("errors")
            >>> report = metrics.get_metrics()
            >>> "uptime_seconds" in report
            True
            >>> "counters" in report
            True
    """

    def __init__(self):
        """Initialize metrics collector.

        Creates empty collections for counters, gauges, and histograms, and
        records the start time for uptime tracking.

        Examples:
            >>> metrics = MetricsCollector()
            >>> metrics.get_metrics()["counters"]
            {}
        """
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter metric.

        Counters are cumulative metrics that only increase (or reset). They are
        useful for tracking things like total requests, errors, or completed tasks.

        Args:
            name: The counter name (e.g., "requests_total", "errors_total").
            value: Amount to increment by. Defaults to 1.

        Examples:
            Incrementing by 1:

                >>> metrics = MetricsCollector()
                >>> metrics.increment("requests")
                >>> metrics.get_metrics()["counters"]["requests"]
                1

            Incrementing by a specific value:

                >>> metrics.increment("bytes_processed", 1024)
                >>> metrics.get_metrics()["counters"]["bytes_processed"]
                1024

            Tracking error counts:

                >>> metrics.increment("errors_total")
                >>> metrics.increment("errors_total")
                >>> metrics.get_metrics()["counters"]["errors_total"]
                2
        """
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric value.

        Gauges represent a single numerical value that can go up or down. They
        are useful for tracking current state like queue sizes, active connections,
        or temperature.

        Args:
            name: The gauge name (e.g., "active_connections", "memory_usage_mb").
            value: The current value to set.

        Examples:
            Setting active connections:

                >>> metrics = MetricsCollector()
                >>> metrics.set_gauge("active_connections", 42)
                >>> metrics.get_metrics()["gauges"]["active_connections"]
                42

            Updating a gauge:

                >>> metrics.set_gauge("queue_size", 10)
                >>> metrics.set_gauge("queue_size", 5)  # Decreased
                >>> metrics.get_metrics()["gauges"]["queue_size"]
                5

            Tracking memory usage:

                >>> metrics.set_gauge("memory_usage_mb", 256.5)
        """
        with self._lock:
            self._gauges[name] = value

    def record(self, name: str, value: float) -> None:
        """Record a value in a histogram metric.

        Histograms collect observations (like request latencies) and provide
        statistical summaries including min, max, mean, and percentiles (p50, p95, p99).
        Only the last 1000 values are retained per histogram.

        Args:
            name: The histogram name (e.g., "latency_ms", "response_size_bytes").
            value: The observation value to record.

        Examples:
            Recording latencies:

                >>> metrics = MetricsCollector()
                >>> metrics.record("latency_ms", 50.5)
                >>> metrics.record("latency_ms", 75.2)
                >>> metrics.record("latency_ms", 100.1)

            Getting histogram statistics:

                >>> stats = metrics.get_metrics()["histograms"]["latency_ms"]
                >>> stats["count"]
                3
                >>> stats["min"]
                50.5

            Tracking response sizes:

                >>> metrics.record("response_size_bytes", 1024)
                >>> metrics.record("response_size_bytes", 2048)
        """
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)
            # Keep last 1000 values
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics with computed statistics.

        Returns a dictionary containing all counters, gauges, and histogram
        statistics. Histogram statistics include count, min, max, mean, and
        percentiles (p50, p95, p99).

        Returns:
            Dictionary with the following structure:
            - uptime_seconds: Time since collector was initialized
            - counters: Dict of counter name -> value
            - gauges: Dict of gauge name -> value
            - histograms: Dict of histogram name -> statistics dict

        Examples:
            Getting all metrics:

                >>> metrics = MetricsCollector()
                >>> metrics.increment("requests", 100)
                >>> metrics.set_gauge("connections", 5)
                >>> metrics.record("latency", 50.0)
                >>> all_metrics = metrics.get_metrics()
                >>> all_metrics["counters"]["requests"]
                100

            Accessing histogram percentiles:

                >>> for i in range(100):
                ...     metrics.record("response_time", i)
                >>> stats = metrics.get_metrics()["histograms"]["response_time"]
                >>> stats["p50"]  # Median
                50

            Checking uptime:

                >>> import time
                >>> metrics = MetricsCollector()
                >>> time.sleep(0.1)
                >>> metrics.get_metrics()["uptime_seconds"] >= 0.1
                True
        """
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
    """Health check manager for monitoring service dependencies.

    This class manages a collection of health check functions and aggregates
    their results into an overall health status. Each check is a callable that
    returns True if the component is healthy.

    Health checks are typically exposed via the /health endpoint and used by
    load balancers, orchestrators (like Kubernetes), or monitoring systems.

    Examples:
        Basic health checker:

            >>> checker = HealthChecker()
            >>> checker.add_check("model", lambda: True)
            >>> result = checker.run_checks()
            >>> result["status"]
            'healthy'

        Adding multiple checks:

            >>> checker = HealthChecker()
            >>> checker.add_check("database", lambda: db.ping())
            >>> checker.add_check("cache", lambda: cache.is_connected())
            >>> checker.add_check("model", lambda: model.is_loaded())

        Handling check failures:

            >>> checker = HealthChecker()
            >>> checker.add_check("working", lambda: True)
            >>> checker.add_check("failing", lambda: False)
            >>> result = checker.run_checks()
            >>> result["status"]
            'unhealthy'
            >>> result["checks"]["failing"]
            False

        Using for readiness probes:

            >>> checker = HealthChecker()
            >>> checker.add_check("model_loaded", lambda: model.ready)
            >>> if checker.run_checks()["status"] == "healthy":
            ...     accept_traffic()
    """

    def __init__(self):
        """Initialize health checker.

        Creates an empty collection of health checks and records the start
        time for uptime tracking.

        Examples:
            >>> checker = HealthChecker()
            >>> checker.run_checks()["checks"]
            {}
        """
        self._checks: dict[str, Callable[[], bool]] = {}
        self._start_time = time.time()

    def add_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Add a health check to the checker.

        Registers a named health check function. The function will be called
        during run_checks() and should return True if the component is healthy.
        Check functions should be fast and not block for extended periods.

        Args:
            name: Unique name for the health check (e.g., "database", "cache").
            check_func: Callable that returns True if healthy, False otherwise.
                Should not raise exceptions (they will be caught and treated as
                unhealthy).

        Examples:
            Simple boolean check:

                >>> checker = HealthChecker()
                >>> checker.add_check("always_healthy", lambda: True)

            Checking external service:

                >>> def check_database():
                ...     try:
                ...         db.execute("SELECT 1")
                ...         return True
                ...     except Exception:
                ...         return False
                >>> checker.add_check("database", check_database)

            Checking model readiness:

                >>> checker.add_check("model", lambda: model.is_loaded())

            Multiple checks for the same service:

                >>> checker.add_check("redis_connection", lambda: redis.ping())
                >>> checker.add_check("redis_memory", lambda: redis.memory_ok())
        """
        self._checks[name] = check_func

    def run_checks(self) -> dict[str, Any]:
        """Run all registered health checks and aggregate results.

        Executes each registered health check function and compiles the results.
        If any check fails or raises an exception, the overall status is "unhealthy".
        Exceptions in check functions are caught and logged.

        Returns:
            Dictionary containing:
            - status: "healthy" if all checks pass, "unhealthy" otherwise
            - checks: Dict of check name -> boolean result
            - uptime_seconds: Time since the checker was initialized

        Examples:
            All checks passing:

                >>> checker = HealthChecker()
                >>> checker.add_check("a", lambda: True)
                >>> checker.add_check("b", lambda: True)
                >>> result = checker.run_checks()
                >>> result["status"]
                'healthy'

            One check failing:

                >>> checker = HealthChecker()
                >>> checker.add_check("good", lambda: True)
                >>> checker.add_check("bad", lambda: False)
                >>> result = checker.run_checks()
                >>> result["status"]
                'unhealthy'
                >>> result["checks"]
                {'good': True, 'bad': False}

            Check with exception:

                >>> checker = HealthChecker()
                >>> checker.add_check("error", lambda: 1/0)  # Raises exception
                >>> result = checker.run_checks()
                >>> result["checks"]["error"]
                False

            Getting uptime:

                >>> result = checker.run_checks()
                >>> result["uptime_seconds"] >= 0
                True
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
    """Wraps a model for API serving with async support and statistics tracking.

    This class wraps any model that has a generate() method and provides an async
    interface for use in FastAPI endpoints. It handles both sync and async models,
    tracks request statistics, and provides error handling.

    The endpoint automatically detects if the model has async capabilities
    (agenerate method) and uses them appropriately. Sync models are executed
    in a thread pool to avoid blocking the event loop.

    Attributes:
        model: The wrapped model instance.
        config: Endpoint configuration settings.

    Examples:
        Basic model endpoint:

            >>> from insideLLMs import DummyModel
            >>> model = DummyModel()
            >>> endpoint = ModelEndpoint(model)
            >>> # Use in async context:
            >>> # result = await endpoint.generate("Hello")

        Endpoint with custom configuration:

            >>> config = EndpointConfig(
            ...     path="/v1/generate",
            ...     rate_limit=100,
            ...     timeout_seconds=60.0
            ... )
            >>> endpoint = ModelEndpoint(model, config)

        Getting endpoint statistics:

            >>> endpoint = ModelEndpoint(model)
            >>> # After some requests...
            >>> stats = endpoint.get_stats()
            >>> print(f"Total requests: {stats['request_count']}")
            >>> print(f"Error rate: {stats['error_count'] / stats['request_count']}")

        Using with FastAPI:

            >>> app = FastAPI()
            >>> endpoint = ModelEndpoint(model)
            >>> @app.post("/generate")
            ... async def generate(request: GenerateRequest):
            ...     result = await endpoint.generate(request.prompt)
            ...     return result
    """

    def __init__(
        self,
        model: Any,
        config: Optional[EndpointConfig] = None,
    ):
        """Initialize model endpoint.

        Args:
            model: Model instance to serve. Must have a generate() method that
                accepts a prompt string. May optionally have an agenerate() method
                for async generation.
            config: Endpoint configuration. If None, uses default EndpointConfig.

        Examples:
            >>> from insideLLMs import DummyModel
            >>> endpoint = ModelEndpoint(DummyModel())

            >>> endpoint = ModelEndpoint(
            ...     model=DummyModel(),
            ...     config=EndpointConfig(path="/custom/generate")
            ... )
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
    ) -> dict[str, Any]:
        """Generate a response from the wrapped model asynchronously.

        This method handles both async and sync models. For async models (those
        with an agenerate method), it awaits directly. For sync models, it runs
        the generation in a thread pool executor to avoid blocking.

        Args:
            prompt: Input text prompt for generation.
            temperature: Sampling temperature controlling output randomness.
                Higher values produce more diverse outputs. Default is 0.7.
            max_tokens: Maximum number of tokens to generate. If None, uses
                the model's default setting.
            **kwargs: Additional keyword arguments passed to the model's
                generate method.

        Returns:
            Dictionary containing:
            - response: The generated text string
            - model_id: Model identifier if available
            - latency_ms: Generation time in milliseconds
            - request_id: Unique identifier for this request

        Raises:
            RuntimeError: If generation fails for any reason.

        Examples:
            Basic generation:

                >>> endpoint = ModelEndpoint(model)
                >>> result = await endpoint.generate("What is Python?")
                >>> print(result["response"])
                >>> print(f"Took {result['latency_ms']:.2f}ms")

            Generation with parameters:

                >>> result = await endpoint.generate(
                ...     prompt="Write a creative story",
                ...     temperature=0.9,
                ...     max_tokens=500
                ... )

            Handling errors:

                >>> try:
                ...     result = await endpoint.generate(prompt)
                ... except RuntimeError as e:
                ...     log_error(e)
                ...     return error_response()

            Using the request_id for logging:

                >>> result = await endpoint.generate("Hello")
                >>> logger.info(f"Request {result['request_id']} completed")
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Check if model has async generate
            if hasattr(self.model, "agenerate"):
                response = await self.model.agenerate(
                    prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
                )
            elif asyncio.iscoroutinefunction(getattr(self.model, "generate", None)):
                response = await self.model.generate(
                    prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
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
                        **kwargs,
                    ),
                )

            latency_ms = (time.time() - start_time) * 1000
            self._request_count += 1
            self._total_latency += latency_ms

            return {
                "response": response,
                "model_id": getattr(self.model, "model_id", None),
                "latency_ms": latency_ms,
                "request_id": request_id,
            }

        except Exception as e:
            self._error_count += 1
            raise RuntimeError(f"Generation failed: {str(e)}") from e

    def get_stats(self) -> dict[str, Any]:
        """Get endpoint statistics for monitoring.

        Returns accumulated statistics about requests processed by this endpoint,
        including counts and average latency.

        Returns:
            Dictionary containing:
            - request_count: Total number of successful requests processed
            - error_count: Total number of requests that resulted in errors
            - average_latency_ms: Average processing time in milliseconds

        Examples:
            Getting basic stats:

                >>> endpoint = ModelEndpoint(model)
                >>> # After processing some requests...
                >>> stats = endpoint.get_stats()
                >>> print(f"Requests: {stats['request_count']}")
                >>> print(f"Errors: {stats['error_count']}")

            Calculating error rate:

                >>> stats = endpoint.get_stats()
                >>> if stats["request_count"] > 0:
                ...     error_rate = stats["error_count"] / stats["request_count"]
                ...     print(f"Error rate: {error_rate:.2%}")

            Monitoring latency:

                >>> stats = endpoint.get_stats()
                >>> if stats["average_latency_ms"] > 1000:
                ...     alert("High latency detected!")

            Fresh endpoint (no requests yet):

                >>> endpoint = ModelEndpoint(model)
                >>> stats = endpoint.get_stats()
                >>> stats["request_count"]
                0
                >>> stats["average_latency_ms"]
                0
        """
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "average_latency_ms": (
                self._total_latency / self._request_count if self._request_count > 0 else 0
            ),
        }


class ProbeEndpoint:
    """Wraps evaluation probes for API serving.

    This class manages a collection of probes and provides an async interface
    for running them against a model via API requests. Each probe evaluates
    model responses for specific qualities like truthfulness, safety, or bias.

    Probes can be added dynamically and run individually or all together.
    Results include per-prompt evaluations and any errors encountered.

    Attributes:
        model: The model instance to evaluate.
        probes: Dictionary mapping probe names to probe instances.
        config: Endpoint configuration settings.

    Examples:
        Basic probe endpoint:

            >>> endpoint = ProbeEndpoint(model)
            >>> endpoint.add_probe("safety", SafetyProbe())
            >>> endpoint.add_probe("bias", BiasProbe())

        Initializing with probes:

            >>> probes = {
            ...     "truthfulness": TruthfulnessProbe(),
            ...     "toxicity": ToxicityProbe()
            ... }
            >>> endpoint = ProbeEndpoint(model, probes=probes)

        Running specific probe:

            >>> result = await endpoint.run_probe(
            ...     prompts=["Tell me something false"],
            ...     probe_type="truthfulness"
            ... )

        Running all probes:

            >>> result = await endpoint.run_probe(
            ...     prompts=["Test prompt 1", "Test prompt 2"]
            ... )
            >>> for probe_result in result["results"]:
            ...     print(f"Probe: {probe_result['probe']}")
    """

    def __init__(
        self,
        model: Any,
        probes: Optional[dict[str, Any]] = None,
        config: Optional[EndpointConfig] = None,
    ):
        """Initialize probe endpoint.

        Args:
            model: Model instance to evaluate with probes. Must have a generate()
                method.
            probes: Dictionary mapping probe names to probe instances. Each probe
                should have an evaluate() method. If None, starts with empty dict.
            config: Endpoint configuration. If None, uses default with /probe path.

        Examples:
            >>> endpoint = ProbeEndpoint(model)
            >>> endpoint.probes
            {}

            >>> endpoint = ProbeEndpoint(
            ...     model=model,
            ...     probes={"safety": SafetyProbe()},
            ...     config=EndpointConfig(path="/v1/probe")
            ... )
        """
        self.model = model
        self.probes = probes or {}
        self.config = config or EndpointConfig(path="/probe")

    def add_probe(self, name: str, probe: Any) -> None:
        """Add a probe to the endpoint.

        Registers a named probe that can be run against the model. The probe
        should have an evaluate() method that takes a prompt and response.

        Args:
            name: Unique name for the probe (e.g., "safety", "truthfulness").
            probe: Probe instance with an evaluate() method.

        Examples:
            Adding a single probe:

                >>> endpoint = ProbeEndpoint(model)
                >>> endpoint.add_probe("safety", SafetyProbe())
                >>> "safety" in endpoint.probes
                True

            Adding multiple probes:

                >>> endpoint.add_probe("bias", BiasProbe())
                >>> endpoint.add_probe("toxicity", ToxicityProbe())
                >>> len(endpoint.probes)
                3

            Replacing an existing probe:

                >>> endpoint.add_probe("safety", UpdatedSafetyProbe())
        """
        self.probes[name] = probe

    async def run_probe(
        self,
        prompts: list[str],
        probe_type: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run probes on a list of prompts and collect results.

        Executes the specified probe (or all probes if none specified) against
        each prompt. For each prompt, the model generates a response which is
        then evaluated by the probe(s).

        Args:
            prompts: List of input prompts to evaluate.
            probe_type: Name of specific probe to run. If None, runs all
                registered probes.
            **kwargs: Additional keyword arguments (reserved for future use).

        Returns:
            Dictionary containing:
            - results: List of probe results, each with probe name and per-prompt results
            - latency_ms: Total processing time in milliseconds
            - request_id: Unique identifier for this request

        Examples:
            Running a specific probe:

                >>> endpoint = ProbeEndpoint(model, {"safety": SafetyProbe()})
                >>> result = await endpoint.run_probe(
                ...     prompts=["Generate harmful content"],
                ...     probe_type="safety"
                ... )
                >>> result["results"][0]["probe"]
                'safety'

            Running all probes:

                >>> endpoint = ProbeEndpoint(model, {
                ...     "safety": SafetyProbe(),
                ...     "bias": BiasProbe()
                ... })
                >>> result = await endpoint.run_probe(prompts=["Test prompt"])
                >>> len(result["results"])  # One per probe
                2

            Processing results:

                >>> result = await endpoint.run_probe(prompts=["Test"])
                >>> for probe_result in result["results"]:
                ...     print(f"Probe: {probe_result['probe']}")
                ...     for item in probe_result.get("results", []):
                ...         print(f"  Score: {item.get('result')}")

            Handling probe errors:

                >>> result = await endpoint.run_probe(prompts=["Test"])
                >>> for probe_result in result["results"]:
                ...     if "error" in probe_result:
                ...         print(f"Probe {probe_result['probe']} failed")
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
                    if hasattr(probe, "evaluate"):
                        response = self.model.generate(prompt)
                        result = probe.evaluate(prompt, response)
                        probe_results.append(
                            {
                                "prompt": prompt,
                                "response": response,
                                "result": result,
                            }
                        )
                    else:
                        probe_results.append(
                            {
                                "prompt": prompt,
                                "error": "Probe does not support evaluation",
                            }
                        )

                results.append(
                    {
                        "probe": probe_name,
                        "results": probe_results,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "probe": probe_name,
                        "error": str(e),
                    }
                )

        latency_ms = (time.time() - start_time) * 1000

        return {
            "results": results,
            "latency_ms": latency_ms,
            "request_id": request_id,
        }


class BatchEndpoint:
    """Handles batch generation requests for processing multiple prompts efficiently.

    This class provides a batch processing endpoint that handles multiple prompts
    in a single API call. It enforces batch size limits and tracks aggregate
    statistics across all prompts in the batch.

    Batch processing is useful for:
    - Reducing API call overhead when processing many prompts
    - Evaluating models on test datasets
    - Running batch evaluations or benchmarks

    Attributes:
        model: The model instance to use for generation.
        max_batch_size: Maximum number of prompts allowed per batch.
        config: Endpoint configuration settings.

    Examples:
        Basic batch endpoint:

            >>> endpoint = BatchEndpoint(model)
            >>> result = await endpoint.generate_batch(
            ...     prompts=["What is 2+2?", "What is 3+3?"]
            ... )
            >>> len(result["responses"])
            2

        Endpoint with custom batch limit:

            >>> endpoint = BatchEndpoint(model, max_batch_size=50)
            >>> # Attempting to exceed limit raises ValueError
            >>> large_batch = ["prompt"] * 100
            >>> await endpoint.generate_batch(large_batch)  # Raises ValueError

        Processing benchmark dataset:

            >>> endpoint = BatchEndpoint(model, max_batch_size=100)
            >>> prompts = load_benchmark_prompts()
            >>> for batch in chunks(prompts, 100):
            ...     results = await endpoint.generate_batch(batch)
            ...     save_results(results)

        Getting batch statistics:

            >>> result = await endpoint.generate_batch(prompts)
            >>> print(f"Total tokens: {result['total_tokens']}")
            >>> print(f"Batch latency: {result['latency_ms']}ms")
    """

    def __init__(
        self,
        model: Any,
        max_batch_size: int = 100,
        config: Optional[EndpointConfig] = None,
    ):
        """Initialize batch endpoint.

        Args:
            model: Model instance to use for generation. Must have a generate()
                method, and optionally an agenerate() method for async operation.
            max_batch_size: Maximum number of prompts allowed in a single batch
                request. Requests exceeding this limit will raise ValueError.
                Default is 100.
            config: Endpoint configuration. If None, uses default with /batch path.

        Examples:
            >>> endpoint = BatchEndpoint(model)
            >>> endpoint.max_batch_size
            100

            >>> endpoint = BatchEndpoint(
            ...     model=model,
            ...     max_batch_size=50,
            ...     config=EndpointConfig(path="/v1/batch")
            ... )
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.config = config or EndpointConfig(path="/batch")

    async def generate_batch(
        self,
        prompts: list[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate responses for a batch of prompts.

        Processes all prompts in the batch, generating a response for each.
        Individual prompt failures are captured as error messages in the response
        list rather than failing the entire batch.

        Args:
            prompts: List of input prompts to process. Must not exceed max_batch_size.
            temperature: Sampling temperature applied to all generations.
                Default is 0.7.
            max_tokens: Maximum tokens per response. If None, uses model default.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            Dictionary containing:
            - responses: List of generated responses (or error messages)
            - total_tokens: Approximate total token count for the batch
            - latency_ms: Total processing time in milliseconds
            - request_id: Unique identifier for this batch request

        Raises:
            ValueError: If the number of prompts exceeds max_batch_size.

        Examples:
            Basic batch generation:

                >>> endpoint = BatchEndpoint(model)
                >>> result = await endpoint.generate_batch(
                ...     prompts=["Hello", "World"]
                ... )
                >>> result["responses"]
                ['Hello response...', 'World response...']

            Batch with parameters:

                >>> result = await endpoint.generate_batch(
                ...     prompts=["Write a poem", "Write a story"],
                ...     temperature=0.9,
                ...     max_tokens=200
                ... )

            Handling batch size limit:

                >>> endpoint = BatchEndpoint(model, max_batch_size=10)
                >>> try:
                ...     await endpoint.generate_batch(["p"] * 100)
                ... except ValueError as e:
                ...     print(f"Batch too large: {e}")

            Processing results with potential errors:

                >>> result = await endpoint.generate_batch(prompts)
                >>> for i, response in enumerate(result["responses"]):
                ...     if response.startswith("Error:"):
                ...         print(f"Prompt {i} failed: {response}")
                ...     else:
                ...         process_response(response)

            Monitoring batch performance:

                >>> result = await endpoint.generate_batch(prompts)
                >>> avg_tokens = result["total_tokens"] / len(prompts)
                >>> print(f"Avg tokens per prompt: {avg_tokens:.1f}")
                >>> print(f"Total latency: {result['latency_ms']:.2f}ms")
        """
        if len(prompts) > self.max_batch_size:
            raise ValueError(f"Batch size exceeds maximum of {self.max_batch_size}")

        request_id = str(uuid.uuid4())
        start_time = time.time()

        responses = []
        total_tokens = 0

        for prompt in prompts:
            try:
                if hasattr(self.model, "agenerate"):
                    response = await self.model.agenerate(prompt, temperature=temperature)
                else:
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None, lambda p=prompt: self.model.generate(p, temperature=temperature)
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
    """Main wrapper for deploying models as FastAPI applications.

    This class is the primary interface for creating production-ready API
    deployments. It orchestrates all components including endpoints, middleware,
    monitoring, and authentication into a cohesive FastAPI application.

    The deployment includes:
    - Generation endpoint (/generate) for single prompt completion
    - Batch endpoint (/batch) for multiple prompts
    - Probe endpoint (/probe) for model evaluation
    - Health endpoint (/health) for monitoring
    - Metrics endpoint (/metrics) for statistics
    - CORS middleware for cross-origin requests
    - Request ID middleware for tracing

    Attributes:
        model: The model being served.
        config: Application configuration.

    Examples:
        Basic deployment:

            >>> from insideLLMs import DummyModel
            >>> model = DummyModel()
            >>> deployment = DeploymentApp(model)
            >>> app = deployment.build_app()
            >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000

        Deployment with custom configuration:

            >>> config = AppConfig(
            ...     deployment=DeploymentConfig(
            ...         title="My LLM API",
            ...         api_key="secret-key"
            ...     )
            ... )
            >>> deployment = DeploymentApp(model, config)
            >>> app = deployment.app  # Auto-builds if needed

        Adding probes to deployment:

            >>> deployment = DeploymentApp(model)
            >>> deployment.add_probe("safety", SafetyProbe())
            >>> deployment.add_probe("bias", BiasProbe())
            >>> app = deployment.build_app()

        Adding custom health checks:

            >>> deployment = DeploymentApp(model)
            >>> deployment.add_health_check("gpu", lambda: gpu.is_available())
            >>> deployment.add_health_check("memory", lambda: memory.ok())
            >>> app = deployment.build_app()

    Note:
        Requires FastAPI and uvicorn to be installed. Install with:
        ``pip install fastapi uvicorn``
    """

    def __init__(
        self,
        model: Any,
        config: Optional[AppConfig] = None,
    ):
        """Initialize deployment application.

        Creates all necessary components for the deployment including endpoints,
        rate limiter, authentication, metrics collector, and health checker.

        Args:
            model: Model instance to serve. Must have a generate() method.
            config: Application configuration. If None, uses default AppConfig.

        Raises:
            ImportError: If FastAPI is not installed.

        Examples:
            >>> from insideLLMs import DummyModel
            >>> deployment = DeploymentApp(DummyModel())

            >>> deployment = DeploymentApp(
            ...     model=DummyModel(),
            ...     config=AppConfig(
            ...         deployment=DeploymentConfig(title="Custom API")
            ...     )
            ... )
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for deployment. Install with: pip install fastapi uvicorn"
            )

        self.model = model
        self.config = config or AppConfig()
        self._app: Optional[FastAPI] = None
        self._model_endpoint = ModelEndpoint(model, self.config.generate_endpoint)
        self._batch_endpoint = BatchEndpoint(model, self.config.deployment.max_batch_size)
        self._probe_endpoint = ProbeEndpoint(model, config=self.config.probe_endpoint)
        self._metrics = MetricsCollector()
        self._health_checker = HealthChecker()
        self._rate_limiter = KeyedTokenBucketRateLimiter()
        self._auth = APIKeyAuth(
            [self.config.deployment.api_key] if self.config.deployment.api_key else None
        )
        self._request_logger = RequestLogger()

    def build_app(self) -> "FastAPI":
        """Build and configure the FastAPI application.

        Creates a new FastAPI application instance with all endpoints, middleware,
        and configuration applied. This method can be called multiple times to
        rebuild the application with updated settings.

        Returns:
            Configured FastAPI application instance ready to be run with uvicorn.

        Examples:
            Building and running the app:

                >>> deployment = DeploymentApp(model)
                >>> app = deployment.build_app()
                >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000

            Rebuilding after configuration changes:

                >>> deployment = DeploymentApp(model)
                >>> deployment.add_probe("safety", SafetyProbe())
                >>> app = deployment.build_app()  # Includes the new probe

            Using in tests:

                >>> from fastapi.testclient import TestClient
                >>> deployment = DeploymentApp(model)
                >>> app = deployment.build_app()
                >>> client = TestClient(app)
                >>> response = client.post("/generate", json={"prompt": "test"})

            Accessing app configuration:

                >>> app = deployment.build_app()
                >>> app.title
                'insideLLMs API'
        """
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware

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
                if not self._rate_limiter.is_allowed(getattr(req.state, "request_id", "default")):
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
                    model_id=getattr(self.model, "model_id", None),
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
                "model_id": getattr(self.model, "model_id", None),
            }

    def add_probe(self, name: str, probe: Any) -> None:
        """Add a probe to the deployment's probe endpoint.

        Registers a named probe that will be available at the /probe endpoint.
        Probes should have an evaluate() method for assessing model responses.

        Args:
            name: Unique name for the probe (e.g., "safety", "truthfulness").
            probe: Probe instance with an evaluate() method.

        Examples:
            Adding evaluation probes:

                >>> deployment = DeploymentApp(model)
                >>> deployment.add_probe("safety", SafetyProbe())
                >>> deployment.add_probe("bias", BiasProbe())
                >>> deployment.add_probe("truthfulness", TruthfulnessProbe())

            Probes are available via API:

                >>> # POST /probe with {"prompts": [...], "probe_type": "safety"}
                >>> app = deployment.build_app()

            Replacing a probe:

                >>> deployment.add_probe("safety", ImprovedSafetyProbe())
        """
        self._probe_endpoint.add_probe(name, probe)

    def add_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Add a health check to the deployment.

        Registers a named health check that will be executed when the /health
        endpoint is called. All checks must pass for the service to be considered
        healthy.

        Args:
            name: Unique name for the health check (e.g., "database", "gpu").
            check_func: Callable that returns True if healthy, False otherwise.
                Should not raise exceptions (they will be caught and treated
                as failures).

        Examples:
            Adding infrastructure checks:

                >>> deployment = DeploymentApp(model)
                >>> deployment.add_health_check("gpu", lambda: gpu.is_available())
                >>> deployment.add_health_check("memory", lambda: get_free_memory() > 1e9)
                >>> deployment.add_health_check("model", lambda: model.is_loaded())

            Custom database check:

                >>> def check_database():
                ...     try:
                ...         db.execute("SELECT 1")
                ...         return True
                ...     except Exception:
                ...         return False
                >>> deployment.add_health_check("database", check_database)

            Health endpoint returns check results:

                >>> # GET /health returns:
                >>> # {"status": "healthy", "checks": {"gpu": true, "memory": true}}
        """
        self._health_checker.add_check(name, check_func)

    @property
    def app(self) -> "FastAPI":
        """Get or build the FastAPI application.

        This property provides lazy initialization of the FastAPI application.
        If the app hasn't been built yet, it calls build_app() automatically.

        Returns:
            The configured FastAPI application instance.

        Examples:
            Accessing the app (auto-builds if needed):

                >>> deployment = DeploymentApp(model)
                >>> app = deployment.app  # Builds automatically
                >>> type(app).__name__
                'FastAPI'

            Multiple accesses return the same instance:

                >>> app1 = deployment.app
                >>> app2 = deployment.app
                >>> app1 is app2
                True

            Use in uvicorn module:

                >>> # In myapi.py:
                >>> deployment = DeploymentApp(model)
                >>> app = deployment.app
                >>> # Run: uvicorn myapi:app --host 0.0.0.0 --port 8000
        """
        if self._app is None:
            self.build_app()
        return self._app  # type: ignore[return-value]  # _app is set by build_app(), checked above


def create_app(
    model: Any,
    title: str = "insideLLMs API",
    description: str = "API for LLM probing and evaluation",
    version: str = "1.0.0",
    cors_origins: Optional[list[str]] = None,
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
        >>> from insideLLMs.system.deployment import create_app
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
        >>> from insideLLMs.system.deployment import quick_deploy
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
    probes: Optional[dict[str, Any]] = None,
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


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import RateLimiter. The canonical name is
# KeyedTokenBucketRateLimiter.
RateLimiter = KeyedTokenBucketRateLimiter
