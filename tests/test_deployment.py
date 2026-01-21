"""Tests for Deployment module."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, Mock

import pytest

from insideLLMs.deployment import (
    FASTAPI_AVAILABLE,
    APIKeyAuth,
    AppConfig,
    BatchEndpoint,
    DeploymentConfig,
    # Configuration
    EndpointConfig,
    HealthChecker,
    # Monitoring
    MetricsCollector,
    # Endpoints
    ModelEndpoint,
    ProbeEndpoint,
    # Middleware
    RateLimiter,
    RequestLogger,
    # Convenience
    create_model_endpoint,
    create_probe_endpoint,
)

# Marker for FastAPI-dependent tests
requires_fastapi = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")


# =============================================================================
# Test EndpointConfig
# =============================================================================


class TestEndpointConfig:
    """Tests for EndpointConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = EndpointConfig()
        assert config.path == "/generate"
        assert config.method == "POST"
        assert config.enabled is True
        assert config.rate_limit is None
        assert config.timeout_seconds == 30.0
        assert config.require_auth is False

    def test_custom_values(self):
        """Test custom config values."""
        config = EndpointConfig(
            path="/custom",
            method="PUT",
            rate_limit=100,
            require_auth=True,
        )
        assert config.path == "/custom"
        assert config.method == "PUT"
        assert config.rate_limit == 100
        assert config.require_auth is True

    def test_to_dict(self):
        """Test config serialization."""
        config = EndpointConfig(
            path="/test",
            tags=["api", "v1"],
            description="Test endpoint",
        )
        data = config.to_dict()
        assert data["path"] == "/test"
        assert data["tags"] == ["api", "v1"]
        assert data["description"] == "Test endpoint"


# =============================================================================
# Test DeploymentConfig
# =============================================================================


class TestDeploymentConfig:
    """Tests for DeploymentConfig."""

    def test_default_values(self):
        """Test default deployment config."""
        config = DeploymentConfig()
        assert config.title == "insideLLMs API"
        assert config.version == "1.0.0"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.enable_docs is True
        assert config.enable_metrics is True
        assert config.enable_health is True
        assert config.api_key is None

    def test_custom_values(self):
        """Test custom deployment config."""
        config = DeploymentConfig(
            title="My API",
            port=9000,
            api_key="secret123",
        )
        assert config.title == "My API"
        assert config.port == 9000
        assert config.api_key == "secret123"

    def test_to_dict(self):
        """Test config serialization."""
        config = DeploymentConfig(
            title="Test API",
            enable_docs=False,
        )
        data = config.to_dict()
        assert data["title"] == "Test API"
        assert data["enable_docs"] is False


# =============================================================================
# Test AppConfig
# =============================================================================


class TestAppConfig:
    """Tests for AppConfig."""

    def test_default_configs(self):
        """Test default app config."""
        config = AppConfig()
        assert isinstance(config.deployment, DeploymentConfig)
        assert isinstance(config.generate_endpoint, EndpointConfig)
        assert isinstance(config.batch_endpoint, EndpointConfig)
        assert isinstance(config.probe_endpoint, EndpointConfig)

    def test_endpoint_paths(self):
        """Test default endpoint paths."""
        config = AppConfig()
        assert config.generate_endpoint.path == "/generate"
        assert config.batch_endpoint.path == "/batch"
        assert config.probe_endpoint.path == "/probe"


# =============================================================================
# Test RateLimiter
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allows_within_limit(self):
        """Test requests within limit are allowed."""
        limiter = RateLimiter(requests_per_minute=60)

        for _ in range(10):
            assert limiter.is_allowed("test_key")

    def test_blocks_over_limit(self):
        """Test requests over burst limit are blocked."""
        limiter = RateLimiter(requests_per_minute=5, burst_size=5)

        # Use all burst tokens
        for _ in range(5):
            assert limiter.is_allowed("test_key")

        # Should be blocked
        assert not limiter.is_allowed("test_key")

    def test_different_keys(self):
        """Test different keys have separate limits."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)

        assert limiter.is_allowed("key1")
        assert limiter.is_allowed("key1")
        assert not limiter.is_allowed("key1")

        # Different key should have fresh limit
        assert limiter.is_allowed("key2")
        assert limiter.is_allowed("key2")

    def test_token_refill(self):
        """Test tokens refill over time."""
        limiter = RateLimiter(requests_per_minute=6000, burst_size=1)  # 100/sec

        assert limiter.is_allowed("key")
        assert not limiter.is_allowed("key")

        # Wait for refill
        time.sleep(0.02)  # 20ms should give ~2 tokens at 100/sec
        assert limiter.is_allowed("key")

    def test_get_wait_time(self):
        """Test wait time calculation."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)

        assert limiter.is_allowed("key")
        assert not limiter.is_allowed("key")

        wait_time = limiter.get_wait_time("key")
        assert wait_time > 0

    def test_no_wait_with_tokens(self):
        """Test zero wait time when tokens available."""
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.get_wait_time("key") == 0


# =============================================================================
# Test APIKeyAuth
# =============================================================================


class TestAPIKeyAuth:
    """Tests for APIKeyAuth."""

    def test_no_keys_allows_all(self):
        """Test open access when no keys configured."""
        auth = APIKeyAuth()
        assert auth.validate("any_key")
        assert auth.validate(None)

    def test_valid_key(self):
        """Test valid key is accepted."""
        auth = APIKeyAuth(valid_keys=["key1", "key2"])
        assert auth.validate("key1")
        assert auth.validate("key2")

    def test_invalid_key(self):
        """Test invalid key is rejected."""
        auth = APIKeyAuth(valid_keys=["valid_key"])
        assert not auth.validate("invalid_key")
        assert not auth.validate(None)

    def test_add_key(self):
        """Test adding keys dynamically."""
        auth = APIKeyAuth(valid_keys=["key1"])
        assert not auth.validate("key2")

        auth.add_key("key2")
        assert auth.validate("key2")

    def test_remove_key(self):
        """Test removing keys."""
        auth = APIKeyAuth(valid_keys=["key1", "key2"])
        auth.remove_key("key1")

        assert not auth.validate("key1")
        assert auth.validate("key2")

    def test_usage_tracking(self):
        """Test key usage tracking."""
        auth = APIKeyAuth(valid_keys=["key1"])

        auth.validate("key1")
        auth.validate("key1")
        auth.validate("key1")

        assert auth.get_usage("key1") == 3
        assert auth.get_usage("unknown") == 0


# =============================================================================
# Test RequestLogger
# =============================================================================


class TestRequestLogger:
    """Tests for RequestLogger."""

    def test_log_request(self):
        """Test basic request logging."""
        logger = RequestLogger()

        logger.log_request(
            method="POST",
            path="/generate",
            status_code=200,
            latency_ms=50.0,
            request_id="req123",
        )

        logs = logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["method"] == "POST"
        assert logs[0]["path"] == "/generate"
        assert logs[0]["status_code"] == 200

    def test_log_with_body(self):
        """Test logging with request body."""
        logger = RequestLogger(include_body=True)

        logger.log_request(
            method="POST",
            path="/generate",
            status_code=200,
            latency_ms=50.0,
            request_id="req123",
            body={"prompt": "test"},
        )

        logs = logger.get_logs()
        assert logs[0]["body"] == {"prompt": "test"}

    def test_log_with_response(self):
        """Test logging with response."""
        logger = RequestLogger(include_response=True)

        logger.log_request(
            method="POST",
            path="/generate",
            status_code=200,
            latency_ms=50.0,
            request_id="req123",
            response={"response": "output"},
        )

        logs = logger.get_logs()
        assert logs[0]["response"] == {"response": "output"}

    def test_api_key_masking(self):
        """Test API key is masked in logs."""
        logger = RequestLogger()

        logger.log_request(
            method="POST",
            path="/generate",
            status_code=200,
            latency_ms=50.0,
            request_id="req123",
            api_key="secret_api_key_12345",
        )

        logs = logger.get_logs()
        assert "secret" not in logs[0]["api_key"]
        assert "***" in logs[0]["api_key"]

    def test_custom_log_func(self):
        """Test custom logging function."""
        custom_logs = []
        logger = RequestLogger(log_func=lambda entry: custom_logs.append(entry))

        logger.log_request(
            method="GET",
            path="/health",
            status_code=200,
            latency_ms=10.0,
            request_id="req123",
        )

        assert len(custom_logs) == 1

    def test_get_logs_limit(self):
        """Test log retrieval with limit."""
        logger = RequestLogger()

        for i in range(150):
            logger.log_request(
                method="GET",
                path=f"/path{i}",
                status_code=200,
                latency_ms=10.0,
                request_id=f"req{i}",
            )

        logs = logger.get_logs(limit=50)
        assert len(logs) == 50


# =============================================================================
# Test MetricsCollector
# =============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_increment_counter(self):
        """Test counter increment."""
        metrics = MetricsCollector()

        metrics.increment("requests")
        metrics.increment("requests")
        metrics.increment("requests", value=3)

        data = metrics.get_metrics()
        assert data["counters"]["requests"] == 5

    def test_set_gauge(self):
        """Test gauge setting."""
        metrics = MetricsCollector()

        metrics.set_gauge("active_connections", 10)
        metrics.set_gauge("active_connections", 15)

        data = metrics.get_metrics()
        assert data["gauges"]["active_connections"] == 15

    def test_record_histogram(self):
        """Test histogram recording."""
        metrics = MetricsCollector()

        for i in range(100):
            metrics.record("latency_ms", float(i))

        data = metrics.get_metrics()
        hist = data["histograms"]["latency_ms"]

        assert hist["count"] == 100
        assert hist["min"] == 0
        assert hist["max"] == 99
        assert hist["mean"] == 49.5

    def test_histogram_percentiles(self):
        """Test histogram percentile calculation."""
        metrics = MetricsCollector()

        for i in range(100):
            metrics.record("latency", float(i))

        data = metrics.get_metrics()
        hist = data["histograms"]["latency"]

        assert hist["p50"] == 50
        assert hist["p95"] == 95
        assert hist["p99"] == 99

    def test_uptime(self):
        """Test uptime tracking."""
        metrics = MetricsCollector()
        time.sleep(0.1)

        data = metrics.get_metrics()
        assert data["uptime_seconds"] >= 0.1

    def test_histogram_size_limit(self):
        """Test histogram keeps last 1000 values."""
        metrics = MetricsCollector()

        for i in range(1500):
            metrics.record("metric", float(i))

        data = metrics.get_metrics()
        assert data["histograms"]["metric"]["count"] == 1000


# =============================================================================
# Test HealthChecker
# =============================================================================


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_add_check(self):
        """Test adding health checks."""
        checker = HealthChecker()
        checker.add_check("database", lambda: True)
        checker.add_check("cache", lambda: True)

        result = checker.run_checks()
        assert result["status"] == "healthy"
        assert result["checks"]["database"] is True
        assert result["checks"]["cache"] is True

    def test_failing_check(self):
        """Test failing health check."""
        checker = HealthChecker()
        checker.add_check("healthy", lambda: True)
        checker.add_check("unhealthy", lambda: False)

        result = checker.run_checks()
        assert result["status"] == "unhealthy"
        assert result["checks"]["healthy"] is True
        assert result["checks"]["unhealthy"] is False

    def test_exception_in_check(self):
        """Test health check that raises exception."""
        checker = HealthChecker()
        checker.add_check("healthy", lambda: True)
        checker.add_check("broken", lambda: 1 / 0)

        result = checker.run_checks()
        assert result["status"] == "unhealthy"
        assert result["checks"]["broken"] is False

    def test_uptime_tracking(self):
        """Test uptime in health check."""
        checker = HealthChecker()
        time.sleep(0.1)

        result = checker.run_checks()
        assert result["uptime_seconds"] >= 0.1


# =============================================================================
# Test ModelEndpoint
# =============================================================================


@requires_fastapi
class TestModelEndpoint:
    """Tests for ModelEndpoint."""

    @pytest.mark.asyncio
    async def test_generate_sync_model(self):
        """Test generate with sync model."""
        # Create a mock without agenerate to test sync path
        model = Mock(spec=["generate"])
        model.generate.return_value = "Generated response"

        endpoint = ModelEndpoint(model)
        result = await endpoint.generate("Test prompt")

        assert result["response"] == "Generated response"
        assert "latency_ms" in result
        assert "request_id" in result

    @pytest.mark.asyncio
    async def test_generate_async_model(self):
        """Test generate with async model."""
        model = Mock()
        model.agenerate = AsyncMock(return_value="Async response")

        endpoint = ModelEndpoint(model)
        result = await endpoint.generate("Test prompt")

        assert result["response"] == "Async response"

    @pytest.mark.asyncio
    async def test_generate_with_params(self):
        """Test generate with parameters."""
        # Create a mock without agenerate to test sync path with params
        model = Mock(spec=["generate"])
        model.generate.return_value = "Response"

        endpoint = ModelEndpoint(model)
        await endpoint.generate(
            "Prompt",
            temperature=0.5,
            max_tokens=100,
        )

        model.generate.assert_called_once()
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_generate_error_handling(self):
        """Test error handling in generate."""
        # Create a mock without agenerate to test error handling on sync path
        model = Mock(spec=["generate"])
        model.generate.side_effect = Exception("Model error")

        endpoint = ModelEndpoint(model)

        with pytest.raises(RuntimeError, match="Generation failed"):
            await endpoint.generate("Prompt")

        assert endpoint._error_count == 1

    def test_get_stats(self):
        """Test endpoint statistics."""
        # Create a mock without agenerate to test stats collection
        model = Mock(spec=["generate"])
        model.generate.return_value = "Response"

        endpoint = ModelEndpoint(model)

        async def run_generates():
            await endpoint.generate("Prompt 1")
            await endpoint.generate("Prompt 2")

        asyncio.run(run_generates())

        stats = endpoint.get_stats()
        assert stats["request_count"] == 2
        assert stats["error_count"] == 0
        assert stats["average_latency_ms"] > 0


# =============================================================================
# Test ProbeEndpoint
# =============================================================================


@requires_fastapi
class TestProbeEndpoint:
    """Tests for ProbeEndpoint."""

    def test_add_probe(self):
        """Test adding probes."""
        model = Mock()
        endpoint = ProbeEndpoint(model)

        probe = Mock()
        endpoint.add_probe("test_probe", probe)

        assert "test_probe" in endpoint.probes

    @pytest.mark.asyncio
    async def test_run_single_probe(self):
        """Test running a single probe."""
        model = Mock()
        model.generate.return_value = "Response"

        probe = Mock()
        probe.evaluate.return_value = {"score": 0.8}

        endpoint = ProbeEndpoint(model, probes={"test": probe})
        result = await endpoint.run_probe(["Prompt"], probe_type="test")

        assert len(result["results"]) == 1
        assert result["results"][0]["probe"] == "test"
        assert "request_id" in result

    @pytest.mark.asyncio
    async def test_run_all_probes(self):
        """Test running all probes."""
        model = Mock()
        model.generate.return_value = "Response"

        probe1 = Mock()
        probe1.evaluate.return_value = {"score": 0.8}
        probe2 = Mock()
        probe2.evaluate.return_value = {"score": 0.9}

        endpoint = ProbeEndpoint(model, probes={"p1": probe1, "p2": probe2})
        result = await endpoint.run_probe(["Prompt"])

        assert len(result["results"]) == 2


# =============================================================================
# Test BatchEndpoint
# =============================================================================


@requires_fastapi
class TestBatchEndpoint:
    """Tests for BatchEndpoint."""

    @pytest.mark.asyncio
    async def test_batch_generate(self):
        """Test batch generation."""
        # Create a mock without agenerate to test sync path
        model = Mock(spec=["generate"])
        model.generate.side_effect = ["Response 1", "Response 2", "Response 3"]

        endpoint = BatchEndpoint(model, max_batch_size=10)
        result = await endpoint.generate_batch(["P1", "P2", "P3"])

        assert len(result["responses"]) == 3
        assert result["responses"][0] == "Response 1"
        assert "total_tokens" in result

    @pytest.mark.asyncio
    async def test_batch_size_limit(self):
        """Test batch size limit enforcement."""
        model = Mock()
        endpoint = BatchEndpoint(model, max_batch_size=2)

        with pytest.raises(ValueError, match="exceeds maximum"):
            await endpoint.generate_batch(["P1", "P2", "P3"])

    @pytest.mark.asyncio
    async def test_batch_handles_errors(self):
        """Test batch handles individual errors gracefully."""
        model = Mock()
        model.generate.side_effect = [
            "Response 1",
            Exception("Error"),
            "Response 3",
        ]

        endpoint = BatchEndpoint(model, max_batch_size=10)
        result = await endpoint.generate_batch(["P1", "P2", "P3"])

        assert len(result["responses"]) == 3
        assert "Error" in result["responses"][1]


# =============================================================================
# Test Convenience Functions
# =============================================================================


@requires_fastapi
class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_model_endpoint(self):
        """Test model endpoint creation."""
        model = Mock()
        endpoint = create_model_endpoint(
            model,
            path="/custom",
            rate_limit=100,
        )

        assert endpoint.model == model
        assert endpoint.config.path == "/custom"
        assert endpoint.config.rate_limit == 100

    def test_create_probe_endpoint(self):
        """Test probe endpoint creation."""
        model = Mock()
        probes = {"test": Mock()}

        endpoint = create_probe_endpoint(
            model,
            probes=probes,
            path="/custom_probe",
        )

        assert endpoint.model == model
        assert "test" in endpoint.probes
        assert endpoint.config.path == "/custom_probe"


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_rate_limiter_concurrent(self):
        """Test rate limiter under concurrent access."""
        limiter = RateLimiter(requests_per_minute=1000, burst_size=100)
        results = []

        def check_limit():
            for _ in range(50):
                results.append(limiter.is_allowed("shared_key"))

        threads = [
            threading.Thread(target=check_limit),
            threading.Thread(target=check_limit),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have had some allowed and some blocked
        assert len(results) == 100
        assert sum(results) <= 100  # At most 100 allowed

    def test_metrics_concurrent(self):
        """Test metrics collector under concurrent access."""
        metrics = MetricsCollector()

        def increment_metrics():
            for _ in range(100):
                metrics.increment("counter")
                metrics.record("histogram", 1.0)

        threads = [
            threading.Thread(target=increment_metrics),
            threading.Thread(target=increment_metrics),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        data = metrics.get_metrics()
        assert data["counters"]["counter"] == 200


# =============================================================================
# Test FastAPI Integration (if available)
# =============================================================================


@requires_fastapi
class TestFastAPIIntegration:
    """Tests for FastAPI integration."""

    def test_create_app(self):
        """Test app creation."""
        from insideLLMs.deployment import create_app

        model = Mock()
        model.generate.return_value = "Response"

        app = create_app(model, title="Test API")
        assert app.title == "Test API"

    def test_deployment_app_build(self):
        """Test DeploymentApp building."""
        from insideLLMs.deployment import DeploymentApp

        model = Mock()
        model.generate.return_value = "Response"

        deployment = DeploymentApp(model)
        app = deployment.build_app()

        assert app is not None
        assert deployment.app is app

    def test_add_probe_to_deployment(self):
        """Test adding probe to deployment."""
        from insideLLMs.deployment import DeploymentApp

        model = Mock()
        probe = Mock()

        deployment = DeploymentApp(model)
        deployment.add_probe("test_probe", probe)

        assert "test_probe" in deployment._probe_endpoint.probes

    def test_add_health_check(self):
        """Test adding health check to deployment."""
        from insideLLMs.deployment import DeploymentApp

        model = Mock()

        deployment = DeploymentApp(model)
        deployment.add_health_check("custom_check", lambda: True)

        health = deployment._health_checker.run_checks()
        assert health["checks"]["custom_check"] is True


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_api_key_list(self):
        """Test auth with empty key list."""
        auth = APIKeyAuth(valid_keys=[])
        # Empty list means open access
        assert auth.validate("any")

    def test_rate_limiter_zero_burst(self):
        """Test rate limiter with minimal burst."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)

        assert limiter.is_allowed("key")
        assert not limiter.is_allowed("key")

    @requires_fastapi
    @pytest.mark.asyncio
    async def test_model_endpoint_with_model_id(self):
        """Test model endpoint preserves model_id."""
        # Create a mock without agenerate to test sync path
        model = Mock(spec=["generate", "model_id"])
        model.generate.return_value = "Response"
        model.model_id = "test-model-v1"

        endpoint = ModelEndpoint(model)
        result = await endpoint.generate("Test")

        assert result["model_id"] == "test-model-v1"

    def test_request_logger_empty_logs(self):
        """Test getting logs when empty."""
        logger = RequestLogger()
        assert logger.get_logs() == []

    def test_health_checker_no_checks(self):
        """Test health checker with no checks."""
        checker = HealthChecker()
        result = checker.run_checks()

        assert result["status"] == "healthy"
        assert result["checks"] == {}

    def test_metrics_empty_histogram(self):
        """Test metrics with no histogram data."""
        metrics = MetricsCollector()
        data = metrics.get_metrics()

        assert data["histograms"] == {}
