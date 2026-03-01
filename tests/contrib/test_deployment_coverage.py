"""Extended tests for Deployment module to achieve 90%+ coverage.

These tests complement tests/test_deployment.py by covering:
- Pydantic schema validation (GenerateRequest, GenerateResponse, etc.)
- DeploymentApp route handlers via FastAPI TestClient
- Authentication enforcement in routes
- Rate-limiting enforcement in routes
- Error responses from route handlers
- quick_deploy() and create_app() convenience functions
- ProbeEndpoint edge cases (no evaluate, exceptions)
- BatchEndpoint with async model
- ModelEndpoint coroutine-generate path
- Disabled endpoints/features
- RequestLogger edge cases (short API key, client_ip)
- DeploymentApp.app lazy property
- Integration between components
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from insideLLMs.contrib.deployment import (
    FASTAPI_AVAILABLE,
    APIKeyAuth,
    AppConfig,
    BatchEndpoint,
    DeploymentConfig,
    EndpointConfig,
    HealthChecker,
    KeyedTokenBucketRateLimiter,
    MetricsCollector,
    ModelEndpoint,
    ProbeEndpoint,
    RateLimiter,
    RequestLogger,
    create_app,
    create_model_endpoint,
    create_probe_endpoint,
)

requires_fastapi = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")


# =============================================================================
# Pydantic Schema Tests
# =============================================================================


@requires_fastapi
class TestGenerateRequestSchema:
    """Tests for GenerateRequest Pydantic model."""

    def test_minimal_request(self):
        from insideLLMs.contrib.deployment import GenerateRequest

        req = GenerateRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.temperature == 0.7
        assert req.max_tokens is None
        assert req.stop_sequences is None
        assert req.stream is False
        assert req.metadata is None

    def test_full_request(self):
        from insideLLMs.contrib.deployment import GenerateRequest

        req = GenerateRequest(
            prompt="Write a poem",
            temperature=0.9,
            max_tokens=200,
            stop_sequences=["\n\n"],
            stream=True,
            metadata={"user": "test"},
        )
        assert req.prompt == "Write a poem"
        assert req.temperature == 0.9
        assert req.max_tokens == 200
        assert req.stop_sequences == ["\n\n"]
        assert req.stream is True
        assert req.metadata == {"user": "test"}

    def test_temperature_validation_bounds(self):
        from pydantic import ValidationError

        from insideLLMs.contrib.deployment import GenerateRequest

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", temperature=-0.1)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", temperature=2.1)

        # Boundary values should work
        req_low = GenerateRequest(prompt="test", temperature=0.0)
        assert req_low.temperature == 0.0
        req_high = GenerateRequest(prompt="test", temperature=2.0)
        assert req_high.temperature == 2.0

    def test_max_tokens_validation(self):
        from pydantic import ValidationError

        from insideLLMs.contrib.deployment import GenerateRequest

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", max_tokens=0)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", max_tokens=-1)

    def test_prompt_required(self):
        from pydantic import ValidationError

        from insideLLMs.contrib.deployment import GenerateRequest

        with pytest.raises(ValidationError):
            GenerateRequest()


@requires_fastapi
class TestGenerateResponseSchema:
    """Tests for GenerateResponse Pydantic model."""

    def test_minimal_response(self):
        from insideLLMs.contrib.deployment import GenerateResponse

        resp = GenerateResponse(
            response="Hello world",
            latency_ms=10.5,
            request_id="req-001",
        )
        assert resp.response == "Hello world"
        assert resp.model_id is None
        assert resp.prompt_tokens is None
        assert resp.completion_tokens is None
        assert resp.latency_ms == 10.5
        assert resp.request_id == "req-001"
        assert resp.metadata is None

    def test_full_response(self):
        from insideLLMs.contrib.deployment import GenerateResponse

        resp = GenerateResponse(
            response="Answer",
            model_id="gpt-4",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=120.0,
            request_id="req-002",
            metadata={"cached": True},
        )
        assert resp.model_id == "gpt-4"
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.metadata == {"cached": True}

    def test_model_dump(self):
        from insideLLMs.contrib.deployment import GenerateResponse

        resp = GenerateResponse(response="text", latency_ms=1.0, request_id="r")
        d = resp.model_dump()
        assert d["response"] == "text"
        assert "request_id" in d


@requires_fastapi
class TestProbeRequestSchema:
    """Tests for ProbeRequest Pydantic model."""

    def test_minimal_probe_request(self):
        from insideLLMs.contrib.deployment import ProbeRequest

        req = ProbeRequest(prompts=["Hello"])
        assert req.prompts == ["Hello"]
        assert req.probe_type is None
        assert req.metadata is None

    def test_full_probe_request(self):
        from insideLLMs.contrib.deployment import ProbeRequest

        req = ProbeRequest(
            prompts=["P1", "P2"],
            probe_type="safety",
            metadata={"suite": "v2"},
        )
        assert len(req.prompts) == 2
        assert req.probe_type == "safety"


@requires_fastapi
class TestProbeResponseSchema:
    """Tests for ProbeResponse Pydantic model."""

    def test_minimal_probe_response(self):
        from insideLLMs.contrib.deployment import ProbeResponse

        resp = ProbeResponse(
            results=[{"probe": "safety", "results": []}],
            latency_ms=50.0,
            request_id="probe-001",
        )
        assert len(resp.results) == 1
        assert resp.summary is None

    def test_with_summary(self):
        from insideLLMs.contrib.deployment import ProbeResponse

        resp = ProbeResponse(
            results=[],
            summary={"total": 10, "avg_score": 0.8},
            latency_ms=100.0,
            request_id="probe-002",
        )
        assert resp.summary["total"] == 10


@requires_fastapi
class TestBatchRequestSchema:
    """Tests for BatchRequest Pydantic model."""

    def test_minimal_batch_request(self):
        from insideLLMs.contrib.deployment import BatchRequest

        req = BatchRequest(prompts=["A", "B"])
        assert req.prompts == ["A", "B"]
        assert req.temperature == 0.7
        assert req.max_tokens is None

    def test_temperature_bounds(self):
        from pydantic import ValidationError

        from insideLLMs.contrib.deployment import BatchRequest

        with pytest.raises(ValidationError):
            BatchRequest(prompts=["x"], temperature=-0.5)
        with pytest.raises(ValidationError):
            BatchRequest(prompts=["x"], temperature=3.0)


@requires_fastapi
class TestBatchResponseSchema:
    """Tests for BatchResponse Pydantic model."""

    def test_batch_response(self):
        from insideLLMs.contrib.deployment import BatchResponse

        resp = BatchResponse(
            responses=["R1", "R2"],
            total_tokens=20,
            latency_ms=500.0,
            request_id="batch-001",
        )
        assert len(resp.responses) == 2
        assert resp.total_tokens == 20


@requires_fastapi
class TestHealthResponseSchema:
    """Tests for HealthResponse Pydantic model."""

    def test_healthy_response(self):
        from insideLLMs.contrib.deployment import HealthResponse

        resp = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.0,
            checks={"model": True},
        )
        assert resp.status == "healthy"
        assert resp.model_id is None

    def test_unhealthy_with_model_id(self):
        from insideLLMs.contrib.deployment import HealthResponse

        resp = HealthResponse(
            status="unhealthy",
            version="2.0.0",
            model_id="gpt-4",
            uptime_seconds=100.0,
            checks={"db": False},
        )
        assert resp.model_id == "gpt-4"
        assert resp.checks["db"] is False


@requires_fastapi
class TestErrorResponseSchema:
    """Tests for ErrorResponse Pydantic model."""

    def test_basic_error(self):
        from insideLLMs.contrib.deployment import ErrorResponse

        err = ErrorResponse(
            error="Something went wrong",
            error_code="INTERNAL_ERROR",
            request_id="req-err",
        )
        assert err.error == "Something went wrong"
        assert err.details is None

    def test_error_with_details(self):
        from insideLLMs.contrib.deployment import ErrorResponse

        err = ErrorResponse(
            error="Rate limited",
            error_code="RATE_LIMITED",
            request_id="req-rl",
            details={"retry_after": 60},
        )
        assert err.details["retry_after"] == 60


# =============================================================================
# DeploymentApp Route Handler Tests via TestClient
# =============================================================================


@requires_fastapi
class TestDeploymentAppRoutes:
    """Test FastAPI route handlers through TestClient."""

    def _make_client(self, model=None, config=None):
        from fastapi.testclient import TestClient

        from insideLLMs.contrib.deployment import DeploymentApp

        if model is None:
            model = Mock(spec=["generate"])
            model.generate.return_value = "Mock response"
        deployment = DeploymentApp(model, config)
        app = deployment.build_app()
        return TestClient(app), deployment

    def test_root_endpoint(self):
        client, _ = self._make_client()
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "insideLLMs API"
        assert data["version"] == "1.0.0"

    def test_root_endpoint_with_model_id(self):
        model = Mock(spec=["generate", "model_id"])
        model.generate.return_value = "response"
        model.model_id = "test-model-v1"
        client, _ = self._make_client(model=model)
        resp = client.get("/")
        assert resp.json()["model_id"] == "test-model-v1"

    def test_health_endpoint(self):
        client, _ = self._make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert data["version"] == "1.0.0"

    def test_health_endpoint_with_checks(self):
        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        model.generate.return_value = "resp"
        deployment = DeploymentApp(model)
        deployment.add_health_check("ok_check", lambda: True)
        deployment.add_health_check("bad_check", lambda: False)

        from fastapi.testclient import TestClient

        app = deployment.build_app()
        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["ok_check"] is True
        assert data["checks"]["bad_check"] is False

    def test_metrics_endpoint(self):
        client, _ = self._make_client()
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "counters" in data
        assert "gauges" in data
        assert "histograms" in data
        assert "uptime_seconds" in data

    def test_generate_endpoint_success(self):
        model = Mock(spec=["generate"])
        model.generate.return_value = "Generated text"
        client, _ = self._make_client(model=model)
        resp = client.post("/generate", json={"prompt": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Generated text"
        assert "latency_ms" in data
        assert "request_id" in data

    def test_generate_endpoint_with_params(self):
        model = Mock(spec=["generate"])
        model.generate.return_value = "Custom response"
        client, _ = self._make_client(model=model)
        resp = client.post(
            "/generate",
            json={"prompt": "Test", "temperature": 0.5, "max_tokens": 50},
        )
        assert resp.status_code == 200
        # Verify model was called with correct params
        model.generate.assert_called_once()
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 50

    def test_generate_endpoint_model_error(self):
        model = Mock(spec=["generate"])
        model.generate.side_effect = Exception("Model crashed")
        client, _ = self._make_client(model=model)
        resp = client.post("/generate", json={"prompt": "Hello"})
        assert resp.status_code == 500

    def test_generate_endpoint_invalid_request(self):
        client, _ = self._make_client()
        # Missing required 'prompt' field
        resp = client.post("/generate", json={})
        assert resp.status_code == 422

    def test_generate_endpoint_invalid_temperature(self):
        client, _ = self._make_client()
        resp = client.post("/generate", json={"prompt": "x", "temperature": 5.0})
        assert resp.status_code == 422

    def test_batch_endpoint_success(self):
        model = Mock(spec=["generate"])
        model.generate.side_effect = ["R1", "R2", "R3"]
        client, _ = self._make_client(model=model)
        resp = client.post("/batch", json={"prompts": ["A", "B", "C"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["responses"] == ["R1", "R2", "R3"]
        assert "total_tokens" in data
        assert "request_id" in data

    def test_batch_endpoint_exceeds_max_size(self):
        model = Mock(spec=["generate"])
        config = AppConfig(deployment=DeploymentConfig(max_batch_size=2))
        client, _ = self._make_client(model=model, config=config)
        resp = client.post("/batch", json={"prompts": ["A", "B", "C"]})
        assert resp.status_code == 400

    def test_batch_endpoint_model_error(self):
        model = Mock(spec=["generate"])
        model.generate.side_effect = ["R1", Exception("fail"), "R3"]
        client, _ = self._make_client(model=model)
        resp = client.post("/batch", json={"prompts": ["A", "B", "C"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "Error" in data["responses"][1]

    def test_probe_endpoint_success(self):
        model = Mock(spec=["generate"])
        model.generate.return_value = "Model response"

        from insideLLMs.contrib.deployment import DeploymentApp

        deployment = DeploymentApp(model)
        probe = Mock()
        probe.evaluate.return_value = {"score": 0.8}
        deployment.add_probe("test_probe", probe)

        from fastapi.testclient import TestClient

        app = deployment.build_app()
        client = TestClient(app)
        resp = client.post(
            "/probe",
            json={"prompts": ["Tell me something"], "probe_type": "test_probe"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["probe"] == "test_probe"

    def test_probe_endpoint_all_probes(self):
        model = Mock(spec=["generate"])
        model.generate.return_value = "Model response"

        from insideLLMs.contrib.deployment import DeploymentApp

        deployment = DeploymentApp(model)
        probe1 = Mock()
        probe1.evaluate.return_value = {"score": 0.7}
        probe2 = Mock()
        probe2.evaluate.return_value = {"score": 0.9}
        deployment.add_probe("p1", probe1)
        deployment.add_probe("p2", probe2)

        from fastapi.testclient import TestClient

        app = deployment.build_app()
        client = TestClient(app)
        resp = client.post("/probe", json={"prompts": ["Test"]})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 2

    def test_request_id_header(self):
        client, _ = self._make_client()
        resp = client.get("/")
        assert "X-Request-ID" in resp.headers

    def test_generate_increments_metrics(self):
        model = Mock(spec=["generate"])
        model.generate.return_value = "OK"
        client, deployment = self._make_client(model=model)
        client.post("/generate", json={"prompt": "test"})
        metrics = deployment._metrics.get_metrics()
        assert metrics["counters"].get("requests_total", 0) >= 1
        assert metrics["counters"].get("requests_success", 0) >= 1

    def test_generate_error_increments_error_metric(self):
        model = Mock(spec=["generate"])
        model.generate.side_effect = Exception("boom")
        client, deployment = self._make_client(model=model)
        client.post("/generate", json={"prompt": "test"})
        metrics = deployment._metrics.get_metrics()
        assert metrics["counters"].get("requests_error", 0) >= 1

    def test_batch_increments_metrics(self):
        model = Mock(spec=["generate"])
        model.generate.return_value = "OK"
        client, deployment = self._make_client(model=model)
        client.post("/batch", json={"prompts": ["a"]})
        metrics = deployment._metrics.get_metrics()
        assert metrics["counters"].get("batch_requests_total", 0) >= 1

    def test_probe_increments_metrics(self):
        model = Mock(spec=["generate"])
        model.generate.return_value = "resp"
        from insideLLMs.contrib.deployment import DeploymentApp

        deployment = DeploymentApp(model)
        from fastapi.testclient import TestClient

        app = deployment.build_app()
        client = TestClient(app)
        client.post("/probe", json={"prompts": ["test"]})
        metrics = deployment._metrics.get_metrics()
        assert metrics["counters"].get("probe_requests_total", 0) >= 1


# =============================================================================
# Disabled Endpoints / Features
# =============================================================================


@requires_fastapi
class TestDisabledFeatures:
    """Test deployment with disabled endpoints and features."""

    def test_disabled_generate_endpoint(self):
        from fastapi.testclient import TestClient

        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        config = AppConfig(generate_endpoint=EndpointConfig(path="/generate", enabled=False))
        deployment = DeploymentApp(model, config)
        app = deployment.build_app()
        client = TestClient(app)
        resp = client.post("/generate", json={"prompt": "test"})
        # Should return 404/405 because endpoint is not registered
        assert resp.status_code in (404, 405)

    def test_disabled_batch_endpoint(self):
        from fastapi.testclient import TestClient

        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        config = AppConfig(batch_endpoint=EndpointConfig(path="/batch", enabled=False))
        deployment = DeploymentApp(model, config)
        app = deployment.build_app()
        client = TestClient(app)
        resp = client.post("/batch", json={"prompts": ["a"]})
        assert resp.status_code in (404, 405)

    def test_disabled_probe_endpoint(self):
        from fastapi.testclient import TestClient

        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        config = AppConfig(probe_endpoint=EndpointConfig(path="/probe", enabled=False))
        deployment = DeploymentApp(model, config)
        app = deployment.build_app()
        client = TestClient(app)
        resp = client.post("/probe", json={"prompts": ["a"]})
        assert resp.status_code in (404, 405)

    def test_disabled_health_endpoint(self):
        from fastapi.testclient import TestClient

        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        config = AppConfig(deployment=DeploymentConfig(enable_health=False))
        deployment = DeploymentApp(model, config)
        app = deployment.build_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 404

    def test_disabled_metrics_endpoint(self):
        from fastapi.testclient import TestClient

        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        config = AppConfig(deployment=DeploymentConfig(enable_metrics=False))
        deployment = DeploymentApp(model, config)
        app = deployment.build_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 404

    def test_disabled_docs(self):
        from fastapi.testclient import TestClient

        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        config = AppConfig(deployment=DeploymentConfig(enable_docs=False))
        deployment = DeploymentApp(model, config)
        app = deployment.build_app()
        client = TestClient(app)
        resp = client.get("/docs")
        assert resp.status_code == 404
        resp2 = client.get("/redoc")
        assert resp2.status_code == 404


# =============================================================================
# DeploymentApp with Authentication
# =============================================================================


@requires_fastapi
class TestDeploymentAppAuth:
    """Test authentication enforcement via the DeploymentApp."""

    def test_deployment_with_api_key_sets_auth(self):
        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        config = AppConfig(deployment=DeploymentConfig(api_key="my-secret-key"))
        deployment = DeploymentApp(model, config)
        assert deployment._auth.validate("my-secret-key")
        assert not deployment._auth.validate("wrong-key")

    def test_deployment_without_api_key_open_access(self):
        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        deployment = DeploymentApp(model)
        # No key configured means open access
        assert deployment._auth.validate("anything")
        assert deployment._auth.validate(None)


# =============================================================================
# Rate Limiting in Routes
# =============================================================================


@requires_fastapi
class TestRateLimitingInRoutes:
    """Test that rate limiting works within the generate endpoint."""

    def test_rate_limiting_triggers_429(self):
        from fastapi.testclient import TestClient

        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        model.generate.return_value = "resp"
        deployment = DeploymentApp(model)
        app = deployment.build_app()
        client = TestClient(app)

        # The route uses request_id (UUID) as rate limit key, so each request
        # gets a fresh bucket. To test the 429 path, mock is_allowed to return False.
        deployment._rate_limiter.is_allowed = Mock(return_value=False)

        resp = client.post("/generate", json={"prompt": "test"})
        assert resp.status_code == 429


# =============================================================================
# quick_deploy() Tests
# =============================================================================


@requires_fastapi
class TestQuickDeploy:
    """Tests for quick_deploy convenience function."""

    def test_quick_deploy_calls_uvicorn_run(self):
        from insideLLMs.contrib.deployment import quick_deploy

        model = Mock(spec=["generate"])
        model.generate.return_value = "resp"

        mock_uvicorn = Mock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            quick_deploy(model, host="127.0.0.1", port=9000)
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[1]["host"] == "127.0.0.1"
            assert call_kwargs[1]["port"] == 9000

    def test_quick_deploy_passes_kwargs_to_create_app(self):
        from insideLLMs.contrib.deployment import quick_deploy

        model = Mock(spec=["generate"])
        model.generate.return_value = "resp"

        mock_uvicorn = Mock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            quick_deploy(model, title="Custom Title")
            mock_uvicorn.run.assert_called_once()
            app = mock_uvicorn.run.call_args[0][0]
            assert app.title == "Custom Title"

    def test_quick_deploy_missing_uvicorn(self):
        import builtins

        from insideLLMs.contrib.deployment import quick_deploy

        model = Mock(spec=["generate"])

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "uvicorn":
                raise ImportError("no uvicorn")
            return real_import(name, *args, **kwargs)

        # Remove uvicorn from sys.modules so the local import triggers
        import sys

        saved = sys.modules.pop("uvicorn", None)
        try:
            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ImportError, match="uvicorn required"):
                    quick_deploy(model)
        finally:
            if saved is not None:
                sys.modules["uvicorn"] = saved


# =============================================================================
# create_app() Tests
# =============================================================================


@requires_fastapi
class TestCreateApp:
    """Tests for create_app convenience function."""

    def test_create_app_default_params(self):
        model = Mock(spec=["generate"])
        model.generate.return_value = "resp"
        app = create_app(model)
        assert app.title == "insideLLMs API"
        assert app.version == "1.0.0"

    def test_create_app_custom_params(self):
        model = Mock(spec=["generate"])
        model.generate.return_value = "resp"
        app = create_app(
            model,
            title="Custom API",
            description="My custom API",
            version="2.0.0",
            cors_origins=["http://localhost"],
            enable_docs=False,
            enable_metrics=False,
            api_key="secret",
        )
        assert app.title == "Custom API"
        assert app.version == "2.0.0"
        # docs should be disabled
        assert app.docs_url is None
        assert app.redoc_url is None

    def test_create_app_with_cors_origins_none(self):
        model = Mock(spec=["generate"])
        model.generate.return_value = "resp"
        # Should default to ["*"]
        app = create_app(model, cors_origins=None)
        assert app is not None


# =============================================================================
# DeploymentApp.app Property (Lazy Initialization)
# =============================================================================


@requires_fastapi
class TestDeploymentAppProperty:
    """Test the lazy app property on DeploymentApp."""

    def test_app_property_builds_on_first_access(self):
        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        deployment = DeploymentApp(model)
        assert deployment._app is None
        app = deployment.app
        assert app is not None
        assert deployment._app is app

    def test_app_property_returns_same_instance(self):
        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        deployment = DeploymentApp(model)
        app1 = deployment.app
        app2 = deployment.app
        assert app1 is app2

    def test_build_app_can_be_called_explicitly(self):
        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        deployment = DeploymentApp(model)
        app = deployment.build_app()
        assert deployment.app is app


# =============================================================================
# ProbeEndpoint Edge Cases
# =============================================================================


@requires_fastapi
class TestProbeEndpointEdgeCases:
    """Test edge cases in ProbeEndpoint."""

    @pytest.mark.asyncio
    async def test_probe_without_evaluate_method(self):
        """Test probe that lacks evaluate method."""
        model = Mock(spec=["generate"])
        model.generate.return_value = "Response"

        probe = Mock(spec=[])  # No evaluate method
        endpoint = ProbeEndpoint(model, probes={"bad_probe": probe})
        result = await endpoint.run_probe(["Test prompt"], probe_type="bad_probe")

        assert len(result["results"]) == 1
        probe_result = result["results"][0]
        assert probe_result["probe"] == "bad_probe"
        assert "error" in probe_result["results"][0]
        assert "does not support evaluation" in probe_result["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_probe_raises_exception(self):
        """Test probe that raises an exception during evaluation."""
        model = Mock(spec=["generate"])
        model.generate.return_value = "Response"

        probe = Mock()
        probe.evaluate.side_effect = RuntimeError("Probe crashed")
        endpoint = ProbeEndpoint(model, probes={"crash_probe": probe})
        result = await endpoint.run_probe(["Test"], probe_type="crash_probe")

        assert len(result["results"]) == 1
        assert "error" in result["results"][0]
        assert "Probe crashed" in result["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_probe_nonexistent_type_runs_all(self):
        """When probe_type doesn't match any registered probe, run all probes."""
        model = Mock(spec=["generate"])
        model.generate.return_value = "Resp"

        probe = Mock()
        probe.evaluate.return_value = {"score": 0.5}
        endpoint = ProbeEndpoint(model, probes={"existing": probe})
        # probe_type "nonexistent" doesn't match, falls through to all probes
        result = await endpoint.run_probe(["Test"], probe_type="nonexistent")
        # Since "nonexistent" is not in probes dict, the ternary goes to all probes
        assert len(result["results"]) == 1
        assert result["results"][0]["probe"] == "existing"

    @pytest.mark.asyncio
    async def test_probe_with_multiple_prompts(self):
        """Test probe with multiple prompts."""
        model = Mock(spec=["generate"])
        model.generate.return_value = "Model response"

        probe = Mock()
        probe.evaluate.return_value = {"score": 0.9}
        endpoint = ProbeEndpoint(model, probes={"p": probe})
        result = await endpoint.run_probe(["P1", "P2", "P3"], probe_type="p")
        assert len(result["results"][0]["results"]) == 3

    @pytest.mark.asyncio
    async def test_probe_empty_probes_dict(self):
        """Test running probes when none are registered."""
        model = Mock(spec=["generate"])
        endpoint = ProbeEndpoint(model)
        result = await endpoint.run_probe(["Test"])
        assert result["results"] == []


# =============================================================================
# BatchEndpoint Edge Cases
# =============================================================================


@requires_fastapi
class TestBatchEndpointEdgeCases:
    """Test edge cases in BatchEndpoint."""

    @pytest.mark.asyncio
    async def test_batch_with_async_model(self):
        """Test batch endpoint with a model that has agenerate."""
        model = Mock()
        model.agenerate = AsyncMock(side_effect=["AR1", "AR2"])
        endpoint = BatchEndpoint(model, max_batch_size=10)
        result = await endpoint.generate_batch(["P1", "P2"])
        assert result["responses"] == ["AR1", "AR2"]
        assert model.agenerate.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_empty_prompts(self):
        """Test batch with empty prompts list."""
        model = Mock(spec=["generate"])
        endpoint = BatchEndpoint(model, max_batch_size=10)
        result = await endpoint.generate_batch([])
        assert result["responses"] == []
        assert result["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_batch_single_prompt(self):
        """Test batch with single prompt."""
        model = Mock(spec=["generate"])
        model.generate.return_value = "Single response"
        endpoint = BatchEndpoint(model, max_batch_size=10)
        result = await endpoint.generate_batch(["Single"])
        assert len(result["responses"]) == 1
        assert result["responses"][0] == "Single response"

    @pytest.mark.asyncio
    async def test_batch_with_temperature_and_max_tokens(self):
        """Test batch passes temperature to model."""
        model = Mock(spec=["generate"])
        model.generate.return_value = "Resp"
        endpoint = BatchEndpoint(model, max_batch_size=10)
        await endpoint.generate_batch(["P1"], temperature=0.3, max_tokens=50)
        model.generate.assert_called_once()
        call_args = model.generate.call_args
        assert call_args[1]["temperature"] == 0.3


# =============================================================================
# ModelEndpoint Edge Cases
# =============================================================================


@requires_fastapi
class TestModelEndpointEdgeCases:
    """Test edge cases in ModelEndpoint."""

    @pytest.mark.asyncio
    async def test_generate_with_async_coroutine_generate(self):
        """Test model whose generate is a coroutine function."""
        model = Mock()
        # Remove agenerate so the second branch is tested
        if hasattr(model, "agenerate"):
            del model.agenerate

        async def async_generate(prompt, temperature=0.7, max_tokens=None, **kwargs):
            return "Async coroutine result"

        model.generate = async_generate
        model.spec = ["generate"]

        endpoint = ModelEndpoint(model)
        result = await endpoint.generate("Test")
        assert result["response"] == "Async coroutine result"

    @pytest.mark.asyncio
    async def test_generate_no_model_id(self):
        """Test model without model_id attribute."""
        model = Mock(spec=["generate"])
        model.generate.return_value = "Response"
        endpoint = ModelEndpoint(model)
        result = await endpoint.generate("Test")
        assert result["model_id"] is None

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self):
        """Test that extra kwargs are passed through."""
        model = Mock(spec=["generate"])
        model.generate.return_value = "Response"
        endpoint = ModelEndpoint(model)
        await endpoint.generate("Test", stop_sequences=["\n"])
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["stop_sequences"] == ["\n"]

    def test_get_stats_no_requests(self):
        """Test stats with zero requests."""
        model = Mock(spec=["generate"])
        endpoint = ModelEndpoint(model)
        stats = endpoint.get_stats()
        assert stats["request_count"] == 0
        assert stats["error_count"] == 0
        assert stats["average_latency_ms"] == 0

    @pytest.mark.asyncio
    async def test_generate_async_model_error(self):
        """Test error in async model."""
        model = Mock()
        model.agenerate = AsyncMock(side_effect=Exception("Async boom"))
        endpoint = ModelEndpoint(model)
        with pytest.raises(RuntimeError, match="Generation failed"):
            await endpoint.generate("Test")
        assert endpoint._error_count == 1


# =============================================================================
# RequestLogger Edge Cases
# =============================================================================


class TestRequestLoggerEdgeCases:
    """Test edge cases in RequestLogger."""

    def test_short_api_key_masking(self):
        """Test API key masking when key is 4 chars or less."""
        rl = RequestLogger()
        rl.log_request(
            method="POST",
            path="/generate",
            status_code=200,
            latency_ms=10.0,
            request_id="req",
            api_key="abcd",
        )
        logs = rl.get_logs()
        assert logs[0]["api_key"] == "***"

    def test_five_char_api_key_masking(self):
        """Test API key masking when key is exactly 5 chars."""
        rl = RequestLogger()
        rl.log_request(
            method="POST",
            path="/generate",
            status_code=200,
            latency_ms=10.0,
            request_id="req",
            api_key="abcde",
        )
        logs = rl.get_logs()
        assert logs[0]["api_key"] == "***bcde"

    def test_log_with_client_ip(self):
        """Test logging with client IP."""
        rl = RequestLogger()
        rl.log_request(
            method="GET",
            path="/health",
            status_code=200,
            latency_ms=5.0,
            request_id="req",
            client_ip="192.168.1.100",
        )
        logs = rl.get_logs()
        assert logs[0]["client_ip"] == "192.168.1.100"

    def test_log_without_api_key_no_key_field(self):
        """Test that api_key field is not set when no key provided."""
        rl = RequestLogger()
        rl.log_request(
            method="GET",
            path="/health",
            status_code=200,
            latency_ms=5.0,
            request_id="req",
        )
        logs = rl.get_logs()
        assert "api_key" not in logs[0]

    def test_body_not_included_by_default(self):
        """Test that body is not logged when include_body is False."""
        rl = RequestLogger()
        rl.log_request(
            method="POST",
            path="/generate",
            status_code=200,
            latency_ms=10.0,
            request_id="req",
            body={"prompt": "secret"},
        )
        logs = rl.get_logs()
        assert "body" not in logs[0]

    def test_response_not_included_by_default(self):
        """Test that response is not logged when include_response is False."""
        rl = RequestLogger()
        rl.log_request(
            method="POST",
            path="/generate",
            status_code=200,
            latency_ms=10.0,
            request_id="req",
            response={"text": "output"},
        )
        logs = rl.get_logs()
        assert "response" not in logs[0]

    def test_log_has_timestamp(self):
        """Test that each log entry has a timestamp."""
        rl = RequestLogger()
        rl.log_request(
            method="GET",
            path="/",
            status_code=200,
            latency_ms=1.0,
            request_id="req",
        )
        logs = rl.get_logs()
        assert "timestamp" in logs[0]

    def test_get_logs_default_limit(self):
        """Test get_logs default limit is 100."""
        rl = RequestLogger()
        for i in range(200):
            rl.log_request(
                method="GET",
                path=f"/p{i}",
                status_code=200,
                latency_ms=1.0,
                request_id=f"r{i}",
            )
        logs = rl.get_logs()
        assert len(logs) == 100
        # Should be the last 100
        assert logs[0]["path"] == "/p100"


# =============================================================================
# KeyedTokenBucketRateLimiter (alias RateLimiter) Additional Tests
# =============================================================================


class TestKeyedTokenBucketRateLimiterExtended:
    """Extended tests for KeyedTokenBucketRateLimiter."""

    def test_default_burst_size(self):
        """Test that default burst_size equals requests_per_minute."""
        limiter = KeyedTokenBucketRateLimiter(requests_per_minute=120)
        assert limiter.burst_size == 120

    def test_custom_burst_size(self):
        """Test custom burst_size."""
        limiter = KeyedTokenBucketRateLimiter(requests_per_minute=120, burst_size=5)
        assert limiter.burst_size == 5

    def test_rate_calculation(self):
        """Test rate is correctly derived from requests_per_minute."""
        limiter = KeyedTokenBucketRateLimiter(requests_per_minute=120)
        assert limiter.rate == 2.0  # 120 / 60

    def test_get_wait_time_new_key(self):
        """Test wait time for a brand new key (has full bucket)."""
        limiter = KeyedTokenBucketRateLimiter(requests_per_minute=60)
        assert limiter.get_wait_time("new_key") == 0.0

    def test_is_allowed_default_key(self):
        """Test is_allowed with default key parameter."""
        limiter = KeyedTokenBucketRateLimiter(requests_per_minute=60)
        assert limiter.is_allowed()  # uses "default" key

    def test_get_wait_time_default_key(self):
        """Test get_wait_time with default key parameter."""
        limiter = KeyedTokenBucketRateLimiter(requests_per_minute=60)
        assert limiter.get_wait_time() == 0.0  # uses "default" key

    def test_alias_is_same_class(self):
        """Test that RateLimiter is an alias for KeyedTokenBucketRateLimiter."""
        assert RateLimiter is KeyedTokenBucketRateLimiter


# =============================================================================
# APIKeyAuth Extended Tests
# =============================================================================


class TestAPIKeyAuthExtended:
    """Extended tests for APIKeyAuth."""

    def test_custom_header_name(self):
        """Test custom header name."""
        auth = APIKeyAuth(valid_keys=["key"], header_name="Authorization")
        assert auth.header_name == "Authorization"

    def test_remove_nonexistent_key(self):
        """Test removing a key that doesn't exist (discard, no error)."""
        auth = APIKeyAuth(valid_keys=["key1"])
        auth.remove_key("nonexistent")  # Should not raise
        assert auth.validate("key1")


# =============================================================================
# DeploymentConfig Extended Tests
# =============================================================================


class TestDeploymentConfigExtended:
    """Extended tests for DeploymentConfig."""

    def test_cors_origins_default(self):
        """Test default CORS origins."""
        config = DeploymentConfig()
        assert config.cors_origins == ["*"]

    def test_to_dict_does_not_include_api_key(self):
        """Test that to_dict omits api_key for security."""
        config = DeploymentConfig(api_key="secret123")
        data = config.to_dict()
        # The api_key is not in to_dict output
        assert "api_key" not in data

    def test_log_requests_default(self):
        """Test default log_requests setting."""
        config = DeploymentConfig()
        assert config.log_requests is True

    def test_max_batch_size_default(self):
        """Test default max_batch_size."""
        config = DeploymentConfig()
        assert config.max_batch_size == 100


# =============================================================================
# EndpointConfig Extended Tests
# =============================================================================


class TestEndpointConfigExtended:
    """Extended tests for EndpointConfig."""

    def test_default_tags_empty(self):
        """Test default tags is empty list."""
        config = EndpointConfig()
        assert config.tags == []

    def test_default_description_empty(self):
        """Test default description is empty."""
        config = EndpointConfig()
        assert config.description == ""

    def test_to_dict_all_fields(self):
        """Test to_dict includes all fields."""
        config = EndpointConfig(
            path="/test",
            method="PUT",
            enabled=False,
            rate_limit=50,
            timeout_seconds=60.0,
            require_auth=True,
            tags=["v1"],
            description="Test endpoint",
        )
        d = config.to_dict()
        assert d["path"] == "/test"
        assert d["method"] == "PUT"
        assert d["enabled"] is False
        assert d["rate_limit"] == 50
        assert d["timeout_seconds"] == 60.0
        assert d["require_auth"] is True
        assert d["tags"] == ["v1"]
        assert d["description"] == "Test endpoint"


# =============================================================================
# AppConfig Extended Tests
# =============================================================================


class TestAppConfigExtended:
    """Extended tests for AppConfig."""

    def test_default_generate_endpoint_config(self):
        """Test default generate endpoint has proper tags/description."""
        config = AppConfig()
        assert config.generate_endpoint.tags == ["generation"]
        assert config.generate_endpoint.description == "Generate text from prompt"

    def test_default_batch_endpoint_config(self):
        """Test default batch endpoint config."""
        config = AppConfig()
        assert config.batch_endpoint.path == "/batch"
        assert config.batch_endpoint.tags == ["generation"]

    def test_default_probe_endpoint_config(self):
        """Test default probe endpoint config."""
        config = AppConfig()
        assert config.probe_endpoint.path == "/probe"
        assert config.probe_endpoint.tags == ["probing"]

    def test_custom_deployment_config(self):
        """Test AppConfig with custom deployment config."""
        config = AppConfig(deployment=DeploymentConfig(title="Custom", port=9000))
        assert config.deployment.title == "Custom"
        assert config.deployment.port == 9000


# =============================================================================
# MetricsCollector Extended Tests
# =============================================================================


class TestMetricsCollectorExtended:
    """Extended tests for MetricsCollector."""

    def test_empty_counters(self):
        """Test fresh MetricsCollector has empty counters."""
        m = MetricsCollector()
        assert m.get_metrics()["counters"] == {}

    def test_empty_gauges(self):
        """Test fresh MetricsCollector has empty gauges."""
        m = MetricsCollector()
        assert m.get_metrics()["gauges"] == {}

    def test_multiple_counters(self):
        """Test tracking multiple independent counters."""
        m = MetricsCollector()
        m.increment("a")
        m.increment("b", 5)
        m.increment("a", 2)
        data = m.get_metrics()
        assert data["counters"]["a"] == 3
        assert data["counters"]["b"] == 5

    def test_multiple_gauges(self):
        """Test tracking multiple gauges."""
        m = MetricsCollector()
        m.set_gauge("x", 10.0)
        m.set_gauge("y", 20.0)
        data = m.get_metrics()
        assert data["gauges"]["x"] == 10.0
        assert data["gauges"]["y"] == 20.0

    def test_multiple_histograms(self):
        """Test tracking multiple histograms."""
        m = MetricsCollector()
        m.record("latency", 10.0)
        m.record("size", 100.0)
        data = m.get_metrics()
        assert "latency" in data["histograms"]
        assert "size" in data["histograms"]

    def test_histogram_single_value(self):
        """Test histogram stats with a single value."""
        m = MetricsCollector()
        m.record("single", 42.0)
        data = m.get_metrics()
        h = data["histograms"]["single"]
        assert h["count"] == 1
        assert h["min"] == 42.0
        assert h["max"] == 42.0
        assert h["mean"] == 42.0


# =============================================================================
# HealthChecker Extended Tests
# =============================================================================


class TestHealthCheckerExtended:
    """Extended tests for HealthChecker."""

    def test_multiple_failing_checks(self):
        """Test health checker with all checks failing."""
        checker = HealthChecker()
        checker.add_check("a", lambda: False)
        checker.add_check("b", lambda: False)
        result = checker.run_checks()
        assert result["status"] == "unhealthy"
        assert result["checks"]["a"] is False
        assert result["checks"]["b"] is False

    def test_exception_message_logged(self):
        """Test that exception in health check is caught gracefully."""
        checker = HealthChecker()
        checker.add_check("err", lambda: (_ for _ in ()).throw(ValueError("bad")))
        result = checker.run_checks()
        assert result["status"] == "unhealthy"
        assert result["checks"]["err"] is False


# =============================================================================
# Integration Tests: DeploymentApp end-to-end workflows
# =============================================================================


@requires_fastapi
class TestDeploymentAppIntegration:
    """Integration tests exercising multiple components together."""

    def test_full_workflow(self):
        """Test building and using a fully-featured deployment."""
        from fastapi.testclient import TestClient

        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate", "model_id"])
        model.generate.return_value = "Integrated response"
        model.model_id = "integration-model"

        config = AppConfig(
            deployment=DeploymentConfig(
                title="Integration Test API",
                version="3.0.0",
            )
        )
        deployment = DeploymentApp(model, config)
        probe = Mock()
        probe.evaluate.return_value = {"score": 1.0}
        deployment.add_probe("quality", probe)
        deployment.add_health_check("model_ready", lambda: True)

        app = deployment.build_app()
        client = TestClient(app)

        # Root
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Integration Test API"
        assert resp.json()["model_id"] == "integration-model"

        # Generate
        resp = client.post("/generate", json={"prompt": "Integration test"})
        assert resp.status_code == 200
        assert resp.json()["response"] == "Integrated response"
        assert resp.json()["model_id"] == "integration-model"

        # Batch
        model.generate.side_effect = ["B1", "B2"]
        resp = client.post("/batch", json={"prompts": ["p1", "p2"]})
        assert resp.status_code == 200
        assert resp.json()["responses"] == ["B1", "B2"]

        # Probe
        model.generate.side_effect = None
        model.generate.return_value = "probed resp"
        resp = client.post(
            "/probe",
            json={"prompts": ["probe test"], "probe_type": "quality"},
        )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["probe"] == "quality"

        # Health
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
        assert resp.json()["version"] == "3.0.0"
        assert resp.json()["checks"]["model_ready"] is True

        # Metrics
        resp = client.get("/metrics")
        assert resp.status_code == 200
        metrics_data = resp.json()
        assert metrics_data["counters"]["requests_total"] >= 1

    def test_custom_endpoint_paths(self):
        """Test deployment with custom endpoint paths."""
        from fastapi.testclient import TestClient

        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        model.generate.return_value = "custom path response"

        config = AppConfig(
            generate_endpoint=EndpointConfig(
                path="/v1/completions",
                tags=["v1"],
                description="V1 completions",
            ),
            batch_endpoint=EndpointConfig(
                path="/v1/batch",
                tags=["v1"],
            ),
            probe_endpoint=EndpointConfig(
                path="/v1/probe",
                tags=["v1"],
            ),
        )
        deployment = DeploymentApp(model, config)
        app = deployment.build_app()
        client = TestClient(app)

        resp = client.post("/v1/completions", json={"prompt": "test"})
        assert resp.status_code == 200
        assert resp.json()["response"] == "custom path response"

        # Old paths should not exist
        resp_old = client.post("/generate", json={"prompt": "test"})
        assert resp_old.status_code in (404, 405)

    def test_deployment_app_without_fastapi_raises(self):
        """Test that DeploymentApp raises ImportError when FastAPI is missing."""
        from insideLLMs.contrib.deployment import DeploymentApp

        model = Mock(spec=["generate"])
        with patch("insideLLMs.contrib.deployment.FASTAPI_AVAILABLE", False):
            with pytest.raises(ImportError, match="FastAPI is required"):
                DeploymentApp(model)

    def test_quick_deploy_without_fastapi_raises(self):
        """Test that quick_deploy raises ImportError when FastAPI is missing."""
        from insideLLMs.contrib.deployment import quick_deploy

        model = Mock(spec=["generate"])
        with patch("insideLLMs.contrib.deployment.FASTAPI_AVAILABLE", False):
            with pytest.raises(ImportError, match="FastAPI required"):
                quick_deploy(model)
