"""Tests for the model adapter factory module."""

import pytest
import time
from unittest.mock import Mock, patch

from insideLLMs.adapters import (
    # Enums
    Provider,
    AdapterStatus,
    ModelCapability,
    # Dataclasses
    ModelInfo,
    AdapterConfig,
    GenerationParams,
    GenerationResult,
    HealthCheckResult,
    ConnectionInfo,
    # Classes
    BaseAdapter,
    MockAdapter,
    ModelRegistry,
    ProviderDetector,
    AdapterFactory,
    FallbackChain,
    AdapterPool,
    ConnectionMonitor,
    # Functions
    create_adapter,
    create_mock_adapter,
    create_fallback_chain,
    create_adapter_pool,
    detect_provider,
    list_providers,
    list_models,
    get_model_info,
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestProvider:
    """Tests for Provider enum."""

    def test_all_providers_exist(self):
        """Test all providers exist."""
        assert Provider.OPENAI.value == "openai"
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.GOOGLE.value == "google"
        assert Provider.COHERE.value == "cohere"
        assert Provider.MISTRAL.value == "mistral"
        assert Provider.GROQ.value == "groq"
        assert Provider.MOCK.value == "mock"


class TestAdapterStatus:
    """Tests for AdapterStatus enum."""

    def test_all_statuses_exist(self):
        """Test all statuses exist."""
        assert AdapterStatus.READY.value == "ready"
        assert AdapterStatus.INITIALIZING.value == "initializing"
        assert AdapterStatus.ERROR.value == "error"
        assert AdapterStatus.RATE_LIMITED.value == "rate_limited"
        assert AdapterStatus.UNAVAILABLE.value == "unavailable"


class TestModelCapability:
    """Tests for ModelCapability enum."""

    def test_all_capabilities_exist(self):
        """Test all capabilities exist."""
        assert ModelCapability.CHAT.value == "chat"
        assert ModelCapability.COMPLETION.value == "completion"
        assert ModelCapability.EMBEDDING.value == "embedding"
        assert ModelCapability.STREAMING.value == "streaming"
        assert ModelCapability.FUNCTION_CALLING.value == "function_calling"
        assert ModelCapability.JSON_MODE.value == "json_mode"


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_create_model_info(self):
        """Test creating model info."""
        info = ModelInfo(
            model_id="gpt-4",
            provider=Provider.OPENAI,
            display_name="GPT-4",
            capabilities=[ModelCapability.CHAT],
            context_window=8192,
        )

        assert info.model_id == "gpt-4"
        assert info.provider == Provider.OPENAI
        assert ModelCapability.CHAT in info.capabilities

    def test_model_info_to_dict(self):
        """Test model info serialization."""
        info = ModelInfo(
            model_id="gpt-4",
            provider=Provider.OPENAI,
            display_name="GPT-4",
            input_price_per_1k=0.03,
        )

        d = info.to_dict()

        assert d["model_id"] == "gpt-4"
        assert d["provider"] == "openai"
        assert d["input_price_per_1k"] == 0.03


class TestAdapterConfig:
    """Tests for AdapterConfig dataclass."""

    def test_create_config(self):
        """Test creating adapter config."""
        config = AdapterConfig(
            model_id="gpt-4",
            provider=Provider.OPENAI,
            api_key="sk-test",
            timeout=30.0,
        )

        assert config.model_id == "gpt-4"
        assert config.provider == Provider.OPENAI
        assert config.api_key == "sk-test"
        assert config.timeout == 30.0

    def test_config_to_dict_excludes_key(self):
        """Test config serialization excludes API key."""
        config = AdapterConfig(
            model_id="gpt-4",
            provider=Provider.OPENAI,
            api_key="sk-test",
        )

        d = config.to_dict()

        assert "api_key" not in d
        assert d["model_id"] == "gpt-4"


class TestGenerationParams:
    """Tests for GenerationParams dataclass."""

    def test_default_params(self):
        """Test default generation params."""
        params = GenerationParams()

        assert params.temperature == 0.7
        assert params.max_tokens == 1024
        assert params.top_p == 1.0
        assert params.stream is False

    def test_custom_params(self):
        """Test custom generation params."""
        params = GenerationParams(
            temperature=0.0,
            max_tokens=100,
            stop_sequences=["END"],
            seed=42,
        )

        assert params.temperature == 0.0
        assert params.max_tokens == 100
        assert "END" in params.stop_sequences
        assert params.seed == 42

    def test_params_to_dict(self):
        """Test params serialization."""
        params = GenerationParams(temperature=0.5)

        d = params.to_dict()

        assert d["temperature"] == 0.5


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_create_result(self):
        """Test creating generation result."""
        result = GenerationResult(
            text="Hello, world!",
            model="gpt-4",
            provider="openai",
            tokens_input=10,
            tokens_output=5,
            latency_ms=150.0,
        )

        assert result.text == "Hello, world!"
        assert result.model == "gpt-4"
        assert result.tokens_input == 10

    def test_result_to_dict(self):
        """Test result serialization."""
        result = GenerationResult(
            text="Test",
            model="gpt-4",
            provider="openai",
        )

        d = result.to_dict()

        assert d["text"] == "Test"
        assert d["model"] == "gpt-4"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_healthy_result(self):
        """Test healthy result."""
        result = HealthCheckResult(
            healthy=True,
            latency_ms=50.0,
        )

        assert result.healthy is True
        assert result.latency_ms == 50.0
        assert result.error is None

    def test_unhealthy_result(self):
        """Test unhealthy result."""
        result = HealthCheckResult(
            healthy=False,
            latency_ms=1000.0,
            error="Connection timeout",
        )

        assert result.healthy is False
        assert result.error == "Connection timeout"


class TestConnectionInfo:
    """Tests for ConnectionInfo dataclass."""

    def test_create_connection_info(self):
        """Test creating connection info."""
        info = ConnectionInfo(
            adapter_id="openai:gpt-4",
            provider=Provider.OPENAI,
            connected=True,
            latency_ms=50.0,
        )

        assert info.connected is True
        assert info.provider == Provider.OPENAI

    def test_connection_info_to_dict(self):
        """Test connection info serialization."""
        info = ConnectionInfo(
            adapter_id="test",
            provider=Provider.MOCK,
            connected=True,
        )

        d = info.to_dict()

        assert d["connected"] is True
        assert d["provider"] == "mock"


# =============================================================================
# MockAdapter Tests
# =============================================================================

class TestMockAdapter:
    """Tests for MockAdapter class."""

    def test_create_mock_adapter(self):
        """Test creating mock adapter."""
        config = AdapterConfig(model_id="test-model", provider=Provider.MOCK)
        adapter = MockAdapter(config)

        assert adapter.model_id == "test-model"
        assert adapter.provider == Provider.MOCK
        assert adapter.status == AdapterStatus.READY

    def test_generate_default_response(self):
        """Test generating default response."""
        config = AdapterConfig(model_id="test", provider=Provider.MOCK)
        adapter = MockAdapter(config)

        result = adapter.generate("Hello")

        assert result.text == "This is a mock response."
        assert result.model == "test"

    def test_set_custom_response(self):
        """Test setting custom response."""
        config = AdapterConfig(model_id="test", provider=Provider.MOCK)
        adapter = MockAdapter(config)
        adapter.set_response("greeting", "Hi there!")

        result = adapter.generate("greeting please")

        assert result.text == "Hi there!"

    def test_set_default_response(self):
        """Test setting default response."""
        config = AdapterConfig(model_id="test", provider=Provider.MOCK)
        adapter = MockAdapter(config)
        adapter.set_default_response("Custom default")

        result = adapter.generate("anything")

        assert result.text == "Custom default"

    def test_generate_chat(self):
        """Test generating from chat messages."""
        config = AdapterConfig(model_id="test", provider=Provider.MOCK)
        adapter = MockAdapter(config)

        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = adapter.generate_chat(messages)

        assert result.text is not None

    def test_max_tokens_respected(self):
        """Test max tokens is respected."""
        config = AdapterConfig(model_id="test", provider=Provider.MOCK)
        adapter = MockAdapter(config)
        adapter.set_default_response("A" * 100)

        params = GenerationParams(max_tokens=10)
        result = adapter.generate("test", params)

        assert len(result.text) <= 10

    def test_request_counting(self):
        """Test request counting."""
        config = AdapterConfig(model_id="test", provider=Provider.MOCK)
        adapter = MockAdapter(config)

        adapter.generate("test 1")
        adapter.generate("test 2")

        stats = adapter.get_stats()

        assert stats["request_count"] == 2


# =============================================================================
# ModelRegistry Tests
# =============================================================================

class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_default_models_registered(self):
        """Test default models are registered."""
        registry = ModelRegistry()

        gpt4 = registry.get("gpt-4o")
        claude = registry.get("claude-3-5-sonnet-20241022")

        assert gpt4 is not None
        assert claude is not None

    def test_register_model(self):
        """Test registering a model."""
        registry = ModelRegistry()

        info = ModelInfo(
            model_id="custom-model",
            provider=Provider.LOCAL,
            display_name="Custom Model",
        )
        registry.register(info)

        retrieved = registry.get("custom-model")

        assert retrieved is not None
        assert retrieved.display_name == "Custom Model"

    def test_alias_resolution(self):
        """Test alias resolution."""
        registry = ModelRegistry()

        # Register with aliases
        info = ModelInfo(
            model_id="model-v2",
            provider=Provider.LOCAL,
            display_name="Model V2",
            aliases=["model", "model-latest"],
        )
        registry.register(info)

        # Resolve alias
        resolved = registry.resolve_alias("model-latest")

        assert resolved == "model-v2"

    def test_get_by_alias(self):
        """Test getting model by alias."""
        registry = ModelRegistry()

        # Claude has aliases
        claude = registry.get("claude-sonnet")

        assert claude is not None
        assert "claude" in claude.model_id.lower()

    def test_list_models(self):
        """Test listing models."""
        registry = ModelRegistry()

        all_models = registry.list_models()

        assert len(all_models) > 0

    def test_list_models_by_provider(self):
        """Test listing models by provider."""
        registry = ModelRegistry()

        openai_models = registry.list_models(Provider.OPENAI)
        anthropic_models = registry.list_models(Provider.ANTHROPIC)

        assert all(m.provider == Provider.OPENAI for m in openai_models)
        assert all(m.provider == Provider.ANTHROPIC for m in anthropic_models)

    def test_get_by_capability(self):
        """Test getting models by capability."""
        registry = ModelRegistry()

        chat_models = registry.get_by_capability(ModelCapability.CHAT)

        assert len(chat_models) > 0
        assert all(ModelCapability.CHAT in m.capabilities for m in chat_models)


# =============================================================================
# ProviderDetector Tests
# =============================================================================

class TestProviderDetector:
    """Tests for ProviderDetector class."""

    def test_detect_openai(self):
        """Test detecting OpenAI models."""
        assert ProviderDetector.detect("gpt-4") == Provider.OPENAI
        assert ProviderDetector.detect("gpt-3.5-turbo") == Provider.OPENAI
        assert ProviderDetector.detect("text-davinci-003") == Provider.OPENAI

    def test_detect_anthropic(self):
        """Test detecting Anthropic models."""
        assert ProviderDetector.detect("claude-3-opus") == Provider.ANTHROPIC
        assert ProviderDetector.detect("claude-2") == Provider.ANTHROPIC

    def test_detect_mistral(self):
        """Test detecting Mistral models."""
        assert ProviderDetector.detect("mistral-large") == Provider.MISTRAL
        assert ProviderDetector.detect("mixtral-8x7b") == Provider.MISTRAL

    def test_detect_google(self):
        """Test detecting Google models."""
        assert ProviderDetector.detect("gemini-pro") == Provider.GOOGLE
        assert ProviderDetector.detect("palm-2") == Provider.GOOGLE

    def test_detect_unknown(self):
        """Test detecting unknown model."""
        assert ProviderDetector.detect("unknown-model") is None


# =============================================================================
# AdapterFactory Tests
# =============================================================================

class TestAdapterFactory:
    """Tests for AdapterFactory class."""

    def test_create_mock_adapter(self):
        """Test creating mock adapter."""
        factory = AdapterFactory()

        adapter = factory.create("test-model", provider=Provider.MOCK)

        assert adapter is not None
        assert adapter.provider == Provider.MOCK

    def test_auto_detect_provider(self):
        """Test auto-detecting provider."""
        factory = AdapterFactory()

        # Mock should work since it doesn't need API keys
        adapter = factory.create("mock-model", provider=Provider.MOCK)

        assert adapter.provider == Provider.MOCK

    def test_set_api_key(self):
        """Test setting API key."""
        factory = AdapterFactory()
        factory.set_api_key(Provider.OPENAI, "sk-test")

        # API key is stored
        assert factory._api_keys[Provider.OPENAI] == "sk-test"

    def test_set_default_config(self):
        """Test setting default config."""
        factory = AdapterFactory()
        factory.set_default_config(Provider.OPENAI, timeout=120.0)

        assert factory._default_configs[Provider.OPENAI]["timeout"] == 120.0

    def test_list_available_providers(self):
        """Test listing available providers."""
        factory = AdapterFactory()

        available = factory.list_available()

        assert Provider.MOCK in available

    def test_get_model_info(self):
        """Test getting model info through factory."""
        factory = AdapterFactory()

        info = factory.get_model_info("gpt-4o")

        assert info is not None
        assert info.provider == Provider.OPENAI

    def test_list_models_through_factory(self):
        """Test listing models through factory."""
        factory = AdapterFactory()

        models = factory.list_models()

        assert len(models) > 0


# =============================================================================
# FallbackChain Tests
# =============================================================================

class TestFallbackChain:
    """Tests for FallbackChain class."""

    def test_create_chain(self):
        """Test creating fallback chain."""
        adapters = [
            MockAdapter(AdapterConfig("model1", Provider.MOCK)),
            MockAdapter(AdapterConfig("model2", Provider.MOCK)),
        ]

        chain = FallbackChain(adapters)

        assert len(chain.adapters) == 2

    def test_generate_uses_first_adapter(self):
        """Test generate uses first adapter."""
        adapter1 = MockAdapter(AdapterConfig("model1", Provider.MOCK))
        adapter1.set_default_response("Response 1")

        adapter2 = MockAdapter(AdapterConfig("model2", Provider.MOCK))
        adapter2.set_default_response("Response 2")

        chain = FallbackChain([adapter1, adapter2])

        result = chain.generate("test")

        assert result.text == "Response 1"
        assert result.model == "model1"

    def test_fallback_on_error(self):
        """Test fallback on error."""
        # First adapter that always fails
        class FailingAdapter(MockAdapter):
            def generate(self, prompt, params=None, **kwargs):
                raise RuntimeError("Simulated failure")

        failing = FailingAdapter(AdapterConfig("failing", Provider.MOCK))
        working = MockAdapter(AdapterConfig("working", Provider.MOCK))
        working.set_default_response("Working response")

        chain = FallbackChain([failing, working])

        result = chain.generate("test")

        assert result.text == "Working response"
        assert chain._fallback_count == 1

    def test_all_fail(self):
        """Test when all adapters fail."""
        class FailingAdapter(MockAdapter):
            def generate(self, prompt, params=None, **kwargs):
                raise RuntimeError("Failed")

        adapters = [
            FailingAdapter(AdapterConfig("fail1", Provider.MOCK)),
            FailingAdapter(AdapterConfig("fail2", Provider.MOCK)),
        ]

        chain = FallbackChain(adapters)

        with pytest.raises(RuntimeError):
            chain.generate("test")

    def test_get_stats(self):
        """Test getting chain stats."""
        adapters = [
            MockAdapter(AdapterConfig("m1", Provider.MOCK)),
            MockAdapter(AdapterConfig("m2", Provider.MOCK)),
        ]

        chain = FallbackChain(adapters)
        chain.generate("test")

        stats = chain.get_stats()

        assert stats["adapter_count"] == 2
        assert "adapters" in stats


# =============================================================================
# AdapterPool Tests
# =============================================================================

class TestAdapterPool:
    """Tests for AdapterPool class."""

    def test_create_pool(self):
        """Test creating adapter pool."""
        adapters = [
            MockAdapter(AdapterConfig("m1", Provider.MOCK)),
            MockAdapter(AdapterConfig("m2", Provider.MOCK)),
        ]

        pool = AdapterPool(adapters)

        assert len(pool.adapters) == 2

    def test_round_robin_selection(self):
        """Test round robin selection."""
        adapters = [
            MockAdapter(AdapterConfig("m1", Provider.MOCK)),
            MockAdapter(AdapterConfig("m2", Provider.MOCK)),
            MockAdapter(AdapterConfig("m3", Provider.MOCK)),
        ]

        pool = AdapterPool(adapters, strategy="round_robin")

        # Should cycle through adapters
        assert pool.get_adapter().model_id == "m1"
        assert pool.get_adapter().model_id == "m2"
        assert pool.get_adapter().model_id == "m3"
        assert pool.get_adapter().model_id == "m1"  # Cycles back

    def test_least_requests_selection(self):
        """Test least requests selection."""
        adapter1 = MockAdapter(AdapterConfig("m1", Provider.MOCK))
        adapter2 = MockAdapter(AdapterConfig("m2", Provider.MOCK))

        # Simulate requests to adapter1
        adapter1._request_count = 10
        adapter2._request_count = 2

        pool = AdapterPool([adapter1, adapter2], strategy="least_requests")

        selected = pool.get_adapter()

        assert selected.model_id == "m2"

    def test_empty_pool(self):
        """Test empty pool raises error."""
        pool = AdapterPool([])

        with pytest.raises(ValueError):
            pool.get_adapter()

    def test_generate_through_pool(self):
        """Test generating through pool."""
        adapters = [
            MockAdapter(AdapterConfig("m1", Provider.MOCK)),
            MockAdapter(AdapterConfig("m2", Provider.MOCK)),
        ]
        adapters[0].set_default_response("Response 1")
        adapters[1].set_default_response("Response 2")

        pool = AdapterPool(adapters, strategy="round_robin")

        result1 = pool.generate("test")
        result2 = pool.generate("test")

        assert result1.model == "m1"
        assert result2.model == "m2"


# =============================================================================
# ConnectionMonitor Tests
# =============================================================================

class TestConnectionMonitor:
    """Tests for ConnectionMonitor class."""

    def test_check_adapter(self):
        """Test checking adapter connection."""
        monitor = ConnectionMonitor()
        adapter = MockAdapter(AdapterConfig("test", Provider.MOCK))

        info = monitor.check_adapter(adapter)

        assert info.connected is True
        assert info.provider == Provider.MOCK

    def test_get_status(self):
        """Test getting all connection statuses."""
        monitor = ConnectionMonitor()
        adapter = MockAdapter(AdapterConfig("test", Provider.MOCK))

        monitor.check_adapter(adapter)
        status = monitor.get_status()

        assert len(status) == 1


# =============================================================================
# Convenience Functions Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_adapter_mock(self):
        """Test create_adapter with mock."""
        # Direct creation with provider specified
        factory = AdapterFactory()
        adapter = factory.create("test", provider=Provider.MOCK)

        assert adapter is not None

    def test_create_mock_adapter_func(self):
        """Test create_mock_adapter function."""
        adapter = create_mock_adapter()

        assert adapter is not None
        assert adapter.provider == Provider.MOCK

    def test_create_mock_adapter_custom_id(self):
        """Test create_mock_adapter with custom ID."""
        adapter = create_mock_adapter("custom-model")

        assert adapter.model_id == "custom-model"

    def test_detect_provider_func(self):
        """Test detect_provider function."""
        assert detect_provider("gpt-4") == Provider.OPENAI
        assert detect_provider("claude-3") == Provider.ANTHROPIC

    def test_list_providers_func(self):
        """Test list_providers function."""
        providers = list_providers()

        assert Provider.OPENAI in providers
        assert Provider.ANTHROPIC in providers

    def test_list_models_func(self):
        """Test list_models function."""
        models = list_models()

        assert len(models) > 0

    def test_list_models_filtered(self):
        """Test list_models with provider filter."""
        models = list_models("openai")

        assert all(m.provider == Provider.OPENAI for m in models)

    def test_get_model_info_func(self):
        """Test get_model_info function."""
        info = get_model_info("gpt-4o")

        assert info is not None
        assert info.provider == Provider.OPENAI


# =============================================================================
# Integration Tests
# =============================================================================

class TestAdaptersIntegration:
    """Integration tests for adapters."""

    def test_full_mock_workflow(self):
        """Test complete workflow with mock adapter."""
        # Create adapter
        adapter = create_mock_adapter("test-model")
        adapter.set_response("hello", "Hello back!")

        # Generate response
        params = GenerationParams(temperature=0.5, max_tokens=100)
        result = adapter.generate("Say hello", params)

        assert result.text == "Hello back!"
        assert result.model == "test-model"

        # Check stats
        stats = adapter.get_stats()
        assert stats["request_count"] == 1

        # Health check
        health = adapter.health_check()
        assert health.healthy is True

    def test_fallback_chain_workflow(self):
        """Test fallback chain workflow."""
        # Create chain
        adapter1 = create_mock_adapter("primary")
        adapter1.set_default_response("Primary response")

        adapter2 = create_mock_adapter("secondary")
        adapter2.set_default_response("Secondary response")

        chain = FallbackChain([adapter1, adapter2])

        # Generate
        result = chain.generate("test")

        assert result.text == "Primary response"

        # Check stats
        stats = chain.get_stats()
        assert stats["adapter_count"] == 2

    def test_adapter_pool_workflow(self):
        """Test adapter pool workflow."""
        # Create pool
        adapters = [create_mock_adapter(f"model-{i}") for i in range(3)]
        pool = AdapterPool(adapters, strategy="round_robin")

        # Generate multiple times
        results = [pool.generate("test") for _ in range(6)]

        # Verify round-robin distribution
        models = [r.model for r in results]
        assert models == ["model-0", "model-1", "model-2", "model-0", "model-1", "model-2"]

    def test_chat_generation(self):
        """Test chat message generation."""
        adapter = create_mock_adapter()
        adapter.set_default_response("I am well, thank you!")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How are you?"},
        ]

        result = adapter.generate_chat(messages)

        assert result.text is not None
        assert result.provider == "mock"
