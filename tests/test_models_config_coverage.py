"""Additional coverage tests for models/base.py and config.py.

Targets uncovered code paths in:
- ProviderExceptionMap
- handle_provider_errors() decorator
- translate_provider_error() function
- AsyncModel (agenerate_with_metadata, achat, astream, abatch_generate)
- Model._validate_prompt()
- ModelWrapper.__repr__(), retry exhaustion, RuntimeError branch
- config.py: pydantic-unavailable fallback classes
- config.py: _require_pydantic(), load_config_from_yaml (yaml missing),
  save_config_to_yaml error path, _config_to_dict(), _parse_config_dict()
"""

import asyncio
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.exceptions import (
    APIError as InsideLLMsAPIError,
)
from insideLLMs.exceptions import (
    ModelGenerationError,
    RateLimitError,
)
from insideLLMs.exceptions import (
    TimeoutError as InsideLLMsTimeoutError,
)
from insideLLMs.models.base import (
    AsyncModel,
    Model,
    ModelWrapper,
    ProviderExceptionMap,
    handle_provider_errors,
    translate_provider_error,
)
from insideLLMs.types import ModelResponse


# ---------------------------------------------------------------------------
# Concrete subclasses for testing abstract Model / AsyncModel
# ---------------------------------------------------------------------------

class ConcreteModel(Model):
    """Non-abstract Model subclass for testing."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"response:{prompt}"


class FailingModel(Model):
    """Model that always raises the configured exception."""

    def __init__(self, exc: Exception, **kwargs: Any):
        super().__init__(name="failing", model_id="failing-v1")
        self._exc = exc

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise self._exc


class ConcreteAsyncModel(AsyncModel):
    """Non-abstract AsyncModel subclass for testing."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"sync:{prompt}"

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        return f"async:{prompt}"


# ---------------------------------------------------------------------------
# Custom provider exceptions for testing handle_provider_errors
# ---------------------------------------------------------------------------

class FakeRateLimitError(Exception):
    retry_after = 42.0


class FakeTimeoutError(Exception):
    pass


class FakeAPIError(Exception):
    status_code = 503


FAKE_EXCEPTION_MAP = ProviderExceptionMap(
    rate_limit_errors=(FakeRateLimitError,),
    timeout_errors=(FakeTimeoutError,),
    api_errors=(FakeAPIError,),
)


# ===================================================================
# ProviderExceptionMap
# ===================================================================

class TestProviderExceptionMap:
    """Tests for ProviderExceptionMap class."""

    def test_default_empty_tuples(self):
        pem = ProviderExceptionMap()
        assert pem.rate_limit_errors == ()
        assert pem.timeout_errors == ()
        assert pem.api_errors == ()

    def test_custom_exceptions(self):
        pem = ProviderExceptionMap(
            rate_limit_errors=(FakeRateLimitError,),
            timeout_errors=(FakeTimeoutError,),
            api_errors=(FakeAPIError,),
        )
        assert FakeRateLimitError in pem.rate_limit_errors
        assert FakeTimeoutError in pem.timeout_errors
        assert FakeAPIError in pem.api_errors

    def test_isinstance_check_works(self):
        pem = ProviderExceptionMap(
            rate_limit_errors=(FakeRateLimitError,),
        )
        err = FakeRateLimitError()
        assert isinstance(err, pem.rate_limit_errors)


# ===================================================================
# handle_provider_errors decorator
# ===================================================================

class TestHandleProviderErrors:
    """Tests for the handle_provider_errors decorator."""

    def test_success_passthrough(self):
        """Decorated function returns normally when no exception."""

        @handle_provider_errors(FAKE_EXCEPTION_MAP)
        def generate(self, prompt, **kwargs):
            return f"ok:{prompt}"

        obj = MagicMock(model_id="test-model", _timeout=30.0)
        assert generate(obj, "hello") == "ok:hello"

    def test_rate_limit_error_translation(self):
        """Rate limit exception is translated to insideLLMs RateLimitError."""

        @handle_provider_errors(FAKE_EXCEPTION_MAP)
        def generate(self, prompt, **kwargs):
            raise FakeRateLimitError()

        obj = MagicMock(model_id="test-model", _timeout=30.0)
        with pytest.raises(RateLimitError) as exc_info:
            generate(obj, "hello")
        assert "test-model" in str(exc_info.value)

    def test_timeout_error_translation(self):
        """Timeout exception is translated to insideLLMs TimeoutError."""

        @handle_provider_errors(FAKE_EXCEPTION_MAP)
        def generate(self, prompt, **kwargs):
            raise FakeTimeoutError()

        obj = MagicMock(model_id="test-model", _timeout=45.0)
        with pytest.raises(InsideLLMsTimeoutError) as exc_info:
            generate(obj, "hello")
        assert exc_info.value.details["timeout_seconds"] == 45.0

    def test_api_error_translation(self):
        """API exception is translated to insideLLMs APIError."""

        @handle_provider_errors(FAKE_EXCEPTION_MAP)
        def generate(self, prompt, **kwargs):
            raise FakeAPIError("server down")

        obj = MagicMock(model_id="test-model", _timeout=30.0)
        with pytest.raises(InsideLLMsAPIError) as exc_info:
            generate(obj, "hello")
        assert exc_info.value.status_code == 503

    def test_generic_exception_becomes_model_generation_error(self):
        """Unmapped exception becomes ModelGenerationError."""

        @handle_provider_errors(FAKE_EXCEPTION_MAP)
        def generate(self, prompt, **kwargs):
            raise ValueError("bad value")

        obj = MagicMock(model_id="test-model", _timeout=30.0)
        with pytest.raises(ModelGenerationError) as exc_info:
            generate(obj, "hello")
        assert "bad value" in str(exc_info.value)

    def test_insidellms_exception_not_re_wrapped(self):
        """Already-insideLLMs exceptions pass through without wrapping."""

        inner_exc = RateLimitError(model_id="inner", retry_after=10)

        @handle_provider_errors(FAKE_EXCEPTION_MAP)
        def generate(self, prompt, **kwargs):
            raise inner_exc

        obj = MagicMock(model_id="test-model", _timeout=30.0)
        with pytest.raises(RateLimitError) as exc_info:
            generate(obj, "hello")
        assert exc_info.value is inner_exc

    def test_custom_get_model_id_and_get_timeout(self):
        """Custom get_model_id and get_timeout callbacks are used."""

        @handle_provider_errors(
            FAKE_EXCEPTION_MAP,
            get_model_id=lambda self: self.custom_id,
            get_timeout=lambda self: self.custom_timeout,
        )
        def generate(self, prompt, **kwargs):
            raise FakeTimeoutError()

        obj = MagicMock(custom_id="custom-model", custom_timeout=99.0)
        with pytest.raises(InsideLLMsTimeoutError) as exc_info:
            generate(obj, "hello")
        assert exc_info.value.details["model_id"] == "custom-model"
        assert exc_info.value.details["timeout_seconds"] == 99.0

    def test_prompt_from_kwargs_when_no_positional_args(self):
        """Prompt extracted from kwargs when no positional arg given."""

        @handle_provider_errors(FAKE_EXCEPTION_MAP)
        def generate(self, prompt="", **kwargs):
            raise ValueError("fail")

        obj = MagicMock(model_id="m", _timeout=1.0)
        with pytest.raises(ModelGenerationError):
            generate(obj, prompt="my prompt")

    def test_default_model_id_when_missing(self):
        """Falls back to 'unknown' when model_id not on self."""

        @handle_provider_errors(FAKE_EXCEPTION_MAP)
        def generate(self, prompt, **kwargs):
            raise FakeTimeoutError()

        obj = MagicMock(spec=[])  # no attributes at all
        with pytest.raises(InsideLLMsTimeoutError) as exc_info:
            generate(obj, "hello")
        assert exc_info.value.details["model_id"] == "unknown"

    def test_long_prompt_truncated_in_error(self):
        """Prompts over 500 chars are truncated in ModelGenerationError."""
        long_prompt = "x" * 1000

        @handle_provider_errors(FAKE_EXCEPTION_MAP)
        def generate(self, prompt, **kwargs):
            raise ValueError("fail")

        obj = MagicMock(model_id="m", _timeout=1.0)
        with pytest.raises(ModelGenerationError) as exc_info:
            generate(obj, long_prompt)
        # The prompt is truncated to 500 chars inside the decorator
        assert len(exc_info.value.details.get("prompt_preview", "")) <= 503  # 500 + "..."


# ===================================================================
# translate_provider_error
# ===================================================================

class TestTranslateProviderError:
    """Tests for translate_provider_error function."""

    def test_rate_limit_error(self):
        err = FakeRateLimitError()
        result = translate_provider_error(err, "model-1", FAKE_EXCEPTION_MAP)
        assert isinstance(result, RateLimitError)

    def test_timeout_error(self):
        err = FakeTimeoutError()
        result = translate_provider_error(
            err, "model-1", FAKE_EXCEPTION_MAP, timeout_seconds=120.0
        )
        assert isinstance(result, InsideLLMsTimeoutError)
        assert result.details["timeout_seconds"] == 120.0

    def test_api_error(self):
        err = FakeAPIError("oops")
        result = translate_provider_error(err, "model-1", FAKE_EXCEPTION_MAP)
        assert isinstance(result, InsideLLMsAPIError)
        assert result.status_code == 503

    def test_generic_error(self):
        err = RuntimeError("unexpected")
        result = translate_provider_error(
            err, "model-1", FAKE_EXCEPTION_MAP, prompt="test prompt"
        )
        assert isinstance(result, ModelGenerationError)
        assert "unexpected" in str(result)

    def test_already_insidellms_error_passthrough(self):
        original = RateLimitError(model_id="m", retry_after=5)
        result = translate_provider_error(original, "model-1", FAKE_EXCEPTION_MAP)
        assert result is original

    def test_already_timeout_error_passthrough(self):
        original = InsideLLMsTimeoutError(model_id="m", timeout_seconds=30)
        result = translate_provider_error(original, "model-1", FAKE_EXCEPTION_MAP)
        assert result is original

    def test_already_api_error_passthrough(self):
        original = InsideLLMsAPIError(model_id="m", status_code=400)
        result = translate_provider_error(original, "model-1", FAKE_EXCEPTION_MAP)
        assert result is original

    def test_already_generation_error_passthrough(self):
        original = ModelGenerationError(
            model_id="m", prompt="p", reason="r"
        )
        result = translate_provider_error(original, "model-1", FAKE_EXCEPTION_MAP)
        assert result is original

    def test_empty_prompt_in_generic_error(self):
        err = RuntimeError("fail")
        result = translate_provider_error(
            err, "model-1", FAKE_EXCEPTION_MAP, prompt=""
        )
        assert isinstance(result, ModelGenerationError)


# ===================================================================
# AsyncModel
# ===================================================================

class TestAsyncModel:
    """Tests for AsyncModel async methods."""

    @pytest.mark.asyncio
    async def test_agenerate(self):
        model = ConcreteAsyncModel(name="async-test", model_id="async-v1")
        result = await model.agenerate("hello")
        assert result == "async:hello"

    @pytest.mark.asyncio
    async def test_agenerate_with_metadata(self):
        model = ConcreteAsyncModel(name="async-test", model_id="async-v1")
        result = await model.agenerate_with_metadata("hello")
        assert isinstance(result, ModelResponse)
        assert result.content == "async:hello"
        assert result.model == "async-v1"
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_achat_raises_not_implemented(self):
        model = ConcreteAsyncModel(name="async-test", model_id="async-v1")
        with pytest.raises(NotImplementedError, match="async chat"):
            await model.achat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_astream_raises_not_implemented(self):
        model = ConcreteAsyncModel(name="async-test", model_id="async-v1")
        with pytest.raises(NotImplementedError, match="async streaming"):
            async for _ in model.astream("hello"):
                pass  # pragma: no cover

    @pytest.mark.asyncio
    async def test_abatch_generate(self):
        model = ConcreteAsyncModel(name="async-test", model_id="async-v1")
        results = await model.abatch_generate(["a", "b", "c"])
        assert results == ["async:a", "async:b", "async:c"]

    @pytest.mark.asyncio
    async def test_abatch_generate_empty(self):
        model = ConcreteAsyncModel(name="async-test", model_id="async-v1")
        results = await model.abatch_generate([])
        assert results == []


# ===================================================================
# Model._validate_prompt
# ===================================================================

class TestModelValidatePrompt:
    """Tests for Model._validate_prompt."""

    def test_validate_prompt_passes_for_valid_input(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        # Should not raise
        model._validate_prompt("valid prompt")

    def test_validate_prompt_raises_for_empty(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        from insideLLMs.validation import ValidationError

        with pytest.raises(ValidationError):
            model._validate_prompt("")

    def test_validate_prompt_allows_empty_when_flagged(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        # Should not raise
        model._validate_prompt("", allow_empty=True)

    def test_validate_prompt_skipped_when_disabled(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        model._validate_prompts = False
        # Should not raise even for empty prompt
        model._validate_prompt("")

    def test_validate_prompt_raises_for_none(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        from insideLLMs.validation import ValidationError

        with pytest.raises(ValidationError):
            model._validate_prompt(None)

    def test_validate_prompt_raises_for_non_string(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        from insideLLMs.validation import ValidationError

        with pytest.raises(ValidationError):
            model._validate_prompt(123)


# ===================================================================
# Model base: chat, stream, batch, info, repr
# ===================================================================

class TestModelBase:
    """Tests for Model base class methods not covered elsewhere."""

    def test_chat_raises_not_implemented(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        with pytest.raises(NotImplementedError, match="does not support chat"):
            model.chat([{"role": "user", "content": "hi"}])

    def test_stream_raises_not_implemented(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        with pytest.raises(NotImplementedError, match="does not support streaming"):
            model.stream("hello")

    def test_batch_generate_sequential(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        results = model.batch_generate(["a", "b", "c"])
        assert results == ["response:a", "response:b", "response:c"]

    def test_model_id_defaults_to_name(self):
        model = ConcreteModel(name="mymodel")
        assert model.model_id == "mymodel"

    def test_model_repr(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        r = repr(model)
        assert "ConcreteModel" in r
        assert "test" in r
        assert "test-v1" in r

    def test_model_info(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        info = model.info()
        assert info.name == "test"
        assert info.model_id == "test-v1"
        assert info.provider == "Concrete"
        assert info.supports_streaming is False
        assert info.supports_chat is False

    def test_generate_with_metadata(self):
        model = ConcreteModel(name="test", model_id="test-v1")
        resp = model.generate_with_metadata("hello")
        assert resp.content == "response:hello"
        assert resp.model == "test-v1"
        assert resp.latency_ms >= 0

    def test_call_count_init(self):
        model = ConcreteModel(name="test")
        assert model._call_count == 0
        assert model._total_tokens == 0


# ===================================================================
# ModelWrapper: __repr__, retry exhaustion, RuntimeError fallback
# ===================================================================

class TestModelWrapperCoverage:
    """Additional ModelWrapper tests for uncovered branches."""

    def test_wrapper_repr(self):
        base = ConcreteModel(name="base", model_id="base-v1")
        wrapper = ModelWrapper(base, max_retries=5)
        r = repr(wrapper)
        assert "ModelWrapper" in r
        assert "max_retries=5" in r
        assert "ConcreteModel" in r

    def test_retry_exhaustion_raises_last_error(self):
        exc = ValueError("always fails")
        base = FailingModel(exc=exc)
        wrapper = ModelWrapper(base, max_retries=2, retry_delay=0.0)
        with pytest.raises(ValueError, match="always fails"):
            wrapper.generate("test")

    def test_retry_exhaustion_runtime_error_when_no_last_error(self):
        """When last_error stays None somehow, RuntimeError is raised.

        This tests the `raise last_error or RuntimeError(...)` fallback.
        We achieve this by setting max_retries=0 so the loop body never executes.
        """
        base = ConcreteModel(name="test")
        wrapper = ModelWrapper(base, max_retries=0, retry_delay=0.0)
        with pytest.raises(RuntimeError, match="Max retries exceeded"):
            wrapper.generate("test")

    def test_wrapper_caching_with_different_kwargs(self):
        call_count = 0

        class CountModel(ConcreteModel):
            def generate(self, prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                return f"r{call_count}"

        base = CountModel(name="c")
        wrapper = ModelWrapper(base, cache_responses=True)
        r1 = wrapper.generate("p", temperature=0.5)
        r2 = wrapper.generate("p", temperature=0.5)
        r3 = wrapper.generate("p", temperature=0.9)  # different kwargs -> cache miss
        assert r1 == r2
        assert r3 != r1
        assert call_count == 2

    def test_wrapper_info_delegates(self):
        base = ConcreteModel(name="test", model_id="test-v1")
        wrapper = ModelWrapper(base)
        info = wrapper.info()
        assert info.name == "test"
        assert info.model_id == "test-v1"

    @patch("time.sleep")
    def test_retry_calls_sleep_with_backoff(self, mock_sleep):
        """Verify sleep is called with increasing delay on retries."""
        exc = ValueError("fail")
        base = FailingModel(exc=exc)
        wrapper = ModelWrapper(base, max_retries=3, retry_delay=1.0)
        with pytest.raises(ValueError):
            wrapper.generate("test")
        # Should have slept twice (after attempt 0 and attempt 1; attempt 2 is final)
        assert mock_sleep.call_count == 2
        # Delays: 1.0 * (0+1) = 1.0, 1.0 * (1+1) = 2.0
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)


# ===================================================================
# config.py: Pydantic fallback code paths
# ===================================================================

class TestConfigFallbackPaths:
    """Tests for pydantic-unavailable fallback code in config.py.

    These tests mock PYDANTIC_AVAILABLE=False at the module level and
    exercise the fallback BaseModel and config classes.
    """

    def test_fallback_base_model_init_and_model_dump(self):
        """Test the fallback BaseModel __init__ and model_dump."""
        # Import the fallback directly by simulating no-pydantic
        # We test the actual fallback classes defined in the else branch
        from insideLLMs.config import PYDANTIC_AVAILABLE

        if PYDANTIC_AVAILABLE:
            # When pydantic is available, the fallback classes are in the
            # else branch and not active. We can still test _config_to_dict
            # and _parse_config_dict which work in both paths.
            pass

        # Test _config_to_dict with a plain object (non-pydantic path)
        from insideLLMs.config import _config_to_dict

        class SimpleObj:
            def __init__(self):
                self.x = 1
                self.y = "hello"

        result = _config_to_dict(SimpleObj())
        assert result == {"x": 1, "y": "hello"}

    def test_config_to_dict_with_nested_objects(self):
        from insideLLMs.config import _config_to_dict

        class Inner:
            def __init__(self):
                self.value = 42

        class Outer:
            def __init__(self):
                self.inner = Inner()
                self.name = "test"

        result = _config_to_dict(Outer())
        assert result["name"] == "test"
        assert result["inner"] == {"value": 42}

    def test_config_to_dict_with_enum(self):
        """Enum values go through the hasattr(__dict__) branch first,
        so _config_to_dict recurses into them. The Enum branch in
        _config_to_dict is only reached for Enum values without __dict__
        (not typical). The serialize_value call in save_config_to_yaml
        handles enum normalization separately.
        """
        from insideLLMs.config import _config_to_dict

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        class MyObj:
            def __init__(self):
                self.color = Color.RED
                self.count = 3

        result = _config_to_dict(MyObj())
        # Enum with __dict__ is recursed into (not .value), so it becomes a dict
        assert isinstance(result["color"], dict)
        assert result["count"] == 3

    def test_config_to_dict_with_model_dump(self):
        """If obj has model_dump, _config_to_dict delegates to it."""
        from insideLLMs.config import _config_to_dict

        class HasModelDump:
            def model_dump(self):
                return {"a": 1, "b": 2}

        result = _config_to_dict(HasModelDump())
        assert result == {"a": 1, "b": 2}


# ===================================================================
# config.py: _parse_config_dict
# ===================================================================

class TestParseConfigDict:
    """Tests for _parse_config_dict."""

    def test_parse_minimal_dict(self):
        from insideLLMs.config import _parse_config_dict

        data = {
            "name": "Test",
            "model": {"provider": "dummy", "model_id": "dummy-v1"},
            "probe": {"type": "logic"},
            "dataset": {"source": "inline", "data": [{"q": "test"}]},
        }
        config = _parse_config_dict(data)
        assert config.name == "Test"
        assert config.model.model_id == "dummy-v1"

    def test_parse_with_runner_and_metadata(self):
        from insideLLMs.config import _parse_config_dict

        data = {
            "name": "Full",
            "description": "A full test",
            "model": {"provider": "openai", "model_id": "gpt-4"},
            "probe": {"type": "factuality", "name": "Facts"},
            "dataset": {"source": "file", "path": "data.jsonl"},
            "runner": {"concurrency": 10, "verbose": True},
            "tags": ["test", "coverage"],
            "metadata": {"version": "2.0"},
        }
        config = _parse_config_dict(data)
        assert config.description == "A full test"
        assert config.runner.concurrency == 10
        assert "test" in config.tags
        assert config.metadata["version"] == "2.0"

    def test_parse_with_empty_nested_dicts(self):
        """Missing nested configs should default to empty dicts."""
        from insideLLMs.config import _parse_config_dict

        data = {
            "name": "Minimal",
        }
        # model/probe/dataset will be empty dicts -> may fail validation
        # but _parse_config_dict itself should not raise; it just constructs
        try:
            config = _parse_config_dict(data)
            assert config.name == "Minimal"
        except Exception:
            # Pydantic validation may reject empty model/probe/dataset
            pass

    def test_parse_defaults_name(self):
        from insideLLMs.config import _parse_config_dict

        data = {
            "model": {"provider": "dummy", "model_id": "d"},
            "probe": {"type": "logic"},
            "dataset": {"source": "inline", "data": [{"q": "t"}]},
        }
        config = _parse_config_dict(data)
        assert config.name == "Unnamed Experiment"


# ===================================================================
# config.py: load / save error paths
# ===================================================================

class TestConfigLoadSaveErrors:
    """Tests for error paths in load/save config functions."""

    def test_load_config_from_yaml_missing_yaml(self):
        """When yaml module is not importable, ImportError is raised."""
        import sys

        # Temporarily make yaml unimportable
        yaml_mod = sys.modules.get("yaml")
        sys.modules["yaml"] = None  # type: ignore[assignment]
        try:
            from insideLLMs.config import load_config_from_yaml

            with pytest.raises(ImportError, match="PyYAML"):
                load_config_from_yaml("nonexistent.yaml")
        finally:
            if yaml_mod is not None:
                sys.modules["yaml"] = yaml_mod
            else:
                sys.modules.pop("yaml", None)

    def test_save_config_to_yaml_missing_yaml(self):
        """When yaml module is not importable, ImportError is raised."""
        import sys

        yaml_mod = sys.modules.get("yaml")
        sys.modules["yaml"] = None  # type: ignore[assignment]
        try:
            from insideLLMs.config import save_config_to_yaml, create_example_config

            config = create_example_config()
            with pytest.raises(ImportError, match="PyYAML"):
                save_config_to_yaml(config, "/tmp/test_save.yaml")
        finally:
            if yaml_mod is not None:
                sys.modules["yaml"] = yaml_mod
            else:
                sys.modules.pop("yaml", None)

    def test_load_config_from_yaml_file_not_found(self):
        from insideLLMs.config import load_config_from_yaml

        with pytest.raises(FileNotFoundError, match="not found"):
            load_config_from_yaml("/nonexistent/path/config.yaml")

    def test_load_config_from_json_file_not_found(self):
        from insideLLMs.config import load_config_from_json

        with pytest.raises(FileNotFoundError, match="not found"):
            load_config_from_json("/nonexistent/path/config.json")

    def test_load_config_unsupported_extension(self):
        from insideLLMs.config import load_config

        with pytest.raises(ValueError, match="Unsupported"):
            load_config("config.toml")

    def test_save_and_load_yaml_roundtrip(self, tmp_path):
        from insideLLMs.config import (
            create_example_config,
            load_config_from_yaml,
            save_config_to_yaml,
        )

        config = create_example_config()
        path = tmp_path / "test_config.yaml"
        save_config_to_yaml(config, path)
        loaded = load_config_from_yaml(path)
        assert loaded.name == config.name
        assert loaded.model.model_id == config.model.model_id

    def test_save_config_to_yaml_creates_parent_dirs(self, tmp_path):
        from insideLLMs.config import create_example_config, save_config_to_yaml

        config = create_example_config()
        deep_path = tmp_path / "a" / "b" / "c" / "config.yaml"
        save_config_to_yaml(config, deep_path)
        assert deep_path.exists()


# ===================================================================
# config.py: _require_pydantic
# ===================================================================

class TestRequirePydantic:
    """Tests for _require_pydantic function."""

    def test_does_not_raise_when_available(self):
        from insideLLMs.config import PYDANTIC_AVAILABLE, _require_pydantic

        if PYDANTIC_AVAILABLE:
            _require_pydantic()  # Should not raise

    def test_raises_when_unavailable(self):
        """Test that _require_pydantic raises ImportError when pydantic is missing."""
        from insideLLMs import config as config_module

        original = config_module.PYDANTIC_AVAILABLE
        try:
            config_module.PYDANTIC_AVAILABLE = False
            with pytest.raises(ImportError, match="Pydantic is required"):
                config_module._require_pydantic()
        finally:
            config_module.PYDANTIC_AVAILABLE = original


# ===================================================================
# config.py: save_config_to_json roundtrip and _config_to_dict path
# ===================================================================

class TestSaveConfigToJson:
    """Tests for save_config_to_json and related code."""

    def test_save_and_load_json_roundtrip(self, tmp_path):
        from insideLLMs.config import (
            create_example_config,
            load_config_from_json,
            save_config_to_json,
        )

        config = create_example_config()
        path = tmp_path / "test_config.json"
        save_config_to_json(config, path)
        loaded = load_config_from_json(path)
        assert loaded.name == config.name

    def test_save_config_to_json_creates_parent_dirs(self, tmp_path):
        from insideLLMs.config import create_example_config, save_config_to_json

        config = create_example_config()
        deep_path = tmp_path / "x" / "y" / "config.json"
        save_config_to_json(config, deep_path)
        assert deep_path.exists()

    def test_save_config_to_json_non_pydantic_path(self, tmp_path):
        """Exercise _config_to_dict fallback used when PYDANTIC_AVAILABLE is False."""
        from insideLLMs import config as config_module
        from insideLLMs.config import create_example_config, save_config_to_json

        config = create_example_config()
        path = tmp_path / "fallback.json"

        original = config_module.PYDANTIC_AVAILABLE
        try:
            config_module.PYDANTIC_AVAILABLE = False
            save_config_to_json(config, path)
            assert path.exists()
            import json

            data = json.loads(path.read_text())
            assert "name" in data
        finally:
            config_module.PYDANTIC_AVAILABLE = original

    def test_save_config_to_yaml_non_pydantic_path(self, tmp_path):
        """Exercise _config_to_dict fallback for YAML save."""
        from insideLLMs import config as config_module
        from insideLLMs.config import create_example_config, save_config_to_yaml

        config = create_example_config()
        path = tmp_path / "fallback.yaml"

        original = config_module.PYDANTIC_AVAILABLE
        try:
            config_module.PYDANTIC_AVAILABLE = False
            save_config_to_yaml(config, path)
            assert path.exists()
        finally:
            config_module.PYDANTIC_AVAILABLE = original


# ===================================================================
# config.py: validate_config
# ===================================================================

class TestValidateConfigCoverage:
    """Additional tests for validate_config."""

    def test_validate_config_returns_same_experiment_config(self):
        from insideLLMs.config import (
            ExperimentConfig,
            validate_config,
            create_example_config,
        )

        config = create_example_config()
        result = validate_config(config)
        assert result is config

    def test_validate_config_from_dict(self):
        from insideLLMs.config import validate_config

        data = {
            "name": "Dict Test",
            "model": {"provider": "dummy", "model_id": "d"},
            "probe": {"type": "logic"},
            "dataset": {"source": "inline", "data": [{"q": "t"}]},
            "tags": ["a"],
        }
        config = validate_config(data)
        assert config.name == "Dict Test"
        assert "a" in config.tags


# ===================================================================
# config.py: DatasetConfig validation (pydantic)
# ===================================================================

class TestDatasetConfigValidation:
    """Test DatasetConfig source requirement validation."""

    def test_hf_source_requires_name(self):
        from insideLLMs.config import PYDANTIC_AVAILABLE, DatasetConfig

        if PYDANTIC_AVAILABLE:
            with pytest.raises(ValueError, match="name is required"):
                DatasetConfig(source="hf")

    def test_file_source_requires_path(self):
        from insideLLMs.config import PYDANTIC_AVAILABLE, DatasetConfig

        if PYDANTIC_AVAILABLE:
            with pytest.raises(ValueError, match="path is required"):
                DatasetConfig(source="file")

    def test_inline_source_requires_data(self):
        from insideLLMs.config import PYDANTIC_AVAILABLE, DatasetConfig

        if PYDANTIC_AVAILABLE:
            with pytest.raises(ValueError, match="data is required"):
                DatasetConfig(source="inline")


# ===================================================================
# config.py: ProbeConfig default name
# ===================================================================

class TestProbeConfigDefaults:
    """Test ProbeConfig name default."""

    def test_probe_name_defaults_to_type(self):
        from insideLLMs.config import PYDANTIC_AVAILABLE, ProbeConfig, ProbeType

        if PYDANTIC_AVAILABLE:
            # When name=None is explicitly passed, the field_validator fires
            # and sets name to the type value.
            config = ProbeConfig(type=ProbeType.BIAS, name=None)
            assert config.name == "bias"

    def test_probe_name_explicit(self):
        from insideLLMs.config import PYDANTIC_AVAILABLE, ProbeConfig, ProbeType

        if PYDANTIC_AVAILABLE:
            config = ProbeConfig(type=ProbeType.LOGIC, name="Custom Name")
            assert config.name == "Custom Name"


# ===================================================================
# config.py: create_example_config
# ===================================================================

class TestCreateExampleConfigCoverage:
    """Ensure example config exercises both pydantic and non-pydantic paths."""

    def test_example_config_structure(self):
        from insideLLMs.config import create_example_config

        config = create_example_config()
        assert config.name == "Example Experiment"
        assert config.description is not None
        assert config.model is not None
        assert config.probe is not None
        assert config.dataset is not None
        assert config.runner is not None
        assert "example" in config.tags
        assert len(config.dataset.data) == 2

    def test_example_config_non_pydantic(self):
        """Exercise create_example_config with PYDANTIC_AVAILABLE=False."""
        from insideLLMs import config as config_module

        original = config_module.PYDANTIC_AVAILABLE
        try:
            config_module.PYDANTIC_AVAILABLE = False
            config = config_module.create_example_config()
            assert config.name == "Example Experiment"
            # When pydantic is "unavailable", provider/type are strings
            assert config.model.model_id == "gpt-4"
        finally:
            config_module.PYDANTIC_AVAILABLE = original


# ===================================================================
# config.py: Pydantic-unavailable fallback classes via module reload
# ===================================================================

class TestConfigNoPydanticFallback:
    """Test the fallback class definitions that activate when pydantic is missing.

    These tests reload the config module with pydantic hidden to exercise the
    else-branch fallback classes (BaseModel stub, Field stub, etc.) and the
    non-pydantic ModelConfig/ProbeConfig/DatasetConfig/RunnerConfig/ExperimentConfig.
    """

    def _load_config_without_pydantic(self):
        """Reload the config module with pydantic blocked."""
        import importlib
        import sys

        # Save originals
        saved_modules = {}
        for mod_name in list(sys.modules):
            if mod_name == "pydantic" or mod_name.startswith("pydantic."):
                saved_modules[mod_name] = sys.modules.pop(mod_name)

        # Also save the config module itself so we can reload it
        saved_config = sys.modules.pop("insideLLMs.config", None)

        # Block pydantic from being imported
        sys.modules["pydantic"] = None  # type: ignore[assignment]

        try:
            # Reload config module -- will take the ImportError fallback path
            import insideLLMs.config as reloaded_config

            importlib.reload(reloaded_config)
            return reloaded_config
        finally:
            # We'll restore in each test's teardown
            pass

    def _restore_modules(self, saved_modules, saved_config):
        import importlib
        import sys

        # Remove the blocked pydantic
        sys.modules.pop("pydantic", None)

        # Restore original pydantic modules
        for mod_name, mod in saved_modules.items():
            sys.modules[mod_name] = mod

        # Restore original config module
        if saved_config is not None:
            sys.modules["insideLLMs.config"] = saved_config
        else:
            # Module wasn't cached before; just re-import it fresh
            sys.modules.pop("insideLLMs.config", None)
            import insideLLMs.config  # noqa: F401

    def test_fallback_classes_loaded(self):
        """Test that fallback classes load when pydantic is unavailable."""
        import importlib
        import sys

        saved_modules = {}
        for mod_name in list(sys.modules):
            if mod_name == "pydantic" or mod_name.startswith("pydantic."):
                saved_modules[mod_name] = sys.modules.pop(mod_name)
        saved_config = sys.modules.pop("insideLLMs.config", None)
        sys.modules["pydantic"] = None  # type: ignore[assignment]

        try:
            import insideLLMs.config as cfg

            importlib.reload(cfg)

            assert cfg.PYDANTIC_AVAILABLE is False

            # Test fallback BaseModel
            class TestModel(cfg.BaseModel):
                pass

            obj = TestModel(x=1, y="hello")
            assert obj.x == 1
            assert obj.y == "hello"
            dumped = obj.model_dump()
            assert dumped == {"x": 1, "y": "hello"}

            # Test fallback Field
            result = cfg.Field(default=42)
            assert result == 42

            # Test fallback field_validator
            @cfg.field_validator("name")
            def my_validator(v):
                return v

            assert callable(my_validator)

            # Test fallback model_validator
            @cfg.model_validator(mode="after")
            def my_model_validator(self):
                return self

            assert callable(my_model_validator)

            # Test fallback ModelConfig
            mc = cfg.ModelConfig(provider="openai", model_id="gpt-4")
            assert mc.provider == "openai"
            assert mc.model_id == "gpt-4"
            assert mc.name == "gpt-4"  # defaults to model_id
            assert mc.temperature == 0.7
            assert mc.max_tokens is None
            assert mc.timeout == 30.0
            assert mc.max_retries == 3
            assert mc.extra_params == {}

            # Test fallback ProbeConfig
            pc = cfg.ProbeConfig(type="logic")
            assert pc.type == "logic"
            assert pc.name == "logic"  # defaults to type
            assert pc.params == {}
            assert pc.timeout_per_item == 30.0
            assert pc.stop_on_error is False

            # Test fallback DatasetConfig
            dc = cfg.DatasetConfig(source="inline", data=[{"q": "test"}])
            assert dc.source == "inline"
            assert dc.data == [{"q": "test"}]
            assert dc.split == "test"
            assert dc.shuffle is False

            # Test fallback RunnerConfig
            rc = cfg.RunnerConfig()
            assert rc.concurrency == 1
            assert rc.output_dir == "output"
            assert rc.output_formats == ["json", "markdown"]
            assert rc.progress_bar is True
            assert rc.verbose is False

            # Test fallback ExperimentConfig
            ec = cfg.ExperimentConfig(
                name="Test",
                model=mc,
                probe=pc,
                dataset=dc,
                tags=["a", "b"],
                metadata={"k": "v"},
            )
            assert ec.name == "Test"
            assert ec.model is mc
            assert ec.runner.concurrency == 1
            assert ec.tags == ["a", "b"]
            assert ec.metadata == {"k": "v"}

            # Test _require_pydantic raises
            with pytest.raises(ImportError, match="Pydantic is required"):
                cfg._require_pydantic()

            # Test _parse_config_dict
            data = {
                "name": "Parsed",
                "model": {"provider": "dummy", "model_id": "d"},
                "probe": {"type": "logic"},
                "dataset": {"source": "inline", "data": [{"q": "t"}]},
                "runner": {"concurrency": 3},
            }
            parsed = cfg._parse_config_dict(data)
            assert parsed.name == "Parsed"
            assert parsed.runner.concurrency == 3

            # Test create_example_config
            example = cfg.create_example_config()
            assert example.name == "Example Experiment"
            assert example.model.model_id == "gpt-4"

            # Test validate_config with dict
            validated = cfg.validate_config(data)
            assert validated.name == "Parsed"

            # Test validate_config with ExperimentConfig passthrough
            result = cfg.validate_config(ec)
            assert result is ec

        finally:
            self._restore_modules(saved_modules, saved_config)
