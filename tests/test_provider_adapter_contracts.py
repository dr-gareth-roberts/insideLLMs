"""Provider adapter behavior against deterministic SDK doubles."""

from __future__ import annotations

import builtins
import sys
import types
from unittest.mock import MagicMock

import pytest

from insideLLMs.exceptions import (
    APIError as InsideLLMsAPIError,
)
from insideLLMs.exceptions import (
    ModelGenerationError,
    ModelInitializationError,
    RateLimitError,
)
from insideLLMs.exceptions import (
    ModelTimeoutError as InsideLLMsTimeoutError,
)


@pytest.fixture()
def anthropic_stub(monkeypatch: pytest.MonkeyPatch):
    anthropic_mod = types.ModuleType("anthropic")

    class APIError(Exception):
        def __init__(self, message="api", status_code=500):
            super().__init__(message)
            self.status_code = status_code

    class APITimeoutError(Exception):
        pass

    class RateLimitErrorSDK(Exception):
        def __init__(self, message="rl"):
            super().__init__(message)
            self.retry_after = 2.0

    class _Block:
        def __init__(self, text=None, as_str=None):
            if text is not None:
                self.text = text
            self._as_str = as_str or "fallback"

        def __str__(self):
            return self._as_str

    class _Resp:
        def __init__(self, text="hi", empty=False, bare_block=False):
            if empty:
                self.content = []
            elif bare_block:
                self.content = [_Block(as_str="bare")]
            else:
                self.content = [_Block(text=text)]

    class _StreamCtx:
        def __init__(self, chunks):
            self.text_stream = iter(chunks)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Anthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = types.SimpleNamespace(
                create=MagicMock(return_value=_Resp("hi")),
                stream=MagicMock(return_value=_StreamCtx(["a", "b"])),
            )

    anthropic_mod.APIError = APIError
    anthropic_mod.APITimeoutError = APITimeoutError
    anthropic_mod.RateLimitError = RateLimitErrorSDK
    anthropic_mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_mod)

    for key in list(sys.modules):
        if key == "insideLLMs.models.anthropic" or key.startswith("insideLLMs.models.anthropic."):
            del sys.modules[key]

    import insideLLMs.models.anthropic as anth

    try:
        yield anth, Anthropic, APIError, APITimeoutError, RateLimitErrorSDK, _Resp, _StreamCtx
    finally:
        for key in list(sys.modules):
            if key == "insideLLMs.models.anthropic" or key.startswith(
                "insideLLMs.models.anthropic."
            ):
                del sys.modules[key]


def test_anthropic_happy_and_errors(anthropic_stub, monkeypatch: pytest.MonkeyPatch) -> None:
    anth, Anthropic, APIError, APITimeoutError, RateLimitErrorSDK, _Resp, _StreamCtx = (
        anthropic_stub
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ModelInitializationError):
        anth.AnthropicModel(api_key=None)

    def boom(**kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(anth.anthropic, "Anthropic", boom)
    with pytest.raises(ModelInitializationError, match="Failed to initialize"):
        anth.AnthropicModel(api_key="sk")
    monkeypatch.setattr(anth.anthropic, "Anthropic", Anthropic)

    m = anth.AnthropicModel(model_name="claude-3", api_key="sk")
    assert m.generate("x") == "hi"
    assert m.chat([{"role": "user", "content": "x"}]) == "hi"
    assert "".join(m.stream("x")) == "ab"
    assert m.info().extra["model_name"] == "claude-3"

    # system message + explicit system= + empty chat + bare block str()
    assert (
        m.chat(
            [
                {"role": "system", "content": "be nice"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "hi"},
            ],
            system="also",
        )
        == "hi"
    )
    m.client.messages.create.return_value = _Resp(empty=True)
    assert m.generate("x") == ""
    assert m.chat([{"role": "user", "content": "x"}]) == ""
    m.client.messages.create.return_value = _Resp(bare_block=True)
    assert m.generate("x") == "bare"

    m.client.messages.create = MagicMock(side_effect=RateLimitErrorSDK())
    with pytest.raises(RateLimitError):
        m.generate("x")
    with pytest.raises(RateLimitError):
        m.chat([{"role": "user", "content": "x"}])
    m.client.messages.stream = MagicMock(side_effect=RateLimitErrorSDK())
    with pytest.raises(RateLimitError):
        list(m.stream("x"))

    m.client.messages.create = MagicMock(side_effect=APITimeoutError())
    m.client.messages.stream = MagicMock(side_effect=APITimeoutError())
    with pytest.raises(InsideLLMsTimeoutError):
        m.generate("x")
    with pytest.raises(InsideLLMsTimeoutError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(InsideLLMsTimeoutError):
        list(m.stream("x"))

    m.client.messages.create = MagicMock(side_effect=APIError("bad", 400))
    m.client.messages.stream = MagicMock(side_effect=APIError("bad", 400))
    with pytest.raises(InsideLLMsAPIError):
        m.generate("x")
    with pytest.raises(InsideLLMsAPIError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(InsideLLMsAPIError):
        list(m.stream("x"))

    m.client.messages.create = MagicMock(side_effect=ValueError("x"))
    m.client.messages.stream = MagicMock(side_effect=ValueError("x"))
    with pytest.raises(ModelGenerationError):
        m.generate("x")
    with pytest.raises(ModelGenerationError):
        m.chat([])
    with pytest.raises(ModelGenerationError):
        list(m.stream("x"))


@pytest.fixture()
def cohere_stub(monkeypatch: pytest.MonkeyPatch):
    # CohereModel imports cohere lazily inside _get_client
    for key in list(sys.modules):
        if key == "insideLLMs.models.cohere" or key.startswith("insideLLMs.models.cohere."):
            del sys.modules[key]

    import insideLLMs.models.cohere as cohere_model

    try:
        yield cohere_model
    finally:
        for key in list(sys.modules):
            if key == "insideLLMs.models.cohere" or key.startswith("insideLLMs.models.cohere."):
                del sys.modules[key]


def _install_cohere(monkeypatch: pytest.MonkeyPatch, client: MagicMock | None = None):
    mod = types.ModuleType("cohere")
    if client is None:
        client = MagicMock()
        client.chat.return_value = types.SimpleNamespace(text="hi")
        client.chat_stream.return_value = [
            types.SimpleNamespace(event_type="stream-start", text=""),
            types.SimpleNamespace(event_type="text-generation", text="a"),
            types.SimpleNamespace(event_type="text-generation", text="b"),
        ]
        client.embed.return_value = types.SimpleNamespace(embeddings=[[0.1, 0.2]])
        client.rerank.return_value = types.SimpleNamespace(
            results=[types.SimpleNamespace(index=1, relevance_score=0.9)]
        )
    mod.Client = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "cohere", mod)
    return client, mod


def test_cohere_full_paths(cohere_stub, monkeypatch: pytest.MonkeyPatch) -> None:
    cohere_model = cohere_stub
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        cohere_model.CohereModel(api_key=None)

    client, _ = _install_cohere(monkeypatch)
    m = cohere_model.CohereModel(
        api_key="ck",
        model_name="command-r",
        default_preamble="be brief",
    )
    assert m.generate("x", temperature=0.1, max_tokens=10, top_p=0.9, top_k=5) == "hi"
    assert m._client is client
    # second call hits cached client; p/k aliases; preamble override
    assert m.generate("y", preamble="override", p=0.5, k=3) == "hi"

    m_plain = cohere_model.CohereModel(api_key="ck")
    # no default preamble; only top_p/top_k via alternate names already covered — bare generate
    assert m_plain.generate("bare") == "hi"

    assert (
        m.chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
            ],
            temperature=0.2,
            max_tokens=8,
        )
        == "hi"
    )
    # recover last user after completed turn cleared current_message (708-711)
    assert (
        m.chat([{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]) == "hi"
    )
    assert m.chat([{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]) == "hi"
    # chat without preamble
    assert m_plain.chat([{"role": "user", "content": "q"}]) == "hi"
    # no messages -> empty current_message still calls chat
    assert m.chat([]) == "hi"
    # recovery loop finds no user role
    assert (
        m.chat([{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]) == "hi"
    )

    assert "".join(m.stream("x", temperature=0.3, max_tokens=4)) == "ab"
    # stream without default preamble
    assert "".join(m_plain.stream("z")) == "ab"

    assert m.info().provider == "Cohere"
    assert m.embed(["a"], embedding_types=["float"]) == [[0.1, 0.2]]
    assert m.embed(["b"]) == [[0.1, 0.2]]
    ranked = m.rerank("q", ["d0", "d1"], top_n=1)
    assert ranked == [{"index": 1, "relevance_score": 0.9, "document": "d1"}]
    client.rerank.return_value = types.SimpleNamespace(
        results=[types.SimpleNamespace(index=0, relevance_score=0.5)]
    )
    assert m.rerank("q", ["d0"]) == [{"index": 0, "relevance_score": 0.5, "document": "d0"}]

    # ImportError path
    m2 = cohere_model.CohereModel(api_key="ck")
    sys.modules.pop("cohere", None)
    real_import = builtins.__import__

    def block(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cohere":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(ImportError, match="cohere package"):
        m2._get_client()


@pytest.fixture()
def gemini_mod(monkeypatch: pytest.MonkeyPatch):
    for key in list(sys.modules):
        if key == "insideLLMs.models.gemini" or key.startswith("insideLLMs.models.gemini."):
            del sys.modules[key]
    import insideLLMs.models.gemini as gemini

    try:
        yield gemini
    finally:
        for key in list(sys.modules):
            if key == "insideLLMs.models.gemini" or key.startswith("insideLLMs.models.gemini."):
                del sys.modules[key]


def _install_genai(monkeypatch: pytest.MonkeyPatch):
    google = types.ModuleType("google")
    genai = types.ModuleType("google.generativeai")

    class GenerativeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.generate_content = MagicMock(
                return_value=types.SimpleNamespace(text="hi"),
            )
            chat = MagicMock()
            chat.send_message.return_value = types.SimpleNamespace(text="chat-hi")
            self.start_chat = MagicMock(return_value=chat)
            self.count_tokens = MagicMock(
                return_value=types.SimpleNamespace(total_tokens=7),
            )

        def stream_chunks(self, *args, **kwargs):
            return [
                types.SimpleNamespace(text="a"),
                types.SimpleNamespace(text=""),
                types.SimpleNamespace(text="b"),
            ]

    def configure(**kwargs):
        genai._configured = kwargs

    def list_models():
        return [
            types.SimpleNamespace(
                name="models/gemini-1.5-flash",
                supported_generation_methods=["generateContent"],
            ),
            types.SimpleNamespace(
                name="models/embed",
                supported_generation_methods=["embedContent"],
            ),
        ]

    genai.configure = configure
    genai.GenerativeModel = GenerativeModel
    genai.list_models = list_models
    google.generativeai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai)
    return genai, GenerativeModel


def test_gemini_full_paths(gemini_mod, monkeypatch: pytest.MonkeyPatch) -> None:
    gemini = gemini_mod
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        gemini.GeminiModel(api_key=None)

    genai, GenerativeModel = _install_genai(monkeypatch)
    m = gemini.GeminiModel(
        api_key="gk",
        model_name="gemini-1.5-flash",
        safety_settings=[{"category": "x", "threshold": "y"}],
        generation_config={"temperature": 0.1},
    )
    assert m.generate("x", temperature=0.5, max_tokens=10, top_p=0.9, top_k=4) == "hi"
    assert m.generate("y", max_output_tokens=5) == "hi"

    class _BadText:
        @property
        def text(self):
            raise ValueError("blocked")

    m._model.generate_content.return_value = _BadText()
    assert m.generate("z") == ""

    class _BadIndex:
        @property
        def text(self):
            raise IndexError("empty")

    m._model.generate_content.return_value = _BadIndex()
    assert m.generate("z2") == ""

    m._model.generate_content.return_value = types.SimpleNamespace(text="hi")
    assert (
        m.chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
            ],
            temperature=0.2,
            max_tokens=9,
        )
        == "chat-hi"
    )
    # trailing system-only
    assert m.chat([{"role": "system", "content": "only"}]) == "chat-hi"
    # second system while current_message already set is skipped
    assert (
        m.chat(
            [
                {"role": "system", "content": "first"},
                {"role": "system", "content": "second"},
                {"role": "user", "content": "u"},
            ]
        )
        == "chat-hi"
    )
    # system after history is skipped
    assert (
        m.chat(
            [
                {"role": "user", "content": "u"},
                {"role": "system", "content": "late"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "u2"},
            ]
        )
        == "chat-hi"
    )
    # empty history edge (no user) — still sends
    assert m.chat([{"role": "assistant", "content": "only-asst"}]) == "chat-hi"
    # unknown role falls through the role chain
    assert (
        m.chat([{"role": "tool", "content": "ignored"}, {"role": "user", "content": "u"}])
        == "chat-hi"
    )

    def stream_gc(*args, **kwargs):
        assert kwargs.get("stream") is True
        return [
            types.SimpleNamespace(text="a"),
            types.SimpleNamespace(text=""),
            types.SimpleNamespace(text="b"),
        ]

    m._model.generate_content = stream_gc
    assert "".join(m.stream("x", temperature=0.3, max_tokens=3)) == "ab"
    assert "".join(m.stream("plain")) == "ab"  # no temp/max_tokens kwargs
    assert m.info().provider == "Google"
    assert m.count_tokens("abc") == 7
    assert m.list_models() == ["models/gemini-1.5-flash"]
    # cached client path
    assert m._get_client() is m._model

    # ImportError on missing SDK
    m3 = gemini.GeminiModel(api_key="gk")
    m3._client = None
    m3._model = None
    for key in ("google.generativeai", "google"):
        sys.modules.pop(key, None)
    real_import = builtins.__import__

    def block(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("google", "google.generativeai") or name.startswith("google."):
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(ImportError, match="google-generativeai"):
        m3._get_client()


@pytest.fixture()
def hf_stub(monkeypatch: pytest.MonkeyPatch):
    transformers = types.ModuleType("transformers")
    state = {"tok": "ok", "mdl": "ok", "pipe": "ok"}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            if state["tok"] != "ok":
                raise RuntimeError(state["tok"])
            return MagicMock(name=f"tok:{name}")

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name):
            if state["mdl"] != "ok":
                raise RuntimeError(state["mdl"])
            return MagicMock(name=f"model:{name}")

    def pipeline(task, model=None, tokenizer=None, device=None):
        if state["pipe"] != "ok":
            raise RuntimeError(state["pipe"])
        gen = MagicMock(return_value=[{"generated_text": "out"}])
        gen.task = task
        gen.device = device
        return gen

    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    transformers.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    for key in list(sys.modules):
        if key == "insideLLMs.models.huggingface" or key.startswith(
            "insideLLMs.models.huggingface."
        ):
            del sys.modules[key]

    import insideLLMs.models.huggingface as hf

    try:
        yield hf, state
    finally:
        for key in list(sys.modules):
            if key == "insideLLMs.models.huggingface" or key.startswith(
                "insideLLMs.models.huggingface."
            ):
                del sys.modules[key]
        # Drop stub transformers if we installed it (monkeypatch restores after).
        # Ensure model module is not left bound to stub classes.


def test_huggingface_full_paths(hf_stub) -> None:
    hf, state = hf_stub

    m = hf.HuggingFaceModel(model_name="gpt2", device=-1)
    assert m.generate("hi") == "out"
    assert m.chat([{"role": "user", "content": "hi"}]) == "out"
    assert list(m.stream("hi")) == ["out"]
    info = m.info()
    assert info.extra["model_name"] == "gpt2"
    assert info.extra["device"] == -1

    m.generator.return_value = []
    assert m.generate("x") == ""
    assert m.chat([{"role": "user", "content": "x"}]) == ""
    assert list(m.stream("x")) == []

    m.generator.side_effect = RuntimeError("boom")
    with pytest.raises(ModelGenerationError):
        m.generate("x")
    with pytest.raises(ModelGenerationError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(ModelGenerationError):
        list(m.stream("x"))
    with pytest.raises(ModelGenerationError):
        m.chat([])

    state["tok"] = "tok-fail"
    with pytest.raises(ModelInitializationError, match="tokenizer"):
        hf.HuggingFaceModel(model_name="bad")
    state["tok"] = "ok"

    state["mdl"] = "mdl-fail"
    with pytest.raises(ModelInitializationError, match="model"):
        hf.HuggingFaceModel(model_name="bad")
    state["mdl"] = "ok"

    state["pipe"] = "pipe-fail"
    with pytest.raises(ModelInitializationError, match="pipeline"):
        hf.HuggingFaceModel(model_name="bad")
    state["pipe"] = "ok"


@pytest.fixture()
def openai_stub(monkeypatch: pytest.MonkeyPatch):
    """Install a fake openai package and force-reload insideLLMs.models.openai."""
    openai_mod = types.ModuleType("openai")

    class APIError(Exception):
        def __init__(self, message="api", status_code=500):
            super().__init__(message)
            self.status_code = status_code

    class APITimeoutError(Exception):
        pass

    class RateLimitErrorSDK(Exception):
        def __init__(self, message="rl"):
            super().__init__(message)
            self.retry_after = 1.5

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)
            self.delta = types.SimpleNamespace(content=content)

    class _Resp:
        def __init__(self, content="hi", empty=False):
            self.choices = [] if empty else [_Choice(content)]

    class OpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=MagicMock(return_value=_Resp("hi")))
            )

    openai_mod.APIError = APIError
    openai_mod.APITimeoutError = APITimeoutError
    openai_mod.RateLimitError = RateLimitErrorSDK
    openai_mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_mod)

    for key in list(sys.modules):
        if key == "insideLLMs.models.openai" or key.startswith("insideLLMs.models.openai."):
            del sys.modules[key]

    import insideLLMs.models.openai as openai_model

    try:
        yield openai_model, OpenAI, APIError, APITimeoutError, RateLimitErrorSDK, _Resp
    finally:
        # Drop poisoned module so later suites re-import against real/absent SDK.
        for key in list(sys.modules):
            if key == "insideLLMs.models.openai" or key.startswith("insideLLMs.models.openai."):
                del sys.modules[key]


def test_openai_init_generate_chat_stream_info(
    openai_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    openai_model, OpenAI, APIError, APITimeoutError, RateLimitErrorSDK, _Resp = openai_stub
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ModelInitializationError):
        openai_model.OpenAIModel(api_key=None)

    # client init failure
    def boom(**kwargs):
        raise RuntimeError("no client")

    monkeypatch.setattr(openai_model, "OpenAI", boom)
    with pytest.raises(ModelInitializationError, match="Failed to initialize"):
        openai_model.OpenAIModel(api_key="sk-x")
    monkeypatch.setattr(openai_model, "OpenAI", OpenAI)

    m = openai_model.OpenAIModel(
        model_name="gpt-4",
        api_key="sk-test",
        base_url="https://example.com",
        organization="org",
        project="proj",
        default_headers={"Authorization": "secret", "X-Custom": "1"},
    )
    assert m.generate("hello") == "hi"
    assert m.chat([{"role": "user", "content": "x"}]) == "hi"

    # empty choices
    m._client.chat.completions.create.return_value = _Resp(empty=True)
    assert m.generate("x") == ""
    assert m.chat([{"role": "user", "content": "x"}]) == ""

    # stream
    def stream_create(**kwargs):
        assert kwargs.get("stream") is True
        return iter(
            [
                types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="a"))]
                ),
                types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="b"))]
                ),
                types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=None))]
                ),
            ]
        )

    m._client.chat.completions.create = stream_create
    assert "".join(m.stream("x")) == "ab"

    info = m.info()
    assert info.extra["model_name"] == "gpt-4"
    assert info.extra["default_headers"]["Authorization"] == "***"
    assert info.extra["default_headers"]["X-Custom"] == "1"
    assert openai_model.OpenAIModel._redact_headers(None) is None


def test_openai_error_mapping(openai_stub) -> None:
    openai_model, OpenAI, APIError, APITimeoutError, RateLimitErrorSDK, _Resp = openai_stub
    m = openai_model.OpenAIModel(model_name="gpt-4", api_key="sk")

    m._client.chat.completions.create = MagicMock(side_effect=RateLimitErrorSDK())
    with pytest.raises(RateLimitError):
        m.generate("x")
    with pytest.raises(RateLimitError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(RateLimitError):
        list(m.stream("x"))

    m._client.chat.completions.create = MagicMock(side_effect=APITimeoutError())
    with pytest.raises(InsideLLMsTimeoutError):
        m.generate("x")
    with pytest.raises(InsideLLMsTimeoutError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(InsideLLMsTimeoutError):
        list(m.stream("x"))

    m._client.chat.completions.create = MagicMock(side_effect=APIError("bad", status_code=400))
    with pytest.raises(InsideLLMsAPIError):
        m.generate("x")
    with pytest.raises(InsideLLMsAPIError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(InsideLLMsAPIError):
        list(m.stream("x"))

    m._client.chat.completions.create = MagicMock(side_effect=ValueError("weird"))
    with pytest.raises(ModelGenerationError):
        m.generate("x")
    with pytest.raises(ModelGenerationError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(ModelGenerationError):
        list(m.stream("x"))
