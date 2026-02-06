import sys
import types
from types import SimpleNamespace

import insideLLMs.integrations.langchain as langchain_integration


class _FakeAIMessage:
    def __init__(self, content):
        self.content = content


class _FakeAIMessageChunk:
    def __init__(self, content):
        self.content = content


class _FakeChatGeneration:
    def __init__(self, message):
        self.message = message


class _FakeChatGenerationChunk:
    def __init__(self, message):
        self.message = message


class _FakeChatResult:
    def __init__(self, generations, llm_output=None):
        self.generations = generations
        self.llm_output = llm_output or {}


class _FakeRunnableLambda:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, inp):
        return self._fn(inp)


class _FakeBaseChatModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_fake_langchain(monkeypatch):
    pydantic_module = types.ModuleType("pydantic")

    def _field(**_kwargs):
        return None

    class _ConfigDict(dict):
        pass

    pydantic_module.Field = _field
    pydantic_module.ConfigDict = _ConfigDict

    langchain_core = types.ModuleType("langchain_core")
    language_models = types.ModuleType("langchain_core.language_models")
    chat_models = types.ModuleType("langchain_core.language_models.chat_models")
    messages = types.ModuleType("langchain_core.messages")
    outputs = types.ModuleType("langchain_core.outputs")
    runnables = types.ModuleType("langchain_core.runnables")

    chat_models.BaseChatModel = _FakeBaseChatModel
    messages.AIMessage = _FakeAIMessage
    messages.AIMessageChunk = _FakeAIMessageChunk
    outputs.ChatGeneration = _FakeChatGeneration
    outputs.ChatGenerationChunk = _FakeChatGenerationChunk
    outputs.ChatResult = _FakeChatResult
    runnables.RunnableLambda = _FakeRunnableLambda

    language_models.chat_models = chat_models
    langchain_core.language_models = language_models
    langchain_core.messages = messages
    langchain_core.outputs = outputs
    langchain_core.runnables = runnables

    monkeypatch.setitem(sys.modules, "pydantic", pydantic_module)
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.language_models", language_models)
    monkeypatch.setitem(sys.modules, "langchain_core.language_models.chat_models", chat_models)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages)
    monkeypatch.setitem(sys.modules, "langchain_core.outputs", outputs)
    monkeypatch.setitem(sys.modules, "langchain_core.runnables", runnables)


def test_message_content_to_text_handles_supported_types():
    class _Unserializable:
        def __str__(self):
            return "UNSERIALIZABLE"

    assert langchain_integration._message_content_to_text(None) == ""
    assert langchain_integration._message_content_to_text("hello") == "hello"
    assert langchain_integration._message_content_to_text(42) == "42"
    assert langchain_integration._message_content_to_text(["a", None, 5]) == "a\n5"
    assert langchain_integration._message_content_to_text({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'
    assert langchain_integration._message_content_to_text(_Unserializable()) == "UNSERIALIZABLE"


def test_lc_messages_to_insidellms_maps_roles_and_names():
    messages = [
        SimpleNamespace(type="system", content="s", name="sys"),
        SimpleNamespace(type="human", content="u", name=None),
        SimpleNamespace(type="ai", content="a", name="assistant-name"),
        SimpleNamespace(type="tool", content="tool-output", name="tool"),
        SimpleNamespace(type="unknown", content=["x", None, 9], name="other"),
    ]

    converted = langchain_integration._lc_messages_to_insidellms(messages)

    assert [msg["role"] for msg in converted] == ["system", "user", "assistant", "assistant", "user"]
    assert converted[0]["name"] == "sys"
    assert converted[4]["content"] == "x\n9"


def test_insidellms_messages_to_prompt_skips_empty_content():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "question"},
    ]

    prompt = langchain_integration._insidellms_messages_to_prompt(messages)

    assert prompt == "SYSTEM: rules\nUSER: question\nASSISTANT:"


def test_call_with_stop_tries_stop_then_stop_sequences_then_plain_call():
    stop_sequence_calls = []

    def _stop_sequences_only(value, **kwargs):
        stop_sequence_calls.append(kwargs)
        if "stop" in kwargs:
            raise TypeError("unsupported")
        return kwargs.get("stop_sequences")

    result = langchain_integration._call_with_stop(
        _stop_sequences_only,
        "prompt",
        stop=["END"],
        temperature=0.1,
    )
    assert result == ["END"]
    assert stop_sequence_calls[0] == {"temperature": 0.1, "stop": ["END"]}
    assert stop_sequence_calls[1] == {"temperature": 0.1, "stop_sequences": ["END"]}

    plain_calls = []

    def _no_stop_support(value, **kwargs):
        plain_calls.append(kwargs)
        if "stop" in kwargs or "stop_sequences" in kwargs:
            raise TypeError("unsupported")
        return value

    plain_result = langchain_integration._call_with_stop(
        _no_stop_support,
        "fallback",
        stop=["END"],
        max_tokens=5,
    )
    assert plain_result == "fallback"
    assert plain_calls[-1] == {"max_tokens": 5}


def test_as_langchain_chat_model_generate_uses_chat_path(monkeypatch):
    _install_fake_langchain(monkeypatch)

    class _Model:
        model_id = "model-123"

        def __init__(self):
            self.chat_calls = []

        def chat(self, messages, **kwargs):
            self.chat_calls.append((messages, kwargs))
            return "chat-response"

    model = _Model()
    wrapped = langchain_integration.as_langchain_chat_model(model)

    result = wrapped._generate(
        [SimpleNamespace(type="human", content="hello", name="u1")],
        stop=["DONE"],
        temperature=0.3,
    )

    assert wrapped._llm_type == "insidellms-chat"
    assert model.chat_calls[0][0] == [{"role": "user", "content": "hello", "name": "u1"}]
    assert model.chat_calls[0][1] == {"temperature": 0.3, "stop": ["DONE"]}
    assert result.generations[0].message.content == "chat-response"
    assert result.llm_output["model_id"] == "model-123"


def test_as_langchain_chat_model_generate_falls_back_to_prompt_generate(monkeypatch):
    _install_fake_langchain(monkeypatch)

    class _Model:
        model_id = "fallback-model"

        def __init__(self):
            self.prompt_seen = None
            self.kwargs_seen = None

        def chat(self, _messages, **_kwargs):
            raise NotImplementedError

        def generate(self, prompt, **kwargs):
            self.prompt_seen = prompt
            self.kwargs_seen = kwargs
            return "generated-response"

    model = _Model()
    wrapped = langchain_integration.as_langchain_chat_model(model)
    result = wrapped._generate(
        [SimpleNamespace(type="system", content="policy"), SimpleNamespace(type="human", content="question")],
        stop=["STOP"],
    )

    assert "SYSTEM: policy" in model.prompt_seen
    assert "USER: question" in model.prompt_seen
    assert model.prompt_seen.endswith("ASSISTANT:")
    assert model.kwargs_seen == {"stop": ["STOP"]}
    assert result.generations[0].message.content == "generated-response"


def test_as_langchain_chat_model_stream_emits_chunks_and_notifies_run_manager(monkeypatch):
    _install_fake_langchain(monkeypatch)

    class _RunManager:
        def __init__(self):
            self.tokens = []

        def on_llm_new_token(self, token):
            self.tokens.append(token)

    class _Model:
        def stream(self, _prompt, **_kwargs):
            return ["a", "b"]

        def chat(self, _messages, **_kwargs):
            return "unused"

    model = _Model()
    wrapped = langchain_integration.as_langchain_chat_model(model)
    run_manager = _RunManager()

    chunks = list(
        wrapped._stream([SimpleNamespace(type="human", content="hello")], run_manager=run_manager)
    )

    assert [chunk.message.content for chunk in chunks] == ["a", "b"]
    assert run_manager.tokens == ["a", "b"]


def test_as_langchain_chat_model_stream_falls_back_to_generate_on_stream_error(monkeypatch):
    _install_fake_langchain(monkeypatch)

    class _Model:
        def stream(self, _prompt, **_kwargs):
            raise RuntimeError("stream failure")

        def chat(self, _messages, **_kwargs):
            return "fallback-chat"

    model = _Model()
    wrapped = langchain_integration.as_langchain_chat_model(model)

    chunks = list(wrapped._stream([SimpleNamespace(type="human", content="hello")]))

    assert len(chunks) == 1
    assert chunks[0].message.content == "fallback-chat"


def test_as_langchain_runnable_supports_str_messages_and_other_inputs(monkeypatch):
    _install_fake_langchain(monkeypatch)

    class _Model:
        def __init__(self):
            self.generated = []
            self.chatted = []

        def generate(self, prompt):
            self.generated.append(prompt)
            return f"gen:{prompt}"

        def chat(self, messages):
            self.chatted.append(messages)
            if messages and messages[0].get("content") == "use-fallback":
                raise NotImplementedError
            return "chat:ok"

    model = _Model()
    runnable = langchain_integration.as_langchain_runnable(model)

    assert runnable.invoke("hello") == "gen:hello"

    message_input = [SimpleNamespace(type="human", content="hi")]
    assert runnable.invoke(message_input) == "chat:ok"
    assert model.chatted[-1] == [{"role": "user", "content": "hi", "name": None}]

    fallback_input = [SimpleNamespace(type="human", content="use-fallback")]
    fallback_result = runnable.invoke(fallback_input)
    assert fallback_result.startswith("gen:USER: use-fallback")
    assert fallback_result.endswith("ASSISTANT:")

    assert runnable.invoke(123) == "gen:123"
    assert model.generated[-1] == "123"
