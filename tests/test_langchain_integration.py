import importlib.util

import pytest

from insideLLMs.integrations.langchain import (
    LangChainIntegrationError,
    as_langchain_chat_model,
    as_langchain_runnable,
)
from insideLLMs.models import DummyModel


def test_as_langchain_chat_model_raises_without_langchain():
    if importlib.util.find_spec("langchain_core") is not None:
        pytest.skip("langchain_core installed; this test validates the no-deps path")

    with pytest.raises(LangChainIntegrationError):
        as_langchain_chat_model(DummyModel())


def test_as_langchain_runnable_raises_without_langchain():
    if importlib.util.find_spec("langchain_core") is not None:
        pytest.skip("langchain_core installed; this test validates the no-deps path")

    with pytest.raises(LangChainIntegrationError):
        as_langchain_runnable(DummyModel())
