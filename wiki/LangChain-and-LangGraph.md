---
title: LangChain and LangGraph
nav_order: 14
---

This integration lets you use insideLLMs models inside the LangChain / LangGraph ecosystem.

## Install

insideLLMs keeps LangChain/LangGraph as optional dependencies:

```bash
pip install -e ".[langchain]"
```

## Use an insideLLMs Model in LangChain

Wrap an insideLLMs model as a LangChain chat model:

```python
from langchain_core.messages import HumanMessage

from insideLLMs.models import DummyModel
from insideLLMs.integrations.langchain import as_langchain_chat_model

lc_llm = as_langchain_chat_model(DummyModel())
result = lc_llm.invoke([HumanMessage(content="What is 2 + 2?")])
print(result.content)
```

If the underlying insideLLMs model does not implement `chat(...)`, the adapter falls back to a
deterministic prompt rendering and calls `generate(...)`.

## Use in LangGraph

LangGraph nodes commonly accept LangChain Runnables (chat models are Runnables). The wrapped
chat model returned by `as_langchain_chat_model(...)` can be used anywhere a chat model is expected.

If you prefer a lightweight adapter, you can wrap as a Runnable:

```python
from insideLLMs.models import DummyModel
from insideLLMs.integrations.langchain import as_langchain_runnable

runnable = as_langchain_runnable(DummyModel())
print(runnable.invoke("Say hello"))
```

## Notes

- This integration is best-effort and intentionally minimal.
- insideLLMs does not currently expose a chat-streaming interface; streaming uses the prompt fallback.
- For deterministic CI workflows, prefer insideLLMs run directories + `insidellms diff`.

## See Also

- [Providers and Models](Providers-and-Models.md)
- [Experiment Tracking](Experiment-Tracking.md)
- [Determinism and CI](Determinism-and-CI.md)
