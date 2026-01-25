"""Example: LangChain / LangGraph integration.

This example shows how to wrap an insideLLMs model for use in LangChain/LangGraph.

Install:
    pip install -e ".[langchain]"
"""

from __future__ import annotations


def run_langchain_chat_example() -> None:
    from langchain_core.messages import HumanMessage

    from insideLLMs.integrations.langchain import as_langchain_chat_model
    from insideLLMs.models import DummyModel

    lc_llm = as_langchain_chat_model(DummyModel())
    out = lc_llm.invoke([HumanMessage(content="What is 2 + 2?")])
    print(out.content)


def run_langchain_runnable_example() -> None:
    from insideLLMs.integrations.langchain import as_langchain_runnable
    from insideLLMs.models import DummyModel

    runnable = as_langchain_runnable(DummyModel())
    print(runnable.invoke("Say hello"))


if __name__ == "__main__":
    run_langchain_chat_example()
    run_langchain_runnable_example()
