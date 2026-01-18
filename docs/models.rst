Models
======

insideLLMs provides a unified interface for various LLM providers.


Model Base Class
----------------

All models inherit from the ``Model`` base class:

.. code-block:: python

    from insideLLMs import Model

    class CustomModel(Model):
        def generate(self, prompt: str, **kwargs) -> str:
            ...

        def chat(self, messages: list, **kwargs) -> str:
            ...

        def stream(self, prompt: str, **kwargs):
            ...


Available Models
----------------

DummyModel
^^^^^^^^^^

A testing model that doesn't require API keys:

.. code-block:: python

    from insideLLMs import DummyModel

    model = DummyModel()
    response = model.generate("Hello!")
    # Output: "[DummyModel] You said: Hello!"

    # Custom responses
    model = DummyModel(canned_response="Always this!")


OpenAIModel
^^^^^^^^^^^

For OpenAI's GPT models:

.. code-block:: python

    import os
    os.environ["OPENAI_API_KEY"] = "your-key"

    from insideLLMs.models import OpenAIModel

    model = OpenAIModel(model_name="gpt-4")
    response = model.generate("Explain quantum computing")


AnthropicModel
^^^^^^^^^^^^^^

For Anthropic's Claude models:

.. code-block:: python

    import os
    os.environ["ANTHROPIC_API_KEY"] = "your-key"

    from insideLLMs.models import AnthropicModel

    model = AnthropicModel(model_name="claude-3-opus-20240229")
    response = model.generate("Write a haiku about AI")


HuggingFaceModel
^^^^^^^^^^^^^^^^

For local HuggingFace Transformers models:

.. code-block:: python

    from insideLLMs.models import HuggingFaceModel

    model = HuggingFaceModel(model_name="gpt2")
    response = model.generate("Once upon a time", max_length=50)


Using the Registry
------------------

Get models by name from the registry:

.. code-block:: python

    from insideLLMs import model_registry

    # List available
    print(model_registry.list())

    # Get by name
    model = model_registry.get("openai", model_name="gpt-4")


Streaming
---------

All models support streaming:

.. code-block:: python

    model = DummyModel()
    for chunk in model.stream("Tell me a story"):
        print(chunk, end="", flush=True)


Chat Interface
--------------

Models support multi-turn conversations:

.. code-block:: python

    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What's 2+2?"},
    ]

    response = model.chat(messages)


Error Handling
--------------

Models raise specific exceptions for different failure modes:

.. code-block:: python

    from insideLLMs.exceptions import (
        ModelInitializationError,
        ModelGenerationError,
        RateLimitError,
        TimeoutError,
        APIError,
    )

    try:
        response = model.generate("...")
    except RateLimitError as e:
        print(f"Rate limited, retry after {e.retry_after}s")
    except TimeoutError as e:
        print(f"Request timed out after {e.details['timeout_seconds']}s")
    except ModelGenerationError as e:
        print(f"Generation failed: {e.message}")
