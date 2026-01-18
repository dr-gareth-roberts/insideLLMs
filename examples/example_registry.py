"""Registry System Example for insideLLMs.

This example demonstrates how to use the plugin registry system:
1. Listing available registrations
2. Getting instances from the registry
3. Registering custom models and probes
4. Using decorator-based registration
"""

from insideLLMs import (
    Model,
    Probe,
    model_registry,
    probe_registry,
    ensure_builtins_registered,
)
from insideLLMs.models.base import ChatMessage
from insideLLMs.types import ProbeCategory
from typing import Iterator, List, Any


def list_available_items():
    """Show all registered models and probes."""
    print("=" * 60)
    print("1. LISTING AVAILABLE REGISTRATIONS")
    print("=" * 60)

    print("\nRegistered Models:")
    for name in model_registry.list():
        info = model_registry.info(name)
        print(f"  - {name}: {info['factory']}")

    print("\nRegistered Probes:")
    for name in probe_registry.list():
        info = probe_registry.info(name)
        print(f"  - {name}: {info['factory']}")


def get_from_registry():
    """Demonstrate getting instances from the registry."""
    print("\n" + "=" * 60)
    print("2. GETTING INSTANCES FROM REGISTRY")
    print("=" * 60)

    # Get a model by name
    model = model_registry.get("dummy")
    print(f"Got model: {model.name}")

    # Get a probe by name
    probe = probe_registry.get("logic")
    print(f"Got probe: {probe.name}")

    # Get with custom parameters
    custom_model = model_registry.get(
        "dummy",
        name="CustomDummy",
        canned_response="Always return this!",
    )
    print(f"Custom model response: {custom_model.generate('test')}")


def register_custom_model():
    """Demonstrate registering a custom model."""
    print("\n" + "=" * 60)
    print("3. REGISTERING CUSTOM MODELS")
    print("=" * 60)

    # Define a custom model
    class EchoModel(Model):
        """A simple model that echoes input with a prefix."""

        def __init__(self, name: str = "EchoModel", prefix: str = "[ECHO]"):
            super().__init__(name=name, model_id="echo-v1")
            self.prefix = prefix

        def generate(self, prompt: str, **kwargs: Any) -> str:
            return f"{self.prefix} {prompt}"

        def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
            last = messages[-1]["content"] if messages else ""
            return f"{self.prefix} {last}"

        def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
            yield self.generate(prompt)

    # Register the custom model
    model_registry.register("echo", EchoModel, prefix=">>>")
    print("Registered 'echo' model")

    # Use it from the registry
    echo_model = model_registry.get("echo")
    print(f"Echo response: {echo_model.generate('Hello World')}")

    # Check it's listed
    print(f"Available models: {model_registry.list()}")


def register_custom_probe():
    """Demonstrate registering a custom probe."""
    print("\n" + "=" * 60)
    print("4. REGISTERING CUSTOM PROBES")
    print("=" * 60)

    # Define a custom probe
    class WordCountProbe(Probe[int]):
        """A probe that counts words in responses."""

        def __init__(self, name: str = "WordCountProbe"):
            super().__init__(name=name, category=ProbeCategory.CUSTOM)

        def run(self, model: Model, data: str, **kwargs: Any) -> int:
            response = model.generate(data, **kwargs)
            return len(response.split())

    # Register the probe
    probe_registry.register("word_count", WordCountProbe)
    print("Registered 'word_count' probe")

    # Use it
    model = model_registry.get("dummy")
    probe = probe_registry.get("word_count")
    count = probe.run(model, "How many words?")
    print(f"Word count result: {count}")


def decorator_registration():
    """Demonstrate decorator-based registration."""
    print("\n" + "=" * 60)
    print("5. DECORATOR-BASED REGISTRATION")
    print("=" * 60)

    # Use decorator to register
    @model_registry.register_decorator("upper_echo")
    class UpperEchoModel(Model):
        """Echo model that uppercases responses."""

        def __init__(self, name: str = "UpperEchoModel"):
            super().__init__(name=name, model_id="upper-echo-v1")

        def generate(self, prompt: str, **kwargs: Any) -> str:
            return prompt.upper()

        def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
            return messages[-1]["content"].upper() if messages else ""

        def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
            yield self.generate(prompt)

    print("Registered 'upper_echo' via decorator")

    # Use it
    upper_model = model_registry.get("upper_echo")
    print(f"Upper echo: {upper_model.generate('hello world')}")


def registry_utilities():
    """Demonstrate registry utility methods."""
    print("\n" + "=" * 60)
    print("6. REGISTRY UTILITIES")
    print("=" * 60)

    # Check if registered
    print(f"Is 'dummy' registered? {model_registry.is_registered('dummy')}")
    print(f"Is 'nonexistent' registered? {model_registry.is_registered('nonexistent')}")

    # Get factory without instantiating
    DummyClass = model_registry.get_factory("dummy")
    print(f"Factory class: {DummyClass.__name__}")

    # Get detailed info
    info = model_registry.info("dummy")
    print(f"Info for 'dummy': {info}")

    # Registry length and contains
    print(f"Number of registered models: {len(model_registry)}")
    print(f"'logic' in probe_registry: {'logic' in probe_registry}")


def main():
    """Run all registry examples."""
    print("\n" + "#" * 60)
    print("# insideLLMs Registry System Guide")
    print("#" * 60)

    # Ensure builtins are registered
    ensure_builtins_registered()

    list_available_items()
    get_from_registry()
    register_custom_model()
    register_custom_probe()
    decorator_registration()
    registry_utilities()

    print("\n" + "=" * 60)
    print("Registry examples complete!")
    print("=" * 60)
    print("\nThe registry system allows you to:")
    print("  - Discover available models and probes")
    print("  - Instantiate by name with custom parameters")
    print("  - Register your own implementations")
    print("  - Build plugin systems on top of insideLLMs")


if __name__ == "__main__":
    main()
