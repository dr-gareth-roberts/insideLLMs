"""Plugin Development Example for insideLLMs.

This example shows how to create a custom plugin that registers
a new model and probe with the insideLLMs registry system.

Plugins are discovered via Python entry points (the ``insidellms.plugins``
group). Any installed package that declares the entry point will have its
``register`` function called at startup.

Sections:
1. Creating a custom model
2. Creating a custom probe
3. Registering via the plugin system
4. Using the registered components

No API keys are required -- this example uses only local code.
"""

from typing import Any

from insideLLMs.models.base import Model
from insideLLMs.probes.base import Probe
from insideLLMs.registry import (
    ensure_builtins_registered,
    model_registry,
    probe_registry,
)
from insideLLMs.types import ProbeCategory

# ---------------------------------------------------------------------------
# 1. Custom model
# ---------------------------------------------------------------------------


class ReverseModel(Model):
    """A trivial model that reverses the input prompt.

    Useful as a starting point for understanding the Model interface.
    """

    def __init__(self) -> None:
        super().__init__(name="reverse", model_id="reverse-v1")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return prompt[::-1]

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "provider": "custom", "model_id": "reverse-v1"}


# ---------------------------------------------------------------------------
# 2. Custom probe
# ---------------------------------------------------------------------------


class PalindromeProbe(Probe):
    """A probe that checks whether a model's response is a palindrome.

    This is intentionally simple to illustrate the Probe interface.
    """

    def __init__(self) -> None:
        super().__init__(
            name="palindrome",
            category=ProbeCategory.CUSTOM,
            description="Checks if the model response reads the same forwards and backwards.",
        )

    def run(self, model: Model, data: Any, **kwargs: Any) -> dict[str, Any]:
        response = model.generate(str(data), **kwargs)
        normalized = response.lower().replace(" ", "")
        is_palindrome = normalized == normalized[::-1]
        return {
            "input": str(data),
            "output": response,
            "is_palindrome": is_palindrome,
            "status": "success",
        }


# ---------------------------------------------------------------------------
# 3. Plugin registration function
# ---------------------------------------------------------------------------


def register() -> None:
    """Entry point called by insideLLMs plugin loader.

    To wire this up as a real plugin, add to your pyproject.toml::

        [project.entry-points."insidellms.plugins"]
        my_plugin = "my_package.plugin:register"

    The registries can also be received as arguments::

        def register(model_registry, probe_registry, dataset_registry):
            ...
    """
    model_registry.register("reverse", ReverseModel, overwrite=True)
    probe_registry.register("palindrome", PalindromeProbe, overwrite=True)


# ---------------------------------------------------------------------------
# 4. Demo: use the plugin components directly
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_builtins_registered()
    register()

    print("=" * 60)
    print("PLUGIN EXAMPLE")
    print("=" * 60)

    # Use the custom model via the registry
    model = model_registry.get("reverse")
    print(f"\nModel: {model.name}")
    print(f"  generate('hello') -> {model.generate('hello')!r}")

    # Use the custom probe via the registry
    probe = probe_registry.get("palindrome")
    result = probe.run(model, "racecar")
    print(f"\nProbe: {probe.name}")
    print(f"  Input:        {result['input']}")
    print(f"  Output:       {result['output']}")
    print(f"  Palindrome?   {result['is_palindrome']}")

    # Verify they appear in the registry listings
    print(f"\nRegistered models:  {model_registry.list()}")
    print(f"Registered probes:  {probe_registry.list()}")


if __name__ == "__main__":
    main()
