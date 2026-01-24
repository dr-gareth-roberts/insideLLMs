"""Plugin registry system for models, probes, and datasets.

This module provides a flexible registration system that allows
models, probes, and datasets to be registered and retrieved by name.
Supports both direct registration and decorator-based registration.
"""

from __future__ import annotations

import os
import warnings
from importlib import metadata
from inspect import Signature, signature
from typing import Any, Callable, Generic, Optional, TypeVar, Union

T = TypeVar("T")
FactoryType = Callable[..., T]


class RegistrationError(Exception):
    """Raised when a registration fails."""

    pass


class NotFoundError(Exception):
    """Raised when a registered item is not found."""

    pass


class Registry(Generic[T]):
    """A generic registry for storing and retrieving registered items.

    The registry supports storing either instances or factory functions,
    and provides both direct registration and decorator-based registration.

    Type Parameters:
        T: The type of items stored in this registry.

    Example:
        >>> model_registry = Registry[Model]("models")
        >>> model_registry.register("gpt4", OpenAIModel, model_name="gpt-4")
        >>> model = model_registry.get("gpt4")
    """

    def __init__(self, name: str):
        """Initialize the registry.

        Args:
            name: A descriptive name for this registry (e.g., "models", "probes").
        """
        self.name = name
        self._registry: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        factory: Union[type[T], FactoryType[T]],
        *,
        overwrite: bool = False,
        **default_kwargs: Any,
    ) -> None:
        """Register a factory or class with the given name.

        Args:
            name: The unique identifier for this registration.
            factory: A class or factory function that creates instances.
            overwrite: If True, allows overwriting existing registrations.
            **default_kwargs: Default arguments to pass when instantiating.

        Raises:
            RegistrationError: If name is already registered and overwrite is False.

        Example:
            >>> registry.register("openai", OpenAIModel, model_name="gpt-4")
        """
        if name in self._registry and not overwrite:
            raise RegistrationError(
                f"'{name}' is already registered in {self.name} registry. "
                f"Use overwrite=True to replace."
            )

        self._registry[name] = {
            "factory": factory,
            "default_kwargs": default_kwargs,
        }

    def register_decorator(
        self,
        name: Optional[str] = None,
        **default_kwargs: Any,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator for registering classes.

        Args:
            name: The registration name. If not provided, uses the class name.
            **default_kwargs: Default arguments for instantiation.

        Returns:
            A decorator that registers the class.

        Example:
            >>> @model_registry.register_decorator("custom")
            ... class CustomModel(Model):
            ...     pass
        """

        def decorator(cls: type[T]) -> type[T]:
            registration_name = name or cls.__name__
            self.register(registration_name, cls, **default_kwargs)
            return cls

        return decorator

    def get(self, registration_name: str, **override_kwargs: Any) -> T:
        """Get an instance of the registered item.

        Args:
            registration_name: The registration name.
            **override_kwargs: Arguments that override the defaults.

        Returns:
            An instance created by the registered factory.

        Raises:
            NotFoundError: If the name is not registered.

        Example:
            >>> model = registry.get("openai", temperature=0.7)
        """
        if registration_name not in self._registry:
            available = ", ".join(self.list()) or "(none)"
            raise NotFoundError(
                f"'{registration_name}' not found in {self.name} registry. Available: {available}"
            )

        entry = self._registry[registration_name]
        kwargs = {**entry["default_kwargs"], **override_kwargs}
        return entry["factory"](**kwargs)

    def get_factory(self, name: str) -> Union[type[T], FactoryType[T]]:
        """Get the factory function/class without instantiating.

        Args:
            name: The registration name.

        Returns:
            The registered factory or class.

        Raises:
            NotFoundError: If the name is not registered.
        """
        if name not in self._registry:
            raise NotFoundError(f"'{name}' not found in {self.name} registry.")
        return self._registry[name]["factory"]

    def list(self) -> list[str]:
        """List all registered names.

        Returns:
            A list of all registration names.
        """
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a name is registered.

        Args:
            name: The name to check.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._registry

    def unregister(self, name: str) -> None:
        """Remove a registration.

        Args:
            name: The name to unregister.

        Raises:
            NotFoundError: If the name is not registered.
        """
        if name not in self._registry:
            raise NotFoundError(f"'{name}' not found in {self.name} registry.")
        del self._registry[name]

    def clear(self) -> None:
        """Remove all registrations."""
        self._registry.clear()

    def info(self, name: str) -> dict[str, Any]:
        """Get information about a registration.

        Args:
            name: The registration name.

        Returns:
            A dictionary with factory and default kwargs information.

        Raises:
            NotFoundError: If the name is not registered.
        """
        if name not in self._registry:
            raise NotFoundError(f"'{name}' not found in {self.name} registry.")

        entry = self._registry[name]
        factory = entry["factory"]

        return {
            "name": name,
            "factory": factory.__name__ if hasattr(factory, "__name__") else str(factory),
            "default_kwargs": entry["default_kwargs"],
            "doc": factory.__doc__,
        }

    def __contains__(self, name: str) -> bool:
        return self.is_registered(name)

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, items={self.list()})"


# Global registries for the library
from insideLLMs.models.base import Model  # noqa: E402
from insideLLMs.probes.base import Probe  # noqa: E402

model_registry: Registry[Model] = Registry("models")
probe_registry: Registry[Probe] = Registry("probes")
dataset_registry: Registry[Any] = Registry("datasets")

PLUGIN_ENTRYPOINT_GROUP = "insidellms.plugins"


def _lazy_import_factory(module_path: str, class_name: str):
    """Create a factory that lazily imports a class when called.

    This enables registration without importing heavy dependencies upfront.
    """

    def factory(**kwargs):
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    factory.__name__ = class_name
    factory.__doc__ = f"Lazy factory for {class_name}"
    return factory


def register_builtins() -> None:
    """Register all built-in models, probes, and datasets.

    This function is called automatically when the library is imported,
    but can be called again to reset the registries to defaults.

    Uses lazy imports for heavy dependencies (like HuggingFace transformers)
    to keep import times fast.
    """
    # Clear existing registrations
    model_registry.clear()
    probe_registry.clear()
    dataset_registry.clear()

    # Register models - use lazy loading for optional/heavy dependencies.
    # DummyModel is lightweight, can import directly.
    from insideLLMs.models import DummyModel

    model_registry.register("dummy", DummyModel)

    # Hosted/API models (optional SDK dependencies).
    model_registry.register(
        "openai", _lazy_import_factory("insideLLMs.models.openai", "OpenAIModel")
    )
    model_registry.register(
        "anthropic", _lazy_import_factory("insideLLMs.models.anthropic", "AnthropicModel")
    )
    model_registry.register(
        "gemini", _lazy_import_factory("insideLLMs.models.gemini", "GeminiModel")
    )
    model_registry.register(
        "cohere", _lazy_import_factory("insideLLMs.models.cohere", "CohereModel")
    )

    # HuggingFace is heavy - use lazy loading.
    model_registry.register(
        "huggingface", _lazy_import_factory("insideLLMs.models.huggingface", "HuggingFaceModel")
    )

    # Local models (optional deps / local services).
    model_registry.register(
        "llamacpp", _lazy_import_factory("insideLLMs.models.local", "LlamaCppModel")
    )
    model_registry.register(
        "ollama", _lazy_import_factory("insideLLMs.models.local", "OllamaModel")
    )
    model_registry.register("vllm", _lazy_import_factory("insideLLMs.models.local", "VLLMModel"))

    # Register probes - these are generally lightweight
    from insideLLMs.probes import (
        AttackProbe,
        BiasProbe,
        CodeDebugProbe,
        CodeExplanationProbe,
        CodeGenerationProbe,
        ConstraintComplianceProbe,
        FactualityProbe,
        InstructionFollowingProbe,
        JailbreakProbe,
        LogicProbe,
        MultiStepTaskProbe,
        PromptInjectionProbe,
    )

    probe_registry.register("logic", LogicProbe)
    probe_registry.register("factuality", FactualityProbe)
    probe_registry.register("bias", BiasProbe)
    probe_registry.register("attack", AttackProbe)
    probe_registry.register("prompt_injection", PromptInjectionProbe)
    probe_registry.register("jailbreak", JailbreakProbe)
    probe_registry.register("code_generation", CodeGenerationProbe)
    probe_registry.register("code_explanation", CodeExplanationProbe)
    probe_registry.register("code_debug", CodeDebugProbe)
    probe_registry.register("instruction_following", InstructionFollowingProbe)
    probe_registry.register("multi_step_task", MultiStepTaskProbe)
    probe_registry.register("constraint_compliance", ConstraintComplianceProbe)

    # Register dataset loaders
    from insideLLMs.dataset_utils import (
        load_csv_dataset,
        load_hf_dataset,
        load_jsonl_dataset,
    )

    dataset_registry.register("csv", load_csv_dataset)
    dataset_registry.register("jsonl", load_jsonl_dataset)
    dataset_registry.register("hf", load_hf_dataset)


# Auto-register on import (deferred to avoid circular imports at module level)
_builtins_registered = False
_plugins_loaded = False


def _call_plugin_register(fn: Callable[..., Any]) -> None:
    sig: Signature
    try:
        sig = signature(fn)
    except Exception:
        fn()
        return

    if len(sig.parameters) == 0:
        fn()
        return

    fn(
        model_registry=model_registry,
        probe_registry=probe_registry,
        dataset_registry=dataset_registry,
    )


def load_entrypoint_plugins(
    *,
    group: str = PLUGIN_ENTRYPOINT_GROUP,
    enabled: Optional[bool] = None,
) -> dict[str, str]:
    """Load and execute plugin registration entry points.

    Plugins are discovered via Python entry points. Each entry point should
    resolve to a callable that registers models/probes/datasets using the
    provided registries (or by importing insideLLMs and registering globally).

    By default, plugins are enabled unless INSIDELLMS_DISABLE_PLUGINS=1.

    Returns:
        Mapping of entry point name -> entry point value (import path).
    """
    if enabled is None:
        enabled = os.environ.get("INSIDELLMS_DISABLE_PLUGINS", "").strip() not in {
            "1",
            "true",
            "yes",
        }
    if not enabled:
        return {}

    loaded: dict[str, str] = {}

    try:
        eps = metadata.entry_points()
        if hasattr(eps, "select"):
            selected = list(eps.select(group=group))
        else:  # pragma: no cover (older Python)
            selected = list(eps.get(group, []))  # type: ignore[attr-defined]
    except Exception:
        return {}

    for ep in selected:
        try:
            fn = ep.load()
            if not callable(fn):
                warnings.warn(
                    f"Plugin entry point {ep.name!r} did not resolve to a callable: {ep.value}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            _call_plugin_register(fn)
            loaded[ep.name] = ep.value
        except Exception as e:
            warnings.warn(
                f"Failed to load plugin {ep.name!r} ({ep.value}): {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    return loaded


def ensure_builtins_registered() -> None:
    """Ensure built-in registrations are loaded. Called lazily."""
    global _builtins_registered, _plugins_loaded
    if not _builtins_registered:
        try:
            register_builtins()
            _builtins_registered = True
        except ImportError:
            # Some dependencies might not be installed
            pass

    if _builtins_registered and not _plugins_loaded:
        load_entrypoint_plugins()
        _plugins_loaded = True
