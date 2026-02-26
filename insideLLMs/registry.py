"""Plugin registry system for models, probes, and datasets.

This module provides a flexible registration system that allows
models, probes, and datasets to be registered and retrieved by name.
Supports both direct registration and decorator-based registration.

The registry system enables:
    - Lazy loading of heavy dependencies (e.g., HuggingFace transformers)
    - Plugin architecture via Python entry points
    - Runtime discovery and instantiation of registered components

Module Components:
    - Registry: Generic class for storing and retrieving registered items
    - model_registry: Global registry for model implementations
    - probe_registry: Global registry for probe implementations
    - dataset_registry: Global registry for dataset loaders
    - RegistrationError: Exception for registration failures
    - NotFoundError: Exception when items are not found

Examples:
    Basic model registration and retrieval:

        >>> from insideLLMs.registry import model_registry
        >>> from insideLLMs.models import DummyModel
        >>> model_registry.register("my_model", DummyModel)
        >>> model = model_registry.get("my_model")
        >>> print(model)
        <DummyModel instance>

    Using the decorator pattern for registration:

        >>> from insideLLMs.registry import probe_registry
        >>> @probe_registry.register_decorator("custom_probe")
        ... class CustomProbe:
        ...     def __init__(self, threshold=0.5):
        ...         self.threshold = threshold
        ...
        >>> probe = probe_registry.get("custom_probe", threshold=0.8)

    Listing available registrations:

        >>> from insideLLMs.registry import model_registry, ensure_builtins_registered
        >>> ensure_builtins_registered()
        >>> available_models = model_registry.list()
        >>> print("openai" in available_models)
        True

    Creating a custom registry:

        >>> from insideLLMs.registry import Registry
        >>> custom_registry = Registry[str]("custom_items")
        >>> custom_registry.register("greeting", lambda: "Hello, World!")
        >>> custom_registry.get("greeting")
        'Hello, World!'

See Also:
    - insideLLMs.models: Model implementations
    - insideLLMs.probes: Probe implementations
    - insideLLMs.dataset_utils: Dataset loading utilities
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
    """Raised when a registration operation fails.

    This exception is raised when attempting to register an item with a name
    that is already in use and the `overwrite` flag is not set to True.

    Attributes:
        args: The exception message describing the registration failure.

    Examples:
        Attempting to register a duplicate name:

            >>> from insideLLMs.registry import Registry, RegistrationError
            >>> registry = Registry("test")
            >>> registry.register("item1", lambda: "first")
            >>> try:
            ...     registry.register("item1", lambda: "second")
            ... except RegistrationError as e:
            ...     print(f"Registration failed: {e}")
            Registration failed: 'item1' is already registered in test registry...

        Avoiding the error with overwrite=True:

            >>> registry.register("item1", lambda: "second", overwrite=True)
            >>> registry.get("item1")
            'second'

        Catching the error in plugin registration:

            >>> def register_plugin(registry):
            ...     try:
            ...         registry.register("conflicting_name", MyPlugin)
            ...     except RegistrationError:
            ...         # Handle gracefully, perhaps use a different name
            ...         registry.register("conflicting_name_v2", MyPlugin)

    See Also:
        Registry.register: The method that raises this exception.
    """

    pass


class NotFoundError(Exception):
    """Raised when a registered item is not found in the registry.

    This exception is raised when attempting to retrieve, get information about,
    or unregister an item using a name that has not been registered.

    Attributes:
        args: The exception message describing what was not found and
              listing available items.

    Examples:
        Attempting to get a non-existent item:

            >>> from insideLLMs.registry import Registry, NotFoundError
            >>> registry = Registry("models")
            >>> registry.register("gpt4", lambda: "GPT-4 model")
            >>> try:
            ...     registry.get("nonexistent")
            ... except NotFoundError as e:
            ...     print(f"Error: {e}")
            Error: 'nonexistent' not found in models registry. Available: gpt4

        Handling missing items gracefully:

            >>> def get_model_or_default(registry, name, default_name="dummy"):
            ...     try:
            ...         return registry.get(name)
            ...     except NotFoundError:
            ...         return registry.get(default_name)

        Checking before retrieval to avoid the exception:

            >>> if registry.is_registered("my_model"):
            ...     model = registry.get("my_model")
            ... else:
            ...     print("Model not registered, using fallback")

        Using the `in` operator for checking:

            >>> if "my_model" in registry:
            ...     model = registry.get("my_model")

    See Also:
        Registry.get: The retrieval method that raises this exception.
        Registry.get_factory: Another method that raises this exception.
        Registry.unregister: Unregistration method that raises this exception.
        Registry.is_registered: Method to check existence without exceptions.
    """

    pass


class Registry(Generic[T]):
    """A generic registry for storing and retrieving registered items.

    The registry supports storing either instances or factory functions,
    and provides both direct registration and decorator-based registration.
    It is the core component of the insideLLMs plugin system, enabling
    dynamic discovery and instantiation of models, probes, and datasets.

    Type Parameters:
        T: The type of items stored in this registry. For example,
           Registry[Model] stores model factories/classes.

    Attributes:
        name (str): A descriptive name for this registry (e.g., "models").
        _registry (dict): Internal storage mapping names to factory info.

    Examples:
        Creating and using a basic registry:

            >>> from insideLLMs.registry import Registry
            >>> registry = Registry[str]("greetings")
            >>> registry.register("hello", lambda: "Hello, World!")
            >>> registry.register("goodbye", lambda: "Goodbye!")
            >>> registry.get("hello")
            'Hello, World!'
            >>> registry.list()
            ['hello', 'goodbye']

        Registering a class with default arguments:

            >>> class MyModel:
            ...     def __init__(self, temperature=0.7, max_tokens=100):
            ...         self.temperature = temperature
            ...         self.max_tokens = max_tokens
            ...
            >>> model_reg = Registry("models")
            >>> model_reg.register("fast", MyModel, temperature=0.0, max_tokens=50)
            >>> model_reg.register("creative", MyModel, temperature=1.0, max_tokens=500)
            >>> fast_model = model_reg.get("fast")
            >>> fast_model.temperature
            0.0
            >>> creative_model = model_reg.get("creative")
            >>> creative_model.max_tokens
            500

        Using the decorator pattern:

            >>> evaluator_registry = Registry("evaluators")
            >>> @evaluator_registry.register_decorator("accuracy")
            ... class AccuracyEvaluator:
            ...     def evaluate(self, predictions, labels):
            ...         return sum(p == l for p, l in zip(predictions, labels)) / len(labels)
            ...
            >>> evaluator = evaluator_registry.get("accuracy")
            >>> evaluator.evaluate([1, 2, 3], [1, 2, 4])
            0.666...

        Overriding default kwargs at retrieval time:

            >>> registry = Registry("models")
            >>> registry.register("gpt", MyModel, temperature=0.5)
            >>> # Use default temperature
            >>> default_model = registry.get("gpt")
            >>> default_model.temperature
            0.5
            >>> # Override temperature
            >>> hot_model = registry.get("gpt", temperature=0.9)
            >>> hot_model.temperature
            0.9

        Registry introspection and management:

            >>> registry = Registry("test")
            >>> registry.register("item1", lambda: "one")
            >>> registry.register("item2", lambda: "two")
            >>> len(registry)
            2
            >>> "item1" in registry
            True
            >>> "item3" in registry
            False
            >>> registry.unregister("item1")
            >>> len(registry)
            1
            >>> registry.clear()
            >>> len(registry)
            0

    Notes:
        - Factories are called each time `get()` is invoked, creating new instances.
        - Use `get_factory()` to access the factory itself without instantiation.
        - The registry is thread-safe for reads but not for concurrent modifications.

    See Also:
        model_registry: The global registry for model implementations.
        probe_registry: The global registry for probe implementations.
        dataset_registry: The global registry for dataset loaders.
    """

    def __init__(self, name: str):
        """Initialize the registry with a descriptive name.

        Creates an empty registry that can store factories and their default
        keyword arguments. The name is used in error messages and the repr.

        Args:
            name: A descriptive name for this registry (e.g., "models", "probes").
                  This name appears in error messages to help identify which
                  registry an operation failed on.

        Examples:
            Creating a registry for models:

                >>> model_registry = Registry[Model]("models")
                >>> print(model_registry)
                Registry(name='models', items=[])

            Creating a registry for custom evaluators:

                >>> evaluator_registry = Registry("evaluators")
                >>> evaluator_registry.name
                'evaluators'

            Creating typed registries for different components:

                >>> from typing import Callable
                >>> metric_registry = Registry[Callable[[list, list], float]]("metrics")
                >>> transform_registry = Registry[Callable[[str], str]]("transforms")

            Using the name in error context:

                >>> registry = Registry("custom_plugins")
                >>> try:
                ...     registry.get("nonexistent")
                ... except NotFoundError as e:
                ...     print("custom_plugins" in str(e))
                True
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

        Stores the factory function or class along with default keyword arguments
        that will be used when creating instances via the `get()` method.

        Args:
            name: The unique identifier for this registration. This is the key
                  used to retrieve the item later via `get()` or `get_factory()`.
            factory: A class or factory function that creates instances of type T.
                     When `get()` is called, this factory is invoked with the
                     merged default and override kwargs.
            overwrite: If True, allows overwriting existing registrations without
                       raising an exception. Defaults to False.
            **default_kwargs: Default keyword arguments to pass when instantiating.
                              These can be overridden at retrieval time.

        Raises:
            RegistrationError: If name is already registered and overwrite is False.

        Examples:
            Basic registration of a class:

                >>> from insideLLMs.registry import Registry
                >>> class SimpleModel:
                ...     def __init__(self):
                ...         self.name = "simple"
                ...
                >>> registry = Registry("models")
                >>> registry.register("simple", SimpleModel)
                >>> registry.get("simple").name
                'simple'

            Registration with default arguments:

                >>> class ConfigurableModel:
                ...     def __init__(self, temp=0.7, max_tokens=100):
                ...         self.temp = temp
                ...         self.max_tokens = max_tokens
                ...
                >>> registry = Registry("models")
                >>> registry.register("default", ConfigurableModel)
                >>> registry.register("fast", ConfigurableModel, temp=0.0, max_tokens=50)
                >>> registry.register("creative", ConfigurableModel, temp=1.0, max_tokens=500)
                >>> registry.get("fast").temp
                0.0
                >>> registry.get("creative").max_tokens
                500

            Using a factory function instead of a class:

                >>> def create_model(name, version="1.0"):
                ...     return {"name": name, "version": version}
                ...
                >>> registry = Registry("factories")
                >>> registry.register("v1_model", create_model, name="GPT", version="1.0")
                >>> registry.register("v2_model", create_model, name="GPT", version="2.0")
                >>> registry.get("v2_model")
                {'name': 'GPT', 'version': '2.0'}

            Handling registration conflicts:

                >>> registry = Registry("test")
                >>> registry.register("item", lambda: "first")
                >>> try:
                ...     registry.register("item", lambda: "second")
                ... except RegistrationError:
                ...     print("Cannot overwrite without flag")
                Cannot overwrite without flag
                >>> registry.register("item", lambda: "second", overwrite=True)
                >>> registry.get("item")
                'second'

            Registering lazy-loaded factories:

                >>> def lazy_heavy_model(**kwargs):
                ...     # Heavy imports happen only when get() is called
                ...     import some_heavy_library  # doctest: +SKIP
                ...     return HeavyModel(**kwargs)  # doctest: +SKIP
                ...
                >>> registry.register("heavy", lazy_heavy_model)  # doctest: +SKIP

        See Also:
            register_decorator: Decorator-based registration.
            get: Retrieve and instantiate registered items.
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
        """Decorator for registering classes with the registry.

        Provides a convenient way to register classes at definition time
        using Python's decorator syntax. The class is registered immediately
        when the module containing it is imported.

        Args:
            name: The registration name. If not provided, uses the class's
                  __name__ attribute automatically.
            **default_kwargs: Default keyword arguments for instantiation.
                              These are passed to the class constructor when
                              `get()` is called.

        Returns:
            A decorator function that registers the class and returns it
            unchanged, allowing normal class instantiation to still work.

        Examples:
            Basic decorator usage with explicit name:

                >>> from insideLLMs.registry import Registry
                >>> registry = Registry("plugins")
                >>> @registry.register_decorator("my_plugin")
                ... class MyPlugin:
                ...     def __init__(self, value=10):
                ...         self.value = value
                ...
                >>> registry.get("my_plugin").value
                10

            Using class name as registration name:

                >>> registry = Registry("handlers")
                >>> @registry.register_decorator()
                ... class JSONHandler:
                ...     def handle(self, data):
                ...         return f"Handling JSON: {data}"
                ...
                >>> handler = registry.get("JSONHandler")
                >>> handler.handle({"key": "value"})
                "Handling JSON: {'key': 'value'}"

            Decorator with default arguments:

                >>> registry = Registry("models")
                >>> @registry.register_decorator("fast_model", temperature=0.0, max_tokens=50)
                ... class FastModel:
                ...     def __init__(self, temperature, max_tokens):
                ...         self.temperature = temperature
                ...         self.max_tokens = max_tokens
                ...
                >>> model = registry.get("fast_model")
                >>> model.temperature
                0.0
                >>> # Override at retrieval time
                >>> hot_model = registry.get("fast_model", temperature=1.0)
                >>> hot_model.temperature
                1.0

            Chaining decorators:

                >>> registry1 = Registry("primary")
                >>> registry2 = Registry("secondary")
                >>> @registry1.register_decorator("shared")
                ... @registry2.register_decorator("shared")
                ... class SharedClass:
                ...     pass
                ...
                >>> "shared" in registry1
                True
                >>> "shared" in registry2
                True

            The decorated class remains usable directly:

                >>> @registry.register_decorator("direct_use")
                ... class DirectUse:
                ...     def __init__(self, x=1):
                ...         self.x = x
                ...
                >>> # Can still instantiate directly
                >>> direct = DirectUse(x=5)
                >>> direct.x
                5
                >>> # Or via registry
                >>> from_registry = registry.get("direct_use")
                >>> from_registry.x
                1

        See Also:
            register: Direct registration without decorators.
        """

        def decorator(cls: type[T]) -> type[T]:
            registration_name = name or cls.__name__
            self.register(registration_name, cls, **default_kwargs)
            return cls

        return decorator

    def get(self, registration_name: str, **override_kwargs: Any) -> T:
        """Get an instance of the registered item by invoking its factory.

        Creates and returns a new instance by calling the registered factory
        function or class constructor with the merged default and override
        keyword arguments. Each call to `get()` creates a new instance.

        Args:
            registration_name: The name under which the item was registered.
            **override_kwargs: Keyword arguments that override the defaults
                               specified at registration time. These take
                               precedence over default_kwargs.

        Returns:
            An instance of type T created by the registered factory.

        Raises:
            NotFoundError: If the name is not registered. The error message
                           includes a list of available registrations.

        Examples:
            Basic retrieval:

                >>> from insideLLMs.registry import Registry
                >>> registry = Registry("greetings")
                >>> registry.register("hello", lambda: "Hello, World!")
                >>> registry.get("hello")
                'Hello, World!'

            Retrieval with default arguments:

                >>> class Model:
                ...     def __init__(self, temp=0.7, tokens=100):
                ...         self.temp = temp
                ...         self.tokens = tokens
                ...
                >>> registry = Registry("models")
                >>> registry.register("gpt", Model, temp=0.5, tokens=200)
                >>> model = registry.get("gpt")
                >>> model.temp
                0.5
                >>> model.tokens
                200

            Overriding defaults at retrieval time:

                >>> # Override just one parameter
                >>> hot_model = registry.get("gpt", temp=1.0)
                >>> hot_model.temp
                1.0
                >>> hot_model.tokens  # Still uses default
                200
                >>> # Override all parameters
                >>> custom = registry.get("gpt", temp=0.0, tokens=50)
                >>> custom.temp
                0.0
                >>> custom.tokens
                50

            Each call creates a new instance:

                >>> registry = Registry("counters")
                >>> class Counter:
                ...     count = 0
                ...     def increment(self):
                ...         self.count += 1
                ...         return self.count
                ...
                >>> registry.register("counter", Counter)
                >>> c1 = registry.get("counter")
                >>> c1.increment()
                1
                >>> c2 = registry.get("counter")
                >>> c2.increment()  # New instance, count starts at 0
                1

            Handling missing registrations:

                >>> registry = Registry("test")
                >>> registry.register("exists", lambda: "found")
                >>> try:
                ...     registry.get("nonexistent")
                ... except NotFoundError as e:
                ...     print("Available:", "exists" in str(e))
                Available: True

        Notes:
            - Override kwargs completely replace default kwargs with the same key
            - The factory is called synchronously; for async factories, use get_factory()

        See Also:
            get_factory: Get the factory without instantiation.
            register: Register items with default arguments.
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
        """Get the factory function or class without instantiating.

        Returns the raw factory that was registered, allowing you to inspect
        it, call it with custom logic, or use it for subclassing. Unlike
        `get()`, this does not invoke the factory or apply default kwargs.

        Args:
            name: The registration name to look up.

        Returns:
            The registered factory function or class. This is exactly what
            was passed to `register()` as the `factory` argument.

        Raises:
            NotFoundError: If the name is not registered.

        Examples:
            Getting a class for subclassing:

                >>> from insideLLMs.registry import Registry
                >>> class BaseModel:
                ...     def predict(self, x):
                ...         return x * 2
                ...
                >>> registry = Registry("models")
                >>> registry.register("base", BaseModel)
                >>> ModelClass = registry.get_factory("base")
                >>> class ExtendedModel(ModelClass):
                ...     def predict(self, x):
                ...         return super().predict(x) + 1
                ...
                >>> ExtendedModel().predict(5)
                11

            Inspecting factory properties:

                >>> registry = Registry("functions")
                >>> def complex_factory(a, b, c=10):
                ...     '''Create a complex object.'''
                ...     return {"a": a, "b": b, "c": c}
                ...
                >>> registry.register("complex", complex_factory)
                >>> factory = registry.get_factory("complex")
                >>> factory.__name__
                'complex_factory'
                >>> factory.__doc__
                'Create a complex object.'

            Custom instantiation logic:

                >>> registry = Registry("async_models")
                >>> class AsyncModel:
                ...     def __init__(self, config):
                ...         self.config = config
                ...
                >>> registry.register("async", AsyncModel)
                >>> Factory = registry.get_factory("async")
                >>> # Custom async initialization
                >>> async def create_model():  # doctest: +SKIP
                ...     config = await load_config_async()
                ...     return Factory(config=config)

            Comparing factories:

                >>> registry = Registry("test")
                >>> class MyClass:
                ...     pass
                ...
                >>> registry.register("item", MyClass)
                >>> registry.get_factory("item") is MyClass
                True

        See Also:
            get: Get an instantiated object.
            info: Get metadata about the registration.
        """
        if name not in self._registry:
            raise NotFoundError(f"'{name}' not found in {self.name} registry.")
        return self._registry[name]["factory"]

    def list(self) -> list[str]:
        """List all registered names in the registry.

        Returns all names that have been registered, in the order they
        were added. This is useful for discovery, debugging, and
        presenting available options to users.

        Returns:
            A list of all registration names as strings. Returns an
            empty list if no items are registered.

        Examples:
            Basic listing:

                >>> from insideLLMs.registry import Registry
                >>> registry = Registry("models")
                >>> registry.register("gpt3", lambda: "GPT-3")
                >>> registry.register("gpt4", lambda: "GPT-4")
                >>> registry.register("claude", lambda: "Claude")
                >>> registry.list()
                ['gpt3', 'gpt4', 'claude']

            Empty registry:

                >>> empty_registry = Registry("empty")
                >>> empty_registry.list()
                []

            Using list for discovery:

                >>> from insideLLMs.registry import model_registry, ensure_builtins_registered
                >>> ensure_builtins_registered()
                >>> available = model_registry.list()
                >>> print("Available models:", available)  # doctest: +SKIP
                Available models: ['dummy', 'openai', 'anthropic', ...]

            Filtering available items:

                >>> registry = Registry("items")
                >>> registry.register("fast_model", lambda: "fast")
                >>> registry.register("slow_model", lambda: "slow")
                >>> registry.register("fast_probe", lambda: "probe")
                >>> fast_items = [n for n in registry.list() if n.startswith("fast")]
                >>> fast_items
                ['fast_model', 'fast_probe']

            Presenting options to users:

                >>> def select_model(registry):
                ...     available = registry.list()
                ...     if not available:
                ...         print("No models available")
                ...         return None
                ...     print("Available models:")
                ...     for i, name in enumerate(available, 1):
                ...         print(f"  {i}. {name}")
                ...     return available[0]  # Default to first

        See Also:
            is_registered: Check if a specific name is registered.
            info: Get detailed information about a registration.
        """
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a name is registered in this registry.

        Provides a safe way to check for existence before attempting to
        retrieve an item, avoiding the need to catch NotFoundError.
        This is equivalent to using the `in` operator.

        Args:
            name: The registration name to check for.

        Returns:
            True if the name is registered, False otherwise.

        Examples:
            Basic existence check:

                >>> from insideLLMs.registry import Registry
                >>> registry = Registry("models")
                >>> registry.register("gpt4", lambda: "GPT-4")
                >>> registry.is_registered("gpt4")
                True
                >>> registry.is_registered("nonexistent")
                False

            Conditional retrieval:

                >>> def safe_get(registry, name, default=None):
                ...     if registry.is_registered(name):
                ...         return registry.get(name)
                ...     return default
                ...
                >>> registry = Registry("test")
                >>> registry.register("item", lambda: "value")
                >>> safe_get(registry, "item")
                'value'
                >>> safe_get(registry, "missing", "default")
                'default'

            Using in conditional logic:

                >>> registry = Registry("models")
                >>> registry.register("openai", lambda: "OpenAI Model")
                >>> if registry.is_registered("openai"):
                ...     model = registry.get("openai")
                ...     print(f"Using: {model}")
                ... else:
                ...     print("OpenAI not available, using fallback")
                Using: OpenAI Model

            Equivalent to `in` operator:

                >>> registry = Registry("test")
                >>> registry.register("item", lambda: "value")
                >>> registry.is_registered("item") == ("item" in registry)
                True

        See Also:
            __contains__: The `in` operator implementation.
            get: Retrieve registered items.
            list: Get all registered names.
        """
        return name in self._registry

    def unregister(self, name: str) -> None:
        """Remove a registration from the registry.

        Removes the specified registration, freeing the name for reuse.
        This is useful for testing, plugin unloading, or replacing
        implementations at runtime.

        Args:
            name: The registration name to remove.

        Raises:
            NotFoundError: If the name is not currently registered.

        Examples:
            Basic unregistration:

                >>> from insideLLMs.registry import Registry
                >>> registry = Registry("models")
                >>> registry.register("temp", lambda: "temporary")
                >>> "temp" in registry
                True
                >>> registry.unregister("temp")
                >>> "temp" in registry
                False

            Replacing a registration:

                >>> registry = Registry("models")
                >>> registry.register("model", lambda: "version1")
                >>> # Remove old version before re-registering
                >>> registry.unregister("model")
                >>> registry.register("model", lambda: "version2")
                >>> registry.get("model")
                'version2'

            Safe unregistration:

                >>> def safe_unregister(registry, name):
                ...     if registry.is_registered(name):
                ...         registry.unregister(name)
                ...         return True
                ...     return False
                ...
                >>> registry = Registry("test")
                >>> registry.register("item", lambda: "value")
                >>> safe_unregister(registry, "item")
                True
                >>> safe_unregister(registry, "nonexistent")
                False

            Handling the error:

                >>> registry = Registry("test")
                >>> try:
                ...     registry.unregister("nonexistent")
                ... except NotFoundError as e:
                ...     print("Item not found")
                Item not found

        Notes:
            - Unlike `clear()`, this only removes a single registration
            - The factory is not called during unregistration
            - Any existing instances created from the factory remain valid

        See Also:
            clear: Remove all registrations.
            register: Add new registrations.
        """
        if name not in self._registry:
            raise NotFoundError(f"'{name}' not found in {self.name} registry.")
        del self._registry[name]

    def clear(self) -> None:
        """Remove all registrations from the registry.

        Clears the entire registry, removing all registered items. This is
        useful for testing scenarios where you need a clean slate, or when
        re-initializing the system.

        Examples:
            Basic clearing:

                >>> from insideLLMs.registry import Registry
                >>> registry = Registry("models")
                >>> registry.register("a", lambda: "A")
                >>> registry.register("b", lambda: "B")
                >>> len(registry)
                2
                >>> registry.clear()
                >>> len(registry)
                0

            Clearing before re-registration:

                >>> registry = Registry("plugins")
                >>> registry.register("plugin1", lambda: "v1")
                >>> registry.register("plugin2", lambda: "v1")
                >>> # Reset and register new versions
                >>> registry.clear()
                >>> registry.register("plugin1", lambda: "v2")
                >>> registry.list()
                ['plugin1']

            Testing isolation:

                >>> def test_my_feature():
                ...     registry = Registry("test")
                ...     try:
                ...         registry.register("test_item", lambda: "test")
                ...         result = registry.get("test_item")
                ...         assert result == "test"
                ...     finally:
                ...         registry.clear()  # Clean up after test

            Safe to call on empty registry:

                >>> empty = Registry("empty")
                >>> empty.clear()  # No error
                >>> len(empty)
                0

        Notes:
            - This operation cannot be undone
            - Factories are not called during clearing
            - Existing instances remain valid after clearing

        See Also:
            unregister: Remove a single registration.
            register_builtins: Re-register default items.
        """
        self._registry.clear()

    def info(self, name: str) -> dict[str, Any]:
        """Get detailed information about a registration.

        Returns metadata about the registered item including its factory
        name, default arguments, and documentation. Useful for introspection,
        debugging, and building discovery UIs.

        Args:
            name: The registration name to look up.

        Returns:
            A dictionary containing:
                - name (str): The registration name
                - factory (str): The factory's __name__ or string representation
                - default_kwargs (dict): Default arguments for instantiation
                - doc (str or None): The factory's docstring

        Raises:
            NotFoundError: If the name is not registered.

        Examples:
            Getting info about a registered class:

                >>> from insideLLMs.registry import Registry
                >>> class MyModel:
                ...     '''A custom model implementation.'''
                ...     def __init__(self, temp=0.7):
                ...         self.temp = temp
                ...
                >>> registry = Registry("models")
                >>> registry.register("my_model", MyModel, temp=0.5)
                >>> info = registry.info("my_model")
                >>> info["name"]
                'my_model'
                >>> info["factory"]
                'MyModel'
                >>> info["default_kwargs"]
                {'temp': 0.5}
                >>> info["doc"]
                'A custom model implementation.'

            Building a help/discovery system:

                >>> def print_available(registry):
                ...     for name in registry.list():
                ...         info = registry.info(name)
                ...         print(f"{name}: {info['doc'] or 'No description'}")
                ...
                >>> registry = Registry("commands")
                >>> registry.register("help", lambda: None)
                >>> # print_available(registry)  # doctest: +SKIP

            Comparing registrations:

                >>> registry = Registry("test")
                >>> registry.register("v1", lambda: "1", version="1.0")
                >>> registry.register("v2", lambda: "2", version="2.0")
                >>> registry.info("v1")["default_kwargs"]["version"]
                '1.0'
                >>> registry.info("v2")["default_kwargs"]["version"]
                '2.0'

            Debugging registration issues:

                >>> def debug_registration(registry, name):
                ...     try:
                ...         info = registry.info(name)
                ...         print(f"Factory: {info['factory']}")
                ...         print(f"Defaults: {info['default_kwargs']}")
                ...     except NotFoundError:
                ...         print(f"'{name}' not found. Available: {registry.list()}")

        See Also:
            get_factory: Get the actual factory object.
            list: Get all registration names.
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
        """Check if a name is registered using the `in` operator.

        Enables Pythonic membership testing with the `in` keyword.
        Equivalent to calling `is_registered(name)`.

        Args:
            name: The registration name to check.

        Returns:
            True if registered, False otherwise.

        Examples:
            Using the `in` operator:

                >>> from insideLLMs.registry import Registry
                >>> registry = Registry("test")
                >>> registry.register("item", lambda: "value")
                >>> "item" in registry
                True
                >>> "missing" in registry
                False

            In conditional statements:

                >>> if "item" in registry:
                ...     value = registry.get("item")
                ... else:
                ...     value = "default"
                >>> value
                'value'

            With `not in`:

                >>> "nonexistent" not in registry
                True

        See Also:
            is_registered: Explicit method for the same check.
        """
        return self.is_registered(name)

    def __len__(self) -> int:
        """Return the number of registered items.

        Enables use of `len()` with the registry to get the count
        of registered items.

        Returns:
            The number of items currently registered.

        Examples:
            Getting registry size:

                >>> from insideLLMs.registry import Registry
                >>> registry = Registry("test")
                >>> len(registry)
                0
                >>> registry.register("a", lambda: "A")
                >>> registry.register("b", lambda: "B")
                >>> len(registry)
                2
                >>> registry.unregister("a")
                >>> len(registry)
                1

            Checking if registry is empty:

                >>> registry = Registry("empty")
                >>> if len(registry) == 0:
                ...     print("Registry is empty")
                Registry is empty

            Or using truthiness (empty registry is falsy for len):

                >>> bool(len(registry))
                False
        """
        return len(self._registry)

    def __repr__(self) -> str:
        """Return a string representation of the registry.

        Provides a readable representation showing the registry name
        and all registered items. Useful for debugging and interactive
        exploration.

        Returns:
            A string in the format: Registry(name='...', items=[...])

        Examples:
            Viewing registry state:

                >>> from insideLLMs.registry import Registry
                >>> registry = Registry("models")
                >>> registry.register("gpt", lambda: "GPT")
                >>> registry.register("claude", lambda: "Claude")
                >>> print(registry)
                Registry(name='models', items=['gpt', 'claude'])

            In the REPL:

                >>> registry = Registry("empty")
                >>> registry
                Registry(name='empty', items=[])

            Debugging:

                >>> registry = Registry("debug")
                >>> registry.register("item", lambda: "value")
                >>> repr(registry)
                "Registry(name='debug', items=['item'])"
        """
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

    This is an internal utility function that enables registration of
    heavy dependencies (like HuggingFace transformers or ML frameworks)
    without importing them at registration time. The actual import
    happens only when the factory is invoked via `registry.get()`.

    Args:
        module_path: The fully qualified module path to import from
                     (e.g., "insideLLMs.models.huggingface").
        class_name: The name of the class to import from the module
                    (e.g., "HuggingFaceModel").

    Returns:
        A factory function that, when called with kwargs, imports the
        specified module, retrieves the class, and instantiates it.

    Examples:
        Creating a lazy factory for a heavy model:

            >>> factory = _lazy_import_factory(
            ...     "insideLLMs.models.huggingface",
            ...     "HuggingFaceModel"
            ... )  # doctest: +SKIP
            >>> factory.__name__
            'HuggingFaceModel'  # doctest: +SKIP
            >>> # The import only happens when factory is called:
            >>> model = factory(model_name="gpt2")  # doctest: +SKIP

        Registering with lazy loading:

            >>> from insideLLMs.registry import model_registry
            >>> model_registry.register(
            ...     "heavy_model",
            ...     _lazy_import_factory("some.heavy.module", "HeavyClass")
            ... )  # doctest: +SKIP
            >>> # No import has happened yet
            >>> model = model_registry.get("heavy_model")  # doctest: +SKIP
            >>> # Now the import occurs

        How it differs from direct registration:

            >>> # Direct registration - imports immediately:
            >>> from some.module import HeavyClass  # doctest: +SKIP
            >>> registry.register("direct", HeavyClass)  # doctest: +SKIP
            >>>
            >>> # Lazy registration - imports only on use:
            >>> registry.register(
            ...     "lazy",
            ...     _lazy_import_factory("some.module", "HeavyClass")
            ... )  # doctest: +SKIP

    Notes:
        - This is a private function used internally by register_builtins()
        - The returned factory sets __name__ and __doc__ for introspection
        - Import errors surface when get() is called, not at registration

    See Also:
        register_builtins: Uses this for heavy dependencies.
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

    This function populates the global model_registry, probe_registry, and
    dataset_registry with all built-in implementations. It is called
    automatically via ensure_builtins_registered() when the library is used,
    but can also be called manually to reset registries to their defaults.

    The function uses lazy imports for heavy dependencies (like HuggingFace
    transformers) to keep import times fast. Light dependencies are imported
    directly.

    Registered Models:
        - dummy: DummyModel for testing
        - openai: OpenAI API models (GPT-3.5, GPT-4, etc.)
        - openrouter: OpenRouter OpenAI-compatible API models
        - anthropic: Anthropic API models (Claude)
        - gemini: Google Gemini models
        - cohere: Cohere API models
        - huggingface: HuggingFace transformers (lazy loaded)
        - llamacpp: llama.cpp local models
        - ollama: Ollama local models
        - vllm: vLLM server models

    Registered Probes:
        - logic: LogicProbe for logical reasoning evaluation
        - factuality: FactualityProbe for fact-checking
        - bias: BiasProbe for bias detection
        - attack: AttackProbe for adversarial testing
        - prompt_injection: PromptInjectionProbe
        - jailbreak: JailbreakProbe
        - code_generation: CodeGenerationProbe
        - code_explanation: CodeExplanationProbe
        - code_debug: CodeDebugProbe
        - instruction_following: InstructionFollowingProbe
        - multi_step_task: MultiStepTaskProbe
        - constraint_compliance: ConstraintComplianceProbe

    Registered Dataset Loaders:
        - csv: load_csv_dataset for CSV files
        - jsonl: load_jsonl_dataset for JSON Lines files
        - hf: load_hf_dataset for HuggingFace datasets

    Examples:
        Resetting registries to defaults:

            >>> from insideLLMs.registry import (
            ...     register_builtins, model_registry, probe_registry
            ... )
            >>> # Add a custom model
            >>> model_registry.register("custom", lambda: "custom")
            >>> "custom" in model_registry
            True
            >>> # Reset to defaults
            >>> register_builtins()
            >>> "custom" in model_registry
            False
            >>> "openai" in model_registry
            True

        Checking available built-ins:

            >>> from insideLLMs.registry import model_registry, ensure_builtins_registered
            >>> ensure_builtins_registered()
            >>> "dummy" in model_registry
            True
            >>> "anthropic" in model_registry
            True

        Using built-in models:

            >>> from insideLLMs.registry import model_registry, ensure_builtins_registered
            >>> ensure_builtins_registered()
            >>> dummy = model_registry.get("dummy")  # doctest: +SKIP
            >>> # OpenAI model (requires API key)
            >>> openai = model_registry.get("openai", model_name="gpt-4")  # doctest: +SKIP

        Using built-in probes:

            >>> from insideLLMs.registry import probe_registry, ensure_builtins_registered
            >>> ensure_builtins_registered()
            >>> logic = probe_registry.get("logic")  # doctest: +SKIP
            >>> bias = probe_registry.get("bias")  # doctest: +SKIP

    Notes:
        - Calling this function clears all existing registrations first
        - Heavy dependencies are lazy-loaded for fast import times
        - ImportErrors for optional dependencies are raised at get() time

    See Also:
        ensure_builtins_registered: Lazy initialization wrapper.
        load_entrypoint_plugins: Loads third-party plugin registrations.
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
        "openrouter", _lazy_import_factory("insideLLMs.models.openrouter", "OpenRouterModel")
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
        JudgeScoredProbe,
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
    probe_registry.register("judge", JudgeScoredProbe)

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
    """Call a plugin registration function with appropriate arguments.

    This is an internal utility that intelligently invokes plugin registration
    functions. It inspects the function signature and either:
    - Calls it with no arguments if it takes none
    - Passes the global registries if it accepts parameters

    This allows plugins to be written in two styles:

    1. Zero-argument (accesses globals):
       def register():
           from insideLLMs.registry import model_registry
           model_registry.register("my_model", MyModel)

    2. Parameterized (receives registries):
       def register(model_registry, probe_registry, dataset_registry):
           model_registry.register("my_model", MyModel)

    Args:
        fn: A callable plugin registration function. Can take either
            no parameters or named registry parameters.

    Examples:
        Calling a zero-argument plugin:

            >>> def simple_plugin():
            ...     print("Plugin loaded!")
            ...
            >>> _call_plugin_register(simple_plugin)  # doctest: +SKIP
            Plugin loaded!

        Calling a parameterized plugin:

            >>> def param_plugin(model_registry, probe_registry, dataset_registry):
            ...     model_registry.register("plugin_model", lambda: "model")
            ...
            >>> _call_plugin_register(param_plugin)  # doctest: +SKIP

        Plugins can use either style:

            >>> # Style 1: Global access
            >>> def global_style():
            ...     from insideLLMs.registry import model_registry
            ...     model_registry.register("v1", lambda: "1")
            ...
            >>> # Style 2: Parameter injection
            >>> def injected_style(model_registry, **_):
            ...     model_registry.register("v2", lambda: "2")

    Notes:
        - This is a private function used by load_entrypoint_plugins()
        - If signature inspection fails, the function is called with no args
        - Only model_registry, probe_registry, and dataset_registry are passed

    See Also:
        load_entrypoint_plugins: Uses this to invoke discovered plugins.
    """
    sig: Signature
    try:
        sig = signature(fn)
    except (ValueError, TypeError):
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

    Discovers and loads plugins via Python's entry points mechanism. Each
    entry point should resolve to a callable that registers models, probes,
    or datasets with the global registries.

    Plugins are discovered from the "insidellms.plugins" entry point group
    by default, allowing third-party packages to extend insideLLMs.

    Args:
        group: The entry point group to search for plugins.
               Defaults to "insidellms.plugins".
        enabled: Override whether plugins are enabled. If None (default),
                 checks INSIDELLMS_DISABLE_PLUGINS environment variable.
                 Set to False to disable all plugin loading.

    Returns:
        A dictionary mapping entry point names to their import paths
        for all successfully loaded plugins.

    Environment Variables:
        INSIDELLMS_DISABLE_PLUGINS: Set to "1", "true", or "yes" to disable
                                    plugin loading. Default is enabled.

    Examples:
        Default plugin loading:

            >>> from insideLLMs.registry import load_entrypoint_plugins
            >>> loaded = load_entrypoint_plugins()  # doctest: +SKIP
            >>> print(loaded)  # doctest: +SKIP
            {'my_plugin': 'my_package.plugin:register'}

        Disabling plugins via parameter:

            >>> loaded = load_entrypoint_plugins(enabled=False)
            >>> loaded
            {}

        Disabling plugins via environment:

            >>> import os
            >>> os.environ["INSIDELLMS_DISABLE_PLUGINS"] = "1"
            >>> loaded = load_entrypoint_plugins()
            >>> loaded
            {}
            >>> del os.environ["INSIDELLMS_DISABLE_PLUGINS"]

        Custom entry point group:

            >>> loaded = load_entrypoint_plugins(group="myapp.models")  # doctest: +SKIP

    Plugin Development:
        To create a plugin, add an entry point to your package's setup:

        pyproject.toml:
            [project.entry-points."insidellms.plugins"]
            my_plugin = "my_package.plugin:register"

        setup.py:
            entry_points={
                "insidellms.plugins": [
                    "my_plugin = my_package.plugin:register"
                ]
            }

        Plugin registration function (zero-argument style):

            # my_package/plugin.py
            def register():
                from insideLLMs.registry import model_registry
                from my_package import MyCustomModel
                model_registry.register("my_custom", MyCustomModel)

        Plugin registration function (parameterized style):

            def register(model_registry, probe_registry, dataset_registry):
                from my_package import MyCustomModel, MyCustomProbe
                model_registry.register("my_custom", MyCustomModel)
                probe_registry.register("my_probe", MyCustomProbe)

    Notes:
        - Failed plugin loads emit RuntimeWarning but don't stop execution
        - Non-callable entry points are skipped with a warning
        - This function is called automatically by ensure_builtins_registered()

    See Also:
        ensure_builtins_registered: Calls this after registering builtins.
        _call_plugin_register: Invokes individual plugin functions.
        PLUGIN_ENTRYPOINT_GROUP: The default entry point group name.
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
            selected = list(eps.get(group, []))  # type: ignore[attr-defined]  # Python <3.10 EntryPoints API
    except (ImportError, AttributeError):
        return {}

    # Deterministic plugin load order: entry point iteration order is not guaranteed.
    selected = sorted(selected, key=lambda ep: (getattr(ep, "name", ""), getattr(ep, "value", "")))

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
    """Ensure built-in registrations and plugins are loaded.

    This is the primary initialization function for the registry system.
    It lazily loads built-in registrations the first time it's called,
    and then loads any installed plugins. Subsequent calls are no-ops.

    This function is safe to call multiple times - it only initializes
    once per process. It's automatically called by various parts of the
    library that need the registries to be populated.

    The initialization order is:
    1. Register all built-in models, probes, and datasets
    2. Discover and load any installed plugins via entry points

    Examples:
        Ensuring registries are ready before use:

            >>> from insideLLMs.registry import (
            ...     ensure_builtins_registered, model_registry
            ... )
            >>> ensure_builtins_registered()
            >>> "dummy" in model_registry
            True
            >>> "openai" in model_registry
            True

        Safe to call multiple times:

            >>> ensure_builtins_registered()  # First call initializes
            >>> ensure_builtins_registered()  # Subsequent calls are no-ops
            >>> ensure_builtins_registered()  # Still a no-op

        Using with model retrieval:

            >>> from insideLLMs.registry import model_registry, ensure_builtins_registered
            >>> ensure_builtins_registered()
            >>> # Now safe to get models
            >>> model_registry.list()  # doctest: +SKIP
            ['dummy', 'openai', 'anthropic', ...]

        In application startup:

            >>> def initialize_app():
            ...     ensure_builtins_registered()
            ...     # Now registries are ready for use
            ...     return True
            ...
            >>> initialize_app()
            True

    Notes:
        - Thread-safe for reading but not for concurrent initialization
        - ImportErrors during builtin registration are silently ignored
          (some optional dependencies may not be installed)
        - Plugin failures emit warnings but don't prevent initialization
        - Use register_builtins() directly if you need to reset registries

    See Also:
        register_builtins: Manually reset and re-register builtins.
        load_entrypoint_plugins: Manually load/reload plugins.
    """
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
