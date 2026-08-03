"""Registry exception types."""

from insideLLMs._exceptions.base import InsideLLMsError


class RegistryError(InsideLLMsError):
    """Base exception for registry errors.

    This is the parent class for all exceptions that occur during registry
    operations, including registration and lookup of models, probes, and
    evaluators. Catching this exception will handle any registry-related
    failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any registry-related error:

    >>> try:
    ...     registry.register("my_model", model_class)
    ...     model = registry.get("my_model")
    ... except RegistryError as e:
    ...     print(f"Registry operation failed: {e}")

    Distinguishing between registry error types:

    >>> try:
    ...     registry.register("custom_probe", ProbeClass)
    ... except AlreadyRegisteredError:
    ...     print("Probe already exists, skipping registration")
    ... except RegistryError as e:
    ...     print(f"Registry error: {e}")

    Notes
    -----
    Subclasses include: AlreadyRegisteredError, NotRegisteredError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    AlreadyRegisteredError : When registering duplicate entries.
    NotRegisteredError : When looking up missing entries.
    """

    pass


class AlreadyRegisteredError(RegistryError):
    """Raised when trying to register a duplicate entry.

    This exception is raised when attempting to register an item with
    a name that already exists in the registry. This prevents accidental
    overwrites of existing registrations.

    Parameters
    ----------
    name : str
        The name that was already registered.
    registry_type : str, default "item"
        The type of item being registered (e.g., "model", "probe").

    Attributes
    ----------
    details : dict
        Contains 'name' and 'registry_type'.

    Examples
    --------
    Handling duplicate registration gracefully:

    >>> try:
    ...     registry.register("gpt-4", GPT4Model)
    ... except AlreadyRegisteredError as e:
    ...     print(f"{e.details['registry_type']} '{e.details['name']}' exists")
    ...     # Use existing registration
    ...     pass

    Implementing register-or-update pattern:

    >>> def register_or_update(registry, name, item, item_type="item"):
    ...     try:
    ...         registry.register(name, item)
    ...         print(f"Registered new {item_type}: {name}")
    ...     except AlreadyRegisteredError:
    ...         registry.update(name, item)
    ...         print(f"Updated existing {item_type}: {name}")

    Force-registering with warning:

    >>> def force_register(registry, name, item):
    ...     try:
    ...         registry.register(name, item)
    ...     except AlreadyRegisteredError:
    ...         logging.warning(f"Overwriting existing registration: {name}")
    ...         registry.unregister(name)
    ...         registry.register(name, item)

    See Also
    --------
    NotRegisteredError : When looking up missing entries.
    """

    def __init__(self, name: str, registry_type: str = "item"):
        super().__init__(
            f"{registry_type.capitalize()} already registered: {name}",
            {"name": name, "registry_type": registry_type},
        )


class NotRegisteredError(RegistryError):
    """Raised when looking up an unregistered entry.

    This exception is raised when attempting to retrieve or use an item
    that has not been registered. This helps identify missing dependencies
    or configuration issues.

    Parameters
    ----------
    name : str
        The name that was not found in the registry.
    registry_type : str, default "item"
        The type of item being looked up (e.g., "model", "probe").

    Attributes
    ----------
    details : dict
        Contains 'name' and 'registry_type'.

    Examples
    --------
    Handling missing registration:

    >>> try:
    ...     model_class = registry.get("custom_model")
    ... except NotRegisteredError as e:
    ...     print(f"{e.details['registry_type']} not found: {e.details['name']}")
    ...     print("Available items:", registry.list())

    Implementing lazy registration:

    >>> def get_or_register(registry, name, factory_fn, item_type="item"):
    ...     try:
    ...         return registry.get(name)
    ...     except NotRegisteredError:
    ...         item = factory_fn()
    ...         registry.register(name, item)
    ...         return item

    Providing helpful error messages:

    >>> try:
    ...     probe = registry.get(probe_name)
    ... except NotRegisteredError as e:
    ...     available = registry.list()
    ...     print(f"Probe '{e.details['name']}' not registered.")
    ...     print(f"Available probes: {', '.join(available)}")
    ...     # Suggest similar names
    ...     from difflib import get_close_matches
    ...     suggestions = get_close_matches(e.details['name'], available)
    ...     if suggestions:
    ...         print(f"Did you mean: {suggestions[0]}?")

    See Also
    --------
    AlreadyRegisteredError : When registering duplicate entries.
    ModelNotFoundError : Specific error for models.
    ProbeNotFoundError : Specific error for probes.
    """

    def __init__(self, name: str, registry_type: str = "item"):
        super().__init__(
            f"{registry_type.capitalize()} not registered: {name}",
            {"name": name, "registry_type": registry_type},
        )
