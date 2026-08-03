"""Configuration exception types."""

from typing import Any, Optional

from insideLLMs._exceptions.base import InsideLLMsError


class ConfigurationError(InsideLLMsError):
    """Base exception for configuration errors.

    This is the parent class for all exceptions that occur during
    configuration operations, including loading, parsing, and validation.
    Catching this exception will handle any configuration-related failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any configuration-related error:

    >>> try:
    ...     config = ConfigLoader.load("config.yaml")
    ...     config.validate()
    ... except ConfigurationError as e:
    ...     print(f"Configuration error: {e}")
    ...     # Fall back to default configuration
    ...     config = Config.default()

    Distinguishing between configuration error types:

    >>> try:
    ...     config = ConfigLoader.load(path)
    ... except ConfigNotFoundError:
    ...     print("Config file not found, creating default...")
    ...     config = Config.default()
    ...     config.save(path)
    ... except ConfigParseError as e:
    ...     print(f"Parse error at line {e.details.get('line')}")
    ... except ConfigValidationError as e:
    ...     print(f"Invalid value for {e.details['field']}")

    Notes
    -----
    Subclasses include: ConfigValidationError, ConfigNotFoundError,
    ConfigParseError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    ConfigValidationError : For field-level validation failures.
    """

    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails.

    This exception is raised when a configuration value does not meet
    the required constraints, type requirements, or business rules.

    Parameters
    ----------
    field : str
        The name of the configuration field that failed validation.
    reason : str
        A description of why validation failed.
    value : Any, optional
        The invalid value (truncated to 100 chars in details).

    Attributes
    ----------
    details : dict
        Contains 'field', 'reason', and optionally 'value'.

    Examples
    --------
    Handling validation errors with field information:

    >>> try:
    ...     config.validate()
    ... except ConfigValidationError as e:
    ...     print(f"Invalid config: {e.details['field']}")
    ...     print(f"  Reason: {e.details['reason']}")
    ...     if 'value' in e.details:
    ...         print(f"  Provided value: {e.details['value']}")

    Providing helpful error messages to users:

    >>> try:
    ...     config = Config(temperature=2.5)
    ... except ConfigValidationError as e:
    ...     if e.details['field'] == 'temperature':
    ...         print("Temperature must be between 0.0 and 2.0")
    ...     else:
    ...         print(f"Configuration error: {e}")

    Validating multiple fields and collecting errors:

    >>> errors = []
    >>> for field, value in user_config.items():
    ...     try:
    ...         validate_field(field, value)
    ...     except ConfigValidationError as e:
    ...         errors.append(e)
    >>> if errors:
    ...     print(f"Found {len(errors)} configuration errors")

    See Also
    --------
    ConfigParseError : When configuration cannot be parsed.
    DatasetValidationError : For dataset validation failures.
    """

    def __init__(self, field: str, reason: str, value: Any = None):
        details = {"field": field, "reason": reason}
        if value is not None:
            details["value"] = str(value)[:100]
        super().__init__(f"Invalid configuration for '{field}': {reason}", details)


class ConfigNotFoundError(ConfigurationError):
    """Raised when configuration file is not found.

    This exception is raised when attempting to load a configuration
    file from a path that does not exist or is not accessible.

    Parameters
    ----------
    path : str
        The path to the configuration file that was not found.

    Attributes
    ----------
    details : dict
        Contains 'path' key with the attempted path.

    Examples
    --------
    Handling missing config with defaults:

    >>> try:
    ...     config = ConfigLoader.load("custom_config.yaml")
    ... except ConfigNotFoundError as e:
    ...     print(f"Config not found: {e.details['path']}")
    ...     config = Config.default()

    Creating config file if missing:

    >>> def load_or_create_config(path: str):
    ...     try:
    ...         return ConfigLoader.load(path)
    ...     except ConfigNotFoundError:
    ...         config = Config.default()
    ...         config.save(path)
    ...         print(f"Created default config at {path}")
    ...         return config

    Searching multiple config locations:

    >>> config_paths = ["./config.yaml", "~/.insidellms/config.yaml"]
    >>> for path in config_paths:
    ...     try:
    ...         config = ConfigLoader.load(os.path.expanduser(path))
    ...         break
    ...     except ConfigNotFoundError:
    ...         continue
    ... else:
    ...     raise ConfigNotFoundError("No configuration file found")

    See Also
    --------
    ConfigParseError : When config exists but cannot be parsed.
    DatasetNotFoundError : For missing dataset files.
    """

    def __init__(self, path: str):
        super().__init__(
            f"Configuration file not found: {path}",
            {"path": path},
        )


class ConfigParseError(ConfigurationError):
    """Raised when configuration parsing fails.

    This exception is raised when a configuration file exists but
    cannot be parsed due to syntax errors, encoding issues, or
    invalid structure.

    Parameters
    ----------
    path : str
        The path to the configuration file.
    reason : str
        A description of why parsing failed.
    line : int, optional
        The line number where the error occurred.

    Attributes
    ----------
    details : dict
        Contains 'path', 'reason', and optionally 'line'.

    Examples
    --------
    Handling parse errors with line information:

    >>> try:
    ...     config = ConfigLoader.load("config.yaml")
    ... except ConfigParseError as e:
    ...     print(f"Failed to parse {e.details['path']}")
    ...     print(f"  Error: {e.details['reason']}")
    ...     if 'line' in e.details:
    ...         print(f"  At line: {e.details['line']}")

    Providing helpful debugging information:

    >>> try:
    ...     config = ConfigLoader.load(path)
    ... except ConfigParseError as e:
    ...     if 'line' in e.details:
    ...         with open(path) as f:
    ...             lines = f.readlines()
    ...             error_line = lines[e.details['line'] - 1]
    ...             print(f"Problematic line: {error_line.strip()}")

    Attempting different parsers:

    >>> def load_config_flexible(path: str):
    ...     for loader in [yaml_loader, json_loader, toml_loader]:
    ...         try:
    ...             return loader(path)
    ...         except ConfigParseError:
    ...             continue
    ...     raise ConfigParseError(path, "No compatible parser found")

    See Also
    --------
    ConfigNotFoundError : When config file doesn't exist.
    ConfigValidationError : When config values are invalid.
    DatasetFormatError : For dataset parsing failures.
    """

    def __init__(self, path: str, reason: str, line: Optional[int] = None):
        details = {"path": path, "reason": reason}
        if line is not None:
            details["line"] = line
        super().__init__(f"Failed to parse configuration: {reason}", details)
