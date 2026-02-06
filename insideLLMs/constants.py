"""Constants and default values for insideLLMs.

This module defines constants used throughout the insideLLMs codebase,
providing a single source of truth for configuration defaults, model
identifiers, and other magic values.

Using constants from this module instead of hardcoded values improves:
- Maintainability: Change values in one place
- Discoverability: Find all configurable values easily
- Type safety: IDE support for constant names
- Documentation: Values are documented in context

Example - Using Constants:
    >>> from insideLLMs.constants import DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES
    >>> from insideLLMs.constants import OpenAIModels, AnthropicModels
    >>>
    >>> # Use default timeout
    >>> model = OpenAIModel(timeout=DEFAULT_TIMEOUT)
    >>>
    >>> # Use model name constant
    >>> model = OpenAIModel(model_name=OpenAIModels.GPT_4)
"""

from typing import Final

# =============================================================================
# Timeouts and Retries
# =============================================================================

#: Default request timeout in seconds for API calls
DEFAULT_TIMEOUT: Final[float] = 60.0

#: Default maximum number of retry attempts for transient failures
DEFAULT_MAX_RETRIES: Final[int] = 2

#: Maximum timeout allowed (10 minutes)
MAX_TIMEOUT: Final[float] = 600.0

#: Default retry delay base in seconds for exponential backoff
DEFAULT_RETRY_DELAY: Final[float] = 1.0

#: Maximum retry delay in seconds
MAX_RETRY_DELAY: Final[float] = 60.0


# =============================================================================
# Token and Context Limits
# =============================================================================

#: Default max tokens for generation if not specified
DEFAULT_MAX_TOKENS: Final[int] = 1024

#: Default temperature for text generation (0.0 = deterministic, 1.0 = creative)
DEFAULT_TEMPERATURE: Final[float] = 0.7


# =============================================================================
# OpenAI Model Identifiers
# =============================================================================


class OpenAIModels:
    """Constants for OpenAI model identifiers.

    Example:
        >>> from insideLLMs.constants import OpenAIModels
        >>> model = OpenAIModel(model_name=OpenAIModels.GPT_4_TURBO)
    """

    #: GPT-4 Turbo - Latest and most capable GPT-4 variant
    GPT_4_TURBO: Final[str] = "gpt-4-turbo"

    #: GPT-4 - Original GPT-4 model
    GPT_4: Final[str] = "gpt-4"

    #: GPT-4o - Optimized GPT-4 variant
    GPT_4O: Final[str] = "gpt-4o"

    #: GPT-4o Mini - Smaller, faster GPT-4o variant
    GPT_4O_MINI: Final[str] = "gpt-4o-mini"

    #: GPT-3.5 Turbo - Fast and cost-effective
    GPT_35_TURBO: Final[str] = "gpt-3.5-turbo"

    #: Default model to use if none specified
    DEFAULT: Final[str] = GPT_35_TURBO


# =============================================================================
# Anthropic Model Identifiers
# =============================================================================


class AnthropicModels:
    """Constants for Anthropic Claude model identifiers.

    Example:
        >>> from insideLLMs.constants import AnthropicModels
        >>> model = AnthropicModel(model_name=AnthropicModels.CLAUDE_3_OPUS)
    """

    #: Claude 3 Opus - Most capable Claude model
    CLAUDE_3_OPUS: Final[str] = "claude-3-opus-20240229"

    #: Claude 3.5 Sonnet - Improved balanced model
    CLAUDE_35_SONNET: Final[str] = "claude-3-5-sonnet-20240620"

    #: Claude 3 Sonnet - Balanced performance and cost
    CLAUDE_3_SONNET: Final[str] = "claude-3-sonnet-20240229"

    #: Claude 3 Haiku - Fastest and most cost-effective
    CLAUDE_3_HAIKU: Final[str] = "claude-3-haiku-20240307"

    #: Default model to use if none specified
    DEFAULT: Final[str] = CLAUDE_3_OPUS


# =============================================================================
# Google Gemini Model Identifiers
# =============================================================================


class GeminiModels:
    """Constants for Google Gemini model identifiers.

    Example:
        >>> from insideLLMs.constants import GeminiModels
        >>> model = GeminiModel(model_name=GeminiModels.GEMINI_PRO)
    """

    #: Gemini Pro - General-purpose model
    GEMINI_PRO: Final[str] = "gemini-pro"

    #: Gemini 1.5 Pro - Enhanced capabilities
    GEMINI_15_PRO: Final[str] = "gemini-1.5-pro"

    #: Gemini 1.5 Flash - Fast, cost-effective
    GEMINI_15_FLASH: Final[str] = "gemini-1.5-flash"

    #: Default model to use if none specified
    DEFAULT: Final[str] = GEMINI_PRO


# =============================================================================
# Cohere Model Identifiers
# =============================================================================


class CohereModels:
    """Constants for Cohere model identifiers.

    Example:
        >>> from insideLLMs.constants import CohereModels
        >>> model = CohereModel(model_name=CohereModels.COMMAND)
    """

    #: Command - Most capable model
    COMMAND: Final[str] = "command"

    #: Command Light - Faster, lighter model
    COMMAND_LIGHT: Final[str] = "command-light"

    #: Command R - Latest command model
    COMMAND_R: Final[str] = "command-r"

    #: Command R Plus - Enhanced command model
    COMMAND_R_PLUS: Final[str] = "command-r-plus"

    #: Default model to use if none specified
    DEFAULT: Final[str] = COMMAND


# =============================================================================
# Environment Variable Names
# =============================================================================


class EnvVars:
    """Environment variable names used by insideLLMs.

    Example:
        >>> import os
        >>> from insideLLMs.constants import EnvVars
        >>> api_key = os.getenv(EnvVars.OPENAI_API_KEY)
    """

    #: OpenAI API key
    OPENAI_API_KEY: Final[str] = "OPENAI_API_KEY"

    #: Anthropic API key
    ANTHROPIC_API_KEY: Final[str] = "ANTHROPIC_API_KEY"

    #: Google API key
    GOOGLE_API_KEY: Final[str] = "GOOGLE_API_KEY"

    #: Cohere API key
    COHERE_API_KEY: Final[str] = "COHERE_API_KEY"

    #: HuggingFace token
    HF_TOKEN: Final[str] = "HF_TOKEN"

    #: Directory for insideLLMs run artifacts
    INSIDELLMS_RUN_ROOT: Final[str] = "INSIDELLMS_RUN_ROOT"

    #: Disable plugin loading
    INSIDELLMS_DISABLE_PLUGINS: Final[str] = "INSIDELLMS_DISABLE_PLUGINS"


# =============================================================================
# Cache Configuration
# =============================================================================


class CacheDefaults:
    """Default values for cache configuration.

    Example:
        >>> from insideLLMs.constants import CacheDefaults
        >>> cache = InMemoryCache(
        ...     max_size=CacheDefaults.MAX_SIZE,
        ...     default_ttl=CacheDefaults.TTL_SECONDS,
        ... )
    """

    #: Default maximum cache entries
    MAX_SIZE: Final[int] = 1000

    #: Default cache TTL in seconds (1 hour)
    TTL_SECONDS: Final[int] = 3600

    #: Default disk cache size in MB
    DISK_MAX_SIZE_MB: Final[int] = 100

    #: Default hash algorithm for cache keys
    HASH_ALGORITHM: Final[str] = "sha256"


# =============================================================================
# Rate Limiting
# =============================================================================


class RateLimitDefaults:
    """Default values for rate limiting configuration.

    Example:
        >>> from insideLLMs.constants import RateLimitDefaults
        >>> limiter = TokenBucketLimiter(
        ...     requests_per_minute=RateLimitDefaults.REQUESTS_PER_MINUTE,
        ... )
    """

    #: Default requests per minute limit
    REQUESTS_PER_MINUTE: Final[int] = 60

    #: Default tokens per minute limit
    TOKENS_PER_MINUTE: Final[int] = 90000

    #: Default burst size for token bucket
    BURST_SIZE: Final[int] = 10


# =============================================================================
# Logging
# =============================================================================

#: Logger name prefix for all insideLLMs loggers
LOGGER_PREFIX: Final[str] = "insideLLMs"


# =============================================================================
# File Extensions and Formats
# =============================================================================


class FileExtensions:
    """File extensions used by insideLLMs.

    Example:
        >>> from insideLLMs.constants import FileExtensions
        >>> output_path = f"results{FileExtensions.JSONL}"
    """

    #: JSON Lines format for streaming records
    JSONL: Final[str] = ".jsonl"

    #: Standard JSON format
    JSON: Final[str] = ".json"

    #: YAML configuration files
    YAML: Final[str] = ".yaml"

    #: Alternative YAML extension
    YML: Final[str] = ".yml"

    #: HTML reports
    HTML: Final[str] = ".html"

    #: CSV exports
    CSV: Final[str] = ".csv"


# =============================================================================
# Canonical Artifact Names
# =============================================================================


class ArtifactNames:
    """Names for canonical run artifacts.

    Example:
        >>> from insideLLMs.constants import ArtifactNames
        >>> records_path = run_dir / ArtifactNames.RECORDS
    """

    #: Canonical output records
    RECORDS: Final[str] = "records.jsonl"

    #: Run manifest with metadata
    MANIFEST: Final[str] = "manifest.json"

    #: Resolved configuration snapshot
    CONFIG_RESOLVED: Final[str] = "config.resolved.yaml"

    #: Summary statistics
    SUMMARY: Final[str] = "summary.json"

    #: HTML report
    REPORT: Final[str] = "report.html"

    #: Diff report for CI
    DIFF: Final[str] = "diff.json"

    #: Run marker file
    RUN_MARKER: Final[str] = ".insidellms_run"


__all__ = [
    # Timeouts and retries
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "MAX_TIMEOUT",
    "DEFAULT_RETRY_DELAY",
    "MAX_RETRY_DELAY",
    # Token limits
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    # Model classes
    "OpenAIModels",
    "AnthropicModels",
    "GeminiModels",
    "CohereModels",
    # Environment variables
    "EnvVars",
    # Configuration classes
    "CacheDefaults",
    "RateLimitDefaults",
    # Logging
    "LOGGER_PREFIX",
    # Files
    "FileExtensions",
    "ArtifactNames",
]
