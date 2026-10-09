"""Provider-package view of the builtin adapter catalogue.

Declarations live in :mod:`insideLLMs.provider_catalogue` so core code can load
them without importing the providers layer. Importing from this module remains
supported.
"""

from insideLLMs.provider_catalogue import (
    PROVIDER_CATALOGUE,
    DependencySpec,
    OperationSupport,
    ProviderCapabilities,
    ProviderSpec,
)

__all__ = [
    "DependencySpec",
    "OperationSupport",
    "ProviderCapabilities",
    "ProviderSpec",
    "PROVIDER_CATALOGUE",
]
