"""Compatibility re-export of the dispatch-limit contracts.

The implementation lives in :mod:`insideLLMs.dispatch_limits`, which both the
inference package and matched-compute analysis can import.
"""

from insideLLMs.dispatch_limits import (
    ComputeMismatchError,
    DispatchCollector,
    DispatchEvidence,
    OutputLimitBinding,
    current_collector,
    dispatch_scope,
    validate_output_limit,
    validate_output_limit_preflight,
)

__all__ = [
    "ComputeMismatchError",
    "DispatchCollector",
    "DispatchEvidence",
    "OutputLimitBinding",
    "current_collector",
    "dispatch_scope",
    "validate_output_limit",
    "validate_output_limit_preflight",
]
