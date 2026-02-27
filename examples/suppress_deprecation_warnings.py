"""
Example: Suppressing deprecation warnings during migration.

This example shows different strategies for handling deprecation warnings
while migrating a large codebase.
"""

import warnings


# Strategy 1: Suppress all deprecation warnings (not recommended)
def suppress_all_deprecations():
    """Suppress all deprecation warnings globally."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from insideLLMs.visualization import TraceVisualizer

    TraceVisualizer()
    print("✓ No warnings (all suppressed)")


# Strategy 2: Suppress specific module warnings (recommended for gradual migration)
def suppress_specific_module():
    """Suppress warnings from specific module only."""
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="insideLLMs.visualization"
    )

    from insideLLMs.visualization import TraceVisualizer

    TraceVisualizer()
    print("✓ No warnings (specific module suppressed)")


# Strategy 3: Context manager for temporary suppression
def suppress_with_context_manager():
    """Suppress warnings in specific code block."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        from insideLLMs.visualization import TraceVisualizer

        TraceVisualizer()

    print("✓ No warnings (context manager)")


# Strategy 4: Capture and log warnings (recommended for tracking migration)
def capture_and_log_warnings():
    """Capture warnings and log them for tracking."""
    import logging

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from insideLLMs.visualization import TraceVisualizer

        # Log deprecation warnings
        for warning in w:
            if issubclass(warning.category, DeprecationWarning):
                logger.warning(
                    f"Deprecation in {warning.filename}:{warning.lineno}: {warning.message}"
                )

        TraceVisualizer()

    print(f"✓ Captured {len(w)} warnings")


# Strategy 5: Compatibility layer (recommended for libraries)
def create_compatibility_layer():
    """Create compatibility layer for your library."""
    import sys

    # Check insideLLMs version
    import insideLLMs

    version = tuple(map(int, insideLLMs.__version__.split(".")[:2]))

    if version >= (2, 0):
        # Use new import path
        from insideLLMs.analysis.visualization import TraceVisualizer
    else:
        # Use old import path with suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from insideLLMs.visualization import TraceVisualizer

    print(f"✓ Compatibility layer (version {insideLLMs.__version__})")
    return TraceVisualizer


if __name__ == "__main__":
    print("Deprecation Warning Suppression Examples\n")

    print("1. Suppress all deprecations:")
    suppress_all_deprecations()

    print("\n2. Suppress specific module:")
    suppress_specific_module()

    print("\n3. Context manager:")
    suppress_with_context_manager()

    print("\n4. Capture and log:")
    capture_and_log_warnings()

    print("\n5. Compatibility layer:")
    viz_class = create_compatibility_layer()

    print("\n✓ All strategies demonstrated")
