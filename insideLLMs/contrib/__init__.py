"""insideLLMs.contrib — Disconnected research and analysis modules.

These modules provide standalone analysis capabilities (hallucination detection,
calibration, adversarial generation, etc.) but are NOT part of the core
probe-execution pipeline (CLI -> Runner -> Probes -> Models -> Artifacts -> Diff).

They remain importable via ``from insideLLMs.contrib import <module>`` or via
the top-level lazy imports (e.g., ``from insideLLMs import ReActAgent``).

If you need a module that was moved here, your imports still work — the
package-level ``__getattr__`` redirects transparently.
"""
