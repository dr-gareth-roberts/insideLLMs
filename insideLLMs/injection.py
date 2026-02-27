"""Public prompt-injection API facade.

This module preserves the long-standing ``insideLLMs.injection`` import path
while delegating implementation to ``insideLLMs.security.injection_engine``.
Use this facade for stable, user-facing imports.
"""

from insideLLMs.security.injection_engine import *  # noqa: F401,F403
