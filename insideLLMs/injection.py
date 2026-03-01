"""Public prompt-injection API facade.

This module preserves the long-standing ``insideLLMs.injection`` import path
while delegating implementation to ``insideLLMs.contrib.security.injection_engine``.
Use this facade for stable, user-facing imports.
"""

from insideLLMs.contrib.security.injection_engine import *  # noqa: F401,F403
