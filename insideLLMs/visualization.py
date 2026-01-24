"""Compatibility shim for insideLLMs.analysis.visualization."""

import sys

from insideLLMs.analysis import visualization as _visualization

sys.modules[__name__] = _visualization
