"""Public prompt-injection API facade.

This module preserves the long-standing ``insideLLMs.injection`` import path
while delegating implementation to ``insideLLMs.contrib.security.injection_engine``.
Use this facade for stable, user-facing imports.

The re-exports are listed explicitly rather than star-imported so that the
public surface of this module is declared here, not inherited from whatever
the engine happens to export.
"""

from insideLLMs.contrib.security.injection_engine import (
    DefenseReport,
    DefenseStrategy,
    DefensivePromptBuilder,
    DetectionResult,
    InjectionDetector,
    InjectionPattern,
    InjectionTester,
    InjectionType,
    InputSanitizer,
    RiskLevel,
    SanitizationResult,
    assess_injection_resistance,
    build_defensive_prompt,
    detect_injection,
    is_safe_input,
    sanitize_input,
)

__all__ = [
    "DefenseReport",
    "DefenseStrategy",
    "DefensivePromptBuilder",
    "DetectionResult",
    "InjectionDetector",
    "InjectionPattern",
    "InjectionTester",
    "InjectionType",
    "InputSanitizer",
    "RiskLevel",
    "SanitizationResult",
    "assess_injection_resistance",
    "build_defensive_prompt",
    "detect_injection",
    "is_safe_input",
    "sanitize_input",
]
