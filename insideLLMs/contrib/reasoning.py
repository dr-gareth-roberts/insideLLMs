"""Reasoning-chain analysis exposed through a stable compatibility facade."""

from insideLLMs.contrib._reasoning.analysis import ReasoningAnalyzer
from insideLLMs.contrib._reasoning.api import (
    analyze_reasoning,
    assess_reasoning_quality,
    evaluate_cot,
    extract_reasoning,
    generate_cot_prompt,
)
from insideLLMs.contrib._reasoning.evaluation import CoTEvaluator
from insideLLMs.contrib._reasoning.extraction import ReasoningExtractor
from insideLLMs.contrib._reasoning.models import (
    ChainAnalysis,
    CoTEvaluation,
    ReasoningChain,
    ReasoningQuality,
    ReasoningReport,
    ReasoningStep,
    ReasoningStepType,
    ReasoningType,
)
from insideLLMs.contrib._reasoning.prompting import CoTPromptGenerator

StepType = ReasoningStepType

__all__ = [
    "ReasoningType",
    "ReasoningStepType",
    "StepType",
    "ReasoningQuality",
    "ReasoningStep",
    "ReasoningChain",
    "ChainAnalysis",
    "CoTEvaluation",
    "ReasoningReport",
    "ReasoningExtractor",
    "ReasoningAnalyzer",
    "CoTEvaluator",
    "CoTPromptGenerator",
    "extract_reasoning",
    "analyze_reasoning",
    "evaluate_cot",
    "generate_cot_prompt",
    "assess_reasoning_quality",
]

for _name in __all__:
    _value = globals()[_name]
    if callable(_value):
        _value.__module__ = __name__
del _name, _value
