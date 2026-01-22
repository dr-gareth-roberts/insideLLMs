"""Probes for testing various aspects of LLM behavior.

This module provides probes for evaluating:
- Logic and reasoning capabilities
- Factual accuracy and knowledge
- Bias and fairness
- Security and adversarial robustness
- Code generation and debugging
- Instruction following and compliance
"""

from insideLLMs.probes.agent_probe import AgentProbe, AgentProbeResult, ToolDefinition
from insideLLMs.probes.attack import AttackProbe, JailbreakProbe, PromptInjectionProbe
from insideLLMs.probes.base import ComparativeProbe, Probe, ScoredProbe
from insideLLMs.probes.bias import BiasProbe
from insideLLMs.probes.code import CodeDebugProbe, CodeExplanationProbe, CodeGenerationProbe
from insideLLMs.probes.factuality import FactualityProbe
from insideLLMs.probes.instruction import (
    ConstraintComplianceProbe,
    InstructionFollowingProbe,
    MultiStepTaskProbe,
)
from insideLLMs.probes.logic import LogicProbe
from insideLLMs.types import ProbeCategory


class CustomProbe(Probe[str]):
    """Template for creating custom probes.

    Extend this class to create your own probe implementations.

    Example:
        >>> class MyProbe(CustomProbe):
        ...     def run(self, model, data, **kwargs):
        ...         prompt = f"Analyze: {data}"
        ...         return model.generate(prompt, **kwargs)
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(self, name: str = "CustomProbe"):
        super().__init__(name=name, category=ProbeCategory.CUSTOM)

    def run(self, model, data, **kwargs):
        """Implement custom probe logic here.

        Args:
            model: The model to test.
            data: The input data for the probe.
            **kwargs: Additional arguments.

        Returns:
            The probe output.
        """
        raise NotImplementedError("CustomProbe.run() must be implemented by subclass")


__all__ = [
    # Base classes
    "Probe",
    "ScoredProbe",
    "ComparativeProbe",
    # Agent probe
    "AgentProbe",
    "AgentProbeResult",
    "ToolDefinition",
    # Built-in probes
    "LogicProbe",
    "BiasProbe",
    "AttackProbe",
    "FactualityProbe",
    # Attack probe variants
    "PromptInjectionProbe",
    "JailbreakProbe",
    # Code probes
    "CodeGenerationProbe",
    "CodeExplanationProbe",
    "CodeDebugProbe",
    # Instruction probes
    "InstructionFollowingProbe",
    "MultiStepTaskProbe",
    "ConstraintComplianceProbe",
    # Template
    "CustomProbe",
    # Types
    "ProbeCategory",
]
