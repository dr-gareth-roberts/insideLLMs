"""Shared, provider-neutral records for inference-time strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class PromptParts:
    """Canonical prompt sections with stable content kept first."""

    stable: tuple[str, ...] = ()
    dynamic: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def compose(self) -> str:
        return "\n\n".join((*self.stable, *self.dynamic, *self.evidence))


@dataclass(frozen=True)
class InferenceRequest:
    """Input shared by every inference strategy."""

    prompt: str
    parts: PromptParts | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """One generated answer and its derivation metadata."""

    id: str
    output: str
    normalized_answer: str | None = None
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verification:
    """Objective or model-based assessment of a candidate."""

    verifier_id: str
    score: float
    passed: bool
    evidence: Any = None


@dataclass(frozen=True)
class TraceEvent:
    """A node in an inference trace DAG."""

    id: str
    kind: str
    parent_ids: tuple[str, ...] = ()
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Budget:
    """Hard limits shared by online and offline strategies."""

    max_calls: int | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None
    max_evaluations: int | None = None
    max_generations: int | None = None
    max_nodes: int | None = None


@dataclass(frozen=True)
class Spend:
    """Actual resources consumed by a strategy."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    cost: float = 0.0
    evaluations: int = 0


class StopReason(str, Enum):
    COMPLETED = "completed"
    VERIFIED = "verified"
    AGREEMENT = "agreement"
    BUDGET = "budget"
    EXHAUSTED = "exhausted"
    NO_IMPROVEMENT = "no_improvement"


@dataclass(frozen=True)
class InferenceResult:
    """Common result envelope returned by inference strategies."""

    answer: Any
    confidence: float = 0.0
    candidates: tuple[Candidate, ...] = ()
    verifications: tuple[Verification, ...] = ()
    trace: tuple[TraceEvent, ...] = ()
    spend: Spend = field(default_factory=Spend)
    stop_reason: StopReason = StopReason.COMPLETED
    provenance: dict[str, Any] = field(default_factory=dict)
