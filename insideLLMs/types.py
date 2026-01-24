"""Type definitions and dataclasses for insideLLMs.

This module provides strongly-typed data structures for representing
experiment results, model responses, and probe outputs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Optional, TypeVar, Union

# Type variable for generic result types
T = TypeVar("T")


class ProbeCategory(Enum):
    """Categories of probes for organizing and filtering."""

    LOGIC = "logic"
    FACTUALITY = "factuality"
    BIAS = "bias"
    ATTACK = "attack"
    SAFETY = "safety"
    REASONING = "reasoning"
    KNOWLEDGE = "knowledge"
    CUSTOM = "custom"


class ResultStatus(Enum):
    """Status of a probe result."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    SKIPPED = "skipped"


@dataclass
class ModelInfo:
    """Information about a language model."""

    name: str
    provider: str
    model_id: str
    max_tokens: Optional[int] = None
    supports_streaming: bool = False
    supports_chat: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Token usage statistics for a model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ModelResponse:
    """A response from a language model."""

    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None
    latency_ms: Optional[float] = None
    raw_response: Optional[Any] = None


@dataclass
class ProbeResult(Generic[T]):
    """Result from running a single probe item.

    Generic type T represents the specific output type for each probe.
    """

    input: Any
    output: Optional[T] = None
    status: ResultStatus = ResultStatus.SUCCESS
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FactualityResult:
    """Result from a factuality probe."""

    question: str
    reference_answer: str
    model_answer: str
    extracted_answer: str
    category: str = "general"
    is_correct: Optional[bool] = None
    confidence: Optional[float] = None
    similarity_score: Optional[float] = None


@dataclass
class BiasResult:
    """Result from a bias probe."""

    prompt_a: str
    prompt_b: str
    response_a: str
    response_b: str
    bias_dimension: str = "unknown"
    sentiment_diff: Optional[float] = None
    length_diff: Optional[int] = None
    semantic_similarity: Optional[float] = None


@dataclass
class LogicResult:
    """Result from a logic probe."""

    problem: str
    model_answer: str
    expected_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    reasoning_steps: Optional[list[str]] = None
    problem_type: str = "general"


@dataclass
class ProbeAttackResult:
    """Result from an attack/adversarial probe."""

    attack_prompt: str
    model_response: str
    attack_type: str
    attack_succeeded: Optional[bool] = None
    severity: Optional[str] = None
    indicators: list[str] = field(default_factory=list)


# Backward compatibility alias
AttackResult = ProbeAttackResult


@dataclass
class ProbeScore:
    """Scoring metrics for a probe run."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mean_latency_ms: Optional[float] = None
    total_tokens: Optional[int] = None
    error_rate: float = 0.0
    custom_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ProbeExperimentResult:
    """Complete result from running an experiment."""

    experiment_id: str
    model_info: ModelInfo
    probe_name: str
    probe_category: ProbeCategory
    results: list[ProbeResult]
    score: Optional[ProbeScore] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        """Count of successful results."""
        return sum(1 for r in self.results if r.status == ResultStatus.SUCCESS)

    @property
    def error_count(self) -> int:
        """Count of error results."""
        return sum(1 for r in self.results if r.status == ResultStatus.ERROR)

    @property
    def total_count(self) -> int:
        """Total number of results."""
        return len(self.results)

    @property
    def success_rate(self) -> float:
        """Percentage of successful results."""
        if not self.results:
            return 0.0
        return self.success_count / self.total_count

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of the experiment in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# Backward compatibility alias
ExperimentResult = ProbeExperimentResult


@dataclass
class BenchmarkComparison:
    """Comparison of multiple models or probes."""

    name: str
    experiments: list[ProbeExperimentResult]
    rankings: dict[str, list[str]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


# Type aliases for common patterns
ProbeInput = Union[str, dict[str, Any], list[Any]]
ProbeOutput = Union[str, dict[str, Any], list[dict[str, Any]]]
ConfigDict = dict[str, Any]
