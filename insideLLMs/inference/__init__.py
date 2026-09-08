"""Composable, async-first inference-time harness strategies."""

from ._limits import OutputLimitBinding
from .adapters import ModelProposer
from .best_of_n import NoVerifiedCandidateError, VerifierSpec, select_best
from .client import InferenceClient
from .dag import DagBudgetExceeded, PlanNode, execute_dag
from .escalation import EscalationStep, escalate_adaptively
from .evolution import (
    EvolutionBudgetExceeded,
    EvolutionCandidate,
    EvolutionConfig,
    EvolutionResult,
    Fitness,
    GenerationRecord,
    evolve_artifacts,
)
from .prefix_cache import (
    CachedPrompt,
    PrefixCacheTelemetry,
    compose_cached_prompt,
    record_cache_usage,
)
from .repair import repair_with_evidence
from .retrieval import AssembledEvidence, rerank_and_assemble
from .schemas import (
    Budget,
    Candidate,
    InferenceRequest,
    InferenceResult,
    PromptParts,
    Spend,
    StopReason,
    TraceEvent,
    Verification,
)
from .search import SearchBudgetExceeded, beam_search
from .self_consistency import sample_consistent
from .structured import generate_validated
from .sync import run_sync
from .tools import (
    Observation,
    ToolAction,
    ToolLimits,
    ToolOutputTooLarge,
    ToolPolicyError,
    execute_tool,
)

__all__ = [
    "Budget",
    "AssembledEvidence",
    "CachedPrompt",
    "Candidate",
    "DagBudgetExceeded",
    "EscalationStep",
    "EvolutionCandidate",
    "EvolutionBudgetExceeded",
    "EvolutionConfig",
    "EvolutionResult",
    "Fitness",
    "GenerationRecord",
    "InferenceRequest",
    "InferenceResult",
    "InferenceClient",
    "ModelProposer",
    "NoVerifiedCandidateError",
    "Observation",
    "OutputLimitBinding",
    "PlanNode",
    "PrefixCacheTelemetry",
    "PromptParts",
    "Spend",
    "SearchBudgetExceeded",
    "StopReason",
    "ToolAction",
    "ToolLimits",
    "ToolOutputTooLarge",
    "ToolPolicyError",
    "TraceEvent",
    "VerifierSpec",
    "Verification",
    "beam_search",
    "compose_cached_prompt",
    "escalate_adaptively",
    "evolve_artifacts",
    "execute_dag",
    "execute_tool",
    "generate_validated",
    "repair_with_evidence",
    "record_cache_usage",
    "rerank_and_assemble",
    "run_sync",
    "sample_consistent",
    "select_best",
]
