"""LangGraph typed state definition.

Uses TypedDict for LangGraph compatibility while wrapping our Pydantic
PipelineState for validation at node boundaries.
"""

from __future__ import annotations

from typing import TypedDict

from app.models import (
    Alert,
    ComplianceDecision,
    ComplianceReport,
    GeopoliticalFinding,
    KYCFinding,
    RiskScore,
    Transaction,
    TransactionPatternFinding,
)


class GraphState(TypedDict, total=False):
    """LangGraph state — mirrors PipelineState but as TypedDict for graph compatibility."""

    transaction: Transaction
    kyc_findings: list[KYCFinding]
    pattern_finding: TransactionPatternFinding | None
    geopolitical_finding: GeopoliticalFinding | None
    risk_score: RiskScore | None
    decision: ComplianceDecision | None
    alerts: list[Alert]
    report: ComplianceReport | None
    processing_steps: list[str]
    reanalysis_count: int
    max_reanalysis: int
    error: str | None
    status: str
