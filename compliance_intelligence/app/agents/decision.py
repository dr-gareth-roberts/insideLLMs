"""Compliance Decision Deep Agent.

Makes the final approve / flag / escalate / block decision based on
the composite risk score and all upstream findings. Can request
re-analysis (cycle) if confidence is insufficient.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.config import settings
from app.models import (
    Alert,
    AlertSeverity,
    ComplianceDecision,
    DecisionVerdict,
    PipelineState,
    RiskLevel,
)

logger = logging.getLogger(__name__)


def run_decision(state: PipelineState) -> PipelineState:
    """Decision node — renders a compliance verdict."""
    logger.info("▶ Decision Agent processing TXN %s", state.transaction.transaction_id)
    state.status = "decision_processing"
    state.processing_steps.append(f"[{_now()}] Decision: evaluating risk score and findings")

    if not state.risk_score:
        state.decision = ComplianceDecision(
            verdict=DecisionVerdict.REQUEST_MORE_INFO,
            confidence=0.0,
            rationale="Risk score unavailable — cannot render decision.",
            needs_reanalysis=True,
            reanalysis_reason="Missing risk score",
        )
        state.status = "decision_complete"
        return state

    risk = state.risk_score
    if settings.simulation_mode:
        decision = _rule_based_decision(state, risk)
    else:
        decision = _llm_decision(state, risk)

    state.decision = decision

    # Log and alert
    state.processing_steps.append(
        f"[{_now()}] Decision: verdict={decision.verdict.value}, "
        f"confidence={decision.confidence:.0%}, "
        f"escalation={decision.escalation_required}, "
        f"reanalysis={decision.needs_reanalysis}"
    )

    if decision.verdict in (DecisionVerdict.BLOCK, DecisionVerdict.ESCALATE):
        state.alerts.append(Alert(
            severity=AlertSeverity.CRITICAL,
            title=f"Transaction {decision.verdict.value.replace('_', ' ').title()}",
            description=decision.rationale[:300],
        ))

    state.status = "decision_complete"
    return state


def _rule_based_decision(state: PipelineState, risk) -> ComplianceDecision:
    """Deterministic rule-based decision engine."""
    overall = risk.overall_score
    level = risk.overall_level
    # Check for hard blocks
    has_sanctions = any(k.sanctions_match for k in state.kyc_findings)
    has_embargo = state.geopolitical_finding and state.geopolitical_finding.embargo_active

    if has_sanctions or has_embargo:
        return ComplianceDecision(
            verdict=DecisionVerdict.BLOCK,
            confidence=0.97,
            rationale=_build_rationale("BLOCK", state, risk,
                "Transaction blocked due to sanctions/embargo match. "
                "Immediate escalation to BSA/AML officer required."),
            regulatory_references=_get_regulatory_refs(state),
            recommended_actions=[
                "Freeze transaction immediately",
                "File SAR (Suspicious Activity Report) within 30 days",
                "Notify BSA/AML Compliance Officer",
                "Preserve all related documentation",
                "Consider filing CTR if applicable",
            ],
            escalation_required=True,
            sla_hours=4,
        )

    # Critical risk
    if level == RiskLevel.CRITICAL or overall >= 75:
        return ComplianceDecision(
            verdict=DecisionVerdict.ESCALATE,
            confidence=0.88,
            rationale=_build_rationale("ESCALATE", state, risk,
                f"Critical risk score ({overall:.1f}/100). Multiple risk indicators triggered."),
            regulatory_references=_get_regulatory_refs(state),
            recommended_actions=[
                "Escalate to Senior Compliance Analyst",
                "Conduct enhanced due diligence (EDD)",
                "Request additional documentation from counterparties",
                "Consider filing SAR",
            ],
            escalation_required=True,
            sla_hours=8,
        )

    # High risk — flag or re-analyze
    if level == RiskLevel.HIGH or overall >= 50:
        # If first pass and confidence would be low, request reanalysis
        if state.reanalysis_count == 0 and overall >= 60:
            return ComplianceDecision(
                verdict=DecisionVerdict.FLAG_FOR_REVIEW,
                confidence=0.72,
                rationale=_build_rationale("FLAG + REANALYSIS", state, risk,
                    f"High risk score ({overall:.1f}/100) with moderate confidence. "
                    "Requesting deeper analysis pass."),
                regulatory_references=_get_regulatory_refs(state),
                recommended_actions=[
                    "Conduct additional pattern analysis",
                    "Cross-reference with historical data",
                    "Review entity documentation",
                ],
                needs_reanalysis=True,
                reanalysis_reason=f"High risk ({overall:.1f}) on first pass — deeper analysis needed",
                sla_hours=24,
            )

        return ComplianceDecision(
            verdict=DecisionVerdict.FLAG_FOR_REVIEW,
            confidence=0.82,
            rationale=_build_rationale("FLAG", state, risk,
                f"Elevated risk score ({overall:.1f}/100). Flagged for manual review."),
            regulatory_references=_get_regulatory_refs(state),
            recommended_actions=[
                "Assign to compliance analyst for manual review",
                "Request additional KYC documentation",
                "Monitor entity for 90 days",
            ],
            sla_hours=48,
        )

    # Medium risk
    if level == RiskLevel.MEDIUM or overall >= 25:
        return ComplianceDecision(
            verdict=DecisionVerdict.APPROVE,
            confidence=0.90,
            rationale=_build_rationale("APPROVE (with monitoring)", state, risk,
                f"Moderate risk score ({overall:.1f}/100). Approved with standard monitoring."),
            regulatory_references=_get_regulatory_refs(state),
            recommended_actions=["Standard transaction monitoring", "Periodic KYC refresh"],
            sla_hours=72,
        )

    # Low risk — approve
    return ComplianceDecision(
        verdict=DecisionVerdict.APPROVE,
        confidence=0.95,
        rationale=_build_rationale("APPROVE", state, risk,
            f"Low risk score ({overall:.1f}/100). No significant risk indicators."),
        regulatory_references=[],
        recommended_actions=["Standard monitoring"],
        sla_hours=168,
    )


def _build_rationale(action: str, state: PipelineState, risk, summary: str) -> str:
    """Build a detailed rationale string."""
    txn = state.transaction
    lines = [
        summary,
        f"\nTransaction: {txn.transaction_id} | {txn.transaction_type.value} | "
        f"{txn.currency.value} {txn.amount:,.2f} (~${txn.amount_usd_approx:,.2f} USD)",
        f"Corridor: {txn.source_entity.country_code} → {txn.destination_entity.country_code}",
        f"Risk breakdown: entity={risk.entity_risk_score:.0f}, txn={risk.transaction_risk_score:.0f}, "
        f"pattern={risk.pattern_risk_score:.0f}, geo={risk.geopolitical_risk_score:.0f}",
    ]
    if risk.contributing_factors:
        lines.append(f"Contributing factors: {'; '.join(risk.contributing_factors[:5])}")
    if risk.mitigating_factors:
        lines.append(f"Mitigating factors: {'; '.join(risk.mitigating_factors[:3])}")
    if state.reanalysis_count > 0:
        lines.append(f"Analysis depth: pass #{state.reanalysis_count + 1}")
    return "\n".join(lines)


def _get_regulatory_refs(state: PipelineState) -> list[str]:
    """Generate relevant regulatory references."""
    refs = ["BSA/AML Act (31 USC §5311)", "FinCEN Guidance FIN-2021-A001"]
    if any(k.sanctions_match for k in state.kyc_findings):
        refs.extend(["OFAC 50% Rule", "IEEPA (50 USC §1701)"])
    if state.geopolitical_finding and state.geopolitical_finding.embargo_active:
        refs.append("EAR Part 746 — Embargoes and Other Special Controls")
    if state.pattern_finding and state.pattern_finding.structuring_detected:
        refs.append("31 USC §5324 — Structuring Transactions")
    if state.transaction.amount_usd_approx >= 10_000:
        refs.append("31 CFR §1010.311 — CTR Filing Requirement")
    return refs


def _llm_decision(state: PipelineState, risk) -> ComplianceDecision:
    """LLM-powered decision (requires API key)."""
    import json

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key)

    context = {
        "risk_score": risk.model_dump(),
        "kyc_findings": [k.model_dump() for k in state.kyc_findings],
        "pattern_finding": state.pattern_finding.model_dump() if state.pattern_finding else None,
        "geopolitical_finding": state.geopolitical_finding.model_dump() if state.geopolitical_finding else None,
        "alerts_count": len(state.alerts),
        "reanalysis_count": state.reanalysis_count,
    }

    messages = [
        SystemMessage(content=(
            "You are a Chief Compliance Officer. Based on the risk analysis, render a verdict: "
            "approve, flag_for_review, escalate, block, or request_more_info. "
            "Return JSON matching ComplianceDecision schema. Be conservative."
        )),
        HumanMessage(content=f"Render compliance decision:\n\n{json.dumps(context, indent=2, default=str)}"),
    ]
    response = llm.invoke(messages)
    try:
        data = json.loads(response.content)
        return ComplianceDecision(**data)
    except Exception:
        logger.warning("LLM decision parsing failed, falling back to rules")
        return _rule_based_decision(state, risk)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
