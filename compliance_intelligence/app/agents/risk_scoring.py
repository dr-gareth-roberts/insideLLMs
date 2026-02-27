"""Composite Risk Scoring Deep Agent.

Aggregates findings from KYC, Transaction Pattern, and Geopolitical agents
into a single composite risk score using a weighted scoring model.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.models import (
    PipelineState,
    RiskLevel,
    RiskScore,
)

logger = logging.getLogger(__name__)

# Scoring weights (must sum to 1.0)
WEIGHTS = {
    "entity": 0.30,
    "transaction": 0.25,
    "pattern": 0.25,
    "geopolitical": 0.20,
}

RISK_LEVEL_TO_SCORE = {
    RiskLevel.LOW: 15.0,
    RiskLevel.MEDIUM: 45.0,
    RiskLevel.HIGH: 75.0,
    RiskLevel.CRITICAL: 95.0,
}


def run_risk_scoring(state: PipelineState) -> PipelineState:
    """Risk Scoring node — computes composite risk from all findings."""
    logger.info("▶ Risk Scoring Agent processing TXN %s", state.transaction.transaction_id)
    state.status = "risk_scoring"
    state.processing_steps.append(f"[{_now()}] Risk: computing composite score")

    txn = state.transaction

    # --- Entity risk (average of source + destination) ---
    entity_scores = []
    for entity in [txn.source_entity, txn.destination_entity]:
        entity_scores.append(RISK_LEVEL_TO_SCORE[entity.inherent_risk])
    for kyc in state.kyc_findings:
        entity_scores.append(RISK_LEVEL_TO_SCORE[kyc.kyc_risk])
    entity_risk = sum(entity_scores) / max(len(entity_scores), 1)

    # --- Transaction risk (amount-based + type-based) ---
    amount_usd = txn.amount_usd_approx
    if amount_usd > 500_000:
        txn_base = 80.0
    elif amount_usd > 100_000:
        txn_base = 60.0
    elif amount_usd > 50_000:
        txn_base = 40.0
    elif amount_usd > 10_000:
        txn_base = 25.0
    else:
        txn_base = 10.0

    # Crypto and cash get a premium
    type_premium = 0.0
    if txn.transaction_type.value in ("crypto", "cash_deposit", "cash_withdrawal"):
        type_premium = 15.0
    txn_risk = min(100.0, txn_base + type_premium)

    # --- Pattern risk ---
    pattern_risk = 15.0
    if state.pattern_finding:
        pattern_risk = RISK_LEVEL_TO_SCORE[state.pattern_finding.pattern_risk]
        # Boost for specific indicators
        if state.pattern_finding.structuring_detected:
            pattern_risk = min(100.0, pattern_risk + 10)
        if state.pattern_finding.layering_detected:
            pattern_risk = min(100.0, pattern_risk + 15)
        if state.pattern_finding.round_tripping_detected:
            pattern_risk = min(100.0, pattern_risk + 10)

    # --- Geopolitical risk ---
    geo_risk = 15.0
    if state.geopolitical_finding:
        geo_risk = RISK_LEVEL_TO_SCORE[state.geopolitical_finding.geo_risk]
        if state.geopolitical_finding.embargo_active:
            geo_risk = 100.0

    # --- Composite score ---
    overall = (
        entity_risk * WEIGHTS["entity"]
        + txn_risk * WEIGHTS["transaction"]
        + pattern_risk * WEIGHTS["pattern"]
        + geo_risk * WEIGHTS["geopolitical"]
    )
    overall = round(min(100.0, overall), 1)

    # --- Contributing & mitigating factors ---
    contributing = []
    mitigating = []

    if entity_risk >= 60:
        contributing.append(f"High entity risk score ({entity_risk:.0f}/100)")
    if txn_risk >= 50:
        contributing.append(
            f"Elevated transaction risk ({txn_risk:.0f}/100) — amount ${amount_usd:,.2f}"
        )
    if pattern_risk >= 60:
        contributing.append(f"Suspicious transaction patterns detected ({pattern_risk:.0f}/100)")
    if geo_risk >= 60:
        contributing.append(f"Geopolitical risk factors ({geo_risk:.0f}/100)")

    for kyc in state.kyc_findings:
        if kyc.sanctions_match:
            contributing.append(f"Sanctions match: entity {kyc.entity_id}")
        if kyc.pep_match:
            contributing.append(f"PEP match: entity {kyc.entity_id}")

    if entity_risk < 30:
        mitigating.append("Both entities have low inherent risk profiles")
    if all(kyc.identity_verified for kyc in state.kyc_findings):
        mitigating.append("All entities passed identity verification")
    if txn.transaction_type.value in ("internal_transfer", "ach"):
        mitigating.append(f"Lower-risk transaction type: {txn.transaction_type.value}")

    # --- Overall level ---
    if overall >= 75:
        level = RiskLevel.CRITICAL
    elif overall >= 50:
        level = RiskLevel.HIGH
    elif overall >= 25:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    state.risk_score = RiskScore(
        overall_score=overall,
        overall_level=level,
        entity_risk_score=round(entity_risk, 1),
        transaction_risk_score=round(txn_risk, 1),
        pattern_risk_score=round(pattern_risk, 1),
        geopolitical_risk_score=round(geo_risk, 1),
        contributing_factors=contributing,
        mitigating_factors=mitigating,
    )

    state.processing_steps.append(
        f"[{_now()}] Risk: composite score = {overall:.1f}/100 ({level.value}) "
        f"[entity={entity_risk:.0f}, txn={txn_risk:.0f}, pattern={pattern_risk:.0f}, geo={geo_risk:.0f}]"
    )
    state.status = "risk_scoring_complete"
    return state


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
