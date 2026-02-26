"""Transaction Pattern Analysis Deep Agent.

Analyzes transaction patterns for structuring, layering, round-tripping,
velocity anomalies, and other suspicious behaviors.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone

from app.config import settings
from app.models import (
    Alert,
    AlertSeverity,
    PipelineState,
    TransactionPatternFinding,
    TransactionType,
)

logger = logging.getLogger(__name__)

STRUCTURING_THRESHOLD_USD = 10_000.0
RAPID_MOVEMENT_HOURS = 24


def run_transaction_pattern(state: PipelineState) -> PipelineState:
    """Transaction Pattern node — detects suspicious transaction behaviors."""
    logger.info("▶ Transaction Pattern Agent analyzing TXN %s", state.transaction.transaction_id)
    state.status = "pattern_processing"
    state.processing_steps.append(f"[{_now()}] Pattern: starting behavioral analysis")

    txn = state.transaction

    if settings.simulation_mode:
        finding = _simulate_pattern(txn, state)
    else:
        finding = _llm_pattern(txn, state)

    state.pattern_finding = finding

    # Raise alerts for critical patterns
    if finding.pattern_risk.value in ("critical", "high"):
        state.alerts.append(Alert(
            severity=AlertSeverity.CRITICAL if finding.pattern_risk.value == "critical" else AlertSeverity.WARNING,
            title=f"Suspicious Pattern Detected: {finding.pattern_type}",
            description=finding.description,
        ))

    state.processing_steps.append(
        f"[{_now()}] Pattern: analysis complete — type='{finding.pattern_type}', "
        f"risk={finding.pattern_risk.value}, "
        f"structuring={finding.structuring_detected}, layering={finding.layering_detected}"
    )
    state.status = "pattern_complete"
    return state


def _simulate_pattern(txn, state: PipelineState) -> TransactionPatternFinding:
    """Realistic pattern simulation."""
    amount_usd = txn.amount_usd_approx
    evidence = []

    # Structuring detection (just below reporting threshold)
    structuring = False
    if 8_000 <= amount_usd <= 10_000:
        structuring = True
        evidence.append(f"Amount ${amount_usd:,.2f} is within structuring range (80-100% of CTR threshold)")

    # Rapid movement (crypto or wire transfers with intermediaries)
    rapid = False
    if txn.transaction_type in (TransactionType.CRYPTO, TransactionType.WIRE_TRANSFER):
        if len(txn.intermediary_banks) >= 2 or txn.transaction_type == TransactionType.CRYPTO:
            rapid = True
            evidence.append("Rapid cross-border movement pattern detected via multiple hops")

    # Round-tripping (same country, different entities, high value)
    round_trip = False
    if (txn.source_entity.country_code == txn.destination_entity.country_code
            and amount_usd > 100_000):
        if random.random() > 0.6:
            round_trip = True
            evidence.append("Potential round-tripping: high-value same-jurisdiction transfer with complex structure")

    # Layering (multiple intermediaries)
    layering = False
    if len(txn.intermediary_banks) >= 3:
        layering = True
        evidence.append(f"Layering indicators: {len(txn.intermediary_banks)} intermediary banks in chain")

    # Unusual timing
    hour = txn.timestamp.hour
    unusual_timing = hour < 6 or hour > 22
    if unusual_timing:
        evidence.append(f"Transaction submitted at unusual hour ({hour}:00 UTC)")

    # Velocity anomaly (simulated based on entity risk)
    velocity = txn.source_entity.adverse_media_hits > 2
    if velocity:
        evidence.append("Velocity anomaly: entity has elevated recent transaction frequency")

    # Historical deviation
    deviation = min(1.0, (amount_usd / 500_000) * 0.5 + (0.3 if structuring else 0))
    if txn.source_entity.pep or txn.source_entity.sanctioned:
        deviation = min(1.0, deviation + 0.3)

    # Determine pattern type
    if layering:
        pattern_type = "Layering / Complex Structuring"
    elif round_trip:
        pattern_type = "Potential Round-Tripping"
    elif structuring:
        pattern_type = "Structuring (Smurfing)"
    elif rapid:
        pattern_type = "Rapid Fund Movement"
    elif unusual_timing:
        pattern_type = "Unusual Timing Pattern"
    else:
        pattern_type = "Normal Transaction Pattern"

    description = f"Analysis of {txn.transaction_type.value} transaction for ${amount_usd:,.2f}. "
    if evidence:
        description += "Findings: " + "; ".join(evidence) + "."
    else:
        description += "No significant anomalies detected."

    return TransactionPatternFinding(
        pattern_type=pattern_type,
        description=description,
        structuring_detected=structuring,
        rapid_movement_detected=rapid,
        round_tripping_detected=round_trip,
        layering_detected=layering,
        unusual_timing=unusual_timing,
        velocity_anomaly=velocity,
        historical_deviation_score=round(deviation, 3),
        similar_suspicious_cases=random.randint(0, 12) if any([structuring, rapid, round_trip, layering]) else 0,
        evidence=evidence,
    )


def _llm_pattern(txn, state: PipelineState) -> TransactionPatternFinding:
    """LLM-powered pattern analysis."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    import json

    llm = ChatOpenAI(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key)

    messages = [
        SystemMessage(content=(
            "You are a financial crime analyst specializing in transaction pattern detection. "
            "Analyze the transaction for: structuring, layering, round-tripping, rapid movement, "
            "unusual timing, and velocity anomalies. Return structured JSON matching TransactionPatternFinding schema."
        )),
        HumanMessage(content=f"Analyze this transaction:\n\n{txn.model_dump_json(indent=2)}"),
    ]
    response = llm.invoke(messages)
    try:
        data = json.loads(response.content)
        return TransactionPatternFinding(**data)
    except Exception:
        logger.warning("LLM pattern output parsing failed, falling back to simulation")
        return _simulate_pattern(txn, state)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
