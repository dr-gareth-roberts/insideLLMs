"""Document Intake & Enrichment Agent.

Validates incoming transaction data, enriches entities with external data,
and prepares the pipeline state for downstream analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.models import (
    Alert,
    AlertSeverity,
    PipelineState,
    RiskLevel,
)

logger = logging.getLogger(__name__)


HIGH_VALUE_THRESHOLD_USD = 50_000.0
REPORTING_THRESHOLD_USD = 10_000.0


def run_intake(state: PipelineState) -> PipelineState:
    """Intake node — validates, enriches, and triages the transaction."""
    logger.info("▶ Intake Agent processing TXN %s", state.transaction.transaction_id)
    state.status = "intake_processing"
    state.processing_steps.append(
        f"[{_now()}] Intake: received transaction {state.transaction.transaction_id}"
    )

    txn = state.transaction
    alerts: list[Alert] = []

    # --- Enrichment: flag high-value transactions ---
    if txn.amount_usd_approx >= HIGH_VALUE_THRESHOLD_USD:
        alerts.append(
            Alert(
                severity=AlertSeverity.WARNING,
                title="High-Value Transaction",
                description=f"Approximate USD value ${txn.amount_usd_approx:,.2f} exceeds ${HIGH_VALUE_THRESHOLD_USD:,.0f} threshold.",
            )
        )
        state.processing_steps.append(
            f"[{_now()}] Intake: flagged high-value (${txn.amount_usd_approx:,.2f})"
        )

    if txn.amount_usd_approx >= REPORTING_THRESHOLD_USD:
        state.processing_steps.append(f"[{_now()}] Intake: CTR reporting threshold triggered")

    # --- Enrichment: inherent entity risk ---
    for label, entity in [("source", txn.source_entity), ("destination", txn.destination_entity)]:
        if entity.inherent_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            alerts.append(
                Alert(
                    severity=AlertSeverity.CRITICAL
                    if entity.inherent_risk == RiskLevel.CRITICAL
                    else AlertSeverity.WARNING,
                    title=f"High-Risk {label.title()} Entity",
                    description=f"{entity.name} ({entity.entity_type.value}) has inherent risk: {entity.inherent_risk.value}. "
                    f"PEP={entity.pep}, Sanctioned={entity.sanctioned}, "
                    f"Jurisdiction={entity.country_code}, Adverse media={entity.adverse_media_hits}.",
                )
            )
        state.processing_steps.append(
            f"[{_now()}] Intake: {label} entity '{entity.name}' inherent risk = {entity.inherent_risk.value}"
        )

    # --- Enrichment: crypto & cross-border flags ---
    if txn.transaction_type.value == "crypto":
        alerts.append(
            Alert(
                severity=AlertSeverity.INFO,
                title="Cryptocurrency Transaction",
                description="Transaction involves cryptocurrency — enhanced due diligence required.",
            )
        )

    if txn.source_entity.country_code != txn.destination_entity.country_code:
        state.processing_steps.append(
            f"[{_now()}] Intake: cross-border corridor {txn.source_entity.country_code} → {txn.destination_entity.country_code}"
        )

    # --- Enrichment: intermediary banks ---
    if len(txn.intermediary_banks) > 2:
        alerts.append(
            Alert(
                severity=AlertSeverity.WARNING,
                title="Multiple Intermediary Banks",
                description=f"Transaction routes through {len(txn.intermediary_banks)} intermediaries, which may indicate layering.",
            )
        )

    state.alerts.extend(alerts)
    state.processing_steps.append(f"[{_now()}] Intake: complete — {len(alerts)} alert(s) raised")
    state.status = "intake_complete"
    return state


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
