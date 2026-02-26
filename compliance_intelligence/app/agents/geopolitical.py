"""Geopolitical Risk Deep Agent.

Evaluates the geopolitical risk of a transaction corridor — sanctions programs,
embargoes, trade restrictions, and regulatory environment.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.config import settings
from app.models import (
    Alert,
    AlertSeverity,
    GeopoliticalFinding,
    PipelineState,
)

logger = logging.getLogger(__name__)

# Sanctions and embargo data (simplified for demonstration)
SANCTIONS_DB: dict[str, list[str]] = {
    "IR": ["OFAC Iran Sanctions", "EU Iran Embargo", "UN Resolution 2231"],
    "KP": ["OFAC DPRK Sanctions", "EU DPRK Embargo", "UN Resolution 2397"],
    "SY": ["OFAC Syria Sanctions", "EU Syria Embargo"],
    "CU": ["OFAC Cuba Embargo"],
    "RU": ["OFAC Russia/Ukraine Sanctions", "EU Russia Sanctions Package"],
    "BY": ["EU Belarus Sanctions", "OFAC Belarus Sanctions"],
    "VE": ["OFAC Venezuela Sanctions"],
    "MM": ["OFAC Burma Sanctions", "EU Myanmar Sanctions"],
}

EMBARGOED_PAIRS: set[tuple[str, str]] = {
    ("US", "IR"), ("US", "KP"), ("US", "SY"), ("US", "CU"),
    ("GB", "IR"), ("GB", "KP"), ("GB", "SY"),
}

TRADE_RESTRICTIONS_DB: dict[str, list[str]] = {
    "CN": ["Export control restrictions (Entity List)", "Technology transfer limitations"],
    "RU": ["SWIFT disconnection for select banks", "Energy sector restrictions", "Luxury goods ban"],
    "IR": ["Total trade embargo", "Financial sector sanctions"],
    "KP": ["Comprehensive sanctions regime", "Maritime interdiction"],
}


def run_geopolitical(state: PipelineState) -> PipelineState:
    """Geopolitical Risk node — assesses corridor-level risk."""
    logger.info("▶ Geopolitical Agent analyzing TXN %s", state.transaction.transaction_id)
    state.status = "geopolitical_processing"

    txn = state.transaction
    src_cc = txn.source_entity.country_code
    dst_cc = txn.destination_entity.country_code
    corridor = f"{src_cc} → {dst_cc}"

    state.processing_steps.append(f"[{_now()}] Geo: analyzing corridor {corridor}")

    finding = _analyze_corridor(src_cc, dst_cc, corridor)
    state.geopolitical_finding = finding

    if finding.geo_risk.value in ("critical", "high"):
        state.alerts.append(Alert(
            severity=AlertSeverity.CRITICAL if finding.embargo_active else AlertSeverity.WARNING,
            title=f"Geopolitical Risk: {corridor}",
            description=finding.risk_narrative,
        ))

    state.processing_steps.append(
        f"[{_now()}] Geo: corridor {corridor} → risk={finding.geo_risk.value}, "
        f"embargo={finding.embargo_active}, sanctions_programs={len(finding.sanctions_programs)}"
    )
    state.status = "geopolitical_complete"
    return state


def _analyze_corridor(src_cc: str, dst_cc: str, corridor: str) -> GeopoliticalFinding:
    """Analyze geopolitical risk for a transaction corridor."""
    sanctions_programs: list[str] = []
    trade_restrictions: list[str] = []
    regulatory_warnings: list[str] = []

    # Check sanctions for both countries
    for cc in (src_cc, dst_cc):
        if cc in SANCTIONS_DB:
            sanctions_programs.extend(SANCTIONS_DB[cc])
        if cc in TRADE_RESTRICTIONS_DB:
            trade_restrictions.extend(TRADE_RESTRICTIONS_DB[cc])

    # Check embargo
    embargo_active = (
        (src_cc, dst_cc) in EMBARGOED_PAIRS
        or (dst_cc, src_cc) in EMBARGOED_PAIRS
    )

    # Add corridor-specific warnings
    if src_cc != dst_cc:
        regulatory_warnings.append(f"Cross-border transaction: {corridor} — verify compliance with both jurisdictions")
    if any(cc in {"PA", "VU", "KY", "VG", "BZ"} for cc in (src_cc, dst_cc)):
        regulatory_warnings.append("Transaction involves offshore financial center — enhanced due diligence recommended")

    # Intermediary country risks (from intermediary bank locations could be added here)

    # Build narrative
    if embargo_active:
        narrative = (
            f"CRITICAL: Active embargo detected on corridor {corridor}. "
            f"Applicable sanctions programs: {', '.join(sanctions_programs)}. "
            f"This transaction MUST be blocked pending OFAC/sanctions compliance review."
        )
    elif sanctions_programs:
        narrative = (
            f"Elevated geopolitical risk on corridor {corridor}. "
            f"Active sanctions programs: {', '.join(sanctions_programs)}. "
            f"Enhanced due diligence and sanctions screening required."
        )
    elif trade_restrictions:
        narrative = (
            f"Moderate geopolitical risk on corridor {corridor}. "
            f"Trade restrictions apply: {', '.join(trade_restrictions)}. "
            f"Verify transaction does not violate sector-specific restrictions."
        )
    else:
        narrative = f"Low geopolitical risk on corridor {corridor}. Standard compliance procedures apply."

    return GeopoliticalFinding(
        corridor=corridor,
        sanctions_programs=sanctions_programs,
        embargo_active=embargo_active,
        trade_restrictions=trade_restrictions,
        regulatory_warnings=regulatory_warnings,
        risk_narrative=narrative,
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
