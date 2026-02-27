"""Compliance Report Generation Agent.

Assembles all findings, risk scores, decisions, and alerts into a
structured ComplianceReport — the final artifact of the pipeline.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.models import (
    ComplianceReport,
    PipelineState,
)

logger = logging.getLogger(__name__)


def run_report(state: PipelineState) -> PipelineState:
    """Report node — assembles the final compliance report."""
    logger.info("▶ Report Agent assembling report for TXN %s", state.transaction.transaction_id)
    state.status = "report_generation"
    state.processing_steps.append(f"[{_now()}] Report: assembling final compliance report")

    # Collect all findings
    findings = []
    findings.extend(state.kyc_findings)
    if state.pattern_finding:
        findings.append(state.pattern_finding)
    if state.geopolitical_finding:
        findings.append(state.geopolitical_finding)

    report = ComplianceReport(
        transaction=state.transaction,
        findings=findings,
        risk_score=state.risk_score,
        decision=state.decision,
        alerts=state.alerts,
        processing_steps=state.processing_steps.copy(),
        reanalysis_count=state.reanalysis_count,
    )

    state.report = report
    state.processing_steps.append(
        f"[{_now()}] Report: complete — {report.report_id} "
        f"({len(findings)} findings, {len(state.alerts)} alerts)"
    )
    state.status = "complete"

    logger.info("✅ Report %s generated: %s", report.report_id, report.executive_summary)
    return state


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
