"""LangGraph workflow — the orchestration engine.

Demonstrates:
- StateGraph with typed state
- Conditional edges (decision routing)
- Cycles (re-analysis loop when confidence is low)
- Parallel-like fan-out (KYC + Pattern + Geopolitical run in sequence but
  are logically independent analysis branches)
- Error handling edges
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from langgraph.graph import END, StateGraph

from app.agents.decision import run_decision
from app.agents.geopolitical import run_geopolitical
from app.agents.intake import run_intake
from app.agents.kyc import run_kyc
from app.agents.report import run_report
from app.agents.risk_scoring import run_risk_scoring
from app.agents.transaction_pattern import run_transaction_pattern
from app.graph.state import GraphState
from app.models import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node wrappers (PipelineState ↔ GraphState bridge)
# ---------------------------------------------------------------------------


def _to_pipeline(state: GraphState) -> PipelineState:
    """Convert LangGraph dict state to our validated Pydantic state."""
    return PipelineState(**state)


def _from_pipeline(ps: PipelineState) -> GraphState:
    """Convert Pydantic state back to LangGraph dict state."""
    return ps.model_dump()


def intake_node(state: GraphState) -> GraphState:
    ps = _to_pipeline(state)
    ps = run_intake(ps)
    return _from_pipeline(ps)


def kyc_node(state: GraphState) -> GraphState:
    ps = _to_pipeline(state)
    ps = run_kyc(ps)
    return _from_pipeline(ps)


def transaction_pattern_node(state: GraphState) -> GraphState:
    ps = _to_pipeline(state)
    ps = run_transaction_pattern(ps)
    return _from_pipeline(ps)


def geopolitical_node(state: GraphState) -> GraphState:
    ps = _to_pipeline(state)
    ps = run_geopolitical(ps)
    return _from_pipeline(ps)


def risk_scoring_node(state: GraphState) -> GraphState:
    ps = _to_pipeline(state)
    ps = run_risk_scoring(ps)
    return _from_pipeline(ps)


def decision_node(state: GraphState) -> GraphState:
    ps = _to_pipeline(state)
    ps = run_decision(ps)
    return _from_pipeline(ps)


def report_node(state: GraphState) -> GraphState:
    ps = _to_pipeline(state)
    ps = run_report(ps)
    return _from_pipeline(ps)


def reanalysis_node(state: GraphState) -> GraphState:
    """Increment reanalysis counter and reset partial findings for deeper pass."""
    ps = _to_pipeline(state)
    ps.reanalysis_count += 1
    ps.processing_steps.append(
        f"[{_now()}] ♻ Re-analysis cycle #{ps.reanalysis_count} — "
        f"reason: {ps.decision.reanalysis_reason if ps.decision else 'unknown'}"
    )
    # Clear previous analysis for fresh deeper pass
    ps.pattern_finding = None
    ps.geopolitical_finding = None
    ps.risk_score = None
    ps.decision = None
    ps.status = "reanalysis"
    logger.info(
        "♻ Re-analysis cycle #%d for TXN %s", ps.reanalysis_count, ps.transaction.transaction_id
    )
    return _from_pipeline(ps)


# ---------------------------------------------------------------------------
# Conditional edge: after decision → report OR re-analysis cycle
# ---------------------------------------------------------------------------


def should_reanalyze(state: GraphState) -> str:
    """Conditional edge function: decide whether to cycle back or finalize."""
    decision = state.get("decision")
    reanalysis_count = state.get("reanalysis_count", 0)
    max_reanalysis = state.get("max_reanalysis", 2)

    needs_reanalysis = (
        decision.get("needs_reanalysis", False)
        if isinstance(decision, dict)
        else getattr(decision, "needs_reanalysis", False)
    )

    if decision and needs_reanalysis and reanalysis_count < max_reanalysis:
        logger.info("→ Routing to re-analysis (cycle %d/%d)", reanalysis_count + 1, max_reanalysis)
        return "reanalyze"

    logger.info("→ Routing to report generation (final)")
    return "finalize"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_compliance_graph() -> StateGraph:
    """Construct the LangGraph compliance workflow.

    Flow:
        intake → kyc → transaction_pattern → geopolitical → risk_scoring → decision
                                                                              ↓
                                                                     [conditional]
                                                                    /            \\
                                                              reanalyze        finalize
                                                                 ↓                ↓
                                                          (cycle back         report → END
                                                           to kyc)
    """
    graph = StateGraph(GraphState)

    # --- Add nodes ---
    graph.add_node("intake", intake_node)
    graph.add_node("kyc", kyc_node)
    graph.add_node("transaction_pattern", transaction_pattern_node)
    graph.add_node("geopolitical", geopolitical_node)
    graph.add_node("risk_scoring", risk_scoring_node)
    graph.add_node("decision", decision_node)
    graph.add_node("report", report_node)
    graph.add_node("reanalysis", reanalysis_node)

    # --- Add edges (linear spine) ---
    graph.set_entry_point("intake")
    graph.add_edge("intake", "kyc")
    graph.add_edge("kyc", "transaction_pattern")
    graph.add_edge("transaction_pattern", "geopolitical")
    graph.add_edge("geopolitical", "risk_scoring")
    graph.add_edge("risk_scoring", "decision")

    # --- Conditional edge: decision → reanalyze or finalize ---
    graph.add_conditional_edges(
        "decision",
        should_reanalyze,
        {
            "reanalyze": "reanalysis",
            "finalize": "report",
        },
    )

    # --- Re-analysis cycle: loops back to KYC for deeper pass ---
    graph.add_edge("reanalysis", "kyc")

    # --- Terminal edge ---
    graph.add_edge("report", END)

    return graph


def compile_graph():
    """Build and compile the graph for execution."""
    graph = build_compliance_graph()
    return graph.compile()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


def run_compliance_pipeline(transaction) -> PipelineState:
    """Execute the full compliance pipeline for a transaction.

    Returns the final PipelineState with report attached.
    """
    start = time.time()

    initial_state: GraphState = {
        "transaction": transaction,
        "kyc_findings": [],
        "pattern_finding": None,
        "geopolitical_finding": None,
        "risk_score": None,
        "decision": None,
        "alerts": [],
        "report": None,
        "processing_steps": [],
        "reanalysis_count": 0,
        "max_reanalysis": 2,
        "error": None,
        "status": "initialized",
    }

    app = compile_graph()
    final_state = app.invoke(initial_state)

    elapsed_ms = (time.time() - start) * 1000
    ps = _to_pipeline(final_state)

    if ps.report:
        ps.report.processing_time_ms = round(elapsed_ms, 1)

    logger.info("Pipeline complete in %.0fms for TXN %s", elapsed_ms, transaction.transaction_id)
    return ps


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
