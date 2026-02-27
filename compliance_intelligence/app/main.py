"""FastAPI backend — REST API + WebSocket streaming for the compliance pipeline.

Endpoints:
  GET  /                  → Serve the frontend
  GET  /api/scenarios     → List available demo scenarios
  POST /api/analyze       → Run a scenario through the pipeline
  POST /api/analyze/custom → Run a custom transaction
  GET  /api/health        → Health check
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.graph.workflow import run_compliance_pipeline
from app.models import Transaction
from app.scenarios import SCENARIOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Compliance Intelligence",
    description="Multi-Agent AML/KYC Transaction Monitoring powered by LangGraph + Pydantic",
    version="1.0.0",
)

# NOTE: Wildcard CORS is acceptable for this demo app only.
# Production deployments must restrict origins to known domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ScenarioInfo(BaseModel):
    key: str
    name: str
    description: str
    expected_verdict: str


class AnalyzeRequest(BaseModel):
    scenario_key: str = Field(description="Key from /api/scenarios")


class AnalyzeResponse(BaseModel):
    success: bool
    processing_time_ms: float
    report: dict
    executive_summary: str
    graph_trace: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def serve_frontend():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(index)


@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/scenarios", response_model=list[ScenarioInfo])
async def list_scenarios():
    return [
        ScenarioInfo(
            key=key,
            name=info["name"],
            description=info["description"],
            expected_verdict=info["expected_verdict"],
        )
        for key, info in SCENARIOS.items()
    ]


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_scenario(req: AnalyzeRequest):
    if req.scenario_key not in SCENARIOS:
        raise HTTPException(
            400, f"Unknown scenario: {req.scenario_key}. Available: {list(SCENARIOS.keys())}"
        )

    scenario = SCENARIOS[req.scenario_key]
    transaction = scenario["factory"]()

    logger.info("Analyzing scenario '%s' — TXN %s", req.scenario_key, transaction.transaction_id)
    start = time.time()

    try:
        final_state = run_compliance_pipeline(transaction)
    except Exception:
        logger.exception("Pipeline failed for scenario '%s'", req.scenario_key)
        raise HTTPException(500, "Pipeline error — see server logs for details")

    elapsed = (time.time() - start) * 1000

    if not final_state.report:
        raise HTTPException(500, "Pipeline completed but no report was generated")

    report = final_state.report

    return AnalyzeResponse(
        success=True,
        processing_time_ms=round(elapsed, 1),
        report=report.model_dump(mode="json"),
        executive_summary=report.executive_summary,
        graph_trace=final_state.processing_steps,
    )


@app.post("/api/analyze/custom")
async def analyze_custom(transaction: Transaction):
    """Run a fully custom transaction through the pipeline."""
    logger.info("Analyzing custom TXN %s", transaction.transaction_id)
    start = time.time()

    try:
        final_state = run_compliance_pipeline(transaction)
    except Exception:
        logger.exception("Pipeline failed for custom TXN")
        raise HTTPException(500, "Pipeline error — see server logs for details")

    elapsed = (time.time() - start) * 1000

    if not final_state.report:
        raise HTTPException(500, "Pipeline completed but no report was generated")

    report = final_state.report

    return AnalyzeResponse(
        success=True,
        processing_time_ms=round(elapsed, 1),
        report=report.model_dump(mode="json"),
        executive_summary=report.executive_summary,
        graph_trace=final_state.processing_steps,
    )
