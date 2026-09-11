from __future__ import annotations

from unittest.mock import Mock

import pytest
from app.agents import decision as decision_agent
from app.models import (
    AlertSeverity,
    ComplianceDecision,
    DecisionVerdict,
    GeopoliticalFinding,
    KYCFinding,
    PipelineState,
    RiskLevel,
    RiskScore,
)
from app.scenarios import scenario_low_risk
from pydantic import ValidationError


def _risk_score() -> RiskScore:
    return RiskScore(
        overall_score=10,
        overall_level=RiskLevel.LOW,
        entity_risk_score=10,
        transaction_risk_score=10,
        pattern_risk_score=10,
        geopolitical_risk_score=10,
    )


def _state(*, sanctions: bool = False, embargo: bool = False) -> PipelineState:
    transaction = scenario_low_risk()
    return PipelineState(
        transaction=transaction,
        kyc_findings=[
            KYCFinding(
                entity_id=transaction.destination_entity.entity_id,
                identity_verified=True,
                sanctions_match=sanctions,
                pep_match=False,
                confidence=1.0,
            )
        ],
        geopolitical_finding=GeopoliticalFinding(
            corridor="US → US",
            embargo_active=embargo,
        ),
    )


def _approval() -> ComplianceDecision:
    return ComplianceDecision(
        verdict=DecisionVerdict.APPROVE,
        confidence=1.0,
        rationale="Fake live approval",
    )


@pytest.mark.parametrize(
    ("sanctions", "embargo"),
    [(True, False), (False, True), (True, True)],
    ids=["sanctions", "embargo", "sanctions-and-embargo"],
)
@pytest.mark.parametrize("simulation_mode", [True, False], ids=["simulation", "live"])
@pytest.mark.parametrize("risk_present", [True, False], ids=["risk", "missing-risk"])
def test_known_hard_blocks_precede_risk_and_decision_routing(
    monkeypatch: pytest.MonkeyPatch,
    sanctions: bool,
    embargo: bool,
    simulation_mode: bool,
    risk_present: bool,
) -> None:
    state = _state(sanctions=sanctions, embargo=embargo)
    state.risk_score = _risk_score() if risk_present else None
    live_decision = Mock(return_value=_approval())
    monkeypatch.setattr(decision_agent.settings, "simulation_mode", simulation_mode)
    monkeypatch.setattr(decision_agent, "_llm_decision", live_decision)

    result = decision_agent.run_decision(state)

    assert result is state
    assert state.decision is not None
    assert state.decision.verdict is DecisionVerdict.BLOCK
    assert state.decision.escalation_required is True
    assert state.decision.needs_reanalysis is False
    assert state.status == "decision_complete"
    assert state.alerts[-1].severity is AlertSeverity.CRITICAL
    assert "verdict=block" in state.processing_steps[-1]
    live_decision.assert_not_called()


def test_clear_live_state_reaches_live_decision_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    state.risk_score = _risk_score()
    live_decision = Mock(return_value=_approval())
    monkeypatch.setattr(decision_agent.settings, "simulation_mode", False)
    monkeypatch.setattr(decision_agent, "_llm_decision", live_decision)

    decision_agent.run_decision(state)

    live_decision.assert_called_once_with(state, state.risk_score)
    assert state.decision is not None
    assert state.decision.verdict is DecisionVerdict.APPROVE


def test_clear_state_with_missing_risk_requests_reanalysis_without_live_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    live_decision = Mock(return_value=_approval())
    monkeypatch.setattr(decision_agent.settings, "simulation_mode", False)
    monkeypatch.setattr(decision_agent, "_llm_decision", live_decision)

    decision_agent.run_decision(state)

    assert state.decision is not None
    assert state.decision.verdict is DecisionVerdict.REQUEST_MORE_INFO
    assert state.decision.needs_reanalysis is True
    assert state.status == "decision_complete"
    assert "verdict=request_more_info" in state.processing_steps[-1]
    live_decision.assert_not_called()


def test_pipeline_state_rejects_malformed_risk_input() -> None:
    with pytest.raises(ValidationError):
        PipelineState.model_validate(
            {
                "transaction": scenario_low_risk().model_dump(),
                "risk_score": {"overall_score": "not-a-score"},
            }
        )


def test_hard_block_does_not_inspect_malformed_in_memory_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(sanctions=True)
    state.risk_score = "not-a-risk"  # type: ignore[assignment]
    live_decision = Mock(return_value=_approval())
    monkeypatch.setattr(decision_agent.settings, "simulation_mode", False)
    monkeypatch.setattr(decision_agent, "_llm_decision", live_decision)

    decision_agent.run_decision(state)

    assert state.decision is not None
    assert state.decision.verdict is DecisionVerdict.BLOCK
    live_decision.assert_not_called()
