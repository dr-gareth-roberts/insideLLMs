"""KYC/AML Deep Agent.

Performs Know-Your-Customer and Anti-Money-Laundering checks on transaction
entities. In simulation mode, produces realistic synthetic results.
In live mode, calls an LLM to reason over entity data + external databases.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.models import (
    Alert,
    AlertSeverity,
    EntityType,
    KYCFinding,
    PipelineState,
    RiskLevel,
)

logger = logging.getLogger(__name__)

KYC_SYSTEM_PROMPT = """You are a senior KYC/AML compliance analyst. Given entity data, perform:
1. Identity verification assessment
2. Sanctions screening (OFAC, EU, UN lists)
3. PEP (Politically Exposed Person) screening
4. Adverse media analysis
5. Watchlist cross-reference

Return your analysis as structured JSON matching the KYCFinding schema.
Be thorough and conservative — when in doubt, flag for review."""


WATCHLISTS = [
    "OFAC SDN List",
    "EU Consolidated Sanctions",
    "UN Security Council",
    "UK HMT Sanctions",
    "FATF Blacklist",
    "Interpol Red Notices",
    "World Bank Debarment",
    "FinCEN 311 Special Measures",
]

SANCTIONS_COUNTRIES = {"IR", "KP", "SY", "CU", "VE", "RU", "BY"}


def run_kyc(state: PipelineState) -> PipelineState:
    """KYC/AML node — screens both entities in the transaction."""
    logger.info("▶ KYC Agent screening entities for TXN %s", state.transaction.transaction_id)
    state.status = "kyc_processing"

    for label, entity in [
        ("source", state.transaction.source_entity),
        ("destination", state.transaction.destination_entity),
    ]:
        state.processing_steps.append(f"[{_now()}] KYC: screening {label} entity '{entity.name}'")

        if settings.simulation_mode:
            finding = _simulate_kyc(entity, state)
        else:
            finding = _llm_kyc(entity, state)

        state.kyc_findings.append(finding)

        # Raise alerts for critical findings
        if finding.sanctions_match:
            state.alerts.append(
                Alert(
                    severity=AlertSeverity.CRITICAL,
                    title=f"Sanctions Match — {entity.name}",
                    description=f"Entity matched against: {', '.join(finding.watchlist_hits)}. IMMEDIATE ESCALATION REQUIRED.",
                )
            )
        elif finding.kyc_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            state.alerts.append(
                Alert(
                    severity=AlertSeverity.WARNING,
                    title=f"Elevated KYC Risk — {entity.name}",
                    description=f"KYC risk level: {finding.kyc_risk.value}. Flags: {', '.join(finding.risk_flags)}.",
                )
            )

        state.processing_steps.append(
            f"[{_now()}] KYC: {label} '{entity.name}' → risk={finding.kyc_risk.value}, "
            f"sanctions={finding.sanctions_match}, pep={finding.pep_match}"
        )

    state.status = "kyc_complete"
    state.processing_steps.append(f"[{_now()}] KYC: screening complete for both entities")
    return state


def _simulate_kyc(entity, state: PipelineState) -> KYCFinding:
    """Realistic simulation without LLM calls."""
    is_sanctioned = entity.sanctioned or entity.country_code in SANCTIONS_COUNTRIES
    is_pep = entity.pep
    has_adverse = entity.adverse_media_hits > 0

    # Simulate watchlist hits
    hits = []
    if is_sanctioned:
        hits = random.sample(WATCHLISTS[:4], k=random.randint(1, 3))
    elif entity.high_risk_jurisdiction:
        if random.random() > 0.5:
            hits = [random.choice(WATCHLISTS)]

    # Simulate verification
    verified = not (entity.entity_type == EntityType.SHELL_COMPANY)
    if entity.entity_type == EntityType.TRUST and random.random() > 0.6:
        verified = False

    risk_flags = []
    if is_sanctioned:
        risk_flags.append("sanctions_list_match")
    if is_pep:
        risk_flags.append("politically_exposed_person")
    if not verified:
        risk_flags.append("identity_verification_failed")
    if has_adverse:
        risk_flags.append(f"adverse_media_{entity.adverse_media_hits}_hits")
    if entity.high_risk_jurisdiction:
        risk_flags.append(f"high_risk_jurisdiction_{entity.country_code}")
    if entity.entity_type == EntityType.SHELL_COMPANY:
        risk_flags.append("shell_company_structure")

    adverse_summary = ""
    if has_adverse:
        adverse_summary = (
            f"Found {entity.adverse_media_hits} adverse media mention(s) related to "
            f"{entity.name}. Topics include regulatory enforcement actions and "
            f"suspicious financial activity reports."
        )

    confidence = 0.92 if verified else 0.65
    if is_sanctioned:
        confidence = 0.98

    return KYCFinding(
        entity_id=entity.entity_id,
        identity_verified=verified,
        sanctions_match=is_sanctioned,
        pep_match=is_pep,
        adverse_media_summary=adverse_summary,
        watchlist_hits=hits,
        verification_sources=[
            "Internal KYC DB",
            "Thomson Reuters World-Check",
            "Dow Jones Risk & Compliance",
        ],
        confidence=confidence,
        risk_flags=risk_flags,
    )


def _llm_kyc(entity, state: PipelineState) -> KYCFinding:
    """LLM-powered KYC analysis (requires OpenAI API key)."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key)

    entity_data = entity.model_dump_json(indent=2)
    messages = [
        SystemMessage(content=KYC_SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze this entity for KYC/AML compliance:\n\n{entity_data}"),
    ]
    response = llm.invoke(messages)

    # Parse structured output
    import json

    try:
        data = json.loads(response.content)
        return KYCFinding(entity_id=entity.entity_id, **data)
    except (json.JSONDecodeError, Exception):
        # Fallback to simulation if parsing fails
        logger.warning("LLM KYC output parsing failed, falling back to simulation")
        return _simulate_kyc(entity, state)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
