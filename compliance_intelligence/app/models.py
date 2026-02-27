"""Pydantic v2 models — the structured data backbone of the compliance system.

Demonstrates: validators, computed fields, discriminated unions, nested models,
custom serialization, and strict type enforcement.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"
    CNY = "CNY"
    AED = "AED"
    RUB = "RUB"
    BTC = "BTC"
    ETH = "ETH"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EntityType(str, Enum):
    INDIVIDUAL = "individual"
    CORPORATION = "corporation"
    TRUST = "trust"
    GOVERNMENT = "government"
    NGO = "ngo"
    SHELL_COMPANY = "shell_company"


class TransactionType(str, Enum):
    WIRE_TRANSFER = "wire_transfer"
    ACH = "ach"
    SWIFT = "swift"
    CRYPTO = "crypto"
    CASH_DEPOSIT = "cash_deposit"
    CASH_WITHDRAWAL = "cash_withdrawal"
    INTERNAL_TRANSFER = "internal_transfer"
    TRADE_SETTLEMENT = "trade_settlement"


class DecisionVerdict(str, Enum):
    APPROVE = "approve"
    FLAG_FOR_REVIEW = "flag_for_review"
    ESCALATE = "escalate"
    BLOCK = "block"
    REQUEST_MORE_INFO = "request_more_info"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------

class Address(BaseModel):
    """Physical or registered address."""
    model_config = ConfigDict(frozen=True)

    street: str = Field(min_length=1)
    city: str = Field(min_length=1)
    country_code: str = Field(pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2")
    postal_code: str | None = None
    state_province: str | None = None


class Entity(BaseModel):
    """A counterparty in a transaction — individual or organisation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    entity_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = Field(min_length=1, max_length=256)
    entity_type: EntityType
    country_code: str = Field(pattern=r"^[A-Z]{2}$")
    address: Address | None = None
    tax_id: str | None = None
    incorporation_date: datetime | None = None
    pep: bool = Field(default=False, description="Politically Exposed Person")
    sanctioned: bool = False
    adverse_media_hits: int = Field(default=0, ge=0)

    @computed_field
    @property
    def high_risk_jurisdiction(self) -> bool:
        """Automatically flag entities in FATF high-risk jurisdictions."""
        HIGH_RISK = {"IR", "KP", "MM", "SY", "YE", "AF", "AL", "BF", "CM", "CD",
                     "HT", "JM", "JO", "ML", "MZ", "NI", "PK", "PA", "PH", "SN",
                     "SS", "TZ", "TR", "UG", "VU"}
        return self.country_code in HIGH_RISK

    @computed_field
    @property
    def inherent_risk(self) -> RiskLevel:
        """Compute inherent entity risk from static attributes."""
        score = 0
        if self.pep:
            score += 3
        if self.sanctioned:
            score += 5
        if self.high_risk_jurisdiction:
            score += 2
        if self.entity_type in (EntityType.SHELL_COMPANY, EntityType.TRUST):
            score += 2
        if self.adverse_media_hits > 0:
            score += min(self.adverse_media_hits, 3)
        if score >= 5:
            return RiskLevel.CRITICAL
        if score >= 3:
            return RiskLevel.HIGH
        if score >= 1:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class Transaction(BaseModel):
    """A single financial transaction submitted for compliance review."""
    model_config = ConfigDict(str_strip_whitespace=True)

    transaction_id: str = Field(default_factory=lambda: f"TXN-{uuid.uuid4().hex[:10].upper()}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    transaction_type: TransactionType
    amount: float = Field(gt=0, description="Transaction amount in source currency")
    currency: Currency
    source_entity: Entity
    destination_entity: Entity
    description: str = Field(default="", max_length=1024)
    reference_id: str | None = None
    intermediary_banks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("amount")
    @classmethod
    def round_amount(cls, v: float) -> float:
        return round(v, 8 if v < 1 else 2)  # 8 decimals for crypto

    @model_validator(mode="after")
    def source_and_dest_differ(self) -> "Transaction":
        if self.source_entity.entity_id == self.destination_entity.entity_id:
            raise ValueError("Source and destination entities must differ")
        return self

    @computed_field
    @property
    def amount_usd_approx(self) -> float:
        """Rough USD equivalent for triage (real system would use live FX)."""
        FX = {
            Currency.USD: 1.0, Currency.EUR: 1.08, Currency.GBP: 1.26,
            Currency.CHF: 1.12, Currency.JPY: 0.0067, Currency.CNY: 0.14,
            Currency.AED: 0.27, Currency.RUB: 0.011, Currency.BTC: 62_000.0,
            Currency.ETH: 3_400.0,
        }
        return round(self.amount * FX.get(self.currency, 1.0), 2)

    @computed_field
    @property
    def fingerprint(self) -> str:
        """Content-addressable hash for deduplication."""
        raw = f"{self.source_entity.entity_id}|{self.destination_entity.entity_id}|{self.amount}|{self.currency.value}|{self.timestamp.isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Agent analysis outputs — discriminated union pattern
# ---------------------------------------------------------------------------

class KYCFinding(BaseModel):
    """Output of the KYC/AML deep agent."""
    agent: Literal["kyc_aml"] = "kyc_aml"
    entity_id: str
    identity_verified: bool
    sanctions_match: bool
    pep_match: bool
    adverse_media_summary: str = ""
    watchlist_hits: list[str] = Field(default_factory=list)
    verification_sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def kyc_risk(self) -> RiskLevel:
        if self.sanctions_match:
            return RiskLevel.CRITICAL
        score = 0
        if self.pep_match:
            score += 2
        if not self.identity_verified:
            score += 2
        if len(self.watchlist_hits) > 0:
            score += min(len(self.watchlist_hits), 3)
        if self.adverse_media_summary:
            score += 1
        if score >= 4:
            return RiskLevel.HIGH
        if score >= 2:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class TransactionPatternFinding(BaseModel):
    """Output of the Transaction Pattern Analysis deep agent."""
    agent: Literal["transaction_pattern"] = "transaction_pattern"
    pattern_type: str
    description: str
    structuring_detected: bool = False
    rapid_movement_detected: bool = False
    round_tripping_detected: bool = False
    layering_detected: bool = False
    unusual_timing: bool = False
    velocity_anomaly: bool = False
    historical_deviation_score: float = Field(ge=0.0, le=1.0)
    similar_suspicious_cases: int = Field(ge=0, default=0)
    evidence: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def pattern_risk(self) -> RiskLevel:
        flags = sum([
            self.structuring_detected,
            self.rapid_movement_detected,
            self.round_tripping_detected,
            self.layering_detected,
            self.unusual_timing,
            self.velocity_anomaly,
        ])
        if flags >= 3 or self.round_tripping_detected or self.layering_detected:
            return RiskLevel.CRITICAL
        if flags >= 2 or self.structuring_detected:
            return RiskLevel.HIGH
        if flags >= 1 or self.historical_deviation_score > 0.7:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class GeopoliticalFinding(BaseModel):
    """Output of the Geopolitical Risk deep agent."""
    agent: Literal["geopolitical"] = "geopolitical"
    corridor: str = Field(description="e.g. 'US → IR' for country corridor")
    sanctions_programs: list[str] = Field(default_factory=list)
    embargo_active: bool = False
    trade_restrictions: list[str] = Field(default_factory=list)
    regulatory_warnings: list[str] = Field(default_factory=list)
    risk_narrative: str = ""

    @computed_field
    @property
    def geo_risk(self) -> RiskLevel:
        if self.embargo_active:
            return RiskLevel.CRITICAL
        if len(self.sanctions_programs) > 0:
            return RiskLevel.HIGH
        if len(self.trade_restrictions) > 0:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


AgentFinding = Annotated[
    Union[KYCFinding, TransactionPatternFinding, GeopoliticalFinding],
    Field(discriminator="agent"),
]


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

class RiskScore(BaseModel):
    """Composite risk score produced by the Risk Scoring agent."""
    model_config = ConfigDict(frozen=True)

    overall_score: float = Field(ge=0.0, le=100.0)
    overall_level: RiskLevel
    entity_risk_score: float = Field(ge=0.0, le=100.0)
    transaction_risk_score: float = Field(ge=0.0, le=100.0)
    pattern_risk_score: float = Field(ge=0.0, le=100.0)
    geopolitical_risk_score: float = Field(ge=0.0, le=100.0)
    contributing_factors: list[str] = Field(default_factory=list)
    mitigating_factors: list[str] = Field(default_factory=list)
    model_version: str = "v2.1.0"

    @computed_field
    @property
    def risk_breakdown(self) -> dict[str, float]:
        return {
            "entity": self.entity_risk_score,
            "transaction": self.transaction_risk_score,
            "pattern": self.pattern_risk_score,
            "geopolitical": self.geopolitical_risk_score,
        }


# ---------------------------------------------------------------------------
# Decision & compliance report
# ---------------------------------------------------------------------------

class ComplianceDecision(BaseModel):
    """Final decision from the Compliance Decision agent."""
    verdict: DecisionVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    regulatory_references: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    escalation_required: bool = False
    sla_hours: int = Field(ge=1, le=720, default=24)
    needs_reanalysis: bool = Field(
        default=False,
        description="If True, the workflow cycles back for deeper analysis",
    )
    reanalysis_reason: str = ""


class Alert(BaseModel):
    """A single alert generated during processing."""
    alert_id: str = Field(default_factory=lambda: f"ALT-{uuid.uuid4().hex[:8].upper()}")
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ComplianceReport(BaseModel):
    """Full compliance report — the final artifact of the pipeline."""
    report_id: str = Field(default_factory=lambda: f"RPT-{uuid.uuid4().hex[:10].upper()}")
    transaction: Transaction
    findings: list[AgentFinding] = Field(default_factory=list)
    risk_score: RiskScore | None = None
    decision: ComplianceDecision | None = None
    alerts: list[Alert] = Field(default_factory=list)
    processing_steps: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_ms: float = 0.0
    reanalysis_count: int = Field(default=0, ge=0)

    @computed_field
    @property
    def executive_summary(self) -> str:
        if not self.decision:
            return "Processing incomplete."
        src = self.transaction.source_entity.name
        dst = self.transaction.destination_entity.name
        amt = f"{self.transaction.currency.value} {self.transaction.amount:,.2f}"
        verdict = self.decision.verdict.value.replace("_", " ").title()
        risk = self.risk_score.overall_level.value.upper() if self.risk_score else "N/A"
        return (
            f"Transaction {self.transaction.transaction_id}: {amt} from {src} to {dst}. "
            f"Risk: {risk}. Verdict: {verdict}. "
            f"{len(self.alerts)} alert(s) raised. "
            f"Confidence: {self.decision.confidence:.0%}."
        )


# ---------------------------------------------------------------------------
# LangGraph state (used as the shared state across nodes)
# ---------------------------------------------------------------------------

class PipelineState(BaseModel):
    """Mutable state flowing through the LangGraph workflow.

    This is the single source of truth for all nodes in the graph.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    transaction: Transaction
    kyc_findings: list[KYCFinding] = Field(default_factory=list)
    pattern_finding: TransactionPatternFinding | None = None
    geopolitical_finding: GeopoliticalFinding | None = None
    risk_score: RiskScore | None = None
    decision: ComplianceDecision | None = None
    alerts: list[Alert] = Field(default_factory=list)
    report: ComplianceReport | None = None
    processing_steps: list[str] = Field(default_factory=list)
    reanalysis_count: int = 0
    max_reanalysis: int = 2
    error: str | None = None
    status: str = "initialized"
