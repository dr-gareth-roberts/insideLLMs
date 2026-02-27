"""Pre-built demo scenarios showcasing different risk profiles.

Each scenario creates a realistic Transaction that exercises different
parts of the compliance pipeline — from low-risk domestic transfers
to sanctions-violating cross-border wire transfers.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.models import (
    Address,
    Currency,
    Entity,
    EntityType,
    Transaction,
    TransactionType,
)


def scenario_low_risk() -> Transaction:
    """Routine domestic ACH between two verified US corporations."""
    return Transaction(
        transaction_type=TransactionType.ACH,
        amount=4_250.00,
        currency=Currency.USD,
        source_entity=Entity(
            name="Acme Manufacturing LLC",
            entity_type=EntityType.CORPORATION,
            country_code="US",
            address=Address(street="123 Industrial Blvd", city="Detroit", country_code="US", postal_code="48201"),
            tax_id="12-3456789",
        ),
        destination_entity=Entity(
            name="Great Lakes Supply Co.",
            entity_type=EntityType.CORPORATION,
            country_code="US",
            address=Address(street="456 Commerce Dr", city="Chicago", country_code="US", postal_code="60601"),
            tax_id="98-7654321",
        ),
        description="Monthly supplier payment for raw materials — PO #2024-0891",
    )


def scenario_medium_risk() -> Transaction:
    """Cross-border wire to a high-risk jurisdiction with a PEP."""
    return Transaction(
        transaction_type=TransactionType.WIRE_TRANSFER,
        amount=87_500.00,
        currency=Currency.EUR,
        source_entity=Entity(
            name="Meridian Capital Partners AG",
            entity_type=EntityType.CORPORATION,
            country_code="CH",
            address=Address(street="Bahnhofstrasse 42", city="Zurich", country_code="CH", postal_code="8001"),
        ),
        destination_entity=Entity(
            name="Hassan Al-Rashid",
            entity_type=EntityType.INDIVIDUAL,
            country_code="TR",
            pep=True,
            adverse_media_hits=2,
            address=Address(street="Istiklal Caddesi 15", city="Istanbul", country_code="TR"),
        ),
        description="Investment advisory fee — Q4 2024",
        intermediary_banks=["Deutsche Bank AG", "Turkiye Is Bankasi"],
    )


def scenario_high_risk() -> Transaction:
    """Large crypto transfer involving a shell company and multiple intermediaries."""
    return Transaction(
        transaction_type=TransactionType.CRYPTO,
        amount=3.75,
        currency=Currency.BTC,
        source_entity=Entity(
            name="NovaTech Digital Holdings Ltd",
            entity_type=EntityType.SHELL_COMPANY,
            country_code="VG",
            adverse_media_hits=4,
            address=Address(street="P.O. Box 3175", city="Road Town", country_code="VG"),
        ),
        destination_entity=Entity(
            name="Constellation Asset Management",
            entity_type=EntityType.TRUST,
            country_code="PA",
            adverse_media_hits=1,
            address=Address(street="Calle 50, Torre Global", city="Panama City", country_code="PA"),
        ),
        description="Digital asset transfer — portfolio rebalancing",
        intermediary_banks=["Silvergate Exchange Network", "Signature Bank", "Tether Treasury"],
    )


def scenario_critical_sanctions() -> Transaction:
    """SWIFT transfer to a sanctioned Iranian entity — should be blocked."""
    return Transaction(
        transaction_type=TransactionType.SWIFT,
        amount=245_000.00,
        currency=Currency.EUR,
        source_entity=Entity(
            name="Global Trade Facilitators GmbH",
            entity_type=EntityType.CORPORATION,
            country_code="DE",
            address=Address(street="Friedrichstraße 123", city="Berlin", country_code="DE", postal_code="10117"),
        ),
        destination_entity=Entity(
            name="Pars Oil & Gas Engineering Co.",
            entity_type=EntityType.CORPORATION,
            country_code="IR",
            sanctioned=True,
            adverse_media_hits=7,
            address=Address(street="Keshavarz Blvd", city="Tehran", country_code="IR"),
        ),
        description="Equipment procurement — energy sector",
        intermediary_banks=["Commerzbank AG", "Central Bank of Iran"],
    )


def scenario_structuring() -> Transaction:
    """Cash deposit just below CTR threshold — classic structuring pattern."""
    return Transaction(
        transaction_type=TransactionType.CASH_DEPOSIT,
        amount=9_850.00,
        currency=Currency.USD,
        source_entity=Entity(
            name="Maria Santos",
            entity_type=EntityType.INDIVIDUAL,
            country_code="US",
            address=Address(street="789 Oak Street", city="Miami", country_code="US", postal_code="33101"),
        ),
        destination_entity=Entity(
            name="Sunshine Import Export Inc.",
            entity_type=EntityType.CORPORATION,
            country_code="US",
            adverse_media_hits=1,
            address=Address(street="1200 Brickell Ave", city="Miami", country_code="US", postal_code="33131"),
        ),
        description="Business operating deposit",
    )


SCENARIOS = {
    "low_risk": {
        "name": "Low Risk — Routine Domestic ACH",
        "description": "Standard $4,250 supplier payment between two verified US corporations. Expected: APPROVE.",
        "factory": scenario_low_risk,
        "expected_verdict": "approve",
    },
    "medium_risk": {
        "name": "Medium Risk — Cross-Border PEP Wire",
        "description": "€87,500 wire from Switzerland to a Politically Exposed Person in Turkey. Expected: FLAG or APPROVE with monitoring.",
        "factory": scenario_medium_risk,
        "expected_verdict": "flag_for_review",
    },
    "high_risk": {
        "name": "High Risk — Crypto Shell Company Transfer",
        "description": "3.75 BTC (~$232K) between a BVI shell company and a Panamanian trust. Expected: ESCALATE with re-analysis cycle.",
        "factory": scenario_high_risk,
        "expected_verdict": "escalate",
    },
    "critical_sanctions": {
        "name": "Critical — Sanctions Violation (Iran)",
        "description": "€245,000 SWIFT to a sanctioned Iranian oil company. Expected: BLOCK immediately.",
        "factory": scenario_critical_sanctions,
        "expected_verdict": "block",
    },
    "structuring": {
        "name": "Structuring — Sub-Threshold Cash Deposit",
        "description": "$9,850 cash deposit just below $10K CTR threshold. Expected: FLAG for structuring pattern.",
        "factory": scenario_structuring,
        "expected_verdict": "flag_for_review",
    },
}
