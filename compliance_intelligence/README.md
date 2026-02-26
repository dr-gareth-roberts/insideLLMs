# Compliance Intelligence

**Multi-Agent AML/KYC Transaction Monitoring Platform**

A production-grade demonstration of **LangGraph**, **Deep Agents**, and **Pydantic v2** working together to solve a real-world compliance problem worth billions to the financial services industry.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-purple)
![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-green)

---

## What It Does

Financial institutions process millions of transactions daily and must screen each one for money laundering, sanctions violations, terrorist financing, and fraud. This system automates that process using a **multi-agent pipeline** that:

1. **Ingests & enriches** transaction data (entity risk profiling, threshold detection)
2. **Screens entities** against sanctions lists, PEP databases, and adverse media (KYC/AML Agent)
3. **Detects suspicious patterns** — structuring, layering, round-tripping, velocity anomalies (Pattern Agent)
4. **Evaluates geopolitical risk** — sanctions programs, embargoes, trade restrictions (Geo Agent)
5. **Computes composite risk scores** using a weighted multi-factor model (Risk Scoring Agent)
6. **Renders compliance decisions** — approve, flag, escalate, or block (Decision Agent)
7. **Generates structured reports** with full audit trails (Report Agent)

The Decision Agent can **trigger re-analysis cycles** (LangGraph conditional edges + cycles) when confidence is insufficient — a key differentiator from simple linear pipelines.

## Architecture

```
START → Intake → KYC/AML → Transaction Pattern → Geopolitical → Risk Scoring → Decision
                                                                                  ↓
                                                                           [conditional]
                                                                          /            \
                                                                   reanalyze        finalize
                                                                      ↓                ↓
                                                               (cycle back         Report → END
                                                                to KYC)
```

### Technology Showcase

| Technology | What It Demonstrates |
|---|---|
| **LangGraph** | `StateGraph` with typed state, conditional edges, cycles (re-analysis loop), error handling |
| **Deep Agents** | 6 specialized agents with domain expertise, tools, and reasoning capabilities |
| **Pydantic v2** | Validators, computed fields, discriminated unions, frozen models, custom serialization, strict type enforcement |

## Quick Start

### 1. Install Dependencies

```bash
cd compliance_intelligence
pip install -r requirements.txt
```

### 2. Run the CLI Demo

```bash
# Run all 5 scenarios with rich terminal output
python demo.py

# Run a specific scenario
python demo.py --scenario critical_sanctions

# Show full agent trace
python demo.py --scenario high_risk --trace

# List available scenarios
python demo.py --list
```

### 3. Run the Web UI

```bash
python run_server.py
# Open http://localhost:8000
```

## Demo Scenarios

| Scenario | Description | Expected Verdict |
|---|---|---|
| **Low Risk** | $4,250 domestic ACH between verified US corps | ✅ Approve |
| **Medium Risk** | €87,500 wire to a PEP in Turkey | ⚠️ Flag / Approve with monitoring |
| **High Risk** | 3.75 BTC between BVI shell company & Panama trust | 🔴 Escalate (with re-analysis) |
| **Critical Sanctions** | €245K SWIFT to sanctioned Iranian oil company | 🚫 Block immediately |
| **Structuring** | $9,850 cash deposit just below CTR threshold | ⚠️ Flag for structuring |

## Pydantic v2 Features Demonstrated

- **Computed fields**: `Entity.inherent_risk`, `Transaction.amount_usd_approx`, `Transaction.fingerprint`
- **Discriminated unions**: `AgentFinding = Union[KYCFinding, TransactionPatternFinding, GeopoliticalFinding]`
- **Validators**: `@field_validator` for amount rounding, `@model_validator` for entity uniqueness
- **Frozen models**: `RiskScore`, `Address` — immutable after creation
- **Nested models**: Deep entity → address → transaction → report hierarchy
- **Settings management**: `pydantic-settings` for config from env vars

## LangGraph Features Demonstrated

- **StateGraph**: Typed state flowing through all nodes
- **Conditional edges**: Decision → reanalyze or finalize
- **Cycles**: Re-analysis loop back to KYC when confidence is low
- **Node composition**: Each agent is a self-contained node with Pydantic validation at boundaries

## Project Structure

```
compliance_intelligence/
├── app/
│   ├── models.py              # Pydantic v2 models (20+ models)
│   ├── config.py              # pydantic-settings configuration
│   ├── scenarios.py           # 5 pre-built demo scenarios
│   ├── main.py                # FastAPI backend
│   ├── agents/
│   │   ├── intake.py          # Document Intake & Enrichment
│   │   ├── kyc.py             # KYC/AML Deep Agent
│   │   ├── transaction_pattern.py  # Pattern Detection Agent
│   │   ├── geopolitical.py    # Geopolitical Risk Agent
│   │   ├── risk_scoring.py    # Composite Risk Scoring
│   │   ├── decision.py        # Compliance Decision Agent
│   │   └── report.py          # Report Generation Agent
│   ├── graph/
│   │   ├── state.py           # LangGraph typed state
│   │   └── workflow.py        # LangGraph workflow orchestration
│   └── static/
│       └── index.html         # Beautiful web UI (Tailwind + Alpine.js)
├── demo.py                    # CLI demo with rich output
├── run_server.py              # Web server launcher
├── requirements.txt
└── README.md
```

## Configuration

Set `CI_SIMULATION_MODE=false` and provide `CI_OPENAI_API_KEY` to use real LLM-powered agents instead of the simulation engine. The simulation mode produces realistic outputs without any API calls.

```bash
# .env
CI_SIMULATION_MODE=false
CI_OPENAI_API_KEY=sk-...
CI_OPENAI_MODEL=gpt-4o
```

## Industry Impact

This system addresses a **$274B+ global compliance market** (2024). Financial institutions spend:
- **$35B+/year** on AML compliance alone
- **10-15% of operating costs** on regulatory compliance
- **60-80% of compliance budget** on manual review processes

Multi-agent AI systems like this can reduce manual review time by 70-90% while improving detection accuracy.
