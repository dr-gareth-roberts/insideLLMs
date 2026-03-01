# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously at insideLLMs. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email your findings to the maintainers (see repository for contact info)
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability within 7 days
- **Updates**: We will keep you informed of our progress
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days
- **Credit**: We will credit you in our release notes (unless you prefer anonymity)

### Scope

The following are in scope for security reports:

- Arbitrary code execution via deserialization or `eval`
- Injection vulnerabilities (command, prompt, path traversal)
- Sensitive data exposure (API keys, PII in logs)
- Supply chain vulnerabilities in dependencies
- Security misconfigurations

### Out of Scope

- Vulnerabilities in third-party services/APIs (report to them directly)
- Social engineering attacks
- Issues already known/reported

## Threat Model

insideLLMs is a library for evaluating LLM behavior. Its primary threat vectors are:

| Threat                         | Mitigation                                                     |
| ------------------------------ | -------------------------------------------------------------- |
| **Malicious checkpoint files** | Pickle deserialization fully removed; JSON only                |
| **Code injection via tools**   | `safe_eval` restricts to arithmetic; no `eval()` on user input |
| **Supply chain attacks**       | OIDC Trusted Publisher for PyPI; `pip-audit` in CI             |
| **Credential leakage**         | Pre-commit hooks; `detect-private-key`; custom secret scanner  |
| **Path traversal**             | Checkpoint IDs validated to reject `../` and separators        |

## Security-Sensitive Modules

These modules receive stricter static analysis (`bandit`, strict `mypy`):

| Module                                | Concern                    |
| ------------------------------------- | -------------------------- |
| `insideLLMs/injection.py`             | Prompt injection detection |
| `insideLLMs/safety.py`                | Content safety analysis    |
| `insideLLMs/contrib/distributed.py`   | Checkpoint serialization   |
| `insideLLMs/contrib/agents.py`        | Tool execution             |
| `insideLLMs/runtime/observability.py` | Telemetry data handling    |

## Security Hardening (Implemented)

### Serialization Safety

- **No `pickle.load`** anywhere in the codebase. Legacy pickle checkpoints are rejected with a clear error and migration instructions.
- All serialization uses JSON exclusively.

### Code Execution Safety

- All docstring examples use `ast.literal_eval` or the library's `safe_eval` (arithmetic-only sandbox).
- No use of `eval()`, `exec()`, or `compile()` on user-controlled input.

### CI/CD Security

- PyPI publishing uses **OIDC Trusted Publisher** (no API token secrets).
- Dependencies audited via `pip-audit`.
- `bandit` scans security-sensitive modules on every commit via pre-commit hook.

## Security Best Practices for Users

### API Key Management

```python
# DO: Use environment variables
import os
model = OpenAIModel(api_key=os.getenv("OPENAI_API_KEY"))

# DON'T: Hardcode API keys
model = OpenAIModel(api_key="sk-...")  # NEVER do this!
```

### Input Sanitization

```python
from insideLLMs.injection import InputSanitizer

# Sanitize user inputs before passing to models
sanitizer = InputSanitizer()
safe_input = sanitizer.sanitize(user_input)
```

### PII Detection

```python
from insideLLMs.safety import detect_pii

# Check for PII before logging or storing responses
pii_report = detect_pii(model_response)
if pii_report.found:
    # Handle PII appropriately
    pass
```

## Dependency Security

We regularly audit our dependencies for known vulnerabilities:

```bash
pip install pip-audit
pip-audit
```

## Security Updates

Security updates will be released as patch versions. We recommend:

1. Subscribing to GitHub releases for notifications
2. Keeping your installation up to date
3. Reviewing the CHANGELOG for security-related changes

Thank you for helping keep insideLLMs secure!
