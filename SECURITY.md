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

- Authentication/authorization vulnerabilities
- Injection vulnerabilities (SQL, command, etc.)
- Sensitive data exposure
- Security misconfigurations
- Vulnerabilities in dependencies

### Out of Scope

- Vulnerabilities in third-party services/APIs (report to them directly)
- Social engineering attacks
- Physical security issues
- Issues already known/reported

## Security Best Practices

When using insideLLMs:

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

We regularly audit our dependencies for known vulnerabilities. You can check for vulnerabilities in your installation:

```bash
pip install pip-audit
pip-audit
```

## Security Updates

Security updates will be released as patch versions (e.g., 0.1.1, 0.1.2). We recommend:

1. Subscribing to GitHub releases for notifications
2. Keeping your installation up to date
3. Reviewing the CHANGELOG for security-related changes

Thank you for helping keep insideLLMs secure!
