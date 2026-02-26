# Safety Module Design and Consolidation

## Current State

insideLLMs currently has two related but separate modules for safety:

1. **`insideLLMs/safety.py`**: General safety utilities
   - PII detection and masking
   - Content moderation
   - Risk assessment
   - General safety checks

2. **`insideLLMs/injection.py`**: Prompt injection specific
   - Injection detection
   - Input sanitization
   - Defensive prompt building
   - Injection testing

## Design Decision: Keep Modules Separate

After careful consideration, we recommend **keeping the modules separate** for the following reasons:

### Rationale

1. **Clear Separation of Concerns**
   - `safety.py`: Reactive safety (detect problems in content)
   - `injection.py`: Proactive defense (prevent attacks)

2. **Different Use Cases**
   - Safety: Content filtering, compliance, monitoring
   - Injection: Security hardening, attack prevention

3. **Import Clarity**
   - Users can import only what they need
   - Clear intent: `from insideLLMs.safety import detect_pii` vs `from insideLLMs.injection import detect_injection`

4. **Maintenance**
   - Each module can evolve independently
   - Easier to test and maintain focused modules

### Module Responsibilities

#### `insideLLMs/safety.py`

**Purpose**: Detect and handle unsafe content in inputs/outputs

**Key Functions**:
- `detect_pii()`: Find personally identifiable information
- `detect_toxicity()`: Identify toxic/harmful content
- `assess_risk()`: Overall risk assessment
- `moderate_content()`: Content moderation

**When to use**:
- Checking model outputs for PII before logging
- Content moderation for user-facing applications
- Compliance with data protection regulations
- Monitoring for harmful content

**Example**:
```python
from insideLLMs.safety import detect_pii, PIIDetector

# Check output for PII before logging
response = model.generate(prompt)
pii_report = detect_pii(response)

if pii_report.has_pii:
    # Mask PII before logging
    safe_response = pii_report.masked_text
    logger.info(f"Response: {safe_response}")
```

#### `insideLLMs/injection.py`

**Purpose**: Prevent and detect prompt injection attacks

**Key Functions**:
- `detect_injection()`: Identify injection attempts
- `sanitize_input()`: Clean malicious input
- `build_defensive_prompt()`: Create injection-resistant prompts
- `test_resistance()`: Test model vulnerability

**When to use**:
- Accepting user input for LLM prompts
- Building security-critical applications
- Testing model robustness
- Implementing defense-in-depth

**Example**:
```python
from insideLLMs.injection import detect_injection, sanitize_input

# Check user input for injection attempts
user_input = request.form['query']
injection_report = detect_injection(user_input)

if injection_report.is_suspicious:
    # Sanitize or reject
    safe_input = sanitize_input(user_input).sanitized
    response = model.generate(safe_input)
```

## Usage Guidelines

### Decision Tree

```
Is your concern about...

├─ Content in model outputs?
│  ├─ PII/sensitive data? → safety.detect_pii()
│  ├─ Toxic/harmful content? → safety.detect_toxicity()
│  └─ General risk? → safety.assess_risk()
│
└─ Malicious user inputs?
   ├─ Injection attempts? → injection.detect_injection()
   ├─ Need to sanitize? → injection.sanitize_input()
   └─ Building prompts? → injection.build_defensive_prompt()
```

### Common Patterns

#### Pattern 1: Input Validation + Output Safety

```python
from insideLLMs.injection import detect_injection, sanitize_input
from insideLLMs.safety import detect_pii

# 1. Check input for injection
user_input = get_user_input()
if detect_injection(user_input).is_suspicious:
    user_input = sanitize_input(user_input).sanitized

# 2. Generate response
response = model.generate(user_input)

# 3. Check output for PII
pii_report = detect_pii(response)
if pii_report.has_pii:
    response = pii_report.masked_text

return response
```

#### Pattern 2: Defense-in-Depth

```python
from insideLLMs.injection import build_defensive_prompt, DefenseStrategy
from insideLLMs.safety import assess_risk

# 1. Build defensive prompt
system_prompt = "You are a helpful assistant."
user_input = get_user_input()

prompt = build_defensive_prompt(
    system_prompt,
    user_input,
    strategy=DefenseStrategy.INSTRUCTION_DEFENSE
)

# 2. Generate with defensive prompt
response = model.generate(prompt)

# 3. Assess overall risk
risk = assess_risk(response)
if risk.level == RiskLevel.HIGH:
    return "I cannot provide that information."

return response
```

#### Pattern 3: Security Testing

```python
from insideLLMs.injection import InjectionTester
from insideLLMs.safety import SafetyTester

# Test injection resistance
injection_tester = InjectionTester()
injection_report = injection_tester.test_resistance(model.generate)

# Test safety compliance
safety_tester = SafetyTester()
safety_report = safety_tester.test_safety(model.generate)

# Combined report
print(f"Injection block rate: {injection_report.block_rate}")
print(f"Safety compliance: {safety_report.compliance_rate}")
```

## Cross-Module Integration

While modules are separate, they work together seamlessly:

```python
from insideLLMs import safety, injection

# Combined safety pipeline
class SafetyPipeline:
    def __init__(self):
        self.injection_detector = injection.InjectionDetector()
        self.pii_detector = safety.PIIDetector()
    
    def process(self, user_input: str, model_response: str):
        # Check input
        if self.injection_detector.detect(user_input).is_suspicious:
            raise SecurityError("Injection detected")
        
        # Check output
        pii_report = self.pii_detector.detect(model_response)
        if pii_report.has_pii:
            return pii_report.masked_text
        
        return model_response
```

## Future Considerations

### Potential Consolidation (v3.0+)

If consolidation becomes necessary in the future, we would:

1. Create `insideLLMs.security` package:
   ```
   insideLLMs/security/
   ├── __init__.py
   ├── injection.py      # Prompt injection
   ├── content.py        # Content safety (PII, toxicity)
   ├── assessment.py     # Risk assessment
   └── testing.py        # Security testing
   ```

2. Maintain compatibility shims:
   ```python
   # insideLLMs/safety.py
   from insideLLMs.security.content import *
   
   # insideLLMs/injection.py
   from insideLLMs.security.injection import *
   ```

3. Deprecation timeline:
   - v3.0: New structure, old imports work with warnings
   - v3.1-3.x: Continued warnings
   - v4.0: Remove old imports

## Recommendations

### For Users

1. **Use both modules as needed** - they complement each other
2. **Follow the decision tree** above to choose the right module
3. **Combine for defense-in-depth** - input validation + output safety

### For Contributors

1. **Keep modules focused** - don't mix concerns
2. **Add cross-references** in docstrings (see below)
3. **Update this document** when adding new functionality

## See Also

- [Security Best Practices](SECURITY.md)
- [Prompt Injection Guide](PROMPT_INJECTION.md)
- [PII Detection Guide](PII_DETECTION.md)