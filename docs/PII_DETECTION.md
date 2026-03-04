# PII Detection Guide

## Overview

insideLLMs provides comprehensive PII (Personally Identifiable Information) detection with support for multiple regions and international formats.

## Supported Regions

### United States (US)
- ✅ Social Security Numbers (SSN)
- ✅ Phone Numbers (various formats)
- ✅ ZIP Codes

### European Union (EU)
- ⚠️ IBAN (International Bank Account Numbers)
- ⚠️ VAT Numbers
- ⚠️ E.164 Phone Numbers
- ⚠️ **Limited coverage** - basic patterns only

### United Kingdom (UK)
- ⚠️ National Insurance Numbers
- ⚠️ Postcodes
- ⚠️ **Limited coverage** - basic patterns only

### Global
- ✅ Email Addresses
- ✅ Credit Card Numbers (Visa, MasterCard, Amex, Discover)
- ✅ URLs
- ✅ IPv4 Addresses

**Legend:**
- ✅ Comprehensive coverage

## Usage Examples

### Basic US Detection

```python
from insideLLMs.safety import PIIDetector

# US + Global patterns (default)
detector = PIIDetector()

text = "Contact: john@example.com, SSN: 123-45-6789, Phone: 555-123-4567"
report = detector.detect(text)

print(f"Found {len(report.matches)} PII items")
for match in report.matches:
    print(f"  {match.pii_type}: {match.value}")
```

### Multi-Region Detection

```python
detector = PIIDetector()

# Add region-specific patterns explicitly (examples only)
detector.add_pattern("iban", r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", "[IBAN]")
detector.add_pattern("uk_postcode", r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", "[UK_POSTCODE]")

text = "IBAN: GB82WEST12345698765432, Postcode: SW1A 1AA, Email: test@example.com"
report = detector.detect(text)
print(f"Found {len(report.matches)} PII items")
```

### Custom Patterns

```python
detector = PIIDetector()

# Add company-specific pattern
detector.add_pattern(
    name="employee_id",
    pattern=r"EMP-\d{6}",
    mask="[EMP_ID]",
)

text = "Employee EMP-123456 submitted the report"
report = detector.detect(text)
```

## Known Limitations

### EU/UK Patterns
- Basic regex patterns only
- No checksum validation
- May not cover all regional variations
- False positives possible with similar number formats

### Credit Cards
- Pattern matching only (no Luhn algorithm validation)
- May match invalid card numbers

### Phone Numbers
- Format variations may not be caught
- International formats outside E.164 may be missed

## Production Recommendations

For production systems with strict compliance requirements (GDPR, HIPAA, etc.), consider:

1. **Use specialized libraries:**
   ```bash
   pip install presidio-analyzer presidio-anonymizer
   ```

2. **Combine with manual review:**
   - Use insideLLMs for initial detection
   - Manual review for high-stakes decisions

3. **Regular pattern updates:**
   - Subscribe to pattern updates
   - Test with real data samples

4. **Regional expertise:**
   - Consult local compliance experts
   - Validate patterns with regional data

## GDPR Compliance Note

This PII detector is a tool to help identify common PII types but is **not a complete GDPR compliance solution**. GDPR compliance requires:

- Legal basis for processing
- Data subject rights implementation
- Privacy by design
- Data protection impact assessments
- And more...

**Consult legal counsel for GDPR compliance requirements.**

## Contributing Patterns

To contribute regional patterns:

1. Add pattern to appropriate region list in `insideLLMs/safety.py`
2. Include comprehensive test cases
3. Document limitations and false positive rates
4. Provide references to official format specifications

See `CONTRIBUTING.md` for details.
