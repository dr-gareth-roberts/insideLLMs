#!/usr/bin/env python3
"""Pre-commit hook to detect accidentally committed secrets."""

import re
import sys
from pathlib import Path

# Patterns that indicate potential secrets
SECRET_PATTERNS = [
    (r'sk-[a-zA-Z0-9]{32,}', "OpenAI API key"),
    (r'sk-ant-[a-zA-Z0-9-]{95,}', "Anthropic API key"),
    (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
    (r'OPENAI_API_KEY\s*=\s*["\']sk-', "Hardcoded OpenAI key in env assignment"),
    (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
    (r'token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "Hardcoded token"),
]

ALLOWED_FILES = {
    ".env.example",
    "SECURITY.md",
    "README.md",
    "scripts/check_secrets.py",
}


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Check a file for potential secrets.
    
    Returns:
        List of (line_number, pattern_name, matched_text) tuples
    """
    if filepath.name in ALLOWED_FILES:
        return []
    
    if not filepath.is_file():
        return []
    
    try:
        content = filepath.read_text()
    except (UnicodeDecodeError, PermissionError):
        return []
    
    findings = []
    for line_num, line in enumerate(content.splitlines(), 1):
        for pattern, name in SECRET_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                findings.append((line_num, name, line.strip()))
    
    return findings


def main():
    """Check all tracked files for secrets."""
    repo_root = Path(__file__).parent.parent
    
    # Check Python files and config files
    patterns = ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.md"]
    
    all_findings = []
    for pattern in patterns:
        for filepath in repo_root.glob(pattern):
            if ".git" in str(filepath) or "node_modules" in str(filepath):
                continue
            
            findings = check_file(filepath)
            if findings:
                all_findings.append((filepath, findings))
    
    if all_findings:
        print("🚨 POTENTIAL SECRETS DETECTED 🚨\n")
        for filepath, findings in all_findings:
            print(f"File: {filepath}")
            for line_num, pattern_name, line in findings:
                print(f"  Line {line_num}: {pattern_name}")
                print(f"    {line}")
            print()
        
        print("❌ Commit blocked. Remove secrets before committing.")
        print("💡 Use environment variables instead. See README.md for guidance.")
        return 1
    
    print("✅ No secrets detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())