#!/usr/bin/env python3
"""Pre-commit hook to detect accidentally committed secrets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Patterns that indicate potential secrets
SECRET_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{32,}", "OpenAI API key"),
    (r"sk-ant-[a-zA-Z0-9-]{95,}", "Anthropic API key"),
    (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
    (r'OPENAI_API_KEY\s*=\s*["\']sk-', "Hardcoded OpenAI key in env assignment"),
    (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
    (r'token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "Hardcoded token"),
]

DEFAULT_GLOBS = ("**/*.py", "**/*.yaml", "**/*.yml", "**/*.json")
STRICT_EXTRA_GLOBS = ("**/*.md",)

ALLOWED_PATHS = {
    Path(".env.example"),
    Path("SECURITY.md"),
    Path("README.md"),
    Path("scripts/check_secrets.py"),
}

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "site-packages",
}

EXCLUDED_ROOT_DIRS = {
    "tests",
    "docs",
    "wiki",
    "examples",
}

EXAMPLE_LINE_PREFIXES = ("#", ">>>", "...")
EXAMPLE_LINE_MARKERS = (
    "```",
    "As a parameter:",
)

PLACEHOLDER_VALUE_FRAGMENTS = (
    "...",
    "your-",
    "your_",
    "example",
    "test",
    "dummy",
    "invalid",
    "secret123",
)


def _should_skip_file(filepath: Path, repo_root: Path, *, strict: bool) -> bool:
    rel = filepath.relative_to(repo_root)
    if rel in ALLOWED_PATHS:
        return True
    if any(part in EXCLUDED_DIRS for part in rel.parts):
        return True
    if not strict and rel.parts and rel.parts[0] in EXCLUDED_ROOT_DIRS:
        return True
    return False


def _candidate_files(repo_root: Path, *, strict: bool) -> list[Path]:
    patterns = list(DEFAULT_GLOBS)
    if strict:
        patterns.extend(STRICT_EXTRA_GLOBS)

    files: set[Path] = set()
    for pattern in patterns:
        for path in repo_root.glob(pattern):
            if path.is_file() and not _should_skip_file(path, repo_root, strict=strict):
                files.add(path)
    return sorted(files)


def _is_example_line(line: str) -> bool:
    stripped = line.strip()
    if any(stripped.startswith(prefix) for prefix in EXAMPLE_LINE_PREFIXES):
        return True
    return any(marker in line for marker in EXAMPLE_LINE_MARKERS)


def _extract_assigned_value(line: str) -> str | None:
    match = re.search(r'=\s*["\']([^"\']+)["\']', line)
    if not match:
        return None
    return match.group(1).strip()


def _is_placeholder_assignment(line: str) -> bool:
    value = _extract_assigned_value(line)
    if not value:
        return False
    lower = value.lower()
    return any(fragment in lower for fragment in PLACEHOLDER_VALUE_FRAGMENTS)


def _is_false_positive(line: str, pattern_name: str, *, strict: bool) -> bool:
    if strict:
        return False
    if _is_example_line(line):
        return True
    if pattern_name in {"Hardcoded API key", "Hardcoded password", "Hardcoded token"}:
        return _is_placeholder_assignment(line)
    return False


def check_file(filepath: Path, *, strict: bool) -> list[tuple[int, str, str]]:
    """Check a file for potential secrets.

    Returns:
        List of (line_number, pattern_name, matched_text) tuples
    """
    if not filepath.is_file():
        return []

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except (UnicodeDecodeError, PermissionError):
        return []

    findings = []
    for line_num, line in enumerate(content.splitlines(), 1):
        for pattern, name in SECRET_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                if _is_false_positive(line, name, strict=strict):
                    continue
                findings.append((line_num, name, line.strip()))

    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Also scan docs/tests/examples/wiki files. Default mode scans source/config files "
            "and excludes directories that produce high false-positive rates."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check all tracked files for secrets."""
    args = parse_args(argv)
    repo_root = Path(__file__).parent.parent

    all_findings = []
    for filepath in _candidate_files(repo_root, strict=args.strict):
        findings = check_file(filepath, strict=args.strict)
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
    sys.exit(main(sys.argv[1:]))
