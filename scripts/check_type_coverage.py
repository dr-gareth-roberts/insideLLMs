#!/usr/bin/env python3
"""Check mypy type coverage and fail if below threshold."""

import re
import sys
from pathlib import Path


def check_type_coverage(report_dir: str, threshold: float = 0.95) -> bool:
    """Check type coverage from mypy report.

    Args:
        report_dir: Directory containing mypy coverage report
        threshold: Minimum acceptable coverage (0.0-1.0)

    Returns:
        True if coverage meets threshold
    """
    report_path = Path(report_dir) / "index.txt"

    if not report_path.exists():
        print(f"Error: Report not found at {report_path}")
        return False

    with open(report_path) as f:
        content = f.read()

    match = re.search(r"Total coverage:\s*(\d+\.?\d*)%", content)

    if not match:
        print("Error: Could not parse coverage from report — failing as a precaution")
        return False

    coverage = float(match.group(1)) / 100.0

    print(f"Type coverage: {coverage * 100:.1f}%")
    print(f"Threshold: {threshold * 100:.1f}%")

    if coverage < threshold:
        print("❌ Type coverage below threshold!")
        return False

    print("✓ Type coverage meets threshold")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_type_coverage.py <report_dir> [threshold]")
        sys.exit(1)

    report_dir = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95

    success = check_type_coverage(report_dir, threshold)
    sys.exit(0 if success else 1)
