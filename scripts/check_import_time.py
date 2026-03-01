#!/usr/bin/env python3
"""Check that importing insideLLMs is fast (lazy loading verification).

This script verifies the lazy-loading architecture by timing the import
and failing if it exceeds a threshold. Run as part of CI to prevent
accidental import-time regressions.

Usage:
    python scripts/check_import_time.py [--threshold-ms 200]
"""

import argparse
import subprocess
import sys


def measure_import_time() -> float:
    """Measure import time of insideLLMs in milliseconds."""
    code = (
        "import time; "
        "start = time.perf_counter(); "
        "import insideLLMs; "
        "elapsed_ms = (time.perf_counter() - start) * 1000; "
        "print(f'{elapsed_ms:.1f}')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"Import failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check insideLLMs import time.")
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=200.0,
        help="Maximum allowed import time in milliseconds (default: 200)",
    )
    args = parser.parse_args()

    # Take the median of 3 runs to reduce noise
    times = [measure_import_time() for _ in range(3)]
    times.sort()
    median_ms = times[1]

    print(f"Import times: {', '.join(f'{t:.1f}ms' for t in times)}")
    print(f"Median: {median_ms:.1f}ms (threshold: {args.threshold_ms:.0f}ms)")

    if median_ms > args.threshold_ms:
        print(
            f"\n❌ FAIL: Import time ({median_ms:.1f}ms) exceeds "
            f"threshold ({args.threshold_ms:.0f}ms).",
            file=sys.stderr,
        )
        print(
            "This may indicate heavy dependencies being imported eagerly. "
            "Use `python -X importtime -c 'import insideLLMs'` to profile.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("\n✅ PASS: Import time is within threshold.")


if __name__ == "__main__":
    main()
