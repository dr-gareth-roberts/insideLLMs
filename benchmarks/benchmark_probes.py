"""Benchmark probe execution times.

This benchmark measures probe execution overhead
(separate from actual model inference time).
"""

import time
from typing import List


def benchmark_probe_creation():
    """Benchmark probe instantiation times."""
    from insideLLMs import LogicProbe, BiasProbe, AttackProbe, FactualityProbe

    probes = [
        ("LogicProbe", LogicProbe),
        ("BiasProbe", BiasProbe),
        ("AttackProbe", AttackProbe),
        ("FactualityProbe", FactualityProbe),
    ]

    print("Probe instantiation times:")
    for name, ProbeClass in probes:
        start = time.time()
        for _ in range(100):
            _ = ProbeClass()
        avg_time = (time.time() - start) * 1000 / 100
        print(f"  {name}: {avg_time:.3f}ms")


def benchmark_probe_execution():
    """Benchmark probe execution with DummyModel."""
    from insideLLMs import DummyModel, LogicProbe, BiasProbe

    model = DummyModel()
    test_data = [
        "What comes next: 1, 2, 3, ?",
        "Solve: 2x + 5 = 15",
        "If A implies B, and A is true, what is B?",
    ]

    probes = [
        ("LogicProbe", LogicProbe()),
        ("BiasProbe", BiasProbe()),
    ]

    print("\nProbe execution times (with DummyModel):")
    for name, probe in probes:
        start = time.time()
        for _ in range(10):
            for data in test_data:
                probe.run(model, data)
        avg_time = (time.time() - start) * 1000 / (10 * len(test_data))
        print(f"  {name}: {avg_time:.3f}ms per input")


def benchmark_runner():
    """Benchmark ProbeRunner execution."""
    from insideLLMs import DummyModel, LogicProbe, ProbeRunner

    model = DummyModel()
    probe = LogicProbe()
    runner = ProbeRunner(model, probe)

    # Generate test data
    test_data = [f"Question {i}: What is {i} + {i}?" for i in range(100)]

    print("\nProbeRunner execution times:")

    start = time.time()
    results = runner.run(test_data)
    total_time = (time.time() - start) * 1000
    per_item = total_time / len(test_data)

    print(f"  Total for {len(test_data)} items: {total_time:.2f}ms")
    print(f"  Average per item: {per_item:.3f}ms")
    print(f"  Throughput: {len(test_data) / (total_time / 1000):.0f} items/sec")


def main():
    print("=" * 60)
    print("insideLLMs Probe Benchmark")
    print("=" * 60)

    benchmark_probe_creation()
    benchmark_probe_execution()
    benchmark_runner()

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
