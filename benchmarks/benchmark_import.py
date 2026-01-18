"""Benchmark import times for insideLLMs.

This benchmark measures:
1. Cold import time (fresh Python process)
2. Warm import time (module already cached)
3. Lazy loading effectiveness
"""

import subprocess
import sys
import time


def measure_cold_import():
    """Measure cold import time in a fresh Python process."""
    code = """
import sys
sys.path.insert(0, '.')
import time
start = time.time()
import insideLLMs
end = time.time()
print(f"{(end-start)*1000:.2f}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def measure_warm_import():
    """Measure warm import time (module cached)."""
    # First import to warm cache
    # Measure reimport
    import importlib

    import insideLLMs

    start = time.time()
    importlib.reload(insideLLMs)
    end = time.time()
    return (end - start) * 1000


def check_lazy_loading():
    """Check if heavy modules are lazily loaded."""
    code = """
import sys
sys.path.insert(0, '.')
import insideLLMs

heavy = ['transformers', 'torch', 'tensorflow']
loaded = [h for h in heavy if h in sys.modules]
print(",".join(loaded) if loaded else "none")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main():
    print("=" * 60)
    print("insideLLMs Import Benchmark")
    print("=" * 60)

    # Measure cold imports (run 3 times)
    print("\nCold import times (fresh Python process):")
    cold_times = []
    for i in range(3):
        t = measure_cold_import()
        cold_times.append(t)
        print(f"  Run {i + 1}: {t:.2f}ms")
    print(f"  Average: {sum(cold_times) / len(cold_times):.2f}ms")

    # Check lazy loading
    print("\nLazy loading check:")
    loaded = check_lazy_loading()
    if loaded == "none":
        print("  Heavy modules NOT loaded on import (good!)")
    else:
        print(f"  Warning: Heavy modules loaded: {loaded}")

    # Measure registry operations
    print("\nRegistry operation times:")
    import insideLLMs

    start = time.time()
    _ = insideLLMs.model_registry.list()
    list_time = (time.time() - start) * 1000
    print(f"  model_registry.list(): {list_time:.2f}ms")

    start = time.time()
    _ = insideLLMs.model_registry.get("dummy")
    get_time = (time.time() - start) * 1000
    print(f"  model_registry.get('dummy'): {get_time:.2f}ms")

    # Model operations
    print("\nModel operation times:")
    model = insideLLMs.DummyModel()

    start = time.time()
    for _ in range(100):
        model.generate("test prompt")
    gen_time = (time.time() - start) * 1000 / 100
    print(f"  DummyModel.generate() avg: {gen_time:.2f}ms")

    start = time.time()
    for _ in range(100):
        list(model.stream("test prompt"))
    stream_time = (time.time() - start) * 1000 / 100
    print(f"  DummyModel.stream() avg: {stream_time:.2f}ms")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    # Summary
    avg_import = sum(cold_times) / len(cold_times)
    if avg_import < 500:
        print(f"\nImport time: {avg_import:.0f}ms - EXCELLENT")
    elif avg_import < 1000:
        print(f"\nImport time: {avg_import:.0f}ms - GOOD")
    else:
        print(f"\nImport time: {avg_import:.0f}ms - NEEDS IMPROVEMENT")


if __name__ == "__main__":
    main()
