# insideLLMs Benchmarks

This directory contains benchmarks for measuring insideLLMs performance.

## Running Benchmarks

### Import Time Benchmark
Measures library import time and lazy loading effectiveness:

```bash
python benchmarks/benchmark_import.py
```

### Probe Execution Benchmark
Measures probe instantiation and execution overhead:

```bash
python benchmarks/benchmark_probes.py
```

## Interpreting Results

### Import Time
- **< 500ms**: Excellent - fast startup
- **500-1000ms**: Good - acceptable for most use cases
- **> 1000ms**: Needs improvement - heavy dependencies may be loading eagerly

### Lazy Loading
The import benchmark checks if heavy modules (transformers, torch, tensorflow)
are loaded during import. They should NOT be loaded until actually used.

### Probe Execution
Probe execution overhead should be minimal compared to actual model inference
time. The benchmarks use DummyModel to isolate the overhead of the probing
framework itself.

## Performance Tips

1. **Use lazy imports**: Don't import model implementations you don't need
2. **Batch operations**: Use ProbeRunner for multiple inputs
3. **Cache results**: Use the caching system for repeated queries
4. **Profile first**: Use the benchmarks to identify bottlenecks
