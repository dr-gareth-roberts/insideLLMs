"""Example: Using Comprehensive Benchmark Datasets

This example demonstrates how to use the built-in benchmark datasets
for systematic LLM evaluation.
"""

from insideLLMs import (
    # Dataset utilities
    list_builtin_datasets,
    load_builtin_dataset,
    get_all_builtin_datasets,
    create_comprehensive_benchmark_suite,
    create_difficulty_stratified_suite,
    DatasetCategory,
    # Models
    DummyModel,
    # Probes
    LogicProbe,
    # Runner
    ProbeRunner,
)


def explore_datasets():
    """Explore available benchmark datasets."""
    print("=" * 60)
    print("Available Benchmark Datasets")
    print("=" * 60)

    for ds_info in list_builtin_datasets():
        print(f"\n{ds_info['name'].upper()}")
        print(f"  Category: {ds_info['category']}")
        print(f"  Examples: {ds_info['num_examples']}")
        print(f"  Description: {ds_info['description']}")
        print(f"  Difficulties: {ds_info['difficulties']}")


def load_and_use_dataset():
    """Load and use a specific dataset."""
    print("\n" + "=" * 60)
    print("Using the Coding Dataset")
    print("=" * 60)

    # Load the coding dataset
    coding = load_builtin_dataset("coding")
    print(f"\nLoaded {len(coding)} examples")

    # Get statistics
    stats = coding.get_stats()
    print(f"Categories: {stats.categories}")
    print(f"Difficulties: {stats.difficulties}")

    # Sample some examples
    print("\nSample examples:")
    for i, example in enumerate(coding.sample(3, seed=42)):
        print(f"\n{i+1}. [{example.difficulty}] {example.category}")
        print(f"   Input: {example.input_text[:80]}...")
        if example.expected_output:
            print(f"   Expected: {example.expected_output[:50]}...")


def create_custom_suite():
    """Create a custom benchmark suite."""
    print("\n" + "=" * 60)
    print("Creating Custom Benchmark Suite")
    print("=" * 60)

    # Create a reasoning-focused suite
    reasoning_suite = create_comprehensive_benchmark_suite(
        categories=[DatasetCategory.REASONING, DatasetCategory.MATH],
        max_examples_per_dataset=5,
        seed=42,
    )
    print(f"\nReasoning suite: {len(reasoning_suite)} examples")

    # Create an easy-only suite
    easy_suite = create_difficulty_stratified_suite(
        difficulty="easy",
        max_total_examples=20,
        seed=42,
    )
    print(f"Easy difficulty suite: {len(easy_suite)} examples")

    # Show some examples from the reasoning suite
    print("\nSample from reasoning suite:")
    for example in reasoning_suite.sample(3, seed=42):
        source = example.metadata.get("source_dataset", "unknown")
        print(f"  [{source}] {example.input_text[:60]}...")


def run_with_dummy_model():
    """Run benchmark with a dummy model for testing."""
    print("\n" + "=" * 60)
    print("Running Benchmark with DummyModel")
    print("=" * 60)

    # Load a small dataset
    reasoning = load_builtin_dataset("reasoning")

    # Use DummyModel (echoes input, good for testing)
    model = DummyModel()
    probe = LogicProbe()
    runner = ProbeRunner(model, probe)

    # Run on a few examples
    inputs = [ex.input_text for ex in reasoning.sample(3, seed=42)]
    results = runner.run(inputs)

    print(f"\nRan {len(results)} examples")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Status: {result.status}")
        print(f"   Input: {result.input[:50]}...")
        print(f"   Output: {result.output[:50] if result.output else 'N/A'}...")


def cross_validation_example():
    """Demonstrate cross-validation with datasets."""
    print("\n" + "=" * 60)
    print("Cross-Validation Example")
    print("=" * 60)

    from insideLLMs import cross_validate

    # Load dataset
    math_ds = load_builtin_dataset("math")

    # Perform 3-fold cross-validation
    print("\n3-fold cross-validation:")
    for fold_idx, (train, test) in enumerate(cross_validate(math_ds, n_folds=3, seed=42)):
        print(f"  Fold {fold_idx + 1}: {len(train)} train, {len(test)} test")


def filtering_example():
    """Demonstrate dataset filtering."""
    print("\n" + "=" * 60)
    print("Dataset Filtering Example")
    print("=" * 60)

    from insideLLMs import filter_dataset

    # Load dataset
    language = load_builtin_dataset("language")
    print(f"Original dataset: {len(language)} examples")

    # Filter by difficulty
    easy_only = filter_dataset(language, difficulty="easy")
    print(f"Easy only: {len(easy_only)} examples")

    # Filter by category
    sentiment = filter_dataset(language, category="sentiment")
    print(f"Sentiment only: {len(sentiment)} examples")


if __name__ == "__main__":
    explore_datasets()
    load_and_use_dataset()
    create_custom_suite()
    run_with_dummy_model()
    cross_validation_example()
    filtering_example()

    print("\n" + "=" * 60)
    print("Done! See the README for more information.")
    print("=" * 60)
