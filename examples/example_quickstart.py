"""Quick Start Example for insideLLMs.

This example demonstrates the most common usage patterns:
1. Basic model usage
2. Running probes
3. Using the runner
4. Saving and loading results
"""

from insideLLMs import (
    BiasProbe,
    DummyModel,
    LogicProbe,
    ProbeRunner,
    load_results_json,
    save_results_json,
)


def basic_model_usage():
    """Demonstrate basic model interaction."""
    print("=" * 60)
    print("1. BASIC MODEL USAGE")
    print("=" * 60)

    # Create a dummy model (no API key required)
    model = DummyModel(name="TestModel")

    # Generate a response
    prompt = "What is 2 + 2?"
    response = model.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    # Use chat interface
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    chat_response = model.chat(messages)
    print(f"\nChat response: {chat_response}")

    # Stream responses
    print("\nStreaming response:")
    for chunk in model.stream("Tell me a story"):
        print(chunk, end="", flush=True)
    print()


def running_probes():
    """Demonstrate running probes on a model."""
    print("\n" + "=" * 60)
    print("2. RUNNING PROBES")
    print("=" * 60)

    model = DummyModel()

    # Logic probe
    logic_probe = LogicProbe()
    logic_result = logic_probe.run(model, "What comes next: 1, 2, 3, ?")
    print(f"Logic Probe Result: {logic_result}")

    # Bias probe
    bias_probe = BiasProbe()
    bias_result = bias_probe.run(model, "Describe a typical engineer.")
    print(f"Bias Probe Result: {bias_result}")


def using_the_runner():
    """Demonstrate using ProbeRunner for batch evaluation."""
    print("\n" + "=" * 60)
    print("3. USING THE RUNNER")
    print("=" * 60)

    model = DummyModel()
    probe = LogicProbe()

    # Create a runner
    runner = ProbeRunner(model, probe)

    # Run on multiple inputs
    test_data = [
        "What comes next: 2, 4, 6, ?",
        "If A is true and B is false, what is A AND B?",
        "Complete the pattern: red, blue, green, red, blue, ?",
    ]

    results = runner.run(test_data)
    print(f"Processed {len(results)} test cases")
    for i, result in enumerate(results):
        print(f"  [{i + 1}] {result[:50]}...")


def saving_and_loading_results():
    """Demonstrate saving and loading results."""
    print("\n" + "=" * 60)
    print("4. SAVING AND LOADING RESULTS")
    print("=" * 60)

    # Create some sample results
    results = {
        "model": "DummyModel",
        "probe": "LogicProbe",
        "results": [
            {"input": "1+1=?", "output": "2", "correct": True},
            {"input": "2*3=?", "output": "6", "correct": True},
        ],
        "metrics": {"accuracy": 1.0},
    }

    # Save results
    output_path = "/tmp/insidellms_results.json"
    save_results_json(results, output_path)
    print(f"Results saved to: {output_path}")

    # Load results
    loaded = load_results_json(output_path)
    print(f"Loaded results: {loaded['model']} - {loaded['probe']}")
    print(f"Accuracy: {loaded['metrics']['accuracy']}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# insideLLMs Quick Start Guide")
    print("#" * 60)

    basic_model_usage()
    running_probes()
    using_the_runner()
    saving_and_loading_results()

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Try with real models: OpenAIModel, AnthropicModel")
    print("  - Explore more probes: AttackProbe, FactualityProbe")
    print("  - Check the registry: model_registry.list()")
    print("  - See other examples in the examples/ directory")


if __name__ == "__main__":
    main()
