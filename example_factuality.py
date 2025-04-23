"""Example script to demonstrate the FactualityProbe and visualization tools."""
from insideLLMs.models import DummyModel, OpenAIModel, AnthropicModel
from insideLLMs.probes import FactualityProbe
from insideLLMs.visualization import plot_factuality_results, create_html_report
from insideLLMs.benchmark import ModelBenchmark
import os

def main():
    # Sample factual questions
    factual_questions = [
        {
            'question': 'What is the capital of France?',
            'reference_answer': 'Paris',
            'category': 'geography'
        },
        {
            'question': 'Who wrote "Romeo and Juliet"?',
            'reference_answer': 'William Shakespeare',
            'category': 'literature'
        },
        {
            'question': 'What is the chemical symbol for gold?',
            'reference_answer': 'Au',
            'category': 'science'
        },
        {
            'question': 'In what year did World War II end?',
            'reference_answer': '1945',
            'category': 'history'
        },
        {
            'question': 'What is the largest planet in our solar system?',
            'reference_answer': 'Jupiter',
            'category': 'science'
        }
    ]
    
    # Create probe
    factuality_probe = FactualityProbe()
    
    # Create models
    dummy_model = DummyModel()
    models = [dummy_model]
    
    # Add OpenAI model if API key is available
    if os.getenv("OPENAI_API_KEY"):
        openai_model = OpenAIModel(model_name="gpt-3.5-turbo")
        models.append(openai_model)
    else:
        print("[OpenAI model skipped: OPENAI_API_KEY not set]")
    
    # Add Anthropic model if API key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_model = AnthropicModel()
        models.append(anthropic_model)
    else:
        print("[Anthropic model skipped: ANTHROPIC_API_KEY not set]")
    
    # Run benchmark
    benchmark = ModelBenchmark(models, factuality_probe, "Factuality Benchmark")
    results = benchmark.run(factual_questions)
    
    # Save benchmark results
    benchmark.save_results("factuality_benchmark.json")
    
    # Compare models
    comparison = benchmark.compare_models()
    print("\nModel Comparison:")
    print(f"Fastest model: {comparison['rankings']['total_time'][0]}")
    print(f"Most reliable model: {comparison['rankings']['success_rate'][0]}")
    
    # Create visualization (if matplotlib is available)
    try:
        plot_factuality_results(results["models"][0]["results"], 
                               title="Factuality Results - " + models[0].name,
                               save_path="factuality_plot.png")
        print("\nVisualization saved to factuality_plot.png")
    except ImportError:
        print("\n[Visualization skipped: matplotlib/pandas not installed]")
    
    # Create HTML report
    report_path = create_html_report(results["models"][0]["results"], 
                                    title="Factuality Probe Report",
                                    save_path="factuality_report.html")
    print(f"HTML report saved to {report_path}")

if __name__ == "__main__":
    main()
