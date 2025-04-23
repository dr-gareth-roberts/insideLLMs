"""Benchmarking tools for comparing models."""
from typing import List, Dict, Any, Optional, Union
import time
import json
import os
from insideLLMs.models import Model
from insideLLMs.probes import Probe
from insideLLMs.runner import run_probe

class ModelBenchmark:
    """Benchmark multiple models on the same probe and dataset."""
    def __init__(self, models: List[Model], probe: Probe, name: str = "Model Benchmark"):
        self.models = models
        self.probe = probe
        self.name = name
        self.results = {}
        
    def run(self, prompt_set: List[Any], **probe_kwargs) -> Dict[str, Any]:
        """Run the benchmark on all models.
        
        Args:
            prompt_set: The dataset to use for benchmarking
            **probe_kwargs: Additional arguments to pass to the probe
            
        Returns:
            Dictionary with benchmark results
        """
        benchmark_results = {
            "name": self.name,
            "probe": self.probe.name,
            "models": [],
            "timestamp": time.time()
        }
        
        for model in self.models:
            model_info = model.info()
            print(f"Benchmarking {model_info['name']}...")
            
            start_time = time.time()
            results = run_probe(model, self.probe, prompt_set, **probe_kwargs)
            end_time = time.time()
            
            model_result = {
                "model": model_info,
                "results": results,
                "metrics": {
                    "total_time": end_time - start_time,
                    "avg_time_per_item": (end_time - start_time) / len(prompt_set) if prompt_set else 0,
                    "total_items": len(prompt_set),
                    "success_rate": sum(1 for r in results if 'error' not in r) / len(results) if results else 0
                }
            }
            
            benchmark_results["models"].append(model_result)
            
        self.results = benchmark_results
        return benchmark_results
    
    def save_results(self, path: str):
        """Save benchmark results to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare models based on benchmark results.
        
        Returns:
            Dictionary with comparison metrics
        """
        if not self.results or "models" not in self.results:
            raise ValueError("No benchmark results available. Run the benchmark first.")
        
        comparison = {
            "name": self.name,
            "metrics": {},
            "rankings": {}
        }
        
        # Extract metrics for comparison
        model_metrics = {}
        for model_result in self.results["models"]:
            model_name = model_result["model"]["name"]
            model_metrics[model_name] = model_result["metrics"]
        
        # Compare total time
        total_times = {name: metrics["total_time"] for name, metrics in model_metrics.items()}
        comparison["metrics"]["total_time"] = total_times
        comparison["rankings"]["total_time"] = sorted(total_times.keys(), key=lambda x: total_times[x])
        
        # Compare average time per item
        avg_times = {name: metrics["avg_time_per_item"] for name, metrics in model_metrics.items()}
        comparison["metrics"]["avg_time_per_item"] = avg_times
        comparison["rankings"]["avg_time_per_item"] = sorted(avg_times.keys(), key=lambda x: avg_times[x])
        
        # Compare success rate
        success_rates = {name: metrics["success_rate"] for name, metrics in model_metrics.items()}
        comparison["metrics"]["success_rate"] = success_rates
        comparison["rankings"]["success_rate"] = sorted(success_rates.keys(), key=lambda x: success_rates[x], reverse=True)
        
        return comparison

class ProbeBenchmark:
    """Benchmark multiple probes on the same model and dataset."""
    def __init__(self, model: Model, probes: List[Probe], name: str = "Probe Benchmark"):
        self.model = model
        self.probes = probes
        self.name = name
        self.results = {}
        
    def run(self, prompt_set: List[Any], **kwargs) -> Dict[str, Any]:
        """Run the benchmark on all probes.
        
        Args:
            prompt_set: The dataset to use for benchmarking
            **kwargs: Additional arguments to pass to the probes
            
        Returns:
            Dictionary with benchmark results
        """
        benchmark_results = {
            "name": self.name,
            "model": self.model.info(),
            "probes": [],
            "timestamp": time.time()
        }
        
        for probe in self.probes:
            print(f"Benchmarking {probe.name}...")
            
            start_time = time.time()
            results = run_probe(self.model, probe, prompt_set, **kwargs)
            end_time = time.time()
            
            probe_result = {
                "probe": probe.name,
                "results": results,
                "metrics": {
                    "total_time": end_time - start_time,
                    "avg_time_per_item": (end_time - start_time) / len(prompt_set) if prompt_set else 0,
                    "total_items": len(prompt_set),
                    "success_rate": sum(1 for r in results if 'error' not in r) / len(results) if results else 0
                }
            }
            
            benchmark_results["probes"].append(probe_result)
            
        self.results = benchmark_results
        return benchmark_results
    
    def save_results(self, path: str):
        """Save benchmark results to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
