"""Experiment runner with YAML/JSON config support."""
import yaml
import json
from insideLLMs.models import DummyModel, OpenAIModel, HuggingFaceModel, AnthropicModel
from insideLLMs.probes import LogicProbe, BiasProbe, AttackProbe, FactualityProbe
from insideLLMs.runner import run_probe
from insideLLMs.dataset_utils import load_csv_dataset, load_jsonl_dataset, load_hf_dataset
from insideLLMs.models import Model
from insideLLMs.probes import Probe
from typing import List, Dict, Any

class ProbeRunner:
    def __init__(self, model: Model, probe: Probe):
        self.model = model
        self.probe = probe

    def run(self, prompt_set: List[Any], **probe_kwargs) -> List[Dict[str, Any]]:
        """Run the probe on the model for each item in the prompt set."""
        results = []
        for item in prompt_set:
            try:
                output = self.probe.run(self.model, item, **probe_kwargs)
                results.append({
                    "input": item,
                    "output": output
                })
            except Exception as e:
                results.append({
                    "input": item,
                    "error": str(e)
                })
        return results

def run_probe(model: Model, probe: Probe, prompt_set: List[Any], **probe_kwargs) -> List[Dict[str, Any]]:
    runner = ProbeRunner(model, probe)
    return runner.run(prompt_set, **probe_kwargs)

def load_config(path: str):
    if path.endswith('.yaml') or path.endswith('.yml'):
        with open(path) as f:
            return yaml.safe_load(f)
    elif path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config file format.")

def run_experiment_from_config(config_path: str):
    config = load_config(config_path)
    # Model selection
    model_type = config['model']['type']
    model_args = config['model'].get('args', {})
    if model_type == 'dummy':
        model = DummyModel(**model_args)
    elif model_type == 'openai':
        model = OpenAIModel(**model_args)
    elif model_type == 'huggingface':
        model = HuggingFaceModel(**model_args)
    elif model_type == 'anthropic':
        model = AnthropicModel(**model_args)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    # Probe selection
    probe_type = config['probe']['type']
    probe_args = config['probe'].get('args', {})
    if probe_type == 'logic':
        probe = LogicProbe(**probe_args)
    elif probe_type == 'bias':
        probe = BiasProbe(**probe_args)
    elif probe_type == 'attack':
        probe = AttackProbe(**probe_args)
    elif probe_type == 'factuality':
        probe = FactualityProbe(**probe_args)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    # Dataset loading
    dataset_cfg = config['dataset']
    if dataset_cfg['format'] == 'csv':
        prompt_set = load_csv_dataset(dataset_cfg['path'])
    elif dataset_cfg['format'] == 'jsonl':
        prompt_set = load_jsonl_dataset(dataset_cfg['path'])
    elif dataset_cfg['format'] == 'hf':
        prompt_set = load_hf_dataset(dataset_cfg['name'], split=dataset_cfg.get('split', 'test'))
    else:
        raise ValueError(f"Unknown dataset format: {dataset_cfg['format']}")
    # Run probe
    results = run_probe(model, probe, prompt_set)
    return results
