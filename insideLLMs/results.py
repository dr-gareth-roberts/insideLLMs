"""Results aggregation and Markdown/HTML reporting."""
from typing import List, Dict, Any
import json

def save_results_json(results: List[Dict[str, Any]], path: str):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

def results_to_markdown(results: List[Dict[str, Any]]) -> str:
    md = "| Input | Output | Error |\n|---|---|---|\n"
    for r in results:
        input_str = str(r.get('input', ''))
        output_str = str(r.get('output', ''))
        error_str = str(r.get('error', ''))
        md += f"| {input_str} | {output_str} | {error_str} |\n"
    return md

def save_results_markdown(results: List[Dict[str, Any]], path: str):
    md = results_to_markdown(results)
    with open(path, 'w') as f:
        f.write(md)
