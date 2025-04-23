"""Visualization tools for probe results."""
from typing import List, Dict, Any, Optional
import json
import os

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

def check_visualization_deps():
    """Check if visualization dependencies are available."""
    if not VISUALIZATION_AVAILABLE:
        raise ImportError(
            "Visualization dependencies not installed. "
            "Please install with: pip install matplotlib pandas seaborn"
        )

def plot_bias_results(results: List[Dict[str, Any]], 
                      title: str = "Bias Probe Results",
                      save_path: Optional[str] = None):
    """Plot results from BiasProbe.
    
    Args:
        results: List of results from BiasProbe
        title: Plot title
        save_path: Path to save the plot (if None, plot is displayed)
    """
    check_visualization_deps()
    
    # Extract data from results
    data = []
    for i, result in enumerate(results):
        for j, (response1, response2) in enumerate(result.get('output', [])):
            data.append({
                'Prompt Pair': f"Pair {i+1}.{j+1}",
                'Response 1 Length': len(response1),
                'Response 2 Length': len(response2),
                'Length Difference': len(response1) - len(response2)
            })
    
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Prompt Pair', y='Length Difference', data=df)
    plt.title(title)
    plt.xlabel('Prompt Pairs')
    plt.ylabel('Response Length Difference')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_factuality_results(results: List[Dict[str, Any]], 
                           title: str = "Factuality Probe Results",
                           save_path: Optional[str] = None):
    """Plot results from FactualityProbe.
    
    Args:
        results: List of results from FactualityProbe
        title: Plot title
        save_path: Path to save the plot (if None, plot is displayed)
    """
    check_visualization_deps()
    
    # Extract data from results
    categories = {}
    for result in results:
        for item in result.get('output', []):
            category = item.get('category', 'general')
            if category not in categories:
                categories[category] = {'total': 0, 'questions': []}
            categories[category]['total'] += 1
            categories[category]['questions'].append(item)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot category distribution
    category_names = list(categories.keys())
    category_counts = [categories[cat]['total'] for cat in category_names]
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=category_names, y=category_counts)
    plt.title(f"{title} - Categories")
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Plot response lengths by category
    plt.subplot(1, 2, 2)
    response_data = []
    for cat, data in categories.items():
        for q in data['questions']:
            response_data.append({
                'Category': cat,
                'Response Length': len(q.get('model_answer', ''))
            })
    
    df = pd.DataFrame(response_data)
    sns.boxplot(x='Category', y='Response Length', data=df)
    plt.title(f"{title} - Response Lengths")
    plt.xlabel('Category')
    plt.ylabel('Response Length')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def create_html_report(results: List[Dict[str, Any]], 
                      title: str = "Probe Results Report",
                      save_path: str = "report.html"):
    """Create an HTML report from probe results.
    
    Args:
        results: List of results from any probe
        title: Report title
        save_path: Path to save the HTML report
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .result {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .input {{ color: #0066cc; }}
            .output {{ color: #009900; }}
            .error {{ color: #cc0000; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
    """
    
    # Add summary
    html += f"<h2>Summary</h2>"
    html += f"<p>Total results: {len(results)}</p>"
    
    # Add results
    html += f"<h2>Results</h2>"
    for i, result in enumerate(results):
        html += f"<div class='result'>"
        html += f"<h3>Result {i+1}</h3>"
        
        # Input
        if 'input' in result:
            html += f"<div class='input'><strong>Input:</strong> {result['input']}</div>"
        
        # Output
        if 'output' in result:
            output = result['output']
            html += f"<div class='output'><strong>Output:</strong></div>"
            
            # Handle different output formats
            if isinstance(output, list):
                html += "<table><tr><th>#</th><th>Output</th></tr>"
                for j, item in enumerate(output):
                    if isinstance(item, tuple):
                        html += f"<tr><td>{j+1}</td><td>{item[0]}<br><em>vs.</em><br>{item[1]}</td></tr>"
                    else:
                        html += f"<tr><td>{j+1}</td><td>{item}</td></tr>"
                html += "</table>"
            elif isinstance(output, dict):
                html += "<table><tr><th>Key</th><th>Value</th></tr>"
                for key, value in output.items():
                    html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                html += "</table>"
            else:
                html += f"<p>{output}</p>"
        
        # Error
        if 'error' in result:
            html += f"<div class='error'><strong>Error:</strong> {result['error']}</div>"
        
        html += "</div>"
    
    html += """
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html)
    
    return save_path
