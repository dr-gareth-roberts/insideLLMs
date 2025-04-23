"""Advanced prompt templating using Jinja2."""
from jinja2 import Template
from typing import Dict, Any, List

class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


def chain_prompts(prompts: List[str]) -> str:
    """Chain multiple prompts together into a single prompt."""
    return "\n".join(prompts)


def apply_prompt_template(template: str, variables: Dict[str, Any]) -> str:
    """Apply a template string with variables."""
    return template.format(**variables)


def render_jinja_template(template_str: str, variables: Dict[str, Any]) -> str:
    template = Template(template_str)
    return template.render(**variables)
