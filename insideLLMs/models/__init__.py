from .base import Model
from .openai import OpenAIModel
from .huggingface import HuggingFaceModel
from .anthropic import AnthropicModel

class DummyModel(Model):
    """A simple model for testing that echoes the prompt or returns canned responses."""
    def __init__(self, name="DummyModel"):
        super().__init__(name)

    def generate(self, prompt: str, **kwargs):
        return f"[DummyModel] You said: {prompt}"

    def chat(self, messages: list, **kwargs):
        last_message = messages[-1]['content'] if messages else ''
        return f"[DummyModel] Last message: {last_message}"

    def stream(self, prompt: str, **kwargs):
        for word in prompt.split():
            yield word

    def info(self):
        base_info = super().info()
        base_info.update({"description": "A dummy model for testing purposes."})
        return base_info
