import os
import anthropic
from .base import Model

class AnthropicModel(Model):
    """Model implementation for Anthropic's Claude models via API."""
    def __init__(self, name="AnthropicModel", model_name="claude-3-opus-20240229"):
        super().__init__(name)
        self.model_name = model_name
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def chat(self, messages: list, **kwargs):
        # Convert messages to Anthropic format if needed
        anthropic_messages = []
        for msg in messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            anthropic_messages.append({"role": role, "content": msg["content"]})
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            messages=anthropic_messages
        )
        return response.content[0].text

    def stream(self, prompt: str, **kwargs):
        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text

    def info(self):
        base_info = super().info()
        base_info.update({
            "provider": "Anthropic",
            "model_name": self.model_name,
            "description": "Anthropic Claude model via API. Requires ANTHROPIC_API_KEY env variable."
        })
        return base_info
