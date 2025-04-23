import os
import openai
from .base import Model

class OpenAIModel(Model):
    """Model implementation for OpenAI's GPT models via API."""
    def __init__(self, name="OpenAIModel", model_name="gpt-3.5-turbo"):
        super().__init__(name)
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        openai.api_key = self.api_key

    def generate(self, prompt: str, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message['content']

    def chat(self, messages: list, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message['content']

    def stream(self, prompt: str, **kwargs):
        stream = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if 'choices' in chunk and chunk.choices[0].delta.get('content'):
                yield chunk.choices[0].delta['content']

    def info(self):
        base_info = super().info()
        base_info.update({
            "provider": "OpenAI",
            "model_name": self.model_name,
            "description": "OpenAI GPT model via API. Requires OPENAI_API_KEY env variable."
        })
        return base_info
