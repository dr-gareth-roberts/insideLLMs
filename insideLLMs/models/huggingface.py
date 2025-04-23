from .base import Model
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class HuggingFaceModel(Model):
    """Model implementation for HuggingFace Transformers models."""
    def __init__(self, name="HuggingFaceModel", model_name="gpt2", device=-1):
        super().__init__(name)
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=device)

    def generate(self, prompt: str, **kwargs):
        outputs = self.generator(prompt, **kwargs)
        return outputs[0]["generated_text"]

    def chat(self, messages: list, **kwargs):
        # Simple chat: concatenate messages
        prompt = "\n".join([m["content"] for m in messages])
        outputs = self.generator(prompt, **kwargs)
        return outputs[0]["generated_text"]

    def stream(self, prompt: str, **kwargs):
        # Streaming not natively supported; yield the full output as one chunk
        outputs = self.generator(prompt, **kwargs)
        yield outputs[0]["generated_text"]

    def info(self):
        base_info = super().info()
        base_info.update({
            "provider": "HuggingFace",
            "model_name": self.model_name,
            "description": "HuggingFace Transformers model via pipeline."
        })
        return base_info
