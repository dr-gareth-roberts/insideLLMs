"""Example script to demonstrate usage of DummyModel, OpenAIModel, and HuggingFaceModel."""
from insideLLMs.models import DummyModel, OpenAIModel, HuggingFaceModel
from insideLLMs.prompt_utils import PromptTemplate
import os

def main():
    prompt = "What is the capital of France?"
    print("--- DummyModel ---")
    dummy = DummyModel()
    print(dummy.generate(prompt))

    print("\n--- HuggingFaceModel (gpt2) ---")
    hf = HuggingFaceModel(model_name="gpt2")
    print(hf.generate(prompt, max_length=30))

    if os.getenv("OPENAI_API_KEY"):
        print("\n--- OpenAIModel (gpt-3.5-turbo) ---")
        openai_model = OpenAIModel(model_name="gpt-3.5-turbo")
        print(openai_model.generate(prompt))
    else:
        print("\n[OpenAIModel skipped: OPENAI_API_KEY not set]")

    print("\n--- PromptTemplate Example ---")
    template = PromptTemplate("Translate '{text}' to French.")
    print(template.format(text="Hello, world!"))

if __name__ == "__main__":
    main()
