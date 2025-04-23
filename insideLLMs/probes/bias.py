from .base import Probe

class BiasProbe(Probe):
    """Probe to test LLMs' propensity for bias."""
    def __init__(self, name="BiasProbe"):
        super().__init__(name)

    def run(self, model, prompt_pairs: list, **kwargs):
        """Run the bias probe on the given model with pairs of prompts to compare for bias."""
        results = []
        for prompt1, prompt2 in prompt_pairs:
            response1 = model.generate(prompt1, **kwargs)
            response2 = model.generate(prompt2, **kwargs)
            results.append((response1, response2))
        return results
