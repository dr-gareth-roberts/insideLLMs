from .base import Probe

class AttackProbe(Probe):
    """Probe to test LLMs' vulnerabilities to attacks (recursion, reframing, tokenization)."""
    def __init__(self, name="AttackProbe"):
        super().__init__(name)

    def run(self, model, attack_prompt: str, **kwargs):
        """Run the attack probe on the given model with an attack prompt."""
        prompt = f"{attack_prompt}"
        return model.generate(prompt, **kwargs)
