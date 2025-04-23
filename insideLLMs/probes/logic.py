from .base import Probe

class LogicProbe(Probe):
    """Probe to test LLMs' zero-shot ability at unseen logic problems."""
    def __init__(self, name="LogicProbe"):
        super().__init__(name)

    def run(self, model, logic_problem: str, **kwargs):
        """Run the logic probe on the given model with a logic problem."""
        prompt = f"Solve this logic problem: {logic_problem}"
        return model.generate(prompt, **kwargs)
