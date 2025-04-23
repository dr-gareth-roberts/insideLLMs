class Probe:
    """Base class for all probes."""
    def __init__(self, name: str):
        self.name = name

    def run(self, model, *args, **kwargs):
        """Run the probe on the given model. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")