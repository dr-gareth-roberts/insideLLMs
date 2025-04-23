class Model:
    """Base class for all language models."""
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, **kwargs):
        """Generate a response from the model given a prompt. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def chat(self, messages: list, **kwargs):
        """Engage in a multi-turn chat. Messages is a list of dicts with 'role' and 'content'."""
        raise NotImplementedError("Subclasses must implement this method.")

    def stream(self, prompt: str, **kwargs):
        """Stream the response from the model as it is generated (if supported)."""
        raise NotImplementedError("Subclasses must implement this method.")

    def info(self):
        """Return model metadata/info as a dict."""
        return {"name": self.name, "type": self.__class__.__name__}