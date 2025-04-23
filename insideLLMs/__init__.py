"""Plugin registry for models, probes, and datasets."""
class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name, obj):
        self._registry[name] = obj

    def get(self, name):
        return self._registry[name]

    def list(self):
        return list(self._registry.keys())

model_registry = Registry()
probe_registry = Registry()
dataset_registry = Registry()
