from .base import Probe
from .logic import LogicProbe
from .bias import BiasProbe
from .attack import AttackProbe
from .factuality import FactualityProbe

# Template for custom probes
class CustomProbe(Probe):
    """Template for creating custom probes."""
    def __init__(self, name="CustomProbe"):
        super().__init__(name)

    def run(self, model, *args, **kwargs):
        """Implement custom probe logic here."""
        pass

__all__ = [
    "Probe",
    "LogicProbe",
    "BiasProbe",
    "AttackProbe",
    "FactualityProbe",
    "CustomProbe",
]
