"""Option pricer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from quant_scenario_engine.models.options import OptionSpec


class OptionPricer(ABC):
    """Base option pricer interface."""

    @abstractmethod
    def price(self, path_slice: np.ndarray, option_spec: OptionSpec) -> np.ndarray:
        """Return mark-to-market price series for the provided path slice."""

