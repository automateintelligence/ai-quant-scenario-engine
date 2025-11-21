"""Strategy interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from qse.schema.signals import StrategySignals
from qse.schema.strategy import StrategyParams


class Strategy(ABC):
    """Base strategy for generating signals over price paths and features."""

    @abstractmethod
    def generate_signals(
        self,
        price_paths: np.ndarray,
        features,
        params: StrategyParams,
    ) -> StrategySignals:
        """Return strategy signals for stock and option legs."""
