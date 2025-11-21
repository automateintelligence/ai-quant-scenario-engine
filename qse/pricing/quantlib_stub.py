"""QuantLib pricer stub (placeholder for future extension)."""

from __future__ import annotations

import numpy as np

from qse.exceptions import PricingError
from qse.interfaces.pricing import OptionPricer
from qse.models.options import OptionSpec


class QuantLibPricer(OptionPricer):
    def __init__(self) -> None:
        # No heavy dependency at MVP time; stub raises until implemented
        pass

    def price(self, path_slice: np.ndarray, option_spec: OptionSpec) -> np.ndarray:  # pragma: no cover - stub
        raise PricingError("QuantLibPricer is a stub; provide a supported pricer")
