"""Shared interfaces for cross-feature compatibility.

This package defines abstract base classes and entities shared between:
- 001-mvp-pipeline: Core backtesting framework
- 009-option-optimizer: Option strategy optimizer

Architecture:
- distribution.py: ReturnDistribution (supports both fit/sample and generate_paths)
- pricing.py: OptionPricer + Greeks (Black-Scholes, Bjerksund-Stensland, future Heston)
- option_leg.py: OptionLeg (harmonizes single-leg and multi-leg representations)
- regime.py: Regime (009-specific, imported by 001 in US7 Phase 2)
- strategy.py: Strategy interface (001-specific currently)
- candidate_selector.py: CandidateSelector interface (001-specific currently)
"""

from qse.interfaces.distribution import (
    DistributionMetadata,
    Estimator,
    FitStatus,
    ReturnDistribution,
)
from qse.interfaces.option_leg import OptionLeg, OptionType, Side
from qse.interfaces.pricing import Greeks, OptionPricer
from qse.interfaces.regime import Regime, RegimeLabel, RegimeMode, RegimeSource

__all__ = [
    # Distribution interfaces
    "ReturnDistribution",
    "DistributionMetadata",
    "Estimator",
    "FitStatus",
    # Pricing interfaces
    "OptionPricer",
    "Greeks",
    # Option leg representation
    "OptionLeg",
    "OptionType",
    "Side",
    # Regime (009-specific)
    "Regime",
    "RegimeLabel",
    "RegimeMode",
    "RegimeSource",
]
