"""Strategy scoring interfaces and plugin system (US5/FR-034, FR-040)."""

from qse.scorers.base import StrategyScorer, load_scorer_plugin

__all__ = ["StrategyScorer", "load_scorer_plugin"]
