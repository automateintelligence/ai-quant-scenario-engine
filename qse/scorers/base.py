"""StrategyScorer interface and plugin loader (US5/T026, FR-034, FR-040)."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class StrategyScorer(ABC):
    """
    Pluggable interface for scoring option strategy candidates.

    Implementations compute a composite score in [0, 1] range from candidate metrics,
    allowing users to rank strategies by different criteria (risk-adjusted return,
    income generation, directional alignment, volatility exposure, etc.).

    Spec References: FR-034 (pluggable interface), FR-040 (custom plugins)
    Tasks: T026
    """

    @abstractmethod
    def score(
        self,
        candidate: dict[str, Any],
        metrics: dict[str, Any],
        config: dict[str, Any],
    ) -> float:
        """
        Compute composite score for a candidate strategy.

        Args:
            candidate: CandidateStructure dict with structure_type, legs, strikes, expiries
            metrics: Monte Carlo metrics dict with E[PnL], POP, ROC, Greeks, VaR, CVaR, etc.
            config: Scoring config section with weights, thresholds, and normalization params

        Returns:
            Composite score in [0, 1] range (higher is better)

        Spec: FR-034
        """
        pass

    @abstractmethod
    def decompose(
        self,
        candidate: dict[str, Any],
        metrics: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, float]:
        """
        Return score decomposition showing individual component contributions.

        Args:
            candidate: CandidateStructure dict
            metrics: Monte Carlo metrics dict
            config: Scoring config section

        Returns:
            Dictionary with component scores (e.g., {"pop_contrib": 0.25, "roc_contrib": 0.18, ...})

        Spec: FR-041 (score decomposition output)
        """
        pass


def load_scorer_plugin(scorer_name: str, scorers_dir: Path | None = None) -> StrategyScorer:
    """
    Load custom scorer plugin from scorers/ directory (FR-040).

    Searches for a Python module named `{scorer_name}.py` in the scorers directory,
    imports it dynamically, and instantiates the first StrategyScorer subclass found.

    Args:
        scorer_name: Name of the scorer plugin (e.g., "directional_bullish")
        scorers_dir: Optional path to custom scorers directory (defaults to cwd/scorers/)

    Returns:
        Instantiated StrategyScorer subclass

    Raises:
        FileNotFoundError: If scorer module not found
        ImportError: If module cannot be imported
        ValueError: If no StrategyScorer subclass found in module

    Spec: FR-040 (auto-discovery of custom scorers)
    Tasks: T026

    Example:
        >>> scorer = load_scorer_plugin("directional_bullish")
        >>> score = scorer.score(candidate, metrics, config)
    """
    if scorers_dir is None:
        scorers_dir = Path.cwd() / "scorers"

    scorer_path = scorers_dir / f"{scorer_name}.py"
    if not scorer_path.exists():
        raise FileNotFoundError(
            f"Scorer plugin '{scorer_name}' not found at {scorer_path}. "
            f"Create {scorer_path} with a StrategyScorer subclass."
        )

    # Dynamically import the module
    spec = importlib.util.spec_from_file_location(f"qse.scorers.{scorer_name}", scorer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scorer module from {scorer_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Find and instantiate the first StrategyScorer subclass
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, StrategyScorer) and obj is not StrategyScorer:
            return obj()

    raise ValueError(
        f"No StrategyScorer subclass found in {scorer_path}. "
        f"Ensure the module defines a class inheriting from StrategyScorer."
    )
