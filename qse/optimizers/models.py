"""Shared optimizer data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd

OptionType = Literal["call", "put"]
Side = Literal["buy", "sell"]
StructureType = Literal["vertical", "iron_condor", "straddle", "strangle"]


@dataclass(slots=True)
class Leg:
    """Single option leg used in candidate structures."""

    option_type: OptionType
    strike: float
    expiry: pd.Timestamp
    side: Side
    premium: float
    quantity: int = 1
    bid: float | None = None
    ask: float | None = None


@dataclass(slots=True)
class CandidateMetrics:
    """Lightweight metrics used across Stage 3/4 filtering."""

    expected_pnl: float
    pop_breakeven: float
    pop_target: float
    capital: float
    max_loss: float
    score: float
    mc_paths: int | None = None
    entry_cash: float | None = None
    expected_exit_cost: float | None = None
    commission: float | None = None


@dataclass(slots=True)
class CandidateStructure:
    """Container for generated option structures."""

    structure_type: StructureType
    legs: Sequence[Leg]
    expiry: pd.Timestamp
    width: float
    metrics: CandidateMetrics | None = None
    composite_score: float | None = None
    score_decomposition: dict[str, float] | None = None

    @property
    def net_premium(self) -> float:
        """Return the net credit (positive) or debit (negative) for the structure."""

        return sum(leg.premium * (1 if leg.side == "sell" else -1) for leg in self.legs)


__all__ = [
    "CandidateMetrics",
    "CandidateStructure",
    "Leg",
    "OptionType",
    "Side",
    "StructureType",
]
