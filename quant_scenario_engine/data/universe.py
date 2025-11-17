"""Universe/watchlist/live set configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from quant_scenario_engine.exceptions import ConfigValidationError


@dataclass
class UniverseConfig:
    universe: List[str] = field(default_factory=list)
    watchlist: List[str] = field(default_factory=list)
    live: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate_tiers(self.universe, "universe")
        self._validate_tiers(self.watchlist, "watchlist")
        self._validate_tiers(self.live, "live")

    @staticmethod
    def _validate_tiers(symbols: List[str], name: str) -> None:
        if len(symbols) != len(set(symbols)):
            raise ConfigValidationError(f"Duplicate symbols in {name} tier")

