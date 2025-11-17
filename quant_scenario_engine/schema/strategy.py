"""Strategy parameter schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal

from quant_scenario_engine.exceptions import ConfigValidationError

StrategyKind = Literal["stock", "option"]
PositionSizing = Literal["fixed_notional", "percent_equity"]


@dataclass(slots=True)
class StrategyParams:
    name: str
    kind: StrategyKind
    params: Dict[str, object] = field(default_factory=dict)
    position_sizing: PositionSizing = "fixed_notional"
    fees: float = 0.0005
    slippage: float = 0.65

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("strategy name required")
        if self.kind not in {"stock", "option"}:
            raise ConfigValidationError("kind must be 'stock' or 'option'")
        if self.fees < 0:
            raise ConfigValidationError("fees must be >= 0")
        if self.slippage < 0:
            raise ConfigValidationError("slippage must be >= 0")
        if self.position_sizing not in {"fixed_notional", "percent_equity"}:
            raise ConfigValidationError("invalid position_sizing")

