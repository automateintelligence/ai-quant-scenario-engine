"""Option specification data model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from quant_scenario_engine.exceptions import ConfigValidationError

OptionType = Literal["call", "put"]
IVSource = Literal["yfinance", "realized_vol", "config_default"]


@dataclass(slots=True)
class OptionSpec:
    option_type: OptionType
    strike: Union[float, str]
    maturity_days: int
    implied_vol: float
    risk_free_rate: float
    contracts: int
    iv_source: IVSource = "config_default"
    early_exercise: bool = False

    def __post_init__(self) -> None:
        if self.option_type not in {"call", "put"}:
            raise ConfigValidationError("option_type must be 'call' or 'put'")
        if isinstance(self.strike, (int, float)) and self.strike <= 0:
            raise ConfigValidationError("strike must be positive")
        if self.maturity_days <= 0:
            raise ConfigValidationError("maturity_days must be positive")
        if self.implied_vol <= 0 or self.implied_vol >= 5:
            raise ConfigValidationError("implied_vol must be between 0 and 5")
        if self.contracts == 0:
            raise ConfigValidationError("contracts cannot be zero")
        if self.risk_free_rate < -1:
            raise ConfigValidationError("risk_free_rate looks invalid")
        if self.iv_source not in {"yfinance", "realized_vol", "config_default"}:
            raise ConfigValidationError("invalid iv_source")

