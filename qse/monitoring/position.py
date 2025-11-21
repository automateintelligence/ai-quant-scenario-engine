from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from qse.exceptions import ConfigValidationError, SchemaError
from qse.models.options import OptionSpec, OptionType


@dataclass(slots=True)
class AlertConfig:
    """Alert thresholds for monitored positions."""

    profit_target: float | None = None
    stop_loss: float | None = None

    def __post_init__(self) -> None:
        if self.profit_target is not None and self.profit_target <= 0:
            raise ConfigValidationError("profit_target must be positive when provided")
        if self.stop_loss is not None and self.stop_loss >= 0:
            raise ConfigValidationError("stop_loss must be negative when provided")

    @property
    def enabled(self) -> bool:
        return self.profit_target is not None or self.stop_loss is not None


@dataclass(slots=True)
class PositionLeg:
    """Single option leg used for monitoring and repricing."""

    option_type: OptionType
    side: Literal["long", "short"]
    strike: float
    expiry: date
    quantity: int
    entry_price: float
    implied_vol: float
    risk_free_rate: float = 0.0
    option_symbol: str | None = None

    def __post_init__(self) -> None:
        if self.option_type not in {"call", "put"}:
            raise SchemaError("option_type must be 'call' or 'put'")
        if self.side not in {"long", "short"}:
            raise SchemaError("side must be 'long' or 'short'")
        if self.strike <= 0:
            raise SchemaError("strike must be positive")
        if self.quantity == 0:
            raise SchemaError("quantity cannot be zero")
        if self.implied_vol <= 0:
            raise SchemaError("implied_vol must be positive")

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    def days_to_expiry(self, as_of: datetime) -> int:
        days = (self.expiry - as_of.date()).days
        return max(1, days)

    def to_option_spec(self, as_of: datetime, remaining_horizon: int) -> OptionSpec:
        maturity = min(self.days_to_expiry(as_of), max(remaining_horizon, 1))
        return OptionSpec(
            option_type=self.option_type,
            strike=self.strike,
            maturity_days=maturity,
            implied_vol=self.implied_vol,
            risk_free_rate=self.risk_free_rate,
            contracts=abs(self.quantity),
        )


@dataclass
class PositionSnapshot:
    """Full position snapshot loaded from exported JSON."""

    underlying: str
    legs: list[PositionLeg]
    entry_time: datetime
    trade_horizon: int
    regime: str
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    alerts: AlertConfig | None = None
    last_mark_time: datetime | None = None
    last_metrics: dict[str, Any] | None = None

    def remaining_horizon(self, now: datetime | None = None) -> int:
        current_time = now or datetime.utcnow()
        days_elapsed = max((current_time - self.entry_time).days, 0)
        return max(1, self.trade_horizon - days_elapsed)


def _parse_datetime(raw: Any, field_name: str) -> datetime:
    if not isinstance(raw, str):
        raise SchemaError(f"{field_name} must be an ISO8601 string")
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SchemaError(f"{field_name} must be ISO8601 formatted") from exc


def _parse_date(raw: Any, field_name: str) -> date:
    if not isinstance(raw, str):
        raise SchemaError(f"{field_name} must be a YYYY-MM-DD string")
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise SchemaError(f"{field_name} must be YYYY-MM-DD formatted") from exc


def _load_alerts(raw_alerts: dict[str, Any] | None) -> AlertConfig | None:
    if raw_alerts is None:
        return None
    if not isinstance(raw_alerts, dict):
        raise SchemaError("alerts must be an object when provided")
    return AlertConfig(
        profit_target=raw_alerts.get("profit_target"),
        stop_loss=raw_alerts.get("stop_loss"),
    )


def _load_leg(raw_leg: dict[str, Any]) -> PositionLeg:
    required_fields = [
        "option_type",
        "side",
        "strike",
        "expiry",
        "quantity",
        "entry_price",
        "implied_vol",
    ]
    missing = [field for field in required_fields if field not in raw_leg]
    if missing:
        raise SchemaError(f"leg is missing required fields: {', '.join(missing)}")

    return PositionLeg(
        option_type=str(raw_leg["option_type"]).lower(),
        side=str(raw_leg["side"]).lower(),
        strike=float(raw_leg["strike"]),
        expiry=_parse_date(raw_leg["expiry"], "expiry"),
        quantity=int(raw_leg["quantity"]),
        entry_price=float(raw_leg["entry_price"]),
        implied_vol=float(raw_leg["implied_vol"]),
        risk_free_rate=float(raw_leg.get("risk_free_rate", 0.0)),
        option_symbol=str(raw_leg.get("option_symbol")) if raw_leg.get("option_symbol") else None,
    )


def _load_legs(raw_legs: list[Any]) -> list[PositionLeg]:
    if not isinstance(raw_legs, list) or not raw_legs:
        raise SchemaError("legs must be a non-empty array")
    return [_load_leg(leg) for leg in raw_legs]


def load_position(position_path: str | Path) -> PositionSnapshot:
    """Load and validate a position JSON file for monitoring (FR-063/FR-064)."""

    path = Path(position_path)
    if not path.exists():
        raise FileNotFoundError(f"Position file not found: {path}")

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise SchemaError("position must be a JSON object")

    underlying = raw.get("underlying")
    if not isinstance(underlying, str) or not underlying:
        raise SchemaError("underlying is required and must be a string")

    trade_horizon = raw.get("trade_horizon")
    if not isinstance(trade_horizon, int) or trade_horizon <= 0:
        raise SchemaError("trade_horizon must be a positive integer")

    entry_time = _parse_datetime(raw.get("entry_time"), "entry_time")
    regime = raw.get("regime")
    if not isinstance(regime, str) or not regime:
        raise SchemaError("regime is required and must be a string")

    position = PositionSnapshot(
        underlying=underlying,
        legs=_load_legs(raw.get("legs", [])),
        entry_time=entry_time,
        trade_horizon=trade_horizon,
        regime=regime,
        config_snapshot=raw.get("config_snapshot", {}),
        alerts=_load_alerts(raw.get("alerts")),
        last_mark_time=_parse_datetime(raw["last_mark_time"], "last_mark_time")
        if raw.get("last_mark_time")
        else None,
        last_metrics=raw.get("last_metrics"),
    )
    return position


__all__ = [
    "AlertConfig",
    "PositionLeg",
    "PositionSnapshot",
    "load_position",
]
