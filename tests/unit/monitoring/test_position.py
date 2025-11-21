from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from qse.exceptions import ConfigValidationError, SchemaError
from qse.monitoring.position import PositionSnapshot, load_position


def test_load_position_happy_path(tmp_path: Path) -> None:
    entry_time = datetime.utcnow() - timedelta(days=2)
    position = {
        "underlying": "SPY",
        "trade_horizon": 5,
        "entry_time": entry_time.isoformat(),
        "regime": "strong-bullish",
        "legs": [
            {
                "option_type": "call",
                "side": "long",
                "strike": 420,
                "expiry": (entry_time + timedelta(days=10)).date().isoformat(),
                "quantity": 1,
                "entry_price": 2.5,
                "implied_vol": 0.2,
            }
        ],
        "alerts": {"profit_target": 50.0, "stop_loss": -25.0},
        "config_snapshot": {"regimes": {"strong-bullish": {"mean_daily_return": 0.01, "daily_vol": 0.2, "skew": 0.0, "kurtosis_excess": 0.0}}},
    }
    path = tmp_path / "position.json"
    path.write_text(json.dumps(position))

    snapshot = load_position(path)

    assert isinstance(snapshot, PositionSnapshot)
    assert snapshot.underlying == "SPY"
    assert snapshot.remaining_horizon(entry_time + timedelta(days=3)) == 2
    assert snapshot.alerts is not None and snapshot.alerts.enabled


def test_load_position_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad_position.json"
    path.write_text(json.dumps({"underlying": "SPY"}))

    with pytest.raises(SchemaError):
        load_position(path)


def test_alert_validation(tmp_path: Path) -> None:
    entry_time = datetime.utcnow().isoformat()
    path = tmp_path / "bad_alert.json"
    path.write_text(
        json.dumps(
            {
                "underlying": "SPY",
                "trade_horizon": 3,
                "entry_time": entry_time,
                "regime": "neutral",
                "legs": [
                    {
                        "option_type": "put",
                        "side": "short",
                        "strike": 400,
                        "expiry": (datetime.utcnow() + timedelta(days=5)).date().isoformat(),
                        "quantity": -1,
                        "entry_price": 1.0,
                        "implied_vol": 0.25,
                    }
                ],
                "alerts": {"profit_target": -10.0},
            }
        )
    )

    with pytest.raises(ConfigValidationError):
        load_position(path)
