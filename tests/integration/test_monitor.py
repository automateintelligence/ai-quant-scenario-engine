from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from qse.monitoring import PositionMonitor, load_position
from qse.pricing.black_scholes import BlackScholesPricer


class StubDataSource:
    def __init__(self, underlying_price: float, option_mark: float) -> None:
        self.underlying_price = underlying_price
        self.option_mark = option_mark

    def get_underlying_price(self, symbol: str) -> float:  # pragma: no cover - trivial
        return self.underlying_price

    def get_option_mark(self, **_: object) -> float:
        return self.option_mark


def test_monitor_triggers_alert_with_stubbed_data(tmp_path: Path) -> None:
    entry_time = datetime.utcnow() - timedelta(days=1)
    expiry = (entry_time + timedelta(days=14)).date()
    position_payload = {
        "underlying": "SPY",
        "trade_horizon": 5,
        "entry_time": entry_time.isoformat(),
        "regime": "strong-bullish",
        "legs": [
            {
                "option_type": "call",
                "side": "long",
                "strike": 420,
                "expiry": expiry.isoformat(),
                "quantity": 1,
                "entry_price": 1.5,
                "implied_vol": 0.25,
            }
        ],
        "alerts": {"profit_target": 1.0, "stop_loss": -2.0},
        "config_snapshot": {
            "regimes": {
                "strong-bullish": {
                    "mean_daily_return": 0.01,
                    "daily_vol": 0.15,
                    "skew": 0.0,
                    "kurtosis_excess": 0.0,
                }
            },
            "mc": {"num_paths": 5, "bars_per_day": 1, "seed": 1},
        },
    }
    position_path = tmp_path / "position.json"
    position_path.write_text(json.dumps(position_payload))

    snapshot = load_position(position_path)
    data_source = StubDataSource(underlying_price=430.0, option_mark=3.0)
    monitor = PositionMonitor(
        data_source,
        pricer=BlackScholesPricer(),
        regimes_config=snapshot.config_snapshot.get("regimes"),
        mc_config=snapshot.config_snapshot.get("mc"),
    )

    result = next(monitor.monitor(snapshot, interval_seconds=0, iterations=1))

    assert result["alert"]["triggered"] is True
    assert result["alert"]["reason"] == "profit_target"
    assert result["mark_pnl"] > 0
    assert result["simulation"]["paths"] == 5
