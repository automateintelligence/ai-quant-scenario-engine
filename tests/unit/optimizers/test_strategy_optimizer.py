from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from qse.optimizers.prefilter import Prefilter, Stage3Config
from qse.optimizers.strategy_optimizer import StrategyOptimizer


AS_OF = datetime(2025, 1, 2)


def _chain_rows() -> pd.DataFrame:
    expiries = [AS_OF + timedelta(days=d) for d in (10, 20, 30, 40)]
    rows = []
    for expiry in expiries:
        for strike, option_type in [(95, "put"), (100, "call"), (105, "call"), (110, "put")]:
            rows.append(
                {
                    "expiry": expiry,
                    "strike": strike,
                    "option_type": option_type,
                    "bid": 1.0,
                    "ask": 1.2,
                    "volume": 50,
                    "open_interest": 100,
                }
            )
    return pd.DataFrame(rows)


def test_strategy_optimizer_runs_pipeline():
    optimizer = StrategyOptimizer(prefilter=Prefilter(Stage3Config(
        max_loss_pct=0.1,
        min_expected_pnl=10.0,
        min_pop_breakeven=0.5,
        min_pop_target=0.5,
    )))

    result = optimizer.run(_chain_rows(), as_of=AS_OF, spot=100.0)

    assert result.survivors
    assert result.stage_counts["stage0_expiries"] == 4
    assert result.stage_counts["stage4_mc"] == len(result.survivors)
    assert all(candidate.metrics and candidate.metrics.mc_paths == optimizer.ci_settings.baseline_paths for candidate in result.survivors)


def test_strategy_optimizer_requires_columns():
    optimizer = StrategyOptimizer()
    chain = pd.DataFrame({"expiry": [AS_OF + timedelta(days=10)], "strike": [100.0]})

    with pytest.raises(ValueError):
        optimizer.run(chain, as_of=AS_OF, spot=100.0)
