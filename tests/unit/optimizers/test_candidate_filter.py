from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from qse.optimizers.candidate_filter import (
    Stage0Config,
    Stage1Config,
    filter_strikes,
    select_expiries,
)


def _build_chain() -> pd.DataFrame:
    as_of = datetime(2025, 1, 2)
    expiries = [as_of + timedelta(days=d) for d in (5, 10, 15, 25, 40, 60)]
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


def test_select_expiries_limits_and_order():
    chain = _build_chain()
    as_of = datetime(2025, 1, 2)

    selected = select_expiries(chain, as_of, Stage0Config(min_dte=7, max_dte=45, max_expiries=4))

    assert len(selected) == 4
    assert all(7 <= (expiry - pd.Timestamp(as_of)).days <= 45 for expiry in selected)
    assert selected == sorted(selected)


def test_select_expiries_insufficient_window_raises():
    chain = _build_chain()
    as_of = datetime(2025, 1, 2)

    with pytest.raises(ValueError):
        select_expiries(chain[chain["expiry"] == chain["expiry"].min()], as_of, Stage0Config(min_dte=1, max_dte=5))


def test_filter_strikes_applies_moneyness_and_liquidity():
    chain = _build_chain()
    as_of = datetime(2025, 1, 2)
    expiries = select_expiries(chain, as_of)

    df = chain.copy()
    df.loc[df.index[0], "volume"] = 0  # should be filtered out
    filtered = filter_strikes(df, spot=100.0, expiries=expiries, config=Stage1Config(min_volume=10))

    assert filtered["expiry"].nunique() == 4  # DTE window retains four expiries within [7, 45]
    assert filtered["volume"].min() >= 10
    assert filtered["moneyness"].between(0.85, 1.15).all()


def test_filter_strikes_requires_columns():
    chain = pd.DataFrame({"expiry": [datetime(2025, 1, 10)], "strike": [100.0]})

    with pytest.raises(ValueError):
        filter_strikes(chain, spot=100.0, expiries=[], config=Stage1Config())
