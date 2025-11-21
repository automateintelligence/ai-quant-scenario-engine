from __future__ import annotations

import pandas as pd

from qse.optimizers.models import CandidateStructure, Leg
from qse.optimizers.prefilter import Prefilter, Stage3Config


EXPIRY = pd.Timestamp("2025-01-20")


def _make_candidate(net_credit: float, short_strike: float, long_strike: float, structure: str = "vertical") -> CandidateStructure:
    legs = [
        Leg(option_type="call", strike=short_strike, expiry=EXPIRY, side="sell", premium=net_credit + 0.5),
        Leg(option_type="call", strike=long_strike, expiry=EXPIRY, side="buy", premium=0.5),
    ]
    return CandidateStructure(structure_type=structure, legs=legs, expiry=EXPIRY, width=abs(short_strike - long_strike) or 1.0)


def test_prefilter_scores_and_limits_top_k():
    candidates = [
        _make_candidate(net_credit=2.0, short_strike=105, long_strike=100),
        _make_candidate(net_credit=1.0, short_strike=110, long_strike=100),
        _make_candidate(net_credit=0.6, short_strike=115, long_strike=100),
    ]
    prefilter = Prefilter(Stage3Config(top_k_per_type=2, max_loss_pct=0.1))

    survivors = prefilter.evaluate(candidates, spot=100.0)

    assert len(survivors) == 2
    scores = [s.metrics.score for s in survivors]
    assert scores == sorted(scores, reverse=True)


def test_prefilter_applies_expected_pnl_threshold():
    low_credit = _make_candidate(net_credit=0.1, short_strike=102, long_strike=101)
    prefilter = Prefilter()

    survivors = prefilter.evaluate([low_credit], spot=100.0)

    assert survivors == []


def test_prefilter_rejects_when_loss_ratio_exceeds_limit():
    risky = _make_candidate(net_credit=0.8, short_strike=150, long_strike=140, structure="strangle")
    prefilter = Prefilter(Stage3Config(max_capital=200.0, max_loss_pct=0.05))

    survivors = prefilter.evaluate([risky], spot=100.0)

    assert survivors == []
