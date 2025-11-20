"""VaR backtest report builder (US6a AS4, T154)."""

from __future__ import annotations

from typing import Sequence
from qse.distributions.backtesting.christoffersen_test import christoffersen_pvalue
from qse.distributions.backtesting.kupiec_test import kupiec_pvalue
from qse.distributions.backtesting.breach_counter import count_breaches
from qse.distributions.backtesting.var_predictor import predict_var_from_samples

def run_var_backtest(returns_test, model_samples, levels: Sequence[float]):
    reports = []
    for level in levels:
        var_level = predict_var_from_samples(model_samples, level)
        n_breaches, breaches_seq = count_breaches(returns_test, var_level)
        n_obs = len(returns_test)
        kupiec = kupiec_pvalue(n_obs, n_breaches, level)
        christ = christoffersen_pvalue(breaches_seq)
        reports.append(
            {
                "level": level,
                "n_obs": n_obs,
                "n_breaches": n_breaches,
                "var_level": var_level,
                "kupiec_pvalue": kupiec,
                "christoffersen_pvalue": christ,
                "passed": kupiec > 0.05 and christ > 0.05,
            }
        )
    return reports


__all__ = ["run_var_backtest"]
