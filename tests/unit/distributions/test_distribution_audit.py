import sys

import numpy as np
import pandas as pd

from qse.distributions.distribution_audit import audit_distributions_for_symbol


def test_audit_selects_best_model_and_reports_failures(tmp_path):
    prices = pd.Series(np.linspace(100, 120, 200))
    result = audit_distributions_for_symbol(
        "TEST",
        prices,
        train_fraction=0.8,
        require_heavy_tails=False,
        cache_dir=tmp_path,
        force_refit=True,
    )

    # Laplace and Student-T should fit; garch_t should report failure without silent fallback
    names = {fr.model_name: fr for fr in result.fit_results}
    assert names["laplace"].fit_success is True
    assert names["student_t"].fit_success is True
    assert names["garch_t"].fit_success is False
    assert "insufficient" in (names["garch_t"].fit_message.lower())
    # Best model should be chosen among successful fits
    assert result.best_model is not None
    assert result.best_model.name in {"laplace", "student_t"}


def test_audit_falls_back_when_no_heavy_tail_available(tmp_path):
    prices = pd.Series(np.linspace(100, 120, 100))
    result = audit_distributions_for_symbol(
        "TEST2",
        prices,
        train_fraction=0.9,
        require_heavy_tails=True,
        cache_dir=tmp_path,
        force_refit=True,
    )

    # Laplace may not meet heavy-tail check, but we should still select best available with warning fallback
    assert result.best_model is not None
    assert result.best_model.name in {"laplace", "student_t"}


def test_garch_t_requires_arch_dependency(monkeypatch, tmp_path):
    prices = pd.Series(np.linspace(100, 120, 300))

    # Force arch import failure
    monkeypatch.setitem(sys.modules, "arch", None)

    result = audit_distributions_for_symbol(
        "TEST3",
        prices,
        require_heavy_tails=False,
        cache_dir=tmp_path,
        force_refit=True,
    )
    garch_result = next(fr for fr in result.fit_results if fr.model_name == "garch_t")
    assert garch_result.fit_success is False
    assert "arch" in (garch_result.error or "")


def test_audit_outputs_tail_var_and_realism_metrics(tmp_path):
    rng = np.random.default_rng(42)
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, size=400)))

    result = audit_distributions_for_symbol(
        "ZZZ",
        prices,
        train_fraction=0.75,
        require_heavy_tails=False,
        seed=7,
        cache_dir=tmp_path,
        force_refit=True,
    )

    assert result.tail_metrics, "expected tail metrics"
    first = result.tail_metrics[0]
    assert first.tail_error_995 >= 0.0

    assert result.var_backtests, "expected VaR backtests"
    var_entry = result.var_backtests[0]
    assert 0.0 <= var_entry.kupiec_pvalue <= 1.0
    assert 0.0 <= var_entry.christoffersen_pvalue <= 1.0

    assert result.simulation_metrics, "expected simulation metrics"
    sim_model = result.simulation_metrics[0].model_name
    realism = result.realism_reports.get(sim_model)
    assert realism and "deltas" in realism
    assert result.selection_report["scores"], "selection report should include scores"
