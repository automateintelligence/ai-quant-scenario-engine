import sys

import numpy as np
import pandas as pd

from quant_scenario_engine.distributions.distribution_audit import audit_distributions_for_symbol


def test_audit_selects_best_model_and_reports_failures():
    prices = pd.Series(np.linspace(100, 120, 200))
    result = audit_distributions_for_symbol("TEST", prices, train_fraction=0.8, require_heavy_tails=False)

    # Laplace and Student-T should fit; garch_t should report failure without silent fallback
    names = {fr.model_name: fr for fr in result.fit_results}
    assert names["laplace"].fit_success is True
    assert names["student_t"].fit_success is True
    assert names["garch_t"].fit_success is False
    assert "insufficient samples" in (names["garch_t"].fit_message.lower())
    # Best model should be chosen among successful fits
    assert result.best_model is not None
    assert result.best_model.name in {"laplace", "student_t"}


def test_audit_falls_back_when_no_heavy_tail_available():
    prices = pd.Series(np.linspace(100, 120, 100))
    result = audit_distributions_for_symbol("TEST", prices, train_fraction=0.9, require_heavy_tails=True)

    # Laplace may not meet heavy-tail check, but we should still select best available with warning fallback
    assert result.best_model is not None
    assert result.best_model.name in {"laplace", "student_t"}


def test_garch_t_requires_arch_dependency(monkeypatch):
    prices = pd.Series(np.linspace(100, 120, 300))

    # Force arch import failure
    monkeypatch.setitem(sys.modules, "arch", None)

    result = audit_distributions_for_symbol("TEST", prices, require_heavy_tails=False)
    garch_result = next(fr for fr in result.fit_results if fr.model_name == "garch_t")
    assert garch_result.fit_success is False
    assert "arch" in (garch_result.error or "")
