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
    assert "not implemented" in (names["garch_t"].fit_message.lower())
    # Best model should be chosen among successful fits
    assert result.best_model is not None
    assert result.best_model.name in {"laplace", "student_t"}
