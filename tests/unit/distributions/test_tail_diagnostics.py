import numpy as np

from quant_scenario_engine.distributions.diagnostics.qq_plots import compute_qq_pairs
from quant_scenario_engine.distributions.diagnostics.tail_metrics import tail_error
from quant_scenario_engine.distributions.diagnostics.kurtosis import compare_kurtosis
from quant_scenario_engine.distributions.diagnostics.tail_report import build_tail_report


def test_qq_pairs_shapes():
    emp = np.random.normal(0, 1, size=500)
    mod = np.random.normal(0, 1, size=500)
    q, emp_q, mod_q = compute_qq_pairs(emp, mod)
    assert len(q) == len(emp_q) == len(mod_q)


def test_tail_error_keys():
    emp = np.random.normal(0, 1, size=500)
    mod = np.random.normal(0, 1, size=500)
    errors = tail_error(emp, mod)
    assert "var_95.0" in errors
    assert "relative_error" in errors["var_95.0"]


def test_kurtosis_compare():
    emp = np.random.standard_t(df=5, size=500)
    mod = np.random.standard_t(df=7, size=500)
    result = compare_kurtosis(emp, mod)
    assert "empirical" in result and "model" in result


def test_tail_report_aggregate():
    emp = np.random.normal(0, 1, size=500)
    mod = np.random.normal(0, 1, size=500)
    report = build_tail_report(emp, mod)
    assert "tail_errors" in report and "kurtosis" in report
