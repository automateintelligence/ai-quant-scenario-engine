import numpy as np

from qse.optimizer.metrics import AdaptiveCISettings
from qse.optimizers.diagnostics import adaptive_diagnostics, empty_result_diagnostics
from qse.optimizers.metrics import adaptive_path_plan


def test_adaptive_paths_doubles_when_ci_wide():
    epnl = np.array([0, 100, 200, 300, 400], dtype=float)
    pop = np.array([0.4, 0.5, 0.6, 0.5, 0.55], dtype=float)
    settings = AdaptiveCISettings(baseline_paths=5000, max_paths=20000, epnl_ci_target=10.0, pop_ci_target=0.01)
    next_paths, diag = adaptive_path_plan(epnl, pop, current_paths=5000, settings=settings)
    assert next_paths == 10000
    assert diag["path_status"] == "continue"


def test_adaptive_paths_caps_at_max():
    epnl = np.array([0, 100, 200, 300, 400], dtype=float)
    pop = np.array([0.4, 0.5, 0.6, 0.5, 0.55], dtype=float)
    settings = AdaptiveCISettings(baseline_paths=16000, max_paths=20000, epnl_ci_target=1.0, pop_ci_target=0.01)
    next_paths, diag = adaptive_path_plan(epnl, pop, current_paths=20000, settings=settings)
    assert next_paths == 20000
    assert diag["path_status"] == "cap_reached"


def test_empty_result_diagnostics_contains_breakdown():
    rejections = {"capital_filter": 5, "pop_filter": 3}
    stage_counts = {"stage0": 10, "stage1": 6}
    diag = empty_result_diagnostics(rejections, stage_counts, hints="tight filters")
    assert diag["rejection_breakdown"]["capital_filter"] == 5
    assert diag["stage_counts"]["stage0"] == 10
    assert diag["hints"] == "tight filters"


def test_adaptive_diagnostics_structure():
    diag = adaptive_diagnostics(epnl_ci_halfwidth=120.0, pop_ci_halfwidth=0.04, status="continue", path_count=10000)
    assert diag["path_status"] == "continue"
    assert diag["path_count"] == 10000
