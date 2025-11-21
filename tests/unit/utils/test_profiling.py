import time

from qse.utils.profiling import budget_checker, track_time


def test_budget_checker_logs_warning(caplog):
    checker = budget_checker(10.0, warn_ratio=0.5, error_ratio=0.9)
    checker(6.0)
    assert any("exceeded 50%" in r.message for r in caplog.records)


def test_track_time_emits_info(caplog):
    with caplog.at_level("INFO"):
        with track_time("segment", warn_budget=0.1):
            time.sleep(0.01)
    assert any(r.levelname == "INFO" for r in caplog.records)
