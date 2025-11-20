"""Unit tests for distribution error helpers (US6a AS11-AS12)."""

from quant_scenario_engine.distributions.errors import (
    handle_insufficient_data,
    has_minimum_samples,
    record_convergence_failure,
)


def test_has_minimum_samples_checks_threshold() -> None:
    assert has_minimum_samples("laplace", 100)
    assert not has_minimum_samples("garch_t", 100, min_required=252)


def test_handle_insufficient_data_marks_fit_result() -> None:
    result = handle_insufficient_data("garch_t", 100, min_required=252, symbol="SPY")
    assert result.fit_success is False
    assert result.converged is False
    assert result.n == 100
    assert "Insufficient data" in (result.error or "")
    assert result.fit_message == "skipped_insufficient_data"


def test_record_convergence_failure_populates_warning() -> None:
    result = record_convergence_failure(
        "student_t",
        error=RuntimeError("optimization diverged"),
        n_samples=200,
        stage="fit",
    )
    assert result.fit_success is False
    assert result.converged is False
    assert result.n == 200
    assert any("diverged" in warning for warning in result.warnings)
