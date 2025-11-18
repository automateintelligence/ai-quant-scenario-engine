import numpy as np
import pytest

from quant_scenario_engine.distributions.validation.stationarity import ensure_min_samples
from quant_scenario_engine.exceptions import DistributionFitError


def test_ensure_min_samples_passes():
    returns = np.random.normal(0, 0.01, size=300)
    ensure_min_samples(returns, "laplace")
    ensure_min_samples(returns, "student_t")


def test_ensure_min_samples_raises_for_garch():
    returns = np.random.normal(0, 0.01, size=100)
    with pytest.raises(DistributionFitError):
        ensure_min_samples(returns, "garch_t")
