import numpy as np
import pytest

from quant_scenario_engine.distributions.laplace import LaplaceDistribution
from quant_scenario_engine.exceptions import DistributionFitError
from quant_scenario_engine.mc.generator import generate_price_paths


def test_generate_price_paths_deterministic_seed():
    returns = np.random.normal(0, 0.01, size=200)
    dist = LaplaceDistribution()
    dist.fit(returns)
    p1 = generate_price_paths(s0=100.0, distribution=dist, n_paths=3, n_steps=5, seed=123)
    p2 = generate_price_paths(s0=100.0, distribution=dist, n_paths=3, n_steps=5, seed=123)
    assert np.array_equal(p1, p2)


def test_generate_price_paths_raises_on_bad_shape():
    class BadDist(LaplaceDistribution):
        def sample(self, n_paths: int, n_steps: int, seed=None):
            return np.ones((1, 1))

    dist = BadDist()
    dist.loc, dist.scale = 0.0, 1.0
    with pytest.raises(DistributionFitError):
        generate_price_paths(100.0, dist, 2, 2, seed=1)

