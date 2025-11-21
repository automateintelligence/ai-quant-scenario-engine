import numpy as np
import pytest

from qse.distributions.garch_t import GarchTDistribution


@pytest.mark.skipif(pytest.importorskip("arch", reason="arch not installed") is None, reason="arch missing")
def test_garch_t_fit_and_sample_shape():
    rng = np.random.default_rng(0)
    returns = rng.standard_t(df=8, size=300)
    dist = GarchTDistribution()
    dist.fit(returns)
    samples = dist.sample(n_paths=5, n_steps=10, seed=123)
    assert samples.shape == (5, 10)
