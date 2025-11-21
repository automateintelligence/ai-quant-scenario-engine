import numpy as np

from qse.distributions.laplace import LaplaceDistribution
from qse.mc.generator import generate_price_paths


def test_generate_price_paths_positive():
    dist = LaplaceDistribution()
    dist.loc = 0.0
    dist.scale = 0.01
    paths = generate_price_paths(s0=100.0, distribution=dist, n_paths=3, n_steps=5, seed=42)
    assert paths.shape == (3, 5)
    assert (paths > 0).all()
