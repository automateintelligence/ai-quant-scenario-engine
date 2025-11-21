import numpy as np

from qse.distributions.student_t import StudentTDistribution


def test_student_t_fit_allows_light_pacf():
    returns = np.random.default_rng(0).standard_t(df=6, size=400)
    dist = StudentTDistribution()
    dist.fit(returns)
    samples = dist.sample(3, 5, seed=123)
    assert samples.shape == (3, 5)
    assert np.isfinite(samples).all()
