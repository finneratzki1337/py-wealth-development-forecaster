from __future__ import annotations

import numpy as np

from wealth_forecaster.returns import monthly_params, sample_monthly_returns


def test_fee_drag_vs_additive():
    rng = np.random.default_rng(123)
    mu_m, sigma_m = monthly_params(0.06, 0.1, "arithmetic")
    net = sample_monthly_returns(10000, mu_m, sigma_m, 0.012, "normal_arith", True, rng)
    gross = rng.normal(mu_m, sigma_m, size=10000)
    gross = np.maximum(gross, -0.999999)
    approx = gross - 0.012 / 12
    assert net.mean() < approx.mean()


def test_lognormal_mean_variance_match():
    rng = np.random.default_rng(456)
    mu_m = 0.005
    sigma_m = 0.02
    samples = sample_monthly_returns(50000, mu_m, sigma_m, 0.0, "lognormal_match", False, rng)
    mean = samples.mean()
    std = samples.std()
    assert np.isclose(mean, mu_m, atol=5e-4)
    assert np.isclose(std, sigma_m, atol=5e-4)
