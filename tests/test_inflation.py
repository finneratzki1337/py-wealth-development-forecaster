from __future__ import annotations

import numpy as np

from wealth_forecaster.inflation import expand_to_monthly_factors, sample_yearly_inflation


def test_inflation_truncation():
    rng = np.random.default_rng(0)
    draws = sample_yearly_inflation(10, -0.5, 0.5, rng)
    assert (draws > -1).all()


def test_monthly_expansion_length():
    yearly = np.array([0.02, 0.03])
    monthly = expand_to_monthly_factors(yearly)
    assert monthly.shape[0] == 24
    assert np.isclose(monthly[:12].prod(), 1.02, atol=1e-6)
