from __future__ import annotations

import numpy as np

from wealth_forecaster.returns import monthly_params


def test_monthly_params_arithmetic():
    mu, sigma = monthly_params(0.12, 0.24, "arithmetic")
    assert np.isclose(mu, 0.01)
    assert np.isclose(sigma, 0.24 / np.sqrt(12))


def test_monthly_params_geometric():
    mu, sigma = monthly_params(0.12, 0.24, "geometric")
    assert np.isclose(mu, (1 + 0.12) ** (1 / 12) - 1)
    assert np.isclose(sigma, 0.24 / np.sqrt(12))
