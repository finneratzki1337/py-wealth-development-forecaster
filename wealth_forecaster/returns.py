"""Return generation helpers."""
from __future__ import annotations

import numpy as np


def monthly_params(mu_y: float, sigma_y: float, mean_type: str) -> tuple[float, float]:
    """Convert annual mean/std to monthly equivalents.

    Parameters
    ----------
    mu_y: float
        Annual mean return.
    sigma_y: float
        Annual standard deviation of returns.
    mean_type: str
        Either "arithmetic" or "geometric".
    """

    mean_type = (mean_type or "arithmetic").lower()
    if mean_type == "geometric":
        mu_m = (1 + mu_y) ** (1 / 12) - 1
    else:
        mu_m = mu_y / 12

    sigma_m = sigma_y / np.sqrt(12.0)
    return mu_m, sigma_m


def _monthly_fee_factor(ter_pa: float) -> float:
    return (1.0 + float(ter_pa)) ** (1 / 12.0) - 1.0


def sample_monthly_returns(
    n: int,
    mu_m: float,
    sigma_m: float,
    ter_pa: float,
    distribution: str,
    truncate_at_minus_100: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample monthly *net* returns after continuous fee drag."""

    fee = _monthly_fee_factor(ter_pa)
    distribution = (distribution or "normal_arith").lower()
    if distribution == "lognormal_match":
        exp_r = 1 + mu_m
        var_r = sigma_m**2
        if exp_r <= 0:
            raise ValueError("Mean return must be greater than -100% for lognormal model")
        sigma2 = np.log(1 + var_r / (exp_r**2))
        mu = np.log(exp_r) - 0.5 * sigma2
        gross_factors = np.exp(rng.normal(mu, np.sqrt(sigma2), size=n))
        gross_returns = gross_factors - 1.0
    else:
        gross_returns = rng.normal(mu_m, sigma_m, size=n)

    if truncate_at_minus_100:
        gross_returns = np.maximum(gross_returns, -0.999999)

    net_factors = (1.0 + gross_returns) * (1.0 - fee)
    return net_factors - 1.0
