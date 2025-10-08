"""Inflation sampling utilities."""
from __future__ import annotations

import numpy as np


MIN_INFLATION = -0.99


def sample_yearly_inflation(years: int, mean_pa: float, sigma_pa: float, rng: np.random.Generator) -> np.ndarray:
    """Draw annual inflation rates with truncation."""
    draws = rng.normal(mean_pa, sigma_pa, size=int(years))
    return np.maximum(draws, MIN_INFLATION)


def expand_to_monthly_factors(yearly_pi: np.ndarray) -> np.ndarray:
    """Convert yearly inflation rates to monthly factors."""
    yearly_pi = np.asarray(yearly_pi)
    if yearly_pi.ndim != 1:
        raise ValueError("Yearly inflation must be 1D")
    factors = np.empty(yearly_pi.size * 12, dtype=float)
    for idx, pi in enumerate(yearly_pi):
        factor = (1.0 + pi) ** (1.0 / 12.0)
        start = idx * 12
        factors[start : start + 12] = factor
    return factors
