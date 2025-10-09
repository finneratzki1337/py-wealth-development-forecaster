"""Tests for metrics aggregation utilities."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from wealth_forecaster import metrics


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    product = np.prod([1 + v for v in values])
    return product ** (1 / len(values)) - 1


def test_estimate_rates_handles_multiple_scenarios_independently() -> None:
    dates = pd.date_range("2025-01-01", periods=3, freq="ME")

    scenario_data = [
        (
            "optimistic",
            1,
            [0.01, 0.015, 0.02],
            list(np.cumprod([1.01, 1.01, 1.01])),
        ),
        (
            "pessimistic",
            1,
            [0.03, 0.025, 0.02],
            list(np.cumprod([1.015, 1.015, 1.015])),
        ),
    ]

    records: list[dict[str, float | str | pd.Timestamp]] = []
    inflation_rates: list[float] = []
    return_rates: list[float] = []

    for scenario, run_id, returns, cpi in scenario_data:
        prev_cpi = 1.0
        for idx, (ret, cpi_val) in enumerate(zip(returns, cpi)):
            records.append(
                {
                    "scenario": scenario,
                    "run_id": run_id,
                    "date": dates[idx],
                    "r_net": ret,
                    "cpi_cum": cpi_val,
                }
            )
            inflation_rates.append(cpi_val / prev_cpi - 1)
            return_rates.append(ret)
            prev_cpi = cpi_val

    df = pd.DataFrame.from_records(records)

    monthly_return, real_monthly = metrics._estimate_rates(df)

    expected_return = _geometric_mean(return_rates)
    expected_inflation = _geometric_mean(inflation_rates)
    expected_real = ((1 + expected_return) / (1 + expected_inflation)) - 1

    assert math.isclose(monthly_return, expected_return, rel_tol=1e-12)
    assert math.isclose(real_monthly, expected_real, rel_tol=1e-12)


def test_estimate_rates_empty_frame() -> None:
    df = pd.DataFrame(columns=["run_id", "date", "r_net", "cpi_cum"])
    monthly_return, real_monthly = metrics._estimate_rates(df)
    assert monthly_return == 0.0
    assert real_monthly == 0.0
