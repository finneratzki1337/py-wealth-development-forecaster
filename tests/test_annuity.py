from __future__ import annotations

import numpy as np
import pandas as pd

from wealth_forecaster.metrics import aggregate


def test_annuity_perpetuity_calculations():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    monthly_rate = 0.01
    value_nom = 1000 * (1 + monthly_rate) ** np.arange(1, 13)
    df = pd.DataFrame(
        {
            "date": np.tile(dates, 1),
            "scenario": "test",
            "run_id": 1,
            "seed_used": 1,
            "value_nom": value_nom,
            "value_real": value_nom,
            "contrib_cum": np.full(12, 1000.0),
            "cpi_cum": np.ones(12),
            "r_net": np.full(12, monthly_rate),
        }
    )

    agg = aggregate(df, tax=0.0, withdraw_params={"r_nom_perp": None, "r_real_perp": None, "r_nom_ann": None, "r_real_ann": None})
    summary = agg["scenarios"]["test"]
    perp_expected = value_nom[-1] * monthly_rate * 12
    ann_expected = value_nom[-1] * (monthly_rate / (1 - (1 + monthly_rate) ** -12)) * 12
    assert np.isclose(summary["withdrawal_nom_perp"], perp_expected, rtol=1e-5)
    assert np.isclose(summary["withdrawal_nom_ann"], ann_expected, rtol=1e-5)


def test_perpetuity_clamped_at_zero_for_negative_rates():
    dates = pd.date_range("2025-01-01", periods=24, freq="MS")
    monthly_rate = -0.005
    value_nom = 100000 * (1 + monthly_rate) ** np.arange(1, len(dates) + 1)
    df = pd.DataFrame(
        {
            "date": dates,
            "scenario": "test",
            "run_id": 1,
            "seed_used": 1,
            "value_nom": value_nom,
            "value_real": value_nom,
            "contrib_cum": np.zeros(len(dates)),
            "cpi_cum": np.ones(len(dates)),
            "r_net": np.full(len(dates), monthly_rate),
        }
    )

    agg = aggregate(df, tax=0.0, withdraw_params={"r_nom_perp": None, "r_real_perp": None, "r_nom_ann": None, "r_real_ann": None})
    summary = agg["scenarios"]["test"]
    perp_stats = summary["perpetuity"]["nominal"]["gross"]
    ann_stats = summary["annuity"]["nominal"]["gross"]

    assert perp_stats["p50"] == 0.0
    assert ann_stats["p50"] > 0.0
