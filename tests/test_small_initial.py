import pandas as pd

from wealth_forecaster.metrics import aggregate
from wealth_forecaster.simulate import simulate_paths
from wealth_forecaster.ui.callbacks import _config_from_inputs


def test_small_initial_no_contributions_produces_positive_paths():
    values = {
        "start_date": "2025-01",
        "horizon_years": 30,
        "initial_capital": 500,
        "monthly_contribution": 0,
        "annual_increase": 0,
        "seed_base": 1000,
        "runs_per_scenario": 100,
        "ter": 0.3,
        "tax_rate": 25,
        "inflation_mean": 2,
        "inflation_sigma": 1,
        "volatility_multiplier": 1,
        "inflation_stochastic": ["on"],
    }
    cfg = _config_from_inputs(values)

    frames = [simulate_paths(cfg, scenario) for scenario in cfg["scenarios"]]
    df_all = pd.concat(frames, ignore_index=True)

    assert not df_all.empty
    assert (df_all["value_nom"] > 0).all()

    agg = aggregate(
        df_all,
        cfg["costs_taxes"]["withholding_tax_rate"],
        cfg["withdrawal_params"],
    )
    moderate_summary = agg["scenarios"].get("moderate", {})
    assert moderate_summary.get("deposits_total", 0) > 0
