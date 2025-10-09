"""Quick test to verify the simulation pipeline produces results."""

import pandas as pd

from wealth_forecaster.config import canonicalize, default_config
from wealth_forecaster.metrics import aggregate
from wealth_forecaster.simulate import simulate_paths


def test_basic_simulation():
    cfg = canonicalize(default_config())
    cfg["runs_per_scenario"] = 10

    frames = []
    for scenario in cfg["scenarios"]:
        df = simulate_paths(cfg, scenario)
        assert not df.empty
        assert {"date", "value_nom", "value_real"}.issubset(df.columns)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    agg = aggregate(
        df_all,
        cfg["costs_taxes"]["withholding_tax_rate"],
        cfg["withdrawal_params"],
    )

    assert agg["scenarios"], "Expected scenario summaries in aggregate output"
