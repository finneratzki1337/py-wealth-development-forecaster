from __future__ import annotations

import pandas as pd

from wealth_forecaster.config import default_config
from wealth_forecaster.simulate import simulate_paths


def test_simulation_reproducibility():
    cfg = default_config()
    df1 = simulate_paths(cfg, "moderate")
    df2 = simulate_paths(cfg, "moderate")
    pd.testing.assert_frame_equal(df1, df2)
