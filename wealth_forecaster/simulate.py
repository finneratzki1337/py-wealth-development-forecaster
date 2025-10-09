"""Simulation engine for wealth development."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from numpy.random import SeedSequence

from . import config as cfg_mod
from . import inflation as infl_mod
from . import returns as ret_mod


@dataclass(frozen=True)
class ContributionEvent:
    date: pd.Period
    new_monthly: float


@dataclass(frozen=True)
class PauseWindow:
    start: pd.Period
    end: pd.Period


def _parse_period(date_str: str) -> pd.Period:
    return pd.Period(str(date_str), freq="M")


def _build_contribution_series(cfg: Dict) -> pd.Series:
    horizon_months = int(cfg["horizon_years"]) * 12
    start = _parse_period(cfg.get("start_date", "2025-01"))
    periods = pd.period_range(start=start, periods=horizon_months, freq="M")

    cash_cfg = cfg.get("cash", {})
    base_monthly = float(cash_cfg.get("monthly", 0.0))
    annual_increase = float(cash_cfg.get("annual_increase", 0.0))

    pause_windows: List[PauseWindow] = []
    for start_s, end_s in cash_cfg.get("pauses", []):
        pause_windows.append(PauseWindow(_parse_period(start_s), _parse_period(end_s)))

    events: List[ContributionEvent] = []
    for event in cash_cfg.get("events", []):
        events.append(ContributionEvent(_parse_period(event["date"]), float(event["new_monthly"])))

    values = []
    current = base_monthly
    for idx, period in enumerate(periods):
        if idx > 0 and period.month == 1:
            current *= (1.0 + annual_increase)
        for event in events:
            if event.date == period:
                current = event.new_monthly
        paused = any(window.start <= period <= window.end for window in pause_windows)
        values.append(0.0 if paused else current)

    series = pd.Series(values, index=periods, name="contribution")
    return series


def _scenario_blocks(scenario_cfg: Dict, sigma_scale: float) -> Iterable[tuple[float, float]]:
    mu_blocks = scenario_cfg.get("mu_pa_blocks", [])
    sigma = float(scenario_cfg.get("sigma_pa", 0.0)) * float(sigma_scale)
    for mu in mu_blocks:
        yield float(mu), sigma


def _scenario_returns(
    cfg: Dict, scenario_cfg: Dict, rng: np.random.Generator, sigma_scale: float
) -> np.ndarray:
    mean_type = cfg.get("return_model", {}).get("mean_type", "arithmetic")
    distribution = cfg.get("return_model", {}).get("distribution", "normal_arith")
    truncate = cfg.get("return_model", {}).get("truncate_at_minus_100", True)
    ter = cfg.get("costs_taxes", {}).get("ter_pa", 0.0)
    optimism_adj = cfg.get("optimism_adjustment", 0.0)

    horizon_months = int(cfg["horizon_years"]) * 12
    returns: List[float] = []
    for mu_y, sigma_y in _scenario_blocks(scenario_cfg, sigma_scale):
        mu_m, sigma_m = ret_mod.monthly_params(mu_y, sigma_y, mean_type)
        block_months = min(60, horizon_months - len(returns))
        if block_months <= 0:
            break
        block_rets = ret_mod.sample_monthly_returns(
            block_months,
            mu_m,
            sigma_m,
            ter,
            distribution,
            truncate,
            rng,
        )
        returns.extend(block_rets.tolist())
    if len(returns) < horizon_months:
        # Extend using final block parameters
        mu_y, sigma_y = list(_scenario_blocks(scenario_cfg, sigma_scale))[-1]
        mu_m, sigma_m = ret_mod.monthly_params(mu_y, sigma_y, mean_type)
        missing = horizon_months - len(returns)
        returns.extend(
            ret_mod.sample_monthly_returns(
                missing, mu_m, sigma_m, ter, distribution, truncate, rng
            ).tolist()
        )
    
    # Apply optimism adjustment (convert annual to monthly)
    optimism_monthly = optimism_adj / 12.0
    returns_array = np.array(returns[:horizon_months])
    if optimism_monthly != 0.0:
        returns_array = returns_array + optimism_monthly
    
    return returns_array


def _inflation_factors(cfg: Dict, rng: np.random.Generator) -> np.ndarray:
    infl_cfg = cfg.get("inflation", {})
    years = int(cfg["horizon_years"])
    if infl_cfg.get("stochastic", True):
        yearly = infl_mod.sample_yearly_inflation(
            years,
            float(infl_cfg.get("mean_pa", 0.0)),
            float(infl_cfg.get("sigma_pa", 0.0)),
            rng,
        )
    else:
        yearly = np.repeat(float(infl_cfg.get("mean_pa", 0.0)), years)
    monthly = infl_mod.expand_to_monthly_factors(yearly)
    return monthly


def _simulate_single_run(
    run_id: int,
    path_seed: int,
    inflation_seed: int,
    scenario_id: str,
    cfg: Dict,
    scenario_cfg: Dict,
    contribs: np.ndarray,
    contrib_cumsum: np.ndarray,
    dates: pd.DatetimeIndex,
    initial_capital: float,
    sigma_scale: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(path_seed)
    scenario_returns = _scenario_returns(cfg, scenario_cfg, rng, sigma_scale)
    horizon_months = len(contribs)
    inflation_rng = np.random.default_rng(inflation_seed)
    monthly_inflation = _inflation_factors(cfg, inflation_rng)[:horizon_months]
    cpi_cum = np.cumprod(monthly_inflation)

    value = initial_capital
    values_nom = np.empty(horizon_months)
    values_real = np.empty(horizon_months)
    r_nets = scenario_returns[:horizon_months]
    for idx in range(horizon_months):
        value = (value + contribs[idx]) * (1.0 + r_nets[idx])
        values_nom[idx] = value
        values_real[idx] = value / cpi_cum[idx]

    df_run = pd.DataFrame(
        {
            "date": dates,
            "scenario": scenario_id,
            "run_id": run_id,
            "seed_used": path_seed,
            "value_nom": values_nom,
            "value_real": values_real,
            "contrib_cum": contrib_cumsum,
            "cpi_cum": cpi_cum,
            "r_net": r_nets,
        }
    )
    return df_run


def simulate_paths(cfg: Dict, scenario_id: str) -> pd.DataFrame:
    """Simulate runs for a scenario using deterministic parallel seeds."""
    cfg = cfg_mod.canonicalize(cfg)
    scenario_cfg = cfg.get("scenarios", {}).get(scenario_id)
    if scenario_cfg is None:
        raise KeyError(f"Scenario '{scenario_id}' not defined")

    contribution_series = _build_contribution_series(cfg)
    horizon_months = contribution_series.shape[0]
    start_date = contribution_series.index[0]
    dates = contribution_series.index.to_timestamp(how="start")

    seed_base = int(cfg.get("seed_base", 1000))
    scenario_offset = abs(hash(scenario_id)) % (2**16)
    runs_per_scenario = max(100, int(cfg.get("runs_per_scenario", 100)))

    initial_capital = float(cfg.get("cash", {}).get("initial", 0.0))
    contribs = contribution_series.to_numpy()
    contrib_cumsum = initial_capital + contribution_series.cumsum().to_numpy()

    base_sequence = SeedSequence(seed_base + scenario_offset)
    child_sequences = base_sequence.spawn(runs_per_scenario)
    seeds: List[tuple[int, int, int]] = []
    for run_idx, child_sequence in enumerate(child_sequences, start=1):
        state = child_sequence.generate_state(2, dtype=np.uint32)
        seeds.append((run_idx, int(state[0]), int(state[1])))

    sigma_scale = float(cfg.get("volatility_multiplier", 1.0))

    def _task(args: tuple[int, int, int]) -> pd.DataFrame:
        run_id, path_seed, inflation_seed = args
        return _simulate_single_run(
            run_id,
            path_seed,
            inflation_seed,
            scenario_id,
            cfg,
            scenario_cfg,
            contribs,
            contrib_cumsum,
            dates,
            initial_capital,
            sigma_scale,
        )

    if runs_per_scenario == 0:
        return pd.DataFrame()

    max_workers = min(len(seeds), 8) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frames = list(executor.map(_task, seeds))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.sort_values(["run_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
