"""Aggregation metrics for simulated paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class ScenarioSummary:
    deposits: float
    returns: float
    end_value_nom: float
    end_value_real: float
    withdrawal_nom_perp: float
    withdrawal_nom_perp_net: float
    withdrawal_real_perp: float
    withdrawal_real_perp_net: float
    withdrawal_nom_ann: float
    withdrawal_nom_ann_net: float
    withdrawal_real_ann: float
    withdrawal_real_ann_net: float


def _geometric_mean(series: Iterable[float]) -> float:
    series = list(series)
    if not series:
        return 0.0
    series = [1 + x for x in series if x > -1]
    if not series:
        return 0.0
    product = np.prod(series)
    return product ** (1 / len(series)) - 1


def _mean_inflation(last_cpi: pd.Series) -> float:
    if last_cpi.empty:
        return 0.0
    pct = last_cpi.pct_change().dropna()
    if pct.empty:
        return 0.0
    return float(pct.mean())


def _estimate_rates(df: pd.DataFrame) -> tuple[float, float]:
    monthly_returns: list[float] = []
    monthly_inflation: list[float] = []
    for _run_id, group in df.sort_values("date").groupby("run_id"):
        tail = group.tail(min(60, len(group)))
        monthly_returns.extend(tail["r_net"].tolist())
        inflation_series = tail["cpi_cum"].pct_change().dropna().tolist()
        monthly_inflation.extend(inflation_series)

    monthly_return = _geometric_mean(monthly_returns)
    inflation_rate = _geometric_mean(monthly_inflation)
    real_monthly = ((1 + monthly_return) / (1 + inflation_rate)) - 1
    return monthly_return, real_monthly


def _payment_perpetuity(value: float, monthly_rate: float) -> float:
    if monthly_rate <= 0:
        return 0.0
    return value * monthly_rate


def _payment_annuity(value: float, monthly_rate: float, months: int) -> float:
    if months <= 0:
        return 0.0
    if abs(monthly_rate) < 1e-9:
        return value / months
    factor = monthly_rate / (1 - (1 + monthly_rate) ** (-months))
    return value * factor


def aggregate(df_paths: pd.DataFrame, tax: float, withdraw_params: Dict) -> Dict:
    summaries: Dict[str, ScenarioSummary] = {}
    n_months = df_paths.groupby("date").ngroups
    ann_months = max(1, n_months)

    for scenario, df_s in df_paths.groupby("scenario"):
        df_sorted = df_s.sort_values(["run_id", "date"])
        end_nom = df_sorted.groupby("run_id")[["value_nom"]].tail(1)["value_nom"].mean()
        end_real = df_sorted.groupby("run_id")[["value_real"]].tail(1)["value_real"].mean()
        deposits = df_sorted.groupby("run_id")[["contrib_cum"]].tail(1)["contrib_cum"].mean()
        returns_value = end_nom - deposits

        monthly_nom, monthly_real = _estimate_rates(df_sorted)

        r_nom_perp = withdraw_params.get("r_nom_perp")
        if r_nom_perp is None:
            r_nom_perp = (1 + monthly_nom) ** 12 - 1
        r_nom_ann = withdraw_params.get("r_nom_ann")
        if r_nom_ann is None:
            r_nom_ann = r_nom_perp
        r_real_perp = withdraw_params.get("r_real_perp")
        if r_real_perp is None:
            r_real_perp = (1 + monthly_real) ** 12 - 1
        r_real_ann = withdraw_params.get("r_real_ann")
        if r_real_ann is None:
            r_real_ann = r_real_perp

        monthly_nom_perp = (1 + r_nom_perp) ** (1 / 12) - 1
        monthly_real_perp = (1 + r_real_perp) ** (1 / 12) - 1
        monthly_nom_ann = (1 + r_nom_ann) ** (1 / 12) - 1
        monthly_real_ann = (1 + r_real_ann) ** (1 / 12) - 1

        perp_nominal = _payment_perpetuity(end_nom, monthly_nom_perp) * 12
        perp_real = _payment_perpetuity(end_real, monthly_real_perp) * 12
        ann_nominal = _payment_annuity(end_nom, monthly_nom_ann, ann_months) * 12
        ann_real = _payment_annuity(end_real, monthly_real_ann, ann_months) * 12

        tax_rate = float(tax)
        perp_nominal_net = perp_nominal * (1 - tax_rate)
        perp_real_net = perp_real * (1 - tax_rate)
        ann_nominal_net = ann_nominal * (1 - tax_rate)
        ann_real_net = ann_real * (1 - tax_rate)

        summaries[scenario] = ScenarioSummary(
            deposits=float(deposits),
            returns=float(returns_value),
            end_value_nom=float(end_nom),
            end_value_real=float(end_real),
            withdrawal_nom_perp=float(perp_nominal),
            withdrawal_nom_perp_net=float(perp_nominal_net),
            withdrawal_real_perp=float(perp_real),
            withdrawal_real_perp_net=float(perp_real_net),
            withdrawal_nom_ann=float(ann_nominal),
            withdrawal_nom_ann_net=float(ann_nominal_net),
            withdrawal_real_ann=float(ann_real),
            withdrawal_real_ann_net=float(ann_real_net),
        )

    if summaries:
        overall = {
            "deposits": float(np.mean([s.deposits for s in summaries.values()])),
            "returns": float(np.mean([s.returns for s in summaries.values()])),
            "end_value_nom": float(np.mean([s.end_value_nom for s in summaries.values()])),
            "end_value_real": float(np.mean([s.end_value_real for s in summaries.values()])),
            "withdrawal_nom_perp": float(
                np.mean([s.withdrawal_nom_perp for s in summaries.values()])
            ),
            "withdrawal_nom_perp_net": float(
                np.mean([s.withdrawal_nom_perp_net for s in summaries.values()])
            ),
            "withdrawal_real_perp": float(
                np.mean([s.withdrawal_real_perp for s in summaries.values()])
            ),
            "withdrawal_real_perp_net": float(
                np.mean([s.withdrawal_real_perp_net for s in summaries.values()])
            ),
            "withdrawal_nom_ann": float(
                np.mean([s.withdrawal_nom_ann for s in summaries.values()])
            ),
            "withdrawal_nom_ann_net": float(
                np.mean([s.withdrawal_nom_ann_net for s in summaries.values()])
            ),
            "withdrawal_real_ann": float(
                np.mean([s.withdrawal_real_ann for s in summaries.values()])
            ),
            "withdrawal_real_ann_net": float(
                np.mean([s.withdrawal_real_ann_net for s in summaries.values()])
            ),
        }
    else:
        overall = {
            "deposits": 0.0,
            "returns": 0.0,
            "end_value_nom": 0.0,
            "end_value_real": 0.0,
            "withdrawal_nom_perp": 0.0,
            "withdrawal_nom_perp_net": 0.0,
            "withdrawal_real_perp": 0.0,
            "withdrawal_real_perp_net": 0.0,
            "withdrawal_nom_ann": 0.0,
            "withdrawal_nom_ann_net": 0.0,
            "withdrawal_real_ann": 0.0,
            "withdrawal_real_ann_net": 0.0,
        }

    return {
        "scenarios": {k: vars(v) for k, v in summaries.items()},
        "overall": overall,
    }


def target_flags(df_paths: pd.DataFrame, target_end_real: float | None, target_income_real: float | None) -> Dict:
    flags = {}
    if target_end_real is not None:
        end_values = df_paths.groupby(["scenario", "run_id"])["value_real"].last()
        flags["target_end_real"] = {
            scenario: (values.mean() >= target_end_real)
            for scenario, values in end_values.groupby(level=0)
        }
    if target_income_real is not None:
        last_real = df_paths.groupby(["scenario", "run_id"])["value_real"].last()
        income = last_real * 0.04
        flags["target_income_real"] = {
            scenario: (values.mean() >= target_income_real)
            for scenario, values in income.groupby(level=0)
        }
    return flags
