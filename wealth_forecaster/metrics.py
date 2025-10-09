"""Aggregation metrics for simulated paths."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


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

    if df.empty:
        return 0.0, 0.0

    group_cols: list[str] = ["run_id"]
    if "scenario" in df.columns:
        group_cols = ["scenario", "run_id"]

    sort_cols = [col for col in group_cols if col in df.columns]
    sort_cols.append("date")
    df_sorted = df.sort_values(sort_cols)

    for _key, group in df_sorted.groupby(group_cols):
        if group.empty:
            continue
        tail = group.tail(min(60, len(group)))
        monthly_returns.extend(tail["r_net"].astype(float).tolist())
        inflation_series = (
            tail["cpi_cum"].astype(float).pct_change().dropna().tolist()
        )
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


def _percentiles(series: Iterable[float]) -> Dict[str, float]:
    values = list(series)
    if not values:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    percentiles = np.nanpercentile(values, [10, 50, 90])
    return {
        "p10": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p90": float(percentiles[2]),
    }


def _build_summary(stats: pd.DataFrame, deposits: float) -> Dict:
    deposits_series = stats["deposits"]
    end_nominal = stats["end_nominal"]
    end_real = stats["end_real"]
    return {
        "deposits_total": float(deposits),
        "deposits_distribution": _percentiles(deposits_series),
        "end_nominal": _percentiles(end_nominal),
        "end_real": _percentiles(end_real),
        "capital_gains_nominal": _percentiles(end_nominal - deposits_series),
        "capital_gains_real": _percentiles(end_real - deposits_series),
        "perpetuity": {
            "nominal": {
                "gross": _percentiles(stats["perp_nominal_gross"]),
                "net": _percentiles(stats["perp_nominal_net"]),
            },
            "real": {
                "gross": _percentiles(stats["perp_real_gross"]),
                "net": _percentiles(stats["perp_real_net"]),
            },
        },
        "annuity": {
            "nominal": {
                "gross": _percentiles(stats["ann_nominal_gross"]),
                "net": _percentiles(stats["ann_nominal_net"]),
            },
            "real": {
                "gross": _percentiles(stats["ann_real_gross"]),
                "net": _percentiles(stats["ann_real_net"]),
            },
        },
    }


def aggregate(df_paths: pd.DataFrame, tax: float, withdraw_params: Dict) -> Dict:
    scenario_summaries: Dict[str, Dict] = {}
    scenario_stats_frames: list[pd.DataFrame] = []

    n_months = df_paths.groupby("date").ngroups
    ann_months = max(1, n_months)
    tax_rate = float(tax)

    for scenario, df_s in df_paths.groupby("scenario"):
        df_sorted = df_s.sort_values(["run_id", "date"])
        grouped = df_sorted.groupby("run_id")
        end_nom = grouped["value_nom"].last()
        end_real = grouped["value_real"].last()
        deposits = grouped["contrib_cum"].last()

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

        perp_nominal = np.array(
            [
                _payment_perpetuity(float(val), float(monthly_nom_perp))
                for val in end_nom
            ]
        )
        perp_real = np.array(
            [
                _payment_perpetuity(float(val), float(monthly_real_perp))
                for val in end_real
            ]
        )
        ann_nominal = np.array(
            [_payment_annuity(float(val), float(monthly_nom_ann), ann_months) for val in end_nom]
        )
        ann_real = np.array(
            [_payment_annuity(float(val), float(monthly_real_ann), ann_months) for val in end_real]
        )

        stats = pd.DataFrame(
            {
                "deposits": deposits.to_numpy(),
                "end_nominal": end_nom.to_numpy(),
                "end_real": end_real.to_numpy(),
                "perp_nominal_gross": perp_nominal,
                "perp_nominal_net": perp_nominal * (1 - tax_rate),
                "perp_real_gross": perp_real,
                "perp_real_net": perp_real * (1 - tax_rate),
                "ann_nominal_gross": ann_nominal,
                "ann_nominal_net": ann_nominal * (1 - tax_rate),
                "ann_real_gross": ann_real,
                "ann_real_net": ann_real * (1 - tax_rate),
            }
        )
        stats["scenario"] = scenario
        scenario_stats_frames.append(stats)
        avg_deposits = float(deposits.mean())
        summary = _build_summary(stats, avg_deposits)
        avg_end_nom = float(end_nom.mean())
        avg_end_real = float(end_real.mean())
        legacy_returns = avg_end_nom - avg_deposits
        summary.update(
            {
                "deposits": avg_deposits,
                "returns": float(legacy_returns),
                "end_value_nom": avg_end_nom,
                "end_value_real": avg_end_real,
                "withdrawal_nom_perp": float(np.mean(perp_nominal) * 12),
                "withdrawal_nom_perp_net": float(np.mean(perp_nominal * (1 - tax_rate)) * 12),
                "withdrawal_real_perp": float(np.mean(perp_real) * 12),
                "withdrawal_real_perp_net": float(np.mean(perp_real * (1 - tax_rate)) * 12),
                "withdrawal_nom_ann": float(np.mean(ann_nominal) * 12),
                "withdrawal_nom_ann_net": float(np.mean(ann_nominal * (1 - tax_rate)) * 12),
                "withdrawal_real_ann": float(np.mean(ann_real) * 12),
                "withdrawal_real_ann_net": float(np.mean(ann_real * (1 - tax_rate)) * 12),
                "market_growth_annual": float((1 + monthly_nom) ** 12 - 1),
                "market_growth_real_annual": float((1 + monthly_real) ** 12 - 1),
            }
        )
        scenario_summaries[scenario] = summary

    if scenario_stats_frames:
        combined = pd.concat(scenario_stats_frames, ignore_index=True)
        overall = _build_summary(combined, combined["deposits"].mean())
        avg_dep = float(combined["deposits"].mean())
        avg_end_nom = float(combined["end_nominal"].mean())
        avg_end_real = float(combined["end_real"].mean())
        monthly_nom_all, monthly_real_all = _estimate_rates(df_paths)
        overall.update(
            {
                "deposits": avg_dep,
                "returns": float(avg_end_nom - avg_dep),
                "end_value_nom": avg_end_nom,
                "end_value_real": avg_end_real,
                "withdrawal_nom_perp": float(combined["perp_nominal_gross"].mean() * 12),
                "withdrawal_nom_perp_net": float(combined["perp_nominal_net"].mean() * 12),
                "withdrawal_real_perp": float(combined["perp_real_gross"].mean() * 12),
                "withdrawal_real_perp_net": float(combined["perp_real_net"].mean() * 12),
                "withdrawal_nom_ann": float(combined["ann_nominal_gross"].mean() * 12),
                "withdrawal_nom_ann_net": float(combined["ann_nominal_net"].mean() * 12),
                "withdrawal_real_ann": float(combined["ann_real_gross"].mean() * 12),
                "withdrawal_real_ann_net": float(combined["ann_real_net"].mean() * 12),
                "market_growth_annual": float((1 + monthly_nom_all) ** 12 - 1),
                "market_growth_real_annual": float((1 + monthly_real_all) ** 12 - 1),
            }
        )
    else:
        overall = {
            "deposits_total": 0.0,
            "end_nominal": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
            "end_real": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
            "perpetuity": {
                "nominal": {
                    "gross": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                    "net": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                },
                "real": {
                    "gross": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                    "net": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                },
            },
            "annuity": {
                "nominal": {
                    "gross": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                    "net": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                },
                "real": {
                    "gross": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                    "net": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                },
            },
        }
        overall.update(
            {
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
        )

    return {
        "scenarios": scenario_summaries,
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
