"""Excel export for simulation runs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, IO, Union

import pandas as pd


def _pivot_paths(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="date",
        columns=["scenario", "run_id"],
        values=value_col,
        aggfunc="first",
    )
    pivot.columns = [f"{scenario.title()} Run {run_id}" for scenario, run_id in pivot.columns]
    scenario_means = (
        df.groupby(["date", "scenario"])[value_col].mean().unstack("scenario")
    )
    for scenario in scenario_means.columns:
        pivot[f"{scenario.title()} Avg"] = scenario_means[scenario]
    return pivot


PathLike = Union[str, Path, IO[bytes]]


def export_xlsx(df_paths: pd.DataFrame, agg: Dict, path: PathLike) -> None:
    writer: pd.ExcelWriter
    if hasattr(path, "write"):
        writer = pd.ExcelWriter(path, engine="xlsxwriter")
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pd.ExcelWriter(path, engine="xlsxwriter")

    nominal = _pivot_paths(df_paths, "value_nom")
    real = _pivot_paths(df_paths, "value_real")

    agg_df = pd.DataFrame(agg["scenarios"]).T
    overall = pd.Series(agg["overall"], name="overall").to_frame()
    meta = agg.get("meta", {})

    with writer:
        df_paths.to_excel(writer, sheet_name="Raw Paths", index=False)
        nominal.to_excel(writer, sheet_name="Nominal Paths")
        real.to_excel(writer, sheet_name="Real Paths")
        agg_df.to_excel(writer, sheet_name="Aggregates")
        overall.to_excel(writer, sheet_name="Overall")
        if meta:
            meta_copy = dict(meta)
            config_json = meta_copy.pop("config_json", None)
            warnings = meta_copy.pop("warnings", [])
            meta_df = pd.DataFrame(meta_copy.items(), columns=["key", "value"])
            meta_df.to_excel(writer, sheet_name="Meta", index=False)
            worksheet = writer.sheets["Meta"]
            row = len(meta_df) + 2
            if config_json:
                worksheet.write(row, 0, "config_json")
                worksheet.write(row, 1, config_json)
                row += 1
            if warnings:
                worksheet.write(row, 0, "warnings")
                worksheet.write(row, 1, "; ".join(warnings))
