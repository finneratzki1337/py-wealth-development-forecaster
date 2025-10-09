"""Excel export for simulation runs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, IO, List, Tuple, Union

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


def _flatten_config(cfg: Dict, prefix: str = "") -> List[Tuple[str, object]]:
    items: List[Tuple[str, object]] = []
    for key, value in sorted(cfg.items()):
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            items.extend(_flatten_config(value, full_key))
        else:
            items.append((full_key, value))
    return items


def export_xlsx(
    df_paths: pd.DataFrame,
    agg: Dict,
    path: PathLike,
    config: Dict | None = None,
) -> None:
    """Export simulation results to Excel with monthly portfolio values and parameters."""
    writer: pd.ExcelWriter
    if hasattr(path, "write"):
        writer = pd.ExcelWriter(path, engine="xlsxwriter")
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pd.ExcelWriter(path, engine="xlsxwriter")

    # Ensure date column is datetime
    if not df_paths.empty and "date" in df_paths.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_paths["date"]):
            df_paths = df_paths.copy()
            df_paths["date"] = pd.to_datetime(df_paths["date"])

    # Create pivot tables for monthly values
    nominal = _pivot_paths(df_paths, "value_nom") if not df_paths.empty else pd.DataFrame()
    real = _pivot_paths(df_paths, "value_real") if not df_paths.empty else pd.DataFrame()

    scenarios = agg.get("scenarios", {}) if isinstance(agg, dict) else {}
    agg_df = pd.DataFrame(scenarios).T if scenarios else pd.DataFrame()
    overall_data = agg.get("overall") if isinstance(agg, dict) else None
    overall = (
        pd.Series(overall_data, name="overall").to_frame()
        if isinstance(overall_data, dict) and overall_data
        else pd.DataFrame()
    )
    meta = agg.get("meta", {})

    with writer:
        # 1. Parameters/Inputs first for reproducibility
        if config:
            config_rows = _flatten_config(config)
            config_df = pd.DataFrame(config_rows, columns=["Parameter", "Value"])
            config_df.to_excel(writer, sheet_name="Parameters", index=False)
            # Auto-adjust column widths
            worksheet = writer.sheets["Parameters"]
            worksheet.set_column('A:A', 35)
            worksheet.set_column('B:B', 25)
        
        # 2. Monthly portfolio values (pivoted by scenario and run)
        if not nominal.empty:
            nominal.to_excel(writer, sheet_name="Monthly Nominal Values")
            worksheet = writer.sheets["Monthly Nominal Values"]
            worksheet.set_column('A:A', 12)  # Date column
        
        if not real.empty:
            real.to_excel(writer, sheet_name="Monthly Real Values")
            worksheet = writer.sheets["Monthly Real Values"]
            worksheet.set_column('A:A', 12)  # Date column
        
        # 3. Summary statistics
        if not agg_df.empty:
            agg_df.to_excel(writer, sheet_name="Summary Statistics")
        
        if not overall.empty:
            overall.to_excel(writer, sheet_name="Overall Statistics")
        
        # 4. Raw data (all individual paths)
        if not df_paths.empty:
            df_paths.to_excel(writer, sheet_name="Raw Simulation Paths", index=False)
        
        # 5. Metadata
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
