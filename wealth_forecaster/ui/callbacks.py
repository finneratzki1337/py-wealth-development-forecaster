"""Dash callbacks for the wealth forecaster UI."""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

from ..config import canonicalize, default_config
from ..export import export_xlsx
from ..hashing import run_hash
from ..metrics import aggregate
from ..simulate import simulate_paths
from ..validate import validate_inputs


SCENARIO_COLORS: Dict[str, str] = {
    "optimistic": "#ff6ec7",
    "moderate": "#2de2e6",
    "pessimistic": "#f9f871",
}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color (e.g., '#ff00aa') to an rgba string with the given alpha."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got '{hex_color}'")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _config_from_inputs(values: Dict[str, Any]) -> Dict[str, Any]:
    cfg = canonicalize(default_config())
    cfg["start_date"] = values["start_date"]
    cfg["horizon_years"] = int(values["horizon_years"])
    cfg["seed_base"] = int(values["seed_base"])

    cash = cfg["cash"]
    cash["initial"] = float(values["initial_capital"])
    cash["monthly"] = float(values["monthly_contribution"])
    cash["annual_increase"] = float(values["annual_increase"]) / 100.0
    cash["pauses"] = []
    cash["events"] = []

    costs = cfg["costs_taxes"]
    costs["ter_pa"] = float(values["ter"]) / 100.0
    costs["withholding_tax_rate"] = float(values["tax_rate"]) / 100.0

    inflation = cfg["inflation"]
    inflation["mean_pa"] = float(values["inflation_mean"]) / 100.0
    inflation["sigma_pa"] = float(values["inflation_sigma"]) / 100.0
    inflation["stochastic"] = "on" in values["inflation_stochastic"]

    return cfg


def _make_figure(df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(family="Courier New, monospace", size=20, color="#f9f871"),
            ),
            template="plotly_dark",
            paper_bgcolor="#0d1026",
            plot_bgcolor="#0d1026",
            font=dict(family="Courier New, monospace", color="#f4f4f4"),
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.update_xaxes(gridcolor="#1f2f4a", zeroline=False, linecolor="#2a3352", showline=True)
        fig.update_yaxes(
            gridcolor="#1f2f4a",
            zeroline=False,
            linecolor="#2a3352",
            showline=True,
            tickprefix="$" if "value" in value_col else "",
        )
        return fig

    for scenario, df_s in df.groupby("scenario"):
        color = SCENARIO_COLORS.get(scenario, "#cccccc")
        run_color = _hex_to_rgba(color, 0.3)
        shade_color = _hex_to_rgba(color, 0.18)

        df_sorted = df_s.sort_values(["run_id", "date"])
        for run_id, df_run in df_sorted.groupby("run_id"):
            hover = (
                f"<b>{scenario.title()}</b><br>%{{x|%Y-%m}}<br>$%{{y:,.0f}}<extra></extra>"
                if "value" in value_col
                else f"<b>{scenario.title()}</b><br>%{{x|%Y-%m}}<br>%{{y:,.2f}}<extra></extra>"
            )
            fig.add_trace(
                go.Scatter(
                    x=df_run["date"],
                    y=df_run[value_col],
                    name=f"{scenario.title()} Run {run_id}",
                    line=dict(width=0.8, color=run_color),
                    mode="lines",
                    opacity=0.25,
                    hovertemplate=hover,
                    legendgroup=scenario,
                    showlegend=False,
                )
            )
        grouped = df_sorted.groupby("date")[value_col]
        mean_vals = grouped.mean()
        min_vals = grouped.min()
        max_vals = grouped.max()
        fig.add_trace(
            go.Scatter(
                x=mean_vals.index,
                y=max_vals,
                line=dict(width=0),
                hoverinfo="skip",
                legendgroup=scenario,
                showlegend=False,
                name=f"{scenario.title()} High",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mean_vals.index,
                y=min_vals,
                fill="tonexty",
                fillcolor=shade_color,
                line=dict(width=0),
                hoverinfo="skip",
                name=f"{scenario.title()} Range",
                showlegend=False,
                legendgroup=scenario,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mean_vals.index,
                y=mean_vals,
                name=f"{scenario.title()} Avg",
                line=dict(width=4, color=color),
                mode="lines",
                legendgroup=scenario,
                hovertemplate=(
                    f"<b>{scenario.title()} Avg</b><br>%{{x|%Y-%m}}<br>$%{{y:,.0f}}<extra></extra>"
                    if "value" in value_col
                    else f"<b>{scenario.title()} Avg</b><br>%{{x|%Y-%m}}<br>%{{y:,.2f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Courier New, monospace", size=20, color="#f9f871"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0d1026",
        plot_bgcolor="#0d1026",
        font=dict(family="Courier New, monospace", color="#f4f4f4"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(13,16,38,0.7)",
            bordercolor="#2a3352",
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(
        gridcolor="#1f2f4a",
        zeroline=False,
        linecolor="#2a3352",
        showline=True,
    )
    fig.update_yaxes(
        gridcolor="#1f2f4a",
        zeroline=False,
        linecolor="#2a3352",
        showline=True,
        tickprefix="$" if "value" in value_col else "",
    )
    return fig


def _build_summary_table(agg: Dict) -> List[html.Tr]:
    header = html.Thead(
        html.Tr(
            [
                html.Th("Scenario"),
                html.Th("Deposits"),
                html.Th("End Nominal"),
                html.Th("End Real"),
                html.Th("Perpetuity (Nom)", title="Gross annual perpetuity"),
                html.Th("Perpetuity Net"),
                html.Th("Annuity (Nom)"),
                html.Th("Annuity Net"),
            ]
        )
    )
    body_rows = []
    for scenario, metrics in agg.get("scenarios", {}).items():
        body_rows.append(
            html.Tr(
                [
                    html.Th(scenario.title()),
                    html.Td(f"${metrics['deposits']:,.0f}"),
                    html.Td(f"${metrics['end_value_nom']:,.0f}"),
                    html.Td(f"${metrics['end_value_real']:,.0f}"),
                    html.Td(f"${metrics['withdrawal_nom_perp']:,.0f}"),
                    html.Td(f"${metrics['withdrawal_nom_perp_net']:,.0f}"),
                    html.Td(f"${metrics['withdrawal_nom_ann']:,.0f}"),
                    html.Td(f"${metrics['withdrawal_nom_ann_net']:,.0f}"),
                ]
            )
        )
    body = html.Tbody(body_rows)
    return [header, body]


def register_callbacks(app):
    @app.callback(
        Output("config-store", "data"),
        Input("run-button", "n_clicks"),
        State("start-date", "value"),
        State("horizon-years", "value"),
        State("initial-capital", "value"),
        State("monthly-contribution", "value"),
        State("annual-increase", "value"),
        State("seed-base", "value"),
        State("ter", "value"),
        State("tax-rate", "value"),
        State("inflation-mean", "value"),
        State("inflation-sigma", "value"),
        State("inflation-stochastic", "value"),
        prevent_initial_call=True,
    )
    def update_config(
        _n,
        start_date,
        horizon_years,
        initial_capital,
        monthly_contribution,
        annual_increase,
        seed_base,
        ter,
        tax_rate,
        inflation_mean,
        inflation_sigma,
        inflation_stochastic,
    ):
        values = {
            "start_date": start_date,
            "horizon_years": horizon_years,
            "initial_capital": initial_capital,
            "monthly_contribution": monthly_contribution,
            "annual_increase": annual_increase,
            "seed_base": seed_base,
            "ter": ter,
            "tax_rate": tax_rate,
            "inflation_mean": inflation_mean,
            "inflation_sigma": inflation_sigma,
            "inflation_stochastic": inflation_stochastic or [],
        }
        cfg = _config_from_inputs(values)
        return cfg

    @app.callback(
        Output("results-store", "data"),
        Input("config-store", "data"),
    )
    def run_simulation(cfg):
        if not cfg:
            return no_update
        warnings = validate_inputs(cfg)
        frames = []
        for scenario_id in cfg.get("scenarios", {}).keys():
            df = simulate_paths(cfg, scenario_id)
            frames.append(df)
        df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        withdraw_params = cfg.get("withdrawal_params", {})
        if df_all.empty:
            agg = {"scenarios": {}, "overall": {}}
        else:
            agg = aggregate(
                df_all,
                cfg["costs_taxes"]["withholding_tax_rate"],
                withdraw_params,
            )
        versions = {"app": "1.0.0"}
        run_digest = run_hash(cfg, versions)
        agg["meta"] = {
            "run_hash": run_digest,
            "versions": versions,
            "seed_base": cfg.get("seed_base"),
            "config_json": json.dumps(cfg, indent=2),
            "warnings": warnings,
        }
        if not df_all.empty:
            df_export = df_all.copy()
            df_export["date"] = df_export["date"].dt.strftime("%Y-%m-%d")
            paths = df_export.to_dict("records")
        else:
            paths = []
        return {
            "paths": paths,
            "agg": agg,
            "warnings": warnings,
            "config": cfg,
        }

    @app.callback(
        Output("nominal-graph", "figure"),
        Output("real-graph", "figure"),
        Output("summary-table", "children"),
        Output("warning-banner", "children"),
        Input("results-store", "data"),
    )
    def update_outputs(data):
        if not data:
            empty_fig = _make_figure(pd.DataFrame(), "value_nom", "Nominal Wealth")
            return empty_fig, empty_fig, [], None
        paths = data.get("paths", [])
        if paths:
            df = pd.DataFrame(paths)
            df["date"] = pd.to_datetime(df["date"])
        else:
            df = pd.DataFrame()
        fig_nom = _make_figure(df, "value_nom", "Nominal Wealth Development")
        fig_real = _make_figure(df, "value_real", "Real Wealth Development")
        table_children = _build_summary_table(data.get("agg", {}))
        warnings = data.get("warnings", [])
        if warnings:
            banner = html.Div(
                [
                    html.Div(
                        [
                            html.H4("Validation Warnings", className="alert-title neon-accent"),
                            html.Ul([html.Li(w) for w in warnings]),
                        ],
                        className="alert neon-warning",
                    )
                ],
                className="warning-container",
            )
        else:
            banner = None
        return fig_nom, fig_real, table_children, banner

    @app.callback(
        Output("download-xlsx", "data"),
        Input("export-button", "n_clicks"),
        State("results-store", "data"),
        prevent_initial_call=True,
    )
    def export_results(n_clicks, data):
        if not data or not data.get("paths"):
            return no_update
        df = pd.DataFrame(data["paths"])
        df["date"] = pd.to_datetime(df["date"])
        agg = data.get("agg", {})
        buffer = io.BytesIO()
        export_xlsx(df, agg, buffer)
        buffer.seek(0)
        return dcc.send_bytes(buffer.read, filename="wealth_forecast.xlsx")
