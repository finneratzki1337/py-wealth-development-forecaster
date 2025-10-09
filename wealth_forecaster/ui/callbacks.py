"""Dash callbacks for the wealth forecaster UI."""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Sequence

import dash_bootstrap_components as dbc
import numpy as np
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


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _percentile_block(stats: Dict[str, float]) -> html.Div:
    return html.Div(
        [
            html.Small(f"p90: {_format_currency(stats.get('p90', 0.0))}"),
            html.Small(f"p50: {_format_currency(stats.get('p50', 0.0))}"),
            html.Small(f"p10: {_format_currency(stats.get('p10', 0.0))}"),
        ],
        className="percentile-block",
    )


def _get_percentiles(metrics: Dict, *keys: str) -> Dict[str, float]:
    cur: Dict | float | int = metrics
    for key in keys:
        if not isinstance(cur, dict):
            return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
        cur = cur.get(key, {})
    if not isinstance(cur, dict):
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "p10": float(cur.get("p10", 0.0)),
        "p50": float(cur.get("p50", 0.0)),
        "p90": float(cur.get("p90", 0.0)),
    }


def _compute_cagr(
    final_value: float, deposits_total: float, horizon_years: float
) -> Optional[float]:
    if final_value <= 0 or deposits_total <= 0 or horizon_years <= 0:
        return None
    try:
        years = float(horizon_years)
        return (final_value / deposits_total) ** (1 / years) - 1
    except (ZeroDivisionError, ValueError):
        return None


def _build_highlight_widget(
    agg: Dict,
    cfg: Optional[Dict],
    value_path: Sequence[str],
    income_path: Sequence[str],
    main_label: str,
    income_label: str,
) -> html.Div:
    scenarios = agg.get("scenarios", {}) if isinstance(agg, dict) else {}
    moderate = scenarios.get("moderate", {}) if isinstance(scenarios, dict) else {}
    if not isinstance(moderate, dict) or not moderate:
        return html.Div(
            "Run the simulation to see highlights.",
            className="widget-placeholder",
        )

    moderate_end = _get_percentiles(moderate, *value_path)
    moderate_income = _get_percentiles(moderate, *income_path)
    optimistic_end = _get_percentiles(
        scenarios.get("optimistic", {}), *value_path
    )
    pessimistic_end = _get_percentiles(
        scenarios.get("pessimistic", {}), *value_path
    )

    horizon_years = 0.0
    if isinstance(cfg, dict):
        try:
            horizon_years = float(cfg.get("horizon_years", 0) or 0)
        except (TypeError, ValueError):
            horizon_years = 0.0
    deposits_total = float(moderate.get("deposits_total", 0.0) or 0.0)
    cagr_value = _compute_cagr(moderate_end["p50"], deposits_total, horizon_years)

    main_row_children = [
        html.Span(
            _format_currency(moderate_end["p50"]), className="widget-main-value"
        )
    ]
    if cagr_value is not None:
        main_row_children.append(
            html.Span(f"{cagr_value * 100:.1f}% CAGR", className="widget-cagr")
        )

    cagr_rows = []
    for scenario_id, label in [
        ("optimistic", "Opt"),
        ("moderate", "Mod"),
        ("pessimistic", "Pes"),
    ]:
        scenario_metrics = scenarios.get(scenario_id, {}) if isinstance(scenarios, dict) else {}
        if not isinstance(scenario_metrics, dict) or not scenario_metrics:
            continue
        deposits = float(scenario_metrics.get("deposits_total", 0.0) or 0.0)
        end_stats = _get_percentiles(scenario_metrics, *value_path)
        cagr_val = _compute_cagr(end_stats["p50"], deposits, horizon_years)
        if cagr_val is None:
            text = f"{label} CAGR: —"
        else:
            text = f"{label} CAGR: {cagr_val * 100:.1f}%"
        cagr_rows.append(html.Span(text))

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        f"Opt median: {_format_currency(optimistic_end['p50'])}"
                    ),
                    html.Span(
                        f"Pes median: {_format_currency(pessimistic_end['p50'])}"
                    ),
                ],
                className="widget-upper",
            ),
            html.Div(
                [
                    html.Div(main_label, className="widget-main-label"),
                    html.Div(main_row_children, className="widget-main-row"),
                    html.Div(
                        f"{income_label}: {_format_currency(moderate_income['p50'])}/mo",
                        className="widget-secondary-value",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Span(
                        f"p90: {_format_currency(moderate_end['p90'])}"
                    ),
                    html.Span(
                        f"p10: {_format_currency(moderate_end['p10'])}"
                    ),
                ],
                className="widget-percentiles",
            ),
            html.Div(cagr_rows, className="widget-cagr-list") if cagr_rows else None,
        ]
    )


def _config_from_inputs(values: Dict[str, Any]) -> Dict[str, Any]:
    cfg = canonicalize(default_config())
    
    # Helper to safely convert to float with default
    def safe_float(val, default=0.0):
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    # Helper to safely convert to int with default
    def safe_int(val, default=0):
        if val is None or val == "":
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default
    
    cfg["start_date"] = values.get("start_date") or cfg.get("start_date", "2025-01")
    cfg["horizon_years"] = safe_int(values.get("horizon_years"), cfg.get("horizon_years", 30))
    cfg["seed_base"] = safe_int(values.get("seed_base"), cfg.get("seed_base", 1000))
    cfg["runs_per_scenario"] = max(100, safe_int(values.get("runs_per_scenario"), 200))
    cfg["volatility_multiplier"] = max(0.01, safe_float(values.get("volatility_multiplier"), 1.0))

    cash = cfg["cash"]
    cash["initial"] = safe_float(values.get("initial_capital"), cash.get("initial", 0.0))
    cash["monthly"] = safe_float(values.get("monthly_contribution"), cash.get("monthly", 0.0))
    cash["annual_increase"] = safe_float(values.get("annual_increase"), 0.0) / 100.0
    cash["pauses"] = []
    cash["events"] = []

    costs = cfg["costs_taxes"]
    costs["ter_pa"] = safe_float(values.get("ter"), costs.get("ter_pa", 0.003) * 100) / 100.0
    costs["withholding_tax_rate"] = safe_float(values.get("tax_rate"), costs.get("withholding_tax_rate", 0.25) * 100) / 100.0

    inflation = cfg["inflation"]
    inflation["mean_pa"] = safe_float(values.get("inflation_mean"), inflation.get("mean_pa", 0.02) * 100) / 100.0
    inflation["sigma_pa"] = safe_float(values.get("inflation_sigma"), inflation.get("sigma_pa", 0.01) * 100) / 100.0
    inflation["stochastic"] = "on" in (values.get("inflation_stochastic") or [])

    return cfg


def _make_figure(df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    fig = go.Figure()
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

    if df.empty:
        return fig

    df_sorted = df.sort_values("date")
    for scenario, df_s in df_sorted.groupby("scenario"):
        color = SCENARIO_COLORS.get(scenario, "#cccccc")
        shade_color = _hex_to_rgba(color, 0.22)

        grouped = df_s.groupby("date")[value_col]
        p10 = grouped.quantile(0.1)
        p50 = grouped.quantile(0.5)
        p90 = grouped.quantile(0.9)
        if p50.empty:
            continue

        custom = np.column_stack([p10.values, p90.values])

        fig.add_trace(
            go.Scatter(
                x=p90.index,
                y=p90.values,
                line=dict(width=0),
                hoverinfo="skip",
                legendgroup=scenario,
                showlegend=False,
                name=f"{scenario.title()} p90",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=p10.index,
                y=p10.values,
                line=dict(width=0),
                fill="tonexty",
                fillcolor=shade_color,
                hoverinfo="skip",
                legendgroup=scenario,
                showlegend=False,
                name=f"{scenario.title()} p10",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=p50.index,
                y=p50.values,
                name=f"{scenario.title()} Median",
                legendgroup=scenario,
                line=dict(width=3, color=color),
                customdata=custom,
                hovertemplate=(
                    f"<b>{scenario.title()}</b><br>%{{x|%Y-%m}}"
                    "<br>p90: $%{customdata[1]:,.0f}"
                    "<br>median: $%{y:,.0f}"
                    "<br>p10: $%{customdata[0]:,.0f}<extra></extra>"
                ),
            )
        )

    return fig



def _build_summary_table(agg: Dict) -> List[html.Tr]:
    scenarios = agg.get("scenarios", {})
    header = html.Thead(
        html.Tr(
            [
                html.Th("Scenario"),
                html.Th("Total Pay-in"),
                html.Th("Nominal Wealth"),
                html.Th("Real Wealth"),
                html.Th("Monthly Nominal Gross"),
                html.Th("Monthly Nominal Net"),
                html.Th("Monthly Real Gross"),
                html.Th("Monthly Real Net"),
            ]
        )
    )
    rows: List[html.Tr] = []
    for scenario, metrics in scenarios.items():
        rows.append(
            html.Tr(
                [
                    html.Th(scenario.title()),
                    html.Td(_format_currency(metrics.get("deposits_total", 0.0))),
                    html.Td(_percentile_block(_get_percentiles(metrics, "end_nominal"))),
                    html.Td(_percentile_block(_get_percentiles(metrics, "end_real"))),
                    html.Td(
                        _percentile_block(
                            _get_percentiles(metrics, "perpetuity", "nominal", "gross")
                        )
                    ),
                    html.Td(
                        _percentile_block(
                            _get_percentiles(metrics, "perpetuity", "nominal", "net")
                        )
                    ),
                    html.Td(
                        _percentile_block(
                            _get_percentiles(metrics, "perpetuity", "real", "gross")
                        )
                    ),
                    html.Td(
                        _percentile_block(
                            _get_percentiles(metrics, "perpetuity", "real", "net")
                        )
                    ),
                ],
                className="summary-row",
            )
        )
    overall = agg.get("overall", {})
    if isinstance(overall, dict) and overall:
        rows.append(
            html.Tr(
                [
                    html.Th("Overall"),
                    html.Td(_format_currency(overall.get("deposits_total", 0.0))),
                    html.Td(_percentile_block(_get_percentiles(overall, "end_nominal"))),
                    html.Td(_percentile_block(_get_percentiles(overall, "end_real"))),
                    html.Td(
                        _percentile_block(
                            _get_percentiles(overall, "perpetuity", "nominal", "gross")
                        )
                    ),
                    html.Td(
                        _percentile_block(
                            _get_percentiles(overall, "perpetuity", "nominal", "net")
                        )
                    ),
                    html.Td(
                        _percentile_block(
                            _get_percentiles(overall, "perpetuity", "real", "gross")
                        )
                    ),
                    html.Td(
                        _percentile_block(
                            _get_percentiles(overall, "perpetuity", "real", "net")
                        )
                    ),
                ],
                className="summary-row overall-row",
            )
        )
    if not rows:
        rows.append(
            html.Tr(
                [html.Td("Run the simulation to populate results.", colSpan=8)],
                className="summary-row empty-row",
            )
        )
    return [header, html.Tbody(rows)]


def _build_scenario_details(agg: Dict) -> html.Component:
    scenarios = agg.get("scenarios", {})
    if not scenarios:
        return html.Div("Run the simulation to view scenario details.", className="details-placeholder")
    items = []
    for scenario, metrics in scenarios.items():
        rows = []
        for label, key in [("Perpetuity", "perpetuity"), ("30-Year Annuity", "annuity")]:
            rows.append(
                html.Tr(
                    [
                        html.Th(label),
                        html.Td(
                            _percentile_block(_get_percentiles(metrics, key, "nominal", "gross"))
                        ),
                        html.Td(
                            _percentile_block(_get_percentiles(metrics, key, "nominal", "net"))
                        ),
                        html.Td(
                            _percentile_block(_get_percentiles(metrics, key, "real", "gross"))
                        ),
                        html.Td(
                            _percentile_block(_get_percentiles(metrics, key, "real", "net"))
                        ),
                    ],
                    className="detail-row",
                )
            )
        table = dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Payout Type"),
                            html.Th("Nominal Gross"),
                            html.Th("Nominal Net"),
                            html.Th("Real Gross"),
                            html.Th("Real Net"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            bordered=True,
            hover=True,
            responsive=True,
            className="neon-table table-sm table-dark",
        )
        items.append(dbc.AccordionItem(table, title=scenario.title()))
    return dbc.Accordion(items, always_open=True, start_collapsed=False, className="scenario-accordion")


def _build_growth_rates_table(df: pd.DataFrame) -> html.Component:
    def _placeholder() -> html.Component:
        placeholder = html.Div(
            "Run the simulation to view yearly growth rates.",
            className="details-placeholder",
        )
        return dbc.Accordion(
            [
                dbc.AccordionItem(
                    placeholder,
                    title="Average YoY Growth (%)",
                )
            ],
            start_collapsed=True,
            className="growth-accordion",
        )

    if df.empty:
        return _placeholder()

    df_sorted = df.sort_values(["scenario", "run_id", "date"])
    df_sorted["year"] = df_sorted["date"].dt.year
    yearly = (
        df_sorted.groupby(["scenario", "run_id", "year"], as_index=False)
        .agg(value_nom=("value_nom", "last"))
    )

    if yearly.empty:
        return _placeholder()

    yearly["growth"] = yearly.groupby(["scenario", "run_id"])["value_nom"].pct_change()
    growth = (
        yearly.groupby(["scenario", "year"], as_index=False)
        ["growth"].mean()
    )

    if growth.empty:
        return _placeholder()

    pivot = (
        growth.pivot(index="year", columns="scenario", values="growth")
        .sort_index()
    )
    scenario_order = ["optimistic", "moderate", "pessimistic"]
    header_cells = [html.Th("Year")] + [html.Th(s.title()) for s in scenario_order]
    rows = []
    for year in pivot.index:
        row_cells = [html.Th(str(int(year)))]
        for scenario in scenario_order:
            value = pivot.loc[year, scenario] if scenario in pivot.columns else float("nan")
            if pd.isna(value):
                display = "—"
            else:
                display = f"{value * 100:.2f}%"
            row_cells.append(html.Td(display))
        rows.append(html.Tr(row_cells))

    table = dbc.Table(
        [html.Thead(html.Tr(header_cells)), html.Tbody(rows)],
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm table-dark growth-table",
    )

    return dbc.Accordion(
        [
            dbc.AccordionItem(
                table,
                title="Average YoY Growth (%)",
            )
        ],
        start_collapsed=True,
        className="growth-accordion",
    )


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
        State("runs-per-scenario", "value"),
        State("ter", "value"),
        State("tax-rate", "value"),
        State("inflation-mean", "value"),
        State("inflation-sigma", "value"),
        State("volatility-multiplier", "value"),
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
        runs_per_scenario,
        ter,
        tax_rate,
        inflation_mean,
        inflation_sigma,
        volatility_multiplier,
        inflation_stochastic,
    ):
        values = {
            "start_date": start_date,
            "horizon_years": horizon_years,
            "initial_capital": initial_capital,
            "monthly_contribution": monthly_contribution,
            "annual_increase": annual_increase,
            "seed_base": seed_base,
            "runs_per_scenario": runs_per_scenario,
            "ter": ter,
            "tax_rate": tax_rate,
            "inflation_mean": inflation_mean,
            "inflation_sigma": inflation_sigma,
            "volatility_multiplier": volatility_multiplier,
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
        Output("scenario-details", "children"),
        Output("nominal-widget", "children"),
        Output("real-widget", "children"),
        Output("growth-rates", "children"),
        Input("results-store", "data"),
    )
    def update_outputs(data):
        if not data:
            empty_fig = _make_figure(pd.DataFrame(), "value_nom", "Nominal Wealth")
            details_placeholder = html.Div(
                "Run the simulation to view scenario details.",
                className="details-placeholder",
            )
            nominal_placeholder = _build_highlight_widget(
                {},
                None,
                ("end_nominal",),
                ("perpetuity", "nominal", "net"),
                "Moderate median end",
                "Perpetual net income",
            )
            real_placeholder = _build_highlight_widget(
                {},
                None,
                ("end_real",),
                ("perpetuity", "real", "net"),
                "Moderate median real end",
                "Perpetual net income (real)",
            )
            growth_placeholder = _build_growth_rates_table(pd.DataFrame())
            return (
                empty_fig,
                empty_fig,
                [],
                None,
                details_placeholder,
                nominal_placeholder,
                real_placeholder,
                growth_placeholder,
            )
        paths = data.get("paths", [])
        if paths:
            df = pd.DataFrame(paths)
            df["date"] = pd.to_datetime(df["date"])
        else:
            df = pd.DataFrame()
        fig_nom = _make_figure(df, "value_nom", "Nominal Wealth Development")
        fig_real = _make_figure(df, "value_real", "Real Wealth Development")
        table_children = _build_summary_table(data.get("agg", {}))
        details = _build_scenario_details(data.get("agg", {}))
        growth_rates = _build_growth_rates_table(df)
        agg = data.get("agg", {})
        cfg = data.get("config")
        widget_nominal = _build_highlight_widget(
            agg,
            cfg,
            ("end_nominal",),
            ("perpetuity", "nominal", "net"),
            "Moderate median end",
            "Perpetual net income",
        )
        widget_real = _build_highlight_widget(
            agg,
            cfg,
            ("end_real",),
            ("perpetuity", "real", "net"),
            "Moderate median real end",
            "Perpetual net income (real)",
        )
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
        return (
            fig_nom,
            fig_real,
            table_children,
            banner,
            details,
            widget_nominal,
            widget_real,
            growth_rates,
        )

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
