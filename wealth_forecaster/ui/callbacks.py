"""Dash callbacks for the wealth forecaster UI."""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Sequence

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
    if value is None:
        return "$0"
    try:
        return f"${float(value):,.0f}"
    except (ValueError, TypeError):
        return "$0"


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
    
    # Helper to safely extract float values, handling None
    def safe_get(key: str, default: float = 0.0) -> float:
        val = cur.get(key, default)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    return {
        "p10": safe_get("p10", 0.0),
        "p50": safe_get("p50", 0.0),
        "p90": safe_get("p90", 0.0),
    }


def _build_highlight_widget(
    agg: Dict,
    value_path: Sequence[str],
    income_path: Sequence[str],
    value_label: str,
    income_label: str,
    growth_key: str,
    growth_label: str,
) -> html.Div:
    scenarios = agg.get("scenarios", {}) if isinstance(agg, dict) else {}
    if not isinstance(scenarios, dict) or not scenarios:
        return html.Div(
            "Run the simulation to see highlights.",
            className="widget-placeholder",
        )

    scenario_cards: List[html.Div] = []
    scenario_order = [
        ("optimistic", "Optimistic"),
        ("moderate", "Moderate"),
        ("pessimistic", "Pessimistic"),
    ]

    for scenario_id, title in scenario_order:
        scenario_metrics = scenarios.get(scenario_id, {})
        if not isinstance(scenario_metrics, dict) or not scenario_metrics:
            continue

        end_stats = _get_percentiles(scenario_metrics, *value_path)
        income_stats = _get_percentiles(scenario_metrics, *income_path)
        deposits_stats = scenario_metrics.get("deposits_distribution", {})
        deposits_median = float(
            deposits_stats.get("p50", scenario_metrics.get("deposits_total", 0.0)) or 0.0
        )
        capital_gain = end_stats["p50"] - deposits_median
        income_median = income_stats["p50"]
        market_growth = scenario_metrics.get(growth_key)
        if market_growth is None:
            market_growth_text = "—"
        else:
            market_growth_text = f"{market_growth * 100:.2f}% p.a."

        card = html.Div(
            [
                html.Div(title, className="widget-scenario-title"),
                html.Div(
                    [
                        html.Span(_format_currency(end_stats["p50"]), className="widget-scenario-value"),
                        html.Small(value_label, className="widget-scenario-label"),
                    ],
                    className="widget-scenario-main",
                ),
                html.Ul(
                    [
                        html.Li(
                            [
                                html.Span("Paid in"),
                                html.Span(_format_currency(deposits_median)),
                            ]
                        ),
                        html.Li(
                            [
                                html.Span("Capital gains"),
                                html.Span(_format_currency(capital_gain)),
                            ]
                        ),
                        html.Li(
                            [
                                html.Span(income_label),
                                html.Span(f"{_format_currency(income_median)}/mo"),
                            ]
                        ),
                        html.Li(
                            [
                                html.Span(growth_label),
                                html.Span(market_growth_text),
                            ]
                        ),
                    ],
                    className="widget-scenario-metrics",
                ),
            ],
            className=f"widget-scenario-card scenario-{scenario_id}",
        )

        scenario_cards.append(card)

    if not scenario_cards:
        return html.Div(
            "Run the simulation to see highlights.",
            className="widget-placeholder",
        )

    return html.Div(
        scenario_cards,
        className="widget-scenario-grid",
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
            "Run the simulation to view market growth assumptions.",
            className="details-placeholder",
        )
        return dbc.Accordion(
            [
                dbc.AccordionItem(
                    placeholder,
                    title="Average Market Growth (%)",
                )
            ],
            start_collapsed=True,
            className="growth-accordion",
        )

    if df.empty:
        return _placeholder()

    df_sorted = df.sort_values(["scenario", "run_id", "date"])
    df_sorted["year"] = df_sorted["date"].dt.year
    yearly_values = (
        df_sorted.groupby(["scenario", "run_id", "year"], as_index=False)
        .agg(value_nom=("value_nom", "last"))
    )

    if yearly_values.empty:
        return _placeholder()

    def _compound_growth(series: pd.Series) -> float:
        return float(np.prod(1.0 + series.values) - 1.0)

    yearly_returns = (
        df_sorted.groupby(["scenario", "run_id", "year"], as_index=False)
        .agg(market_growth=("r_net", _compound_growth))
    )

    if yearly_returns.empty:
        return _placeholder()

    growth = (
        yearly_returns.groupby(["scenario", "year"], as_index=False)[
            "market_growth"
        ].mean()
    )

    if growth.empty:
        return _placeholder()

    pivot_growth = (
        growth.pivot(index="year", columns="scenario", values="market_growth")
        .sort_index()
    )
    pivot_values = (
        yearly_values.groupby(["scenario", "year"], as_index=False)["value_nom"]
        .median()
        .pivot(index="year", columns="scenario", values="value_nom")
        .reindex(pivot_growth.index, fill_value=float("nan"))
    )
    scenario_order = ["optimistic", "moderate", "pessimistic"]
    header_cells = [html.Th("Year")] + [html.Th(s.title()) for s in scenario_order]
    rows = []
    for year in pivot_growth.index:
        row_cells = [html.Th(str(int(year)))]
        for scenario in scenario_order:
            growth_value = (
                pivot_growth.loc[year, scenario]
                if scenario in pivot_growth.columns
                else float("nan")
            )
            abs_value = (
                pivot_values.loc[year, scenario]
                if scenario in pivot_values.columns
                else float("nan")
            )

            if pd.isna(growth_value):
                percent_display = "—"
            else:
                percent_display = f"{growth_value * 100:.2f}%"

            if pd.isna(abs_value):
                abs_display: html.Small | None = None
            else:
                abs_display = html.Small(
                    f"Median: {_format_currency(abs_value)}",
                    className="growth-absolute",
                )

            cell_children: List[html.Component] = [
                html.Div(percent_display, className="growth-percent")
            ]
            if abs_display is not None:
                cell_children.append(abs_display)

            row_cells.append(html.Td(cell_children, className="growth-cell"))
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
                title="Average Market Growth (%)",
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
        prevent_initial_call=True,
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
                ("end_nominal",),
                ("perpetuity", "nominal", "net"),
                "Median end wealth",
                "Net perpetuity income",
                "market_growth_annual",
                "Avg nominal growth",
            )
            real_placeholder = _build_highlight_widget(
                {},
                ("end_real",),
                ("perpetuity", "real", "net"),
                "Median real end wealth",
                "Real net perpetuity",
                "market_growth_real_annual",
                "Avg real growth",
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
        widget_nominal = _build_highlight_widget(
            agg,
            ("end_nominal",),
            ("perpetuity", "nominal", "net"),
            "Median end wealth",
            "Net perpetuity income",
            "market_growth_annual",
            "Avg nominal growth",
        )
        widget_real = _build_highlight_widget(
            agg,
            ("end_real",),
            ("perpetuity", "real", "net"),
            "Median real end wealth",
            "Real net perpetuity",
            "market_growth_real_annual",
            "Avg real growth",
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
        print(f"Export button clicked: {n_clicks}")
        print(f"Data available: {data is not None}")
        
        if not data:
            print("No data in results-store")
            return no_update
            
        if not data.get("paths"):
            print("No paths in data")
            return no_update
        
        try:
            print(f"Number of paths: {len(data['paths'])}")
            df = pd.DataFrame(data["paths"])
            print(f"DataFrame shape: {df.shape}")
            
            if df.empty:
                print("DataFrame is empty")
                return no_update
            
            # Convert date column to datetime
            df["date"] = pd.to_datetime(df["date"])
            print(f"Date conversion successful")
            
            agg = data.get("agg", {})
            cfg = data.get("config")
            print(f"Config available: {cfg is not None}")
            
            buffer = io.BytesIO()
            export_xlsx(df, agg, buffer, config=cfg)
            buffer.seek(0)
            print(f"Excel file created, size: {len(buffer.getvalue())} bytes")
            
            return dcc.send_bytes(buffer.getvalue, filename="wealth_forecast.xlsx")
        except Exception as e:
            print(f"Export error: {e}")
            import traceback
            traceback.print_exc()
            return no_update
