"""Dash layout for the wealth forecaster."""
from __future__ import annotations

import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html

from ..config import default_config


def _placeholder_figure(title: str) -> go.Figure:
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
    fig.add_annotation(
        text="Press Run Simulation to generate results",
        showarrow=False,
        font=dict(size=16, family="Courier New, monospace", color="#f4f4f4"),
        align="center",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    return fig


def build_controls(cfg: dict) -> dbc.Col:
    cash = cfg.get("cash", {})
    costs = cfg.get("costs_taxes", {})
    inflation = cfg.get("inflation", {})

    return dbc.Col(
        [
            html.Div(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.Label("Simulation Seed", className="seed-label-small"),
                                    dbc.Input(
                                        id="seed-base",
                                        type="number",
                                        value=cfg.get("seed_base", 1000),
                                        min=1,
                                        step=1,
                                        className="seed-input-small",
                                    ),
                                ],
                                className="top-control-card",
                            ),
                            width="auto",
                            className="top-control-cell",
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.Label("Runs/Scenario", className="seed-label-small"),
                                    dbc.Input(
                                        id="runs-per-scenario",
                                        type="number",
                                        value=cfg.get("runs_per_scenario", 200),
                                        min=100,
                                        step=10,
                                        className="seed-input-small",
                                    ),
                                ],
                                className="top-control-card",
                            ),
                            width="auto",
                            className="top-control-cell",
                        ),
                        dbc.Col(
                            html.Div(
                                html.Div(
                                    dbc.Button(
                                        "Run Simulation",
                                        id="run-button",
                                        color="primary",
                                        size="sm",
                                        className="w-100 neon-button primary-button",
                                    ),
                                    className="w-100 d-grid",
                                ),
                                className="top-control-card top-control-button",
                            ),
                            width="auto",
                            className="top-control-cell",
                        ),
                        dbc.Col(
                            html.Div(
                                html.Div(
                                    dbc.Button(
                                        "Download XLSX",
                                        id="export-button",
                                        color="secondary",
                                        size="sm",
                                        className="w-100 neon-button secondary-button",
                                    ),
                                    className="w-100 d-grid",
                                ),
                                className="top-control-card top-control-button",
                            ),
                            width="auto",
                            className="top-control-cell",
                        ),
                    ],
                    className="g-2 mb-2 align-items-stretch button-row top-control-row",
                ),
                className="mb-3",
            ),
            html.Hr(),
            html.H2("Parameters", className="panel-title mb-3 neon-accent"),
            html.H3("Timeline", className="section-title mt-2 mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Start Date (YYYY-MM)"),
                            dbc.Input(id="start-date", value=cfg.get("start_date", "2025-01")),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=3,
                        xl=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Horizon (years)"),
                            dbc.Input(
                                id="horizon-years",
                                type="number",
                                value=cfg.get("horizon_years", 30),
                                min=1,
                                step=1,
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                ],
                className="g-2 compact-row",
            ),
            html.Hr(),
            html.H3("Contributions", className="section-title mt-2 mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Initial Capital"),
                            dbc.Input(
                                id="initial-capital",
                                type="number",
                                value=cash.get("initial", 0.0),
                                min=0,
                                step=1,
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Monthly Contribution"),
                            dbc.Input(
                                id="monthly-contribution",
                                type="number",
                                value=cash.get("monthly", 0.0),
                                min=0,
                                step=100,
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                ],
                className="g-2 compact-row",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Annual Increase (%)"),
                            dbc.Input(
                                id="annual-increase",
                                type="number",
                                value=round(cash.get("annual_increase", 0.0) * 100, 2),
                                step=0.5,
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                ],
                className="g-2 compact-row",
            ),
            html.Hr(),
            html.H3("Costs & Taxes", className="section-title mt-2 mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("TER (p.a. %)"),
                            dbc.Input(
                                id="ter",
                                type="number",
                                value=round(costs.get("ter_pa", 0.003) * 100, 3),
                                step=0.05,
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Tax on Withdrawals (%)"),
                            dbc.Input(
                                id="tax-rate",
                                type="number",
                                value=round(costs.get("withholding_tax_rate", 0.25) * 100, 1),
                                step=1,
                                min=0,
                                max=100,
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                ],
                className="g-2 compact-row",
            ),
            html.Hr(),
            html.H3("Inflation", className="section-title mt-2 mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Mean (p.a. %)"),
                            dbc.Input(
                                id="inflation-mean",
                                type="number",
                                value=round(inflation.get("mean_pa", 0.02) * 100, 2),
                                step=0.1,
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Sigma (p.a. %)"),
                            dbc.Input(
                                id="inflation-sigma",
                                type="number",
                                value=round(inflation.get("sigma_pa", 0.01) * 100, 2),
                                step=0.1,
                                min=0,
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                ],
                className="g-2 compact-row",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Stochastic Inflation"),
                            dbc.Checklist(
                                options=[{"label": "Enabled", "value": "on"}],
                                value=["on"] if inflation.get("stochastic", True) else [],
                                id="inflation-stochastic",
                                switch=True,
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                ],
                className="g-2 compact-row align-items-center",
            ),
            html.Hr(),
            html.H3("Risk", className="section-title mt-2 mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Volatility Multiplier"),
                            dbc.Input(
                                id="volatility-multiplier",
                                type="number",
                                value=cfg.get("volatility_multiplier", 1.0),
                                min=0.1,
                                step=0.1,
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                ],
                className="g-2 compact-row",
            ),
        ],
        width=12,
        className="control-panel neon-panel p-3 rounded-3",
    )


def serve_layout() -> dbc.Container:
    cfg = default_config()
    return dbc.Container(
        [
            dcc.Store(id="config-store", data=cfg),
            dcc.Store(id="results-store"),
            dcc.Download(id="download-xlsx"),
            html.H1(
                "Wealth Development Forecaster",
                className="app-title mb-3 neon-accent",
            ),
            html.Div(id="warning-banner"),
            dbc.Row([build_controls(cfg)], className="g-3 layout-row"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(
                                "Nominal Growth Highlights",
                                className="section-title",
                            ),
                            html.Div(
                                id="nominal-widget",
                                className="graph-highlight-widget",
                                children=html.Div(
                                    "Run the simulation to see highlights.",
                                    className="widget-placeholder",
                                ),
                            ),
                        ],
                        lg=4,
                        md=12,
                        className="callout-panel neon-panel p-3 rounded-3",
                    ),
                    dbc.Col(
                        [
                            html.H3(
                                "Nominal Wealth Development",
                                className="section-title",
                            ),
                            dcc.Loading(
                                id="nominal-loading",
                                type="default",
                                className="graph-loading-wrapper",
                                children=dcc.Graph(
                                    id="nominal-graph",
                                    config={"displayModeBar": False},
                                    className="neon-graph",
                                    style={"height": "320px"},
                                    figure=_placeholder_figure(
                                        "Nominal Wealth Development"
                                    ),
                                ),
                            ),
                        ],
                        lg=8,
                        md=12,
                        className="chart-panel neon-panel p-3 rounded-3",
                    ),
                ],
                className="g-3 layout-row",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(
                                "Real Growth Highlights",
                                className="section-title",
                            ),
                            html.Div(
                                id="real-widget",
                                className="graph-highlight-widget",
                                children=html.Div(
                                    "Run the simulation to see highlights.",
                                    className="widget-placeholder",
                                ),
                            ),
                        ],
                        lg=4,
                        md=12,
                        className="callout-panel neon-panel p-3 rounded-3",
                    ),
                    dbc.Col(
                        [
                            html.H3(
                                "Real Wealth Development",
                                className="section-title",
                            ),
                            dcc.Loading(
                                id="real-loading",
                                type="default",
                                className="graph-loading-wrapper",
                                children=dcc.Graph(
                                    id="real-graph",
                                    config={"displayModeBar": False},
                                    className="neon-graph",
                                    style={"height": "320px"},
                                    figure=_placeholder_figure(
                                        "Real Wealth Development"
                                    ),
                                ),
                            ),
                        ],
                        lg=8,
                        md=12,
                        className="chart-panel neon-panel p-3 rounded-3",
                    ),
                ],
                className="g-3 layout-row",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Scenario Summary", className="section-title"),
                            dbc.Table(
                                id="summary-table",
                                bordered=True,
                                hover=True,
                                responsive=True,
                                className="neon-table table-sm table-dark",
                            ),
                            html.H3("Scenario Details", className="mt-4 section-title"),
                            html.Div(id="scenario-details"),
                            html.H3(
                                "Average Market Growth",
                                className="mt-4 section-title",
                            ),
                            html.Div(id="growth-rates"),
                        ],
                        width=12,
                        className="detail-panel neon-panel p-3 rounded-3",
                    )
                ],
                className="g-3 layout-row",
            ),
        ],
        fluid=True,
        className="app-shell pb-4",
    )
