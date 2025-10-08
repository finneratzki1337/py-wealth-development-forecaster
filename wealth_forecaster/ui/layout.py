"""Dash layout for the wealth forecaster."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from ..config import default_config


def build_controls(cfg: dict) -> dbc.Col:
    cash = cfg.get("cash", {})
    costs = cfg.get("costs_taxes", {})
    inflation = cfg.get("inflation", {})

    return dbc.Col(
        [
            html.H2("Simulation Settings", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Start Date (YYYY-MM)"),
                            dbc.Input(id="start-date", value=cfg.get("start_date", "2025-01")),
                        ],
                        md=6,
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
                        md=6,
                    ),
                ],
                className="g-3",
            ),
            html.Hr(),
            html.H4("Contributions"),
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
                                step=1000,
                            ),
                        ],
                        md=6,
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
                        md=6,
                    ),
                ],
                className="g-3",
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
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Seed Base"),
                            dbc.Input(
                                id="seed-base",
                                type="number",
                                value=cfg.get("seed_base", 1000),
                                min=1,
                                step=1,
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="g-3",
            ),
            html.Hr(),
            html.H4("Economics"),
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
                        md=4,
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
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Inflation Mean (p.a. %)"),
                            dbc.Input(
                                id="inflation-mean",
                                type="number",
                                value=round(inflation.get("mean_pa", 0.02) * 100, 2),
                                step=0.1,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Inflation Sigma (p.a. %)"),
                            dbc.Input(
                                id="inflation-sigma",
                                type="number",
                                value=round(inflation.get("sigma_pa", 0.01) * 100, 2),
                                step=0.1,
                                min=0,
                            ),
                        ],
                        md=6,
                    ),
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
                        md=6,
                    ),
                ],
                className="g-3",
            ),
            html.Div(
                dbc.Button(
                    "Run Simulation",
                    id="run-button",
                    color="primary",
                    className="mt-3 w-100",
                )
            ),
            dbc.Button(
                "Download XLSX",
                id="export-button",
                color="secondary",
                className="mt-2 w-100",
            ),
        ],
        md=4,
        className="bg-light p-3 rounded-3 h-100",
    )


def serve_layout() -> dbc.Container:
    cfg = default_config()
    return dbc.Container(
        [
            dcc.Store(id="config-store", data=cfg),
            dcc.Store(id="results-store"),
            dcc.Download(id="download-xlsx"),
            dbc.Row(
                [
                    build_controls(cfg),
                    dbc.Col(
                        [
                            html.H1("Wealth Development Forecaster", className="mb-3"),
                            html.Div(id="warning-banner"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(id="nominal-graph", config={"displayModeBar": False}),
                                        md=12,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(id="real-graph", config={"displayModeBar": False}),
                                        md=12,
                                    ),
                                ]
                            ),
                            html.H3("Scenario Summary", className="mt-4"),
                            dbc.Table(id="summary-table", bordered=True, hover=True, responsive=True),
                        ],
                        md=8,
                    ),
                ],
                className="g-4 mt-2",
            ),
        ],
        fluid=True,
        className="pb-4",
    )
