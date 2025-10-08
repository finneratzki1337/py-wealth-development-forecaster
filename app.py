from __future__ import annotations

import dash
import dash_bootstrap_components as dbc

from wealth_forecaster.ui.callbacks import register_callbacks
from wealth_forecaster.ui.layout import serve_layout


external_stylesheets = [dbc.themes.SANDSTONE]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Wealth Development Forecaster"
app.layout = serve_layout

register_callbacks(app)

server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
