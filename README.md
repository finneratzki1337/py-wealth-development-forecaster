# Wealth Development Forecaster

An interactive Dash application for modelling 30-year wealth development across three market scenarios. The simulator follows the requirements in `specs.txt`, providing reproducible scenario analysis, inflation-aware metrics, and XLSX export.

## Features

- Block-wise return modelling with stochastic monthly sampling and fee drag.
- Inflation process with yearly draws and monthly expansion.
- Cash-flow controls for initial wealth, monthly contributions, and annual step-ups.
- Nine paths per run (three scenarios × three seeds) with scenario averages and bands.
- Withdrawal analytics (perpetuity and 30-year annuity) including tax drag.
- Validation warnings for suspect inputs and deterministic run hashing.
- Downloadable Excel workbook containing raw paths and aggregate tables.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open your browser at [http://127.0.0.1:8050](http://127.0.0.1:8050) and adjust the controls on the left. Use **Run Simulation** to update the charts and tables, and **Download XLSX** to export the results.

## Testing

Run the automated tests with:

```bash
pytest
```

## Docker

Build and run the Dash application with Docker:

```bash
docker build -t wealth-forecaster .
docker run --rm -p 8050:8050 wealth-forecaster
```

The app will be available at [http://127.0.0.1:8050](http://127.0.0.1:8050).
