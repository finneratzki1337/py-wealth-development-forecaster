"""Configuration utilities for the wealth development forecaster."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict

_DEFAULT_CONFIG: Dict[str, Any] = {
    "seed_base": 1000,
    "runs_per_scenario": 200,
    "horizon_years": 30,
    "start_date": "2025-01",
    "cash": {
        "initial": 0.0,
        "monthly": 1000.0,
        "annual_increase": 0.0,
        "pauses": [["2030-01", "2030-12"]],
        "events": [{"date": "2032-01", "new_monthly": 1500.0}],
    },
    "inflation": {
        "mean_pa": 0.02,
        "sigma_pa": 0.005,
        "stochastic": True,
    },
    "costs_taxes": {
        "ter_pa": 0.003,
        "withholding_tax_rate": 0.25,
    },
    "scenarios": {
        "optimistic": {
            "mu_pa_blocks": [0.06, 0.08, 0.08, 0.07, 0.07, 0.06],
            "sigma_pa": 0.10,
        },
        "moderate": {
            "mu_pa_blocks": [0.05, 0.06, 0.06, 0.05, 0.05, 0.05],
            "sigma_pa": 0.14,
        },
        "pessimistic": {
            "mu_pa_blocks": [0.00, 0.02, 0.03, 0.02, 0.02, 0.02],
            "sigma_pa": 0.20,
        },
    },
    "volatility_multiplier": 1.0,
    "optimism_adjustment": 0.0,
    "return_model": {
        "mean_type": "arithmetic",
        "distribution": "normal_arith",
        "truncate_at_minus_100": True,
    },
    "withdrawal_params": {
        "r_nom_perp": None,
        "r_real_perp": None,
        "r_nom_ann": None,
        "r_real_ann": None,
    },
    "targets": {
        "end_real": None,
        "income_real": None,
    },
}


def default_config() -> Dict[str, Any]:
    """Return a deep copy of the default configuration."""
    return json.loads(json.dumps(_DEFAULT_CONFIG))


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Save a configuration as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def canonicalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a canonical (sorted) representation of the configuration."""

    def _canonical(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _canonical(value[k]) for k in sorted(value)}
        if isinstance(value, list):
            return [_canonical(v) for v in value]
        return value

    return _canonical(cfg)
