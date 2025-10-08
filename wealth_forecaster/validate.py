"""Configuration validation utilities."""
from __future__ import annotations

from typing import Dict, List


def validate_inputs(cfg: Dict) -> List[str]:
    warnings: List[str] = []

    if cfg.get("horizon_years", 0) <= 0:
        warnings.append("Horizon must be positive.")

    cash = cfg.get("cash", {})
    if cash.get("monthly", 0) < 0:
        warnings.append("Monthly contribution cannot be negative.")
    if cash.get("initial", 0) < 0:
        warnings.append("Initial capital cannot be negative.")

    inflation = cfg.get("inflation", {})
    if inflation.get("sigma_pa", 0) < 0:
        warnings.append("Inflation sigma must be non-negative.")

    costs = cfg.get("costs_taxes", {})
    ter = costs.get("ter_pa", 0)
    if ter < 0 or ter > 0.05:
        warnings.append("TER appears implausible (0-5% recommended).")
    tax = costs.get("withholding_tax_rate", 0)
    if not 0 <= tax <= 1:
        warnings.append("Withholding tax should be between 0 and 1.")

    for scenario, values in cfg.get("scenarios", {}).items():
        if len(values.get("mu_pa_blocks", [])) < 1:
            warnings.append(f"Scenario {scenario} has no return blocks configured.")

    return warnings
