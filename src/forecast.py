"""
forecast.py - Short-term Forecast Monte Carlo Simulation Engine for RainProof.

Runs 1,000 Monte Carlo iterations for forecast weather days to compute:
- Expected modeled income
- P10, P50, P90 income percentiles
- Income-at-Risk (Baseline reference - P10 modeled income)
- Explainability waterfall components (Baseline, Rain Effect, Weekend Effect, Festival Effect)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.simulator import (
    calculate_rain_stress,
    simulate_single_day,
    SENSITIVITY_SCENARIOS,
    DEFAULT_BASELINE_INCOME,
    DEFAULT_WEEKDAY_FACTOR,
    DEFAULT_WEEKEND_FACTOR,
    DEFAULT_FESTIVAL_FACTOR,
    DEFAULT_NOISE_SIGMA,
)


def run_forecast_monte_carlo(
    rain_mm: float,
    is_weekend: bool = False,
    is_festival: bool = False,
    baseline_income: float = DEFAULT_BASELINE_INCOME,
    sensitivity: str = "MEDIUM",
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    n_simulations: int = 1000,
    seed: int = 100,
) -> Dict[str, Any]:
    """
    Runs Monte Carlo simulation for a single forecast day.

    Returns:
        - rain_mm
        - rain_stress
        - baseline_ref (No-rain income reference)
        - rain_effect_amount (Income reduction due to rain)
        - weekend_effect_amount (Income boost due to weekend)
        - festival_effect_amount (Income boost due to festival)
        - expected_income (Mean modeled income across simulations)
        - p10_income (10th percentile modeled income)
        - p50_income (50th percentile modeled income)
        - p90_income (90th percentile modeled income)
        - income_at_risk (baseline_ref - p10_income)
        - simulations_array
    """
    s_val = SENSITIVITY_SCENARIOS.get(sensitivity.upper(), SENSITIVITY_SCENARIOS["MEDIUM"])
    day_factor = DEFAULT_WEEKEND_FACTOR if is_weekend else DEFAULT_WEEKDAY_FACTOR
    fest_factor = DEFAULT_FESTIVAL_FACTOR if is_festival else 1.00

    baseline_ref = baseline_income * day_factor * fest_factor
    rain_stress = calculate_rain_stress(rain_mm)

    # Explainability breakdown components
    unadjusted_base = baseline_income
    weekend_effect = baseline_income * (day_factor - 1.0)
    festival_effect = (baseline_income * day_factor) * (fest_factor - 1.0)
    rain_effect = -1.0 * baseline_ref * (s_val * rain_stress)

    expected_deterministic = baseline_ref * (1.0 - s_val * rain_stress)

    # Monte Carlo simulation with lognormal noise
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_simulations)
    noise_vals = np.exp(z * noise_sigma - 0.5 * (noise_sigma ** 2)) if noise_sigma > 0 else np.ones(n_simulations)

    simulated_incomes = expected_deterministic * noise_vals

    p10 = float(np.percentile(simulated_incomes, 10))
    p50 = float(np.percentile(simulated_incomes, 50))
    p90 = float(np.percentile(simulated_incomes, 90))
    mean_income = float(np.mean(simulated_incomes))

    income_at_risk = float(max(baseline_ref - p10, 0.0))

    return {
        "rain_mm": float(rain_mm),
        "rain_stress": float(rain_stress),
        "unadjusted_base": float(unadjusted_base),
        "weekend_effect": float(weekend_effect),
        "festival_effect": float(festival_effect),
        "rain_effect": float(rain_effect),
        "baseline_ref": float(baseline_ref),
        "expected_income": mean_income,
        "p10_income": p10,
        "p50_income": p50,
        "p90_income": p90,
        "income_at_risk": income_at_risk,
        "simulations": simulated_incomes,
    }
