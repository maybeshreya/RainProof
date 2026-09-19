"""
simulator.py - Income Simulation Engine for RainProof.

Implements rainfall-induced gig worker income modeling and loss calculations.

Core Equations:
1. RainStress = 1 - exp(-rain_mm / 50)
2. Income = Baseline * DayFactor * FestivalFactor * (1 - s * RainStress) * Noise
3. Baseline Reference = Baseline * DayFactor * FestivalFactor
4. Loss = max(Baseline Reference - Income, 0)
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, Tuple

# Standard Sensitivity Scenarios (Section 11)
SENSITIVITY_SCENARIOS = {
    "LOW": 0.25,
    "MEDIUM": 0.40,
    "HIGH": 0.55,
}

DEFAULT_BASELINE_INCOME = 900.0  # ₹900/day
DEFAULT_WEEKDAY_FACTOR = 1.00
DEFAULT_WEEKEND_FACTOR = 1.10
DEFAULT_FESTIVAL_FACTOR = 1.15
DEFAULT_NOISE_SIGMA = 0.12


def calculate_rain_stress(rain_mm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates saturating rain stress factor.

    Formula:
        RainStress = 1 - exp(-rain_mm / 50.0)

    Properties:
        - rain_mm = 0 -> RainStress = 0.0
        - rain_mm = 50 -> RainStress ≈ 0.6321
        - rain_mm -> inf -> RainStress -> 1.0
    """
    rain_clean = np.maximum(rain_mm, 0.0)
    return 1.0 - np.exp(-rain_clean / 50.0)


def generate_lognormal_noise(
    size: int = 1,
    sigma: float = DEFAULT_NOISE_SIGMA,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates mean-normalized lognormal noise with standard deviation parameter sigma.

    To ensure E[Noise] = 1.0:
        Noise = exp(Z * sigma - 0.5 * sigma^2) where Z ~ N(0, 1)
    """
    if sigma <= 0.0:
        return np.ones(size)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size)
    # Mean correction for lognormal distribution
    lognormal_vals = np.exp(z * sigma - 0.5 * (sigma ** 2))
    return lognormal_vals


def simulate_single_day(
    rain_mm: float,
    is_weekend: bool = False,
    is_festival: bool = False,
    baseline_income: float = DEFAULT_BASELINE_INCOME,
    sensitivity: Union[str, float] = "MEDIUM",
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    seed: Optional[int] = None,
    weekday_factor: float = DEFAULT_WEEKDAY_FACTOR,
    weekend_factor: float = DEFAULT_WEEKEND_FACTOR,
    festival_factor: float = DEFAULT_FESTIVAL_FACTOR,
) -> Dict[str, float]:
    """
    Simulates modeled income and income loss for a single day.

    Returns dictionary containing:
        - rain_mm
        - rain_stress
        - baseline_ref (No-rain income reference)
        - expected_income (Deterministic modeled income without noise)
        - modeled_income (Stochastic modeled income with lognormal noise)
        - modeled_loss (max(baseline_ref - modeled_income, 0))
    """
    # Sensitivity numerical lookup
    if isinstance(sensitivity, str):
        s = SENSITIVITY_SCENARIOS.get(sensitivity.upper(), SENSITIVITY_SCENARIOS["MEDIUM"])
    else:
        s = float(sensitivity)

    day_factor = weekend_factor if is_weekend else weekday_factor
    fest_factor = festival_factor if is_festival else 1.00

    baseline_ref = baseline_income * day_factor * fest_factor
    rain_stress = float(calculate_rain_stress(rain_mm))

    expected_income = baseline_ref * (1.0 - s * rain_stress)

    if noise_sigma > 0.0:
        noise = float(generate_lognormal_noise(size=1, sigma=noise_sigma, seed=seed)[0])
    else:
        noise = 1.0

    modeled_income = expected_income * noise
    modeled_loss = max(baseline_ref - modeled_income, 0.0)

    return {
        "rain_mm": float(rain_mm),
        "rain_stress": rain_stress,
        "baseline_ref": baseline_ref,
        "expected_income": expected_income,
        "modeled_income": modeled_income,
        "modeled_loss": modeled_loss,
        "noise_applied": noise,
    }


def simulate_weather_series(
    df: pd.DataFrame,
    baseline_income: float = DEFAULT_BASELINE_INCOME,
    sensitivity: Union[str, float] = "MEDIUM",
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    seed: Optional[int] = 42,
    weekday_factor: float = DEFAULT_WEEKDAY_FACTOR,
    weekend_factor: float = DEFAULT_WEEKEND_FACTOR,
    festival_factor: float = DEFAULT_FESTIVAL_FACTOR,
) -> pd.DataFrame:
    """
    Simulates modeled income and loss across a weather DataFrame.

    Parameters:
        df: DataFrame with columns ['date', 'precipitation_mm', 'is_weekend', 'is_festival']
        baseline_income: Daily baseline (default ₹900)
        sensitivity: Sensitivity scenario ("LOW", "MEDIUM", "HIGH" or float s)
        noise_sigma: Lognormal noise standard deviation (0.0 for deterministic backtest)
        seed: Random seed for reproducible backtests

    Returns:
        DataFrame enriched with:
        ['rain_stress', 'baseline_ref', 'expected_income', 'modeled_income', 'modeled_loss']
    """
    res_df = df.copy()

    if isinstance(sensitivity, str):
        s = SENSITIVITY_SCENARIOS.get(sensitivity.upper(), SENSITIVITY_SCENARIOS["MEDIUM"])
    else:
        s = float(sensitivity)

    day_factors = np.where(res_df["is_weekend"], weekend_factor, weekday_factor)
    fest_factors = np.where(res_df["is_festival"], festival_factor, 1.00)

    baseline_refs = baseline_income * day_factors * fest_factors
    rain_stresses = calculate_rain_stress(res_df["precipitation_mm"].values)

    expected_incomes = baseline_refs * (1.0 - s * rain_stresses)

    if noise_sigma > 0.0:
        noise = generate_lognormal_noise(size=len(res_df), sigma=noise_sigma, seed=seed)
    else:
        noise = np.ones(len(res_df))

    modeled_incomes = expected_incomes * noise
    modeled_losses = np.maximum(baseline_refs - modeled_incomes, 0.0)

    res_df["rain_stress"] = rain_stresses
    res_df["baseline_ref"] = baseline_refs
    res_df["expected_income"] = expected_incomes
    res_df["modeled_income"] = modeled_incomes
    res_df["modeled_loss"] = modeled_losses

    return res_df
