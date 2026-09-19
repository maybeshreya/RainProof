"""
robustness.py - Robustness Analysis & Grid Search Engine for RainProof.

Evaluates parametric payout designs across multiple rain sensitivity scenarios (LOW, MEDIUM, HIGH).

Core Specification:
- Grid search over StartRain, FullRain, MaxPayout parameter combinations.
- Evaluate each design across LOW (s=0.25), MEDIUM (s=0.40), and HIGH (s=0.55) sensitivity scenarios.
- RobustScore = min(Coverage, Precision) across ALL 3 sensitivity scenarios (worst-case performance).
- Identifies the "Most robust design under tested assumptions".
- Vectorized implementation for maximum computation speed.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from src.simulator import SENSITIVITY_SCENARIOS, simulate_weather_series
from src.metrics import calculate_loss_coverage, calculate_payout_precision, calculate_premium

# Standard Grid Search Ranges (Section 23)
DEFAULT_START_RAIN_GRID = [10.0, 20.0, 30.0, 40.0, 50.0]
DEFAULT_FULL_RAIN_GRID = [40.0, 50.0, 60.0, 70.0, 80.0, 100.0]
DEFAULT_MAX_PAYOUT_GRID = [250.0, 500.0, 750.0, 1000.0, 1500.0]


def run_robustness_grid_search(
    df_weather: pd.DataFrame,
    start_rain_grid: Optional[List[float]] = None,
    full_rain_grid: Optional[List[float]] = None,
    max_payout_grid: Optional[List[float]] = None,
    baseline_income: float = 900.0,
    target_loss_ratio: float = 0.60,
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes fast vectorized grid search over payout designs and evaluates worst-case RobustScore.

    Parameters:
        df_weather: Weather DataFrame containing ['precipitation_mm', 'is_weekend', 'is_festival']
        start_rain_grid: List of StartRain values
        full_rain_grid: List of FullRain values
        max_payout_grid: List of MaxPayout values
        baseline_income: Modeled baseline daily income
        target_loss_ratio: Target loss ratio for premium calculation

    Returns:
        Tuple of (results_dataframe, most_robust_design_dict)
    """
    if start_rain_grid is None:
        start_rain_grid = DEFAULT_START_RAIN_GRID
    if full_rain_grid is None:
        full_rain_grid = DEFAULT_FULL_RAIN_GRID
    if max_payout_grid is None:
        max_payout_grid = DEFAULT_MAX_PAYOUT_GRID

    rain_vals = df_weather["precipitation_mm"].values
    n_days = len(df_weather)

    # 1. Precompute modeled losses for all 3 sensitivity scenarios
    sim_low = simulate_weather_series(
        df_weather, baseline_income=baseline_income, sensitivity="LOW", noise_sigma=noise_sigma, seed=seed
    )
    sim_med = simulate_weather_series(
        df_weather, baseline_income=baseline_income, sensitivity="MEDIUM", noise_sigma=noise_sigma, seed=seed
    )
    sim_high = simulate_weather_series(
        df_weather, baseline_income=baseline_income, sensitivity="HIGH", noise_sigma=noise_sigma, seed=seed
    )

    losses_low = sim_low["modeled_loss"].values
    losses_med = sim_med["modeled_loss"].values
    losses_high = sim_high["modeled_loss"].values

    sum_loss_low = float(np.sum(losses_low))
    sum_loss_med = float(np.sum(losses_med))
    sum_loss_high = float(np.sum(losses_high))

    # 2. Build parameter combination tuples (StartRain, FullRain, MaxPayout) where FullRain > StartRain
    param_combos = []
    for sr in start_rain_grid:
        for fr in full_rain_grid:
            if fr > sr:
                for mp in max_payout_grid:
                    param_combos.append((float(sr), float(fr), float(mp)))

    if not param_combos:
        raise ValueError("No valid parameter combinations where FullRain > StartRain.")

    start_rains = np.array([c[0] for c in param_combos])
    full_rains = np.array([c[1] for c in param_combos])
    max_payouts = np.array([c[2] for c in param_combos])
    n_combos = len(param_combos)

    # 3. Vectorized payout calculation: Payout matrix shape (n_combos, n_days)
    fractions = (rain_vals[None, :] - start_rains[:, None]) / (full_rains[:, None] - start_rains[:, None])
    fractions_clipped = np.clip(fractions, 0.0, 1.0)
    payout_matrix = max_payouts[:, None] * fractions_clipped

    total_payouts = np.sum(payout_matrix, axis=1)  # shape (n_combos,)

    # 4. Evaluate coverage and precision for each sensitivity scenario
    # Low sensitivity
    cov_low_arr = np.sum(np.minimum(payout_matrix, losses_low[None, :]), axis=1)
    coverage_low = np.where(sum_loss_low > 0, cov_low_arr / sum_loss_low, 1.0)
    precision_low = np.where(total_payouts > 0, cov_low_arr / total_payouts, 1.0)
    score_low = np.minimum(coverage_low, precision_low)

    # Medium sensitivity
    cov_med_arr = np.sum(np.minimum(payout_matrix, losses_med[None, :]), axis=1)
    coverage_med = np.where(sum_loss_med > 0, cov_med_arr / sum_loss_med, 1.0)
    precision_med = np.where(total_payouts > 0, cov_med_arr / total_payouts, 1.0)
    score_med = np.minimum(coverage_med, precision_med)

    # High sensitivity
    cov_high_arr = np.sum(np.minimum(payout_matrix, losses_high[None, :]), axis=1)
    coverage_high = np.where(sum_loss_high > 0, cov_high_arr / sum_loss_high, 1.0)
    precision_high = np.where(total_payouts > 0, cov_high_arr / total_payouts, 1.0)
    score_high = np.minimum(coverage_high, precision_high)

    # 5. Worst-case RobustScore across ALL 3 scenarios
    robust_scores = np.minimum(np.minimum(score_low, score_med), score_high)

    # Premium & Expected Annual Payout calculations
    years = n_days / 365.25 if n_days > 0 else 1.0
    expected_annual_payouts = total_payouts / years
    premiums = expected_annual_payouts / target_loss_ratio

    # 6. Assemble DataFrame
    results_df = pd.DataFrame(
        {
            "start_rain": start_rains,
            "full_rain": full_rains,
            "max_payout": max_payouts,
            "robust_score": robust_scores,
            "cov_low": coverage_low,
            "prec_low": precision_low,
            "uncov_low": 1.0 - coverage_low,
            "overpay_low": 1.0 - precision_low,
            "cov_med": coverage_med,
            "prec_med": precision_med,
            "uncov_med": 1.0 - coverage_med,
            "overpay_med": 1.0 - precision_med,
            "cov_high": coverage_high,
            "prec_high": precision_high,
            "uncov_high": 1.0 - coverage_high,
            "overpay_high": 1.0 - precision_high,
            "expected_annual_payout": expected_annual_payouts,
            "premium": premiums,
        }
    )

    # Sort by robust_score descending
    results_df = results_df.sort_values("robust_score", ascending=False).reset_index(drop=True)

    # Identify most robust design
    best_row = results_df.iloc[0]
    most_robust_design = {
        "title": "Most robust design under tested assumptions",
        "start_rain": float(best_row["start_rain"]),
        "full_rain": float(best_row["full_rain"]),
        "max_payout": float(best_row["max_payout"]),
        "robust_score": float(best_row["robust_score"]),
        "expected_annual_payout": float(best_row["expected_annual_payout"]),
        "premium": float(best_row["premium"]),
        "metrics_low": {
            "coverage": float(best_row["cov_low"]),
            "precision": float(best_row["prec_low"]),
            "uncovered_loss": float(best_row["uncov_low"]),
            "overpayment": float(best_row["overpay_low"]),
        },
        "metrics_med": {
            "coverage": float(best_row["cov_med"]),
            "precision": float(best_row["prec_med"]),
            "uncovered_loss": float(best_row["uncov_med"]),
            "overpayment": float(best_row["overpay_med"]),
        },
        "metrics_high": {
            "coverage": float(best_row["cov_high"]),
            "precision": float(best_row["prec_high"]),
            "uncovered_loss": float(best_row["uncov_high"]),
            "overpayment": float(best_row["overpay_high"]),
        },
        "explanation": (
            "This design maintains the highest minimum (worst-case) performance across low, medium, "
            "and high rain sensitivity assumptions without assuming a single true income loss function."
        ),
    }

    return results_df, most_robust_design
