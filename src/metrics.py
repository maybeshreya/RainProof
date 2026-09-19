"""
metrics.py - Performance, Basis Risk, Premium, and Volatility Metrics for RainProof.

Implements exact metrics definitions from Section 20 & 21 of the specification:
1. Loss Coverage = sum(min(Payout, Loss)) / sum(Loss)
2. Payout Precision = sum(min(Payout, Loss)) / sum(Payout)
3. Uncovered Loss = 1 - Loss Coverage
4. Overpayment = 1 - Payout Precision
5. Premium = ExpectedAnnualPayout / TargetLossRatio
6. Income Volatility Reduction using monthly income standard deviation
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, Optional

DEFAULT_TARGET_LOSS_RATIO = 0.60  # 60% default target loss ratio


def calculate_loss_coverage(payouts: np.ndarray, losses: np.ndarray) -> float:
    """
    Calculates Loss Coverage metric.

    Formula:
        Coverage = sum(min(Payout_i, Loss_i)) / sum(Loss_i)

    Returns:
        Float between 0.0 and 1.0 (1.0 if total loss is zero).
    """
    total_loss = np.sum(losses)
    if total_loss <= 0.0:
        return 1.0

    covered_loss = np.sum(np.minimum(payouts, losses))
    coverage = float(covered_loss / total_loss)
    return float(np.clip(coverage, 0.0, 1.0))


def calculate_payout_precision(payouts: np.ndarray, losses: np.ndarray) -> float:
    """
    Calculates Payout Precision metric.

    Formula:
        Precision = sum(min(Payout_i, Loss_i)) / sum(Payout_i)

    Returns:
        Float between 0.0 and 1.0 (1.0 if total payout is zero).
    """
    total_payout = np.sum(payouts)
    if total_payout <= 0.0:
        return 1.0

    covered_loss = np.sum(np.minimum(payouts, losses))
    precision = float(covered_loss / total_payout)
    return float(np.clip(precision, 0.0, 1.0))


def calculate_uncovered_loss(coverage: float) -> float:
    """Uncovered Loss = 1 - Coverage"""
    return float(np.clip(1.0 - coverage, 0.0, 1.0))


def calculate_overpayment(precision: float) -> float:
    """Overpayment = 1 - Precision"""
    return float(np.clip(1.0 - precision, 0.0, 1.0))


def calculate_premium(expected_annual_payout: float, target_loss_ratio: float = DEFAULT_TARGET_LOSS_RATIO) -> float:
    """
    Calculates annual insurance premium based on target loss ratio.

    Formula:
        Premium = ExpectedAnnualPayout / TargetLossRatio
    """
    if target_loss_ratio <= 0.0 or target_loss_ratio > 1.0:
        raise ValueError(f"Target loss ratio must be between (0, 1]. Got {target_loss_ratio}")
    return float(expected_annual_payout / target_loss_ratio)


def calculate_volatility_metrics(
    df: pd.DataFrame,
    modeled_income_col: str = "modeled_income",
    payout_col: str = "payout",
    date_col: str = "date",
) -> Dict[str, float]:
    """
    Calculates monthly income standard deviation for unprotected vs protected modeled income.

    Returns:
        - vol_unprotected (Monthly std dev without protection)
        - vol_protected (Monthly std dev with parametric protection)
        - vol_reduction_pct (Percentage reduction in volatility)
    """
    df_temp = df.copy()
    df_temp["protected_income"] = df_temp[modeled_income_col] + df_temp[payout_col]

    # Convert date to year-month period for grouping
    df_temp["year_month"] = pd.to_datetime(df_temp[date_col]).dt.to_period("M")

    monthly = df_temp.groupby("year_month").agg(
        unprotected_monthly=(modeled_income_col, "sum"),
        protected_monthly=("protected_income", "sum"),
    )

    vol_unprotected = float(monthly["unprotected_monthly"].std(ddof=1)) if len(monthly) > 1 else 0.0
    vol_protected = float(monthly["protected_monthly"].std(ddof=1)) if len(monthly) > 1 else 0.0

    if np.isnan(vol_unprotected):
        vol_unprotected = 0.0
    if np.isnan(vol_protected):
        vol_protected = 0.0

    if vol_unprotected > 0.0:
        vol_reduction_pct = float((vol_unprotected - vol_protected) / vol_unprotected * 100.0)
    else:
        vol_reduction_pct = 0.0

    return {
        "volatility_unprotected": vol_unprotected,
        "volatility_protected": vol_protected,
        "volatility_reduction_pct": vol_reduction_pct,
    }


def calculate_all_metrics(
    payouts: np.ndarray,
    losses: np.ndarray,
    total_days: int,
    target_loss_ratio: float = DEFAULT_TARGET_LOSS_RATIO,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Computes complete performance, premium, and basis risk metrics suite.
    """
    payouts = np.asarray(payouts, dtype=float)
    losses = np.asarray(losses, dtype=float)

    total_loss = float(np.sum(losses))
    total_payout = float(np.sum(payouts))
    effective_covered = float(np.sum(np.minimum(payouts, losses)))

    coverage = calculate_loss_coverage(payouts, losses)
    precision = calculate_payout_precision(payouts, losses)
    uncovered_loss = calculate_uncovered_loss(coverage)
    overpayment = calculate_overpayment(precision)

    # Expected annual payout
    years = total_days / 365.25 if total_days > 0 else 1.0
    expected_annual_payout = total_payout / years if years > 0 else 0.0

    premium = calculate_premium(expected_annual_payout, target_loss_ratio)

    metrics = {
        "total_modeled_loss": total_loss,
        "total_payout": total_payout,
        "effective_covered_loss": effective_covered,
        "loss_coverage": coverage,
        "payout_precision": precision,
        "uncovered_loss": uncovered_loss,
        "overpayment": overpayment,
        "expected_annual_payout": expected_annual_payout,
        "premium": premium,
        "target_loss_ratio": target_loss_ratio,
        "actual_loss_ratio": (total_payout / (premium * years)) if (premium > 0 and years > 0) else target_loss_ratio,
    }

    if df is not None and "modeled_income" in df.columns:
        vol_metrics = calculate_volatility_metrics(df)
        metrics.update(vol_metrics)

    return metrics
