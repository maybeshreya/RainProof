"""
test_metrics.py - Unit tests for protection metrics, basis risk, and premium calculations.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics import (
    calculate_loss_coverage,
    calculate_payout_precision,
    calculate_uncovered_loss,
    calculate_overpayment,
    calculate_premium,
    calculate_all_metrics,
)


def test_manually_verifiable_metrics_vector():
    """
    Test metrics against a manually calculated 4-day vector:
        Losses:  [100, 200,   0, 100]  -> Total Loss   = 400
        Payouts: [100, 100, 100,   0]  -> Total Payout = 300
        min:     [100, 100,   0,   0]  -> Covered Loss = 200

    Expected:
        Coverage  = 200 / 400 = 0.50 (50%)
        Precision = 200 / 300 = 2/3 ≈ 0.6667 (66.67%)
        Uncovered Loss = 1 - 0.50 = 0.50 (50%)
        Overpayment    = 1 - 2/3  = 1/3  ≈ 0.3333 (33.33%)
    """
    losses = np.array([100.0, 200.0, 0.0, 100.0])
    payouts = np.array([100.0, 100.0, 100.0, 0.0])

    coverage = calculate_loss_coverage(payouts, losses)
    precision = calculate_payout_precision(payouts, losses)
    uncovered = calculate_uncovered_loss(coverage)
    overpay = calculate_overpayment(precision)

    assert np.isclose(coverage, 0.50)
    assert np.isclose(precision, 2.0 / 3.0)
    assert np.isclose(uncovered, 0.50)
    assert np.isclose(overpay, 1.0 / 3.0)


def test_zero_loss_and_zero_payout_edge_cases():
    """Test boundary behavior when loss or payout sum is zero."""
    # Zero loss, zero payout -> 1.0 coverage, 1.0 precision
    cov_zero = calculate_loss_coverage(np.array([0.0]), np.array([0.0]))
    prec_zero = calculate_payout_precision(np.array([0.0]), np.array([0.0]))
    assert cov_zero == 1.0
    assert prec_zero == 1.0

    # Loss with zero payout -> 0.0 coverage, 1.0 precision
    cov_no_pay = calculate_loss_coverage(np.array([0.0]), np.array([100.0]))
    prec_no_pay = calculate_payout_precision(np.array([0.0]), np.array([100.0]))
    assert cov_no_pay == 0.0
    assert prec_no_pay == 1.0

    # Zero loss with payout -> 1.0 coverage, 0.0 precision
    cov_overpay = calculate_loss_coverage(np.array([100.0]), np.array([0.0]))
    prec_overpay = calculate_payout_precision(np.array([100.0]), np.array([0.0]))
    assert cov_overpay == 1.0
    assert prec_overpay == 0.0


def test_premium_calculation():
    """Test premium formula: Premium = ExpectedAnnualPayout / TargetLossRatio"""
    # Annual Payout = ₹1200, Target Loss Ratio = 0.60 -> Premium = 2000
    prem = calculate_premium(expected_annual_payout=1200.0, target_loss_ratio=0.60)
    assert np.isclose(prem, 2000.0)

    # Invalid target loss ratio
    with pytest.raises(ValueError, match="Target loss ratio"):
        calculate_premium(1200.0, target_loss_ratio=0.0)

    with pytest.raises(ValueError, match="Target loss ratio"):
        calculate_premium(1200.0, target_loss_ratio=1.5)
