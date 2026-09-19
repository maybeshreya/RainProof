"""
test_payout.py - Unit tests for parametric payout engine.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.payout import calculate_payout, validate_payout_params, calculate_payout_series


def test_payout_below_start_rain():
    """Rain below StartRain must produce 0 payout."""
    payout = calculate_payout(rain_mm=10.0, start_rain=20.0, full_rain=80.0, max_payout=500.0)
    assert payout == 0.0


def test_payout_at_start_rain():
    """Rain exactly at StartRain must produce 0 payout."""
    payout = calculate_payout(rain_mm=20.0, start_rain=20.0, full_rain=80.0, max_payout=500.0)
    assert payout == 0.0


def test_payout_at_midpoint():
    """Rain exactly halfway between StartRain and FullRain must produce half MaxPayout."""
    # (50 - 20) / (80 - 20) = 30 / 60 = 0.5 -> Payout = 500 * 0.5 = 250
    payout = calculate_payout(rain_mm=50.0, start_rain=20.0, full_rain=80.0, max_payout=500.0)
    assert payout == 250.0


def test_payout_at_full_rain():
    """Rain at FullRain must produce MaxPayout."""
    payout = calculate_payout(rain_mm=80.0, start_rain=20.0, full_rain=80.0, max_payout=500.0)
    assert payout == 500.0


def test_payout_above_full_rain():
    """Rain above FullRain must saturate at MaxPayout."""
    payout = calculate_payout(rain_mm=150.0, start_rain=20.0, full_rain=80.0, max_payout=500.0)
    assert payout == 500.0


def test_invalid_parameters():
    """FullRain <= StartRain or negative MaxPayout must raise ValueError."""
    with pytest.raises(ValueError, match="FullRain"):
        validate_payout_params(start_rain=50.0, full_rain=20.0, max_payout=500.0)

    with pytest.raises(ValueError, match="FullRain"):
        validate_payout_params(start_rain=50.0, full_rain=50.0, max_payout=500.0)

    with pytest.raises(ValueError, match="MaxPayout"):
        validate_payout_params(start_rain=20.0, full_rain=80.0, max_payout=-100.0)
