"""
test_simulator.py - Unit tests for income simulation engine.
"""

import os
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import (
    calculate_rain_stress,
    simulate_single_day,
    simulate_weather_series,
    SENSITIVITY_SCENARIOS,
    DEFAULT_BASELINE_INCOME,
)

def test_rain_stress_zero():
    """Test rain stress at zero rainfall."""
    stress = calculate_rain_stress(0.0)
    assert stress == 0.0, f"Expected 0.0 stress at 0mm rain, got {stress}"

def test_rain_stress_values():
    """Test specific rain stress calculation points."""
    stress_50 = calculate_rain_stress(50.0)
    expected_50 = 1.0 - np.exp(-1.0)  # ~0.63212
    assert np.isclose(stress_50, expected_50, atol=1e-4)

    stress_100 = calculate_rain_stress(100.0)
    expected_100 = 1.0 - np.exp(-2.0)  # ~0.86466
    assert np.isclose(stress_100, expected_100, atol=1e-4)

def test_rain_stress_saturation():
    """Test saturating behavior for extreme rainfall."""
    stress_500 = calculate_rain_stress(500.0)
    assert np.isclose(stress_500, 1.0, atol=1e-4), "Rain stress must saturate near 1.0 for large rainfall"

def test_deterministic_income_simulation_zero_rain():
    """Test deterministic income simulation with zero rain on weekday."""
    res = simulate_single_day(
        rain_mm=0.0,
        is_weekend=False,
        is_festival=False,
        baseline_income=900.0,
        sensitivity="MEDIUM",
        noise_sigma=0.0,
    )
    assert np.isclose(res["rain_stress"], 0.0)
    assert np.isclose(res["baseline_ref"], 900.0)
    assert np.isclose(res["expected_income"], 900.0)
    assert np.isclose(res["modeled_income"], 900.0)
    assert np.isclose(res["modeled_loss"], 0.0)

def test_deterministic_income_simulation_heavy_rain():
    """Test deterministic income simulation with 50mm rain on weekend under MEDIUM sensitivity."""
    res = simulate_single_day(
        rain_mm=50.0,
        is_weekend=True,
        is_festival=False,
        baseline_income=900.0,
        sensitivity="MEDIUM",
        noise_sigma=0.0,
    )

    # Weekend baseline = 900 * 1.10 = 990
    assert np.isclose(res["baseline_ref"], 990.0)
    # Rain stress = 1 - e^-1 ≈ 0.63212
    # Medium s = 0.40 -> expected income = 990 * (1 - 0.40 * 0.63212) = 990 * (1 - 0.252848) = 990 * 0.747152 = 739.68
    expected_income = 990.0 * (1.0 - 0.40 * (1.0 - np.exp(-1.0)))
    assert np.isclose(res["expected_income"], expected_income, atol=1e-2)
    assert np.isclose(res["modeled_loss"], 990.0 - expected_income, atol=1e-2)

def test_sensitivity_levels():
    """Test that higher sensitivity yields lower modeled income for rain > 0."""
    rain = 40.0
    low_res = simulate_single_day(rain_mm=rain, sensitivity="LOW", noise_sigma=0.0)
    med_res = simulate_single_day(rain_mm=rain, sensitivity="MEDIUM", noise_sigma=0.0)
    high_res = simulate_single_day(rain_mm=rain, sensitivity="HIGH", noise_sigma=0.0)

    assert low_res["modeled_income"] > med_res["modeled_income"] > high_res["modeled_income"]
    assert low_res["modeled_loss"] < med_res["modeled_loss"] < high_res["modeled_loss"]

def test_seed_reproducibility():
    """Test that identical seeds yield identical stochastic outputs."""
    res1 = simulate_single_day(rain_mm=25.0, noise_sigma=0.12, seed=12345)
    res2 = simulate_single_day(rain_mm=25.0, noise_sigma=0.12, seed=12345)
    assert res1["modeled_income"] == res2["modeled_income"]
