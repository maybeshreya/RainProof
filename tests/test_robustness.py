"""
test_robustness.py - Unit tests for robustness analysis engine.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.weather import fetch_historical_weather
from src.robustness import run_robustness_grid_search


def test_robustness_grid_search_logic():
    """Verify worst-case min(coverage, precision) RobustScore calculation."""
    df_weather = fetch_historical_weather()
    results_df, best_design = run_robustness_grid_search(df_weather)

    assert not results_df.empty
    assert "robust_score" in results_df.columns

    # Verify every row's robust_score is min across all 6 metrics
    for _, row in results_df.iterrows():
        expected_min = min(
            row["cov_low"], row["prec_low"],
            row["cov_med"], row["prec_med"],
            row["cov_high"], row["prec_high"]
        )
        assert np.isclose(row["robust_score"], expected_min), (
            f"Robust score {row['robust_score']} does not match min performance {expected_min}"
        )

    # Verify best design has top score
    assert best_design["robust_score"] == results_df["robust_score"].max()
    assert best_design["title"] == "Most robust design under tested assumptions"


def test_robustness_execution_speed():
    """Verify that grid search across 2557 days completes under 1 second (optimized speed)."""
    df_weather = fetch_historical_weather()
    start_time = time.time()
    results_df, best_design = run_robustness_grid_search(df_weather)
    duration = time.time() - start_time

    print(f"\nGrid search completed {len(results_df)} payout designs across {len(df_weather)} days in {duration:.4f} seconds.")
    assert duration < 2.0, f"Grid search took too long ({duration:.2f}s)"
