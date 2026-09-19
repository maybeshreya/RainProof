"""
test_phase1.py - Verification script for Phase 1 (weather data engine).
"""

import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.weather import fetch_historical_weather, fetch_forecast_weather, DEFAULT_CACHE_PATH

def test_phase1():
    print("==========================================")
    print("RUNNING PHASE 1 VERIFICATION TEST")
    print("==========================================")

    # 1. Clear cache if exists to test API retrieval and saving
    if os.path.exists(DEFAULT_CACHE_PATH):
        print(f"Removing existing cache file at {DEFAULT_CACHE_PATH} for clean test...")
        os.remove(DEFAULT_CACHE_PATH)

    # 2. Fetch historical weather data
    print("Fetching historical weather (2019-01-01 to 2025-12-31)...")
    df = fetch_historical_weather(start_date="2019-01-01", end_date="2025-12-31", force_refresh=True)

    # 3. Assertions
    assert isinstance(df, pd.DataFrame), "Output must be a pandas DataFrame"
    assert not df.empty, "DataFrame must not be empty"
    assert df["date"].iloc[0] == "2019-01-01", f"Start date expected '2019-01-01', got '{df['date'].iloc[0]}'"
    assert df["date"].iloc[-1] == "2025-12-31", f"End date expected '2025-12-31', got '{df['date'].iloc[-1]}'"
    assert not df["date"].duplicated().any(), "Date duplicates detected!"
    assert df.isnull().sum().sum() == 0, "Unexpected missing values found!"
    assert (df["precipitation_mm"] >= 0).all(), "Negative precipitation values found!"
    assert os.path.exists(DEFAULT_CACHE_PATH), "Cache file data/weather.csv was not created!"

    expected_cols = ["date", "precipitation_mm", "temp_max_c", "is_weekend", "is_festival"]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    print(f"SUCCESS!")
    print(f"Cache created at: {DEFAULT_CACHE_PATH}")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rainfall stats (mm): min={df['precipitation_mm'].min()}, max={df['precipitation_mm'].max()}, mean={df['precipitation_mm'].mean():.2f}")
    print("\nFirst 5 rows:")
    print(df.head())

    # 4. Verify loading from cache
    print("\nTesting loading directly from cached file...")
    cached_df = fetch_historical_weather(force_refresh=False)
    assert len(cached_df) == len(df), "Cached dataframe row count mismatch"
    print("Cache reload verified successfully.")

    # 5. Verify forecast retrieval
    print("\nTesting forecast API retrieval...")
    fc_df = fetch_forecast_weather(forecast_days=7)
    print(f"Forecast Rows: {len(fc_df)}")
    print(fc_df.head(3))
    print("\nPhase 1 execution fully verified!")

if __name__ == "__main__":
    test_phase1()
