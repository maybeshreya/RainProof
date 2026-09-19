"""
weather.py - Historical and forecast weather data retrieval and caching for RainProof.

Provides functions to fetch Mumbai historical weather (2019-2025) and short-term forecast
from Open-Meteo, validate the data, apply festival/weekend tags, and handle local CSV caching.
"""

import os
import pandas as pd
import numpy as np
import requests
from typing import Optional, List, Dict, Tuple

# Constants
MUMBAI_LAT = 19.0760
MUMBAI_LON = 72.8777
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "weather.csv")

# Major Indian Festival Dates (Configurable assumptions for Mumbai, 2019-2025)
# Includes key festival days (Diwali, Ganesh Chaturthi, Holi, Eid al-Fitr, Independence Day, Republic Day, New Year)
DEFAULT_FESTIVAL_DATES = {
    # 2019
    "2019-01-26", "2019-03-21", "2019-06-05", "2019-08-15", "2019-09-02", "2019-10-27", "2019-10-28", "2019-12-25",
    # 2020
    "2020-01-26", "2020-03-10", "2020-05-24", "2020-08-15", "2020-08-22", "2020-11-14", "2020-11-15", "2020-12-25",
    # 2021
    "2021-01-26", "2021-03-29", "2021-05-13", "2021-08-15", "2021-09-10", "2021-11-04", "2021-11-05", "2021-12-25",
    # 2022
    "2022-01-26", "2022-03-18", "2022-05-03", "2022-08-15", "2022-08-31", "2022-10-24", "2022-10-25", "2022-12-25",
    # 2023
    "2023-01-26", "2023-03-08", "2023-04-22", "2023-08-15", "2023-09-19", "2023-11-12", "2023-11-13", "2023-12-25",
    # 2024
    "2024-01-26", "2024-03-25", "2024-04-11", "2024-08-15", "2024-09-07", "2024-11-01", "2024-11-02", "2024-12-25",
    # 2025
    "2025-01-26", "2025-03-14", "2025-03-31", "2025-08-15", "2025-08-27", "2025-10-20", "2025-10-21", "2025-12-25",
    # 2026
    "2026-01-26", "2026-03-04", "2026-03-20", "2026-08-15", "2026-09-16", "2026-11-08", "2026-12-25",
}


def validate_weather_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates, cleans, and structures weather DataFrame.

    Checks:
    - Required columns present: date, precipitation_mm, temp_max_c
    - No duplicate dates
    - Non-negative precipitation values
    - No unexpected missing values
    - Adds is_weekend and is_festival indicator columns
    """
    required_cols = {"date", "precipitation_mm", "temp_max_c"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame missing required columns. Expected {required_cols}, got {set(df.columns)}")

    # Ensure clean copy
    df = df.copy()

    # Convert date to string YYYY-MM-DD format
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Check for duplicate dates
    if df["date"].duplicated().any():
        dup_count = df["date"].duplicated().sum()
        raise ValueError(f"Weather dataset contains {dup_count} duplicate date entries.")

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Clean numeric fields
    df["precipitation_mm"] = pd.to_numeric(df["precipitation_mm"], errors="coerce").fillna(0.0)
    df["precipitation_mm"] = df["precipitation_mm"].clip(lower=0.0)

    df["temp_max_c"] = pd.to_numeric(df["temp_max_c"], errors="coerce").ffill().bfill().fillna(30.0)

    # Calculate indicators
    dates_dt = pd.to_datetime(df["date"])
    # Weekend: Saturday (5) or Sunday (6)
    df["is_weekend"] = dates_dt.dt.dayofweek >= 5
    # Festival check
    df["is_festival"] = df["date"].isin(DEFAULT_FESTIVAL_DATES)

    return df


def fetch_historical_weather(
    start_date: str = "2019-01-01",
    end_date: str = "2025-12-31",
    lat: float = MUMBAI_LAT,
    lon: float = MUMBAI_LON,
    cache_path: str = DEFAULT_CACHE_PATH,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetches historical weather for Mumbai from Open-Meteo API or loads from cached CSV.

    Parameters:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        lat: Latitude (default Mumbai 19.0760)
        lon: Longitude (default Mumbai 72.8777)
        cache_path: Path to local CSV cache
        force_refresh: If True, ignores cached CSV and fetches fresh data from API

    Returns:
        Clean, validated pandas DataFrame with columns:
        ['date', 'precipitation_mm', 'temp_max_c', 'is_weekend', 'is_festival']
    """
    # Check cache first unless force refresh requested
    if not force_refresh and os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path)
            valid_df = validate_weather_dataframe(cached_df)
            return valid_df
        except Exception as e:
            # If reading cache fails, log warning and attempt fresh API fetch
            print(f"Warning: Failed to load cached weather data from {cache_path}: {e}. Fetching from API...")

    # Fetch from Open-Meteo Archive API
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if "daily" not in data:
            raise KeyError(f"Open-Meteo API response missing 'daily' payload: {data.keys()}")

        raw_df = pd.DataFrame(
            {
                "date": data["daily"]["time"],
                "precipitation_mm": data["daily"]["precipitation_sum"],
                "temp_max_c": data["daily"]["temperature_2m_max"],
            }
        )

        clean_df = validate_weather_dataframe(raw_df)

        # Cache to CSV
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        clean_df.to_csv(cache_path, index=False)

        return clean_df

    except Exception as api_err:
        # Fallback to cache if API request fails and cache exists
        if os.path.exists(cache_path):
            print(f"Warning: API request failed ({api_err}). Falling back to existing cached dataset.")
            cached_df = pd.read_csv(cache_path)
            return validate_weather_dataframe(cached_df)
        else:
            raise RuntimeError(
                f"Failed to retrieve weather data from Open-Meteo API and no local cache was found: {api_err}"
            ) from api_err


def fetch_forecast_weather(
    forecast_days: int = 7,
    lat: float = MUMBAI_LAT,
    lon: float = MUMBAI_LON,
) -> pd.DataFrame:
    """
    Fetches short-term weather forecast for Mumbai from Open-Meteo Forecast API.

    Returns:
        Clean, validated pandas DataFrame for the next `forecast_days` days.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum",
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        raw_df = pd.DataFrame(
            {
                "date": data["daily"]["time"],
                "precipitation_mm": data["daily"]["precipitation_sum"],
                "temp_max_c": data["daily"]["temperature_2m_max"],
            }
        )

        return validate_weather_dataframe(raw_df)
    except Exception as err:
        print(f"Warning: Forecast API unavailable ({err}). Returning synthetic placeholder forecast.")
        # Fallback synthetic forecast if offline
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        dates = pd.date_range(start=today, periods=forecast_days, freq="D").strftime("%Y-%m-%d")
        placeholder = pd.DataFrame(
            {
                "date": dates,
                "precipitation_mm": [0.0, 15.0, 45.0, 5.0, 0.0, 60.0, 2.0][:forecast_days],
                "temp_max_c": [31.5, 30.0, 28.5, 29.0, 31.0, 27.5, 30.5][:forecast_days],
            }
        )
        return validate_weather_dataframe(placeholder)
