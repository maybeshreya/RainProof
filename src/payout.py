"""
payout.py - Parametric Payout Engine for RainProof.

Implements the linear parametric payout curve bounded by StartRain, FullRain, and MaxPayout.

Formula:
    Payout = MaxPayout * clip((rain - StartRain) / (FullRain - StartRain), 0, 1)

Validation Rules:
    - FullRain > StartRain
    - MaxPayout >= 0
"""

import numpy as np
import pandas as pd
from typing import Union

DEFAULT_START_RAIN = 20.0   # mm
DEFAULT_FULL_RAIN = 80.0    # mm
DEFAULT_MAX_PAYOUT = 500.0  # ₹


def validate_payout_params(start_rain: float, full_rain: float, max_payout: float) -> None:
    """
    Validates payout parameter bounds.

    Raises:
        ValueError if FullRain <= StartRain or MaxPayout < 0.
    """
    if full_rain <= start_rain:
        raise ValueError(
            f"Invalid payout parameters: FullRain ({full_rain}mm) must be strictly greater than StartRain ({start_rain}mm)."
        )
    if max_payout < 0:
        raise ValueError(f"Invalid payout parameters: MaxPayout ({max_payout}) cannot be negative.")


def calculate_payout(
    rain_mm: Union[float, np.ndarray],
    start_rain: float = DEFAULT_START_RAIN,
    full_rain: float = DEFAULT_FULL_RAIN,
    max_payout: float = DEFAULT_MAX_PAYOUT,
) -> Union[float, np.ndarray]:
    """
    Calculates parametric payout for a given rainfall value or array.

    Parameters:
        rain_mm: Single rainfall float or array of values (mm)
        start_rain: Threshold rainfall where payouts begin (mm)
        full_rain: Threshold rainfall where payouts reach max (mm)
        max_payout: Maximum daily payout amount (₹)

    Returns:
        Payout amount (₹) clipped between 0 and max_payout.
    """
    validate_payout_params(start_rain, full_rain, max_payout)

    rain_arr = np.asarray(rain_mm, dtype=float)
    rain_clipped = np.clip(rain_arr, 0.0, None)

    fraction = (rain_clipped - start_rain) / (full_rain - start_rain)
    fraction_clipped = np.clip(fraction, 0.0, 1.0)

    payout = max_payout * fraction_clipped

    if np.isscalar(rain_mm):
        return float(payout)
    return payout


def calculate_payout_series(
    df: pd.DataFrame,
    start_rain: float = DEFAULT_START_RAIN,
    full_rain: float = DEFAULT_FULL_RAIN,
    max_payout: float = DEFAULT_MAX_PAYOUT,
    rain_col: str = "precipitation_mm",
) -> pd.Series:
    """
    Calculates parametric payout series across a weather DataFrame.
    """
    validate_payout_params(start_rain, full_rain, max_payout)
    payouts = calculate_payout(df[rain_col].values, start_rain, full_rain, max_payout)
    return pd.Series(payouts, index=df.index, name="payout")
