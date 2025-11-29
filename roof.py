import numpy as np
import pandas as pd


def roofing_filter(df, hp_period=48, ss_period=10, price_col="Close"):
    """
    Apply John Ehlers' Roofing Filter to an OHLC dataframe.
    Returns a new column 'Roof' containing the roofing filter output.
    """

    price = df[price_col].values
    length = len(price)

    # ----- Step 1: High-Pass Filter -----
    hp = np.zeros(length)

    # High-pass constant
    alpha_hp = (
        np.cos(1.414 * 2 * np.pi / hp_period)
        + np.sin(1.414 * 2 * np.pi / hp_period)
        - 1
    ) / np.cos(1.414 * 2 * np.pi / hp_period)

    for i in range(1, length):
        hp[i] = (1 - alpha_hp / 2) * (price[i] - price[i - 1]) + alpha_hp * hp[i - 1]

    # ----- Step 2: Super Smoother Filter -----
    roof = np.zeros(length)

    # Precompute coefficients
    a1 = np.exp(-1.414 * 2 * np.pi / ss_period)
    b1 = 2 * a1 * np.cos(1.414 * 2 * np.pi / ss_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - b1 - c3

    # Apply 2-pole filter
    for i in range(2, length):
        roof[i] = c1 * hp[i] + c2 * roof[i - 1] + c3 * roof[i - 2]

    df["Roof"] = roof
    return df
