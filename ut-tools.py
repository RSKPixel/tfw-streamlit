import numpy as np
from app import eod
import pandas as pd


def mvf(close, low, lookback=20):
    """
    Compute ASC(lookback) and return MVF for a dataframe with 'close' & 'low' column.
    Returns dataframe with a new column 'asc'.

    mvf = (ASC(lookback) - low) / ASC(lookback) * 100
    """
    asc = np.full(len(close), np.nan)

    for i in range(len(close)):
        if i < lookback:
            continue
        highest = close[i]

        for j in range((i - lookback) + 1, i):
            if close[j] > close[j - 1]:
                highest = close[j]
                break

        asc[i] = highest

    mvf = (asc - low) / asc * 100
    return mvf


def vix_fix(close, low, lookback=20):
    highest_close = close.rolling(window=lookback).max()
    vix_fix = (highest_close - low) / highest_close * 100
    return vix_fix


def atr(data: pd.DataFrame, lookback=4) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    df = data.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]

    df["range"] = high - low
    atr = df["range"].rolling(window=lookback).mean()

    return atr


def acr(data: pd.DataFrame, lookback=4) -> pd.Series:
    """Calculate Average Close Range (ACR)"""
    """Find the highest close range over a lookback period"""
    """Find the lowest close range over a lookback period"""
    """ACR = Highest Close Range - Lowest Close Range over lookback period"""

    df = data.copy()
    df["higest_close"] = df["close"].rolling(window=lookback).max()
    df["lowest_close"] = df["close"].rolling(window=lookback).min()
    df["close_range"] = df["higest_close"] - df["lowest_close"]

    acr = df["close_range"]
    return acr


if __name__ == "__main__":
    # data = eod("CRUDEOIL", "2025-10-01", "2025-11-30")
    data = pd.read_csv("data/bp-data.csv")
    data = data.sort_index()

    data["vix_fix"] = vix_fix(data["close"], data["low"], lookback=20)
    data["mvf"] = mvf(data["close"], data["low"], lookback=20)
    data["atr_4"] = atr(data, lookback=4)
    data["acr_4"] = acr(data, lookback=4)
    data = data.round(4)

    print(data[["acr_4", "atr_4", "mvf", "vix_fix"]].tail(20))
