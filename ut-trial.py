import pandas as pd
import streamlit as st
from swingchart import plot_tv_ohlc_dark_v2
import numpy as np
from datetime import datetime, timedelta


def main():
    st.title("UT Data Tester")
    st.set_page_config(layout="wide")
    symbol = st.selectbox(
        "Select Symbol",
        ["nifty", "banknifty", "gold", "crudeoil"],
    )
    df = pd.read_csv(f"data/{symbol}-dataset.csv")
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
        }
    )
    df["swing_point"] = df["SSC"]
    df["bar_type"] = ""
    df["date"] = pd.to_datetime(df["date"])
    df = roofing_filter(df, hp_period=48, ss_period=10, price_col="close")
    df["buy"] = (df["Roof"].shift(1) < df["Roof"]) & (df["Roof"] < df["Roof"].shift(-1))
    df["sell"] = (df["Roof"].shift(1) > df["Roof"]) & (
        df["Roof"] > df["Roof"].shift(-1)
    )
    df.set_index("date", inplace=True)
    df.sort_values(by=["date"], inplace=True)
    df = df[(df.index >= datetime(2022, 7, 19)) & (df.index <= datetime(2023, 7, 19))]
    df = df.reset_index()
    df = df[:-100]
    df["x"] = range(len(df))
    df["swing"] = np.where((df["high"] == df["SSC"]), "swing-high", "")
    df["swing"] = np.where((df["low"] == df["SSC"]), "swing-low", df["swing"])
    tops = [
        (row["x"], row["high"]) for _, row in df[df["swing"] == "swing-high"].iterrows()
    ]
    bottoms = [
        (row["x"], row["low"]) for _, row in df[df["swing"] == "swing-low"].iterrows()
    ]
    st.write(f"No of Swing tops found: {len(tops)}")
    st.write(f"No of Swing bottoms found: {len(bottoms)}")
    fig = plot_tv_ohlc_dark_v2(
        data=df,
        tops=tops,
        bottoms=bottoms,
        date_fmt="%Y-%m-%d",
        title="Swing Chart",
    )
    st.pyplot(fig)
    st.dataframe(
        df[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "SSC",
                "swing",
                "Roof",
                "buy",
                "sell",
            ]
        ]
    )
    pass


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


if __name__ == "__main__":
    main()
