import pandas as pd
import numpy as np
import ut_tools_ssc as ssc
import streamlit as st
from app import eod
from barchart import plot_tv_ohlc_dark_v2


def main():
    st.set_page_config(page_title="SSC Data Viewer", layout="wide")
    symbol = st.selectbox(
        "Select Symbol",
        ["CRUDEOIL", "GOLD", "SILVER", "NATURALGAS", "NIFTY", "BANKNIFTY"],
    )
    df = eod(symbol, "2023-01-01", "2023-12-31")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = ssc.SSC(df)
    df = df[:-100]
    df["x"] = range(len(df))
    tops = [
        (row["x"], row["high"])
        for _, row in df[df["swing_point"] == df["high"]].iterrows()
    ]

    bottoms = [
        (row["x"], row["low"])
        for _, row in df[df["swing_point"] == df["low"]].iterrows()
    ]

    fig = plot_tv_ohlc_dark_v2(
        df,
        title="SSC Chart",
        tops=tops,
        bottoms=bottoms,
    )
    st.pyplot(fig)
    st.title("SSC Data Viewer")
    st.dataframe(df[["open", "high", "low", "close", "bar_type", "swing_point"]])


if __name__ == "__main__":
    main()
