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

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_values(by=["date"], inplace=True)
    df = df[-120:]
    df["x"] = range(len(df))
    df["swing"] = np.where((df["high"] == df["SSC"]), "swing-high", "")
    df["swing"] = np.where((df["low"] == df["SSC"]), "swing-low", df["swing"])
    tops = [
        (row["x"], row["high"]) for _, row in df[df["swing"] == "swing-high"].iterrows()
    ]
    bottoms = [
        (row["x"], row["low"]) for _, row in df[df["swing"] == "swing-low"].iterrows()
    ]
    from_date = df.index.min().strftime("%Y-%m-%d")
    to_date = df.index.max().strftime("%Y-%m-%d")
    st.text(f"Displaying data from {from_date} to {to_date}")
    title = f"Swing Chart of {symbol.upper()} from {from_date} to {to_date}"
    fig = plot_tv_ohlc_dark_v2(
        data=df,
        tops=tops,
        bottoms=bottoms,
        date_fmt="%Y-%m-%d",
        title=title,
    )
    st.pyplot(fig)
    st.dataframe(df[["open", "high", "low", "close", "SSC", "swing"]])
    pass


if __name__ == "__main__":
    main()
