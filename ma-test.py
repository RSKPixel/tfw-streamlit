import streamlit as st
import pandas as pd
from core import get_sqlalchemy_engine, fetch_ohlcv, fetch_symbols, trend_identification
from datetime import datetime
import talib as ta
import numpy as np


def main():
    st.set_page_config(page_title="TFW Dashboard", layout="wide")

    symbol = "NIFTY-I"
    symbol_list = fetch_symbols().to_dict(orient="records")

    filter_cols = st.columns(2)
    with filter_cols[0]:
        symbol = st.selectbox(
            "Select Symbol", options=[s["symbol"] for s in symbol_list]
        )
    with filter_cols[1]:
        recent_signals = st.slider(
            "Number of Recent Signals to Display", min_value=1, max_value=100, value=50
        )
    data = fetch_ohlcv(
        symbol=symbol,
        start_date="2016-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        timeframe="1day",
    )

    data = trend_identification(data)
    data = data[data["signal"] != "none"]
    data = data[-recent_signals:]

    st.dataframe(data, hide_index=True)


if __name__ == "__main__":
    main()
