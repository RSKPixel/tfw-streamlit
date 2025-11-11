import streamlit as st
import pandas as pd
from core import get_sqlalchemy_engine, ohlcv
from datetime import datetime

engine, db_error = get_sqlalchemy_engine()

if db_error:
    st.error(db_error)
    st.stop()


def main():
    st.title("Moving Average Test App")
    data = ohlcv(
        symbol="NIFTY-I",
        start_date="2016-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        timeframe="1day",
        engine=engine,
    )

    st.dataframe(data)


if __name__ == "__main__":
    main()
