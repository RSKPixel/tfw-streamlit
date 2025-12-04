import pandas as pd
import numpy as np
from ut_tools_ssc import SwingPoints2
import streamlit as st
from app import eod
from barchart import plot_tv_ohlc_dark_v2
from bokeh_chart import plot_tv_ohlc_bokeh
from streamlit_bokeh import streamlit_bokeh
from ut_tools import mvf, vix_fix, atr, acr
from datetime import datetime, timedelta


def main():
    st.set_page_config(page_title="SSC Data Viewer", layout="wide")
    symbol = "NIFTY"
    symbol = st.selectbox(
        "Select Symbol",
        ["CRUDEOIL", "GOLD", "SILVER", "NATURALGAS", "NIFTY", "BANKNIFTY"],
        index=0,
    )
    df_ohlc = eod(
        symbol,
        datetime.now() - timedelta(days=365 * 2),
        datetime.now().strftime("%Y-%m-%d"),
    )
    df_ohlc["date"] = pd.to_datetime(df_ohlc["date"])
    df_ohlc.set_index("date", inplace=True)

    df = SwingPoints2(df_ohlc)

    # Identify duplicate swings
    s = df.loc[df["swing"].isin(["high", "low"]), "swing"]
    dup = s.eq(s.shift(1))
    df["duplicate_swings"] = False
    df.loc[dup.index, "duplicate_swings"] = dup

    st.write(f"Total Error Swings: {df['duplicate_swings'].sum()}")

    df["x"] = range(len(df))
    fig_bokeh = plot_tv_ohlc_bokeh(
        df,
        swing=True,
        debugging=True,
        compare=True,
        title="SSC Chart - Bokeh",
    )

    streamlit_bokeh(
        fig_bokeh, use_container_width=True, theme="streamlit", key="my_unique_key"
    )

    st.title("SSC Data Viewer")
    st.dataframe(
        df[
            [
                "open",
                "high",
                "low",
                "close",
                "bar_type",
                "swing_point",
                "swing",
                "duplicate_swings",
            ]
        ]
    )


def compare_main():
    st.set_page_config(page_title="SSC Data Viewer - Compare", layout="wide")
    df = pd.read_csv("data/crudeoil-dataset.csv")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    # df = df[
    #     df["date"] >= (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    # ]
    df.set_index("date", inplace=True)
    # df.reset_index(inplace=True)
    print(df.head())
    df_ssc = SwingPoints2(df)

    df_ssc["x"] = range(len(df_ssc))
    # df_ssc["error"] = df_ssc["swing_point"] != df_ssc["swing_point_penfold"]
    df_ssc["error"] = (
        ~df_ssc["swing_point"].fillna("X").eq(df_ssc["swing_point_penfold"].fillna("X"))
    )

    df_ssc["error_bartype"] = (
        ~df_ssc["bar_type"].fillna("X").eq(df_ssc["bar_type_penfold"].fillna("X"))
    )
    df_ssc["mvf"] = mvf(df_ssc["close"], df_ssc["low"], lookback=20)
    df_ssc["atr"] = atr(df_ssc, lookback=4)

    fig_bokeh = plot_tv_ohlc_bokeh(
        df_ssc,
        title="SSC Chart - Bokeh",
        compare=True,
        debugging=True,
    )
    st.write(f"Errors count {len(df_ssc[df_ssc['error']])}")
    streamlit_bokeh(
        fig_bokeh, use_container_width=True, theme="streamlit", key="my_unique_key"
    )

    st.dataframe(
        df_ssc[
            [
                "open",
                "high",
                "low",
                "close",
                "bar_type",
                "bar_type_penfold",
                # "bar_type_1",
                "swing_point_penfold",
                "swing_point",
                "swing",
                "mvf",
                "atr",
                "swing_penfold",
                "error",
                "error_bartype",
            ]
        ]
    )


if __name__ == "__main__":
    # main()
    compare_main()
