import pandas as pd
import numpy as np
from nerotrader import rw_extremes
import streamlit as st
from app import eod
from datetime import datetime, timedelta
from barchart import plot_tv_ohlc_dark_v2


def ssc(data: pd.DataFrame):

    df = data.copy()
    df["swing"] = None
    df["previous_db"] = None
    df.iloc[0, df.columns.get_loc("bartype")] = "DB"
    prev_swing_point = 0
    previous_db = 0
    swing = ""

    for i in range(1, len(df)):
        previous_db = i - 1
        previous_high = df.iloc[previous_db]["high"]
        previous_low = df.iloc[previous_db]["low"]
        high = df.iloc[i]["high"]
        low = df.iloc[i]["low"]
        current_barttpe = df.iloc[i]["bartype"]
        previous_bartype = df.iloc[i - 1]["bartype"]

        if current_barttpe == "DB":

            df.at[df.index[i], "previous_db"] = df.index[previous_db]

            if high >= previous_high:
                if swing == "low":
                    continue
                df.at[df.index[previous_db], "swing"] = "low"
                swing = "low"
                swing_value = previous_low

            if low <= previous_low:
                if swing == "high":
                    continue
                df.at[df.index[previous_db], "swing"] = "high"
                swing = "high"
                swing_value = previous_high

        if current_barttpe == "ISB":
            continue

        if current_barttpe == "OSB":
            pass

    # Tops (swing highs)
    tops = [(row["x"], row["high"]) for _, row in df[df["swing"] == "high"].iterrows()]

    # Bottoms (swing lows)
    bottoms = [(row["x"], row["low"]) for _, row in df[df["swing"] == "low"].iterrows()]

    return df, tops, bottoms


def bartype(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["bartype"] = None

    # First bar is always DB
    df.iloc[0, df.columns.get_loc("bartype")] = "DB"

    for i in range(1, len(df)):
        prev_high = df.iloc[i - 1]["high"]
        prev_low = df.iloc[i - 1]["low"]
        high = df.iloc[i]["high"]
        low = df.iloc[i]["low"]
        previous_range = prev_high - prev_low
        current_range = high - low

        # ISB: inside previous bar
        if high <= prev_high and low >= prev_low:
            df.at[df.index[i], "bartype"] = "ISB"

        # OSB: outside previous bar
        elif high >= prev_high and low <= prev_low:
            df.at[df.index[i], "bartype"] = "OSB"

        # All other cases are DB (normal direction bar)
        else:
            df.at[df.index[i], "bartype"] = "DB"
    return df


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    bars = st.slider("Nerotrader RW Extremes and SSC Extremes", 10, 50, 200)
    bpdata_original = pd.read_csv("data/bp-data.csv")
    bpdata = bpdata_original[-bars:]
    bpdata["date"] = bpdata["date"].astype("datetime64[s]")
    bpdata = bpdata.set_index("date")
    bpdata["x"] = range(len(bpdata))
    print(bpdata.head())

    tops = [
        (row["x"], row["highs"]) for _, row in bpdata[bpdata["highs"] != ""].iterrows()
    ]
    bottoms = [
        (row["x"], row["lows"]) for _, row in bpdata[bpdata["lows"] != ""].iterrows()
    ]
    print("Tops:", tops)
    print("Bottoms:", bottoms)
    fig = plot_tv_ohlc_dark_v2(
        bpdata, tops, bottoms, title="BP PLC with Nerotrader RW Extremes"
    )

    # add line call to st.pyplot
    sh = bpdata["sh"]
    sl = bpdata["sl"]
    bpdata["sh"] = pd.to_numeric(bpdata["sh"], errors="coerce")
    bpdata["sl"] = pd.to_numeric(bpdata["sl"], errors="coerce")

    bpdata["sh_ff"] = bpdata["sh"].ffill()
    bpdata["sl_ff"] = bpdata["sl"].ffill()

    st.pyplot(fig)

    to_date = datetime.now().date().strftime("%Y-%m-%d")
    no_of_days = 200
    symbol = "BANKNIFTY"
    from_date = (
        (datetime.now() - timedelta(days=no_of_days)).date().strftime("%Y-%m-%d")
    )
    st.text(f"Fetching EOD data for {symbol} from {from_date} to {to_date}")
    df = eod(symbol, str(from_date), str(to_date))

    # neotrader RW Extremes
    neotrader_data = df.copy()
    neotrader_data["date"] = neotrader_data["date"].astype("datetime64[s]")
    neotrader_data = neotrader_data.set_index("date")
    neotrader_tops, neotrader_bottoms = rw_extremes(
        neotrader_data["high"].to_numpy(), neotrader_data["low"].to_numpy(), 5
    )
    neotrader_data["x"] = range(len(neotrader_data))

    neotrader_tops = [(top[1], top[2]) for top in neotrader_tops]
    neotrader_bottoms = [(bottom[1], bottom[2]) for bottom in neotrader_bottoms]

    # SSC Extremes
    ssc_data = df.copy()
    ssc_data["date"] = ssc_data["date"].astype("datetime64[s]")
    ssc_data["x"] = range(len(ssc_data))
    ssc_data = ssc_data.set_index("date")
    ssc_data = bartype(ssc_data)
    ssc_data, ssc_tops, ssc_bottoms = ssc(ssc_data)

    fig1 = plot_tv_ohlc_dark_v2(
        ssc_data, ssc_tops, ssc_bottoms, title=f"SSC - {symbol}"
    )
    fig2 = plot_tv_ohlc_dark_v2(
        neotrader_data, neotrader_tops, neotrader_bottoms, title="neotrader RW Extremes"
    )

    # plot_ssc_lightweight(ssc_data, ssc_tops, ssc_bottoms)

    st.pyplot(fig1)
    st.pyplot(fig2)
