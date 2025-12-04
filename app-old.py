import streamlit as st
import psycopg2
import pandas as pd
import sqlalchemy
from urllib.parse import quote_plus
import altair as alt

# Time frame mapping
tf = {
    "5min": "idata_5min",
    "15min": "idata_15min",
    "60min": "idata_60min",
    "1day": "idata_1day",
}


# --- Database Connection ---
def get_sqlalchemy_engine():
    try:
        host = "trialnerror.in"
        database = "tfw"
        user = "sysadmin"
        password = quote_plus("Apple@1239")

        conn_string = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{database}"
        engine = sqlalchemy.create_engine(conn_string)

        with engine.connect() as connection:
            pass  # Test connection

    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.stop()
    return engine


engine = get_sqlalchemy_engine()


def ohlcv(
    symbol: str, start_date: str, end_date: str, selected_tf: str
) -> pd.DataFrame:

    timeframe = tf[selected_tf]
    query = f"""
        SELECT datetime AT TIME ZONE 'Asia/Kolkata' AS local_time, *
        FROM {timeframe}
        WHERE symbol = %s AND datetime >= %s AND datetime <= %s
        ORDER BY datetime DESC;
    """
    df = pd.read_sql(query, engine, params=(symbol, start_date, end_date))

    df = df[["local_time", "open", "high", "low", "close", "volume"]]
    df.rename(columns={"local_time": "date"}, inplace=True)
    return df


# Fetch available symbols
def fetch_symbols_daterange(selected_tf) -> pd.DataFrame:
    timeframe = tf[selected_tf]
    query = f"SELECT symbol FROM {timeframe} GROUP BY symbol ORDER BY symbol;"

    symbols = pd.read_sql(query, engine)

    query = f"""
        SELECT
            MIN(date) AT TIME ZONE 'Asia/Kolkata' as min_date,
            MAX(date) AT TIME ZONE 'Asia/Kolkata' as max_date
        FROM {timeframe};
    """

    date_range = pd.read_sql(query, engine)

    date_range["max_date"] = pd.to_datetime(date_range["max_date"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    date_range["min_date"] = pd.to_datetime(date_range["min_date"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    return symbols, date_range


# --- Streamlit UI ---
def streamlit_ui():
    selected_tf = "5min"
    st.set_page_config(page_title="TFW Dashboard", layout="wide")
    st.header("Dashboard")

    datatype = ["OHLC"]

    primary_columns = st.columns([1, 3])

    with primary_columns[0]:
        with st.container(border=True):
            selected_tf = st.segmented_control(
                "Time Frames:", options=list(tf.keys()), default="5min"
            )

            selected_datatype = st.segmented_control(
                "Data:", options=datatype, default="OHLC"
            )

            symbols, date_range = fetch_symbols_daterange(selected_tf)
            selected_symbol = st.selectbox("Symbols", symbols["symbol"].tolist())

            table_data = {
                "Min Date": [date_range["min_date"][0]],
                "Max Date": [date_range["max_date"][0]],
            }
            st.table(table_data)

    with primary_columns[1]:
        with st.container(border=True):
            st.subheader(f"{selected_symbol} - {selected_tf} Data")
            if selected_datatype == "With Technical Data":
                st.info("Technical Data feature is coming soon!")
                st.stop()

            if selected_datatype == "OHLC":
                ohlcv_data = ohlcv(
                    selected_symbol,
                    date_range["min_date"][0],
                    date_range["max_date"][0],
                    selected_tf,
                )

            st.dataframe(ohlcv_data, hide_index=True)
            st.button("Show Chart", on_click=chart_dialog, args=(ohlcv_data,))


@st.dialog("Chart", width="large")
def chart_dialog(data=None):
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("index:T", title="Date"),
            y=alt.Y(
                "close:Q",
                title="Closing Price",
                scale=alt.Scale(domain=[data["close"].min(), data["close"].max()]),
            ),
        )
        .properties(title="Closing Prices Over Time")
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


streamlit_ui()
