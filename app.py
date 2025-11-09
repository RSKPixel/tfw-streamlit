import streamlit as st
import psycopg2
import pandas as pd
import sqlalchemy
from urllib.parse import quote_plus


# Time frame mapping
tf = {
    "5min": "idata_5min",
    "15min": "idata_15min",
    "60min": "idata_60min",
    "1day": "idata_1day",
}


# --- Database Connection ---
def get_db_connection():

    try:
        conn = psycopg2.connect(
            host="trialnerror.in",
            database="tradersframework",
            user="sysadmin",
            password="Apple@1239",
        )
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.stop()
    return conn


def get_sqlalchemy_engine():
    try:
        host = "trialnerror.in"
        database = "tradersframework"
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


def ohlcv(symbol: str, start_date: str, end_date: str, time_frame: str) -> pd.DataFrame:

    engine = get_sqlalchemy_engine()
    query = f"""
        SELECT date AT TIME ZONE 'Asia/Kolkata' AS local_time, *
        FROM {time_frame}
        WHERE symbol = %s AND date >= %s AND date <= %s
        ORDER BY date DESC;
    """
    df = pd.read_sql(query, engine, params=(symbol, start_date, end_date))

    df = df[["local_time", "open", "high", "low", "close", "volume"]]
    df.rename(columns={"local_time": "date"}, inplace=True)
    return df


# Fetch available symbols
def fetch_symbols_daterange(timeframe) -> pd.DataFrame:
    st.set_page_config(page_title="TFW Dashboard", layout="centered")
    st.header("Traders Framework (TFW) Dashboard")
    engine = get_sqlalchemy_engine()
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
    # st.title("Traders Framework (TFW)")
    selected_tf = "5min"
    datatype = ["OHLCV Only", "With Technical Data"]
    timeframe = tf[selected_tf]

    symbols, date_range = fetch_symbols_daterange(timeframe)

    selected_symbol = st.selectbox("Select Symbol", symbols["symbol"].tolist())
    cols = st.columns(2)

    with cols[0]:
        selected_tf = st.segmented_control(
            "Available Time Frames:", options=list(tf.keys()), default="5min"
        )

    with cols[1]:
        selected_datatype = st.segmented_control(
            "Select Data Type:", options=datatype, default="OHLCV Only"
        )
    timeframe = tf[selected_tf]

    table_data = {
        "Min Date": [date_range["min_date"][0]],
        "Max Date": [date_range["max_date"][0]],
    }
    st.table(table_data)

    if selected_datatype == "With Technical Data":
        st.info("Technical Data feature is coming soon!")
        st.stop()

    if selected_datatype == "OHLCV Only":
        ohlcv_data = ohlcv(
            selected_symbol,
            date_range["min_date"][0],
            date_range["max_date"][0],
            timeframe,
        )

    st.dataframe(ohlcv_data, hide_index=True, height=300)


streamlit_ui()
