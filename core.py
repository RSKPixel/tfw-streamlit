import sqlalchemy
from urllib.parse import quote_plus
import pandas as pd
from turtle import st

tf = {
    "5min": "idata_5min",
    "15min": "idata_15min",
    "60min": "idata_60min",
    "1day": "idata_1day",
}


def get_sqlalchemy_engine():
    error = None
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
        error = f"Error connecting to database: {e}"
        engine = None

    return engine, error


def ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str,
    engine: sqlalchemy.engine.Engine,
) -> pd.DataFrame:

    timeframe = tf[timeframe]
    query = f"""
        SELECT date AT TIME ZONE 'Asia/Kolkata' AS local_time, *
        FROM {timeframe}
        WHERE symbol = %s AND date >= %s AND date <= %s
        ORDER BY date DESC;
    """
    df = pd.read_sql(query, engine, params=(symbol, start_date, end_date))

    df = df[["local_time", "open", "high", "low", "close", "volume"]]
    df.rename(columns={"local_time": "date"}, inplace=True)
    return df
