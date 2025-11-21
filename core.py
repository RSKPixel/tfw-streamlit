import sqlalchemy
from urllib.parse import quote_plus
import pandas as pd
import numpy as np
import talib as ta

tf = {
    "3min": "idata_3min",
    "5min": "idata_5min",
    "15min": "idata_15min",
    "60min": "idata_60min",
    "1day": "idata_1day",
}


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
        print(f"Error connecting to database: {e}")

    return engine


def fetch_symbols() -> pd.DataFrame:
    engine = get_sqlalchemy_engine()
    symbols = pd.DataFrame()
    try:
        query = sqlalchemy.text(
            "SELECT DISTINCT symbol FROM idata_1day ORDER BY symbol;"
        )
        symbols = pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error fetching symbols: {e}")
    return symbols


def fetch_ohlcv(
    symbol: str, start_date: str, end_date: str, timeframe: str
) -> pd.DataFrame:

    engine = get_sqlalchemy_engine()
    timeframe = tf[timeframe]
    df = pd.DataFrame()
    status = "success"

    try:
        query = sqlalchemy.text(
            f"""
            SELECT date AT TIME ZONE 'Asia/Kolkata' AS local_time, *
            FROM {timeframe}
            WHERE symbol = :symbol AND date >= :start_date AND date <= :end_date
            ORDER BY date ASC;
        """
        )
        df = pd.read_sql(
            query,
            engine,
            params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
        )

        df = df[["local_time", "open", "high", "low", "close", "volume"]]
        df.rename(columns={"local_time": "date"}, inplace=True)
    except Exception as e:
        status = f"Error fetching data: {e}"
        print(status)
    return df


def trend_identification(
    df: pd.DataFrame,
    short_window: int = 13,
    medium_window: int = 50,
    long_window: int = 200,
    momentum_window: int = 3,
) -> pd.DataFrame:

    data = df.copy()
    data["short_ema"] = ta.EMA(data["close"], timeperiod=short_window)
    data["medium_ema"] = ta.EMA(data["close"], timeperiod=medium_window)
    data["long_ema"] = ta.EMA(data["close"], timeperiod=long_window)
    data["momentum"] = ta.RSI(data["close"], timeperiod=momentum_window)
    data["bull"] = data["close"] > data["open"]
    data["bear"] = data["close"] < data["open"]

    uptrend_condition = (
        (data["short_ema"] > data["medium_ema"])
        & (data["medium_ema"] > data["long_ema"])
        & (data["momentum"] > 80)
        & (data["bull"])
        & data["bull"].shift(1)
        & (data["close"] > data["short_ema"])
        & (data["close"] > data["high"].shift(1))
    )

    downtrend_condition = (
        (data["short_ema"] < data["medium_ema"])
        & (data["medium_ema"] < data["long_ema"])
        & (data["momentum"] < 20)
        & (data["bear"])
        & data["bear"].shift(1)
        & (data["close"] < data["short_ema"])
        & (data["close"] < data["low"].shift(1))
    )

    data["signal"] = np.select(
        [uptrend_condition, downtrend_condition],
        ["buy signal", "sell signal"],
        default="none",
    )

    df["short_ema"] = data["short_ema"]
    df["medium_ema"] = data["medium_ema"]
    df["long_ema"] = data["long_ema"]
    df["momentum"] = data["momentum"]
    df["signal"] = data["signal"]

    return df
