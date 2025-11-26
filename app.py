import sqlalchemy
import pandas as pd
import psycopg2
import streamlit as st
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import altair as alt
import plotly.graph_objects as go


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


def eod(
    symbol: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:

    engine = get_sqlalchemy_engine()
    query = f"""
        SELECT datetime AT TIME ZONE 'Asia/Kolkata' AS local_time, *
        FROM tfw_eod
        WHERE symbol = %s AND datetime >= %s AND datetime <= %s
        ORDER BY datetime ASC;
    """
    df = pd.read_sql(query, engine, params=(symbol, from_date, to_date))

    df = df[["local_time", "open", "high", "low", "close", "volume", "oi"]]
    df.rename(columns={"local_time": "date"}, inplace=True)
    return df


def main():
    st.set_page_config(layout="wide")
    portfolio = ["NIFTY", "GOLD", "SILVER", "NATURALGAS", "CRUDEOIL"]
    symbol = st.sidebar.selectbox("Select Symbol", portfolio, index=0)
    no_of_days = st.sidebar.slider("Select number of days for EOD data", 1, 5000, 200)
    to_date = datetime.now().date().strftime("%Y-%m-%d")
    from_date = (
        (datetime.now() - timedelta(days=no_of_days)).date().strftime("%Y-%m-%d")
    )
    st.text(f"Fetching EOD data for {symbol} from {from_date} to {to_date}")
    df = eod(symbol, str(from_date), str(to_date))
    if st.sidebar.button("Show DataFrame"):
        st.write(df)

    chart_type_radio = st.sidebar.radio(
        "Select Chart Type", ["Candlestick", "Line Chart", "Bar Chart"], index=0
    )

    if chart_type_radio == "Candlestick":
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                )
            ]
        )

        fig.update_layout(xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True, config={"height": 600})
    elif chart_type_radio == "Line Chart":
        df2 = df.set_index("date")

        ymin = df2["high"].min()
        ymax = df2["low"].max()

        chart = (
            alt.Chart(df2.reset_index())
            .mark_line()
            .encode(
                x="date:T", y=alt.Y("close:Q", scale=alt.Scale(domain=[ymin, ymax]))
            )
            .properties(width="container", height=600)
        )

        st.altair_chart(chart, use_container_width=True)
    elif chart_type_radio == "Bar Chart":
        # df["date"] = pd.to_datetime(df["date"])

        # ymax = df["high"].max()
        # ymin = df["low"].min()

        # chart = (
        #     alt.Chart(df)
        #     .mark_rule()
        #     .encode(
        #         x="date:T",
        #         y="low:Q",
        #         y2="high:Q",
        #         color=alt.condition(
        #             "datum.open < datum.close", alt.value("green"), alt.value("red")
        #         ),
        #     )
        #     .properties(height=400)
        # )
        # # set x max and min
        # chart = chart.encode(
        #     y=alt.Y("low:Q", scale=alt.Scale(domain=[ymin, ymax])),
        # )
        # st.altair_chart(chart, use_container_width=True)
        df["date"] = pd.to_datetime(df["date"])

        fig = go.Figure(
            data=[
                go.Ohlc(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    increasing_line_color="red",
                    decreasing_line_color="green",
                )
            ]
        )

        fig.update_layout(
            autosize=False,
            height=600,
            xaxis_rangeslider_visible=False,  # no bottom scroll
        )

        st.plotly_chart(fig, use_container_width=True, config={"height": 1000})


if __name__ == "__main__":
    main()
