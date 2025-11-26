from app import eod
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def main():

    instruments = pd.read_csv("instruments.csv")
    symbols = instruments["name"].unique().tolist()

    quantity = instruments.set_index("name")["lot_size"].to_dict()
    symbol = st.sidebar.selectbox("Select Symbol", symbols, index=0)
    qty = quantity[symbol]
    to_data = datetime.now().date().strftime("%Y-%m-%d")
    from_data = (datetime.now().date() - timedelta(days=365)).strftime("%Y-%m-%d")
    st.text(f"Fetching EOD data for {symbol} from {from_data} to {to_data}")
    ohlc_data = eod(symbol, from_data, to_data)
    ohlc_data.sort_values("date", inplace=True)

    st.set_page_config(layout="wide")

    trades = gap_model(ohlc_data, qty)

    trades_df = pd.DataFrame(trades)
    trades_df["charges"] = (
        trades_df["quantity"] * trades_df["entry_price"] * 0.0005 + 20
    )
    trades_df["profit_loss"] = trades_df["profit_loss"] - trades_df["charges"]

    st.header(f"Backtest Results of {symbol}")

    total_profit_loss = trades_df["profit_loss"].sum()
    st.subheader(f"Total Profit/Loss: ₹{total_profit_loss:.2f}")
    st.subheader(f"Total Trades: {len(trades_df)}")

    # Monthly P/L
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df["month"] = trades_df["entry_date"].dt.to_period("M")
    monthly_profit_loss = trades_df.groupby("month")["profit_loss"].sum()
    profittrades = len(trades_df[trades_df["profit_loss"] > 0])
    lossstrades = len(trades_df[trades_df["profit_loss"] <= 0])
    st.subheader(f"Profitable Trades: {profittrades}")
    st.subheader(f"Losing Trades: {lossstrades}")
    st.subheader("Monthly Profit/Loss")
    st.bar_chart(monthly_profit_loss)

    trades_df = trades_df[trades_df["exit_date"] != ""].copy()
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    trades_df = trades_df.sort_values("exit_date")

    # Running equity
    trades_df["equity"] = trades_df["profit_loss"].cumsum()

    st.subheader("Equity Curve")
    st.line_chart(trades_df.set_index("exit_date")["equity"])
    st.dataframe(trades_df)

    returns = trades_df["profit_loss"].astype(float)

    # Fit Gaussian distribution
    mu, sigma = norm.fit(returns)

    # Generate X-axis for curve
    x = np.linspace(returns.min(), returns.max(), 200)
    bell_curve = norm.pdf(x, mu, sigma)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram of returns
    ax.hist(returns, bins=20, density=True, alpha=0.6)

    # Bell curve line
    ax.plot(x, bell_curve, linewidth=2)

    ax.set_title(f"Bell Curve of Profit/Loss  (μ={mu:.2f},  σ={sigma:.2f})")
    ax.set_xlabel("Profit / Loss per Trade")
    ax.set_ylabel("Density")

    st.subheader("Bell Curve of Profit/Loss")
    st.pyplot(fig)


def gap_model(ohlc_data, qty):
    trades = []
    open_positions = False

    for i in range(1, len(ohlc_data)):
        row = ohlc_data.iloc[i]
        prev = ohlc_data.iloc[i - 1]

        # --------------------------------------------------
        # 1. If a position is open → check only SL
        # --------------------------------------------------
        if open_positions:
            entry = trades[-1]

            if entry["action"] == "LONG":
                stop_level = prev["low"]

                # SL HIT
                if row["low"] <= stop_level:
                    entry["exit_date"] = row["date"]
                    entry["exit_price"] = stop_level
                    entry["profit_loss"] = (stop_level - entry["entry_price"]) * entry[
                        "quantity"
                    ]
                    open_positions = False
                    continue

            elif entry["action"] == "SHORT":
                stop_level = prev["high"]

                # SL HIT
                if row["high"] >= stop_level:
                    entry["exit_date"] = row["date"]
                    entry["exit_price"] = stop_level
                    entry["profit_loss"] = (entry["entry_price"] - stop_level) * entry[
                        "quantity"
                    ]
                    open_positions = False
                    continue

            # NO SL → stay inside trade, continue to next candle
            continue

        # --------------------------------------------------
        # 2. No open position → look for GAP ENTRY
        # --------------------------------------------------
        if row["open"] > prev["close"]:
            # GAP UP → LONG
            trades.append(
                {
                    "action": "LONG",
                    "entry_date": row["date"],
                    "entry_price": row["open"],
                    "exit_date": "",
                    "exit_price": "",
                    "quantity": qty,
                    "profit_loss": 0,
                }
            )
            open_positions = True

        elif row["open"] < prev["close"]:
            # GAP DOWN → SHORT
            trades.append(
                {
                    "action": "SHORT",
                    "entry_date": row["date"],
                    "entry_price": row["open"],
                    "exit_date": "",
                    "exit_price": "",
                    "quantity": qty,
                    "profit_loss": 0,
                }
            )
            open_positions = True

    return trades


if __name__ == "__main__":
    main()
