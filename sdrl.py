# sdrl_sell_strategy.py
from backtesting import Strategy, Backtest
from backtesting.test import GOOG
import pandas as pd
import numpy as np
from app import eod
import streamlit as st


class DelayedDonchianBreakout(Strategy):
    def init(self):
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        dc_high_20 = high.rolling(20).max()
        dc_low_20 = low.rolling(20).min()
        self.dc_delayed_high = self.I(lambda x: dc_high_20.shift(20), high)
        self.dc_delayed_low = self.I(lambda x: dc_low_20.shift(20), low)
        self.ma200 = self.I(lambda c: pd.Series(c).rolling(200).mean(), self.data.Close)
        self.ma50 = self.I(lambda c: pd.Series(c).rolling(50).mean(), self.data.Close)

    def next(self):
        close = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        dch = self.dc_delayed_high[-1]
        dch_prev = self.dc_delayed_high[-2]
        dcl = self.dc_delayed_low[-1]
        dcl_prev = self.dc_delayed_low[-2]
        # ma = self.ma200[-1]
        ma = self.ma50[-1]

        # If position open: compute trailing SL and check for SL breach intrabar
        if self.position:
            if self.position.is_long:
                trailing_sl = min(
                    self.data.Low[-1], self.data.Low[-2], self.data.Low[-3]
                )
                # If current bar already touched/breached trailing SL
                if self.data.Low[-1] <= trailing_sl:
                    # This will trigger an immediate intrabar stop if SL was set on entry,
                    # otherwise it will close at the next allowed market price.
                    self.position.close()
            else:  # short
                trailing_sl = max(
                    self.data.High[-1], self.data.High[-2], self.data.High[-3]
                )
                if self.data.High[-1] >= trailing_sl:
                    self.position.close()
            return  # don't enter while in a trade

        # Entry (exact your logic)
        long_signal = (close > dch) and (prev_close < dch_prev) and (close > ma)
        short_signal = (close < dcl) and (prev_close > dcl_prev) and (close < ma)

        if long_signal:
            # initial SL at entry = last 3-bar low (use bars relative to entry decision)
            init_sl = min(self.data.Low[-1], self.data.Low[-2], self.data.Low[-3])
            self.buy(sl=init_sl)
        elif short_signal:
            init_sl = max(self.data.High[-1], self.data.High[-2], self.data.High[-3])
            self.sell(sl=init_sl)


# Example main runner
if __name__ == "__main__":
    symbols = ["BANKNIFTY", "NIFTY", "CRUDEOIL", "GOLD", "SILVER", "SBIN"]
    selected_symbol = st.selectbox(
        "Select Symbol for SDRL Sell Strategy Backtest", symbols
    )
    ohlc = eod(selected_symbol, "2020-01-01", "2025-11-30")
    df = ohlc.copy()
    df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    df["Date"] = pd.to_datetime(df["Date"])

    df.set_index("Date", inplace=True)

    st.write(df)

    bt = Backtest(df, DelayedDonchianBreakout, cash=100000, commission=0.000375)
    output = bt.run()
    bt.plot(plot_pl=True, plot_trades=True)
    print(output)
    st.write(output)
