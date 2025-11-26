from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
from app import eod
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import ColumnDataSource
import pandas as pd


from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd


class LongShortSMACross(Strategy):

    def init(self):
        price = pd.Series(self.data.Close)

        # Proper SMAs
        self.sma50 = self.I(lambda x: x.rolling(50).mean(), price)
        self.sma200 = self.I(lambda x: x.rolling(200).mean(), price)

        # Setup bar storage
        self.setup_low = None
        self.setup_high = None
        self.entry_index = None

    def next(self):

        # -----------------------------------------
        # If a trade is open — manage exit and SL
        # -----------------------------------------
        if self.position:

            bars_since_entry = len(self.data.Close) - self.entry_index

            # Exit on 5th bar after entry
            if bars_since_entry >= 5:
                self.position.close()
                return

            # Maintain fixed SL for long positions
            if self.position.is_long:
                self.position.sl = self.setup_low

            # Maintain fixed SL for short positions
            if self.position.is_short:
                self.position.sl = self.setup_high

            return  # <--- VERY IMPORTANT: no new trades until current closes

        # -----------------------------------------
        # No open trade — look for new setups
        # -----------------------------------------

        # -------- LONG SETUP --------
        if self.data.Close[-1] > self.sma200[-1]:
            if crossover(self.data.Close, self.sma50):  # cross ABOVE 50MA

                # Setup bar details
                self.setup_low = self.data.Low[-1]

                # Enter long next bar
                self.buy(sl=self.setup_low)
                self.entry_index = len(self.data.Close)
                return

        # -------- SHORT SETUP --------
        if self.data.Close[-1] < self.sma200[-1]:
            if crossover(self.sma50, self.data.Close):  # cross BELOW 50MA

                # Setup bar details
                self.setup_high = self.data.High[-1]

                # Enter short next bar
                self.sell(sl=self.setup_high)
                self.entry_index = len(self.data.Close)
                return


if __name__ == "__main__":
    df = eod("NIFTY", "2023-01-01", "2025-11-30")
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

    bt = Backtest(
        df,
        LongShortSMACross,
        cash=100000,
        commission=0.0005,
        exclusive_orders=True,  # ensures only 1 open trade
    )

    stats = bt.run()
    bt.plot()
    print(stats)
