from roof import roofing_filter
import pandas as pd
from app import eod
from backtesting import Backtest, Strategy


class RoofStrategy(Strategy):
    def init(self):
        # Pass your precomputed Roof column into backtesting as indicator
        self.roof = self.I(lambda *args: self.data.df["Roof"].values)

    def next(self):
        if len(self.roof) < 3:
            return

        # swing low = buy
        if (
            self.roof[-2] < self.roof[-1]
            and self.roof[-2] < self.roof[-3]
            and not self.position
        ):
            self.buy()

        # swing high = sell
        if (
            self.roof[-2] > self.roof[-1]
            and self.roof[-2] > self.roof[-3]
            and self.position
        ):
            self.sell()


def main():
    symbol = "NIFTY"

    df = eod(symbol, "2020-07-19", "2023-07-19")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # --- Roofing Filter + Signals ---
    df = roofing_filter(df, hp_period=48, ss_period=10, price_col="close")
    df["buy"] = (df["Roof"].shift(1) < df["Roof"]) & (df["Roof"] < df["Roof"].shift(-1))
    df["sell"] = (df["Roof"].shift(1) > df["Roof"]) & (
        df["Roof"] > df["Roof"].shift(-1)
    )

    df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"},
        inplace=True,
    )

    bt = Backtest(df, RoofStrategy, cash=10000, commission=0.0005)
    stats = bt.run()
    print(stats)
    bt.plot()


if __name__ == "__main__":
    main()
