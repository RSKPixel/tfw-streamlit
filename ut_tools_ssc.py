import pandas as pd
import numpy as np

import pandas as pd


def BarDefinition_Penfold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified and accurate Python version of Brent Penfold's VBA bar classification.
    Requires df columns: ['High', 'Low'].
    Adds a 'bar_type' column containing: DB, OSB, ISB.
    """

    df = df.copy()
    n = len(df)
    df["bar_type"] = None

    i = 1  # equivalent to VBA row 2
    df.iloc[0, df.columns.get_loc("bar_type")] = "DB"  # first bar is always DB

    while i < n:

        prev_h = df.high.iloc[i - 1]
        prev_l = df.low.iloc[i - 1]

        h = df.high.iloc[i]
        l = df.low.iloc[i]
        bar_type = df.columns.get_loc("bar_type")

        # -------------------------------------
        # 1. Directional UP (DB)
        # -------------------------------------
        if h > prev_h and l >= prev_l:
            df.iloc[i, bar_type] = "DB"

        # -------------------------------------
        # 2. Directional DOWN (DB)
        # -------------------------------------
        elif h <= prev_h and l < prev_l:
            df.iloc[i, bar_type] = "DB"

        # -------------------------------------
        # 3. Outside Bar (OSB)
        # -------------------------------------
        elif h > prev_h and l < prev_l:
            df.iloc[i, bar_type] = "OSB"

        # -------------------------------------
        # 4. Inside Bar (ISB) + Multi-ISB Loop
        # -------------------------------------
        elif h <= prev_h and l >= prev_l:

            df.iloc[i, bar_type] = "ISB"

            # freeze previous bar range
            range_high = prev_h
            range_low = prev_l

            j = i
            # extend ISB chain
            while j < n:
                hh = df.high.iloc[j]
                ll = df.low.iloc[j]

                # breakout?
                if hh > range_high or ll < range_low:
                    break

                df.iloc[j, bar_type] = "ISB"
                j += 1

            # move i to last inside bar (same logic as VBA's i = j - 1)
            i = j - 1

        else:
            # Should never occur if data is valid OHLC
            df.iloc[i, bar_type] = "DB"

        i += 1
    return df


def BarType(previous_bar, current_bar) -> str:
    prev_h, prev_l = previous_bar["high"], previous_bar["low"]
    h, l = current_bar["high"], current_bar["low"]

    # Outside Bar (first!)
    if h >= prev_h and l <= prev_l:
        return "OSB"

    # Inside Bar
    if h <= prev_h and l >= prev_l:
        return "ISB"

    # Directional Bar UP
    if h > prev_h and l > prev_l:
        return "DB"

    # Directional Bar DOWN
    if h < prev_h and l < prev_l:
        return "DB"

    # Safety fallback
    return "DB"


def BarDefination(df: pd.DataFrame) -> pd.DataFrame:
    # Bar type assignment
    # DB - Directional Bar, ISB - Inside Bar, OSB - Outside Bar

    df = df.copy()
    df["bar_type"] = ""
    df.at[df.index[0], "bar_type"] = "DB"
    for i in range(1, len(df)):
        previous_bar = df.iloc[i - 1]
        current_bar = df.iloc[i]
        df.at[df.index[i], "bar_type"] = BarType(previous_bar, current_bar)

    # Refine DB and OSB bars based on the last DB bar
    # Handleling multiple ISB's in a row

    previous_db_index = 0
    for i in range(1, len(df)):
        previous_db_bar = df.iloc[previous_db_index]

        if (
            df.iloc[i]["bar_type"] == "DB" or df.iloc[i]["bar_type"] == "OSB"
        ) and df.iloc[i - 1]["bar_type"] == "ISB":
            df.at[df.index[i], "bar_type"] = BarType(previous_db_bar, df.iloc[i])
        elif df.iloc[i]["bar_type"] == "DB" or df.iloc[i]["bar_type"] == "OSB":
            previous_db_index = i

    return df


def SwingPoints2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["swing_point"] = np.nan
    df["swing"] = ""
    previous_db_index = 0
    swing_index = 0
    swing_point = df.iloc[0]["high"]
    swing = "low"

    # Identifying the first bar direction and mark
    if df.iloc[1]["bar_type"] == "DB":
        if df.iloc[1]["high"] > df.iloc[0]["high"]:
            swing = "low"
            swing_index = 0
            swing_point = df.iloc[0]["low"]
            df.at[df.index[swing_index], "swing"] = swing
            df.at[df.index[swing_index], "swing_point"] = swing_point
        else:
            swing = "high"
            swing_index = 0
            swing_point = df.iloc[0]["high"]
            df.at[df.index[swing_index], "swing"] = swing
            df.at[df.index[swing_index], "swing_point"] = swing_point

    for i in range(1, len(df) - 1):
        previous_h, previous_l = (
            df.iloc[previous_db_index]["high"],
            df.iloc[previous_db_index]["low"],
        )
        current_h, current_l, current_bar = (
            df.iloc[i]["high"],
            df.iloc[i]["low"],
            df.iloc[i]["bar_type"],
        )

        if current_bar == "ISB":
            continue

        if current_bar == "OSB":
            # if swing_index != previous_db_index:
            #     # swing_index = previous_db_index

            if swing == "low":
                if current_l <= swing_point:

                    swing_point = current_l
                    df.at[df.index[swing_index], "swing_point"] = np.nan
                    df.at[df.index[swing_index], "swing"] = swing
                    df.at[df.index[swing_index], "swing_point"] = swing_point
                    swing_index = i
                else:
                    previous_db_index = i
                    continue

            if swing == "high":
                if current_h >= swing_point:

                    swing_point = current_h

                    df.at[df.index[swing_index], "swing_point"] = np.nan
                    df.at[df.index[swing_index], "swing"] = swing
                    df.at[df.index[swing_index], "swing_point"] = swing_point
                    swing_index = i
                else:
                    previous_db_index = i
                    continue

        if current_bar == "DB":
            if current_h >= previous_h:
                if swing == "low":
                    previous_db_index = i
                    continue

                swing = "low"
                swing_index = previous_db_index
                swing_point = previous_l

                df.at[df.index[swing_index], "swing"] = swing
                df.at[df.index[swing_index], "swing_point"] = swing_point
                previous_db_index = i
                if i == 28:
                    print(swing, swing_index)
            if current_l <= previous_l:
                if swing == "high":
                    previous_db_index = i
                    continue

                swing = "high"
                swing_index = previous_db_index
                swing_point = previous_h

                df.at[df.index[swing_index], "swing"] = swing
                df.at[df.index[swing_index], "swing_point"] = swing_point
                previous_db_index = i

    return df


def SSC(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = BarDefination(df)
    df = SwingPoints2(df)

    return df
