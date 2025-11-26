import pandas as pd
import numpy as np


def barType(previous_bar, current_bar) -> str:
    previous_h, previous_l = previous_bar["high"], previous_bar["low"]
    current_h, current_l = current_bar["high"], current_bar["low"]

    if current_h < previous_h and current_l > previous_l:
        return "ISB"

    if current_h > previous_h and current_l < previous_l:
        return "OSB"

    return "DB"


def barDefinition(df: pd.DataFrame) -> pd.DataFrame:
    # Bar type assignment
    # DB - Directional Bar, ISB - Inside Bar, OSB - Outside Bar

    df = df.copy()
    df["bar_type"] = ""
    df.at[df.index[0], "bar_type"] = "DB"
    for i in range(1, len(df)):
        previous_bar = df.iloc[i - 1]
        current_bar = df.iloc[i]
        df.at[df.index[i], "bar_type"] = barType(previous_bar, current_bar)

    # Refine DB and OSB bars based on the last DB bar
    # Handleling multiple ISB's in a row

    previous_db_index = 0
    for i in range(1, len(df)):
        previous_db_bar = df.iloc[previous_db_index]

        if (
            df.iloc[i]["bar_type"] == "DB" or df.iloc[i]["bar_type"] == "OSB"
        ) and df.iloc[i - 1]["bar_type"] == "ISB":
            df.at[df.index[i], "bar_type"] = barType(previous_db_bar, df.iloc[i])
        elif df.iloc[i]["bar_type"] == "DB" or df.iloc[i]["bar_type"] == "OSB":
            previous_db_index = i

    return df


def SwingPoints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["swing_point"] = np.nan
    current_swing = ""
    current_swing_index = 0
    previous_db_index = 0

    for i in range(1, len(df) - 1):
        previous_h, previous_l = df.iloc[i - 1]["high"], df.iloc[i - 1]["low"]
        current_h, current_l, current_bar = (
            df.iloc[i]["high"],
            df.iloc[i]["low"],
            df.iloc[i]["bar_type"],
        )

        if current_bar == "ISB":
            continue

        if current_bar == "OSB":
            if current_swing == "low":
                if current_l < df.iloc[current_swing_index]["low"]:
                    df.at[df.index[current_swing_index], "swing_point"] = np.nan
                    current_swing_index = i
                    df.at[df.index[current_swing_index], "swing_point"] = current_l
            if current_swing == "high":
                if current_h > df.iloc[current_swing_index]["high"]:
                    df.at[df.index[current_swing_index], "swing_point"] = np.nan
                    current_swing_index = i
                    df.at[df.index[current_swing_index], "swing_point"] = current_h

            continue

        if current_bar == "DB":
            previous_db_index = i
            if current_h >= previous_h:
                if current_swing == "low":
                    continue

                df.at[df.index[previous_db_index - 1], "swing_point"] = previous_l
                current_swing = "low"
                current_swing_index = i - 1
                previous_db_index = i - 1
            if current_l <= previous_l:
                if current_swing == "high":
                    continue
                df.at[df.index[previous_db_index - 1], "swing_point"] = previous_h
                current_swing = "high"
                current_swing_index = i - 1
                previous_db_index = i - 1
                pass
        pass

    return df


def SSC(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = barDefinition(df)
    df = SwingPoints(df)

    return df
