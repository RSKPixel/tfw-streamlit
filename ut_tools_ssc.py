import pandas as pd


def barType(previous_bar, current_bar) -> str:
    if (
        current_bar["high"] > previous_bar["high"]
        and current_bar["low"] >= previous_bar["low"]
    ):
        return "DB"

    if (
        current_bar["high"] <= previous_bar["high"]
        and current_bar["low"] < previous_bar["low"]
    ):
        return "DB"

    if (
        current_bar["high"] < previous_bar["high"]
        and current_bar["low"] > previous_bar["low"]
    ):
        return "ISB"

    if (
        current_bar["high"] > previous_bar["high"]
        and current_bar["low"] < previous_bar["low"]
    ):
        return "OSB"


def barDefinition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bar_type"] = ""
    df.at[df.index[0], "bar_type"] = "DB"
    for i in range(1, len(df)):
        previous_bar = df.iloc[i - 1]
        current_bar = df.iloc[i]
        df.at[df.index[i], "bar_type"] = barType(previous_bar, current_bar)

    df = multipleISBs(df)
    return df


def multipleISBs(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    previous_db_index = 0
    for i in range(1, len(df)):
        previous_db_bar = df.iloc[previous_db_index]

        if df.iloc[i]["bar_type"] == "DB" and df.iloc[i - 1]["bar_type"] == "ISB":
            df.at[df.index[i], "bar_type"] = barType(previous_db_bar, df.iloc[i])
        elif df.iloc[i]["bar_type"] == "DB" or df.iloc[i]["bar_type"] == "OSB":
            previous_db_index = i

    return df


def SSC(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SSC"] = 0.0

    df = barDefinition(df)

    return df
