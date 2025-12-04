import pandas as pd
import numpy as np


def BarIdentification(dataframe: pd.DataFrame, key: str = "bar_type") -> pd.DataFrame:
    df = dataframe.copy()

    df["bar_type"] = ""
    anchor_bar_index = 0
    df.at[df.index[0], "bar_type"] = "DB"

    for i in range(1, len(df)):
        anchor_h, anchor_l = df.iloc[anchor_bar_index][["high", "low"]]
        current_h, current_l = df.iloc[i][["high", "low"]]

        if current_h <= anchor_h and current_l >= anchor_l:
            df.at[df.index[i], "bar_type"] = "ISB"

        elif current_h > anchor_h and current_l >= anchor_l:
            df.at[df.index[i], "bar_type"] = "DB"
            anchor_bar_index = i

        elif current_l < anchor_l and current_h <= anchor_h:
            df.at[df.index[i], "bar_type"] = "DB"
            anchor_bar_index = i

        elif current_h > anchor_h and current_l < anchor_l:
            df.at[df.index[i], "bar_type"] = "OSB"
            anchor_bar_index = i

    return df


def SwingPoints2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = BarIdentification(df, key="bar_type")
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

    # Identifying the last bar direction and mark
    if df.iloc[-1]["bar_type"] == "DB":
        if df.iloc[-1]["high"] > df.iloc[-2]["high"]:
            swing = "low"
            swing_index = len(df) - 1
            swing_point = df.iloc[-1]["high"]
            df.at[df.index[swing_index], "swing"] = swing
            df.at[df.index[swing_index], "swing_point"] = swing_point
        else:
            swing = "high"
            swing_index = len(df) - 1
            swing_point = df.iloc[-1]["low"]
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

            if swing == "low":

                if swing_index != previous_db_index:
                    # if previous_db_index != swing_index then
                    # current bar is the swing high
                    # and we need to check for new a swing low which in between previous_db_index and swing_index

                    if current_l < swing_point and current_h > previous_h:

                        if df.iloc[previous_db_index]["high"] > swing_point:
                            resolving_hh = df.iloc[swing_index : previous_db_index + 1][
                                "high"
                            ].max()
                            swing_point = resolving_hh
                            swing = "high"
                            swing_index = df.index.get_loc(
                                df[df["high"] == resolving_hh].index[0]
                            )
                        else:
                            resolving_ll = df.iloc[swing_index : previous_db_index + 1][
                                "low"
                            ].min()

                            swing_point = resolving_ll
                            swing = "low"
                            swing_index = df.index.get_loc(
                                df[df["low"] == resolving_ll].index[0]
                            )

                        df.at[df.index[swing_index], "swing_point"] = swing_point
                        df.at[df.index[swing_index], "swing"] = swing

                        swing_point = current_h
                        swing = "high"
                        swing_index = i
                        df.at[df.index[swing_index], "swing"] = swing
                        df.at[df.index[swing_index], "swing_point"] = swing_point
                        previous_db_index = swing_index

                        continue

                if current_l <= swing_point:

                    swing_point = current_l
                    df.at[df.index[swing_index], "swing_point"] = np.nan
                    df.at[df.index[swing_index], "swing"] = swing
                    df.at[df.index[swing_index], "swing_point"] = swing_point
                else:
                    previous_db_index = i
                    continue

            if swing == "high":
                if i == 786:
                    print(i, swing, swing_point, swing_index, previous_db_index)

                if swing_index != previous_db_index:
                    #     # if previous_db_index != swing_index then
                    #     # current bar is the swing low
                    #     # and we need to check for new a swing high which in between previous_db_index and swing_index
                    if current_h > swing_point and current_l < previous_l:
                        if df.iloc[previous_db_index]["low"] < swing_point:
                            resolving_ll = df.iloc[swing_index : previous_db_index + 1][
                                "low"
                            ].min()
                            swing_point = resolving_ll
                            swing = "low"
                            swing_index = df.index.get_loc(
                                df[df["low"] == resolving_ll].index[0]
                            )
                        else:
                            resolving_hh = df.iloc[swing_index : previous_db_index + 1][
                                "high"
                            ].max()
                            swing_point = resolving_hh
                            swing = "high"
                            swing_index = df.index.get_loc(
                                df[df["high"] == resolving_hh].index[0]
                            )

                        df.at[df.index[swing_index], "swing_point"] = swing_point
                        df.at[df.index[swing_index], "swing"] = swing

                        swing_point = current_l
                        swing = "low"
                        swing_index = i
                        df.at[df.index[swing_index], "swing"] = swing
                        df.at[df.index[swing_index], "swing_point"] = swing_point
                        previous_db_index = swing_index
                        continue

                if current_h >= swing_point:

                    swing_point = current_h

                    df.at[df.index[swing_index], "swing_point"] = np.nan
                    df.at[df.index[swing_index], "swing"] = swing
                    df.at[df.index[swing_index], "swing_point"] = swing_point
                else:

                    previous_db_index = i
                    continue

        if current_bar == "DB":
            if current_h > previous_h:
                if swing == "low":
                    previous_db_index = i
                    continue

                swing = "low"
                swing_index = previous_db_index
                swing_point = previous_l

                df.at[df.index[swing_index], "swing"] = swing
                df.at[df.index[swing_index], "swing_point"] = swing_point
                previous_db_index = i
            if current_l < previous_l:
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
    df = BarIdentification(df)
    df = SwingPoints2(df)

    return df
