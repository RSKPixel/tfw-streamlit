import pandas as pd


def BarType(previous_bar, current_bar) -> str:
    prev_h, prev_l = previous_bar["high"], previous_bar["low"]
    h, l = current_bar["high"], current_bar["low"]

    # Outside Bar (first!)
    if h > prev_h and l < prev_l:
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

    # previous_db_index = 0
    # for i in range(1, len(df)):
    #     previous_db_bar = df.iloc[previous_db_index]

    #     if (
    #         df.iloc[i]["bar_type"] == "DB" or df.iloc[i]["bar_type"] == "OSB"
    #     ) and df.iloc[i - 1]["bar_type"] == "ISB":
    #         df.at[df.index[i], "bar_type"] = BarType(previous_db_bar, df.iloc[i])
    #     elif df.iloc[i]["bar_type"] == "DB" or df.iloc[i]["bar_type"] == "OSB":
    #         previous_db_index = i

    # Resolving ISB Bars
    for i in range(1, len(df)):
        anchor_date, anchor_h, anchor_l, anchor_bt = (
            df.iloc[i - 1].name,
            df.iloc[i - 1]["high"],
            df.iloc[i - 1]["low"],
            df.iloc[i - 1]["bar_type"],
        )

        if i >= 449 and i <= 452:

            print(i, anchor_date, anchor_h, anchor_l, anchor_bt)

        for j in range(i, len(df)):

            current_h, current_l = df.iloc[j]["high"], df.iloc[j]["low"]

            if current_h < anchor_h and current_l > anchor_l:
                df.at[df.index[j], "bar_type"] = "ISB"
            else:
                if j == 450:
                    print(df.iloc[j].name, df.iloc[j]["bar_type"])
                i = j
                break

    return df
