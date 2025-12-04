import matplotlib.pyplot as plt
import pandas as pd


def plot_tv_ohlc_dark_v2(
    data,
    tops=None,
    bottoms=None,
    date_fmt="%Y-%m-%d",
    title="TradingView Dark-Mode OHLC",
    debugging=False,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(14, 6))

    # numeric X axis
    x = data["x"].values

    openp = data["open"].values
    highp = data["high"].values
    lowp = data["low"].values
    closep = data["close"].values
    bar_type = data["bar_type"].values
    swing_points = data["swing_point"].values
    # swing_points_index = data[data["swing_point"].notna()].index
    # bar_type_penfold = data["bar_type_penfold"].values

    tick = 0.25
    wick_width = 1.3
    line_width = 2.0

    # Colors (TradingView Dark)
    up = "#26a69a"  # teal green
    down = "#ef5350"  # soft red
    up = "#777777"  # medium gray
    down = "#777777"  # medium gray
    osb = "#1e80ff"  # more lighter blue
    isb = "#ffeb3b"  # yellow
    bg = "#000000"  # deep dark
    grid_color = "#30363d"
    border_color = "#444c56"

    # need to put bar number on every bar for debugging
    if debugging:
        for i in range(len(x)):

            if bar_type[i] == "OSB":
                ax.text(
                    x[i],
                    highp[i] + (highp[i] * 1 / 100),
                    str(i),
                    color="white",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

    # --- OHLC Bars ---
    for i in range(len(x)):
        bull = closep[i] >= openp[i]
        col = up if bull else down
        if debugging:
            if bar_type[i] == "OSB":
                col = osb
            elif bar_type[i] == "ISB":
                col = isb

        # Wick
        ax.plot([x[i], x[i]], [lowp[i], highp[i]], color=col, linewidth=wick_width)

        # Open tick
        ax.plot(
            [x[i] - tick, x[i]], [openp[i], openp[i]], color=col, linewidth=line_width
        )

        # Close tick
        ax.plot(
            [x[i], x[i] + tick], [closep[i], closep[i]], color=col, linewidth=line_width
        )

    # # --- Tops ---
    # if tops:
    #     for xx, price in tops:
    #         ax.plot(
    #             xx,
    #             price,
    #             "v",
    #             color="#1e90ff",
    #             markersize=7,
    #             markeredgecolor="white",
    #             markeredgewidth=0.9,
    #         )
    # # --- Bottoms ---
    # if bottoms:
    #     for xx, price in bottoms:
    #         ax.plot(
    #             xx,
    #             price,
    #             "^",
    #             color="#ffa726",
    #             markersize=7,
    #             markeredgecolor="white",
    #             markeredgewidth=0.9,
    #         )

    # --- X-axis Date Labels ---
    idx = data.index
    labels = [d.strftime(date_fmt) if hasattr(d, "strftime") else str(d) for d in idx]
    step = max(len(data) // 12, 1)

    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, color="white")

    # --- Styling (Dark Mode) ---
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    ax.tick_params(colors="white")
    ax.grid(True, linestyle="--", alpha=0.15, color=grid_color)

    # Border
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.2)

    ax.set_title(title, fontsize=12, color="white", weight="bold")
    ax.set_ylabel("Price", color="white")

    # --- Overlay Swing Line ---
    df_sw = data[data["swing_point"].notna()]

    if not df_sw.empty:
        sx = df_sw.index.map(
            lambda t: x[data.index.get_loc(t)]
        )  # map datetime to x coordinate
        sy = df_sw["swing_point"].values
        # Connect swing points
        ax.plot(
            sx,
            sy,
            color="#ff9800",
            linewidth=1,
            zorder=0,  # <-- ensures swing line stays on top
        )

        # Optional: small dots for clarity
        ax.scatter(
            sx,
            sy,
            color="#ff9800",
            s=29,
            edgecolor="white",
            linewidth=0.1,
            zorder=0,
        )

    return fig
