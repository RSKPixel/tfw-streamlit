import matplotlib.pyplot as plt
import pandas as pd


def plot_tv_ohlc_dark_v2(
    data,
    tops=None,
    bottoms=None,
    date_fmt="%Y-%m-%d",
    title="TradingView Dark-Mode OHLC",
) -> plt.Figure:
    import matplotlib.pyplot as plt
    import streamlit as st

    fig, ax = plt.subplots(figsize=(14, 6))

    # numeric X axis
    x = data["x"].values

    openp = data["open"].values
    highp = data["high"].values
    lowp = data["low"].values
    closep = data["close"].values
    bar_types = data["bar_type"].values

    tick = 0.25
    wick_width = 1.3
    line_width = 2.0

    # Colors (TradingView Dark)
    up = "#26a69a"  # teal green
    down = "#ef5350"  # soft red
    osb = "#cfd8dc"  # light gray
    bg = "#0d1117"  # deep dark
    grid_color = "#30363d"
    border_color = "#444c56"

    # --- OHLC Bars ---
    for i in range(len(x)):
        bull = closep[i] >= openp[i]
        col = up if bull else down
        if bar_types[i] == "OSB":
            col = osb

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

    # --- Tops ---
    if tops:
        for xx, price in tops:
            ax.plot(
                xx,
                price + (price * 1 / 100),
                "v",
                color="#1e90ff",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.9,
            )
    # --- Bottoms ---
    if bottoms:
        for xx, price in bottoms:
            ax.plot(
                xx,
                price - (price * 1 / 100),
                "^",
                color="#ffa726",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.9,
            )

    # --- X-axis Date Labels ---
    idx = data.index
    labels = [d.strftime(date_fmt) for d in idx]
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

    return fig
