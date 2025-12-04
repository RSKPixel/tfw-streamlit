from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    CrosshairTool,
    CustomJS,
    WheelPanTool,
    WheelZoomTool,
    PanTool,
)
import pandas as pd


def plot_tv_ohlc_bokeh(
    data,
    compare=False,
    date_fmt="%Y-%m-%d",
    title="TradingView OHLC (TV-like)",
    swing=True,
    debugging=False,
):

    data = data.copy()

    # keep a REAL x (for swing plotting) and an integer x for candle alignment
    data["x_real"] = data["x"].values
    data["x_int"] = data["x"].round().astype(int)

    # hover-friendly date string
    if isinstance(data.index, pd.DatetimeIndex):
        data["date_str"] = data.index.strftime(date_fmt)
    else:
        data["date_str"] = data["x"].astype(str)

    # build ColumnDataSource (note: x-> integer positions for candles)
    source = ColumnDataSource(
        {
            "x": data["x_int"].values,  # used by candle glyphs
            "x_real": data["x_real"].values,  # used by swing plotting
            "open": data["open"].values,
            "high": data["high"].values,
            "low": data["low"].values,
            "close": data["close"].values,
            "bar_type": data["bar_type"].values,
            "swing_point": data["swing_point"].values,
            "date_str": data["date_str"].values,
        }
    )

    if compare:
        if "bar_type_penfold" in data.columns:
            source.data["bar_type_penfold"] = data["bar_type_penfold"].values
        if "swing_point_penfold" in data.columns:
            source.data["swing_point_penfold"] = data["swing_point_penfold"].values
        if "swing_penfold" in data.columns:
            source.data["swing_penfold"] = data["swing_penfold"].values

    # Colors

    if debugging:
        up = "#777777"
        down = "#777777"
    else:
        up = "#26a69a"
        down = "#ef5350"

    osb = "#1e80ff"
    isb = "#ffeb3b"
    swing_color = "#ff9800"

    # color map
    if debugging:
        source.data["color"] = [
            osb if b == "OSB" else isb if b == "ISB" else up
            for b in source.data["bar_type"]
        ]
    else:
        source.data["color"] = [
            up if c >= o else down
            for o, c in zip(source.data["open"], source.data["close"])
        ]
    # ticks (for small open/close ticks)
    tick_size = 0.25
    source.data["open_tick_x"] = source.data["x"] - tick_size
    source.data["close_tick_x"] = source.data["x"] + tick_size

    # --- Figure (do NOT set active_scroll in figure) ---
    p = figure(
        width=1200,
        height=520,
        x_axis_type="linear",
        title=title,
        background_fill_color="#000000",
        border_fill_color="#000000",
        outline_line_color="#444c56",
        # active_scroll will be set below to WheelZoomTool
    )

    # styling
    p.title.text_color = "white"
    p.xaxis.major_label_text_color = "white"
    p.yaxis.major_label_text_color = "white"
    p.grid.grid_line_color = "#30363d"
    p.grid.grid_line_alpha = 0.15

    # --- Tools: horizontal trackpad pan + vertical zoom + mouse drag pan ---
    wheel_pan = WheelPanTool(dimension="width")  # horizontal scroll -> pan
    wheel_zoom = WheelZoomTool()  # scroll/gesture zoom
    pan_tool = PanTool(dimensions="width")  # mouse drag horizontally (optional)
    p.add_tools(wheel_pan, wheel_zoom, pan_tool)

    # set default scroll action to zoom (so two-finger up/down or pinch zooms)
    p.toolbar.active_scroll = wheel_zoom
    # optionally set active_drag if you want drag to pan by default:
    # p.toolbar.active_drag = pan_tool

    # --- Prevent candles from leaving frame ---
    xmin = int(data["x_int"].min())
    xmax = int(data["x_int"].max())
    p.x_range.bounds = (xmin - 1, xmax + 1)
    p.x_range.range_padding = 0.6
    p.x_range.min_interval = 1
    p.min_border_left = 10
    p.min_border_right = 10

    # --- Draw candles (using x_int so they're pixel aligned) ---
    p.segment("x", "low", "x", "high", source=source, line_width=2, color="color")
    p.segment(
        "x", "open", "open_tick_x", "open", source=source, line_width=3, color="color"
    )
    p.segment(
        "x",
        "close",
        "close_tick_x",
        "close",
        source=source,
        line_width=3,
        color="color",
    )

    # --- Swing line & markers (use real x so alignment matches data index) ---
    if swing:
        df_sw = data[data["swing_point"].notna()]
        if not df_sw.empty:
            # plot using real x coordinates
            p.line(
                df_sw["x_real"].values,
                df_sw["swing_point"].values,
                color=swing_color,
                line_width=2,
            )
            p.scatter(
                df_sw["x_real"].values,
                df_sw["swing_point"].values,
                size=6,
                fill_color=swing_color,
                line_color="white",
                line_width=0.5,
            )

    if compare:
        # --- Penfold Swing line & markers (use real x so alignment matches data index) ---
        if "swing_point_penfold" in data.columns:
            df_sw_penfold = data[data["swing_point_penfold"].notna()]
            if not df_sw_penfold.empty:
                # plot using real x coordinates
                p.line(
                    df_sw_penfold["x_real"].values,
                    df_sw_penfold["swing_point_penfold"].values,
                    color="#00e676",
                    line_width=2,
                )
                p.scatter(
                    df_sw_penfold["x_real"].values,
                    df_sw_penfold["swing_point_penfold"].values,
                    size=6,
                    fill_color="#00e676",
                    line_color="white",
                    line_width=0.5,
                )

        # --- Highlight Swing/Penfold Swing mismatches ---
        if "error" in data.columns:
            df_sw_penfold_errors = data[data["error"]]
            if not df_sw_penfold_errors.empty:
                p.segment(
                    x0=df_sw_penfold_errors["x_real"].values,
                    y0=data["low"].min(),
                    x1=df_sw_penfold_errors["x_real"].values,
                    y1=data["high"].max(),
                    color="red",
                    line_width=0.5,
                    line_alpha=0.8,
                )
    if debugging:
        # --- Highlight Stacked Swing Bars ---
        if "duplicate_swings" in data.columns:
            df_sw_stacked = data[data["duplicate_swings"]]
            if not df_sw_stacked.empty:
                p.segment(
                    x0=df_sw_stacked["x_real"].values,
                    y0=data["low"].min(),
                    x1=df_sw_stacked["x_real"].values,
                    y1=data["high"].max(),
                    color="white",
                    line_width=1,
                    line_alpha=0.8,
                )

    # --- Hover (single floating auto-aligned) ---
    hover = HoverTool(
        tooltips=[
            ("date", "@date_str"),
            ("open", "@open{0.2f}"),
            ("high", "@high{0.2f}"),
            ("low", "@low{0.2f}"),
            ("close", "@close{0.2f}"),
            ("type", "@bar_type"),
            ("swing", "@swing_point{0.2f}"),
        ],
        mode="mouse",
        # attachment="auto",
        # anchor="auto",
        line_policy="nearest",
    )
    p.add_tools(hover)

    # --- Crosshair ---
    p.add_tools(CrosshairTool(line_color="#888888", line_alpha=0.6, line_width=1))

    # --- Auto-fit Y (includes swing_point) via JS callback ---
    callback = CustomJS(
        args=dict(p=p, source=source),
        code="""
        const xr = p.x_range;
        const start = xr.start;
        const end = xr.end;

        const xs = source.data['x'];
        const highs = source.data['high'];
        const lows = source.data['low'];
        const swings = source.data['swing_point'];

        let min_val = Infinity;
        let max_val = -Infinity;

        for (let i = 0; i < xs.length; i++) {
            if (xs[i] >= start && xs[i] <= end) {
                const h = highs[i];
                const l = lows[i];
                if (h > max_val) max_val = h;
                if (l < min_val) min_val = l;

                const s = swings[i];
                if (s !== null && !isNaN(s)) {
                    if (s > max_val) max_val = s;
                    if (s < min_val) min_val = s;
                }
            }
        }

        if (max_val > min_val) {
            const pad = (max_val - min_val) * 0.12;
            p.y_range.start = min_val - pad;
            p.y_range.end = max_val + pad;
        }
    """,
    )
    p.x_range.js_on_change("start", callback)
    p.x_range.js_on_change("end", callback)

    return p
