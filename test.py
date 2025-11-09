import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# --- Pattern Logic ---
def rw_top(data: np.ndarray, curr_index: int, order: int) -> bool:
    if curr_index < order or curr_index + order >= len(data):
        return False
    v = data[curr_index]
    for i in range(1, order + 1):
        if data[curr_index - i] > v or data[curr_index + i] > v:
            return False
    return True


def rw_bottom(data: np.ndarray, curr_index: int, order: int) -> bool:
    if curr_index < order or curr_index + order >= len(data):
        return False
    v = data[curr_index]
    for i in range(1, order + 1):
        if data[curr_index - i] < v or data[curr_index + i] < v:
            return False
    return True


# --- Streamlit UI ---
st.title("📈 Algo Pattern Visualizer")
st.caption("Visualize TA patterns like swing highs/lows interactively")

# Generate synthetic data
df = pd.read_json('data/nifty-dataset.json')
df["date"] = pd.to_datetime(df["date"])
df=df[["date","open","high","low","close"]]

# Sidebar controls
sidebar = st.sidebar.selectbox("Select Pattern", ["Swing Highs/Lows"])
order = st.sidebar.slider("Swing Order", 1, 10, 3)
data_length = st.sidebar.slider("Data Length", 50, len(df["close"]), 150)

df = df[-data_length:].reset_index(drop=True)


# Detect patterns
tops = [i for i in range(len(df)) if rw_top(df["close"].values, i, order)]
bottoms = [i for i in range(len(df)) if rw_bottom(df["close"].values, i, order)]

# --- Plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close", line=dict(width=2)))
fig.add_trace(go.Scatter(
    x=tops, y=df["close"].iloc[tops],
    mode="markers", name="Tops", marker=dict(color="red", size=10, symbol="triangle-up")
))
fig.add_trace(go.Scatter(
    x=bottoms, y=df["close"].iloc[bottoms],
    mode="markers", name="Bottoms", marker=dict(color="green", size=10, symbol="triangle-down")
))
fig.update_layout(title=f"Swing Tops & Bottoms (Order={order})", height=500)
st.plotly_chart(fig, use_container_width=True)

# Optional data display
if st.checkbox("Show Data Table"):
    st.dataframe(df)
