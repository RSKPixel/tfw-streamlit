import pandas as pd
import numpy as np
import ut_tools_ssc as ssc
import streamlit as st
from app import eod


def main():
    df = eod("NIFTY", "2023-01-01", "2023-12-31")
    df = ssc.SSC(df)

    st.title("SSC Data Viewer")
    st.dataframe(df[["date", "open", "high", "low", "close", "bar_type"]])


if __name__ == "__main__":
    main()
