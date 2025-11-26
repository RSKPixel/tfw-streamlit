import pandas as pd
from core import get_sqlalchemy_engine
import streamlit as st
from app import eod
from datetime import datetime, timedelta
import numpy as np
from swingchart import plot_tv_ohlc_dark_v2

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timedelta


engine = get_sqlalchemy_engine()


def main():
    st.title("SSC Swing Calculator and Plotter")
    st.set_page_config(layout="wide")
    to_date = datetime.now().date().strftime("%Y-%m-%d")
    no_of_days = 100
    symbol = "FINNIFTY"
    from_date = (
        (datetime.now() - timedelta(days=no_of_days)).date().strftime("%Y-%m-%d")
    )
    st.text(f"Fetching EOD data for {symbol} from {from_date} to {to_date}")
    df = eod(symbol, str(from_date), str(to_date))

    # df: must have columns: ['date','open','high','low','close'] or be indexed by datetime
    # ensure your df has date column or datetime index and columns ['open','high','low','close']
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    # 1) classify bars (optional - compute_ssc will call if missing)
    df = classify_bars_ssc(df)

    # 2) compute swings
    df_with_swings, swings = compute_ssc(df)

    # 3) convert swings to plotting lists
    df_with_swings["x"] = range(len(df_with_swings))
    tops = [(int(s[0]), float(s[2])) for s in swings if s[1] == "high"]
    bottoms = [(int(s[0]), float(s[2])) for s in swings if s[1] == "low"]

    # 4) plot with your existing plotting function
    fig = plot_tv_ohlc_dark_v2(df_with_swings, tops, bottoms)
    st.pyplot(fig)
    st.dataframe(df_with_swings)


# Types:
# swings_list -> List[ (idx_pos:int, 'high'|'low', price:float, confirmed:bool, source_bar_index:int) ]


def classify_bars_ssc(df: pd.DataFrame) -> pd.DataFrame:

    def barType(prev_h, prev_l, h, l):
        if (h > prev_h) and (l >= prev_l):
            return "DB"
        if (l < prev_l) and (h <= prev_h):
            return "DB"
        if (h <= prev_h) and (l >= prev_l):
            return "ISB"
        if (h >= prev_h) and (l <= prev_l):
            return "OSB"
        return "DB"  # fallback directional bar

    """
    Classify each bar as:
      - 'DBU' (directional up), 'DBD' (directional down), 'ISB', 'OSB'
    Uses the strict SSC rules:
      DBU: higher high AND higher/equal low  (h > prev_h and l >= prev_l)
      DBD: lower low AND lower/equal high   (l < prev_l and h <= prev_h)
      ISB: h <= prev_h AND l >= prev_l
      OSB: h >= prev_h AND l <= prev_l
    Note: DBU/DBD take precedence.
    """
    df = df.copy()
    df["bartype"] = None
    if len(df) == 0:
        return df

    df.iloc[0, df.columns.get_loc("bartype")] = "DB"  # first bar considered DB
    previous_db = 0

    for i in range(1, len(df)):
        prev_h = df.iloc[i - 1]["high"]
        prev_l = df.iloc[i - 1]["low"]
        h = df.iloc[i]["high"]
        l = df.iloc[i]["low"]

        # Strict DB up/down
        if (h > prev_h) and (l >= prev_l):
            df.iat[i, df.columns.get_loc("bartype")] = "DB"
            previous_db = i
            continue
        if (l < prev_l) and (h <= prev_h):
            df.iat[i, df.columns.get_loc("bartype")] = "DB"
            previous_db = i
            continue

        # ISB
        if (h <= prev_h) and (l >= prev_l):
            df.iat[i, df.columns.get_loc("bartype")] = "ISB"
            for j in range(i + 1, len(df)):
                prev_db_high = df.iloc[previous_db]["high"]
                prev_db_low = df.iloc[previous_db]["low"]
                h = df.iloc[j]["high"]
                l = df.iloc[j]["low"]

                if (h <= prev_db_high) and (l >= prev_db_low):
                    df.iat[j, df.columns.get_loc("bartype")] = "ISB"
                else:
                    break
            i = j - 1
            continue

        # OSB
        if (h >= prev_h) and (l <= prev_l):
            df.iat[i, df.columns.get_loc("bartype")] = "OSB"
            continue

        # Fallback directional bar if ambiguous
        df.iat[i, df.columns.get_loc("bartype")] = "DB"
    return df


def compute_ssc(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Tuple[int, str, float, bool, int]]]:
    """
    Compute SSC swings faithfully to the manual.

    Returns:
      df_out (copy with columns added):
         - 'previous_db' (Timestamp), 'swing' ('high'|'low' or None), 'swing_confirmed' (bool),
           'swg_idx' (int), 'is_stacked_swing' (bool)
      swings_list: list of tuples (idx_pos, 'high'|'low', price, confirmed, source_bar_index)
    """

    df = df.copy()
    if "bartype" not in df.columns:
        df = classify_bars_ssc(df)

    n = len(df)
    bartypes = df["bartype"].values
    highs = df["high"].values
    lows = df["low"].values

    # result columns
    df["previous_db"] = pd.NaT
    df["swing"] = None
    df["swing_confirmed"] = False
    df["swg_idx"] = np.nan
    df["is_stacked_swing"] = False

    # helpers
    def find_prev_db_index(i):
        """Find previous non-ISB DB (index j < i where bartype is DB*)."""
        for j in range(i - 1, -1, -1):
            b = bartypes[j]
            if b is not None and b.startswith("DB"):
                return j
        return None

    def find_prev_non_isb_before(idx):
        for j in range(idx - 1, -1, -1):
            if (
                bartypes[j] != "ISB"
                and bartypes[j] is not None
                and bartypes[j].startswith("DB")
            ):
                return j
        return None

    # trackers
    snap_hh: Optional[float] = None
    snap_ll: Optional[float] = None
    prev_confirmed_swing_type: Optional[str] = None  # 'high'/'low' or None
    prev_confirmed_swing_index: Optional[int] = None
    prev_trend: Optional[str] = None  # 'Up' or 'Down'
    last_swing_idx: Optional[int] = None

    # temporary list of OSBs encountered since last DB
    pending_osbs: List[int] = []

    swings: List[Tuple[int, str, float, bool, int]] = []

    for i in range(n):
        bt = bartypes[i]

        # Record previous_db pointer
        pdb = find_prev_db_index(i)
        if pdb is not None:
            df.at[df.index[i], "previous_db"] = df.index[pdb]

        # --- ISB: SSC ignores ISB for swing creation (but record previous_db) ---
        if bt == "ISB":
            # just continue, but keep OSB pending unchanged
            continue

        # --- OSB: initially ignored, but stored for later review ---
        if bt == "OSB":
            # add to pending list for later review
            pending_osbs.append(i)
            # update snaps as OSBs can hold extremes used for snapping later
            if snap_hh is None or highs[i] > snap_hh:
                snap_hh = highs[i]
            if snap_ll is None or lows[i] < snap_ll:
                snap_ll = lows[i]
            # We do not change swings now; will evaluate OSBs when a DB occurs or when they break confirmed swing
            # But SSC requires: if OSB breaks the most recent confirmed swing, create the opposite swing on OSB extents
            # So check that immediately:
            if prev_confirmed_swing_index is not None:
                if prev_confirmed_swing_type == "high":
                    # if OSB high breaks previous confirmed swing high -> create new swing low at OSB low
                    if highs[i] > df.iloc[prev_confirmed_swing_index]["high"]:
                        # create a swing low at OSB low (index of OSB)
                        sw_idx = i
                        price = lows[i]
                        confirmed = False  # only DBs confirm; OSB creates swing and waits for confirm by DB
                        df.at[df.index[sw_idx], "swing"] = "low"
                        df.at[df.index[sw_idx], "swing_confirmed"] = False
                        df.at[df.index[sw_idx], "swg_idx"] = sw_idx
                        swings.append((sw_idx, "low", price, False, i))
                        last_swing_idx = sw_idx
                        # After creating this swing low, also set prev_confirmed_swing? Not yet: confirmation occurs on DB
                elif prev_confirmed_swing_type == "low":
                    if lows[i] < df.iloc[prev_confirmed_swing_index]["low"]:
                        sw_idx = i
                        price = highs[i]
                        df.at[df.index[sw_idx], "swing"] = "high"
                        df.at[df.index[sw_idx], "swing_confirmed"] = False
                        df.at[df.index[sw_idx], "swg_idx"] = sw_idx
                        swings.append((sw_idx, "high", price, False, i))
                        last_swing_idx = sw_idx
            continue  # OSB handled (defer full resolution)

        # --- DB* handling (DBU / DBD / DB) - these bars confirm swings and resolve pending OSBs ---
        if bt is not None and bt.startswith("DB"):
            # Before processing this DB, review pending OSBs (if any) for stacked/duplicate rules
            if pending_osbs:
                # Evaluate pending OSBs in chronological order
                # For each OSB we already may have added preliminary swings when it broke confirmed swings.
                # Now we must resolve stacked swings and duplicates using the rules in manual.
                # We'll implement the manual's rules:
                # - If multiple OSBs produced alternating duplicates, wait for this DB to confirm which side.
                # - If stacking occurred (both high & low on same OSB), confirm one and move the other to prior non-ISB DB.
                # Implementing a faithful automated resolution:
                for osb_idx in pending_osbs:
                    # check if this OSB currently has both a swing high and low created on it (stacked)
                    existing_swings_at_osb = [s for s in swings if s[0] == osb_idx]
                    if len(existing_swings_at_osb) >= 2:
                        # stacked swings exist. Manual says: examine current DB direction:
                        if bt.startswith("DBD") or (
                            bt == "DB" and highs[i] < highs[i - 1]
                        ):  # DB down-ish
                            # confirm swing high on OSB high, move swing low to prior non-ISB DB low
                            # find the low swing entry at this osb and move it
                            # locate prior non-ISB DB index
                            move_to = find_prev_non_isb_before(osb_idx)
                            # find the high swing entry and mark confirmed
                            for s in existing_swings_at_osb:
                                if s[1] == "high":
                                    # confirm this high
                                    sw_idx = s[0]
                                    df.at[df.index[sw_idx], "swing_confirmed"] = True
                                    # update swings list entry to confirmed True
                                    # (we will rebuild unique/sorted list later)
                            if move_to is not None:
                                # move low to prior non ISB (create new) and mark the OSB record as stacked
                                df.at[df.index[osb_idx], "is_stacked_swing"] = True
                                # place a low on move_to (conservative: not confirmed)
                                if df.at[df.index[move_to], "swing"] is None:
                                    df.at[df.index[move_to], "swing"] = "low"
                                    df.at[df.index[move_to], "swing_confirmed"] = False
                                    df.at[df.index[move_to], "swg_idx"] = move_to
                                    swings.append(
                                        (
                                            move_to,
                                            "low",
                                            df.iloc[move_to]["low"],
                                            False,
                                            osb_idx,
                                        )
                                    )
                        else:
                            # DB up-ish: confirm low on OSB low and move high to prior non-ISB DB high
                            move_to = find_prev_non_isb_before(osb_idx)
                            for s in existing_swings_at_osb:
                                if s[1] == "low":
                                    sw_idx = s[0]
                                    df.at[df.index[sw_idx], "swing_confirmed"] = True
                            if move_to is not None:
                                df.at[df.index[osb_idx], "is_stacked_swing"] = True
                                if df.at[df.index[move_to], "swing"] is None:
                                    df.at[df.index[move_to], "swing"] = "high"
                                    df.at[df.index[move_to], "swing_confirmed"] = False
                                    df.at[df.index[move_to], "swg_idx"] = move_to
                                    swings.append(
                                        (
                                            move_to,
                                            "high",
                                            df.iloc[move_to]["high"],
                                            False,
                                            osb_idx,
                                        )
                                    )

                # handle duplicate OSBs producing consecutive same swings:
                # if swings produce duplicate 'high' then next DB decides reallocation
                # Find duplicates in chronological order
                # We'll defer a definitive restructure until we see the DB direction (we have it now).
                # Build chronological swings mapped by type
                # Simple approach: ensure alternation by post-processing after we append this DB's confirmations below.

                pending_osbs = []  # cleared after resolution

            # Now DB may confirm earlier unconfirmed swings (including those created on OSBs)
            # Check if a swing should be confirmed now (manual: confirmed only by DBs)
            # If the previous bars indicate a reversal (DBU -> DBD or DBD -> DBU), create/confirm swing on previous_db
            prev_db_idx = find_prev_db_index(i)
            if prev_db_idx is not None:
                # Determine direction flip: compare prev_db vs current DB
                prev_db_type = bartypes[prev_db_idx]
                # If prev_db was DBU and current DB is DBD (or vice-versa), a confirmed swing occurs on prev_db
                # We follow manual: a DB up followed by DB down => confirmed swing high on previous DB's high
                if prev_db_type is not None and prev_db_type.startswith("DB"):
                    # current is down and previous was up -> confirmed swing high
                    if (prev_db_type.startswith("DBU") or prev_db_type == "DB") and (
                        bt.startswith("DBD")
                        or (bt == "DB" and lows[i] < lows[prev_db_idx])
                    ):
                        sw_idx = prev_db_idx
                        price = highs[sw_idx]
                        # Snap: check for any higher high between prev_confirmed_swing_index and this DB (usually on OSBs)
                        # Manual: snap to highest high between current confirmed swing high and previous confirmed swing low
                        if (
                            prev_confirmed_swing_index is not None
                            and prev_confirmed_swing_type == "low"
                        ):
                            # find highest high in range (prev_confirmed_swing_index+1 .. i)
                            seg_hi = df.iloc[prev_confirmed_swing_index + 1 : i + 1][
                                "high"
                            ]
                            if len(seg_hi):
                                hh = seg_hi.max()
                                if hh > price:
                                    price = hh
                        df.at[df.index[sw_idx], "swing"] = "high"
                        df.at[df.index[sw_idx], "swing_confirmed"] = True
                        df.at[df.index[sw_idx], "swg_idx"] = sw_idx
                        swings.append((sw_idx, "high", price, True, i))
                        prev_confirmed_swing_type = "high"
                        prev_confirmed_swing_index = sw_idx
                        last_swing_idx = sw_idx
                        prev_trend = "Down"

                    # current is up and previous was down -> confirmed swing low
                    elif (prev_db_type.startswith("DBD") or prev_db_type == "DB") and (
                        bt.startswith("DBU")
                        or (bt == "DB" and highs[i] > highs[prev_db_idx])
                    ):
                        sw_idx = prev_db_idx
                        price = lows[sw_idx]
                        if (
                            prev_confirmed_swing_index is not None
                            and prev_confirmed_swing_type == "high"
                        ):
                            seg_lo = df.iloc[prev_confirmed_swing_index + 1 : i + 1][
                                "low"
                            ]
                            if len(seg_lo):
                                ll = seg_lo.min()
                                if ll < price:
                                    price = ll
                        df.at[df.index[sw_idx], "swing"] = "low"
                        df.at[df.index[sw_idx], "swing_confirmed"] = True
                        df.at[df.index[sw_idx], "swg_idx"] = sw_idx
                        swings.append((sw_idx, "low", price, True, i))
                        prev_confirmed_swing_type = "low"
                        prev_confirmed_swing_index = sw_idx
                        last_swing_idx = sw_idx
                        prev_trend = "Up"

            # update snap extremes with this DB
            if snap_hh is None or highs[i] > snap_hh:
                snap_hh = highs[i]
            if snap_ll is None or lows[i] < snap_ll:
                snap_ll = lows[i]

            # Also confirm any earlier OSB-created swings if this DB confirms them per manual:
            # e.g., if an OSB created a swing low (unconfirmed) and now a DB up occurs, that DB confirms it.
            # We'll scan swings list for unconfirmed entries and confirm if this DB direction confirms them.
            newly_confirmed = []
            for idx_s, typ_s, price_s, conf_s, source_idx in swings:
                if not conf_s:
                    # If type is low and current DB is up -> confirm
                    if typ_s == "low" and (
                        bt.startswith("DBU") or (bt == "DB" and highs[i] > highs[idx_s])
                    ):
                        # confirm and maybe snap
                        conf_price = price_s
                        if (
                            prev_confirmed_swing_index is not None
                            and prev_confirmed_swing_type == "high"
                        ):
                            seg_lo = df.iloc[prev_confirmed_swing_index + 1 : i + 1][
                                "low"
                            ]
                            if len(seg_lo):
                                ll = seg_lo.min()
                                if ll < conf_price:
                                    conf_price = ll
                        # update df and list
                        df.at[df.index[idx_s], "swing_confirmed"] = True
                        newly_confirmed.append((idx_s, typ_s, conf_price))
                    # If type is high and current DB is down -> confirm
                    if typ_s == "high" and (
                        bt.startswith("DBD") or (bt == "DB" and lows[i] < lows[idx_s])
                    ):
                        conf_price = price_s
                        if (
                            prev_confirmed_swing_index is not None
                            and prev_confirmed_swing_type == "low"
                        ):
                            seg_hi = df.iloc[prev_confirmed_swing_index + 1 : i + 1][
                                "high"
                            ]
                            if len(seg_hi):
                                hh = seg_hi.max()
                                if hh > conf_price:
                                    conf_price = hh
                        df.at[df.index[idx_s], "swing_confirmed"] = True
                        newly_confirmed.append((idx_s, typ_s, conf_price))
            # apply newly confirmed changes (and set prev_confirmed_swing accordingly to maintain alternation)
            for idx_s, typ_s, conf_price in newly_confirmed:
                # update swing tuple in swings list: mark confirmed True (we'll dedupe later)
                swings.append((idx_s, typ_s, conf_price, True, i))
                prev_confirmed_swing_type = typ_s
                prev_confirmed_swing_index = idx_s
                last_swing_idx = idx_s

            # After DB processing, clear pending osbs
            pending_osbs = []
            continue

        # other types ignored
        continue

    # Post-processing to enforce alternation and resolve duplicates more cleanly:
    # Build a map of the highest-priority swing for each index & type and then force alternation.
    # Keep earliest index for each alternating sequence.
    swings_sorted = sorted(
        swings, key=lambda s: (s[0], 0 if s[3] else 1)
    )  # prefer confirmed entries
    unique = {}
    for s in swings_sorted:
        key = (s[0], s[1])
        if key not in unique:
            unique[key] = s

    swings_list = sorted(unique.values(), key=lambda x: x[0])

    # enforce alternation: convert to final list by scanning in chronological order and ensuring high/low alternation
    final = []
    last_type = None
    for s in swings_list:
        idx_pos, typ, price, conf, src = s
        if last_type is None:
            final.append(s)
            last_type = typ
        else:
            if typ == last_type:
                # duplicate. Manual: wait for next DB to resolve, but as we are post-processing,
                # attempt to transform earlier duplicate to opposite on prior non-ISB DB to preserve alternation.
                move_to = find_prev_non_isb_before(idx_pos)
                if move_to is not None:
                    # create opposite swing on move_to
                    opp = "low" if typ == "high" else "high"
                    p = df.iloc[move_to]["low" if opp == "low" else "high"]
                    final.append((move_to, opp, p, False, idx_pos))
                    # then append current
                    final.append(s)
                    last_type = s[1]
                else:
                    # fallback: skip the duplicate (rare)
                    continue
            else:
                final.append(s)
                last_type = typ

    # write final swings back into df columns (ensure existing marks updated)
    # Clear previous swing columns and re-write
    df["swing"] = None
    df["swing_confirmed"] = False
    df["swg_idx"] = np.nan
    df["is_stacked_swing"] = False

    for s in final:
        idx_pos, typ, price, conf, src = s
        df.at[df.index[idx_pos], "swing"] = typ
        df.at[df.index[idx_pos], "swing_confirmed"] = bool(conf)
        df.at[df.index[idx_pos], "swg_idx"] = idx_pos
        # mark stacked if source differs and source was OSB
        if src != idx_pos and bartypes[src] == "OSB":
            df.at[df.index[idx_pos], "is_stacked_swing"] = True

    return df, final


if __name__ == "__main__":
    main()
