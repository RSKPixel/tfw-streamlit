import numpy as np
import pandas as pd
from typing import Optional
from app import eod
import streamlit as st
from barchart import plot_tv_ohlc_dark_v2
from ut_tools_ssc import BarDefination, SwingPoints2


def SwingPoints2_realtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Option B: Use RealTest SSC engine internally, but produce the same
    user-facing output as your original SwingPoints2:
      - df['swing'] -> 'high' / 'low' (only set on the bar where RealTest confirms)
      - df['swing_point'] -> value (NaN otherwise)

    Also adds intermediate columns for inspection:
      ssc01 (current dir per bar), ssc02 (unconfirmed val), ssc03 (bars back),
      ssc_confirmed (confirmed value written to earlier bar).
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return df

    # Ensure bar types are present (use your BarDefination)
    if "bar_type" not in df.columns:
        from math import nan

        try:
            df = BarDefination(df)
        except Exception as e:
            raise RuntimeError("BarDefination must be available and working.") from e

    # Allocate arrays (C code conceptually used indices from 1..count-1)
    bsNonISB = np.zeros(n, dtype=int)
    sscDir = np.zeros(n, dtype=int)
    sscBB = np.zeros(n, dtype=int)
    unresolvedOSBbreaks = np.zeros(n, dtype=int)
    OSBbreaksPrevSwing = np.zeros(n, dtype=int)
    osbbpsHigh = np.zeros(n, dtype=float)
    osbbpsLow = np.zeros(n, dtype=float)
    sscValArr = np.zeros(n, dtype=float)  # ssc02 per bar history
    ssc_confirmed = np.full(n, np.nan)  # placed at earlier bar index when confirmed

    # Swing output columns
    swing_col = np.full(n, "", dtype=object)
    swing_point_col = np.full(n, np.nan, dtype=float)

    # Scalar state variables (init to C initial state)
    lastDBHigh = prevLastDBHigh = 0.0
    lastDBLow = prevLastDBLow = 0.0

    ISB = OSB = DB = DBU = DBD = 0
    bsDB = prevbsDB = 0
    ssc01 = 0  # current direction
    ssc02 = 0.0  # current unconfirmed SSC value
    ssc03 = 0  # bars since unconfirmed SSC
    ssc11 = 0  # last confirmed direction (kept for final columns)
    ssc12 = 0.0
    ssc13 = 0
    tempSwingDir = 0
    tempBarsBack = 0

    # main loop: replicate iBar = 1..n-1 behavior from C
    for iBar in range(1, n):
        # reset / preserve previous
        unresolvedOSBbreaks[iBar] = 0
        prevLastDBHigh = lastDBHigh
        prevLastDBLow = lastDBLow
        prevbsDB = bsDB

        nHigh = float(df.loc[iBar, "high"])
        nLow = float(df.loc[iBar, "low"])
        OSBbreaksPrevSwing[iBar] = 0

        # ISB detection using prevLastDB values (C logic)
        if iBar > 1:
            if (nHigh <= prevLastDBHigh) and (nLow >= prevLastDBLow):
                lastDBHigh = prevLastDBHigh
                lastDBLow = prevLastDBLow
                ISB = 1
            else:
                lastDBHigh = nHigh
                lastDBLow = nLow
                ISB = 0
        else:
            lastDBHigh = nHigh
            lastDBLow = nLow
            ISB = 0

        # OSB detection vs immediate previous bar
        prev_high = float(df.loc[iBar - 1, "high"])
        prev_low = float(df.loc[iBar - 1, "low"])
        OSB = 1 if (ISB == 0 and nHigh > prev_high and nLow < prev_low) else 0

        # DB detection
        DB = 1 if (OSB == 0 and ISB == 0) else 0

        # DBU / DBD
        DBU = 1 if (DB == 1 and nHigh > prevLastDBHigh) else 0
        DBD = 1 if (DB == 1 and nLow < prevLastDBLow) else 0

        # bsNonISB and bsDB (bookkeeping)
        bsNonISB[iBar] = 0 if ISB == 0 else (bsNonISB[iBar - 1] + 1)
        bsDB = 0 if DB == 1 else (prevbsDB + 1)

        # counters increment as in C
        ssc03 += 1
        ssc13 += 1

        # unresolved OSB handling
        if (
            (unresolvedOSBbreaks[iBar - 1] == 0)
            and (OSB == 1)
            and (
                ((ssc01 == -1) and (nHigh > ssc12)) or ((ssc01 == 1) and (nLow < ssc12))
            )
        ):
            osbbpsHigh[iBar] = nHigh
            osbbpsLow[iBar] = nLow
            unresolvedOSBbreaks[iBar] = 1
            OSBbreaksPrevSwing[iBar] = 1

        elif unresolvedOSBbreaks[iBar - 1] > 0:
            if DB == 1:
                unresolvedOSBbreaks[iBar] = 0
            elif (nHigh > osbbpsHigh[iBar - 1]) and (nLow < osbbpsLow[iBar - 1]):
                osbbpsHigh[iBar] = nHigh
                osbbpsLow[iBar] = nLow
                unresolvedOSBbreaks[iBar] = unresolvedOSBbreaks[iBar - 1] + 1
                OSBbreaksPrevSwing[iBar] = 1
            else:
                unresolvedOSBbreaks[iBar] = unresolvedOSBbreaks[iBar - 1]
                osbbpsHigh[iBar] = osbbpsHigh[iBar - 1]
                osbbpsLow[iBar] = osbbpsLow[iBar - 1]

        # Complex backtracking when DB occurs and unresolved OSB chain existed
        if (DB == 1) and (unresolvedOSBbreaks[iBar - 1] > 0):
            tempSwingDir = 1 if DBU == 1 else -1

            # search up to ssc13+1 bars back (bounded by available history)
            max_search = min(ssc13 + 1, iBar - 1)
            tempBarsBack = 1
            while tempBarsBack <= max_search:
                idx = iBar - tempBarsBack
                cond1 = OSBbreaksPrevSwing[idx] == 1
                cond2 = tempSwingDir == 1 and (
                    osbbpsLow[idx] > float(df.loc[idx, "low"])
                )
                cond3 = tempSwingDir == -1 and (
                    osbbpsHigh[idx] < float(df.loc[idx, "high"])
                )
                if cond1 or cond2 or cond3:
                    tempSwingDir *= -1

                # break condition in C: if unresolvedOSBbreaks[iBar - (tempBarsBack+1)] == 0 -> break
                check_idx = iBar - (tempBarsBack + 1)
                if check_idx >= 0:
                    if unresolvedOSBbreaks[check_idx] == 0:
                        break
                else:
                    break

                tempBarsBack += 1

            comp_index = max(0, iBar - (tempBarsBack + 1))
            prior_sscDir = int(sscDir[comp_index]) if comp_index < n else 0

            if tempSwingDir != prior_sscDir:
                ssc11 = -1 * tempSwingDir
                bb_index = iBar - (tempBarsBack + 1)
                if bb_index >= 0:
                    ssc13 = int(sscBB[bb_index]) + tempBarsBack + 1
                else:
                    ssc13 = tempBarsBack + 1

                # write confirmed ssc value to earlier bar (same index as RealTest)
                write_idx = iBar - ssc13
                if write_idx >= 0:
                    if ssc11 == 1:
                        ssc12 = float(df.loc[write_idx, "high"])
                    else:
                        ssc12 = float(df.loc[write_idx, "low"])
                    ssc_confirmed[write_idx] = ssc12
                    # mark swing outputs on that write index (match expected user-facing output)
                    swing_col[write_idx] = "high" if ssc11 == 1 else "low"
                    swing_point_col[write_idx] = ssc12

            # set unconfirmed direction and value
            ssc01 = tempSwingDir

            # compute ssc02 using previous sscValArr at bb_index
            bb_index = iBar - (tempBarsBack + 1)
            prev_ssc_val = sscValArr[bb_index] if bb_index >= 0 else 0.0

            if tempSwingDir == 1:
                ssc02 = max(prev_ssc_val, float(df.loc[iBar - tempBarsBack, "high"]))
            else:
                ssc02 = min(prev_ssc_val, float(df.loc[iBar - tempBarsBack, "low"]))

            # update ssc03
            compare_val = (
                float(df.loc[iBar - tempBarsBack, "high"])
                if tempSwingDir == 1
                else float(df.loc[iBar - tempBarsBack, "low"])
            )
            if ssc02 == compare_val:
                ssc03 = tempBarsBack
            else:
                ssc03 = tempBarsBack + (sscBB[bb_index] if bb_index >= 0 else 0) + 1

            # step back one and iterate inner while like C
            tempBarsBack -= 1
            while tempBarsBack > 0:
                idx = iBar - tempBarsBack
                condA = OSBbreaksPrevSwing[idx] == 1
                condB = ssc01 == 1 and float(df.loc[idx, "low"]) < osbbpsLow[idx]
                condC = ssc01 == -1 and float(df.loc[idx, "high"]) > osbbpsHigh[idx]
                if condA or condB or condC:
                    # commit unconfirmed as confirmed
                    ssc11 = ssc01
                    ssc12 = ssc02
                    ssc13 = ssc03
                    write_idx = iBar - ssc13
                    if write_idx >= 0:
                        ssc_confirmed[write_idx] = ssc12
                        swing_col[write_idx] = "high" if ssc11 == 1 else "low"
                        swing_point_col[write_idx] = ssc12

                    # flip and set new unconfirmed
                    ssc01 = -1 * ssc11
                    if ssc01 == 1:
                        ssc02 = float(df.loc[idx, "high"])
                    else:
                        ssc02 = float(df.loc[idx, "low"])
                    ssc03 = tempBarsBack
                tempBarsBack -= 1

        # OSB extension of unconfirmed when it doesn't break previous swing
        if (OSB == 1) and (OSBbreaksPrevSwing[iBar] == 0):
            if ssc01 == 1:
                if nHigh >= ssc02:
                    ssc02 = nHigh
                    ssc03 = 0
            elif ssc01 == -1:
                if nLow <= ssc02:
                    ssc02 = nLow
                    ssc03 = 0

        # DB handling: confirm, flip or extend logic (closely follows C code)
        if DB == 1:
            if ssc01 == 1:
                if DBU == 1:
                    # attempt extension
                    ref_idx = iBar - prevbsDB - 1
                    if ref_idx >= 0 and nHigh > float(df.loc[ref_idx, "high"]):
                        ssc02 = max(ssc02, nHigh)
                        ssc03 = (sscBB[iBar - 1] + 1) if (iBar - 1) >= 0 else 0
                    else:
                        # commit and flip
                        ssc11 = ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        widx = iBar - ssc13
                        if widx >= 0:
                            ssc_confirmed[widx] = ssc12
                            swing_col[widx] = "high"
                            swing_point_col[widx] = ssc12

                        # flip to -1 and find occurrence of prevLastDBLow to set ssc03
                        ssc01 = -1
                        ssc02 = prevLastDBLow
                        found = False
                        for tb in range(0, prevbsDB + 1):
                            idx = iBar - tb
                            idx_prev = iBar - tb - 1
                            if idx >= 0 and idx_prev >= 0:
                                if (float(df.loc[idx, "low"]) == ssc02) and (
                                    float(df.loc[idx_prev, "low"]) != ssc02
                                ):
                                    ssc03 = tb
                                    found = True
                                    break
                        if not found:
                            ssc03 = 0

                        # commit and set new upward unconfirmed
                        ssc11 = -1 * ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        widx = iBar - ssc13
                        if widx >= 0:
                            ssc_confirmed[widx] = ssc12
                            swing_col[widx] = "high" if ssc11 == 1 else "low"
                            swing_point_col[widx] = ssc12

                        ssc01 = 1
                        ssc02 = nHigh
                        ssc03 = 0
                else:
                    if DBD == 1:
                        ssc11 = ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        widx = iBar - ssc13
                        if widx >= 0:
                            ssc_confirmed[widx] = ssc12
                            swing_col[widx] = "high"
                            swing_point_col[widx] = ssc12
                        ssc01 = -1
                        ssc02 = nLow
                        ssc03 = 0

            elif ssc01 == -1:
                if DBD == 1:
                    ref_idx = iBar - prevbsDB - 1
                    if ref_idx >= 0 and nLow < float(df.loc[ref_idx, "low"]):
                        ssc02 = min(ssc02, nLow)
                        ssc03 = (sscBB[iBar - 1] + 1) if (iBar - 1) >= 0 else 0
                    else:
                        ssc11 = ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        widx = iBar - ssc13
                        if widx >= 0:
                            ssc_confirmed[widx] = ssc12
                            swing_col[widx] = "low"
                            swing_point_col[widx] = ssc12

                        ssc01 = 1
                        ssc02 = prevLastDBHigh
                        found = False
                        for tb in range(0, prevbsDB + 1):
                            idx = iBar - tb
                            idx_prev = iBar - tb - 1
                            if idx >= 0 and idx_prev >= 0:
                                if (float(df.loc[idx, "high"]) == ssc02) and (
                                    float(df.loc[idx_prev, "high"]) != ssc02
                                ):
                                    ssc03 = tb
                                    found = True
                                    break
                        if not found:
                            ssc03 = 0

                        ssc11 = ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        widx = iBar - ssc13
                        if widx >= 0:
                            ssc_confirmed[widx] = ssc12
                            swing_col[widx] = "high" if ssc11 == 1 else "low"
                            swing_point_col[widx] = ssc12

                        ssc01 = -1
                        ssc02 = nLow
                        ssc03 = 0

                else:
                    if DBU == 1:
                        ssc11 = ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        widx = iBar - ssc13
                        if widx >= 0:
                            ssc_confirmed[widx] = ssc12
                            swing_col[widx] = "low"
                            swing_point_col[widx] = ssc12
                        ssc01 = 1
                        ssc02 = nHigh
                        ssc03 = 0

            else:
                # initial ssc01 == 0 handling
                ssc03 = 0
                if DBU == 1:
                    ssc01 = 1
                    ssc02 = nHigh
                else:
                    ssc01 = -1
                    ssc02 = nLow

        # store per-bar arrays (like C code)
        sscDir[iBar] = ssc01
        sscValArr[iBar] = ssc02
        sscBB[iBar] = ssc03

    # Compose outputs: attach columns and return same shape as original SwingPoints2
    df["swing"] = swing_col
    df["swing_point"] = swing_point_col

    # Helpful debug columns (can be removed later)
    df["ssc01"] = sscDir
    df["ssc02"] = sscValArr
    df["ssc03"] = sscBB
    df["ssc_confirmed"] = ssc_confirmed

    return df


def SSC_realtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Port of the provided RealTest C SSC routine to Python.
    Expects df with columns 'high' and 'low' (float).
    Returns a copy of df with added SSC columns:
      ssc01, ssc02, ssc03, ssc11, ssc12, ssc13,
      sscDir, sscVal, sscBB, ssc_confirmed
    This aims to reproduce C behavior exactly (indexing and back-writes).
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return df

    # allocate arrays same length (0..n-1); C code used index from 1..countOfBars-1
    bsNonISB = np.zeros(n, dtype=int)
    sscDir = np.zeros(n, dtype=int)
    sscBB = np.zeros(n, dtype=int)
    unresolvedOSBbreaks = np.zeros(n, dtype=int)
    OSBbreaksPrevSwing = np.zeros(n, dtype=int)
    osbbpsHigh = np.zeros(n, dtype=float)
    osbbpsLow = np.zeros(n, dtype=float)
    sscValArr = np.zeros(n, dtype=float)  # stores ssc02 per bar (for history)
    ssc_confirmed = np.full(n, np.nan)  # corresponds to *(pnVal - ssc13)

    # scalar variables
    lastDBHigh = 0.0
    prevLastDBHigh = 0.0
    lastDBLow = 0.0
    prevLastDBLow = 0.0

    ISB = OSB = DB = DBU = DBD = 0
    bsDB = prevbsDB = 0
    ssc01 = 0  # current direction
    ssc02 = 0.0  # current unconfirmed SSC value
    ssc03 = 0  # bars since unconfirmed SSC
    ssc11 = 0  # last confirmed direction
    ssc12 = 0.0  # last confirmed value
    ssc13 = 0  # bars since confirmed
    tempSwingDir = 0
    tempBarsBack = 0

    # loop: C code uses iBar from 1 .. countOfBars-1, referencing previous bars via pBar - k
    # We'll replicate same indexing: iBar in [1..n-1]
    for iBar in range(1, n):
        # setup
        unresolvedOSBbreaks[iBar] = 0
        prevLastDBHigh = lastDBHigh
        prevLastDBLow = lastDBLow
        prevbsDB = bsDB

        nHigh = float(df.loc[iBar, "high"])
        nLow = float(df.loc[iBar, "low"])
        OSBbreaksPrevSwing[iBar] = 0

        # ISB detection using prevLastDB* values
        if iBar > 1:
            if (nHigh <= prevLastDBHigh) and (nLow >= prevLastDBLow):
                # ISB
                lastDBHigh = prevLastDBHigh
                lastDBLow = prevLastDBLow
                ISB = 1
            else:
                lastDBHigh = nHigh
                lastDBLow = nLow
                ISB = 0
        else:
            # for iBar == 1, treat as non-ISB and set lastDB to current
            lastDBHigh = nHigh
            lastDBLow = nLow
            ISB = 0

        # OSB detection: compare to immediate previous bar highs/lows
        prev_high = float(df.loc[iBar - 1, "high"])
        prev_low = float(df.loc[iBar - 1, "low"])
        if (ISB == 0) and (nHigh > prev_high) and (nLow < prev_low):
            OSB = 1
        else:
            OSB = 0

        # DB detection
        if (OSB == 0) and (ISB == 0):
            DB = 1
        else:
            DB = 0

        # DBU / DBD
        DBU = 1 if (DB == 1 and nHigh > prevLastDBHigh) else 0
        DBD = 1 if (DB == 1 and nLow < prevLastDBLow) else 0

        # bsNonISB
        if ISB == 0:
            bsNonISB[iBar] = 0
        else:
            bsNonISB[iBar] = bsNonISB[iBar - 1] + 1

        # bsDB
        if DB == 1:
            bsDB = 0
        else:
            bsDB = prevbsDB + 1

        # increment counters as in C
        ssc03 += 1
        ssc13 += 1

        # OSB unresolved handling
        if (
            (unresolvedOSBbreaks[iBar - 1] == 0)
            and (OSB == 1)
            and (
                ((ssc01 == -1) and (nHigh > ssc12)) or ((ssc01 == 1) and (nLow < ssc12))
            )
        ):
            osbbpsHigh[iBar] = nHigh
            osbbpsLow[iBar] = nLow
            unresolvedOSBbreaks[iBar] = 1
            OSBbreaksPrevSwing[iBar] = 1

        elif unresolvedOSBbreaks[iBar - 1] > 0:
            # chain of unresolved OSBs continues or resets
            if DB == 1:
                unresolvedOSBbreaks[iBar] = 0
            elif (nHigh > osbbpsHigh[iBar - 1]) and (nLow < osbbpsLow[iBar - 1]):
                osbbpsHigh[iBar] = nHigh
                osbbpsLow[iBar] = nLow
                unresolvedOSBbreaks[iBar] = unresolvedOSBbreaks[iBar - 1] + 1
                OSBbreaksPrevSwing[iBar] = 1
            else:
                unresolvedOSBbreaks[iBar] = unresolvedOSBbreaks[iBar - 1]
                osbbpsHigh[iBar] = osbbpsHigh[iBar - 1]
                osbbpsLow[iBar] = osbbpsLow[iBar - 1]

        # If DB and there were unresolved OSB breaks previously -> complex backtracking logic
        if (DB == 1) and (unresolvedOSBbreaks[iBar - 1] > 0):
            tempSwingDir = 1 if DBU == 1 else -1

            # tempBarsBack loop (1 .. ssc13+1 in C)
            tempBarsBack = 1
            # We'll search up to ssc13+1 bars back, but don't exceed iBar-1
            max_search = min(ssc13 + 1, iBar - 1)  # safe cap
            while tempBarsBack <= max_search:
                idx = iBar - tempBarsBack
                # read conditions (guard indices)
                cond1 = OSBbreaksPrevSwing[idx] == 1
                cond2 = False
                cond3 = False
                if tempSwingDir == 1:
                    # osbbpsLow[idx] > (pBar - tempBarsBack)->nLow
                    cond2 = osbbpsLow[idx] > float(df.loc[idx, "low"])
                else:
                    cond3 = osbbpsHigh[idx] < float(df.loc[idx, "high"])

                if cond1 or cond2 or cond3:
                    tempSwingDir *= -1

                # break if unresolvedOSBbreaks at iBar - (tempBarsBack + 1) == 0
                check_idx = iBar - (tempBarsBack + 1)
                if check_idx >= 0:
                    if unresolvedOSBbreaks[check_idx] == 0:
                        break
                else:
                    break

                tempBarsBack += 1

            # compute comparison index used in next steps (iBar - (tempBarsBack + 1))
            comp_index = iBar - (tempBarsBack + 1)
            if comp_index < 0:
                comp_index = 0

            # compare to sscDir at that index (guard)
            prior_sscDir = sscDir[comp_index] if comp_index < n else 0
            if tempSwingDir != prior_sscDir:
                # set last confirmed values ssc11, ssc12, ssc13 and write into ssc_confirmed
                ssc11 = -1 * tempSwingDir
                # ssc13 = sscBB[iBar - (tempBarsBack + 1)] + tempBarsBack + 1
                bb_index = iBar - (tempBarsBack + 1)
                if bb_index >= 0:
                    ssc13 = int(sscBB[bb_index]) + tempBarsBack + 1
                else:
                    ssc13 = tempBarsBack + 1

                if ssc11 == 1:
                    write_index = iBar - ssc13
                    if write_index >= 0:
                        ssc12 = float(df.loc[write_index, "high"])
                        ssc_confirmed[write_index] = ssc12
                else:
                    write_index = iBar - ssc13
                    if write_index >= 0:
                        ssc12 = float(df.loc[write_index, "low"])
                        ssc_confirmed[write_index] = ssc12

            # update ssc01
            ssc01 = tempSwingDir

            # compute ssc02 using historical sscValArr at bb_index
            bb_index = iBar - (tempBarsBack + 1)
            if bb_index >= 0:
                prev_ssc_val = sscValArr[bb_index]
            else:
                prev_ssc_val = 0.0

            if tempSwingDir == 1:
                ssc02 = max(prev_ssc_val, float(df.loc[iBar - tempBarsBack, "high"]))
            else:
                ssc02 = min(prev_ssc_val, float(df.loc[iBar - tempBarsBack, "low"]))

            # update ssc03
            if tempSwingDir == 1:
                compare_val = float(df.loc[iBar - tempBarsBack, "high"])
            else:
                compare_val = float(df.loc[iBar - tempBarsBack, "low"])

            if ssc02 == compare_val:
                ssc03 = tempBarsBack
            else:
                if bb_index >= 0:
                    ssc03 = int(tempBarsBack + sscBB[bb_index] + 1)
                else:
                    ssc03 = tempBarsBack + 1

            tempBarsBack -= 1

            # inner while loop: while tempBarsBack > 0
            while tempBarsBack > 0:
                idx = iBar - tempBarsBack
                # if OSBbreaksPrevSwing[idx] == 1 or conditional comparing pBar-tempBarsBack lows/highs with osbbps
                condA = OSBbreaksPrevSwing[idx] == 1
                condB = False
                condC = False
                if ssc01 == 1:
                    condB = float(df.loc[idx, "low"]) < osbbpsLow[idx]
                elif ssc01 == -1:
                    condC = float(df.loc[idx, "high"]) > osbbpsHigh[idx]

                if condA or condB or condC:
                    # commit current unconfirmed as confirmed
                    ssc11 = ssc01
                    ssc12 = ssc02
                    ssc13 = ssc03
                    write_index = iBar - ssc13
                    if write_index >= 0:
                        ssc_confirmed[write_index] = ssc12

                    # flip ssc01 and set new ssc02, ssc03
                    ssc01 = -1 * ssc11
                    if ssc01 == 1:
                        ssc02 = float(df.loc[idx, "high"])
                    else:
                        ssc02 = float(df.loc[idx, "low"])
                    ssc03 = tempBarsBack
                tempBarsBack -= 1

        # OSB and not OSBbreaksPrevSwing: extend unconfirmed if matching direction
        if (OSB == 1) and (OSBbreaksPrevSwing[iBar] == 0):
            if ssc01 == 1:
                if nHigh >= ssc02:
                    ssc02 = nHigh
                    ssc03 = 0
            elif ssc01 == -1:
                if nLow <= ssc02:
                    ssc02 = nLow
                    ssc03 = 0

        # DB handling
        if DB == 1:
            if ssc01 == 1:
                if DBU == 1:
                    # if DBU and new high improved
                    # The C code uses a condition with prevbsDB and comparisons; approximate exact behavior:
                    if nHigh > float(df.loc[iBar - prevbsDB - 1, "high"]):
                        ssc02 = max(ssc02, nHigh)
                        ssc03 = (sscBB[iBar - 1] + 1) if iBar - 1 >= 0 else 0
                    else:
                        # commit current unconfirmed to confirmed
                        ssc11 = ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        write_index = iBar - ssc13
                        if write_index >= 0:
                            ssc_confirmed[write_index] = ssc12

                        # flip and search prevLastDBLow occurrence to set ssc03
                        ssc01 = -1
                        ssc02 = prevLastDBLow
                        # find bars back where low equals prevLastDBLow but previous bar low is different
                        found = False
                        for tb in range(0, prevbsDB + 1):
                            idx = iBar - tb
                            idx_prev = iBar - tb - 1
                            if idx >= 0 and idx_prev >= 0:
                                if (float(df.loc[idx, "low"]) == ssc02) and (
                                    float(df.loc[idx_prev, "low"]) != ssc02
                                ):
                                    ssc03 = tb
                                    found = True
                                    break
                        if not found:
                            ssc03 = 0

                        # commit again and set new unconfirmed as upward
                        ssc11 = -1 * ssc01  # note: matches C sequence
                        ssc12 = ssc02
                        ssc13 = ssc03
                        write_index = iBar - ssc13
                        if write_index >= 0:
                            ssc_confirmed[write_index] = ssc12

                        ssc01 = 1
                        ssc02 = nHigh
                        ssc03 = 0
                else:
                    # DBU == 0 -> check DBD
                    if DBD == 1:
                        ssc11 = ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        write_index = iBar - ssc13
                        if write_index >= 0:
                            ssc_confirmed[write_index] = ssc12
                        ssc01 = -1
                        ssc02 = nLow
                        ssc03 = 0

            elif ssc01 == -1:
                if DBD == 1:
                    if nLow < float(df.loc[iBar - prevbsDB - 1, "low"]):
                        ssc02 = min(ssc02, nLow)
                        ssc03 = (sscBB[iBar - 1] + 1) if iBar - 1 >= 0 else 0
                    else:
                        ssc11 = ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        write_index = iBar - ssc13
                        if write_index >= 0:
                            ssc_confirmed[write_index] = ssc12

                        ssc01 = 1
                        ssc02 = prevLastDBHigh
                        found = False
                        for tb in range(0, prevbsDB + 1):
                            idx = iBar - tb
                            idx_prev = iBar - tb - 1
                            if idx >= 0 and idx_prev >= 0:
                                if (float(df.loc[idx, "high"]) == ssc02) and (
                                    float(df.loc[idx_prev, "high"]) != ssc02
                                ):
                                    ssc03 = tb
                                    found = True
                                    break
                        if not found:
                            ssc03 = 0

                        ssc11 = ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        write_index = iBar - ssc13
                        if write_index >= 0:
                            ssc_confirmed[write_index] = ssc12

                        ssc01 = -1
                        ssc02 = nLow
                        ssc03 = 0

                else:
                    if DBU == 1:
                        ssc11 = ssc01
                        ssc12 = ssc02
                        ssc13 = ssc03
                        write_index = iBar - ssc13
                        if write_index >= 0:
                            ssc_confirmed[write_index] = ssc12
                        ssc01 = 1
                        ssc02 = nHigh
                        ssc03 = 0

            else:  # ssc01 == 0 initial state
                ssc03 = 0
                if DBU == 1:
                    ssc01 = 1
                    ssc02 = nHigh
                else:
                    ssc01 = -1
                    ssc02 = nLow

        # store per-bar arrays as C code does at end of loop
        sscDir[iBar] = ssc01
        sscValArr[iBar] = ssc02
        sscBB[iBar] = ssc03

    # add columns to df (aligning indices)
    df["ssc01"] = sscDir
    df["ssc02"] = sscValArr
    df["ssc03"] = sscBB
    df["ssc11"] = ssc11
    df["ssc12"] = ssc12
    df["ssc13"] = ssc13
    df["sscDir"] = sscDir
    df["sscVal"] = sscValArr
    df["sscBB"] = sscBB
    df["ssc_confirmed"] = ssc_confirmed

    return df


def main():
    st.set_page_config(page_title="SSC Data Viewer", layout="wide")
    symbol = "NIFTY"
    symbol = st.selectbox(
        "Select Symbol",
        ["CRUDEOIL", "GOLD", "SILVER", "NATURALGAS", "NIFTY", "BANKNIFTY"],
        index=0,
    )
    # df = eod(symbol, "2023-01-01", "2023-12-31")
    ohlc = eod(symbol, "2022-07-19", "2023-07-19")

    df = ohlc.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    # df = SSC_realtest(df)
    df = SwingPoints2_realtest(df)
    df["swing_point"] = df["ssc_confirmed"]
    # df["swing"] = np.where((df["sscDir"] == 1), "swing-high", "")
    # df["swing"] = np.where((df["sscDir"] == -1), "swing-low", df["swing"])
    # df["bar_type"] = np.where(
    #     (df["sscBB"] == 0), "DB", np.where((df["sscBB"] > 0), "OSB", "ISB")
    # )
    # df["x"] = range(len(df))
    # df = df[:-120]

    # tops = [
    #     (row["x"], row["high"])
    #     for _, row in df[df["swing_point"] == df["high"]].iterrows()
    # ]

    # bottoms = [
    #     (row["x"], row["low"])
    #     for _, row in df[df["swing_point"] == df["low"]].iterrows()
    # ]

    # fig = plot_tv_ohlc_dark_v2(
    #     df,
    #     title="SSC RealTest Chart",
    #     tops=tops,
    #     bottoms=bottoms,
    # )
    # st.pyplot(fig)

    st.dataframe(df)

    df1 = ohlc.copy()
    df1["date"] = pd.to_datetime(df1["date"])
    df1.set_index("date", inplace=True)
    df1 = BarDefination(df1)
    df1 = SwingPoints2(df1)
    st.dataframe(df1)


if __name__ == "__main__":
    main()
