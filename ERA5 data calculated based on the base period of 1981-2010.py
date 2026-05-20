# -*- coding: utf-8 -*-
"""
NCC-grade recalculation for ERA5-Land JJA grid CSVs (one file per grid).
Fixes:
- DOY-based thresholds are per-row (each day has its own p10..p90)
- SM_decile must be computed row-wise (cannot use np.digitize with 2D bins)
- Preflight + failure logging

Input : E:\temp_csv_with_SM90\*.csv  (16653 files)
Output: E:\temp_csv_with_SM90_NCC_1981_2010_deciles\*.csv
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================
# Paths
# =========================
IN_DIR = Path(r"E:\temp_csv_with_SM90")
OUT_DIR = Path(r"E:\temp_csv_with_SM90_NCC_1981_2010_deciles")

# =========================
# Baseline + method
# =========================
BASELINE_START = 1981
BASELINE_END = 2010
DOY_WINDOW = 5
CONSEC_DAYS = 3

# =========================
# Column names in your CSV
# =========================
COL_DATE = "date"
COL_T = "temp_air"
COL_SM = "soil_moist"

# Soil moisture decile thresholds: p10..p90
SM_PCTS = list(range(10, 100, 10))  # [10,20,...,90]

# Parallel
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# IO
WRITE_FLOAT_FMT = "%.6f"
PREFLIGHT_N = 10  # first 10 files single-thread test


# -------------------------
# Utilities
# -------------------------
def read_csv_robust(path: Path) -> pd.DataFrame:
    """Try utf-8-sig then gbk (some Windows CSVs)."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="gbk", errors="ignore")


def parse_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, format="%Y/%m/%d", errors="coerce")
    if dt.isna().any():
        dt2 = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        dt = dt.fillna(dt2)
    if dt.isna().any():
        dt3 = pd.to_datetime(s, errors="coerce")
        dt = dt.fillna(dt3)
    return dt


def mark_runs_ge_k(x01: np.ndarray, k: int) -> np.ndarray:
    """Return 1 for elements belonging to runs of 1s with length >= k."""
    x = np.asarray(x01, dtype=np.int8)
    n = x.size
    if n == 0:
        return x
    padded = np.r_[0, x, 0]
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    out = np.zeros(n, dtype=np.int8)
    for st, en in zip(starts, ends):
        if (en - st) >= k:
            out[st:en] = 1
    return out


def compute_doy_thresholds(df_base: pd.DataFrame, doy_window: int):
    """
    Compute DOY-based thresholds (baseline-only):
      - T90 for temperature
      - SM10..SM90 for soil moisture (deciles)
    Return:
      - doy_map: dict doy -> (t90, sm10, sm20, ..., sm90)
      - fallback: same tuple for whole baseline season (in case missing doy)
    """
    base = df_base[[COL_T, COL_SM, "doy"]].copy()
    doys = np.sort(base["doy"].unique())
    doy_set = set(doys.tolist())

    t90_f = float(np.nanpercentile(base[COL_T].to_numpy(float), 90))
    sm_f = [float(np.nanpercentile(base[COL_SM].to_numpy(float), p)) for p in SM_PCTS]
    fallback = (t90_f, *sm_f)

    doy_map = {}
    for d in doys:
        window_doys = [dd for dd in range(int(d) - doy_window, int(d) + doy_window + 1) if dd in doy_set]
        sub = base[base["doy"].isin(window_doys)]
        if sub.empty:
            doy_map[int(d)] = fallback
            continue
        t90 = float(np.nanpercentile(sub[COL_T].to_numpy(float), 90))
        sm_vals = [float(np.nanpercentile(sub[COL_SM].to_numpy(float), p)) for p in SM_PCTS]
        doy_map[int(d)] = (t90, *sm_vals)

    return doy_map, fallback


def soil_decile_rowwise(sm: np.ndarray, p10_p90_rowwise: np.ndarray, fallback_bins_1d: np.ndarray) -> np.ndarray:
    """
    Row-wise decile classification for DOY-based thresholds.

    sm: (n,)
    p10_p90_rowwise: (n,9) where columns are [p10, p20, ..., p90] for each row/day.
    fallback_bins_1d: (9,) fallback thresholds if some row has NaNs.

    Decile definition (same as np.digitize with right=False):
      1: < p10
      2: [p10, p20)
      ...
      10: >= p90

    Implemented as:
      decile = 1 + sum(sm >= thresholds)
    """
    sm = np.asarray(sm, dtype=float).reshape(-1)
    bins = np.asarray(p10_p90_rowwise, dtype=float)

    if bins.ndim != 2 or bins.shape[1] != 9 or bins.shape[0] != sm.shape[0]:
        raise ValueError(f"bins shape must be (n,9). Got {bins.shape}, sm={sm.shape}")

    # Replace rows with any non-finite threshold using fallback bins
    bad = ~np.isfinite(bins).all(axis=1)
    if np.any(bad):
        bins[bad, :] = fallback_bins_1d

    # Vectorized compare: (n,9) boolean -> count -> 0..9 -> +1 => 1..10
    dec = 1 + (sm[:, None] >= bins).sum(axis=1)
    dec = np.clip(dec, 1, 10).astype(np.int8)
    return dec


# -------------------------
# One-file processing
# -------------------------
def process_one_file(csv_path: str):
    p = Path(csv_path)
    try:
        df = read_csv_robust(p)
        df.columns = [c.strip() for c in df.columns]

        for c in [COL_DATE, COL_T, COL_SM]:
            if c not in df.columns:
                return (p.name, False, f"Missing required column: {c}. Columns={df.columns.tolist()[:30]}")

        df[COL_DATE] = parse_date_series(df[COL_DATE].astype(str))
        df = df.dropna(subset=[COL_DATE]).copy()
        df = df.sort_values(COL_DATE).reset_index(drop=True)

        df["year"] = df[COL_DATE].dt.year.astype(int)
        df["month"] = df[COL_DATE].dt.month.astype(int)
        df["doy"] = df[COL_DATE].dt.dayofyear.astype(int)

        # keep only JJA
        df = df[df["month"].between(6, 8)].copy().reset_index(drop=True)

        # rename old columns
        for col in ["T90", "SM10", "SM90", "heatwave", "dry_lag1"]:
            if col in df.columns and f"{col}_old" not in df.columns:
                df.rename(columns={col: f"{col}_old"}, inplace=True)

        # baseline subset
        df_base = df[(df["year"] >= BASELINE_START) & (df["year"] <= BASELINE_END)].copy()
        if df_base.shape[0] < 500:
            return (p.name, False, f"Too few baseline rows: {df_base.shape[0]}")

        doy_map, fallback = compute_doy_thresholds(df_base, DOY_WINDOW)
        fallback_bins_1d = np.asarray(fallback[1:], dtype=float)  # p10..p90

        # map thresholds for each row
        n = len(df)
        t90_arr = np.empty(n, dtype=float)
        sm_p_arr = np.empty((n, 9), dtype=float)  # p10..p90

        doys = df["doy"].to_numpy(int)
        for i, d in enumerate(doys):
            vals = doy_map.get(int(d), fallback)
            t90_arr[i] = vals[0]
            sm_p_arr[i, :] = np.asarray(vals[1:], dtype=float)

        # write new threshold columns
        df["T90"] = t90_arr
        df["SM10"] = sm_p_arr[:, 0]
        df["SM90"] = sm_p_arr[:, 8]
        # optional: SM20..SM80
        for j, pctl in enumerate(SM_PCTS[1:-1], start=1):
            df[f"SM{pctl}"] = sm_p_arr[:, j]

        # compute raw flags
        t = df[COL_T].to_numpy(float)
        sm = df[COL_SM].to_numpy(float)

        df["heat_raw"] = (t > df["T90"].to_numpy(float)).astype(np.int8)
        df["dry_raw"] = (sm < df["SM10"].to_numpy(float)).astype(np.int8)
        df["wet90_raw"] = (sm > df["SM90"].to_numpy(float)).astype(np.int8)

        # lag dryness
        df["dry_lag1"] = np.r_[0, df["dry_raw"].to_numpy(np.int8)[:-1]].astype(np.int8)

        # consecutive >=3 heat
        df["heat3"] = mark_runs_ge_k(df["heat_raw"].to_numpy(np.int8), CONSEC_DAYS).astype(np.int8)

        # labels
        df["DHW_sync3"] = (df["heat3"].to_numpy(np.int8) & df["dry_raw"].to_numpy(np.int8)).astype(np.int8)
        df["NDHW3"] = (df["heat3"].to_numpy(np.int8) & (sm > df["SM10"].to_numpy(float)).astype(np.int8)).astype(np.int8)
        df["DHW_predry3"] = (df["heat3"].to_numpy(np.int8) & df["dry_lag1"].to_numpy(np.int8)).astype(np.int8)
        df["WHW90_3"] = (df["heat3"].to_numpy(np.int8) & df["wet90_raw"].to_numpy(np.int8)).astype(np.int8)

        # FIXED: row-wise SM decile
        df["SM_decile"] = soil_decile_rowwise(sm, sm_p_arr, fallback_bins_1d)

        # write file
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / p.name
        df.to_csv(out_path, index=False, float_format=WRITE_FLOAT_FMT, encoding="utf-8")

        return (p.name, True, "ok")

    except Exception as e:
        return (p.name, False, f"{repr(e)}\n{traceback.format_exc(limit=2)}")


# -------------------------
# Main
# -------------------------
def main():
    if not IN_DIR.exists():
        print(f"[ERROR] IN_DIR not found: {IN_DIR}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fail_log = OUT_DIR / "failures_log.txt"
    with open(fail_log, "w", encoding="utf-8") as f:
        f.write("filename\tmessage\n")

    files = sorted(str(x) for x in IN_DIR.glob("*.csv"))
    if not files:
        print(f"[ERROR] No CSV files in {IN_DIR}")
        sys.exit(1)

    print(f"Files: {len(files)}")
    print(f"Baseline: {BASELINE_START}-{BASELINE_END} | DOY_WINDOW=±{DOY_WINDOW} | CONSEC_DAYS={CONSEC_DAYS}")
    print(f"Output: {OUT_DIR}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Failure log: {fail_log}")

    # Preflight
    print(f"\n[Preflight] testing first {PREFLIGHT_N} files in single thread...")
    for fp in files[:PREFLIGHT_N]:
        name, ok, msg = process_one_file(fp)
        print(f"  {name}: {'OK' if ok else 'FAIL'}")
        if not ok:
            print("  ---- error detail ----")
            print(msg)
            print("------------------------")
            print("Stop here: fix this first. After preflight OK, rerun for all files.")
            return

    # Parallel
    ok_cnt, fail_cnt = 0, 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(process_one_file, fp) for fp in files]
        for i, fut in enumerate(as_completed(futs), start=1):
            name, ok, msg = fut.result()
            if ok:
                ok_cnt += 1
            else:
                fail_cnt += 1
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(f"{name}\t{msg.replace(chr(10),' ')}\n")

            if i % 200 == 0:
                print(f"[{i}/{len(files)}] ok={ok_cnt} fail={fail_cnt}")

    print(f"\nDONE. ok={ok_cnt} fail={fail_cnt}")
    print(f"See failure details in: {fail_log}")


if __name__ == "__main__":
    main()
