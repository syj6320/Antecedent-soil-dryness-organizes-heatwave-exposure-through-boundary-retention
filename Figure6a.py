# -*- coding: utf-8 -*-
"""
Figure 5a only -- connected, moderate-size object-evolution schematic
====================================================================

Purpose
-------
Redraw only panel a:
    Object evolution and spatial-role definition

Key corrections relative to the previous version
------------------------------------------------
1. Do NOT choose the largest event. Large events make the panel unreadable.
2. Prefer events with:
       - moderate daily footprint size
       - high spatial connectedness
       - visible S1-S6 variation
       - non-trivial but not overwhelming advancing-front recruitment
3. Use ix/iy grid coordinates for plotting when available, so each grid cell
   is shown as a clean square and S1-S6 colours remain visible.
4. Do NOT read every CSV.
5. Do NOT use rglob or os.walk.
6. Do NOT require yearly summary files.

Expected input structure
------------------------
event_root/
    1950/
        event_1950_00001.csv
        event_1950_00002.csv
        ...
    1951/
        event_1951_00001.csv
        ...

Required columns in event CSV
-----------------------------
At minimum:
    date or doy
    S_bin or SM_decile
    lon/lat or longitude/latitude

Preferred:
    ix, iy
    heat3
    grid_id

Recommended run
---------------
python panel_a_code.py --max-event-id 500 --max-read-events 80

If you want fewer grids:
python panel_a_code.py --target-panel-cells 220 --max-panel-cells 520 --max-read-events 100

If you want one exact event:
python panel_a_code.py --event-csv "E:\\...\\1950\\event_1950_00001.csv"
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path as FilePath
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings("ignore")


# =============================================================================
# 1. DEFAULT PATHS
# =============================================================================

DEFAULT_EVENT_ROOT = FilePath(
    r"E:\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本"
    r"\events_cc3d_with_precip_H_LE_CAPE_IVTDIV_T850_WIND_RH"
)

DEFAULT_OUT_DIR = FilePath(
    r"D:\第二篇\第二篇最终20260407版本\最终20260428\最终版本代码"
    r"\Figure5a_object_evolution_template_NCC"
)

FIG_BASENAME = "Figure5a_object_evolution_template_NCC_connected_moderate"
DPI = 600


# =============================================================================
# 2. STYLE
# =============================================================================

STATE_ORDER = [1, 2, 3, 4, 5, 6]

STATE_COLORS = [
    "#9b5a08",  # S1
    "#c8872d",  # S2
    "#e3c77d",  # S3
    "#7fcdbb",  # S4
    "#2f9c95",  # S5
    "#006c5b",  # S6
]

STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NORM = BoundaryNorm(np.arange(0.5, 7.5, 1), STATE_CMAP.N)

FRONT_COLOR = "#f05a1a"
LOCAL_COLOR = "#1f1f1f"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


# =============================================================================
# 3. UTILITIES
# =============================================================================

def log(msg: str):
    print(msg, flush=True)


def ensure_dir(path: FilePath) -> FilePath:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "0"


def parse_candidate_years(s: str, year_min: int, year_max: int) -> List[int]:
    s = str(s).strip()
    if s.lower() in ["all", "range", "full"]:
        return list(range(int(year_min), int(year_max) + 1))

    years = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))

    return sorted(set(years))


def parse_state_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        out = (
            s.astype(str)
            .str.strip()
            .str.replace("S", "", regex=False)
            .str.replace("s", "", regex=False)
        )
        return pd.to_numeric(out, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def derive_sbin_from_sm_decile(sm: pd.Series) -> pd.Series:
    x = pd.to_numeric(sm, errors="coerce")
    finite = x[np.isfinite(x)]

    if finite.empty:
        return x * np.nan

    xmin = finite.min()
    xmax = finite.max()

    if xmin >= 1 and xmax <= 6:
        return x

    if xmax <= 10:
        return pd.Series(
            np.select(
                [
                    x <= 1,
                    (x > 1) & (x <= 3),
                    (x > 3) & (x <= 5),
                    (x > 5) & (x <= 7),
                    (x > 7) & (x <= 9),
                    x > 9,
                ],
                [1, 2, 3, 4, 5, 6],
                default=np.nan,
            ),
            index=x.index,
        )

    return pd.Series(
        np.select(
            [
                x < 10,
                (x >= 10) & (x < 30),
                (x >= 30) & (x < 50),
                (x >= 50) & (x < 70),
                (x >= 70) & (x < 90),
                x >= 90,
            ],
            [1, 2, 3, 4, 5, 6],
            default=np.nan,
        ),
        index=x.index,
    )


def dominant_state(values: pd.Series) -> float:
    x = parse_state_series(values).dropna()
    x = x[x.between(1, 6)]
    if x.empty:
        return np.nan
    return float(x.astype(int).value_counts().idxmax())


# =============================================================================
# 4. READ ONE EVENT CSV
# =============================================================================

EVENT_USECOLS = {
    "date", "year", "month", "doy",
    "grid_id", "event_id",
    "S_bin", "SM_decile",
    "lon", "lat", "longitude", "latitude",
    "heat3", "heat_raw",
    "ix", "iy",
    "day_key", "coord_key",
}


def event_usecols_filter(c):
    return str(c).strip() in EVENT_USECOLS


def read_csv_robust(fp: FilePath) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]
    seps = [",", "\t", None]

    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                if sep is None:
                    df = pd.read_csv(
                        fp,
                        sep=None,
                        engine="python",
                        encoding=enc,
                        usecols=event_usecols_filter,
                    )
                else:
                    df = pd.read_csv(
                        fp,
                        sep=sep,
                        encoding=enc,
                        usecols=event_usecols_filter,
                        low_memory=False,
                    )

                if df.shape[1] >= 5:
                    return df

            except ValueError as e:
                last_err = e
                try:
                    if sep is None:
                        df = pd.read_csv(
                            fp,
                            sep=None,
                            engine="python",
                            encoding=enc,
                        )
                    else:
                        df = pd.read_csv(
                            fp,
                            sep=sep,
                            encoding=enc,
                            low_memory=False,
                        )
                    if df.shape[1] >= 5:
                        return df
                except Exception as e2:
                    last_err = e2

            except Exception as e:
                last_err = e

    raise RuntimeError(f"Could not read CSV:\n{fp}\nLast error: {last_err}")


def normalize_event_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input event CSV is empty.")

    d = df.rename(columns={c: str(c).strip() for c in df.columns}).copy()

    if "lon" not in d.columns and "longitude" in d.columns:
        d["lon"] = d["longitude"]

    if "lat" not in d.columns and "latitude" in d.columns:
        d["lat"] = d["latitude"]

    if "lon" not in d.columns or "lat" not in d.columns:
        raise ValueError("CSV must contain lon/lat or longitude/latitude.")

    d["lon"] = pd.to_numeric(d["lon"], errors="coerce")
    d["lat"] = pd.to_numeric(d["lat"], errors="coerce")

    if "S_bin" in d.columns:
        d["S_bin"] = parse_state_series(d["S_bin"])
    elif "SM_decile" in d.columns:
        d["S_bin"] = derive_sbin_from_sm_decile(d["SM_decile"])
    else:
        raise ValueError("CSV must contain S_bin or SM_decile.")

    if "heat3" in d.columns:
        d["heat3"] = pd.to_numeric(d["heat3"], errors="coerce")
        if (d["heat3"] == 1).any():
            d = d[d["heat3"] == 1].copy()

    if "day_key" in d.columns:
        d["day_key"] = pd.to_numeric(d["day_key"], errors="coerce")
    elif "doy" in d.columns:
        d["day_key"] = pd.to_numeric(d["doy"], errors="coerce")
    elif "date" in d.columns:
        dt = pd.to_datetime(d["date"], errors="coerce")
        if not dt.notna().any():
            raise ValueError("date column exists but cannot be parsed.")
        unique_dates = sorted(dt.dropna().unique())
        date_to_idx = {v: i + 1 for i, v in enumerate(unique_dates)}
        d["day_key"] = dt.map(date_to_idx)
    else:
        raise ValueError("CSV must contain day_key, doy, or date.")

    if "ix" in d.columns:
        d["ix"] = pd.to_numeric(d["ix"], errors="coerce")
    if "iy" in d.columns:
        d["iy"] = pd.to_numeric(d["iy"], errors="coerce")

    d = d.dropna(subset=["lon", "lat", "S_bin", "day_key"]).copy()
    d = d[d["S_bin"].between(1, 6)].copy()

    if d.empty:
        raise ValueError("No valid heatwave cells after normalization.")

    d["S_bin"] = d["S_bin"].astype(int)
    d["day_key"] = d["day_key"].astype(int)

    if "grid_id" in d.columns:
        d["grid_id"] = d["grid_id"].astype(str)

    if "ix" in d.columns and "iy" in d.columns and d["ix"].notna().any() and d["iy"].notna().any():
        d["ix"] = d["ix"].round().astype("Int64")
        d["iy"] = d["iy"].round().astype("Int64")
        d["coord_key"] = d["ix"].astype(str) + "_" + d["iy"].astype(str)
    elif "grid_id" in d.columns:
        d["coord_key"] = d["grid_id"].astype(str)
    elif "coord_key" not in d.columns:
        d["coord_key"] = (
            d["lon"].round(5).astype(str) + "_" + d["lat"].round(5).astype(str)
        )
    else:
        d["coord_key"] = d["coord_key"].astype(str)

    agg_dict = {
        "lon": ("lon", "mean"),
        "lat": ("lat", "mean"),
        "S_bin": ("S_bin", dominant_state),
    }

    for c in ["ix", "iy"]:
        if c in d.columns:
            agg_dict[c] = (c, "first")

    for c in ["date", "year", "month", "doy", "event_id", "grid_id"]:
        if c in d.columns:
            agg_dict[c] = (c, "first")

    d2 = (
        d.groupby(["day_key", "coord_key"], as_index=False)
        .agg(**agg_dict)
    )

    d2 = d2.dropna(subset=["lon", "lat", "S_bin", "day_key"]).copy()
    d2["S_bin"] = d2["S_bin"].astype(int)
    d2["day_key"] = d2["day_key"].astype(int)

    if "ix" in d2.columns and "iy" in d2.columns:
        if d2["ix"].notna().any() and d2["iy"].notna().any():
            d2["ix"] = pd.to_numeric(d2["ix"], errors="coerce")
            d2["iy"] = pd.to_numeric(d2["iy"], errors="coerce")

    return d2


# =============================================================================
# 5. CONNECTEDNESS METRICS
# =============================================================================

def get_integer_xy(sub: pd.DataFrame) -> Optional[List[Tuple[int, int]]]:
    if "ix" in sub.columns and "iy" in sub.columns:
        ss = sub.dropna(subset=["ix", "iy"]).copy()
        if not ss.empty:
            return list(zip(ss["ix"].round().astype(int), ss["iy"].round().astype(int)))

    # fallback: build integer grid from lon/lat ranks
    if "lon" in sub.columns and "lat" in sub.columns:
        ss = sub.dropna(subset=["lon", "lat"]).copy()
        if ss.empty:
            return None

        lons = sorted(ss["lon"].round(5).unique())
        lats = sorted(ss["lat"].round(5).unique())
        lon_to_i = {v: i for i, v in enumerate(lons)}
        lat_to_j = {v: j for j, v in enumerate(lats)}

        return [
            (lon_to_i[round(r.lon, 5)], lat_to_j[round(r.lat, 5)])
            for r in ss.itertuples(index=False)
        ]

    return None


def connected_components_stats(sub: pd.DataFrame) -> Dict[str, float]:
    pts = get_integer_xy(sub)
    if pts is None or len(pts) == 0:
        return {
            "n": 0,
            "n_components": np.nan,
            "largest": np.nan,
            "largest_fraction": np.nan,
        }

    pts_set = set(pts)
    visited = set()
    comp_sizes = []

    neighbor_steps = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for p in pts_set:
        if p in visited:
            continue

        stack = [p]
        visited.add(p)
        size = 0

        while stack:
            q = stack.pop()
            size += 1
            qx, qy = q

            for dx, dy in neighbor_steps:
                nb = (qx + dx, qy + dy)
                if nb in pts_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        comp_sizes.append(size)

    largest = max(comp_sizes) if comp_sizes else 0
    n = len(pts_set)

    return {
        "n": float(n),
        "n_components": float(len(comp_sizes)),
        "largest": float(largest),
        "largest_fraction": float(largest / n) if n > 0 else np.nan,
    }


def panel_connectedness(panel_cells: pd.DataFrame) -> Dict[str, float]:
    fracs = []
    comps = []

    for k in [0, 1, 2]:
        sub = panel_cells[panel_cells["panel_day"] == k]
        st = connected_components_stats(sub)

        if np.isfinite(st["largest_fraction"]):
            fracs.append(st["largest_fraction"])
        if np.isfinite(st["n_components"]):
            comps.append(st["n_components"])

    return {
        "mean_largest_fraction": float(np.mean(fracs)) if fracs else np.nan,
        "min_largest_fraction": float(np.min(fracs)) if fracs else np.nan,
        "mean_components": float(np.mean(comps)) if comps else np.nan,
        "max_components": float(np.max(comps)) if comps else np.nan,
    }


# =============================================================================
# 6. CANDIDATE SELECTION WITHOUT READING EVERY CSV
# =============================================================================

def build_candidate_files_by_pattern(
    event_root: FilePath,
    years: List[int],
    max_event_id: int,
    max_candidates: int,
    min_size_kb: float,
    max_size_kb: float,
    target_size_kb: float,
) -> List[FilePath]:
    """
    Checks filename pattern only:
        <event_root>/<year>/event_<year>_<event_id:05d>.csv

    It checks existence and file size only.
    It does not read all event CSV files.
    """
    found = []

    for year in years:
        year_dir = event_root / str(year)

        for event_id in range(1, int(max_event_id) + 1):
            fp = year_dir / f"event_{year}_{event_id:05d}.csv"

            if fp.exists():
                try:
                    size_kb = fp.stat().st_size / 1024.0
                except Exception:
                    size_kb = np.nan

                if np.isfinite(size_kb):
                    found.append((size_kb, year, event_id, fp))

    if not found:
        return []

    in_band = [
        x for x in found
        if float(min_size_kb) <= x[0] <= float(max_size_kb)
    ]

    if not in_band:
        log("[WARN] No files in the requested size band. Falling back to all existing candidates.")
        in_band = found

    # Prefer moderate-size events, not the largest events.
    def keyfun(x):
        size_kb = max(x[0], 1e-6)
        return abs(np.log(size_kb / max(float(target_size_kb), 1e-6)))

    in_band = sorted(in_band, key=keyfun)

    files = [x[3] for x in in_band[:int(max_candidates)]]

    return files


def select_three_days(
    d: pd.DataFrame,
    day_start: Optional[int],
    min_panel_cells: int,
    max_panel_cells: int,
    target_panel_cells: int,
) -> List[int]:
    days = np.array(sorted(d["day_key"].unique()))

    if len(days) < 3:
        raise ValueError(f"Only {len(days)} active day(s). At least 3 are required.")

    if day_start is not None:
        valid = days[days >= int(day_start)]
        if len(valid) < 3:
            raise ValueError(f"day_start={day_start} leaves fewer than 3 active days.")
        return [int(valid[0]), int(valid[1]), int(valid[2])]

    day_sets = {
        int(day): set(d.loc[d["day_key"] == day, "coord_key"].astype(str))
        for day in days
    }

    best_score = -np.inf
    best_i = 0

    for i in range(0, len(days) - 2):
        d0, d1, d2 = int(days[i]), int(days[i + 1]), int(days[i + 2])

        n0 = len(day_sets[d0])
        n1 = len(day_sets[d1])
        n2 = len(day_sets[d2])

        f1 = len(day_sets[d1] - day_sets[d0])
        f2 = len(day_sets[d2] - day_sets[d1])

        mean_n = np.mean([n0, n1, n2])
        total_front = f1 + f2
        front_fraction = total_front / max(n1 + n2, 1)

        # Reject very tiny or very large windows softly.
        size_penalty = abs(np.log((mean_n + 1.0) / max(float(target_panel_cells), 1.0)))

        # Prefer some expansion, but not an explosion of newly recruited cells.
        front_target = 0.18
        front_penalty = abs(front_fraction - front_target)

        score = (
            2.0 * np.log1p(total_front)
            - 2.2 * size_penalty
            - 3.0 * front_penalty
        )

        if mean_n < min_panel_cells:
            score -= 8.0
        if mean_n > max_panel_cells:
            score -= 10.0 + 0.01 * (mean_n - max_panel_cells)

        if score > best_score:
            best_score = score
            best_i = i

    return [int(days[best_i]), int(days[best_i + 1]), int(days[best_i + 2])]


def build_panel_cells(d: pd.DataFrame, selected_days: List[int]) -> pd.DataFrame:
    out = []
    previous_set = None
    labels = ["t", "t+1", "t+2"]

    for k, day in enumerate(selected_days):
        sub = d[d["day_key"] == day].copy()
        current_set = set(sub["coord_key"].astype(str))

        sub["panel_day"] = k
        sub["panel_label"] = labels[k]

        if k == 0 or previous_set is None:
            sub["role"] = "initial"
            sub["is_front"] = False
            sub["is_local"] = False
        else:
            is_front = ~sub["coord_key"].astype(str).isin(previous_set)
            sub["is_front"] = is_front
            sub["is_local"] = ~is_front
            sub["role"] = np.where(is_front, "front", "local")

        out.append(sub)
        previous_set = current_set

    return pd.concat(out, ignore_index=True)


def state_diversity(panel_cells: pd.DataFrame) -> float:
    p = panel_cells["S_bin"].value_counts(normalize=True).sort_index()
    return float(-np.sum(p * np.log(p + 1e-12)))


def score_panel_event(
    panel_cells: pd.DataFrame,
    target_panel_cells: int,
    max_panel_cells: int,
) -> float:
    ns = []
    for k in [0, 1, 2]:
        ns.append(len(panel_cells[panel_cells["panel_day"] == k]))

    mean_n = float(np.mean(ns))
    max_n = float(np.max(ns))

    n_front = int((panel_cells["role"] == "front").sum())
    n_local = int((panel_cells["role"] == "local").sum())
    front_fraction = n_front / max(n_front + n_local, 1)

    conn = panel_connectedness(panel_cells)

    div = state_diversity(panel_cells)

    x = panel_cells["plot_x"]
    y = panel_cells["plot_y"]
    aspect = (x.max() - x.min() + 1) / max((y.max() - y.min() + 1), 1e-6)

    size_penalty = abs(np.log((mean_n + 1.0) / max(float(target_panel_cells), 1.0)))
    too_large_penalty = max(0.0, max_n - max_panel_cells) / max(max_panel_cells, 1)
    aspect_penalty = abs(np.log(max(aspect, 1e-6) / 1.75))
    front_penalty = abs(front_fraction - 0.18)

    mean_largest = conn["mean_largest_fraction"]
    min_largest = conn["min_largest_fraction"]
    max_components = conn["max_components"]

    if not np.isfinite(mean_largest):
        mean_largest = 0.0
    if not np.isfinite(min_largest):
        min_largest = 0.0
    if not np.isfinite(max_components):
        max_components = 99.0

    score = (
        7.0 * mean_largest
        + 4.0 * min_largest
        + 2.0 * div
        + 1.3 * np.log1p(n_front)
        - 3.2 * size_penalty
        - 7.0 * too_large_penalty
        - 1.2 * aspect_penalty
        - 3.0 * front_penalty
        - 0.20 * max_components
    )

    # Strongly reject events that are visually too fragmented.
    if min_largest < 0.55:
        score -= 8.0
    if max_components > 12:
        score -= 4.0

    return float(score)


def add_plot_coordinates(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()

    if "ix" in d.columns and "iy" in d.columns:
        if d["ix"].notna().any() and d["iy"].notna().any():
            d["plot_x"] = pd.to_numeric(d["ix"], errors="coerce")
            d["plot_y"] = pd.to_numeric(d["iy"], errors="coerce")
            return d

    # fallback: use ranked lon/lat grid
    lons = sorted(d["lon"].round(5).unique())
    lats = sorted(d["lat"].round(5).unique())

    lon_to_i = {v: i for i, v in enumerate(lons)}
    lat_to_j = {v: j for j, v in enumerate(lats)}

    d["plot_x"] = d["lon"].round(5).map(lon_to_i)
    d["plot_y"] = d["lat"].round(5).map(lat_to_j)

    return d


def choose_event_without_scanning(args):
    event_root = FilePath(args.event_root)
    years = parse_candidate_years(args.candidate_years, args.year_min, args.year_max)

    log(f"[INFO] Candidate years: {years[0]}-{years[-1]} ({len(years)} years)")
    log(f"[INFO] Checking only filename pattern event_YYYY_00001 to event_YYYY_{args.max_event_id:05d}")
    log("[INFO] No rglob. No os.walk. No reading every CSV.")

    candidates = build_candidate_files_by_pattern(
        event_root=event_root,
        years=years,
        max_event_id=args.max_event_id,
        max_candidates=args.max_candidates,
        min_size_kb=args.min_size_kb,
        max_size_kb=args.max_size_kb,
        target_size_kb=args.target_size_kb,
    )

    if not candidates:
        raise FileNotFoundError(
            "\nNo candidate files found by filename pattern.\n"
            f"event_root      : {event_root}\n"
            f"candidate_years : {args.candidate_years}\n"
            f"max_event_id    : {args.max_event_id}\n\n"
            "Try increasing --max-event-id or provide --event-csv directly."
        )

    log(f"[INFO] Candidate files found by pattern/size: {len(candidates)}")
    log(f"[INFO] Will read at most {args.max_read_events} event CSVs.")

    best = None
    best_score = -np.inf
    read_n = 0

    for fp in candidates[:int(args.max_read_events)]:
        read_n += 1

        try:
            raw = read_csv_robust(fp)
            d = normalize_event_df(raw)
            d = add_plot_coordinates(d)

            days = select_three_days(
                d,
                day_start=args.day_start,
                min_panel_cells=args.min_panel_cells,
                max_panel_cells=args.max_panel_cells,
                target_panel_cells=args.target_panel_cells,
            )

            panel_cells = build_panel_cells(d, days)
            panel_cells = add_plot_coordinates(panel_cells)

            sc = score_panel_event(
                panel_cells,
                target_panel_cells=args.target_panel_cells,
                max_panel_cells=args.max_panel_cells,
            )

            conn = panel_connectedness(panel_cells)
            ns = [
                len(panel_cells[panel_cells["panel_day"] == k])
                for k in [0, 1, 2]
            ]

            log(
                f"[CHECK] {read_n:02d}/{args.max_read_events}: "
                f"score={sc:.2f}; days={days}; "
                f"N={ns}; min_conn={conn['min_largest_fraction']:.2f}; "
                f"max_comp={conn['max_components']:.0f}; file={fp}"
            )

            if sc > best_score:
                best_score = sc
                best = (fp, panel_cells, days, conn, ns)

        except Exception as e:
            log(f"[SKIP] {fp} | {e}")
            continue

    if best is None:
        raise RuntimeError("Candidates were found, but none could be plotted.")

    fp, panel_cells, days, conn, ns = best
    log(f"[INFO] Selected event score : {best_score:.2f}")
    log(f"[INFO] Selected panel N     : {ns}")
    log(f"[INFO] Selected connectedness: {conn}")

    return fp, panel_cells, days


# =============================================================================
# 7. PLOTTING HELPERS
# =============================================================================

def compute_extent(panel_cells: pd.DataFrame, target_aspect: float = 1.72):
    x = panel_cells["plot_x"].astype(float)
    y = panel_cells["plot_y"].astype(float)

    xmin, xmax = np.nanquantile(x, [0.005, 0.995])
    ymin, ymax = np.nanquantile(y, [0.005, 0.995])

    dx = max(float(xmax - xmin), 1.0)
    dy = max(float(ymax - ymin), 1.0)

    pad_x = max(dx * 0.10, 2.0)
    pad_y = max(dy * 0.12, 2.0)

    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    dx = xmax - xmin
    dy = ymax - ymin

    current_aspect = dx / max(dy, 1e-6)

    if current_aspect < target_aspect:
        new_dx = target_aspect * dy
        extra = (new_dx - dx) / 2.0
        xmin -= extra
        xmax += extra
    elif current_aspect > target_aspect:
        new_dy = dx / target_aspect
        extra = (new_dy - dy) / 2.0
        ymin -= extra
        ymax += extra

    return xmin, xmax, ymin, ymax


def marker_size_for_panel(n: int) -> float:
    if n <= 80:
        return 95
    if n <= 150:
        return 75
    if n <= 250:
        return 58
    if n <= 400:
        return 44
    if n <= 650:
        return 33
    if n <= 900:
        return 24
    return 18


def linewidth_for_panel(n: int) -> Tuple[float, float, float]:
    if n <= 150:
        return 0.70, 1.45, 1.85
    if n <= 400:
        return 0.55, 1.20, 1.55
    if n <= 650:
        return 0.42, 1.00, 1.30
    if n <= 900:
        return 0.32, 0.82, 1.10
    return 0.24, 0.65, 0.92


def add_rounded_axis_box(ax, edgecolor="0.63", lw=1.05):
    for sp in ax.spines.values():
        sp.set_visible(False)

    patch = FancyBboxPatch(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=0,
        clip_on=False,
    )
    ax.add_patch(patch)


def add_fig_arrow(fig, ax_left, ax_right):
    p0 = ax_left.get_position()
    p1 = ax_right.get_position()

    x0 = p0.x1 + 0.012
    x1 = p1.x0 - 0.012
    y = (p0.y0 + p0.y1) / 2.0

    arrow = FancyArrowPatch(
        (x0, y),
        (x1, y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=1.55,
        color="0.30",
        shrinkA=0,
        shrinkB=0,
        zorder=10,
    )
    fig.add_artist(arrow)


def draw_soil_moisture_legend(fig):
    ax = fig.add_axes([0.24, 0.405, 0.56, 0.095])
    ax.set_axis_off()

    x0 = 0.22
    y0 = 0.42
    w = 0.080
    h = 0.27
    gap = 0.040

    ax.text(
        x0 - 0.045,
        y0 + h / 2,
        "dry",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=15,
        color="0.10",
    )

    for i, col in enumerate(STATE_COLORS):
        xi = x0 + i * (w + gap)

        ax.add_patch(
            Rectangle(
                (xi + 0.004, y0 - 0.012),
                w,
                h,
                transform=ax.transAxes,
                facecolor="0.70",
                edgecolor="none",
                alpha=0.22,
                clip_on=False,
                zorder=1,
            )
        )

        ax.add_patch(
            Rectangle(
                (xi, y0),
                w,
                h,
                transform=ax.transAxes,
                facecolor=col,
                edgecolor="white",
                linewidth=0.50,
                clip_on=False,
                zorder=2,
            )
        )

        ax.text(
            xi + w / 2,
            y0 - 0.100,
            f"S{i + 1}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14.5,
            color="0.05",
        )

    ax.text(
        x0 + 6 * (w + gap) - gap + 0.045,
        y0 + h / 2,
        "wet",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=15,
        color="0.10",
    )


def draw_cell_icon(ax, x, y, size, face, edge, lw=1.8, dashed=False):
    rect = Rectangle(
        (x - size / 2, y - size / 2),
        size,
        size,
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        clip_on=False,
    )
    ax.add_patch(rect)


def draw_role_box(ax, kind: str):
    ax.set_axis_off()

    if kind == "local":
        edge = LOCAL_COLOR
        title = "Interior/local: retained cells"
        box_lw = 1.20
    else:
        edge = FRONT_COLOR
        title = "Advancing front: newly recruited cells"
        box_lw = 1.35

    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.065),
            0.970,
            0.855,
            transform=ax.transAxes,
            boxstyle="round,pad=0.018,rounding_size=0.028",
            facecolor="white",
            edgecolor=edge,
            linewidth=box_lw,
            clip_on=False,
        )
    )

    ax.text(
        0.50,
        0.835,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14.2,
        fontweight="bold",
        color="0.04",
    )

    if kind == "local":
        draw_cell_icon(
            ax,
            0.170,
            0.555,
            0.075,
            face=STATE_COLORS[3],
            edge=LOCAL_COLOR,
            lw=1.9,
        )

        ax.annotate(
            "",
            xy=(0.405, 0.555),
            xytext=(0.295, 0.555),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.15,
                color="0.55",
                mutation_scale=18,
            ),
        )

        draw_cell_icon(
            ax,
            0.525,
            0.555,
            0.075,
            face=STATE_COLORS[3],
            edge=LOCAL_COLOR,
            lw=1.9,
        )

        ax.text(
            0.170,
            0.365,
            "at t\n(inside object)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11.2,
            color="0.05",
            linespacing=1.25,
        )

        ax.text(
            0.525,
            0.365,
            "at t+1\n(remains interior)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11.2,
            color="0.05",
            linespacing=1.25,
        )

        ax.text(
            0.705,
            0.485,
            "Cell remains within\nthe object interior.",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=12.0,
            color="0.05",
            linespacing=1.18,
        )

    else:
        draw_cell_icon(
            ax,
            0.170,
            0.555,
            0.075,
            face="white",
            edge="0.62",
            lw=1.8,
            dashed=True,
        )

        ax.annotate(
            "",
            xy=(0.405, 0.555),
            xytext=(0.295, 0.555),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.15,
                color="0.55",
                mutation_scale=18,
            ),
        )

        draw_cell_icon(
            ax,
            0.525,
            0.555,
            0.075,
            face=STATE_COLORS[2],
            edge=FRONT_COLOR,
            lw=2.10,
        )

        ax.text(
            0.170,
            0.365,
            "at t\n(outside object)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11.2,
            color="0.05",
            linespacing=1.25,
        )

        ax.text(
            0.525,
            0.365,
            "at t+1\n(joins object edge)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11.2,
            color="0.05",
            linespacing=1.25,
        )

        ax.text(
            0.705,
            0.485,
            "Cell newly joins the\nobject at the front.",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=12.0,
            color="0.05",
            linespacing=1.18,
        )


def plot_top_panel(ax, sub: pd.DataFrame, k: int, extent, show_counts: bool):
    ax.set_facecolor("white")
    add_rounded_axis_box(ax)

    xmin, xmax, ymin, ymax = extent

    n = len(sub)
    msize = marker_size_for_panel(n)
    white_lw, local_lw, front_lw = linewidth_for_panel(n)

    # Base layer: S1-S6 is always visible.
    ax.scatter(
        sub["plot_x"],
        sub["plot_y"],
        c=sub["S_bin"],
        cmap=STATE_CMAP,
        norm=STATE_NORM,
        s=msize,
        marker="s",
        linewidths=white_lw,
        edgecolors="white",
        alpha=0.985,
        rasterized=True,
        zorder=2,
    )

    if k > 0:
        local = sub[sub["role"] == "local"]
        front = sub[sub["role"] == "front"]

        # Thin local outline. Kept subtle so it does not cover S1-S6.
        if len(local) > 0:
            ax.scatter(
                local["plot_x"],
                local["plot_y"],
                s=msize * 1.04,
                marker="s",
                facecolors="none",
                edgecolors=LOCAL_COLOR,
                linewidths=local_lw,
                alpha=0.62,
                rasterized=True,
                zorder=5,
            )

        # Front outline. Orange but not over-thick.
        if len(front) > 0:
            ax.scatter(
                front["plot_x"],
                front["plot_y"],
                s=msize * 1.18,
                marker="s",
                facecolors="none",
                edgecolors=FRONT_COLOR,
                linewidths=front_lw,
                alpha=0.96,
                rasterized=True,
                zorder=6,
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        0.50,
        0.925,
        ["t", "t+1", "t+2"][k],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="0.02",
        zorder=20,
    )

    if show_counts and k > 0:
        front_n = int((sub["role"] == "front").sum())
        local_n = int((sub["role"] == "local").sum())

        ax.text(
            0.035,
            0.055,
            f"front +{fmt_int(front_n)}; local {fmt_int(local_n)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.4,
            color="0.15",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.72,
                pad=1.5,
            ),
            zorder=20,
        )


# =============================================================================
# 8. MAIN PLOT
# =============================================================================

def plot_figure(
    panel_cells: pd.DataFrame,
    source_csv: FilePath,
    out_dir: FilePath,
    show_counts: bool = False,
):
    ensure_dir(out_dir)

    selected_days = sorted(panel_cells["day_key"].unique())
    extent = compute_extent(panel_cells)

    fig = plt.figure(figsize=(13.8, 7.35))

    fig.text(
        0.016,
        0.955,
        "a",
        ha="left",
        va="top",
        fontsize=27,
        fontweight="bold",
        color="0.02",
    )

    fig.text(
        0.060,
        0.955,
        "Object evolution and spatial-role definition",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color="0.02",
    )

    top_y = 0.550
    top_h = 0.335
    top_w = 0.280
    x0 = 0.045
    x1 = 0.360
    x2 = 0.675

    ax0 = fig.add_axes([x0, top_y, top_w, top_h])
    ax1 = fig.add_axes([x1, top_y, top_w, top_h])
    ax2 = fig.add_axes([x2, top_y, top_w, top_h])

    top_axes = [ax0, ax1, ax2]

    for k, ax in enumerate(top_axes):
        sub = panel_cells[panel_cells["panel_day"] == k].copy()
        plot_top_panel(ax, sub, k, extent, show_counts=show_counts)

    add_fig_arrow(fig, ax0, ax1)
    add_fig_arrow(fig, ax1, ax2)

    fig.text(
        x0 + 0.002,
        top_y - 0.028,
        "initial footprint (soil–moisture state)",
        ha="left",
        va="top",
        fontsize=13.2,
        color="0.05",
    )

    draw_soil_moisture_legend(fig)

    ax_local = fig.add_axes([0.070, 0.090, 0.415, 0.250])
    ax_front = fig.add_axes([0.520, 0.090, 0.425, 0.250])

    draw_role_box(ax_local, "local")
    draw_role_box(ax_front, "front")

    png = out_dir / f"{FIG_BASENAME}.png"
    pdf = out_dir / f"{FIG_BASENAME}.pdf"

    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    selected_csv = out_dir / "Figure5a_selected_event_cells_connected_moderate.csv"
    panel_cells.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    conn = panel_connectedness(panel_cells)

    meta_fp = out_dir / "Figure5a_selected_event_metadata_connected_moderate.txt"
    with open(meta_fp, "w", encoding="utf-8") as f:
        f.write("Figure 5a selected-event metadata\n")
        f.write("=" * 80 + "\n")
        f.write(f"source_csv: {source_csv}\n")
        f.write(f"selected_days: {selected_days}\n")
        f.write(f"output_png: {png}\n")
        f.write(f"output_pdf: {pdf}\n")
        f.write(f"selected_cells_csv: {selected_csv}\n")
        f.write(f"n_cells_total_selected: {len(panel_cells)}\n")
        f.write(f"connectedness: {conn}\n\n")

        for k in [0, 1, 2]:
            ss = panel_cells[panel_cells["panel_day"] == k]
            if ss.empty:
                continue

            st = connected_components_stats(ss)

            f.write(
                f"panel {k} ({ss['panel_label'].iloc[0]}): "
                f"N={len(ss)}, "
                f"front={(ss['role'] == 'front').sum()}, "
                f"local={(ss['role'] == 'local').sum()}, "
                f"initial={(ss['role'] == 'initial').sum()}, "
                f"largest_component_fraction={st['largest_fraction']:.3f}, "
                f"n_components={st['n_components']:.0f}\n"
            )

    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")
    log(f"[SAVED] {selected_csv}")
    log(f"[SAVED] {meta_fp}")


# =============================================================================
# 9. ARGUMENTS AND MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--event-root",
        default=str(DEFAULT_EVENT_ROOT),
        help="Root directory containing yearly folders such as 1950, 1951, 1952.",
    )

    parser.add_argument(
        "--event-csv",
        default=None,
        help="Read this single event CSV only.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--candidate-years",
        default="1950-2024",
        help="Candidate years. Examples: 1950-2024, 1960-1980, 1950,1951,1952.",
    )

    parser.add_argument(
        "--year-min",
        default=1950,
        type=int,
        help="Minimum year used when --candidate-years is all.",
    )

    parser.add_argument(
        "--year-max",
        default=2024,
        type=int,
        help="Maximum year used when --candidate-years is all.",
    )

    parser.add_argument(
        "--max-event-id",
        default=500,
        type=int,
        help="Maximum event ID checked in each year by filename pattern.",
    )

    parser.add_argument(
        "--min-size-kb",
        default=35.0,
        type=float,
        help="Minimum file size for candidate events. Used only for selecting candidates.",
    )

    parser.add_argument(
        "--max-size-kb",
        default=950.0,
        type=float,
        help="Maximum file size for candidate events. Lower this if the object is still too dense.",
    )

    parser.add_argument(
        "--target-size-kb",
        default=280.0,
        type=float,
        help="Target file size for visually moderate events.",
    )

    parser.add_argument(
        "--max-candidates",
        default=180,
        type=int,
        help="Number of existing candidate files retained before reading.",
    )

    parser.add_argument(
        "--max-read-events",
        default=80,
        type=int,
        help="Maximum number of candidate event CSVs actually read.",
    )

    parser.add_argument(
        "--min-panel-cells",
        default=45,
        type=int,
        help="Soft lower bound for daily panel cells.",
    )

    parser.add_argument(
        "--target-panel-cells",
        default=260,
        type=int,
        help="Preferred mean number of cells per panel.",
    )

    parser.add_argument(
        "--max-panel-cells",
        default=620,
        type=int,
        help="Soft upper bound for daily panel cells. Lower this if the grids are still too dense.",
    )

    parser.add_argument(
        "--day-start",
        default=None,
        type=int,
        help="Optional starting day_key/doy for the three-day sequence.",
    )

    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Show front/local counts inside t+1 and t+2 panels.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(FilePath(args.output))

    log("=" * 100)
    log("[INFO] Rebuilding Figure 5a with connected, moderate-size event selection")
    log("[INFO] This script does NOT read every event CSV and does NOT use rglob.")
    log("[INFO] It prefers fewer grids, higher connectedness, and visible S1-S6 states.")
    log(f"[INFO] event root : {args.event_root}")
    log(f"[INFO] output     : {out_dir}")
    log("=" * 100)

    if args.event_csv:
        source_csv = FilePath(args.event_csv)
        if not source_csv.exists():
            raise FileNotFoundError(f"--event-csv does not exist:\n{source_csv}")

        log("[INFO] Single-event mode. Reading only this CSV:")
        log(f"[INFO] {source_csv}")

        raw = read_csv_robust(source_csv)
        event_df = normalize_event_df(raw)
        event_df = add_plot_coordinates(event_df)

        selected_days = select_three_days(
            event_df,
            day_start=args.day_start,
            min_panel_cells=args.min_panel_cells,
            max_panel_cells=args.max_panel_cells,
            target_panel_cells=args.target_panel_cells,
        )

        panel_cells = build_panel_cells(event_df, selected_days)
        panel_cells = add_plot_coordinates(panel_cells)

    else:
        log("[INFO] No --event-csv supplied.")
        log("[INFO] Using pattern-based limited event selection.")
        log("[INFO] It checks file existence/size and reads only limited candidates.")

        source_csv, panel_cells, selected_days = choose_event_without_scanning(args)

    log(f"[INFO] Selected source CSV: {source_csv}")
    log(f"[INFO] Selected days      : {selected_days}")

    plot_figure(
        panel_cells=panel_cells,
        source_csv=source_csv,
        out_dir=out_dir,
        show_counts=bool(args.show_counts),
    )

    log("=" * 100)
    log("[DONE] Figure 5a finished.")
    log(f"[DONE] Output directory: {out_dir}")
    log("=" * 100)


if __name__ == "__main__":
    main()