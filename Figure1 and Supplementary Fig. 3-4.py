# -*- coding: utf-8 -*-
"""
Rebuild Figure 1, Supplementary Fig. 1 and Supplementary Fig. 2
for journal-facing heatwave-object exposure analysis.

Main revisions in this version:
1. Spatial maps are redrawn in a cleaner Figure-3-like style.
2. Figure 1 map colorbars are placed in a compact dedicated middle band, shifted upward to sit closer to the maps and away from the lower panels.
3. All figure-level super titles are removed.
4. Font sizes are increased.
5. Figure 1 panel d is replaced by a cleaner state-by-metric trend heatmap with significance marks.
6. Supplementary Fig. 1 panel d is changed to an interpretable secondary-attribute trend heatmap.
7. Supplementary Fig. 1 panel f uses clear legend entries: Rolling threshold / Fixed threshold.
8. Supplementary Fig. 2 panel b includes median/IQR trend structure and Spearman statistics.
9. Supplementary Fig. 2 panel d is replaced by annual initial-state composition, avoiding uninformative or zero-inflated mobility boxplots.
10. Supplementary Fig. 2 panel e includes Spearman statistics and a fitted trend line.

Input roots are set for the user's current ERA5 data:

Rolling:
E:\\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本\\events_cc3d_with_precip_H_LE_CAPE_IVTDIV_T850_WIND_RH

Fixed:
E:\\temp_events_ERA5_S1S6_Nature所有数据版本\\events_cc3d_fixed_threshold
"""

import os
import re
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, LogLocator, ScalarFormatter

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


warnings.filterwarnings("ignore")


# =============================================================================
# 1. USER PATHS
# =============================================================================

ROLLING_ROOT = Path(
    r"E:\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本"
    r"\events_cc3d_with_precip_H_LE_CAPE_IVTDIV_T850_WIND_RH"
)

FIXED_ROOT = Path(
    r"E:\temp_events_ERA5_S1S6_Nature所有数据版本"
    r"\events_cc3d_fixed_threshold"
)

OUT_DIR = Path(r"E:\第二篇数据0427\NCC_rebuilt_Figure1_and_Supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(1950, 2025))
MOVING_THRESHOLD_KM = 200.0


# =============================================================================
# 2. STYLE
# =============================================================================

DPI = 500

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 22,
    "axes.titlesize": 25,
    "axes.labelsize": 24,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "legend.fontsize": 19,
    "axes.linewidth": 1.35,
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

STATE_COLORS = {
    1: "#8c510a",
    2: "#bf812d",
    3: "#dfc27d",
    4: "#80cdc1",
    5: "#35978f",
    6: "#01665e",
}

STATE_LABELS = {i: f"S{i}" for i in range(1, 7)}

DRY_CMAP = plt.get_cmap("YlOrRd")
DIVERGE_CMAP = plt.get_cmap("RdBu_r")
GREY_DARK = "#2f2f2f"
GREY_MID = "#7a7a7a"


# =============================================================================
# 3. GENERAL UTILITIES
# =============================================================================

def log(msg):
    print(msg, flush=True)


def p_to_star(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_p(p):
    if not np.isfinite(p):
        return "P=NA"
    if p < 0.001:
        return "P<0.001"
    return f"P={p:.3f}"


def safe_numeric(x):
    return pd.to_numeric(x, errors="coerce")


def detect_sep(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
    except Exception:
        return ","

    n_tab = first.count("\t")
    n_comma = first.count(",")
    if n_tab > n_comma:
        return "\t"
    return ","


def read_csv_auto(path):
    sep = detect_sep(path)
    try:
        return pd.read_csv(path, sep=sep, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df, candidates, required=True):
    if isinstance(candidates, str):
        candidates = [candidates]

    nmap = {norm_name(c): c for c in df.columns}

    for cand in candidates:
        key = norm_name(cand)
        if key in nmap:
            return nmap[key]

    for cand in candidates:
        key = norm_name(cand)
        for nk, orig in nmap.items():
            if key in nk or nk in key:
                return orig

    if required:
        raise KeyError(
            f"Cannot find column from candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def infer_year_from_path(path):
    m = re.search(r"(19[5-9][0-9]|20[0-2][0-9])", str(path))
    if m:
        return int(m.group(1))
    return np.nan


def infer_event_uid(path, fallback_event_id=None):
    stem = Path(path).stem
    year = infer_year_from_path(path)
    if pd.notna(year):
        return f"{int(year)}_{stem}"
    if fallback_event_id is not None:
        return f"{stem}_{fallback_event_id}"
    return stem


def lintrend_per_decade(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 8:
        return np.nan, np.nan, np.nan

    x = x[mask]
    y = y[mask]

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0, np.nan, np.nan

    if stats is not None:
        res = stats.linregress(x, y)
        return float(res.slope * 10.0), float(res.pvalue), float(res.rvalue)

    X = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    slope = beta[1] * 10.0
    r = np.corrcoef(x, y)[0, 1]
    return float(slope), np.nan, float(r)


def spearman_stat(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 8:
        return np.nan, np.nan

    if stats is not None:
        r, p = stats.spearmanr(x[mask], y[mask])
        return float(r), float(p)

    xr = pd.Series(x[mask]).rank().values
    yr = pd.Series(y[mask]).rank().values
    r = np.corrcoef(xr, yr)[0, 1]
    return float(r), np.nan


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def bootstrap_ci(values, func=np.nanmedian, n_boot=1000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 5:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(func(sample))

    boots = np.asarray(boots)
    return (
        float(func(values)),
        float(np.nanpercentile(boots, 2.5)),
        float(np.nanpercentile(boots, 97.5)),
    )


def add_panel_label(ax, letter, x=-0.13, y=1.08, size=28):
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=size,
        fontweight="bold",
        clip_on=False,
    )


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def apply_sci_y(ax, powerlimits=(4, 4)):
    """Use scientific notation on large linear y-axis tick values."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(powerlimits)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=powerlimits)
    ax.yaxis.get_offset_text().set_fontsize(15)


def apply_sci_x(ax, powerlimits=(4, 4)):
    """Use scientific notation on large linear x-axis tick values."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(powerlimits)
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="x", style="sci", scilimits=powerlimits)
    ax.xaxis.get_offset_text().set_fontsize(15)


def fmt_large_number(value, decimals=2, sci_threshold=1.0e4):
    """Compact label formatter for large legend/table numbers."""
    if not np.isfinite(value):
        return "NA"
    if abs(value) >= sci_threshold:
        s = f"{value:.{decimals}e}"
        s = s.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
        return s
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def fmt_trend_legend(value, p=None, decimals=2):
    return f"{fmt_large_number(value, decimals=decimals)} decade$^{{-1}}${p_to_star(p)}"


def save_figure(fig, name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")


# =============================================================================
# 4. EVENT READING AND CACHE BUILDING
# =============================================================================

def list_event_files(root):
    files = []

    for y in YEARS:
        ydir = root / str(y)
        if ydir.exists():
            files.extend(sorted(ydir.glob("*.csv")))
            files.extend(sorted(ydir.glob("*.txt")))

    if len(files) == 0:
        files = sorted(root.rglob("*.csv"))

    return files


def get_heat_rows(df):
    heat_col = find_col(
        df,
        ["heat3", "heat_raw", "is_heat_period_event", "is_heat_period_grid", "heatwave"],
        required=False
    )

    if heat_col is None:
        return df.copy()

    h = safe_numeric(df[heat_col])
    sub = df[h == 1].copy()

    if len(sub) == 0:
        return df.copy()

    return sub


def get_state_series(df):
    state_col = find_col(
        df,
        ["S_bin", "s_bin", "SM_decile", "sm_decile", "state", "sm_state", "soil_moisture_state"],
        required=False
    )

    if state_col is None:
        return None

    s = safe_numeric(df[state_col])

    if state_col.lower() in ["sm_decile", "smdecile"] or s.max(skipna=True) > 6:
        # Convert decile-like values to six regimes:
        # 1-2 -> S1, 3-4 -> S2, 5 -> S3, 6 -> S4, 7-8 -> S5, 9-10 -> S6
        dec = s.round()
        out = pd.Series(np.nan, index=df.index)
        out[(dec >= 1) & (dec <= 2)] = 1
        out[(dec >= 3) & (dec <= 4)] = 2
        out[dec == 5] = 3
        out[dec == 6] = 4
        out[(dec >= 7) & (dec <= 8)] = 5
        out[(dec >= 9) & (dec <= 10)] = 6
        return out

    out = s.round()
    out = out.where(out.between(1, 6))
    return out


def mode_state(s):
    s = pd.Series(s).dropna()
    s = s[s.between(1, 6)]
    if len(s) == 0:
        return np.nan
    return int(s.value_counts().idxmax())


def summarize_one_event(path, tag):
    diag = {
        "file": str(path),
        "tag": tag,
        "status": "unknown",
        "reason": "",
        "n_rows_raw": 0,
        "n_rows_heat": 0,
    }

    try:
        df = read_csv_auto(path)
    except Exception as e:
        diag["status"] = "failed"
        diag["reason"] = f"read_failed: {e}"
        return None, None, diag

    diag["n_rows_raw"] = len(df)

    if len(df) == 0:
        diag["status"] = "failed"
        diag["reason"] = "empty_file"
        return None, None, diag

    try:
        year_col = find_col(df, ["year"], required=False)
        date_col = find_col(df, ["date", "time"], required=False)
        doy_col = find_col(df, ["doy", "dayofyear"], required=False)

        lon_col = find_col(df, ["lon", "longitude", "x"], required=True)
        lat_col = find_col(df, ["lat", "latitude", "y"], required=True)

        grid_col = find_col(df, ["grid_id", "gridid", "coord_key"], required=False)
        event_col = find_col(df, ["event_id", "cc_label", "global_event_id"], required=False)

        temp_col = find_col(df, ["temp_air", "temperature", "t2m"], required=False)
        t90_col = find_col(df, ["T90", "t90"], required=False)

        dfh = get_heat_rows(df)
        diag["n_rows_heat"] = len(dfh)

        state = get_state_series(dfh)
        if state is None:
            diag["status"] = "failed"
            diag["reason"] = "missing_state_column"
            return None, None, diag

        dfh = dfh.copy()
        dfh["_state"] = state

        dfh = dfh[dfh["_state"].between(1, 6)].copy()
        if len(dfh) == 0:
            diag["status"] = "failed"
            diag["reason"] = "no_valid_s1_s6_rows"
            return None, None, diag

        dfh["_lon"] = safe_numeric(dfh[lon_col])
        dfh["_lat"] = safe_numeric(dfh[lat_col])
        dfh = dfh[np.isfinite(dfh["_lon"]) & np.isfinite(dfh["_lat"])].copy()

        if len(dfh) == 0:
            diag["status"] = "failed"
            diag["reason"] = "no_valid_lon_lat"
            return None, None, diag

        if year_col is not None:
            year = int(safe_numeric(dfh[year_col]).dropna().iloc[0])
        else:
            year = infer_year_from_path(path)

        if pd.isna(year):
            diag["status"] = "failed"
            diag["reason"] = "cannot_infer_year"
            return None, None, diag

        if date_col is not None:
            dt = pd.to_datetime(dfh[date_col], errors="coerce")
            dfh["_date"] = dt
        else:
            if doy_col is not None:
                doy = safe_numeric(dfh[doy_col]).fillna(1).astype(int)
                dfh["_date"] = pd.to_datetime(str(year), format="%Y") + pd.to_timedelta(doy - 1, unit="D")
            else:
                dfh["_date"] = pd.NaT

        if dfh["_date"].notna().any():
            first_date = dfh["_date"].min()
            start_rows = dfh[dfh["_date"] == first_date]
            start_state = mode_state(start_rows["_state"])
            duration = int(dfh["_date"].nunique())
        else:
            start_state = mode_state(dfh["_state"])
            duration = np.nan

        if not np.isfinite(start_state):
            diag["status"] = "failed"
            diag["reason"] = "cannot_infer_start_state"
            return None, None, diag

        if event_col is not None:
            event_id_file = str(dfh[event_col].iloc[0])
        else:
            event_id_file = Path(path).stem

        uid = infer_event_uid(path, fallback_event_id=event_id_file)

        if grid_col is not None:
            grid = dfh[grid_col].astype(str)
        else:
            grid = dfh["_lon"].round(5).astype(str) + "_" + dfh["_lat"].round(5).astype(str)

        dfh["_grid_id"] = grid
        dfh["_year"] = int(year)
        dfh["_start_state"] = int(start_state)

        event_grid_days = int(len(dfh))
        n_unique_grids = int(dfh["_grid_id"].nunique())

        if dfh["_date"].notna().any():
            daily_area = dfh.groupby("_date")["_grid_id"].nunique()
            max_area = float(daily_area.max())

            cent = (
                dfh.groupby("_date", as_index=False)
                .agg(lon=("_lon", "mean"), lat=("_lat", "mean"))
                .sort_values("_date")
            )
            if len(cent) >= 2:
                step = haversine_km(
                    cent["lon"].values[:-1],
                    cent["lat"].values[:-1],
                    cent["lon"].values[1:],
                    cent["lat"].values[1:]
                )
                path_length = float(np.nansum(step))
                net_disp = float(haversine_km(
                    cent["lon"].values[0],
                    cent["lat"].values[0],
                    cent["lon"].values[-1],
                    cent["lat"].values[-1]
                ))
                straightness = float(net_disp / path_length) if path_length > 0 else np.nan
            else:
                path_length = 0.0
                net_disp = 0.0
                straightness = np.nan
        else:
            max_area = np.nan
            path_length = np.nan
            net_disp = np.nan
            straightness = np.nan

        if temp_col is not None and t90_col is not None:
            heat_excess = safe_numeric(dfh[temp_col]) - safe_numeric(dfh[t90_col])
            heat_excess_mean = float(np.nanmean(heat_excess))
        else:
            heat_excess_mean = np.nan

        if doy_col is not None:
            doy_mean = float(np.nanmean(safe_numeric(dfh[doy_col])))
        else:
            doy_mean = np.nan

        ev = {
            "dataset": tag,
            "event_uid": uid,
            "file": str(path),
            "year": int(year),
            "start_state": int(start_state),
            "event_grid_days": event_grid_days,
            "n_unique_grids": n_unique_grids,
            "duration_days": duration,
            "max_area_gridcells": max_area,
            "heat_excess_mean": heat_excess_mean,
            "doy_mean": doy_mean,
            "path_length_km": path_length,
            "net_displacement_km": net_disp,
            "straightness": straightness,
            "moving_200km": 1 if np.isfinite(net_disp) and net_disp >= MOVING_THRESHOLD_KM else 0,
        }

        gy = (
            dfh.groupby(["_year", "_grid_id", "_lon", "_lat", "_start_state"], as_index=False)
            .size()
            .rename(columns={
                "_year": "year",
                "_grid_id": "grid_id",
                "_lon": "lon",
                "_lat": "lat",
                "_start_state": "start_state",
                "size": "grid_days",
            })
        )
        gy["dataset"] = tag

        diag["status"] = "ok"
        diag["reason"] = ""

        return ev, gy, diag

    except Exception as e:
        diag["status"] = "failed"
        diag["reason"] = f"parse_failed: {repr(e)}"
        return None, None, diag


def build_dataset(root, tag, force=False):
    event_cache = OUT_DIR / f"cache_{tag}_event_summary.csv"
    grid_cache = OUT_DIR / f"cache_{tag}_grid_year_state.csv"
    annual_cache = OUT_DIR / f"cache_{tag}_annual_state.csv"
    diag_cache = OUT_DIR / f"cache_{tag}_read_diagnostics.csv"

    if (not force) and event_cache.exists() and grid_cache.exists() and annual_cache.exists():
        log(f"[CACHE] Loading {tag} from cache.")
        events = pd.read_csv(event_cache)
        grid = pd.read_csv(grid_cache)
        annual = pd.read_csv(annual_cache)
        return {"events": events, "grid": grid, "annual": annual}

    files = list_event_files(root)
    if len(files) == 0:
        raise RuntimeError(f"No event files found under: {root}")

    log(f"[BUILD] Dataset={tag}, root={root}")
    events = []
    grids = []
    diags = []

    files_by_year = {}
    for f in files:
        y = infer_year_from_path(f)
        files_by_year.setdefault(int(y) if pd.notna(y) else -9999, []).append(f)

    for y in YEARS:
        flist = files_by_year.get(y, [])
        ok_count = 0

        for fp in flist:
            ev, gy, diag = summarize_one_event(fp, tag)
            diags.append(diag)

            if ev is not None and gy is not None:
                events.append(ev)
                grids.append(gy)
                ok_count += 1

        log(f"  [{tag}] finished {y}, files={len(flist)}, valid_events={ok_count}")

    diag_df = pd.DataFrame(diags)
    diag_df.to_csv(diag_cache, index=False, encoding="utf-8-sig")

    if len(events) == 0:
        reason_counts = diag_df["reason"].value_counts().head(15)
        log("[ERROR] No valid event rows read.")
        log(reason_counts.to_string())
        raise RuntimeError(
            f"No valid event rows were read for dataset: {tag}.\n"
            f"Check diagnostics file: {diag_cache}"
        )

    events = pd.DataFrame(events)
    grid = pd.concat(grids, ignore_index=True)

    annual_sum = (
        events.groupby(["dataset", "year", "start_state"], as_index=False)
        .agg(
            object_grid_days=("event_grid_days", "sum"),
            event_starts=("event_uid", "count"),
            duration_days=("duration_days", "mean"),
            heat_excess=("heat_excess_mean", "mean"),
            max_area=("max_area_gridcells", "mean"),
            net_displacement=("net_displacement_km", "mean"),
            path_length=("path_length_km", "mean"),
            moving_fraction=("moving_200km", "mean"),
            straightness=("straightness", "median"),
        )
    )

    events.to_csv(event_cache, index=False, encoding="utf-8-sig")
    grid.to_csv(grid_cache, index=False, encoding="utf-8-sig")
    annual_sum.to_csv(annual_cache, index=False, encoding="utf-8-sig")

    log(f"[SAVED] {event_cache}")
    log(f"[SAVED] {grid_cache}")
    log(f"[SAVED] {annual_cache}")
    log(f"[SAVED] {diag_cache}")

    return {"events": events, "grid": grid, "annual": annual_sum}


# =============================================================================
# 5. SPATIAL PLOTTING
# =============================================================================

def infer_edges(vals):
    vals = np.asarray(sorted(np.unique(vals)), dtype=float)

    if len(vals) == 1:
        d = 0.25
        return np.array([vals[0] - d / 2, vals[0] + d / 2])

    mid = (vals[:-1] + vals[1:]) / 2
    first = vals[0] - (mid[0] - vals[0])
    last = vals[-1] + (vals[-1] - mid[-1])
    return np.r_[first, mid, last]


def make_geo_ax(fig, spec):
    """Create a geographic axis from either a GridSpec slot or a manual rect."""
    is_rect = isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == 4

    if HAS_CARTOPY:
        if is_rect:
            ax = fig.add_axes(spec, projection=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())

        ax.set_extent([-125, -66, 24, 50.5], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, edgecolor="0.55")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, edgecolor="0.55")
        try:
            states = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces_lines",
                scale="50m",
                facecolor="none"
            )
            ax.add_feature(states, linewidth=0.35, edgecolor="0.70")
        except Exception:
            pass

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.45,
            color="0.80",
            alpha=0.75,
            linestyle="-"
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 19}
        gl.ylabel_style = {"size": 19}
        return ax

    if is_rect:
        ax = fig.add_axes(spec)
    else:
        ax = fig.add_subplot(spec)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="0.86", linewidth=0.55)
    return ax


def plot_grid_map(ax, df, value_col, cmap, norm=None, vmin=None, vmax=None):
    d = df[["lon", "lat", value_col]].dropna().copy()
    if len(d) == 0:
        return None

    d = (
        d.groupby(["lon", "lat"], as_index=False)[value_col]
        .mean()
    )

    lons = np.sort(d["lon"].unique())
    lats = np.sort(d["lat"].unique())

    transform = ccrs.PlateCarree() if HAS_CARTOPY else None

    if len(lons) * len(lats) <= len(d) * 1.30:
        pivot = d.pivot_table(index="lat", columns="lon", values=value_col)
        pivot = pivot.reindex(index=lats, columns=lons)
        X = infer_edges(lons)
        Y = infer_edges(lats)
        Z = pivot.values

        if HAS_CARTOPY:
            im = ax.pcolormesh(
                X, Y, Z,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
                transform=transform
            )
        else:
            im = ax.pcolormesh(
                X, Y, Z,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                shading="auto"
            )
    else:
        if HAS_CARTOPY:
            im = ax.scatter(
                d["lon"], d["lat"], c=d[value_col],
                s=5.5,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
                transform=transform,
                rasterized=True
            )
        else:
            im = ax.scatter(
                d["lon"], d["lat"], c=d[value_col],
                s=5.5,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
                rasterized=True
            )

    return im


def compute_dry_exposure_map(grid):
    d = grid[grid["start_state"].isin([1, 2])].copy()
    out = (
        d.groupby(["grid_id", "lon", "lat"], as_index=False)["grid_days"]
        .sum()
        .rename(columns={"grid_days": "dry_grid_days"})
    )
    out["log10_dry_grid_days"] = np.log10(out["dry_grid_days"].clip(lower=1))
    return out


def compute_dry_minus_wet_trend_map(grid):
    d = grid.copy()
    d["group"] = np.where(
        d["start_state"].isin([1, 2]), "dry",
        np.where(d["start_state"].isin([5, 6]), "wet", "other")
    )
    d = d[d["group"].isin(["dry", "wet"])]

    yy = (
        d.groupby(["grid_id", "lon", "lat", "year", "group"], as_index=False)["grid_days"]
        .sum()
    )

    piv = (
        yy.pivot_table(
            index=["grid_id", "lon", "lat", "year"],
            columns="group",
            values="grid_days",
            fill_value=0
        )
        .reset_index()
    )

    if "dry" not in piv.columns:
        piv["dry"] = 0
    if "wet" not in piv.columns:
        piv["wet"] = 0

    piv["diff"] = piv["dry"] - piv["wet"]

    rows = []
    for (gid, lon, lat), sub in piv.groupby(["grid_id", "lon", "lat"]):
        if sub["year"].nunique() < 12:
            continue
        slope, p, r = lintrend_per_decade(sub["year"], sub["diff"])
        rows.append({
            "grid_id": gid,
            "lon": lon,
            "lat": lat,
            "trend_dry_minus_wet": slope,
            "p": p,
            "r": r,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 6. FIGURE 1
# =============================================================================

def build_trend_matrix(annual, dataset="rolling"):
    d = annual[annual["dataset"] == dataset].copy()

    metrics = [
        ("Object grid-days", "object_grid_days"),
        ("Event starts", "event_starts"),
        ("Duration", "duration_days"),
        ("Heat excess", "heat_excess"),
    ]

    rows = []
    for metric_label, col in metrics:
        for s in range(1, 7):
            sub = d[d["start_state"] == s]
            slope, p, r = lintrend_per_decade(sub["year"], sub[col])
            rows.append({
                "metric": metric_label,
                "column": col,
                "state": s,
                "trend": slope,
                "p": p,
                "r": r,
            })

    return pd.DataFrame(rows)


def plot_trend_heatmap(ax, trend_df, title, cbar_label="Trend per decade"):
    metrics = ["Object grid-days", "Event starts", "Duration", "Heat excess"]
    states = [1, 2, 3, 4, 5, 6]

    mat = np.full((len(metrics), len(states)), np.nan)
    pmat = np.full_like(mat, np.nan, dtype=float)

    for i, m in enumerate(metrics):
        for j, s in enumerate(states):
            row = trend_df[(trend_df["metric"] == m) & (trend_df["state"] == s)]
            if len(row):
                mat[i, j] = row["trend"].iloc[0]
                pmat[i, j] = row["p"].iloc[0]

    vmax = np.nanquantile(np.abs(mat), 0.95)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(mat, cmap=DIVERGE_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(states)))
    ax.set_xticklabels([f"S{s}" for s in states])
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Initial state")
    ax.set_title(title)

    for i in range(len(metrics)):
        for j in range(len(states)):
            if np.isfinite(mat[i, j]):
                star = p_to_star(pmat[i, j])
                txt = f"{mat[i, j]:.1f}{star}" if abs(mat[i, j]) < 100 else f"{mat[i, j]:.0f}{star}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=15, color="black")

    ax.set_xticks(np.arange(-.5, len(states), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(metrics), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.14, fraction=0.08)
    cbar.set_label(cbar_label, fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    return im


def plot_dry_wet_contrast_panel(ax, annual):
    """Figure 1d: dry–wet exposure contrast through time, replacing the heatmap."""
    con = dry_wet_annual_contrast(annual, "rolling").sort_values("year")
    slope, p, r = lintrend_per_decade(con["year"], con["contrast"])

    ax.axhline(0, color="0.62", lw=1.1, ls="--", zorder=1)
    ax.plot(
        con["year"],
        con["contrast"],
        color=GREY_DARK,
        lw=2.0,
        alpha=0.68,
        label="Annual"
    )

    roll = con["contrast"].rolling(7, center=True, min_periods=3).mean()
    ax.plot(
        con["year"],
        roll,
        color="#8c510a",
        lw=3.2,
        label="7-yr running mean"
    )

    x = con["year"].astype(float).values
    y = con["contrast"].astype(float).values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 8:
        beta = np.polyfit(x[mask], y[mask], 1)
        yy = beta[0] * x + beta[1]
        ax.plot(
            con["year"],
            yy,
            color="black",
            lw=2.5,
            ls="-",
            label=f"Trend: {fmt_trend_legend(slope, p, decimals=2)}"
        )

    ax.set_title("Dry–wet exposure contrast")
    ax.set_xlabel("Year")
    ax.set_ylabel("S1–S2 minus S5–S6\ngrid-days")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    apply_sci_y(ax, powerlimits=(4, 4))
    ax.legend(frameon=False, loc="upper left", fontsize=15, handlelength=2.4)


def draw_figure1(data):
    rolling = data["rolling"]
    grid = rolling["grid"]
    annual = rolling["annual"]

    dry_map = compute_dry_exposure_map(grid)
    trend_map = compute_dry_minus_wet_trend_map(grid)

    # Manual layout is used here because the two map colorbars must sit close
    # to the maps but remain clearly separated from panels c and d. GridSpec
    # tends to pull the colorbars downward when bbox_inches="tight" is used.
    fig = plt.figure(figsize=(19.2, 13.2))

    ax_a = make_geo_ax(fig, [0.055, 0.690, 0.420, 0.245])
    ax_b = make_geo_ax(fig, [0.545, 0.690, 0.420, 0.245])
    cax_a = fig.add_axes([0.065, 0.625, 0.400, 0.024])
    cax_b = fig.add_axes([0.555, 0.625, 0.400, 0.024])
    ax_c = fig.add_axes([0.055, 0.080, 0.420, 0.365])
    ax_d = fig.add_axes([0.545, 0.080, 0.420, 0.365])

    # Panel a
    add_panel_label(ax_a, "a", x=-0.12, y=1.15)
    ax_a.set_title("Dry-state heatwave-object exposure")

    vals = dry_map["log10_dry_grid_days"].dropna()
    vmin = np.nanquantile(vals, 0.03)
    vmax = np.nanquantile(vals, 0.985)

    im_a = plot_grid_map(
        ax_a,
        dry_map,
        "log10_dry_grid_days",
        cmap=DRY_CMAP,
        vmin=vmin,
        vmax=vmax
    )

    cb_a = fig.colorbar(im_a, cax=cax_a, orientation="horizontal", extend="both")
    cb_a.set_label(r"$\log_{10}$(S1–S2 object grid-days)", fontsize=21, labelpad=1)
    cb_a.ax.tick_params(labelsize=19, pad=1)

    # Panel b
    add_panel_label(ax_b, "b", x=-0.12, y=1.15)
    ax_b.set_title("Trend in dry-minus-wet exposure")

    tv = trend_map["trend_dry_minus_wet"].dropna()
    lim = np.nanquantile(np.abs(tv), 0.985)
    if not np.isfinite(lim) or lim == 0:
        lim = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    im_b = plot_grid_map(
        ax_b,
        trend_map,
        "trend_dry_minus_wet",
        cmap=DIVERGE_CMAP,
        norm=norm
    )

    cb_b = fig.colorbar(im_b, cax=cax_b, orientation="horizontal", extend="both")
    cb_b.set_label("S1–S2 minus S5–S6 object grid-days decade$^{-1}$", fontsize=21, labelpad=1)
    cb_b.ax.tick_params(labelsize=19, pad=1)

    # Panel c
    add_panel_label(ax_c, "c", x=-0.12, y=1.12)

    for s in range(1, 7):
        sub = annual[(annual["dataset"] == "rolling") & (annual["start_state"] == s)].sort_values("year")
        slope, p, r = lintrend_per_decade(sub["year"], sub["object_grid_days"])
        label = f"S{s} ({fmt_trend_legend(slope, p, decimals=2)})"

        ax_c.plot(
            sub["year"],
            sub["object_grid_days"],
            lw=2.4 if s in [1, 2] else 2.0,
            color=STATE_COLORS[s],
            label=label
        )

    ax_c.set_title("Annual exposure by initial soil-moisture state", pad=18)
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Object grid-days")
    ax_c.grid(color="0.88", linewidth=0.9)
    clean_spines(ax_c)
    apply_sci_y(ax_c, powerlimits=(4, 4))
    ax_c.legend(
        ncol=2,
        frameon=False,
        loc="upper left",
        fontsize=13,
        handlelength=2.2
    )

    # Panel d
    add_panel_label(ax_d, "d", x=-0.12, y=1.12)
    plot_dry_wet_contrast_panel(ax_d, annual)
    ax_d.set_title("Dry–wet exposure contrast", pad=18)

    save_figure(fig, "Figure1_dry_state_exposure_rebuilt")


# =============================================================================
# 7. SUPPLEMENTARY FIGURE 1
# =============================================================================

def dry_wet_annual_contrast(annual, dataset):
    d = annual[annual["dataset"] == dataset].copy()

    dry = (
        d[d["start_state"].isin([1, 2])]
        .groupby("year")["object_grid_days"]
        .sum()
    )
    wet = (
        d[d["start_state"].isin([5, 6])]
        .groupby("year")["object_grid_days"]
        .sum()
    )

    out = pd.DataFrame({"dry": dry, "wet": wet}).fillna(0).reset_index()
    out["contrast"] = out["dry"] - out["wet"]
    out["dataset"] = dataset
    return out


def ccdf(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    values = np.sort(values)

    if len(values) == 0:
        return np.array([]), np.array([])

    y = 1.0 - np.arange(1, len(values) + 1) / len(values)
    y = np.maximum(y, 1.0 / len(values))
    return values, y


def draw_supplementary_fig1(data):
    rolling = data["rolling"]
    fixed = data["fixed"]

    fig = plt.figure(figsize=(19.6, 12.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.62, wspace=0.34)

    # Panel a: state hierarchy
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a", x=-0.16, y=1.10)

    for dataset, dd, color, marker in [
        ("rolling", rolling, GREY_DARK, "o"),
        ("fixed", fixed, GREY_MID, "s"),
    ]:
        ann = dd["annual"]
        vals = (
            ann.groupby("start_state")["object_grid_days"]
            .mean()
            .reindex(range(1, 7))
        )
        ax.plot(
            range(1, 7), vals.values,
            marker=marker,
            lw=2.5,
            ms=7.5,
            color=color,
            label="Rolling threshold" if dataset == "rolling" else "Fixed threshold"
        )

    ax.set_title("State hierarchy")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"S{i}" for i in range(1, 7)])
    ax.set_xlabel("Initial state")
    ax.set_ylabel("Grid-days yr$^{-1}$")
    apply_sci_y(ax, powerlimits=(4, 4))
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=17)

    # Panel b: trend hierarchy
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "b", x=-0.16, y=1.10)

    for dataset, dd, color, marker in [
        ("rolling", rolling, GREY_DARK, "o"),
        ("fixed", fixed, GREY_MID, "s"),
    ]:
        ann = dd["annual"]
        slopes = []
        pvals = []
        for s in range(1, 7):
            sub = ann[ann["start_state"] == s]
            slope, p, r = lintrend_per_decade(sub["year"], sub["object_grid_days"])
            slopes.append(slope)
            pvals.append(p)

        ax.plot(
            range(1, 7), slopes,
            marker=marker,
            lw=2.5,
            ms=7.5,
            color=color,
            label="Rolling threshold" if dataset == "rolling" else "Fixed threshold"
        )

        for s, y, p in zip(range(1, 7), slopes, pvals):
            ax.text(s, y, p_to_star(p), ha="center", va="bottom", fontsize=16, color=color)

    ax.axhline(0, color="0.60", lw=1.0, ls="--")
    ax.set_title("Trend hierarchy")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"S{i}" for i in range(1, 7)])
    ax.set_xlabel("Initial state")
    ax.set_ylabel("Trend decade$^{-1}$")
    apply_sci_y(ax, powerlimits=(4, 4))
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)

    # Panel c: dry-wet contrast
    ax = fig.add_subplot(gs[0, 2])
    add_panel_label(ax, "c", x=-0.16, y=1.10)

    for dataset, dd, color in [
        ("rolling", rolling, GREY_DARK),
        ("fixed", fixed, GREY_MID),
    ]:
        con = dry_wet_annual_contrast(dd["annual"], dataset)
        slope, p, r = lintrend_per_decade(con["year"], con["contrast"])
        label = (
            "Rolling threshold"
            if dataset == "rolling"
            else "Fixed threshold"
        )
        label = f"{label}: {fmt_trend_legend(slope, p, decimals=2)}"

        ax.plot(con["year"], con["contrast"], lw=2.4, color=color, label=label)

    ax.axhline(0, color="0.60", lw=1.0, ls="--")
    ax.set_title("Dry–wet contrast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Dry–wet grid-days")
    apply_sci_y(ax, powerlimits=(4, 4))
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=15)

    # Panel d: secondary trend heatmap
    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, "d", x=-0.16, y=1.10)

    trend_matrix = build_trend_matrix(rolling["annual"], dataset="rolling")
    secondary = trend_matrix[trend_matrix["metric"].isin(["Event starts", "Duration", "Heat excess"])].copy()

    metrics = ["Event starts", "Duration", "Heat excess"]
    states = list(range(1, 7))
    mat = np.full((len(metrics), len(states)), np.nan)
    pmat = np.full_like(mat, np.nan)

    for i, m in enumerate(metrics):
        for j, s in enumerate(states):
            row = secondary[(secondary["metric"] == m) & (secondary["state"] == s)]
            if len(row):
                mat[i, j] = row["trend"].iloc[0]
                pmat[i, j] = row["p"].iloc[0]

    vmax = np.nanquantile(np.abs(mat), 0.95)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(mat, cmap=DIVERGE_CMAP, norm=norm, aspect="auto")
    ax.set_title("Secondary-attribute trends")
    ax.set_xticks(np.arange(len(states)))
    ax.set_xticklabels([f"S{s}" for s in states])
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Initial state")

    for i in range(len(metrics)):
        for j in range(len(states)):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j, i,
                    f"{mat[i, j]:.2f}{p_to_star(pmat[i, j])}",
                    ha="center", va="center",
                    fontsize=14,
                    color="black"
                )

    ax.set_xticks(np.arange(-.5, len(states), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(metrics), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.28, fraction=0.065)
    cb.set_label("Trend decade$^{-1}$", fontsize=18, labelpad=3)
    cb.ax.tick_params(labelsize=16)

    # Panel e: largest component fraction
    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "e", x=-0.16, y=1.10)

    for dataset, dd, color in [
        ("rolling", rolling, GREY_DARK),
        ("fixed", fixed, GREY_MID),
    ]:
        ev = dd["events"]
        ysum = (
            ev.groupby("year", as_index=False)
            .agg(total=("event_grid_days", "sum"), largest=("event_grid_days", "max"))
        )
        ysum["lcf"] = ysum["largest"] / ysum["total"]
        slope, p, r = lintrend_per_decade(ysum["year"], ysum["lcf"])
        label = (
            "Rolling threshold"
            if dataset == "rolling"
            else "Fixed threshold"
        )
        label = f"{label}: {slope:.3f} decade$^{{-1}}${p_to_star(p)}"
        ax.plot(ysum["year"], ysum["lcf"], lw=2.4, color=color, label=label)

    ax.set_title("Event-object fragmentation")
    ax.set_xlabel("Year")
    ax.set_ylabel("Largest fraction")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=14)

    # Panel f: CCDF
    ax = fig.add_subplot(gs[1, 2])
    add_panel_label(ax, "f", x=-0.16, y=1.10)

    for s in range(1, 7):
        for dataset, dd, ls in [
            ("rolling", rolling, "-"),
            ("fixed", fixed, "--"),
        ]:
            ev = dd["events"]
            vals = ev.loc[ev["start_state"] == s, "event_grid_days"].values
            x, y = ccdf(vals)
            if len(x) == 0:
                continue
            ax.plot(
                x, y,
                color=STATE_COLORS[s],
                ls=ls,
                lw=2.2,
                alpha=0.96
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Event-size distribution")
    ax.set_xlabel("Event size (grid-days)")
    ax.set_ylabel("CCDF")
    ax.grid(color="0.88", linewidth=0.8, which="both")
    clean_spines(ax)

    state_handles = [
        Line2D([0], [0], color=STATE_COLORS[s], lw=2.6, label=f"S{s}")
        for s in range(1, 7)
    ]
    style_handles = [
        Line2D([0], [0], color="0.25", lw=2.6, ls="-", label="Rolling threshold"),
        Line2D([0], [0], color="0.25", lw=2.6, ls="--", label="Fixed threshold"),
    ]

    leg1 = ax.legend(
        handles=state_handles,
        frameon=False,
        ncol=3,
        loc="upper right",
        fontsize=12,
        title="Initial state",
        title_fontsize=13
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=style_handles,
        frameon=False,
        loc="lower left",
        fontsize=13
    )

    save_figure(fig, "Supplementary_Fig1_threshold_robustness_rebuilt")


# =============================================================================
# 8. SUPPLEMENTARY FIGURE 2
# =============================================================================

def bin_duration_summary(ev):
    d = ev[["duration_days", "max_area_gridcells"]].dropna().copy()
    d = d[(d["duration_days"] > 0) & (d["max_area_gridcells"] > 0)]

    rows = []
    for dur, sub in d.groupby("duration_days"):
        if len(sub) < 8:
            continue
        med = np.nanmedian(sub["max_area_gridcells"])
        q25 = np.nanpercentile(sub["max_area_gridcells"], 25)
        q75 = np.nanpercentile(sub["max_area_gridcells"], 75)
        rows.append({
            "duration_days": dur,
            "median": med,
            "q25": q25,
            "q75": q75,
            "n": len(sub),
        })

    return pd.DataFrame(rows)


def draw_supplementary_fig2(data):
    rolling = data["rolling"]
    fixed = data["fixed"]

    ev = rolling["events"].copy()

    fig = plt.figure(figsize=(19.6, 12.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.36)

    # Panel a: catalogue size
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a", x=-0.16, y=1.10)

    for dataset, dd, color in [
        ("rolling", rolling, GREY_DARK),
        ("fixed", fixed, GREY_MID),
    ]:
        tmp = dd["events"].groupby("year", as_index=False)["event_uid"].count()
        tmp = tmp.rename(columns={"event_uid": "n_events"})
        slope, p, r = lintrend_per_decade(tmp["year"], tmp["n_events"])
        label = (
            "Rolling threshold"
            if dataset == "rolling"
            else "Fixed threshold"
        )
        label = f"{label}: {slope:.1f} decade$^{{-1}}${p_to_star(p)}"
        ax.plot(tmp["year"], tmp["n_events"], lw=2.4, color=color, label=label)

    ax.set_title("Catalogue size")
    ax.set_xlabel("Year")
    ax.set_ylabel("Event objects")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=14)

    # Panel b: size-duration structure
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "b", x=-0.16, y=1.10)

    plot_df = ev[["duration_days", "max_area_gridcells", "start_state"]].dropna().copy()
    plot_df = plot_df[(plot_df["duration_days"] > 0) & (plot_df["max_area_gridcells"] > 0)]

    if len(plot_df) > 4500:
        plot_sub = plot_df.sample(4500, random_state=42)
    else:
        plot_sub = plot_df

    for s in range(1, 7):
        ss = plot_sub[plot_sub["start_state"] == s]
        if len(ss) == 0:
            continue
        ax.scatter(
            ss["duration_days"],
            ss["max_area_gridcells"],
            s=18,
            color=STATE_COLORS[s],
            alpha=0.25,
            linewidths=0,
            rasterized=True
        )

    bins = bin_duration_summary(ev)
    if len(bins):
        ax.plot(
            bins["duration_days"], bins["median"],
            color="black", lw=2.8, marker="o", ms=5.5,
            label="Median"
        )
        ax.fill_between(
            bins["duration_days"],
            bins["q25"],
            bins["q75"],
            color="black",
            alpha=0.16,
            linewidth=0,
            label="IQR"
        )

    rho, p = spearman_stat(plot_df["duration_days"], np.log10(plot_df["max_area_gridcells"]))
    ax.text(
        0.05, 0.95,
        f"Spearman ρ={rho:.2f}\n{format_p(p)}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=17,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.88)
    )

    ax.set_yscale("log")
    ax.set_title("Event size–duration structure")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Max area")
    ax.grid(color="0.88", linewidth=0.8, which="both")
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=14, loc="lower right")

    # Panel c: state-specific object size
    ax = fig.add_subplot(gs[0, 2])
    add_panel_label(ax, "c", x=-0.16, y=1.10)

    data_box = []
    for s in range(1, 7):
        vals = np.log10(ev.loc[ev["start_state"] == s, "event_grid_days"].clip(lower=1))
        data_box.append(vals.dropna().values)

    bp = ax.boxplot(
        data_box,
        positions=np.arange(1, 7),
        widths=0.66,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", lw=1.4),
        boxprops=dict(linewidth=1.1),
        whiskerprops=dict(linewidth=1.1),
        capprops=dict(linewidth=1.1)
    )

    for patch, s in zip(bp["boxes"], range(1, 7)):
        patch.set_facecolor(STATE_COLORS[s])
        patch.set_alpha(0.72)

    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"S{i}" for i in range(1, 7)])
    ax.set_title("State-specific object size")
    ax.set_xlabel("Initial state")
    ax.set_ylabel(r"$\log_{10}$(size)")
    ax.grid(axis="y", color="0.88", linewidth=0.9)
    clean_spines(ax)

    # Panel d: annual composition of initial soil-moisture states
    # This replaces the previous path-length/duration boxplot, which was
    # dominated by zero or near-zero values and was therefore not diagnostic.
    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, "d", x=-0.16, y=1.10)

    comp = (
        ev.groupby(["year", "start_state"], as_index=False)["event_uid"]
        .count()
        .rename(columns={"event_uid": "n"})
    )
    comp = comp.pivot_table(index="year", columns="start_state", values="n", fill_value=0)
    comp = comp.reindex(columns=range(1, 7), fill_value=0).sort_index()
    comp_frac = comp.div(comp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    x_year = comp_frac.index.values.astype(float)
    y_stack = [comp_frac[s].values for s in range(1, 7)]
    ax.stackplot(
        x_year,
        y_stack,
        colors=[STATE_COLORS[s] for s in range(1, 7)],
        labels=[f"S{s}" for s in range(1, 7)],
        alpha=0.86,
        linewidth=0.0
    )

    dry_frac = comp_frac[[1, 2]].sum(axis=1)
    wet_frac = comp_frac[[5, 6]].sum(axis=1)
    ax.plot(x_year, dry_frac.values, color="black", lw=2.2, label="S1–S2 fraction")
    ax.plot(x_year, wet_frac.values, color="white", lw=2.0, ls="--", label="S5–S6 fraction")

    ax.set_title("Initial-state composition")
    ax.set_xlabel("Year")
    ax.set_ylabel("Object fraction")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)

    handles, labels = ax.get_legend_handles_labels()
    # Keep the legend compact: show state colours plus dry/wet summary lines.
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=10.5,
        handlelength=1.5,
        columnspacing=0.8
    )

    # Panel e: trajectory efficiency
    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "e", x=-0.16, y=1.10)

    te = ev[["path_length_km", "net_displacement_km", "start_state"]].dropna().copy()
    te = te[(te["path_length_km"] > 0) & (te["net_displacement_km"] >= 0)]

    if len(te) > 4500:
        te_plot = te.sample(4500, random_state=43)
    else:
        te_plot = te

    for s in range(1, 7):
        ss = te_plot[te_plot["start_state"] == s]
        if len(ss) == 0:
            continue
        ax.scatter(
            ss["path_length_km"],
            ss["net_displacement_km"],
            s=17,
            color=STATE_COLORS[s],
            alpha=0.25,
            linewidths=0,
            rasterized=True
        )

    max_lim = np.nanpercentile(te[["path_length_km", "net_displacement_km"]].values, 99.5)
    max_lim = max(max_lim, 50)

    ax.plot([0, max_lim], [0, max_lim], color="0.35", ls="--", lw=1.4, label="1:1")

    rho, p = spearman_stat(te["path_length_km"], te["net_displacement_km"])

    if len(te) >= 20 and stats is not None:
        lr = stats.linregress(te["path_length_km"], te["net_displacement_km"])
        xx = np.linspace(0, max_lim, 100)
        yy = lr.intercept + lr.slope * xx
        yy = np.clip(yy, 0, None)
        ax.plot(xx, yy, color="black", lw=2.4, label="Fitted trend")

    ax.text(
        0.05, 0.95,
        f"Spearman ρ={rho:.2f}\n{format_p(p)}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=17,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.88)
    )

    ax.set_xlim(0, max_lim)
    ax.set_ylim(0, max_lim)
    apply_sci_x(ax, powerlimits=(3, 3))
    apply_sci_y(ax, powerlimits=(3, 3))
    ax.set_title("Trajectory efficiency")
    ax.set_xlabel("Path length (km)")
    ax.set_ylabel("Net disp. (km)")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=14, loc="lower right")

    # Panel f: mobility support metrics
    ax = fig.add_subplot(gs[1, 2])
    add_panel_label(ax, "f", x=-0.16, y=1.10)

    rows = []
    for s in range(1, 7):
        sub = ev[ev["start_state"] == s]
        mf, mf_lo, mf_hi = bootstrap_ci(sub["moving_200km"].values, func=np.nanmean, seed=40 + s)
        st, st_lo, st_hi = bootstrap_ci(sub["straightness"].dropna().values, func=np.nanmedian, seed=60 + s)

        rows.append({
            "state": s,
            "moving_fraction": mf,
            "moving_low": mf_lo,
            "moving_high": mf_hi,
            "straightness": st,
            "straight_low": st_lo,
            "straight_high": st_hi,
        })

    ms = pd.DataFrame(rows)
    x = np.arange(1, 7)

    ax2 = ax.twinx()

    ax.errorbar(
        x - 0.10,
        ms["straightness"],
        yerr=[
            ms["straightness"] - ms["straight_low"],
            ms["straight_high"] - ms["straightness"]
        ],
        fmt="o-",
        color="#1f78b4",
        lw=2.4,
        ms=7.5,
        capsize=3,
        label="Median straightness"
    )

    ax2.errorbar(
        x + 0.10,
        ms["moving_fraction"],
        yerr=[
            ms["moving_fraction"] - ms["moving_low"],
            ms["moving_high"] - ms["moving_fraction"]
        ],
        fmt="s-",
        color="#e66101",
        lw=2.4,
        ms=7.5,
        capsize=3,
        label=f"Fraction moving ≥{int(MOVING_THRESHOLD_KM)} km"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in x])
    ax.set_xlabel("Initial state")
    ax.set_ylabel("Straightness", color="#1f78b4")
    ax2.set_ylabel(f"Moving ≥{int(MOVING_THRESHOLD_KM)} km", color="#e66101")
    ax.set_title("Mobility support metrics")

    ax.grid(axis="y", color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax2.spines["top"].set_visible(False)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        loc="lower left",
        fontsize=13
    )

    save_figure(fig, "Supplementary_Fig2_event_object_support_rebuilt")


# =============================================================================
# 9. MAIN
# =============================================================================

def main(force_rebuild=False):
    log("=" * 100)
    log("[INFO] Rebuilding journal-facing Figure 1 and compact Supplementary Figures")
    log(f"[INFO] Output directory: {OUT_DIR}")
    log("=" * 100)

    fixed = build_dataset(FIXED_ROOT, "fixed", force=force_rebuild)
    rolling = build_dataset(ROLLING_ROOT, "rolling", force=force_rebuild)

    data = {
        "fixed": fixed,
        "rolling": rolling,
    }

    log("[DRAW] Figure 1")
    draw_figure1(data)

    log("[DRAW] Supplementary Fig. 1")
    draw_supplementary_fig1(data)

    log("[DRAW] Supplementary Fig. 2")
    draw_supplementary_fig2(data)

    log("=" * 100)
    log("[DONE] All figures rebuilt.")
    log(f"[DONE] Output directory: {OUT_DIR}")
    log("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild caches from event CSV files instead of using existing cache."
    )
    args = parser.parse_args()
    main(force_rebuild=args.rebuild)