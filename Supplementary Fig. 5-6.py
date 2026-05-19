# -*- coding: utf-8 -*-
r"""
Redesigned Supplementary Fig. 1 and Supplementary Fig. 2
=======================================================

This script uses existing cached CSV files from Figure 2 and Figure 3.

Main fixes:
1) The common-support PCA no longer assumes unmatched and matched tables
   have the same column names.
   Example:
       unmatched/raw table may have:
           Z500_height_m_anom_onset
           T850_anom_onset
           W500_anom_onset
       matched table has:
           Z500
           T850
           W500
   This script maps them separately and renames them to common labels.

2) Figure 3 exact uploaded columns are supported:
       Bowen_anom_event
       EF_anom_event
       H_anom_event
       Bowen_anom_dry_minus_wet
       EF_anom_dry_minus_wet
       H_anom_dry_minus_wet
       dry_Bowen_anom_count / wet_Bowen_anom_count
       dry_EF_anom_count / wet_EF_anom_count
       dry_H_anom_count / wet_H_anom_count

Outputs:
    E:\第二篇数据0427\Supplementary_Fig_1_2_composites_redesigned_v2
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")


# =============================================================================
# 0. USER PATHS
# =============================================================================

FIG2_ROOT = Path(r"E:\第二篇数据0427\Figure2_circulation_control_outputs")
FIG3_ROOT = Path(r"E:\第二篇数据0427\Figure3_surface_energy_partitioning_outputs")

FIG2_MATCHED_EVENT_TABLE = FIG2_ROOT / "Figure2_main_circulation_matched_event_table.csv"
FIG2_FRONT_LOCAL_EVENT = FIG2_ROOT / "Figure2_panel_f_front_local_drying_contrast_computed_for_matched_events.csv"

OPTIONAL_UNMATCHED_TABLES = [
    FIG2_ROOT / "event_level_summary_with_Z500_W500_T850_controls.csv",
    FIG2_ROOT / "Figure2_event_level_summary_with_Z500_W500_T850_controls.csv",
    FIG2_ROOT / "Figure2_unmatched_event_table.csv",
    FIG2_ROOT / "Figure2_main_unmatched_event_table.csv",
]

FIG3_EVENT_CSV = FIG3_ROOT / "Figure3_event_level_surface_energy_anomaly_summary.csv"
FIG3_SPATIAL_CSV = FIG3_ROOT / "Figure3_spatial_surface_energy_anomaly_dry_wet_contrast.csv"

OUT_DIR = Path(r"E:\第二篇数据0427\Supplementary_Fig_1_2_composites_redesigned_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. STYLE
# =============================================================================

RANDOM_SEED = 42
N_BOOT = 600
DPI = 500

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18.5,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 17.5,
    "ytick.labelsize": 17.5,
    "legend.fontsize": 16.5,
    "axes.linewidth": 1.15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.color": "#e4e4e4",
    "grid.linewidth": 0.8,
})

DRY_COLOR = "#b2182b"
WET_COLOR = "#2166ac"
GREEN = "#1b9e77"
PURPLE = "#756bb1"
ORANGE = "#d95f02"
BROWN = "#8c510a"
GREY = "#4d4d4d"

STATE_COLORS = {
    "S1": "#8c510a",
    "S2": "#bf812d",
    "S3": "#dfc27d",
    "S4": "#80cdc1",
    "S5": "#35978f",
    "S6": "#01665e",
}

METRIC_COLORS = {
    "Event voxels": DRY_COLOR,
    "Maximum area": ORANGE,
    "Net displacement": GREY,
    "Front-local drying": PURPLE,
}

ENERGY_COLORS = {
    "Bowen": BROWN,
    "Evap. suppression": GREEN,
    "Sensible heat": ORANGE,
    "Front-local drying": PURPLE,
}

REGION_ORDER = [
    "Northwest",
    "Northern Great Plains",
    "Midwest",
    "Northeast",
    "Southwest",
    "Southern Great Plains",
    "Southeast",
]

REGION_ABBR = {
    "Northwest": "NW",
    "Northern Great Plains": "NGP",
    "Midwest": "MW",
    "Northeast": "NE",
    "Southwest": "SW",
    "Southern Great Plains": "SGP",
    "Southeast": "SE",
    "CONUS": "CONUS",
    "All pairs": "All",
}

REGION_COLORS = {
    "Northwest": "#c0392b",
    "Northern Great Plains": "#7f8c8d",
    "Midwest": "#8e44ad",
    "Northeast": "#1f78b4",
    "Southwest": "#d95f02",
    "Southern Great Plains": "#1b9e77",
    "Southeast": "#e6ab02",
}

SHORT_METRIC_LABELS = {
    "Event voxels": "Voxels",
    "Maximum area": "Max area",
    "Net displacement": "Displacement",
    "Front-local drying": "Front drying",
}


# =============================================================================
# 2. BASIC UTILITIES
# =============================================================================

def log(msg):
    print(msg, flush=True)


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df, candidates, required=True, desc="column"):
    if isinstance(candidates, str):
        candidates = [candidates]

    norm_map = {norm_name(c): c for c in df.columns}

    for cand in candidates:
        key = norm_name(cand)
        if key in norm_map:
            return norm_map[key]

    for cand in candidates:
        key = norm_name(cand)
        for nk, orig in norm_map.items():
            if key and (key in nk or nk in key):
                return orig

    if required:
        raise KeyError(
            f"Cannot find {desc}.\n"
            f"Candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def to_numeric_if_possible(df):
    out = df.copy()
    for c in out.columns:
        try:
            out[c] = pd.to_numeric(out[c], errors="ignore")
        except Exception:
            pass
    return out


def add_panel_letter(ax, letter, x=-0.14, y=1.06, fs=27):
    """Place panel letters outside the upper-left corner of the axes."""
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fs,
        fontweight="bold",
        clip_on=False,
    )

def show_no_data(ax, message):
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=18,
        color="0.35",
    )


def save_figure(fig, stem):
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")


def gaussian_kernel(sigma=1.2, radius=None):
    if radius is None:
        radius = max(1, int(4 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def smooth1d(arr, sigma=1.2):
    return np.convolve(arr, gaussian_kernel(sigma=sigma), mode="same")


def smooth2d(arr, sigma=1.3):
    k = gaussian_kernel(sigma=sigma)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, tmp)
    return out


def bootstrap_median_ci(values, n_boot=N_BOOT, seed=RANDOM_SEED):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 5:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.nanmedian(values[idx], axis=1)

    return (
        float(np.nanmedian(values)),
        float(np.nanpercentile(boots, 2.5)),
        float(np.nanpercentile(boots, 97.5)),
    )


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=RANDOM_SEED):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 5:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.nanmean(values[idx], axis=1)

    return (
        float(np.nanmean(values)),
        float(np.nanpercentile(boots, 2.5)),
        float(np.nanpercentile(boots, 97.5)),
    )


def cohens_d_independent(x, y):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 3 or len(y) < 3:
        return np.nan

    vx = np.nanvar(x, ddof=1)
    vy = np.nanvar(y, ddof=1)

    pooled = np.sqrt(
        ((len(x) - 1) * vx + (len(y) - 1) * vy) /
        (len(x) + len(y) - 2)
    )

    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan

    return float((np.nanmean(x) - np.nanmean(y)) / pooled)


def standardized_mean_diff(x, y):
    return cohens_d_independent(x, y)


def bootstrap_cohens_d(x, y, n_boot=N_BOOT, seed=RANDOM_SEED):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 5 or len(y) < 5:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        out[i] = cohens_d_independent(xb, yb)

    return (
        cohens_d_independent(x, y),
        float(np.nanpercentile(out, 2.5)),
        float(np.nanpercentile(out, 97.5)),
    )


def paired_standardized_effect(diff):
    diff = np.asarray(pd.Series(diff).dropna(), dtype=float)
    if len(diff) < 5:
        return np.nan

    sd = np.nanstd(diff, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan

    return float(np.nanmean(diff) / sd)


def bootstrap_paired_std_effect(diff, n_boot=N_BOOT, seed=RANDOM_SEED):
    diff = np.asarray(pd.Series(diff).dropna(), dtype=float)
    if len(diff) < 5:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        b = rng.choice(diff, size=len(diff), replace=True)
        out[i] = paired_standardized_effect(b)

    return (
        paired_standardized_effect(diff),
        float(np.nanpercentile(out, 2.5)),
        float(np.nanpercentile(out, 97.5)),
    )


def spearman_rank_corr(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    mask = x.notna() & y.notna()

    if mask.sum() < 10:
        return np.nan

    xr = x[mask].rank().values
    yr = y[mask].rank().values

    if np.nanstd(xr) <= 0 or np.nanstd(yr) <= 0:
        return np.nan

    return float(np.corrcoef(xr, yr)[0, 1])


def paired_wide(df, metric):
    if (
        "matched_pair_id" not in df.columns or
        "dry_vs_wet" not in df.columns or
        metric not in df.columns
    ):
        return None

    tmp = df[["matched_pair_id", "dry_vs_wet", metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna()

    if tmp.empty:
        return None

    wide = tmp.pivot_table(
        index="matched_pair_id",
        columns="dry_vs_wet",
        values=metric,
        aggfunc="first",
    )

    if 1 not in wide.columns or 0 not in wide.columns:
        return None

    wide = wide.dropna(subset=[1, 0]).copy()

    if len(wide) == 0:
        return None

    wide = wide.rename(columns={1: "dry", 0: "wet"})
    wide["diff"] = wide["dry"] - wide["wet"]

    return wide


def smoothed_density(values, bins=100, sigma=2.0, xlim=None):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)

    if len(values) < 10:
        return None, None

    if xlim is None:
        lo, hi = np.nanquantile(values, [0.01, 0.99])
        pad = 0.15 * (hi - lo if hi > lo else 1.0)
        xlim = (lo - pad, hi + pad)

    counts, edges = np.histogram(values, bins=bins, range=xlim, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    dens = smooth1d(counts, sigma=sigma)

    return mids, dens


def parse_state_value(v):
    if pd.isna(v):
        return np.nan

    s = str(v).strip()
    m = re.search(r"([1-6])", s)

    if m:
        return int(m.group(1))

    try:
        x = int(float(s))
        if 1 <= x <= 6:
            return x
    except Exception:
        pass

    return np.nan


# =============================================================================
# 3. REGIONS
# =============================================================================

def assign_fallback_conus_region(lon, lat):
    if not np.isfinite(lon) or not np.isfinite(lat):
        return np.nan

    if lon < -111:
        return "Northwest" if lat >= 42 else "Southwest"

    if -111 <= lon < -95:
        return "Northern Great Plains" if lat >= 42 else "Southern Great Plains"

    if -95 <= lon < -80:
        return "Midwest" if lat >= 37 else "Southeast"

    if lon >= -80:
        return "Northeast" if lat >= 37 else "Southeast"

    return np.nan


def add_region_column(df):
    out = df.copy()

    region_col = find_col(
        out,
        ["region", "climate_region", "nca_region", "us_region", "Region"],
        required=False,
        desc="region",
    )

    if region_col is not None:
        out["region_final"] = out[region_col].astype(str)
        return out

    lon_col = find_col(
        out,
        ["longitude", "lon", "origin_lon", "start_lon", "centroid_lon", "grid_lon"],
        required=False,
        desc="lon",
    )
    lat_col = find_col(
        out,
        ["latitude", "lat", "origin_lat", "start_lat", "centroid_lat", "grid_lat"],
        required=False,
        desc="lat",
    )

    if lon_col is not None and lat_col is not None:
        lon = pd.to_numeric(out[lon_col], errors="coerce")
        lat = pd.to_numeric(out[lat_col], errors="coerce")
        out["region_final"] = [
            assign_fallback_conus_region(x, y)
            for x, y in zip(lon, lat)
        ]
    else:
        out["region_final"] = np.nan

    return out


# =============================================================================
# 4. LOAD FIGURE 2 DATA
# =============================================================================

def infer_dry_wet(df):
    out = df.copy()

    if "dry_vs_wet" in out.columns:
        out["dry_vs_wet"] = pd.to_numeric(out["dry_vs_wet"], errors="coerce")
        return out

    if "regime" in out.columns:
        rr = out["regime"].astype(str).str.lower().str.strip()
        out["dry_vs_wet"] = np.where(
            rr.eq("dry"),
            1,
            np.where(rr.eq("wet"), 0, np.nan),
        )
        return out

    state_col = find_col(
        out,
        ["start_state", "initial_state", "state_label", "soil_moisture_state", "state"],
        required=False,
        desc="state",
    )

    if state_col is not None:
        ss = out[state_col].astype(str).str.upper().str.strip()
        out["dry_vs_wet"] = np.where(
            ss.isin(["S1", "S2", "1", "2"]),
            1,
            np.where(ss.isin(["S5", "S6", "5", "6"]), 0, np.nan),
        )
    else:
        out["dry_vs_wet"] = np.nan

    return out


def add_basic_transforms(df):
    out = df.copy()

    if "event_voxels" in out.columns:
        out["event_voxels"] = pd.to_numeric(out["event_voxels"], errors="coerce").clip(lower=1)
        out["log10_event_voxels"] = np.log10(out["event_voxels"])

    if "max_area_km2" in out.columns:
        out["max_area_km2"] = pd.to_numeric(out["max_area_km2"], errors="coerce").clip(lower=1)
        out["log10_max_area_km2"] = np.log10(out["max_area_km2"])

    # Your matched table also contains natural-log columns.
    # Convert them to log10-like scale if the direct raw variables are unavailable.
    if "log_event_voxels" in out.columns and "log10_event_voxels" not in out.columns:
        out["log10_event_voxels"] = pd.to_numeric(out["log_event_voxels"], errors="coerce") / np.log(10)

    if "log_max_area" in out.columns and "log10_max_area_km2" not in out.columns:
        out["log10_max_area_km2"] = pd.to_numeric(out["log_max_area"], errors="coerce") / np.log(10)

    for raw, new in [
        ("net_displacement_km", "log1p_net_displacement_km"),
        ("path_length_km", "log1p_path_length_km"),
        ("max_daily_step_km", "log1p_max_daily_step_km"),
    ]:
        if raw in out.columns:
            out[raw] = pd.to_numeric(out[raw], errors="coerce").clip(lower=0)
            out[new] = np.log1p(out[raw])

    return out


def load_matched_fig2():
    if not FIG2_MATCHED_EVENT_TABLE.exists():
        raise FileNotFoundError(f"Missing matched table:\n{FIG2_MATCHED_EVENT_TABLE}")

    df = pd.read_csv(FIG2_MATCHED_EVENT_TABLE, low_memory=False)
    df = to_numeric_if_possible(df)
    df = infer_dry_wet(df)

    if "matched_pair_id" not in df.columns:
        raise KeyError("matched_pair_id is required in matched event table.")

    need_front = (
        "front_local_drying_contrast" not in df.columns or
        pd.to_numeric(
            df.get("front_local_drying_contrast", pd.Series(dtype=float)),
            errors="coerce"
        ).notna().sum() < 20
    )

    if need_front and FIG2_FRONT_LOCAL_EVENT.exists() and "event_id" in df.columns:
        fl = pd.read_csv(FIG2_FRONT_LOCAL_EVENT, low_memory=False)

        if "event_id" in fl.columns:
            front_col = find_col(
                fl,
                [
                    "front_local_drying_contrast",
                    "front_local_drying",
                    "front_local_drying_speed_contrast",
                    "front_drying_contrast",
                ],
                required=False,
                desc="front-local drying",
            )

            if front_col is not None:
                fl = fl[["event_id", front_col]].drop_duplicates("event_id")
                fl = fl.rename(columns={front_col: "front_local_drying_contrast"})
                df = df.drop(columns=["front_local_drying_contrast"], errors="ignore")
                df = df.merge(fl, on="event_id", how="left")

    df = df[df["dry_vs_wet"].isin([0, 1])].copy()
    df["dry_vs_wet"] = df["dry_vs_wet"].astype(int)

    df = add_basic_transforms(df)
    df = add_region_column(df)

    log(f"[INFO] Matched rows: {len(df):,}")
    log(f"[INFO] Matched pairs: {df['matched_pair_id'].nunique():,}")

    return df


def load_unmatched_fig2():
    for p in OPTIONAL_UNMATCHED_TABLES:
        if p.exists():
            log(f"[INFO] Using unmatched/raw event table for pre-matching balance:\n       {p}")

            df = pd.read_csv(p, low_memory=False)
            df = to_numeric_if_possible(df)
            df = infer_dry_wet(df)

            df = df[df["dry_vs_wet"].isin([0, 1])].copy()
            df["dry_vs_wet"] = df["dry_vs_wet"].astype(int)

            df = add_basic_transforms(df)
            df = add_region_column(df)

            log(f"[INFO] Unmatched dry/wet rows: {len(df):,}")
            return df

    log("[WARN] No unmatched/raw event table found. Panel a/b will use matched-only support diagnostics.")
    return None


# =============================================================================
# 5. LOAD FIGURE 3 DATA
# =============================================================================

def load_fig3_event():
    if not FIG3_EVENT_CSV.exists():
        raise FileNotFoundError(f"Missing Figure 3 event CSV:\n{FIG3_EVENT_CSV}")

    df = pd.read_csv(FIG3_EVENT_CSV, low_memory=False)
    df = to_numeric_if_possible(df)

    state_col = find_col(
        df,
        [
            "state_label", "initial_state", "sm_state", "soil_moisture_state",
            "s_bin_label", "state", "start_state",
        ],
        desc="state",
    )

    bowen_col = find_col(
        df,
        [
            "Bowen_anom_event",
            "event_mean_bowen_anom",
            "bowen_anom_event_mean",
            "bowen_anomaly_event_mean",
            "event_bowen_anom_mean",
            "mean_bowen_anom",
            "Bowen_anom_mean",
            "Bowen_anom",
        ],
        desc="Bowen anomaly",
    )

    ef_col = find_col(
        df,
        [
            "EF_anom_event",
            "event_mean_ef_anom",
            "ef_anom_event_mean",
            "ef_anomaly_event_mean",
            "event_ef_anom_mean",
            "mean_ef_anom",
            "EF_anom_mean",
            "EF_anom",
        ],
        desc="EF anomaly",
    )

    h_col = find_col(
        df,
        [
            "H_anom_event",
            "event_mean_h_anom",
            "h_anom_event_mean",
            "event_mean_sensible_heat_anom",
            "sensible_heat_anom_event_mean",
            "mean_h_anom",
            "H_anom_mean",
            "H_anom",
        ],
        desc="H anomaly",
    )

    front_col = find_col(
        df,
        [
            "front_local_drying_contrast",
            "front_local_drying_speed_contrast",
            "front_local_drying",
            "front_drying_contrast",
        ],
        required=False,
        desc="front-local drying",
    )

    out = df.copy()
    out["state_num"] = out[state_col].apply(parse_state_value)

    out = out[out["state_num"].between(1, 6, inclusive="both")].copy()
    out["state_num"] = out["state_num"].astype(int)
    out["state_lab"] = out["state_num"].map(lambda i: f"S{i}")

    out["bowen_anom_evt"] = pd.to_numeric(out[bowen_col], errors="coerce")
    out["ef_anom_evt"] = pd.to_numeric(out[ef_col], errors="coerce")
    out["evap_supp_evt"] = -out["ef_anom_evt"]
    out["h_anom_evt"] = pd.to_numeric(out[h_col], errors="coerce")

    if front_col is not None:
        out["front_local_drying"] = pd.to_numeric(out[front_col], errors="coerce")
    else:
        out["front_local_drying"] = np.nan

    out = add_region_column(out)

    log(f"[INFO] Figure 3 event rows: {len(out):,}")

    return out


def load_fig3_spatial():
    if not FIG3_SPATIAL_CSV.exists():
        raise FileNotFoundError(f"Missing Figure 3 spatial CSV:\n{FIG3_SPATIAL_CSV}")

    df = pd.read_csv(FIG3_SPATIAL_CSV, low_memory=False)
    df = to_numeric_if_possible(df)

    lon_col = find_col(df, ["longitude", "lon", "x"], desc="longitude")
    lat_col = find_col(df, ["latitude", "lat", "y"], desc="latitude")

    bowen_col = find_col(
        df,
        [
            "Bowen_anom_dry_minus_wet",
            "dry_wet_bowen_anom",
            "dry_minus_wet_bowen_anom",
            "bowen_anom_contrast",
            "bowen_anomaly_contrast",
        ],
        desc="spatial Bowen contrast",
    )

    ef_col = find_col(
        df,
        [
            "EF_anom_dry_minus_wet",
            "dry_wet_ef_anom",
            "dry_minus_wet_ef_anom",
            "ef_anom_contrast",
            "ef_anomaly_contrast",
        ],
        desc="spatial EF contrast",
    )

    h_col = find_col(
        df,
        [
            "H_anom_dry_minus_wet",
            "dry_wet_h_anom",
            "dry_minus_wet_h_anom",
            "h_anom_contrast",
            "sensible_heat_anom_contrast",
        ],
        desc="spatial H contrast",
    )

    out = df.copy()
    out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")

    out["bowen_map"] = pd.to_numeric(out[bowen_col], errors="coerce")
    out["ef_map"] = pd.to_numeric(out[ef_col], errors="coerce")
    out["evap_supp_map"] = -out["ef_map"]
    out["h_map"] = pd.to_numeric(out[h_col], errors="coerce")

    # Exact variable-specific counts in your uploaded file.
    count_specs = [
        ("bowen", "Bowen_anom"),
        ("ef", "EF_anom"),
        ("h", "H_anom"),
    ]

    for short, prefix in count_specs:
        dry_col = find_col(
            out,
            [
                f"dry_{prefix}_count",
                f"dry_{prefix.lower()}_count",
                f"dry_{short}_count",
            ],
            required=False,
            desc=f"dry {short} count",
        )
        wet_col = find_col(
            out,
            [
                f"wet_{prefix}_count",
                f"wet_{prefix.lower()}_count",
                f"wet_{short}_count",
            ],
            required=False,
            desc=f"wet {short} count",
        )

        out[f"dry_{short}_n"] = pd.to_numeric(out[dry_col], errors="coerce") if dry_col is not None else np.nan
        out[f"wet_{short}_n"] = pd.to_numeric(out[wet_col], errors="coerce") if wet_col is not None else np.nan

    dry_cols = [c for c in ["dry_bowen_n", "dry_ef_n", "dry_h_n"] if c in out.columns]
    wet_cols = [c for c in ["wet_bowen_n", "wet_ef_n", "wet_h_n"] if c in out.columns]

    out["dry_n"] = out[dry_cols].min(axis=1) if dry_cols else np.nan
    out["wet_n"] = out[wet_cols].min(axis=1) if wet_cols else np.nan

    out = out[np.isfinite(out["lon"]) & np.isfinite(out["lat"])].copy()

    out["region_final"] = [
        assign_fallback_conus_region(x, y)
        for x, y in zip(out["lon"], out["lat"])
    ]

    log(f"[INFO] Figure 3 spatial rows: {len(out):,}")

    return out


# =============================================================================
# 6. FIGURE 1 COMPUTATION
# =============================================================================

COVARIATE_SPECS = [
    (
        "Z500",
        [
            "Z500",
            "z500",
            "Z500_height_m_anom_onset",
            "Z500_anom_onset",
            "z500_anom",
            "z500_mean",
            "event_z500",
        ],
    ),
    (
        "W500/ascent",
        [
            "W500",
            "w500",
            "W500_anom_onset",
            "w500_anom",
            "w500_mean",
            "ascent_anom",
            "ascent",
        ],
    ),
    (
        "T850",
        [
            "T850",
            "t850",
            "T850_anom_onset",
            "t850_anom",
            "t850_mean",
            "temperature_850",
            "event_t850",
        ],
    ),
    (
        "Area",
        [
            "log10_max_area_km2",
            "log_max_area",
            "max_area_km2",
            "max_area",
        ],
    ),
    (
        "Duration",
        [
            "duration_days",
            "duration",
            "event_duration",
        ],
    ),
    (
        "DOY",
        [
            "start_doy",
            "doy",
            "event_start_doy",
        ],
    ),
    (
        "Longitude",
        [
            "origin_lon",
            "longitude",
            "lon",
            "start_lon",
            "centroid_lon",
        ],
    ),
    (
        "Latitude",
        [
            "origin_lat",
            "latitude",
            "lat",
            "start_lat",
            "centroid_lat",
        ],
    ),
]


def _find_numeric_col(df, candidates, min_n=20):
    if df is None:
        return None

    norm_map = {norm_name(c): c for c in df.columns}

    for cand in candidates:
        key = norm_name(cand)
        if key in norm_map:
            c = norm_map[key]
            if pd.to_numeric(df[c], errors="coerce").notna().sum() >= min_n:
                return c

    for cand in candidates:
        key = norm_name(cand)
        for nk, orig in norm_map.items():
            if key and (key in nk or nk in key):
                if pd.to_numeric(df[orig], errors="coerce").notna().sum() >= min_n:
                    return orig

    return None


def get_shared_covariates(before_df, after_df):
    pairs = []

    for label, candidates in COVARIATE_SPECS:
        before_col = _find_numeric_col(before_df, candidates, min_n=20)
        after_col = _find_numeric_col(after_df, candidates, min_n=20)

        if before_col is not None and after_col is not None:
            pairs.append({
                "label": label,
                "before_col": before_col,
                "after_col": after_col,
            })

    return pairs


def compute_balance_df(matched, unmatched=None):
    before = unmatched if unmatched is not None else matched
    pairs = get_shared_covariates(before, matched)

    rows = []

    for item in pairs:
        label = item["label"]
        before_col = item["before_col"]
        after_col = item["after_col"]

        before_smd = standardized_mean_diff(
            before.loc[before["dry_vs_wet"] == 1, before_col],
            before.loc[before["dry_vs_wet"] == 0, before_col],
        )

        after_smd = standardized_mean_diff(
            matched.loc[matched["dry_vs_wet"] == 1, after_col],
            matched.loc[matched["dry_vs_wet"] == 0, after_col],
        )

        rows.append({
            "covariate": label,
            "before": before_smd,
            "after": after_smd,
            "before_col": before_col,
            "after_col": after_col,
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out["sorter"] = np.abs(out["after"])
        out = out.sort_values("sorter", ascending=False).drop(columns="sorter")

    out.to_csv(
        OUT_DIR / "Supplementary_Fig_1_panel_b_covariate_balance_mapped.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return out


def _get_common_support_data(matched, unmatched=None):
    """
    Robust common-support diagnostic.

    Critical correction:
    before/unmatched and after/matched are allowed to have different column names.
    They are mapped separately and then renamed to common labels.
    """
    before = unmatched if unmatched is not None else matched
    pairs = get_shared_covariates(before, matched)

    if len(pairs) < 2:
        log("[WARN] Common support skipped: fewer than two shared covariates.")
        return None

    # Use at most six variables to avoid visually unstable PC space.
    pairs = pairs[:6]
    labels = [p["label"] for p in pairs]

    before_cols = [p["before_col"] for p in pairs]
    after_cols = [p["after_col"] for p in pairs]

    before_df = before[before_cols + ["dry_vs_wet"]].copy()
    after_df = matched[after_cols + ["dry_vs_wet"]].copy()

    before_df = before_df.rename(columns={p["before_col"]: p["label"] for p in pairs})
    after_df = after_df.rename(columns={p["after_col"]: p["label"] for p in pairs})

    before_df["_sample"] = "before"
    after_df["_sample"] = "after"

    common_cols = labels + ["dry_vs_wet", "_sample"]

    all_df = pd.concat(
        [before_df[common_cols], after_df[common_cols]],
        axis=0,
        ignore_index=True,
    )

    for c in labels:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    all_df["dry_vs_wet"] = pd.to_numeric(all_df["dry_vs_wet"], errors="coerce")

    all_df = all_df.dropna(subset=labels + ["dry_vs_wet"]).copy()

    if len(all_df) < 50:
        log(f"[WARN] Common support skipped: only {len(all_df)} complete rows.")
        return None

    before_part = all_df[all_df["_sample"] == "before"].copy()
    after_part = all_df[all_df["_sample"] == "after"].copy()

    if len(before_part) > 6000:
        before_part = before_part.sample(6000, random_state=RANDOM_SEED)

    if len(after_part) > 6000:
        after_part = after_part.sample(6000, random_state=RANDOM_SEED + 1)

    all_df = pd.concat([before_part, after_part], axis=0, ignore_index=True)

    X = all_df[labels].values.astype(float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0

    Xz = (X - mu) / sd
    Xz = Xz - np.nanmean(Xz, axis=0, keepdims=True)

    try:
        _, _, Vt = np.linalg.svd(Xz, full_matrices=False)
    except np.linalg.LinAlgError:
        log("[WARN] Common support skipped: PCA SVD did not converge.")
        return None

    pcs = Xz @ Vt[:2].T

    all_df["pc1"] = pcs[:, 0]
    all_df["pc2"] = pcs[:, 1]

    loading_df = pd.DataFrame({
        "covariate": labels,
        "PC1_loading": Vt[0, :],
        "PC2_loading": Vt[1, :],
    })
    loading_df.to_csv(
        OUT_DIR / "Supplementary_Fig_1_panel_a_common_support_PCA_loadings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("[INFO] Common-support PCA covariates:")
    for item in pairs:
        log(f"       {item['label']}: before={item['before_col']} | after={item['after_col']}")

    return all_df, labels


def choose_fig1_metrics(matched):
    all_metrics = [
        ("Event voxels", "log10_event_voxels"),
        ("Maximum area", "log10_max_area_km2"),
        ("Net displacement", "log1p_net_displacement_km"),
        ("Front-local drying", "front_local_drying_contrast"),
    ]

    selected = []

    for label, col in all_metrics:
        if col in matched.columns:
            wide = paired_wide(matched, col)
            if wide is not None and len(wide) >= 20:
                selected.append((label, col))

    return selected


def compute_ridge_payload(matched):
    payload = []
    global_all = []

    for label, col in choose_fig1_metrics(matched):
        wide = paired_wide(matched, col)
        diff = wide["diff"].dropna().values

        sd = np.nanstd(diff, ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            continue

        z = diff / sd
        global_all.extend(list(z))

        est, lo, hi = bootstrap_mean_ci(z, seed=RANDOM_SEED + len(label))

        payload.append({
            "label": label,
            "column": col,
            "z": z,
            "est": est,
            "lo": lo,
            "hi": hi,
            "n": len(z),
            "color": METRIC_COLORS.get(label, GREY),
        })

    if len(global_all) == 0:
        return []

    g = np.asarray(global_all, dtype=float)
    qlo, qhi = np.nanquantile(g, [0.01, 0.99])
    pad = 0.2 * (qhi - qlo if qhi > qlo else 1.0)
    xlim = (qlo - pad, qhi + pad)

    for item in payload:
        xg, dens = smoothed_density(item["z"], bins=100, sigma=2.0, xlim=xlim)
        item["xgrid"] = xg
        item["density"] = dens

    return payload


def compute_trim_df(matched):
    rows = []
    trims = [0.00, 0.01, 0.05, 0.10, 0.15]

    for label, col in choose_fig1_metrics(matched):
        wide = paired_wide(matched, col)
        diff0 = wide["diff"].dropna().values

        for tr in trims:
            diff = diff0.copy()

            if tr > 0 and len(diff) >= 30:
                qlo = np.nanquantile(diff, tr)
                qhi = np.nanquantile(diff, 1 - tr)
                diff = diff[(diff >= qlo) & (diff <= qhi)]

            est, lo, hi = bootstrap_paired_std_effect(
                diff,
                seed=RANDOM_SEED + int(tr * 1000) + len(label),
            )

            rows.append({
                "metric": label,
                "trim": tr,
                "effect": est,
                "lo": lo,
                "hi": hi,
                "n": len(diff),
            })

    out = pd.DataFrame(rows)
    out.to_csv(
        OUT_DIR / "Supplementary_Fig_1_panel_d_trimming_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return out


def compute_leave_one_region_out(matched):
    if "region_final" not in matched.columns:
        return None

    metrics = choose_fig1_metrics(matched)
    specs = [("All pairs", None)] + [(f"Omit {r}", r) for r in REGION_ORDER]

    rows = []

    pair_region = (
        matched.sort_values("dry_vs_wet", ascending=False)
        .groupby("matched_pair_id")["region_final"]
        .first()
        .rename("pair_region")
    )

    tmp = matched.merge(
        pair_region,
        left_on="matched_pair_id",
        right_index=True,
        how="left",
    )

    for label, col in metrics:
        for spec_label, omit_region in specs:
            sub = tmp.copy()

            if omit_region is not None:
                sub = sub[sub["pair_region"] != omit_region].copy()

            wide = paired_wide(sub, col)

            if wide is None or len(wide) < 10:
                rows.append({
                    "metric": label,
                    "spec": spec_label,
                    "effect": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                    "n": 0,
                })
                continue

            diff = wide["diff"].dropna().values

            est, lo, hi = bootstrap_paired_std_effect(
                diff,
                seed=RANDOM_SEED + len(label) + len(spec_label),
            )

            rows.append({
                "metric": label,
                "spec": spec_label,
                "effect": est,
                "lo": lo,
                "hi": hi,
                "n": len(diff),
            })

    out = pd.DataFrame(rows)
    out.to_csv(
        OUT_DIR / "Supplementary_Fig_1_panel_e_leave_one_region_out.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return out


# =============================================================================
# 7. FIGURE 2 COMPUTATION
# =============================================================================

def summarize_states(evt, col):
    rows = []

    for s in range(1, 7):
        vals = evt.loc[evt["state_num"] == s, col]
        med, lo, hi = bootstrap_median_ci(vals, seed=RANDOM_SEED + s + len(col))

        rows.append({
            "state": s,
            "label": f"S{s}",
            "median": med,
            "lo": lo,
            "hi": hi,
            "n": int(pd.Series(vals).dropna().shape[0]),
        })

    return pd.DataFrame(rows)


def compute_energy_pathway(evt):
    out = []

    for s in range(1, 7):
        sub = evt[evt["state_num"] == s]

        x = pd.to_numeric(sub["evap_supp_evt"], errors="coerce").dropna().values
        y = pd.to_numeric(sub["h_anom_evt"], errors="coerce").dropna().values

        out.append({
            "state": s,
            "label": f"S{s}",
            "x": float(np.nanmedian(x)) if len(x) >= 10 else np.nan,
            "y": float(np.nanmedian(y)) if len(y) >= 10 else np.nan,
        })

    return pd.DataFrame(out)


def compute_state_slope_bootstrap(evt, col, n_boot=600, seed=RANDOM_SEED):
    vals = pd.to_numeric(evt[col], errors="coerce")
    mu = np.nanmean(vals.values.astype(float))
    sd = np.nanstd(vals.values.astype(float), ddof=1)

    if not np.isfinite(sd) or sd == 0:
        sd = 1.0

    state_arrays = {}

    for s in range(1, 7):
        arr = pd.to_numeric(
            evt.loc[evt["state_num"] == s, col],
            errors="coerce",
        ).dropna().values.astype(float)

        if len(arr) < 10:
            return np.array([])

        state_arrays[s] = (arr - mu) / sd

    rng = np.random.default_rng(seed + len(col))
    slopes = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        xs = []
        ys = []

        for s in range(1, 7):
            arr = state_arrays[s]
            b = rng.choice(arr, size=len(arr), replace=True)
            xs.append(s)
            ys.append(float(np.nanmedian(b)))

        slopes[i] = np.polyfit(xs, ys, 1)[0]

    return slopes


def compute_support_threshold_curves(spatial):
    thresholds = [5, 10, 15, 20, 25, 30, 40, 50]

    variables = [
        ("Bowen", "bowen_map", "bowen"),
        ("Evap. suppression", "evap_supp_map", "ef"),
        ("Sensible heat", "h_map", "h"),
    ]

    rows = []
    scale_map = {}

    for label, col, short in variables:
        arr = pd.to_numeric(spatial[col], errors="coerce").dropna().values.astype(float)
        sc = np.nanstd(arr, ddof=1)
        scale_map[label] = sc if np.isfinite(sc) and sc > 0 else 1.0

    for thr in thresholds:
        for label, col, short in variables:
            dry_count_col = f"dry_{short}_n"
            wet_count_col = f"wet_{short}_n"

            if (
                dry_count_col in spatial.columns and
                wet_count_col in spatial.columns and
                spatial[dry_count_col].notna().any() and
                spatial[wet_count_col].notna().any()
            ):
                sub = spatial[
                    (spatial[dry_count_col] >= thr) &
                    (spatial[wet_count_col] >= thr)
                ].copy()
            else:
                sub = spatial.copy()

            vals = pd.to_numeric(sub[col], errors="coerce").dropna().values.astype(float)

            if len(vals) < 20:
                rows.append({
                    "threshold": thr,
                    "metric": label,
                    "contrast_std": np.nan,
                    "n": 0,
                })
            else:
                med = float(np.nanmedian(vals))
                rows.append({
                    "threshold": thr,
                    "metric": label,
                    "contrast_std": med / scale_map[label],
                    "n": len(vals),
                })

    out = pd.DataFrame(rows)
    out.to_csv(
        OUT_DIR / "Supplementary_Fig_2_panel_c_support_threshold_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return out


def compute_regional_state_trajectories(evt):
    variables = [
        ("Bowen", "bowen_anom_evt"),
        ("Evap. suppression", "evap_supp_evt"),
        ("Sensible heat", "h_anom_evt"),
    ]

    rows = []

    for region in REGION_ORDER + ["CONUS"]:
        if region == "CONUS":
            subreg = evt.copy()
        else:
            subreg = evt[evt["region_final"] == region].copy()

        for label, col in variables:
            for s in range(1, 7):
                vals = pd.to_numeric(
                    subreg.loc[subreg["state_num"] == s, col],
                    errors="coerce",
                ).dropna().values

                rows.append({
                    "region": region,
                    "metric": label,
                    "state": s,
                    "median": float(np.nanmedian(vals)) if len(vals) >= 10 else np.nan,
                    "n": len(vals),
                })

    out = pd.DataFrame(rows)
    out.to_csv(
        OUT_DIR / "Supplementary_Fig_2_panel_e_regional_state_trajectories.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return out


# =============================================================================
# 8. PLOTTING HELPERS
# =============================================================================

def clear_titles(fig):
    """Remove all figure and axes titles, including nested subplot titles."""
    fig.suptitle("")
    for ax in fig.axes:
        try:
            ax.set_title("")
        except Exception:
            pass


def tighten_axis_fonts(ax):
    """Consistent heavier typography after the global +5 pt increase."""
    ax.tick_params(axis="both", labelsize=17.5, width=1.15, length=4.5)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)


def _density_contours(ax, x, y, color, bins=50, levels=(0.20, 0.45, 0.70), sigma=1.3):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    mask = x.notna() & y.notna()

    x = x[mask].values
    y = y[mask].values

    if len(x) < 20:
        return

    xmin, xmax = np.nanquantile(x, [0.01, 0.99])
    ymin, ymax = np.nanquantile(y, [0.01, 0.99])

    dx = (xmax - xmin) * 0.20 if xmax > xmin else 1.0
    dy = (ymax - ymin) * 0.20 if ymax > ymin else 1.0

    H, xe, ye = np.histogram2d(
        x, y,
        bins=bins,
        range=[[xmin - dx, xmax + dx], [ymin - dy, ymax + dy]],
        density=True,
    )

    H = smooth2d(H, sigma=sigma)

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

    if np.nanmax(H) <= 0:
        return

    levs = sorted([np.nanmax(H) * lv for lv in levels if np.nanmax(H) * lv > 0])

    if levs:
        ax.contour(
            Xc,
            Yc,
            H,
            levels=levs,
            colors=[color] * len(levs),
            linewidths=[1.0, 1.2, 1.5][:len(levs)],
            alpha=0.95,
        )


def plot_common_support_panel(ax_parent, matched, unmatched=None):
    add_panel_letter(ax_parent, "a", x=-0.14, y=1.08)
    ax_parent.axis("off")

    gs = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=ax_parent.get_subplotspec(),
        wspace=0.30,
    )

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    packed = _get_common_support_data(matched, unmatched)

    if packed is None:
        show_no_data(ax1, "No sufficient overlap data")
        ax2.axis("off")
        return

    support_df, cols = packed

    for ax, sample_name in [(ax1, "before"), (ax2, "after")]:
        sub = support_df[support_df["_sample"] == sample_name].copy()
        dry = sub[sub["dry_vs_wet"] == 1]
        wet = sub[sub["dry_vs_wet"] == 0]

        _density_contours(ax, dry["pc1"], dry["pc2"], DRY_COLOR)
        _density_contours(ax, wet["pc1"], wet["pc2"], WET_COLOR)

        ax.scatter(dry["pc1"].median(), dry["pc2"].median(), s=54, color=DRY_COLOR,
                   edgecolors="white", zorder=4)
        ax.scatter(wet["pc1"].median(), wet["pc2"].median(), s=54, color=WET_COLOR,
                   edgecolors="white", zorder=4)

        ax.axhline(0, color="0.85", lw=0.9)
        ax.axvline(0, color="0.85", lw=0.9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2", labelpad=8)
        ax.grid(False)
        tighten_axis_fonts(ax)

        # Panel a keeps the dry/wet count box. Panels c/e remove all n= labels.
        ax.text(
            0.02, 0.98,
            f"Dry n={len(dry):,}\nWet n={len(wet):,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=15.8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.9),
        )

    ax2.legend(
        handles=[
            Line2D([0], [0], color=DRY_COLOR, lw=2.1, label="Dry-start"),
            Line2D([0], [0], color=WET_COLOR, lw=2.1, label="Wet-start"),
        ],
        frameon=False, loc="lower right", fontsize=16.5,
    )

def plot_love_dumbbell(ax, balance_df):
    add_panel_letter(ax, "b", x=-0.16, y=1.08)

    if balance_df is None or balance_df.empty:
        show_no_data(ax, "No covariate-balance data")
        return

    df = balance_df.copy()
    df["abs_before"] = np.abs(df["before"])
    df["abs_after"] = np.abs(df["after"])
    df = df.sort_values("abs_after", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    ax.axvspan(0, 0.10, color="#eef7ee", zorder=0)
    ax.axvline(0.10, color="0.6", lw=1.2, ls="--")
    ax.text(0.02, 1.02, "Target balance zone", color="#3c763d",
            transform=ax.transAxes, fontsize=16, ha="left", va="bottom", clip_on=False)

    for i, row in df.iterrows():
        if np.isfinite(row["abs_before"]):
            ax.plot([row["abs_before"], row["abs_after"]], [i, i], color="0.72", lw=2.4, zorder=1)
            ax.scatter([row["abs_before"]], [i], s=52, facecolors="white",
                       edgecolors="0.35", linewidths=1.2, zorder=3)
        ax.scatter([row["abs_after"]], [i], s=60, color=GREEN,
                   edgecolors="white", linewidths=0.7, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(df["covariate"], fontsize=17.5)
    ax.set_xlabel("|SMD|")
    ax.grid(axis="x")
    ax.set_xlim(left=0)
    tighten_axis_fonts(ax)

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="0.35", label="Before"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markeredgecolor="white", label="After"),
        ],
        frameon=False, loc="lower right", fontsize=16.5,
    )

def plot_ridgeline_effects(ax, ridge_payload):
    add_panel_letter(ax, "c", x=-0.08, y=1.08)

    if not ridge_payload:
        show_no_data(ax, "No paired effects available")
        return

    order = ["Event voxels", "Maximum area", "Net displacement", "Front-local drying"]
    payload = []
    for name in order:
        for item in ridge_payload:
            if item["label"] == name:
                payload.append(item)

    heights = np.arange(len(payload))[::-1]
    x_arrays = [p["xgrid"] for p in payload if p.get("xgrid") is not None]
    if len(x_arrays) == 0:
        show_no_data(ax, "No ridge density available")
        return

    all_x = np.concatenate(x_arrays)
    xmin, xmax = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    ax.axvline(0, color="0.55", lw=1.1, ls="--", zorder=0)
    max_height = max(np.nanmax(p["density"]) for p in payload if p.get("density") is not None)
    scale = 0.65 / max_height if max_height > 0 else 1.0

    for y0, item in zip(heights, payload):
        x = item["xgrid"]
        dens = item["density"] * scale
        color = item["color"]
        ax.fill_between(x, y0, y0 + dens, color=color, alpha=0.35, lw=0)
        ax.plot(x, y0 + dens, color=color, lw=2.0)
        ax.hlines(y0, xmin, xmax, color="0.93", lw=0.9, zorder=0)
        ax.plot([item["lo"], item["hi"]], [y0 - 0.07, y0 - 0.07],
                color="0.15", lw=3.0, solid_capstyle="round")
        ax.scatter([item["est"]], [y0 - 0.07], s=58, color="0.15", zorder=5,
                   edgecolors="white", linewidths=0.6)
        disp_label = SHORT_METRIC_LABELS.get(item["label"], item["label"])
        ax.text(xmin - 0.02 * (xmax - xmin), y0 + 0.18, disp_label,
                ha="right", va="center", fontsize=17.0)
        ax.text(xmax + 0.02 * (xmax - xmin), y0 + 0.18, f"d={item['est']:.2f}",
                ha="left", va="center", fontsize=15.6, color="0.25")

    ax.set_xlim(xmin - 0.11 * (xmax - xmin), xmax + 0.14 * (xmax - xmin))
    ax.set_ylim(-0.4, max(heights) + 0.95)
    ax.set_yticks([])
    ax.set_xlabel("Paired std. contrast")
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    tighten_axis_fonts(ax)

def draw_heatmap(
    ax,
    data,
    row_labels,
    col_labels,
    title,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    center=0,
    annotations=None,
    cbar=True,
    cbar_label="",
    text_size=9.3,
):
    arr = np.asarray(data, dtype=float)
    finite = np.isfinite(arr)

    if finite.any():
        vmax0 = np.nanquantile(np.abs(arr[finite]), 0.98)
    else:
        vmax0 = 1.0

    if not np.isfinite(vmax0) or vmax0 == 0:
        vmax0 = 1.0

    if vmin is None:
        vmin = -vmax0
    if vmax is None:
        vmax = vmax0

    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if annotations is not None:
                txt = annotations[i][j]
            else:
                txt = "" if not np.isfinite(arr[i, j]) else f"{arr[i, j]:.2f}"

            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=text_size, color="black")

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)

    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.055, pad=0.03)
        cb.set_label(cbar_label, fontsize=16.8)
        cb.ax.tick_params(labelsize=15.5, width=1.0, length=3.5)

    return im


def plot_trim_heatmap(ax, trim_df):
    add_panel_letter(ax, "d", x=-0.20, y=1.08)

    if trim_df is None or trim_df.empty:
        show_no_data(ax, "No trimming sensitivity available")
        return

    order = ["Event voxels", "Maximum area", "Front-local drying", "Net displacement"]
    trims = sorted(trim_df["trim"].dropna().unique())
    mat = np.full((len(order), len(trims)), np.nan)
    annot = [["" for _ in trims] for _ in order]

    for i, metric in enumerate(order):
        for j, tr in enumerate(trims):
            sub = trim_df[(trim_df["metric"] == metric) & (trim_df["trim"] == tr)]
            if len(sub) == 0:
                continue
            val = float(sub["effect"].iloc[0])
            mat[i, j] = val
            annot[i][j] = f"{val:.2f}"

    row_labels = [SHORT_METRIC_LABELS.get(m, m) for m in order]
    draw_heatmap(
        ax, mat, row_labels, [f"{int(t * 100)}%" for t in trims], title="",
        cmap="RdBu_r", cbar=True, cbar_label="Effect",
        annotations=annot, text_size=13.6,
    )
    ax.set_xlabel("Trim")
    tighten_axis_fonts(ax)

def plot_leave_one_region_out_panel(ax_parent, loo_df):
    add_panel_letter(ax_parent, "e", x=-0.06, y=1.07)
    ax_parent.axis("off")

    if loo_df is None or loo_df.empty:
        gs = GridSpecFromSubplotSpec(1, 1, subplot_spec=ax_parent.get_subplotspec())
        ax = plt.subplot(gs[0, 0])
        show_no_data(ax, "No leave-one-region-out results")
        return

    order_metrics = ["Event voxels", "Maximum area", "Front-local drying", "Net displacement"]
    subgs = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=ax_parent.get_subplotspec(), wspace=0.42, hspace=1.22,
    )
    axes = [plt.subplot(subgs[0, 0]), plt.subplot(subgs[0, 1]),
            plt.subplot(subgs[1, 0]), plt.subplot(subgs[1, 1])]
    spec_order = ["All pairs"] + [f"Omit {r}" for r in REGION_ORDER]

    for ax, metric in zip(axes, order_metrics):
        sub = loo_df[loo_df["metric"] == metric].copy()
        if sub.empty:
            show_no_data(ax, f"No {metric}")
            continue
        sub["spec"] = pd.Categorical(sub["spec"], categories=spec_order, ordered=True)
        sub = sub.sort_values("spec")
        y = np.arange(len(sub))[::-1]
        allrow = sub[sub["spec"] == "All pairs"]
        ref = float(allrow["effect"].iloc[0]) if len(allrow) else np.nan
        if np.isfinite(ref):
            ax.axvline(ref, color="0.65", lw=1.1, ls="--")
        ax.hlines(y, sub["lo"], sub["hi"], color="0.25", lw=2.0)
        ax.scatter(sub["effect"], y, s=52, color=METRIC_COLORS.get(metric, GREY),
                   edgecolor="white", linewidth=0.7, zorder=3)
        def _abbr_spec(v):
            v = str(v)
            if v == "All pairs":
                return "All"
            if v.startswith("Omit "):
                vv = v.replace("Omit ", "")
                return REGION_ABBR.get(vv, vv)
            return REGION_ABBR.get(v, v)
        ax.set_yticks(y)
        ax.set_yticklabels([_abbr_spec(v) for v in sub["spec"]], fontsize=13.2)
        ax.set_xlabel("Std. effect")
        ax.grid(axis="x")
        tighten_axis_fonts(ax)
        ax.tick_params(axis="y", pad=3)
        # n= labels removed.

# =============================================================================
# 9. FIGURE 2 PLOTTING
# =============================================================================

def plot_state_ribbon(ax, summary_df, title, color, ylabel):
    x = summary_df["state"].values
    med = summary_df["median"].values
    lo = summary_df["lo"].values
    hi = summary_df["hi"].values

    ax.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)
    ax.plot(x, med, color=color, lw=2.4)
    ax.scatter(x, med, s=38, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0, color="0.70", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in x])
    ax.set_xlabel("State")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    tighten_axis_fonts(ax)
    # Sample-size labels removed.

def plot_slope_ridgelines(ax, slope_dict):
    add_panel_letter(ax, "b", x=-0.16, y=1.08)
    order = ["Bowen", "Evap. suppression", "Sensible heat"]
    items = []
    allvals = []
    for lab in order:
        arr = np.asarray(slope_dict.get(lab, np.array([])), dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            items.append((lab, arr))
            allvals.extend(arr.tolist())
    if not items:
        show_no_data(ax, "No gradient bootstrap distributions")
        return
    allvals = np.asarray(allvals)
    lo, hi = np.nanquantile(allvals, [0.01, 0.99])
    pad = 0.15 * (hi - lo if hi > lo else 1)
    xlim = (lo - pad, hi + pad)
    heights = np.arange(len(items))[::-1]
    temp = []
    maxh = 0
    for lab, arr in items:
        xg, dens = smoothed_density(arr, bins=90, sigma=1.8, xlim=xlim)
        temp.append((lab, arr, xg, dens))
        maxh = max(maxh, np.nanmax(dens))
    scale = 0.62 / maxh if maxh > 0 else 1
    ax.axvline(0, color="0.60", lw=1.1, ls="--")
    for y0, (lab, arr, xg, dens) in zip(heights, temp):
        color = ENERGY_COLORS[lab]
        denss = dens * scale
        ax.fill_between(xg, y0, y0 + denss, color=color, alpha=0.35, lw=0)
        ax.plot(xg, y0 + denss, color=color, lw=2.0)
        med = np.nanmedian(arr)
        lo95 = np.nanpercentile(arr, 2.5)
        hi95 = np.nanpercentile(arr, 97.5)
        ax.plot([lo95, hi95], [y0 - 0.07, y0 - 0.07], color="0.15", lw=2.8)
        ax.scatter([med], [y0 - 0.07], s=52, color="0.15",
                   edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(xlim[0] - 0.02 * (xlim[1] - xlim[0]), y0 + 0.16, lab,
                ha="right", va="center", fontsize=17.3)
    ax.set_xlim(xlim[0] - 0.08 * (xlim[1] - xlim[0]), xlim[1] + 0.08 * (xlim[1] - xlim[0]))
    ax.set_ylim(-0.35, max(heights) + 0.92)
    ax.set_yticks([])
    ax.grid(axis="x")
    ax.set_xlabel("S1-S6 slope")
    ax.spines["left"].set_visible(False)
    tighten_axis_fonts(ax)

def plot_threshold_trajectories(ax, support_df):
    add_panel_letter(ax, "c", x=-0.17, y=1.08)
    if support_df is None or support_df.empty:
        show_no_data(ax, "No support-threshold robustness data")
        return
    for lab in ["Bowen", "Evap. suppression", "Sensible heat"]:
        sub = support_df[support_df["metric"] == lab].sort_values("threshold")
        if not sub.empty:
            ax.plot(sub["threshold"], sub["contrast_std"], lw=2.4, marker="o", ms=5.2,
                    color=ENERGY_COLORS[lab], label=lab)
    ax.axhline(0, color="0.65", lw=1.0, ls="--")
    ax.set_xlabel("Min support")
    ax.set_ylabel("Std. contrast")
    ax.grid(True)
    ax.legend(frameon=False, loc="best", fontsize=16.5)
    tighten_axis_fonts(ax)

def plot_hexbin_regime(ax, evt, pathway):
    add_panel_letter(ax, "d", x=-0.08, y=1.08)
    x = pd.to_numeric(evt["evap_supp_evt"], errors="coerce")
    y = pd.to_numeric(evt["h_anom_evt"], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 50:
        show_no_data(ax, "No regime-space data")
        return
    hb = ax.hexbin(x[mask], y[mask], gridsize=38, bins="log", mincnt=1,
                   cmap="Greys", linewidths=0)
    cb = plt.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Density", fontsize=15.5)
    cb.ax.tick_params(labelsize=14.8)
    ax.axvline(0, color="0.75", lw=1.0, ls="--")
    ax.axhline(0, color="0.75", lw=1.0, ls="--")
    p = pathway.sort_values("state")
    ax.plot(p["x"], p["y"], color="0.45", lw=1.8, zorder=3)
    for _, row in p.iterrows():
        if np.isfinite(row["x"]) and np.isfinite(row["y"]):
            col = STATE_COLORS[row["label"]]
            ax.scatter(row["x"], row["y"], s=64, color=col, edgecolor="white",
                       linewidth=0.8, zorder=4)
            ax.text(row["x"] + 0.012, row["y"] + 0.8, row["label"],
                    color=col, fontsize=16.5, weight="bold")
    rho = spearman_rank_corr(x[mask], y[mask])
    ax.text(0.03, 0.97, f"Spearman rho = {rho:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=15.8,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.8", alpha=0.95))
    ax.set_xlabel("-EF'")
    ax.set_ylabel("H' (W m$^{-2}$)")
    tighten_axis_fonts(ax)

def plot_regional_state_trajectories(ax_parent, regional_state_df):
    add_panel_letter(ax_parent, "e", x=-0.05, y=1.07)
    ax_parent.axis("off")
    if regional_state_df is None or regional_state_df.empty:
        gs = GridSpecFromSubplotSpec(1, 1, subplot_spec=ax_parent.get_subplotspec())
        ax = plt.subplot(gs[0, 0])
        show_no_data(ax, "No regional-state data")
        return
    subgs = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax_parent.get_subplotspec(), wspace=0.34)
    metrics = ["Bowen", "Evap. suppression", "Sensible heat"]
    ylabels = ["Bowen", "-EF'", "H' (W m$^{-2}$)"]
    for j, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = plt.subplot(subgs[0, j])
        sub = regional_state_df[regional_state_df["metric"] == metric].copy()
        for region in REGION_ORDER:
            rr = sub[sub["region"] == region].sort_values("state")
            if rr["median"].notna().sum() >= 3:
                ax.plot(
                    rr["state"], rr["median"],
                    color=REGION_COLORS.get(region, "0.75"),
                    lw=1.35, alpha=0.62, zorder=1,
                )
        rr = sub[sub["region"] == "CONUS"].sort_values("state")
        ax.plot(rr["state"], rr["median"], color="0.15", lw=3.0,
                zorder=3)
        ax.plot(rr["state"], rr["median"], color=ENERGY_COLORS[metric], lw=2.2,
                marker="o", ms=5.6, zorder=4)
        ax.axhline(0, color="0.70", lw=1.0, ls="--")
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels([f"S{i}" for i in range(1, 7)])
        ax.set_xlabel("State")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        tighten_axis_fonts(ax)
    handles = [
        Line2D([0], [0], color=REGION_COLORS[r], lw=1.8, alpha=0.85, label=REGION_ABBR[r])
        for r in REGION_ORDER
    ]
    handles.append(Line2D([0], [0], color="0.15", lw=3.0, marker="o", ms=5.4, label="CONUS"))
    ax_parent.legend(
        handles=handles,
        frameon=False,
        ncol=4,
        loc="upper left",
        bbox_to_anchor=(0.015, 1.01),
        fontsize=12.8,
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.45,
    )

# =============================================================================
# 10. FIGURE BUILDERS
# =============================================================================

def make_supplementary_fig_1(matched, unmatched=None):
    balance_df = compute_balance_df(matched, unmatched)
    ridge_payload = compute_ridge_payload(matched)
    trim_df = compute_trim_df(matched)
    loo_df = compute_leave_one_region_out(matched)

    # Taller figure and explicit empty columns: b and d are moved right;
    # bottom row is given more height so the two rows in panel e no longer overlap.
    fig = plt.figure(figsize=(17.8, 16.4))
    gs = GridSpec(
        3, 24, figure=fig,
        height_ratios=[0.96, 1.00, 2.75],
        wspace=0.78, hspace=0.88,
    )
    ax_a = fig.add_subplot(gs[0, 0:10])
    ax_b = fig.add_subplot(gs[0, 14:24])
    ax_c = fig.add_subplot(gs[1, 0:17])
    ax_d = fig.add_subplot(gs[1, 19:24])
    ax_e = fig.add_subplot(gs[2, 0:24])

    plot_common_support_panel(ax_a, matched, unmatched)
    plot_love_dumbbell(ax_b, balance_df)
    plot_ridgeline_effects(ax_c, ridge_payload)
    plot_trim_heatmap(ax_d, trim_df)
    plot_leave_one_region_out_panel(ax_e, loo_df)

    clear_titles(fig)
    fig.subplots_adjust(left=0.092, right=0.985, top=0.975, bottom=0.065)
    save_figure(fig, "Supplementary_Fig_1_circulation_control_robustness_redesigned_v2")

def make_supplementary_fig_2(evt, spatial):
    state_bowen = summarize_states(evt, "bowen_anom_evt")
    state_ef = summarize_states(evt, "evap_supp_evt")
    state_h = summarize_states(evt, "h_anom_evt")

    slope_dict = {
        "Bowen": compute_state_slope_bootstrap(evt, "bowen_anom_evt"),
        "Evap. suppression": compute_state_slope_bootstrap(evt, "evap_supp_evt"),
        "Sensible heat": compute_state_slope_bootstrap(evt, "h_anom_evt"),
    }
    support_df = compute_support_threshold_curves(spatial)
    pathway = compute_energy_pathway(evt)

    fig = plt.figure(figsize=(17.4, 9.8))
    gs = GridSpec(
        2, 4, figure=fig,
        height_ratios=[1.00, 1.08],
        width_ratios=[1.05, 1.00, 1.08, 1.08],
        hspace=0.80, wspace=0.48,
    )
    outer_a = fig.add_subplot(gs[0, 0:4])
    outer_a.axis("off")
    add_panel_letter(outer_a, "a", x=-0.035, y=1.08)
    subgs_a = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_a.get_subplotspec(), wspace=0.42)
    ax_a1 = fig.add_subplot(subgs_a[0, 0])
    ax_a2 = fig.add_subplot(subgs_a[0, 1])
    ax_a3 = fig.add_subplot(subgs_a[0, 2])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2:4])

    plot_state_ribbon(ax_a1, state_bowen, "", BROWN, "Bowen")
    plot_state_ribbon(ax_a2, state_ef, "", GREEN, "-EF'")
    plot_state_ribbon(ax_a3, state_h, "", ORANGE, "H' (W m$^{-2}$)")
    plot_slope_ridgelines(ax_b, slope_dict)
    plot_threshold_trajectories(ax_c, support_df)
    plot_hexbin_regime(ax_d, evt, pathway)

    clear_titles(fig)
    fig.subplots_adjust(left=0.080, right=0.985, top=0.975, bottom=0.090)
    save_figure(fig, "Supplementary_Fig_2_surface_energy_regime_robustness_redesigned_v2")

# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    log("=" * 98)
    log("[INFO] Building redesigned Supplementary Fig. 1 and Supplementary Fig. 2")
    log(f"[INFO] Output directory: {OUT_DIR}")
    log("=" * 98)

    matched = load_matched_fig2()
    unmatched = load_unmatched_fig2()
    evt = load_fig3_event()
    spatial = load_fig3_spatial()

    log("[STEP] Building Supplementary Fig. 1 (redesigned v2)")
    make_supplementary_fig_1(matched, unmatched)

    log("[STEP] Building Supplementary Fig. 2 (redesigned v2)")
    make_supplementary_fig_2(evt, spatial)

    log("=" * 98)
    log("[DONE] All redesigned supplementary figures finished.")
    log("=" * 98)


if __name__ == "__main__":
    main()
