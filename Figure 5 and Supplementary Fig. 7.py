# -*- coding: utf-8 -*-
"""
NCC-polished Figure 5 and Supplementary Figure
==============================================

Redraws:
1. Figure5_all_event_front_mechanism_NCC_polished.png
2. Supplementary_Fig_all_event_front_mechanism_NCC_polished.png

This version fixes the layout problems in the Supplementary Figure:
1. Panel labels are placed outside the upper-left corner of each panel.
2. Top-row maps have the same width, height, title style and colour-bar position.
3. Lower-row panels b/c/d have the same height and top alignment.
4. All legends are moved to a dedicated bottom legend row.
5. The figure height is increased to avoid crowding.
6. The vertical, crowded "Front dry-state maintenance" label is removed.
   The event-morphology scatter now uses a horizontal colour bar.
7. Legends for b/c/d are horizontally aligned in the same legend row.
8. No raw-event extraction is rerun. Only existing cache files are read.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

try:
    from scipy import stats
except Exception:
    stats = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS
# =============================================================================

CACHE_ROOT_CANDIDATES = [
    Path(r"D:\第二篇\第二篇最终20260407版本\最终20260428\最终版本代码\Figure5_front_advection_land_coupling_from_raw_events"),
    Path(r"E:\第二篇的20260513\Figure5_front_advection_land_coupling_from_raw_events"),
]

CACHE_ROOT = None
for p in CACHE_ROOT_CANDIDATES:
    if (p / "front_event_diagnostics_with_model_variables.csv").exists() or (
        p / "front_event_diagnostics_C26_from_raw_events.csv"
    ).exists():
        CACHE_ROOT = p
        break

if CACHE_ROOT is None:
    CACHE_ROOT = CACHE_ROOT_CANDIDATES[0]

EVENT_MODEL_CSV = CACHE_ROOT / "front_event_diagnostics_with_model_variables.csv"
EVENT_RAW_CSV = CACHE_ROOT / "front_event_diagnostics_C26_from_raw_events.csv"
SPATIAL_CSV = CACHE_ROOT / "front_cell_spatial_aggregates_C26.csv"

OUT_DIR = CACHE_ROOT / "_Figure5_all_event_mechanism_NCC_polished_no_rerun"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_PNG = OUT_DIR / "Figure5_all_event_front_mechanism_NCC_polished.png"
MAIN_PDF = OUT_DIR / "Figure5_all_event_front_mechanism_NCC_polished.pdf"

SUPP_PNG = OUT_DIR / "Supplementary_Fig_all_event_front_mechanism_NCC_polished.png"
SUPP_PDF = OUT_DIR / "Supplementary_Fig_all_event_front_mechanism_NCC_polished.pdf"

EVENT_TABLE_OUT = OUT_DIR / "event_table_used_for_Figure5_NCC_polished.csv"
REGION_COEF_OUT = OUT_DIR / "regional_coefficients_Figure5_NCC_polished.csv"
R2_OUT = OUT_DIR / "unique_R2_Figure5_NCC_polished.csv"
NOTE_OUT = OUT_DIR / "Figure5_NCC_polished_interpretation_note.txt"


# =============================================================================
# 2. STYLE
# =============================================================================

DPI = 500
RANDOM_SEED = 42
BOOT_N = 600

CONUS_EXTENT = [-125, -66, 24, 50]

COLOR_LAND = "#8c510a"
COLOR_TADV = "#d95f02"
COLOR_VIMD = "#1b9e77"
COLOR_TRANSPORT = "#2c7fb8"
COLOR_CIRC = "#756bb1"
COLOR_GEOM = "#4d4d4d"
COLOR_INTENSITY = "#b2182b"
COLOR_MIXED = "#9e9e9e"
COLOR_WEAK = "#d9d9d9"

DOM_CMAP = LinearSegmentedColormap.from_list(
    "transport_to_land",
    ["#2166ac", "#f7f7f7", "#8c510a"],
    N=256,
)

DRY_CMAP = LinearSegmentedColormap.from_list(
    "dry_maintenance",
    ["#fff7bc", "#fec44f", "#f03b20", "#7f0000"],
    N=256,
)

MAIN_FIGSIZE = (16.2, 12.0)
SUPP_FIGSIZE = (18.2, 13.2)

FS_BASE = 20.0
FS_TITLE = 21.0
FS_LABEL = 20.0
FS_TICK = 17.2
FS_MAP_TICK = 15.2
FS_LEGEND = 16.0
FS_LEGEND_SMALL = 14.5
FS_CBAR_LABEL = 16.8
FS_CBAR_TICK = 15.0
FS_NOTE = 15.5
FS_STAR = 17.0
FS_PANEL = 27.0

# True = supplementary top three maps are grouped as panel a;
# lower panels are b/c/d. This matches your current figure logic.
# If you want six independent labels a-f, change this to False.
GROUP_TOP_MAPS_AS_PANEL_A = True

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FS_BASE,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.linewidth": 1.20,
    "xtick.major.size": 5.4,
    "ytick.major.size": 5.4,
    "xtick.major.width": 1.10,
    "ytick.major.width": 1.10,
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def log(msg: str):
    print(msg, flush=True)


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {label}:\n{path}\n\n"
            "This script reads existing cache files only. "
            "Run the previous front-advection diagnostic script first."
        )


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, label, x=-0.145, y=1.105, size=FS_PANEL):
    """
    Axis-coordinate panel label.
    Kept for compatibility, but the final supplementary figure now uses
    add_panel_label_figure(), which is more robust.
    """
    if label is None or str(label).strip() == "":
        return

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
        zorder=100,
    )


def add_panel_label_figure(fig, ax, label, dx=0.028, dy=0.010, size=FS_PANEL):
    """
    Place panel letters in figure coordinates, outside the axes.
    This prevents labels from overlapping titles, map frames, y-axis labels,
    or colour bars.
    """
    if label is None or str(label).strip() == "":
        return

    bbox = ax.get_position()
    fig.text(
        bbox.x0 - dx,
        bbox.y1 + dy,
        label,
        fontsize=size,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=200,
    )


def add_fixed_horizontal_cbar(
    fig,
    ax,
    mappable,
    label,
    ticks=None,
    width_frac=0.54,
    height=0.014,
    y_offset=0.040,
    labelpad=2.0,
):
    """
    Add a horizontal colour bar using fixed figure coordinates.
    Unlike plt.colorbar(..., ax=ax), this does not shrink or distort the axes.
    """
    bbox = ax.get_position()
    cbar_width = bbox.width * width_frac
    cbar_x0 = bbox.x0 + (bbox.width - cbar_width) / 2.0
    cbar_y0 = bbox.y0 - y_offset

    cax = fig.add_axes([cbar_x0, cbar_y0, cbar_width, height])

    cb = fig.colorbar(
        mappable,
        cax=cax,
        orientation="horizontal",
        extend="both",
    )

    if ticks is not None:
        cb.set_ticks(ticks)

    cb.set_label(label, fontsize=FS_CBAR_LABEL, labelpad=labelpad)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK, length=3.0, pad=2.0)

    return cb


def add_fixed_horizontal_cbar_noextend(
    fig,
    ax,
    mappable,
    label,
    ticks=None,
    width_frac=0.58,
    height=0.014,
    y_offset=0.070,
    labelpad=2.0,
):
    """
    Horizontal colour bar for the lower scatter panel.
    No vertical label is used.
    """
    bbox = ax.get_position()
    cbar_width = bbox.width * width_frac
    cbar_x0 = bbox.x0 + (bbox.width - cbar_width) / 2.0
    cbar_y0 = bbox.y0 - y_offset

    cax = fig.add_axes([cbar_x0, cbar_y0, cbar_width, height])

    cb = fig.colorbar(
        mappable,
        cax=cax,
        orientation="horizontal",
        extend="neither",
    )

    if ticks is not None:
        cb.set_ticks(ticks)

    cb.set_label(label, fontsize=FS_CBAR_LABEL, labelpad=labelpad)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK, length=3.0, pad=2.0)

    return cb


def zscore(s):
    x = to_num(s)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)

    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)

    return (x - mu) / sd


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


def fmt_p_threshold(p):
    if not np.isfinite(p):
        return "P=NA"

    if p < 0.05:
        return "P<0.05"

    return f"P={p:.3f}"


def robust_abs_lim(x, q=0.985, floor=0.05):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return floor

    v = np.nanquantile(np.abs(arr), q)

    if not np.isfinite(v) or v <= 0:
        return floor

    return max(float(v), floor)


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(w) & (w > 0)

    if m.sum() == 0:
        return np.nan

    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def weighted_quantile(values, weights, qs):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    qs = np.asarray(qs, dtype=float)

    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if m.sum() == 0:
        return np.full(len(qs), np.nan)

    v = values[m]
    w = weights[m]

    order = np.argsort(v)
    v = v[order]
    w = w[order]

    cdf = np.cumsum(w) / np.sum(w)

    return np.interp(qs, cdf, v)


def safe_spearman(x, y):
    if stats is None:
        return np.nan, np.nan

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 20:
        return np.nan, np.nan

    res = stats.spearmanr(x[m], y[m])

    return float(res.correlation), float(res.pvalue)


def coord_edges(vals):
    vals = np.asarray(vals, dtype=float)
    vals = np.sort(vals)

    if vals.size < 2:
        d = 0.25
        return np.array([vals[0] - d, vals[0] + d])

    half = np.diff(vals) / 2

    edges = np.empty(vals.size + 1)
    edges[1:-1] = vals[:-1] + half
    edges[0] = vals[0] - half[0]
    edges[-1] = vals[-1] + half[-1]

    return edges


def smooth_nan_field(Z, sigma=0.85):
    if gaussian_filter is None:
        return Z

    Z = np.asarray(Z, dtype=float)
    mask = np.isfinite(Z)

    if mask.sum() == 0:
        return Z

    num = gaussian_filter(np.where(mask, Z, 0.0), sigma=sigma)
    den = gaussian_filter(mask.astype(float), sigma=sigma)

    out = num / np.where(den <= 0, np.nan, den)

    return out


def setup_conus_map(ax, ticksize=FS_MAP_TICK, show_lat_labels=True):
    if HAS_CARTOPY:
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())

        ax.coastlines(resolution="50m", linewidth=0.90, color="0.30")
        ax.add_feature(cfeature.BORDERS, linewidth=0.60, edgecolor="0.50")
        ax.add_feature(cfeature.STATES, linewidth=0.48, edgecolor="0.62")

        ax.set_xticks([-120, -110, -100, -90, -80, -70], crs=ccrs.PlateCarree())
        ax.set_yticks([25, 30, 35, 40, 45, 50], crs=ccrs.PlateCarree())

        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())

        ax.tick_params(labelsize=ticksize, pad=2.5)

        if not show_lat_labels:
            ax.tick_params(axis="y", labelleft=False, left=False)
    else:
        ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
        ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
        ax.set_xticks([-120, -110, -100, -90, -80, -70])
        ax.set_yticks([25, 30, 35, 40, 45, 50])
        ax.tick_params(labelsize=ticksize, pad=2.5)

        if not show_lat_labels:
            ax.tick_params(axis="y", labelleft=False, left=False)


def enforce_minimum_fontsizes(fig):
    min_size = FS_LEGEND_SMALL

    for ax in fig.axes:
        ax.title.set_fontsize(max(ax.title.get_fontsize(), FS_TITLE))
        ax.xaxis.label.set_fontsize(max(ax.xaxis.label.get_fontsize(), FS_LABEL))
        ax.yaxis.label.set_fontsize(max(ax.yaxis.label.get_fontsize(), FS_LABEL))

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(max(tick.get_fontsize(), min_size))

        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(max(txt.get_fontsize(), FS_LEGEND_SMALL))
            if leg.get_title() is not None:
                leg.get_title().set_fontsize(max(leg.get_title().get_fontsize(), FS_LEGEND_SMALL))

    for txt in fig.findobj(match=matplotlib.text.Text):
        if txt.get_text() and txt.get_fontsize() < min_size:
            txt.set_fontsize(min_size)


def savefig(fig, png, pdf):
    enforce_minimum_fontsizes(fig)

    try:
        fig.align_labels()
    except Exception:
        pass

    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.040)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.040)
    plt.close(fig)

    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")


# =============================================================================
# 4. LOAD AND PREPARE EVENT TABLE
# =============================================================================

def load_event_cache():
    if EVENT_MODEL_CSV.exists():
        log(f"[READ] {EVENT_MODEL_CSV}")
        return pd.read_csv(EVENT_MODEL_CSV, low_memory=False)

    require_file(EVENT_RAW_CSV, "front_event_diagnostics_C26_from_raw_events.csv")

    log(f"[READ] {EVENT_RAW_CSV}")

    return pd.read_csv(EVENT_RAW_CSV, low_memory=False)


def detect_intensity_column(df):
    candidates = [
        "front_heat_excess_mean",
        "heat_excess_mean",
        "event_heat_excess_mean",
        "heat_excess_t95_mean",
        "front_temp_air_anom",
        "temp_air_anom_event_mean",
        "event_temp_air_anom_mean",
        "front_T850_anom",
        "T850_anom_event_mean",
        "front_H_anom",
    ]

    lower = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower:
            c = lower[cand.lower()]
            s = to_num(df[c])

            if s.notna().sum() > 100 and np.nanstd(s) > 0:
                return c

    for c in df.columns:
        lc = c.lower()

        if any(k in lc for k in ["heat_excess", "temp_air_anom", "t850_anom"]):
            s = to_num(df[c])

            if s.notna().sum() > 100 and np.nanstd(s) > 0:
                return c

    return None


def prepare_event_table(ev):
    d = ev.copy()

    numeric_cols = [
        "year", "month", "doy", "event_id",
        "front_dry_fraction",
        "front_drying_tendency",
        "front_soil_moist_change",
        "front_Tadv850_K_day",
        "front_VIMD_anom",
        "front_Z500_height_m_anom",
        "front_ASC500_anom",
        "front_cell_count",
        "object_grid_days",
        "duration_days",
        "max_daily_extent",
        "event_lon",
        "event_lat",
        "local_land_coupling_index",
        "front_Bowen_anom",
        "front_H_anom",
        "front_EF_anom",
    ]

    for c in numeric_cols:
        if c in d.columns:
            d[c] = to_num(d[c])

    required = [
        "front_dry_fraction",
        "front_Tadv850_K_day",
        "front_VIMD_anom",
        "front_cell_count",
    ]

    missing = [c for c in required if c not in d.columns]

    if missing:
        raise KeyError(f"Missing required columns in event cache: {missing}")

    d["y_dryfrac"] = d["front_dry_fraction"]
    d["y_dryfrac_z"] = zscore(d["y_dryfrac"])

    if "local_land_coupling_index" in d.columns and d["local_land_coupling_index"].notna().sum() > 100:
        d["land_index"] = d["local_land_coupling_index"]
    else:
        comps = []

        if "front_Bowen_anom" in d.columns and d["front_Bowen_anom"].notna().sum() > 100:
            d["_z_Bowen"] = zscore(d["front_Bowen_anom"])
            comps.append("_z_Bowen")

        if "front_H_anom" in d.columns and d["front_H_anom"].notna().sum() > 100:
            d["_z_H"] = zscore(d["front_H_anom"])
            comps.append("_z_H")

        if "front_EF_anom" in d.columns and d["front_EF_anom"].notna().sum() > 100:
            d["_z_minus_EF"] = zscore(-d["front_EF_anom"])
            comps.append("_z_minus_EF")

        if not comps:
            raise RuntimeError("Cannot construct local land-coupling index.")

        d["land_index"] = d[comps].mean(axis=1)

    d["x_land_z"] = zscore(d["land_index"])
    d["x_Tadv_z"] = zscore(d["front_Tadv850_K_day"])
    d["x_VIMD_z"] = zscore(d["front_VIMD_anom"])

    if "front_Z500_height_m_anom" in d.columns:
        d["x_Z500_z"] = zscore(d["front_Z500_height_m_anom"])
    else:
        d["x_Z500_z"] = np.nan

    if "front_ASC500_anom" in d.columns:
        d["x_ASC500_z"] = zscore(d["front_ASC500_anom"])
    else:
        d["x_ASC500_z"] = np.nan

    if "object_grid_days" not in d.columns or d["object_grid_days"].notna().sum() < 100:
        d["object_grid_days"] = d["front_cell_count"]

    if "max_daily_extent" not in d.columns or d["max_daily_extent"].notna().sum() < 100:
        d["max_daily_extent"] = d["front_cell_count"]

    if "duration_days" not in d.columns:
        d["duration_days"] = np.nan

    d["object_grid_days"] = to_num(d["object_grid_days"]).clip(lower=1)
    d["max_daily_extent"] = to_num(d["max_daily_extent"]).clip(lower=1)
    d["duration_days"] = to_num(d["duration_days"])

    d["log_object_grid_days"] = np.log10(d["object_grid_days"])
    d["log_max_extent"] = np.log10(d["max_daily_extent"])

    d["x_log_object_z"] = zscore(d["log_object_grid_days"])
    d["x_log_extent_z"] = zscore(d["log_max_extent"])
    d["x_duration_z"] = zscore(d["duration_days"])

    intensity_col = detect_intensity_column(d)

    if intensity_col is not None:
        d["intensity_proxy"] = to_num(d[intensity_col])
        d["x_intensity_z"] = zscore(d["intensity_proxy"])
        d.attrs["intensity_col"] = intensity_col
    else:
        d["intensity_proxy"] = np.nan
        d["x_intensity_z"] = np.nan
        d.attrs["intensity_col"] = None

    d["transport_index"] = d[["x_Tadv_z", "x_VIMD_z"]].mean(axis=1)
    d["transport_index_z"] = zscore(d["transport_index"])
    d["land_minus_transport"] = d["x_land_z"] - d["transport_index_z"]

    if "region" not in d.columns:
        d["region"] = "CONUS"

    d["region"] = d["region"].fillna("Other")

    if "event_lon" not in d.columns or d["event_lon"].notna().sum() < 100:
        for cand in ["lon", "longitude", "mean_lon", "centroid_lon"]:
            if cand in d.columns:
                d["event_lon"] = to_num(d[cand])
                break

    if "event_lat" not in d.columns or d["event_lat"].notna().sum() < 100:
        for cand in ["lat", "latitude", "mean_lat", "centroid_lat"]:
            if cand in d.columns:
                d["event_lat"] = to_num(d[cand])
                break

    d["front_cell_count"] = to_num(d["front_cell_count"]).clip(lower=1)
    d["weight"] = np.sqrt(d["front_cell_count"])

    conditions = [
        (d["x_land_z"] >= 0) &
        (d["transport_index_z"] >= 0) &
        (np.abs(d["land_minus_transport"]) < 0.50),

        (d["land_minus_transport"] >= 0.50) &
        (d["x_land_z"] >= 0),

        (d["land_minus_transport"] <= -0.50) &
        (d["transport_index_z"] >= 0),
    ]

    choices = [
        "Mixed",
        "Land-dominated",
        "Transport-dominated",
    ]

    d["mechanism_class"] = np.select(
        conditions,
        choices,
        default="Weak/ambiguous",
    )

    d = d.replace([np.inf, -np.inf], np.nan)

    d.to_csv(EVENT_TABLE_OUT, index=False, encoding="utf-8-sig")

    log(f"[SAVED] {EVENT_TABLE_OUT}")

    return d


# =============================================================================
# 5. LOAD SPATIAL TABLE
# =============================================================================

def load_spatial_cache():
    if not SPATIAL_CSV.exists():
        log(f"[WARN] spatial cache missing: {SPATIAL_CSV}")
        return pd.DataFrame()

    log(f"[READ] {SPATIAL_CSV}")

    return pd.read_csv(SPATIAL_CSV, low_memory=False)


def prepare_spatial_table(sp):
    if sp.empty:
        return sp

    d = sp.copy()

    for c in d.columns:
        if c != "coord_key":
            d[c] = to_num(d[c])

    comps = []

    if "mean_Bowen_anom" in d.columns:
        d["_z_Bowen"] = zscore(d["mean_Bowen_anom"])
        comps.append("_z_Bowen")

    if "mean_H_anom" in d.columns:
        d["_z_H"] = zscore(d["mean_H_anom"])
        comps.append("_z_H")

    if "mean_EF_anom" in d.columns:
        d["_z_minus_EF"] = zscore(-d["mean_EF_anom"])
        comps.append("_z_minus_EF")

    if comps:
        d["sp_land_index"] = d[comps].mean(axis=1)
    else:
        d["sp_land_index"] = np.nan

    return d


# =============================================================================
# 6. MODEL HELPERS
# =============================================================================

def valid_predictors(df, candidates, min_n=100):
    out = []

    for c in candidates:
        if c in df.columns:
            s = to_num(df[c])

            if s.notna().sum() >= min_n and np.nanstd(s) > 0:
                out.append(c)

    return out


def fit_wls(y, X, w=None):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if w is None:
        w = np.ones(len(y), dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    m = np.isfinite(y) & np.isfinite(X).all(axis=1) & np.isfinite(w) & (w > 0)

    if m.sum() < X.shape[1] + 30:
        return None

    yy = y[m]
    XX = X[m]
    ww = w[m]

    X1 = np.column_stack([np.ones(len(yy)), XX])
    sw = np.sqrt(ww)

    beta = np.linalg.lstsq(X1 * sw[:, None], yy * sw, rcond=None)[0]
    pred = X1 @ beta

    ybar = np.average(yy, weights=ww)
    ss_res = np.sum(ww * (yy - pred) ** 2)
    ss_tot = np.sum(ww * (yy - ybar) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    pred_full = np.full(len(y), np.nan)
    resid_full = np.full(len(y), np.nan)

    pred_full[m] = pred
    resid_full[m] = yy - pred

    return {
        "beta": beta,
        "r2": float(r2),
        "n": int(m.sum()),
        "mask": m,
        "pred_full": pred_full,
        "resid_full": resid_full,
    }


def bootstrap_coefficients(df, y_col, x_cols, n_boot=BOOT_N, seed=RANDOM_SEED):
    use = df[[y_col, "weight"] + x_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(use) < len(x_cols) + 60:
        return pd.DataFrame()

    y = use[y_col].to_numpy(float)
    X = use[x_cols].to_numpy(float)
    w = use["weight"].to_numpy(float)

    base = fit_wls(y, X, w)

    if base is None:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    n = len(use)

    B = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        res = fit_wls(y[idx], X[idx, :], w[idx])

        if res is not None:
            B.append(res["beta"][1:])

    B = np.asarray(B, dtype=float)

    rows = []

    for j, c in enumerate(x_cols):
        vals = B[:, j] if B.size else np.array([])
        vals = vals[np.isfinite(vals)]

        if vals.size:
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            p = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))
        else:
            lo = hi = p = np.nan

        rows.append({
            "predictor": c,
            "beta": base["beta"][j + 1],
            "ci_low": lo,
            "ci_high": hi,
            "p_boot": p,
            "n": base["n"],
            "r2_full": base["r2"],
        })

    return pd.DataFrame(rows)


def regional_coefficients(df):
    y_col = "y_dryfrac_z"

    predictors = valid_predictors(
        df,
        [
            "x_land_z",
            "x_Tadv_z",
            "x_VIMD_z",
            "x_Z500_z",
            "x_ASC500_z",
            "x_log_extent_z",
            "x_duration_z",
            "x_intensity_z",
        ],
    )

    rows = []

    for region, g in df.groupby("region"):
        if len(g) < 60:
            continue

        preds = [
            p for p in predictors
            if g[p].notna().sum() >= 50 and np.nanstd(g[p]) > 0
        ]

        core = [
            c for c in ["x_land_z", "x_Tadv_z", "x_VIMD_z"]
            if c in preds
        ]

        if len(core) < 2:
            continue

        b = bootstrap_coefficients(
            g,
            y_col,
            preds,
            n_boot=450,
            seed=RANDOM_SEED + len(rows),
        )

        if b.empty:
            continue

        b = b[b["predictor"].isin(core)].copy()
        b["region"] = region
        b["n_region"] = len(g)

        rows.append(b)

    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame()

    out.to_csv(REGION_COEF_OUT, index=False, encoding="utf-8-sig")

    log(f"[SAVED] {REGION_COEF_OUT}")

    return out


def weighted_r2(df, y_col, x_cols):
    if not x_cols:
        return 0.0

    use = df[[y_col, "weight"] + x_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(use) < len(x_cols) + 60:
        return np.nan

    res = fit_wls(
        use[y_col].to_numpy(float),
        use[x_cols].to_numpy(float),
        use["weight"].to_numpy(float),
    )

    return res["r2"] if res else np.nan


def unique_r2_table(df):
    y_col = "y_dryfrac_z"

    predictors = valid_predictors(
        df,
        [
            "x_land_z",
            "x_Tadv_z",
            "x_VIMD_z",
            "x_Z500_z",
            "x_ASC500_z",
            "x_log_extent_z",
            "x_duration_z",
            "x_intensity_z",
        ],
    )

    groups = {
        "Local land": ["x_land_z"],
        "Atmospheric transport": ["x_Tadv_z", "x_VIMD_z"],
        "Circulation": ["x_Z500_z", "x_ASC500_z"],
        "Event morphology": ["x_log_extent_z", "x_duration_z"],
        "Intensity": ["x_intensity_z"],
    }

    r2_full = weighted_r2(df, y_col, predictors)

    rows = []

    for name, cols in groups.items():
        cols = [c for c in cols if c in predictors]

        if not cols:
            continue

        reduced = [c for c in predictors if c not in cols]
        r2_red = weighted_r2(df, y_col, reduced)

        uniq = r2_full - r2_red if np.isfinite(r2_full) and np.isfinite(r2_red) else np.nan

        rows.append({
            "component": name,
            "unique_delta_R2": max(uniq, 0.0) if np.isfinite(uniq) else np.nan,
            "full_R2": r2_full,
        })

    out = pd.DataFrame(rows)

    out.to_csv(R2_OUT, index=False, encoding="utf-8-sig")

    log(f"[SAVED] {R2_OUT}")

    return out


def residualize(y, X, w):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    m = np.isfinite(y) & np.isfinite(X).all(axis=1) & np.isfinite(w) & (w > 0)

    out = np.full(len(y), np.nan)

    if m.sum() < X.shape[1] + 30:
        return out

    yy = y[m]
    XX = X[m]
    ww = w[m]

    X1 = np.column_stack([np.ones(len(yy)), XX])
    sw = np.sqrt(ww)

    beta = np.linalg.lstsq(X1 * sw[:, None], yy * sw, rcond=None)[0]

    out[m] = yy - X1 @ beta

    return out


# =============================================================================
# 7. BINNING HELPERS
# =============================================================================

def binned2d_weighted(
    df,
    x,
    y,
    z,
    w="weight",
    xbins=24,
    ybins=20,
    min_n=8,
    xlim=None,
    ylim=None,
):
    d = df[[x, y, z, w]].replace([np.inf, -np.inf], np.nan).dropna()

    if xlim is None:
        xlo, xhi = np.nanpercentile(d[x], [1, 99])
    else:
        xlo, xhi = xlim

    if ylim is None:
        ylo, yhi = np.nanpercentile(d[y], [1, 99])
    else:
        ylo, yhi = ylim

    xb = np.linspace(xlo, xhi, xbins + 1)
    yb = np.linspace(ylo, yhi, ybins + 1)

    Z = np.full((ybins, xbins), np.nan)
    N = np.zeros((ybins, xbins), dtype=int)

    for i in range(xbins):
        for j in range(ybins):
            g = d[
                (d[x] >= xb[i]) &
                (d[x] < xb[i + 1]) &
                (d[y] >= yb[j]) &
                (d[y] < yb[j + 1])
            ]

            N[j, i] = len(g)

            if len(g) >= min_n:
                Z[j, i] = weighted_quantile(g[z], g[w], [0.5])[0]

    return xb, yb, Z, N


def running_window_summary(
    df,
    x_col,
    y_col,
    w_col="weight",
    n_centers=26,
    width_q=0.20,
    min_n=45,
):
    d = df[[x_col, y_col, w_col]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(d) < min_n * 3:
        return pd.DataFrame()

    qs = np.linspace(0.06, 0.94, n_centers)
    xvals = d[x_col].to_numpy(float)

    rows = []

    for q in qs:
        loq = max(0, q - width_q / 2)
        hiq = min(1, q + width_q / 2)

        lo = np.nanquantile(xvals, loq)
        hi = np.nanquantile(xvals, hiq)

        g = d[(d[x_col] >= lo) & (d[x_col] <= hi)]

        if len(g) < min_n:
            continue

        qu = weighted_quantile(g[y_col], g[w_col], [0.25, 0.5, 0.75])

        rows.append({
            "x": np.nanmedian(g[x_col]),
            "q25": qu[0],
            "q50": qu[1],
            "q75": qu[2],
            "n": len(g),
        })

    return pd.DataFrame(rows)


# =============================================================================
# 8. MAIN FIGURE PANELS
# =============================================================================

def event_weighted_spatial_field(df, cell_deg=1.25):
    d = df[
        [
            "event_lon",
            "event_lat",
            "land_minus_transport",
            "object_grid_days",
            "front_cell_count",
        ]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    d = d[
        d["event_lon"].between(CONUS_EXTENT[0], CONUS_EXTENT[1]) &
        d["event_lat"].between(CONUS_EXTENT[2], CONUS_EXTENT[3])
    ].copy()

    lon_edges = np.arange(CONUS_EXTENT[0], CONUS_EXTENT[1] + cell_deg, cell_deg)
    lat_edges = np.arange(CONUS_EXTENT[2], CONUS_EXTENT[3] + cell_deg, cell_deg)

    nlon = len(lon_edges) - 1
    nlat = len(lat_edges) - 1

    d["lon_i"] = np.digitize(d["event_lon"], lon_edges) - 1
    d["lat_i"] = np.digitize(d["event_lat"], lat_edges) - 1

    d = d[
        d["lon_i"].between(0, nlon - 1) &
        d["lat_i"].between(0, nlat - 1)
    ].copy()

    Z_num = np.zeros((nlat, nlon), dtype=float)
    Z_den = np.zeros((nlat, nlon), dtype=float)
    SUPPORT = np.zeros((nlat, nlon), dtype=float)

    for r in d.itertuples(index=False):
        i = int(r.lon_i)
        j = int(r.lat_i)

        w = np.sqrt(max(float(r.front_cell_count), 1.0))

        Z_num[j, i] += float(r.land_minus_transport) * w
        Z_den[j, i] += w
        SUPPORT[j, i] += float(r.object_grid_days)

    Z = Z_num / np.where(Z_den <= 0, np.nan, Z_den)

    Zs = smooth_nan_field(Z, sigma=0.85)
    SUPPORT_s = smooth_nan_field(
        np.where(SUPPORT > 0, np.log10(SUPPORT + 1.0), np.nan),
        sigma=0.85,
    )

    pos_support = SUPPORT[np.isfinite(SUPPORT) & (SUPPORT > 0)]

    if pos_support.size:
        support_cut = np.nanquantile(pos_support, 0.12)
        weak_mask = SUPPORT < support_cut
        Zs[weak_mask] = np.nan

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    return lon_edges, lat_edges, lon_centers, lat_centers, Zs, SUPPORT_s


def plot_event_dominance_field(ax, df, label=None):
    setup_conus_map(ax, ticksize=FS_MAP_TICK, show_lat_labels=True)

    lon_edges, lat_edges, lon_centers, lat_centers, Z, SUPPORT_s = event_weighted_spatial_field(
        df,
        cell_deg=1.25,
    )

    vmax = robust_abs_lim(Z, q=0.985, floor=0.8)

    plot_kwargs = dict(
        cmap=DOM_CMAP,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        shading="auto",
        rasterized=True,
    )

    if HAS_CARTOPY:
        plot_kwargs["transform"] = ccrs.PlateCarree()

    im = ax.pcolormesh(
        lon_edges,
        lat_edges,
        Z,
        **plot_kwargs,
    )

    Xc, Yc = np.meshgrid(lon_centers, lat_centers)

    finite_support = SUPPORT_s[np.isfinite(SUPPORT_s)]

    if finite_support.size > 20:
        levels = np.unique(np.nanpercentile(finite_support, [55, 72, 88]))

        if len(levels) >= 2:
            contour_kwargs = dict(
                levels=levels,
                colors=["0.35"],
                linewidths=[0.95, 1.15, 1.35][:len(levels)],
                alpha=0.76,
            )

            if HAS_CARTOPY:
                contour_kwargs["transform"] = ccrs.PlateCarree()

            ax.contour(
                Xc,
                Yc,
                SUPPORT_s,
                **contour_kwargs,
            )

    ax.set_title(
        "Event-weighted land–transport dominance",
        pad=8,
    )

    cb = plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.110,
        extend="both",
    )

    cb.set_label(
        "Land-coupling index − transport index",
        fontsize=FS_CBAR_LABEL,
    )

    cb.ax.tick_params(labelsize=FS_CBAR_TICK)

    ax.set_ylabel("")

    ax.text(
        0.025,
        0.055,
        "Grey contours: event-support density",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FS_LEGEND_SMALL,
        bbox=dict(
            facecolor="white",
            edgecolor="0.80",
            alpha=0.90,
            pad=3.0,
        ),
        zorder=30,
    )


def plot_process_landscape(ax, df, label=None):
    d = df[
        [
            "transport_index_z",
            "x_land_z",
            "front_dry_fraction",
            "weight",
        ]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    lim = max(
        robust_abs_lim(d["transport_index_z"], q=0.985, floor=1.5),
        robust_abs_lim(d["x_land_z"], q=0.985, floor=1.5),
    )

    lim = min(max(lim, 2.0), 3.2)

    xb, yb, Z, N = binned2d_weighted(
        d,
        "transport_index_z",
        "x_land_z",
        "front_dry_fraction",
        xbins=24,
        ybins=24,
        min_n=8,
        xlim=(-lim, lim),
        ylim=(-lim, lim),
    )

    im = ax.pcolormesh(
        xb,
        yb,
        Z,
        cmap=DRY_CMAP,
        vmin=0,
        vmax=1,
        shading="auto",
        rasterized=True,
    )

    Xc = (xb[:-1] + xb[1:]) / 2
    Yc = (yb[:-1] + yb[1:]) / 2

    if np.nanmax(N) > 0:
        npos = N[N > 0]
        levels = np.nanpercentile(npos, [45, 65, 80, 92])
        levels = np.unique(levels)

        if len(levels) >= 2:
            ax.contour(
                Xc,
                Yc,
                N,
                levels=levels,
                colors="0.25",
                linewidths=1.0,
                alpha=0.70,
            )

    ax.axhline(0, color="0.50", ls="--", lw=1.2)
    ax.axvline(0, color="0.50", ls="--", lw=1.2)
    ax.plot([-lim, lim], [-lim, lim], color="0.45", ls=":", lw=1.2)

    ax.text(
        0.04,
        0.95,
        "Land-dominated\nmaintenance",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_NOTE,
        color=COLOR_LAND,
    )

    ax.text(
        0.96,
        0.05,
        "Transport-dominated\nmaintenance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FS_NOTE,
        color=COLOR_TRANSPORT,
    )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_xlabel("Atmospheric transport index")
    ax.set_ylabel("Local land-coupling index")

    ax.set_title("All-event land–transport process landscape", pad=8)

    ax.grid(color="0.90", lw=1.0)

    clean_spines(ax)

    cb = plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.042,
        pad=0.022,
    )

    cb.set_label("Median front dry-state maintenance", fontsize=FS_CBAR_LABEL)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK)


def plot_region_forest(ax, reg, label=None, show_legend=False):
    if reg.empty:
        ax.text(
            0.5,
            0.5,
            "Regional coefficients unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        clean_spines(ax)
        return

    predictors = ["x_land_z", "x_Tadv_z", "x_VIMD_z"]

    labels = {
        "x_land_z": "Local land",
        "x_Tadv_z": "Tadv850",
        "x_VIMD_z": "VIMD",
    }

    colors = {
        "x_land_z": COLOR_LAND,
        "x_Tadv_z": COLOR_TADV,
        "x_VIMD_z": COLOR_VIMD,
    }

    preferred = [
        "West",
        "Rockies / N. Plains",
        "Great Plains",
        "Midwest",
        "Southwest / S. Plains",
        "Southeast",
        "Northeast",
    ]

    regions = [r for r in preferred if r in set(reg["region"])]
    regions += [r for r in sorted(reg["region"].unique()) if r not in regions]

    y_base = np.arange(len(regions))[::-1]

    offsets = {
        "x_land_z": 0.24,
        "x_Tadv_z": 0.00,
        "x_VIMD_z": -0.24,
    }

    for pred in predictors:
        g = reg[reg["predictor"] == pred].set_index("region").reindex(regions)

        y = y_base + offsets[pred]

        beta = g["beta"].to_numpy(float)
        lo = g["ci_low"].to_numpy(float)
        hi = g["ci_high"].to_numpy(float)
        p = g["p_boot"].to_numpy(float)

        xerr = np.vstack([
            np.maximum(beta - lo, 0),
            np.maximum(hi - beta, 0),
        ])

        ax.errorbar(
            beta,
            y,
            xerr=xerr,
            fmt="o",
            ms=9.5,
            lw=2.4,
            capsize=4.3,
            color=colors[pred],
            ecolor=colors[pred],
            label=labels[pred],
            zorder=4,
        )

        for xi, yi, pi in zip(beta, y, p):
            if np.isfinite(xi) and p_to_star(pi):
                ax.text(
                    xi,
                    yi + 0.115,
                    p_to_star(pi),
                    ha="center",
                    va="bottom",
                    fontsize=FS_NOTE,
                    color=colors[pred],
                )

    ax.set_yticks(y_base)
    ax.set_yticklabels(list(regions))

    ax.axvline(0, color="0.50", lw=1.2, ls="--")

    ax.set_xlabel("Standardized coefficient")
    ax.set_title("Regional robustness of event-level controls", pad=8)

    ax.grid(axis="x", color="0.88", lw=1.0)

    if show_legend:
        ax.legend(
            frameon=True,
            loc="upper right",
            ncol=1,
            facecolor="white",
            edgecolor="0.82",
            framealpha=0.94,
            borderpad=0.55,
            handlelength=1.5,
            handletextpad=0.70,
        )

    clean_spines(ax)


def plot_partial_response(ax, df, label=None, show_legend=False):
    y_col = "y_dryfrac_z"

    land_controls = valid_predictors(
        df,
        [
            "transport_index_z",
            "x_Z500_z",
            "x_ASC500_z",
            "x_log_extent_z",
            "x_duration_z",
            "x_intensity_z",
        ],
    )

    transport_controls = valid_predictors(
        df,
        [
            "x_land_z",
            "x_Z500_z",
            "x_ASC500_z",
            "x_log_extent_z",
            "x_duration_z",
            "x_intensity_z",
        ],
    )

    d1 = df[
        [y_col, "x_land_z", "weight"] + land_controls
    ].replace([np.inf, -np.inf], np.nan).dropna()

    d2 = df[
        [y_col, "transport_index_z", "weight"] + transport_controls
    ].replace([np.inf, -np.inf], np.nan).dropna()

    x1 = residualize(
        d1["x_land_z"].to_numpy(float),
        d1[land_controls].to_numpy(float),
        d1["weight"].to_numpy(float),
    )

    y1 = residualize(
        d1[y_col].to_numpy(float),
        d1[land_controls].to_numpy(float),
        d1["weight"].to_numpy(float),
    )

    x2 = residualize(
        d2["transport_index_z"].to_numpy(float),
        d2[transport_controls].to_numpy(float),
        d2["weight"].to_numpy(float),
    )

    y2 = residualize(
        d2[y_col].to_numpy(float),
        d2[transport_controls].to_numpy(float),
        d2["weight"].to_numpy(float),
    )

    ok1 = np.isfinite(x1) & np.isfinite(y1)
    ok2 = np.isfinite(x2) & np.isfinite(y2)

    rho1, p1 = safe_spearman(x1[ok1], y1[ok1])
    rho2, p2 = safe_spearman(x2[ok2], y2[ok2])

    curve_specs = [
        (
            x1[ok1],
            y1[ok1],
            f"Local land: ρ={rho1:.2f}, {fmt_p_threshold(p1)}",
            COLOR_LAND,
        ),
        (
            x2[ok2],
            y2[ok2],
            f"Transport: ρ={rho2:.2f}, {fmt_p_threshold(p2)}",
            COLOR_TRANSPORT,
        ),
    ]

    for x, y, leg_label, color in curve_specs:
        rb = pd.DataFrame({
            "x": x,
            "y": y,
            "w": np.ones(len(x)),
        }).dropna()

        if len(rb) < 100:
            continue

        sm = running_window_summary(
            rb,
            "x",
            "y",
            w_col="w",
            n_centers=24,
            width_q=0.22,
            min_n=45,
        )

        if sm.empty:
            continue

        ax.plot(
            sm["x"],
            sm["q50"],
            lw=3.8,
            color=color,
            label=leg_label,
        )

        ax.fill_between(
            sm["x"],
            sm["q25"],
            sm["q75"],
            color=color,
            alpha=0.13,
            lw=0,
        )

    ax.axhline(0, color="0.50", ls="--", lw=1.2)
    ax.axvline(0, color="0.50", ls="--", lw=1.2)

    ax.set_xlabel("Residual diagnostic contrast")
    ax.set_ylabel("Residual dry-front maintenance")

    ax.set_title("Partial response after mutual adjustment", pad=8)

    ax.grid(color="0.88", lw=1.0)

    if show_legend:
        ax.legend(
            frameon=True,
            loc="upper left",
            facecolor="white",
            edgecolor="0.82",
            framealpha=0.94,
            borderpad=0.55,
            handlelength=2.0,
            handletextpad=0.70,
        )

    clean_spines(ax)


# =============================================================================
# 9. SUPPLEMENTARY PANELS
# =============================================================================

def grid_from_spatial(sp, value_col):
    d = sp[
        ["longitude", "latitude", value_col]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    lons = np.sort(d["longitude"].dropna().unique())
    lats = np.sort(d["latitude"].dropna().unique())

    Z = (
        d.pivot_table(
            index="latitude",
            columns="longitude",
            values=value_col,
            aggfunc="mean",
        )
        .reindex(index=lats, columns=lons)
        .to_numpy(float)
    )

    return lons, lats, Z


def plot_spatial_panel(
    ax,
    sp,
    value_col,
    title,
    label=None,
    floor=0.05,
    show_lat_labels=True,
):
    setup_conus_map(
        ax,
        ticksize=FS_MAP_TICK,
        show_lat_labels=show_lat_labels,
    )

    if sp.empty or value_col not in sp.columns or sp[value_col].notna().sum() < 30:
        ax.text(
            0.5,
            0.5,
            "Spatial diagnostic\nnot available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FS_NOTE,
        )
        ax.set_title(title, pad=7)
        return None

    lons, lats, Z = grid_from_spatial(sp, value_col)

    Z = smooth_nan_field(Z, sigma=0.65)

    vmax = robust_abs_lim(Z, q=0.985, floor=floor)

    plot_kwargs = dict(
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        shading="auto",
        rasterized=True,
    )

    if HAS_CARTOPY:
        plot_kwargs["transform"] = ccrs.PlateCarree()

    im = ax.pcolormesh(
        coord_edges(lons),
        coord_edges(lats),
        Z,
        **plot_kwargs,
    )

    ax.set_title(title, pad=7)

    return im


def plot_regional_composition(ax, df, label=None):
    classes = [
        "Land-dominated",
        "Mixed",
        "Transport-dominated",
        "Weak/ambiguous",
    ]

    colors = [
        COLOR_LAND,
        COLOR_MIXED,
        COLOR_TRANSPORT,
        COLOR_WEAK,
    ]

    tab = (
        pd.crosstab(
            df["region"],
            df["mechanism_class"],
            normalize="index",
        )
        .reindex(columns=classes)
        .fillna(0)
    )

    preferred = [
        "West",
        "Rockies / N. Plains",
        "Great Plains",
        "Midwest",
        "Southwest / S. Plains",
        "Southeast",
        "Northeast",
    ]

    order = [r for r in preferred if r in tab.index]
    order += [r for r in tab.index if (r not in order) and (r != "Other")]

    tab = tab.reindex(order)
    tab = tab.loc[[r for r in tab.index if r != "Other"]]

    y = np.arange(len(tab))
    left = np.zeros(len(tab))

    for cls, color in zip(classes, colors):
        vals = tab[cls].to_numpy()

        ax.barh(
            y,
            vals,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.90,
            label=cls,
        )

        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(tab.index)

    ax.set_xlim(0, 1)

    ax.set_xlabel("Fraction of events")
    ax.set_ylabel("")
    ax.set_title("")

    clean_spines(ax)


def plot_event_morphology_mechanism_scatter(ax, df, label=None):
    """
    Supplementary panel c:
    x-axis  : Event duration
    y-axis  : Maximum daily extent, log scale
    colour  : Front dry-state maintenance
    marker  : Mechanism class

    The colour bar is NOT drawn here. It is drawn later using fixed figure
    coordinates so it does not shrink the panel or create vertical crowding.
    """

    required = [
        "duration_days",
        "max_daily_extent",
        "front_dry_fraction",
        "mechanism_class",
    ]

    use_cols = required + ["front_cell_count", "object_grid_days"]
    use_cols = [c for c in use_cols if c in df.columns]

    d = df[use_cols].replace([np.inf, -np.inf], np.nan).dropna(
        subset=["duration_days", "max_daily_extent", "front_dry_fraction", "mechanism_class"]
    ).copy()

    d = d[
        (d["duration_days"] > 0) &
        (d["max_daily_extent"] > 0)
    ].copy()

    d = d.reset_index(drop=True)

    if d.empty:
        ax.text(
            0.5,
            0.5,
            "Event morphology data unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        clean_spines(ax)
        return None

    rng = np.random.default_rng(RANDOM_SEED)

    d["duration_jitter"] = (
        d["duration_days"].astype(float) +
        rng.normal(0, 0.08, len(d))
    )

    d["duration_jitter"] = d["duration_jitter"].clip(lower=0.5)

    marker_map = {
        "Land-dominated": "o",
        "Transport-dominated": "^",
        "Mixed": "s",
        "Weak/ambiguous": "x",
    }

    label_order = [
        "Land-dominated",
        "Transport-dominated",
        "Mixed",
        "Weak/ambiguous",
    ]

    edge_map = {
        "Land-dominated": "0.20",
        "Transport-dominated": "0.20",
        "Mixed": "0.20",
        "Weak/ambiguous": "0.35",
    }

    if "front_cell_count" in d.columns:
        size_raw = np.sqrt(d["front_cell_count"].clip(lower=1))
        size = np.clip(size_raw * 3.2, 45, 210).to_numpy(float)
    else:
        size = np.full(len(d), 75.0)

    last_sc = None

    for cls in label_order:
        g = d[d["mechanism_class"] == cls].copy()

        if g.empty:
            continue

        s_use = size[g.index.to_numpy()]

        if cls == "Weak/ambiguous":
            last_sc = ax.scatter(
                g["duration_jitter"],
                g["max_daily_extent"],
                c=g["front_dry_fraction"],
                cmap=DRY_CMAP,
                vmin=0,
                vmax=1,
                s=s_use,
                marker=marker_map[cls],
                alpha=0.55,
                linewidths=1.4,
                rasterized=True,
                zorder=3,
            )
        else:
            last_sc = ax.scatter(
                g["duration_jitter"],
                g["max_daily_extent"],
                c=g["front_dry_fraction"],
                cmap=DRY_CMAP,
                vmin=0,
                vmax=1,
                s=s_use,
                marker=marker_map[cls],
                alpha=0.70,
                edgecolors=edge_map[cls],
                linewidths=0.55,
                rasterized=True,
                zorder=3,
            )

    ax.set_yscale("log")

    xmax = np.nanpercentile(d["duration_days"], 99.2)
    xmax = max(8, xmax)

    ymax = np.nanpercentile(d["max_daily_extent"], 99.5)
    ymin = max(1, np.nanpercentile(d["max_daily_extent"], 0.5))

    ax.set_xlim(0, xmax * 1.06)
    ax.set_ylim(ymin * 0.80, ymax * 1.35)

    ax.set_xlabel("Event duration (days)")
    ax.set_ylabel("Maximum daily extent\n(grid cells)", labelpad=4)

    ax.set_title("")

    ax.grid(True, which="major", color="0.88", lw=1.0)
    ax.grid(True, which="minor", color="0.93", lw=0.7)

    clean_spines(ax)

    return last_sc


def plot_unique_r2(ax, r2tab, label=None):
    if r2tab.empty:
        ax.text(
            0.5,
            0.5,
            "Unique R² unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        clean_spines(ax)
        return

    order = [
        "Local land",
        "Atmospheric transport",
        "Circulation",
        "Event morphology",
        "Intensity",
    ]

    d = (
        r2tab.set_index("component")
        .reindex([o for o in order if o in set(r2tab["component"])])
        .reset_index()
    )

    color_map = {
        "Local land": COLOR_LAND,
        "Atmospheric transport": COLOR_TRANSPORT,
        "Circulation": COLOR_CIRC,
        "Event morphology": COLOR_GEOM,
        "Intensity": COLOR_INTENSITY,
    }

    colors = [color_map.get(c, "0.5") for c in d["component"]]

    x = np.arange(len(d))

    ax.bar(
        x,
        d["unique_delta_R2"],
        color=colors,
        edgecolor="black",
        linewidth=0.95,
        width=0.68,
    )

    ymax = np.nanmax(d["unique_delta_R2"].to_numpy(float))

    if not np.isfinite(ymax):
        ymax = 0.05

    for xi, val in zip(x, d["unique_delta_R2"]):
        if np.isfinite(val):
            ax.text(
                xi,
                val + max(0.004, ymax * 0.04),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=FS_NOTE,
            )

    fullr2 = (
        d["full_R2"].dropna().iloc[0]
        if d["full_R2"].notna().any()
        else np.nan
    )

    if np.isfinite(fullr2):
        ax.text(
            0.98,
            0.96,
            f"Full weighted model R² = {fullr2:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=FS_NOTE,
            bbox=dict(
                facecolor="white",
                edgecolor="0.82",
                alpha=0.90,
                pad=3.0,
            ),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["" for _ in d["component"]])

    ax.set_ylabel("Unique increment in weighted R²", labelpad=4)
    ax.set_title("")

    ax.grid(axis="y", color="0.88", lw=1.0)

    clean_spines(ax)


# =============================================================================
# 10. DRAW FIGURES
# =============================================================================

def draw_main_figure(df, reg):
    fig = plt.figure(figsize=MAIN_FIGSIZE)

    gs = GridSpec(
        3,
        2,
        figure=fig,
        left=0.070,
        right=0.988,
        top=0.930,
        bottom=0.070,
        hspace=0.38,
        wspace=0.285,
        height_ratios=[1.0, 1.0, 0.16],
    )

    if HAS_CARTOPY:
        ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    else:
        ax_a = fig.add_subplot(gs[0, 0])

    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")

    plot_event_dominance_field(ax_a, df, label=None)
    plot_process_landscape(ax_b, df, label=None)
    plot_region_forest(ax_c, reg, label=None, show_legend=False)
    plot_partial_response(ax_d, df, label=None, show_legend=False)

    fig.canvas.draw()

    add_panel_label_figure(fig, ax_a, "a", dx=0.030, dy=0.010)
    add_panel_label_figure(fig, ax_b, "b", dx=0.030, dy=0.010)
    add_panel_label_figure(fig, ax_c, "c", dx=0.030, dy=0.010)
    add_panel_label_figure(fig, ax_d, "d", dx=0.030, dy=0.010)

    handles_main = [
        Line2D([0], [0], marker="o", linestyle="None", color=COLOR_LAND,
               markersize=10, label="Local land"),
        Line2D([0], [0], marker="o", linestyle="None", color=COLOR_TADV,
               markersize=10, label="Tadv850"),
        Line2D([0], [0], marker="o", linestyle="None", color=COLOR_VIMD,
               markersize=10, label="VIMD"),
        Line2D([0], [0], color=COLOR_LAND, lw=3.8,
               label="Local-land partial response"),
        Line2D([0], [0], color=COLOR_TRANSPORT, lw=3.8,
               label="Transport partial response"),
    ]

    ax_leg.legend(
        handles=handles_main,
        loc="center",
        ncol=5,
        frameon=False,
        fontsize=FS_LEGEND_SMALL,
        columnspacing=1.4,
        handletextpad=0.6,
        handlelength=1.8,
    )

    savefig(fig, MAIN_PNG, MAIN_PDF)


def draw_supplementary_figure(df, sp, r2tab):
    fig = plt.figure(figsize=SUPP_FIGSIZE)

    # Reserve a real bottom space for the three aligned legends.
    outer = GridSpec(
        2,
        1,
        figure=fig,
        left=0.055,
        right=0.990,
        top=0.925,
        bottom=0.245,
        hspace=0.58,
        height_ratios=[0.88, 1.12],
    )

    gs_top = outer[0].subgridspec(
        1,
        3,
        wspace=0.030,
    )

    gs_bottom = outer[1].subgridspec(
        1,
        3,
        wspace=0.500,
    )

    if HAS_CARTOPY:
        ax_map1 = fig.add_subplot(gs_top[0, 0], projection=ccrs.PlateCarree())
        ax_map2 = fig.add_subplot(gs_top[0, 1], projection=ccrs.PlateCarree())
        ax_map3 = fig.add_subplot(gs_top[0, 2], projection=ccrs.PlateCarree())
    else:
        ax_map1 = fig.add_subplot(gs_top[0, 0])
        ax_map2 = fig.add_subplot(gs_top[0, 1])
        ax_map3 = fig.add_subplot(gs_top[0, 2])

    ax_b = fig.add_subplot(gs_bottom[0, 0])
    ax_c = fig.add_subplot(gs_bottom[0, 1])
    ax_d = fig.add_subplot(gs_bottom[0, 2])

    # -------------------------------------------------------------------------
    # Top-row maps: same axes geometry. Colour bars are added later using fixed
    # figure coordinates, so the map panels do not shrink unevenly.
    # -------------------------------------------------------------------------
    im1 = plot_spatial_panel(
        ax_map1,
        sp,
        "mean_Tadv850_K_day",
        "Front thermal advection",
        label=None,
        floor=0.05,
        show_lat_labels=True,
    )

    im2 = plot_spatial_panel(
        ax_map2,
        sp,
        "sp_land_index",
        "Front local land-coupling index",
        label=None,
        floor=0.20,
        show_lat_labels=False,
    )

    im3 = plot_spatial_panel(
        ax_map3,
        sp,
        "mean_VIMD_anom",
        "Front moisture-divergence support",
        label=None,
        floor=0.02,
        show_lat_labels=False,
    )

    # -------------------------------------------------------------------------
    # Lower row: b/c/d panels, same height and top alignment.
    # -------------------------------------------------------------------------
    plot_regional_composition(ax_b, df, label=None)
    sc_c = plot_event_morphology_mechanism_scatter(ax_c, df, label=None)
    plot_unique_r2(ax_d, r2tab, label=None)

    fig.canvas.draw()

    # -------------------------------------------------------------------------
    # Fixed colour bars for maps. They do not alter axes size.
    # -------------------------------------------------------------------------
    if im1 is not None:
        add_fixed_horizontal_cbar(
            fig,
            ax_map1,
            im1,
            r"Tadv850 (K day$^{-1}$)",
            ticks=None,
            width_frac=0.52,
            height=0.013,
            y_offset=0.038,
        )

    if im2 is not None:
        add_fixed_horizontal_cbar(
            fig,
            ax_map2,
            im2,
            "Standardized index",
            ticks=None,
            width_frac=0.52,
            height=0.013,
            y_offset=0.038,
        )

    if im3 is not None:
        add_fixed_horizontal_cbar(
            fig,
            ax_map3,
            im3,
            "VIMD anomaly",
            ticks=None,
            width_frac=0.52,
            height=0.013,
            y_offset=0.038,
        )

    # Horizontal colour bar for the scatter panel.
    # This replaces the old vertical crowded label.
    if sc_c is not None:
        add_fixed_horizontal_cbar_noextend(
            fig,
            ax_c,
            sc_c,
            "Dry-state maintenance",
            ticks=[0, 0.5, 1.0],
            width_frac=0.58,
            height=0.013,
            y_offset=0.072,
        )

    fig.canvas.draw()

    # -------------------------------------------------------------------------
    # Panel labels: outside upper-left corners.
    # -------------------------------------------------------------------------
    if GROUP_TOP_MAPS_AS_PANEL_A:
        add_panel_label_figure(fig, ax_map1, "a", dx=0.030, dy=0.010)
        add_panel_label_figure(fig, ax_b, "b", dx=0.032, dy=0.012)
        add_panel_label_figure(fig, ax_c, "c", dx=0.032, dy=0.012)
        add_panel_label_figure(fig, ax_d, "d", dx=0.032, dy=0.012)
    else:
        add_panel_label_figure(fig, ax_map1, "a", dx=0.030, dy=0.010)
        add_panel_label_figure(fig, ax_map2, "b", dx=0.030, dy=0.010)
        add_panel_label_figure(fig, ax_map3, "c", dx=0.030, dy=0.010)
        add_panel_label_figure(fig, ax_b, "d", dx=0.032, dy=0.012)
        add_panel_label_figure(fig, ax_c, "e", dx=0.032, dy=0.012)
        add_panel_label_figure(fig, ax_d, "f", dx=0.032, dy=0.012)

    # -------------------------------------------------------------------------
    # Dedicated bottom legend row.
    # Three legend axes use identical y-position and height.
    # -------------------------------------------------------------------------
    fig.canvas.draw()

    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()
    pos_d = ax_d.get_position()

    legend_y0 = 0.050
    legend_h = 0.115

    ax_leg_b = fig.add_axes([pos_b.x0, legend_y0, pos_b.width, legend_h])
    ax_leg_c = fig.add_axes([pos_c.x0, legend_y0, pos_c.width, legend_h])
    ax_leg_d = fig.add_axes([pos_d.x0, legend_y0, pos_d.width, legend_h])

    for ax in [ax_leg_b, ax_leg_c, ax_leg_d]:
        ax.axis("off")

    handles_b = [
        Patch(facecolor=COLOR_LAND, edgecolor="white", linewidth=0.9, label="Land-dominated"),
        Patch(facecolor=COLOR_MIXED, edgecolor="white", linewidth=0.9, label="Mixed"),
        Patch(facecolor=COLOR_TRANSPORT, edgecolor="white", linewidth=0.9, label="Transport-dominated"),
        Patch(facecolor=COLOR_WEAK, edgecolor="white", linewidth=0.9, label="Weak/ambiguous"),
    ]

    ax_leg_b.legend(
        handles=handles_b,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=FS_LEGEND_SMALL,
        columnspacing=1.1,
        handletextpad=0.55,
        handlelength=1.6,
    )

    handles_c = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="0.72", markeredgecolor="0.20",
               markersize=10.0, label="Land-dominated"),
        Line2D([0], [0], marker="^", linestyle="None",
               markerfacecolor="0.72", markeredgecolor="0.20",
               markersize=10.0, label="Transport-dominated"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor="0.72", markeredgecolor="0.20",
               markersize=10.0, label="Mixed"),
        Line2D([0], [0], marker="x", linestyle="None",
               color="0.35", markersize=10.0,
               markeredgewidth=2.0, label="Weak/ambiguous"),
    ]

    ax_leg_c.legend(
        handles=handles_c,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=FS_LEGEND_SMALL,
        columnspacing=1.1,
        handletextpad=0.55,
        handlelength=1.4,
    )

    handles_d = [
        Patch(facecolor=COLOR_LAND, edgecolor="black", linewidth=0.6, label="Local land"),
        Patch(facecolor=COLOR_TRANSPORT, edgecolor="black", linewidth=0.6, label="Atmospheric transport"),
        Patch(facecolor=COLOR_CIRC, edgecolor="black", linewidth=0.6, label="Circulation"),
        Patch(facecolor=COLOR_GEOM, edgecolor="black", linewidth=0.6, label="Event morphology"),
        Patch(facecolor=COLOR_INTENSITY, edgecolor="black", linewidth=0.6, label="Intensity"),
    ]

    ax_leg_d.legend(
        handles=handles_d,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=FS_LEGEND_SMALL,
        columnspacing=1.0,
        handletextpad=0.50,
        handlelength=1.3,
    )

    savefig(fig, SUPP_PNG, SUPP_PDF)


def audit_font_and_layout():
    old_main_width = 30.0
    old_supp_width = 37.0
    old_base_font = 31.0
    two_col_width_in = 180.0 / 25.4

    old_main_effective = old_base_font * two_col_width_in / old_main_width
    old_supp_effective = old_base_font * two_col_width_in / old_supp_width
    new_main_effective = FS_BASE * two_col_width_in / MAIN_FIGSIZE[0]
    new_supp_effective = FS_BASE * two_col_width_in / SUPP_FIGSIZE[0]

    log("[FONT-AUDIT] Previous main canvas: 30.0 in wide; previous supplementary canvas: 37.0 in wide")
    log(f"[FONT-AUDIT] New main canvas: {MAIN_FIGSIZE[0]:.1f} in wide; new supplementary canvas: {SUPP_FIGSIZE[0]:.1f} in wide")
    log(f"[FONT-AUDIT] Base font={FS_BASE:.1f} pt; axis label={FS_LABEL:.1f} pt; tick={FS_TICK:.1f} pt; panel={FS_PANEL:.1f} pt")
    log(f"[FONT-AUDIT] Approx. effective base font at 180-mm width: main {old_main_effective:.1f} -> {new_main_effective:.1f} pt; supplementary {old_supp_effective:.1f} -> {new_supp_effective:.1f} pt")


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    log("=" * 100)
    log("[INFO] Redraw NCC-polished Figure 5 and Supplementary Figure")
    log("[INFO] No raw-event extraction will be rerun")
    log(f"[INFO] CACHE_ROOT : {CACHE_ROOT}")
    log(f"[INFO] OUT_DIR    : {OUT_DIR}")
    audit_font_and_layout()
    log("=" * 100)

    ev = load_event_cache()
    df = prepare_event_table(ev)

    sp = prepare_spatial_table(load_spatial_cache())

    log(f"[INFO] events used: {len(df):,}")
    log(f"[INFO] detected intensity proxy: {df.attrs.get('intensity_col', None)}")

    reg = regional_coefficients(df)
    r2tab = unique_r2_table(df)

    draw_main_figure(df, reg)
    draw_supplementary_figure(df, sp, r2tab)

    NOTE_OUT.write_text(
        "\n".join([
            "Figure 5 interpretation note",
            "============================",
            "",
            "This version fixes the major layout problems in the previous supplementary figure:",
            "1. panel letters are placed outside the axes in figure coordinates;",
            "2. the three top-row maps use fixed colour bars and therefore keep the same panel size;",
            "3. lower-row panels b/c/d are aligned by a single bottom subgridspec;",
            "4. all legends are moved to a dedicated bottom legend row;",
            "5. the previous vertical 'Front dry-state maintenance' colour-bar label is replaced by a horizontal colour bar;",
            "6. the figure height is increased to prevent label, legend and colour-bar overlap.",
            "",
            "Interpretation boundary:",
            "The figure supports the statement that dry-state maintenance at advancing fronts is more consistently associated",
            "with local land-surface coupling than with Tadv850 or VIMD. It should not be interpreted as evidence that",
            "advection is irrelevant.",
        ]),
        encoding="utf-8",
    )

    log(f"[SAVED] {NOTE_OUT}")

    log("=" * 100)
    log("[DONE]")
    log(f"[DONE] Main figure       : {MAIN_PNG}")
    log(f"[DONE] Main PDF          : {MAIN_PDF}")
    log(f"[DONE] Supplementary fig : {SUPP_PNG}")
    log(f"[DONE] Supplementary PDF : {SUPP_PDF}")
    log("=" * 100)


if __name__ == "__main__":
    main()