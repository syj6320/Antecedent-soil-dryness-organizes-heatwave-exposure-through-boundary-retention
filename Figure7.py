# -*- coding: utf-8 -*-
"""
CMIP6 projection figure package | editorial v11
==============================================

This version fixes:
1) Figure 5d no-data problem:
   - Replace empty agreement map with SSP5-8.5 state-by-year exposure heatmap.
   - This uses the existing annual model cache and will not be empty.

2) Supplementary Fig. f y-axis:
   - Front/local support ratio is plotted with automatic focused y-limits.
   - The variation is no longer compressed by a 0–1.25 axis.

3) Supplementary map/colorbar spacing:
   - Map colorbars are attached directly underneath the map using inset axes.
   - This avoids the excessive gap caused by cartopy aspect-ratio constraints.

4) All subplot titles are removed.
   - Panel information is carried by axis labels, colorbars, scenario tags, and figure captions.

Outputs
-------
Figure5_CMIP6_continuous_spatial_rebuilt_v11.png/pdf  # rebuilt with panels a-h
Figure5_CMIP6_transition_flow_panel_h_v11.png/pdf/svg
Supplementary_Fig_CMIP6_spatial_uncertainty_support_rebuilt_v9.png/pdf
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

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
# 1. Paths
# =============================================================================

CMIP6_ROOT = Path(
    r"E:\第二篇的修改20251229开始修改\CC3D_STATES6_UNIFIED_outputs_ROLL30每个模型的"
)

TS_DIR = Path(
    r"E:\第二篇的修改20251229开始修改"
    r"\CMIP6_MME_Timeseries_2x3_5metrics_ROLL30_fromEvents_ANOM"
)

SPMAP_DIR = Path(
    r"E:\第二篇的修改20251229开始修改"
    r"\CMIP6_MME_DiffMaps_FULL6x3_fromCC3D_ROLL30"
)

OUT_DIR = CMIP6_ROOT / "_cmip6_projection_editorial_v11"


# =============================================================================
# 2. Constants
# =============================================================================

HIST = (1950, 2014)
FUT = (2015, 2100)
BASELINE = (1985, 2014)
ENDCENTURY = (2071, 2100)

SCENARIOS = ["ssp126", "ssp245", "ssp585"]

SC_LABEL = {
    "historical": "Historical",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5",
}

SC_COLOR = {
    "historical": "#111111",
    "ssp126": "#2b83ba",
    "ssp245": "#1a9850",
    "ssp585": "#6a3d9a",
}

DRY_STATES = [1, 2]
WET_STATES = [5, 6]

STATE_ORDER = [1, 2, 3, 4, 5, 6]
STATE_LABELS = [f"S{i}" for i in STATE_ORDER]

CONUS_EXTENT = (-125, -66, 24, 50.5)

DPI = 450
ROLL = 9
AGREE_THR = 0.75

ALLOW_PERIOD_DIFF_FALLBACK = True

PANEL_LABEL_DX_STD = 0.032
PANEL_LABEL_DY_STD = 0.006
PANEL_LABEL_DX_MAP = 0.032
PANEL_LABEL_DY_MAP = 0.006
PANEL_LABEL_DX_HEAT = 0.032
PANEL_LABEL_DY_HEAT = 0.006

# Source-state colours used by the compact transition-flow panel.
# The palette follows the baseline Figure 5f style: dry states are brown/orange,
# intermediate states are light yellow/teal, and wet states are teal/green.
STATE_COLORS = {
    1: "#9b5a08",
    2: "#c8872d",
    3: "#e3c77d",
    4: "#7fcdbb",
    5: "#2f9c95",
    6: "#006c5b",
}

# New standalone CMIP6 flow panel. The original Figure 5 and Supplementary
# Figure are still generated unchanged; this file is an additional output that
# mimics the visual grammar of Figure5_baseline_front_local_transition... panel f.
FLOW_MIN_PROB = 0.080
FLOW_PANEL_SCENARIO = "ssp585"
FLOW_PANEL_PERIOD = ENDCENTURY
FLOW_FIG_STEM = "Figure5_CMIP6_transition_flow_panel_h_v11"


# =============================================================================
# 3. Matplotlib style
# =============================================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 29,
    "axes.labelsize": 32,
    "xtick.labelsize": 27,
    "ytick.labelsize": 27,
    "legend.fontsize": 20,
    "axes.linewidth": 1.25,
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


# =============================================================================
# 4. Utilities
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(fig, ax, letter: str, dx: float = 0.012, dy: float = 0.004, size: int = 40) -> None:
    """Place panel letters outside the full panel bounding box (including axis labels),
    using the tight bounding box so letters do not overlap y-axis labels.
    dx and dy are figure-coordinate offsets.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    x = max(0.002, tight.x0 - dx)
    y = min(0.995, tight.y1 + dy)
    fig.text(
        x, y, letter,
        ha="left",
        va="top",
        fontsize=size,
        fontweight="bold",
        zorder=300,
    )



def add_group_panel_label(fig, axes, letter: str, dx: float = 0.032, dy: float = 0.006, size: int = 42) -> None:
    """Place a panel letter outside the union of several axes.

    This is used for composite panels (f, g, h), so the label is outside the
    full panel box rather than sitting on top of an internal y-axis label.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax in axes:
        if ax is None:
            continue
        try:
            boxes.append(ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()))
        except Exception:
            pass
    if not boxes:
        return
    x0 = min(b.x0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    x = max(0.002, x0 - dx)
    y = min(0.995, y1 + dy)
    fig.text(
        x, y, letter,
        ha="left",
        va="top",
        fontsize=size,
        fontweight="bold",
        zorder=300,
    )


def rolling_mean(y, w: int = ROLL) -> np.ndarray:
    return (
        pd.Series(y, dtype=float)
        .rolling(window=w, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def safe_num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def savefig(fig, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")


def summarize_ensemble(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=group_cols + ["median", "q25", "q75", "n"])

    d = df.copy()
    d[value_col] = safe_num(d[value_col])

    out = (
        d.groupby(group_cols, observed=True)[value_col]
        .agg(
            median=lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan,
            q25=lambda x: float(np.nanpercentile(x.dropna(), 25)) if x.dropna().size else np.nan,
            q75=lambda x: float(np.nanpercentile(x.dropna(), 75)) if x.dropna().size else np.nan,
            n=lambda x: int(np.isfinite(x).sum()),
        )
        .reset_index()
    )
    return out


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {norm_text(c): c for c in cols}

    for cand in candidates:
        key = norm_text(cand)
        if key in norm_map:
            return norm_map[key]

    for cand in candidates:
        key = norm_text(cand)
        for nk, original in norm_map.items():
            if key in nk or nk in key:
                return original

    if required:
        raise KeyError(
            f"Cannot find required column. Candidates={candidates}\n"
            f"Available columns={cols}"
        )
    return None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in [".pkl", ".pickle"]:
        return pd.read_pickle(path)

    return pd.read_csv(path, low_memory=False)


def recursive_files(root: Path, suffixes=(".csv", ".parquet", ".pkl", ".pickle")) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]


def parse_state(v):
    if pd.isna(v):
        return np.nan

    s = str(v).strip()

    m = re.search(r"S\s*([1-6])", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    try:
        iv = int(float(s))
        if 1 <= iv <= 6:
            return iv
    except Exception:
        pass

    return np.nan


def infer_state_from_path(fp: Path) -> Optional[int]:
    text = str(fp).replace("\\", "/")

    m = re.search(r"S([1-6])", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r"state[_\-]?([1-6])", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def add_state_idx(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "state_idx" in out.columns:
        out["state_idx"] = out["state_idx"].apply(parse_state)
        return out

    for c in ["state", "S_bin", "Sbin", "start_state", "initial_state", "state_num"]:
        if c in out.columns:
            out["state_idx"] = out[c].apply(parse_state)
            return out

    return out


def unique_historical(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "year" not in df.columns or "scenario" not in df.columns:
        return df.copy()

    d = df.copy()
    d["year"] = safe_num(d["year"])
    d["scenario"] = d["scenario"].astype(str).str.lower()

    hist = d[d["year"].between(*HIST)].copy()
    fut = d[d["year"] >= FUT[0]].copy()

    if not hist.empty:
        subset = [
            c for c in ["model", "year", "state_idx", "kernel", "from_state", "to_state", "start_state"]
            if c in hist.columns
        ]

        if not subset:
            subset = [c for c in ["model", "year"] if c in hist.columns]

        hist = hist.sort_values([c for c in ["model", "year", "scenario"] if c in hist.columns])
        hist = hist.drop_duplicates(subset=subset, keep="first")
        hist["scenario"] = "historical"

    return pd.concat([hist, fut], ignore_index=True)


def infer_metric_columns(df: pd.DataFrame) -> Tuple[str, str]:
    event_candidates = [
        "event_days_awmean", "event_days", "days_awmean", "days",
        "dry_event_days", "eventday", "event_days_mean"
    ]

    prob_candidates = [
        "prob_awmean", "probability_awmean", "event_probability_awmean",
        "prob", "probability", "event_probability"
    ]

    event_col = None
    prob_col = None

    for c in event_candidates:
        if c in df.columns:
            event_col = c
            break

    for c in prob_candidates:
        if c in df.columns:
            prob_col = c
            break

    if event_col is None:
        for c in df.columns:
            if ("event" in c.lower()) and ("day" in c.lower()):
                event_col = c
                break

    if prob_col is None:
        for c in df.columns:
            if "prob" in c.lower():
                prob_col = c
                break

    if event_col is None or prob_col is None:
        raise KeyError(
            "Cannot infer event-days / probability columns.\n"
            f"Available columns: {list(df.columns)}"
        )

    return event_col, prob_col


def sign_agreement(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    med = np.nanmedian(x)

    if med >= 0:
        return float(np.mean(x >= 0))
    return float(np.mean(x < 0))


# =============================================================================
# 5. Map helpers
# =============================================================================

def infer_edges(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(sorted(np.unique(vals)), dtype=float)

    if vals.size == 0:
        return np.array([])

    if vals.size == 1:
        d = 1.0
        return np.array([vals[0] - d / 2, vals[0] + d / 2])

    mid = (vals[:-1] + vals[1:]) / 2
    first = vals[0] - (mid[0] - vals[0])
    last = vals[-1] + (vals[-1] - mid[-1])
    return np.r_[first, mid, last]


def make_geo_ax(fig, spec):
    if HAS_CARTOPY:
        ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, edgecolor="0.55")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, edgecolor="0.60")

        try:
            states = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces_lines",
                scale="50m",
                facecolor="none",
            )
            ax.add_feature(states, linewidth=0.35, edgecolor="0.75")
        except Exception:
            pass

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.45,
            color="0.82",
            alpha=0.8,
            linestyle="-",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 19}
        gl.ylabel_style = {"size": 19}
        return ax

    ax = fig.add_subplot(spec)
    ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
    ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
    ax.set_xticks([-120, -110, -100, -90, -80, -70])
    ax.set_xticklabels(["120°W", "110°W", "100°W", "90°W", "80°W", "70°W"], fontsize=19)
    ax.set_yticks([25, 30, 35, 40, 45, 50])
    ax.set_yticklabels(["25°N", "30°N", "35°N", "40°N", "45°N", "50°N"], fontsize=19)
    ax.grid(color="0.86", linewidth=0.55)
    clean_spines(ax)
    return ax


def standardize_lon_lat(df: pd.DataFrame) -> pd.DataFrame:
    lon_col = find_col(df, ["lon", "longitude", "x"], required=False)
    lat_col = find_col(df, ["lat", "latitude", "y"], required=False)

    if lon_col is None or lat_col is None:
        return pd.DataFrame()

    out = df.copy()
    out = out.rename(columns={lon_col: "lon", lat_col: "lat"})
    out["lon"] = safe_num(out["lon"])
    out["lat"] = safe_num(out["lat"])
    return out


def choose_value_col(df: pd.DataFrame, preferred_metric: Optional[str] = None) -> Optional[str]:
    preferred = []

    if preferred_metric:
        preferred.extend([
            preferred_metric,
            preferred_metric.replace("_awmean", ""),
            preferred_metric.replace("_", ""),
        ])

    preferred.extend([
        "value", "metric_value", "diff", "delta", "change",
        "response", "trend", "slope", "median", "mean",
        "event_days_awmean", "event_days", "days", "days_awmean"
    ])

    for c in preferred:
        if c in df.columns:
            return c

    banned = {
        "lon", "lat", "longitude", "latitude", "x", "y",
        "year", "model", "scenario", "state", "state_idx",
        "kernel", "from_state", "to_state", "agreement", "iqr",
        "q25", "q75", "n_model", "count", "n", "n_events",
    }

    numeric = []
    for c in df.columns:
        if c in banned:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s).sum() > 10:
            numeric.append(c)

    return numeric[0] if numeric else None


def add_stippling(ax, df: pd.DataFrame, thr: float = AGREE_THR):
    if df.empty or "agreement" not in df.columns:
        return

    g = df.copy()
    g["agreement"] = safe_num(g["agreement"])
    g = g[g["agreement"] >= thr].dropna(subset=["lon", "lat"])

    if g.empty:
        return

    kwargs = dict(
        s=5.2,
        color="black",
        alpha=0.28,
        linewidths=0,
        zorder=10,
        rasterized=True,
    )

    if HAS_CARTOPY:
        kwargs["transform"] = ccrs.PlateCarree()

    ax.scatter(g["lon"], g["lat"], **kwargs)


def plot_grid_map(ax, df: pd.DataFrame, value_col: str, cmap, norm):
    d = df.copy().dropna(subset=["lon", "lat", value_col])

    if d.empty:
        return None

    d = d.groupby(["lon", "lat"], as_index=False)[value_col].mean()
    lons = np.sort(d["lon"].unique())
    lats = np.sort(d["lat"].unique())

    pivot = d.pivot_table(index="lat", columns="lon", values=value_col)
    pivot = pivot.reindex(index=lats, columns=lons)

    X = infer_edges(lons)
    Y = infer_edges(lats)
    Z = pivot.values

    kwargs = dict(cmap=cmap, norm=norm, shading="auto", rasterized=True)

    if HAS_CARTOPY:
        kwargs["transform"] = ccrs.PlateCarree()

    im = ax.pcolormesh(X, Y, Z, **kwargs)
    return im


def draw_spatial_panel(
    fig,
    spec,
    spatial_df: pd.DataFrame,
    value_col: str,
    cmap,
    norm,
    cbar_label: str,
    panel_letter: str,
    show_stipple: bool = True,
    scenario_tag: Optional[str] = None,
    add_colorbar: bool = True,
    colorbar_box: Tuple[float, float, float, float] = (0.02, -0.245, 0.96, 0.075),
):
    # Single map axis + inset colorbar.
    # This prevents the large gap caused by cartopy aspect constraints inside nested GridSpec.
    ax = make_geo_ax(fig, spec)
    im = plot_grid_map(ax, spatial_df, value_col, cmap, norm)

    if show_stipple:
        add_stippling(ax, spatial_df, thr=AGREE_THR)

    add_panel_label(fig, ax, panel_letter, dx=PANEL_LABEL_DX_MAP, dy=PANEL_LABEL_DY_MAP)

    if scenario_tag is not None:
        ax.text(
            0.015, 0.985, scenario_tag,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=22,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.85", alpha=0.92),
            zorder=30,
        )

    if im is not None and add_colorbar:
        cax = ax.inset_axes(list(colorbar_box))
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(cbar_label, fontsize=23, labelpad=3)
        cb.ax.tick_params(labelsize=18)
    else:
        cax = None

    return ax, cax, im


# =============================================================================
# 6. Annual cache and dry/wet series
# =============================================================================

def load_annual_state_cache(ts_dir: Path) -> pd.DataFrame:
    cache_dir = ts_dir / "_cache_modelAnnual_fromEvents"

    if not cache_dir.exists():
        raise FileNotFoundError(f"Annual cache directory not found:\n{cache_dir}")

    files = sorted(cache_dir.glob("annual_*.pkl"))

    if not files:
        raise FileNotFoundError(f"No annual_*.pkl found in:\n{cache_dir}")

    rows = []

    for fp in files:
        name = fp.stem.replace("annual_", "")
        parts = name.split("_")

        if len(parts) < 2:
            continue

        scenario = parts[0].lower()
        model = "_".join(parts[1:])

        df = pd.read_pickle(fp).copy()
        df["scenario"] = scenario
        df["model"] = model

        if "year" in df.columns:
            df["year"] = safe_num(df["year"])

        df = add_state_idx(df)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out = unique_historical(out)
    return out


def aggregate_regime_series(df: pd.DataFrame, metric_col: str, states: List[int], regime: str) -> pd.DataFrame:
    d = df[df["state_idx"].isin(states)].copy()
    d[metric_col] = safe_num(d[metric_col])

    out = (
        d.groupby(["scenario", "model", "year"], as_index=False)[metric_col]
        .sum()
        .rename(columns={metric_col: "value"})
    )
    out["regime"] = regime
    return out


def build_dry_wet_series(df: pd.DataFrame, metric_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dry = aggregate_regime_series(df, metric_col, DRY_STATES, "dry")
    wet = aggregate_regime_series(df, metric_col, WET_STATES, "wet")

    dmw = dry.merge(
        wet[["scenario", "model", "year", "value"]],
        on=["scenario", "model", "year"],
        how="inner",
        suffixes=("_dry", "_wet"),
    )
    dmw["contrast"] = dmw["value_dry"] - dmw["value_wet"]
    return dry, wet, dmw


# =============================================================================
# 7. Spatial loading
# =============================================================================

def has_scenario_in_path(fp: Path, scenario: str) -> bool:
    return scenario.lower() in str(fp).lower()


def has_eventday_in_path(fp: Path) -> bool:
    text = str(fp).lower()
    keys = ["event_days", "eventdays", "event_day", "days", "event-days"]
    return any(k in text for k in keys)


def load_annual_spatial_trend_for_scenario(
    spmap_dir: Path,
    scenario: str,
    preferred_metric: Optional[str] = None,
) -> Optional[pd.DataFrame]:

    files = [
        p for p in recursive_files(spmap_dir)
        if has_scenario_in_path(p, scenario) and has_eventday_in_path(p)
    ]

    annual_parts = []

    for fp in files:
        try:
            df0 = read_table(fp)
        except Exception:
            continue

        if df0.empty:
            continue

        df = standardize_lon_lat(df0)

        if df.empty:
            continue

        if "year" not in df.columns or "model" not in df.columns:
            continue

        val_col = choose_value_col(df, preferred_metric=preferred_metric)

        if val_col is None:
            continue

        df["year"] = safe_num(df["year"])
        df["model"] = df["model"].astype(str)
        df["value"] = safe_num(df[val_col])

        if "scenario" in df.columns:
            scen = df["scenario"].astype(str).str.lower()
            df = df[scen.eq(scenario.lower()) | scen.str.contains(scenario.lower())]
            if df.empty:
                continue

        if "state_idx" not in df.columns:
            if "state" in df.columns:
                df["state_idx"] = df["state"].apply(parse_state)
            else:
                st = infer_state_from_path(fp)
                if st is not None:
                    df["state_idx"] = st

        if "state_idx" not in df.columns:
            continue

        df["state_idx"] = df["state_idx"].apply(parse_state)

        keep = ["lon", "lat", "year", "model", "state_idx", "value"]
        part = df[keep].dropna()

        if not part.empty:
            annual_parts.append(part)

    if not annual_parts:
        return None

    annual = pd.concat(annual_parts, ignore_index=True)
    annual = annual[annual["year"].between(*FUT)].copy()

    annual["regime"] = np.where(
        annual["state_idx"].isin(DRY_STATES),
        "dry",
        np.where(annual["state_idx"].isin(WET_STATES), "wet", np.nan),
    )
    annual = annual.dropna(subset=["regime"])

    reg = (
        annual.groupby(["model", "year", "lon", "lat", "regime"], as_index=False)["value"]
        .sum()
    )

    piv = (
        reg.pivot_table(
            index=["model", "year", "lon", "lat"],
            columns="regime",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    if "dry" not in piv.columns or "wet" not in piv.columns:
        return None

    piv["contrast"] = piv["dry"] - piv["wet"]

    trend_rows = []

    for (model, lon, lat), g in piv.groupby(["model", "lon", "lat"], observed=True):
        x = safe_num(g["year"]).to_numpy(float)
        y = safe_num(g["contrast"]).to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)

        if m.sum() < 20:
            continue

        if stats is not None:
            slope = stats.linregress(x[m], y[m]).slope * 10.0
        else:
            X = np.vstack([np.ones(m.sum()), x[m]]).T
            coef = np.linalg.lstsq(X, y[m], rcond=None)[0]
            slope = coef[1] * 10.0

        trend_rows.append({"model": model, "lon": lon, "lat": lat, "trend": slope})

    tr = pd.DataFrame(trend_rows)

    if tr.empty:
        return None

    summary = (
        tr.groupby(["lon", "lat"], as_index=False)["trend"]
        .agg(
            value=lambda x: float(np.nanmedian(x)),
            q25=lambda x: float(np.nanpercentile(x.dropna(), 25)) if x.dropna().size else np.nan,
            q75=lambda x: float(np.nanpercentile(x.dropna(), 75)) if x.dropna().size else np.nan,
            n_model=lambda x: int(np.isfinite(x).sum()),
        )
    )
    summary["iqr"] = summary["q75"] - summary["q25"]

    agr = tr.groupby(["lon", "lat"])["trend"].apply(sign_agreement).reset_index(name="agreement")
    summary = summary.merge(agr, on=["lon", "lat"], how="left")
    summary["scenario"] = scenario
    summary["spatial_mode"] = "future_trend"
    return summary


def load_cache_diff_for_scenario(
    spmap_dir: Path,
    scenario: str,
    preferred_metric: Optional[str] = None,
) -> Optional[pd.DataFrame]:

    files = [
        p for p in recursive_files(spmap_dir)
        if has_scenario_in_path(p, scenario) and has_eventday_in_path(p)
    ]

    parts = []

    for fp in files:
        try:
            df0 = read_table(fp)
        except Exception:
            continue

        if df0.empty:
            continue

        df = standardize_lon_lat(df0)

        if df.empty:
            continue

        val_col = choose_value_col(df, preferred_metric=preferred_metric)

        if val_col is None:
            continue

        df["value_raw"] = safe_num(df[val_col])

        if "scenario" in df.columns:
            scen = df["scenario"].astype(str).str.lower()
            df = df[scen.eq(scenario.lower()) | scen.str.contains(scenario.lower())]
            if df.empty:
                continue

        if "state_idx" not in df.columns:
            if "state" in df.columns:
                df["state_idx"] = df["state"].apply(parse_state)
            else:
                st = infer_state_from_path(fp)
                if st is not None:
                    df["state_idx"] = st

        if "state_idx" not in df.columns:
            continue

        df["state_idx"] = df["state_idx"].apply(parse_state)

        keep = ["lon", "lat", "state_idx", "value_raw"]

        if "model" in df.columns:
            keep.append("model")

        for c in ["agreement", "model_agreement", "same_sign_fraction"]:
            if c in df.columns:
                df["agreement"] = safe_num(df[c])
                keep.append("agreement")
                break

        for c in ["iqr", "spread", "model_iqr"]:
            if c in df.columns:
                df["iqr"] = safe_num(df[c])
                keep.append("iqr")
                break

        part = df[keep].dropna(subset=["lon", "lat", "state_idx", "value_raw"])

        if not part.empty:
            part["source_file"] = str(fp)
            parts.append(part)

    if not parts:
        return None

    d = pd.concat(parts, ignore_index=True)

    d["regime"] = np.where(
        d["state_idx"].isin(DRY_STATES),
        "dry",
        np.where(d["state_idx"].isin(WET_STATES), "wet", np.nan),
    )
    d = d.dropna(subset=["regime"])

    if "model" in d.columns:
        reg = d.groupby(["model", "lon", "lat", "regime"], as_index=False)["value_raw"].sum()

        piv = (
            reg.pivot_table(
                index=["model", "lon", "lat"],
                columns="regime",
                values="value_raw",
                aggfunc="mean",
            )
            .reset_index()
        )

        if "dry" in piv.columns and "wet" in piv.columns:
            piv["model_response"] = piv["dry"] - piv["wet"]

            summary = (
                piv.groupby(["lon", "lat"], as_index=False)["model_response"]
                .agg(
                    value=lambda x: float(np.nanmedian(x)),
                    q25=lambda x: float(np.nanpercentile(x.dropna(), 25)) if x.dropna().size else np.nan,
                    q75=lambda x: float(np.nanpercentile(x.dropna(), 75)) if x.dropna().size else np.nan,
                    n_model=lambda x: int(np.isfinite(x).sum()),
                )
            )
            summary["iqr"] = summary["q75"] - summary["q25"]

            agr = piv.groupby(["lon", "lat"])["model_response"].apply(sign_agreement).reset_index(name="agreement")
            summary = summary.merge(agr, on=["lon", "lat"], how="left")
            summary["scenario"] = scenario
            summary["spatial_mode"] = "period_response_model_level"
            return summary

    reg = d.groupby(["lon", "lat", "regime"], as_index=False)["value_raw"].sum()

    piv = (
        reg.pivot_table(
            index=["lon", "lat"],
            columns="regime",
            values="value_raw",
            aggfunc="mean",
        )
        .reset_index()
    )

    if "dry" not in piv.columns or "wet" not in piv.columns:
        return None

    piv["value"] = piv["dry"] - piv["wet"]

    if "agreement" in d.columns and np.isfinite(safe_num(d["agreement"])).any():
        agr = d.groupby(["lon", "lat"], as_index=False)["agreement"].mean()
        piv = piv.merge(agr, on=["lon", "lat"], how="left")
    else:
        piv["agreement"] = np.nan

    if "iqr" in d.columns and np.isfinite(safe_num(d["iqr"])).any():
        spr = d.groupby(["lon", "lat"], as_index=False)["iqr"].mean()
        piv = piv.merge(spr, on=["lon", "lat"], how="left")
    else:
        piv["iqr"] = np.nan

    piv["q25"] = np.nan
    piv["q75"] = np.nan
    piv["n_model"] = np.nan
    piv["scenario"] = scenario
    piv["spatial_mode"] = "period_response_mme_fallback"

    return piv[["lon", "lat", "value", "q25", "q75", "iqr", "agreement", "n_model", "scenario", "spatial_mode"]]


def load_spatial_for_scenario(
    spmap_dir: Path,
    scenario: str,
    preferred_metric: Optional[str] = None,
) -> pd.DataFrame:
    log(f"[INFO] Loading spatial data for {scenario}: annual trend -> fallback cache_diff")

    out = load_annual_spatial_trend_for_scenario(
        spmap_dir=spmap_dir,
        scenario=scenario,
        preferred_metric=preferred_metric,
    )

    if out is not None and not out.empty:
        log(f"       using annual trend spatial cache: {out.shape[0]} cells")
        return out

    if not ALLOW_PERIOD_DIFF_FALLBACK:
        raise FileNotFoundError(
            f"No annual spatial cache found for {scenario}, and period-diff fallback is disabled."
        )

    out = load_cache_diff_for_scenario(
        spmap_dir=spmap_dir,
        scenario=scenario,
        preferred_metric=preferred_metric,
    )

    if out is not None and not out.empty:
        mode = str(out["spatial_mode"].iloc[0])
        log(f"       using fallback mode={mode}: {out.shape[0]} cells")
        return out

    raise FileNotFoundError(
        f"No usable spatial cache found for {scenario} under:\n{spmap_dir}\n"
        "The code tried both annual trend cache and cache_diff fallback."
    )


# =============================================================================
# 8. Transition tables
# =============================================================================

def load_transition_tables(root: Path) -> Dict[str, pd.DataFrame]:
    d = root / "_cmip6_result3_transition"

    yearly = read_table(d / "cmip6_result3_transition_yearly_metrics.csv")
    annual = read_table(d / "cmip6_result3_transition_annual_counts.csv")
    start = read_table(d / "cmip6_result3_transition_start_counts.csv")

    if not yearly.empty:
        yearly = yearly.copy()

        if "year" in yearly.columns:
            yearly["year"] = safe_num(yearly["year"])

        if "scenario" in yearly.columns:
            yearly["scenario"] = yearly["scenario"].astype(str).str.lower()

        if "kernel" in yearly.columns:
            yearly["kernel"] = yearly["kernel"].astype(str).str.lower().str.strip()

        if "p11" not in yearly.columns and {"metric", "value"}.issubset(yearly.columns):
            idx = [c for c in ["scenario", "model", "year", "kernel", "mode"] if c in yearly.columns]
            yearly = (
                yearly.pivot_table(index=idx, columns="metric", values="value", aggfunc="mean")
                .reset_index()
            )
            yearly.columns = [str(c) for c in yearly.columns]

        if "mode" in yearly.columns:
            mm = yearly["mode"].astype(str).str.lower()
            if mm.eq("standardized").any():
                yearly = yearly[mm.eq("standardized")].copy()

        yearly = unique_historical(yearly)

    if not annual.empty:
        annual = annual.copy()

        rename = {}

        if "count" not in annual.columns:
            for c in ["n", "n_transitions", "transitions", "support_count"]:
                if c in annual.columns:
                    rename[c] = "count"
                    break

        if "from_state" not in annual.columns:
            for c in ["from", "state_from", "source_state"]:
                if c in annual.columns:
                    rename[c] = "from_state"
                    break

        if "to_state" not in annual.columns:
            for c in ["to", "state_to", "target_state"]:
                if c in annual.columns:
                    rename[c] = "to_state"
                    break

        annual = annual.rename(columns=rename)

        for c in ["year", "count"]:
            if c in annual.columns:
                annual[c] = safe_num(annual[c])

        for c in ["from_state", "to_state"]:
            if c in annual.columns:
                annual[c] = annual[c].apply(parse_state)

        if "scenario" in annual.columns:
            annual["scenario"] = annual["scenario"].astype(str).str.lower()

        if "kernel" in annual.columns:
            annual["kernel"] = annual["kernel"].astype(str).str.lower().str.strip()

        annual = unique_historical(annual)

    if not start.empty:
        start = start.copy()

        if "start_state" not in start.columns:
            for c in ["state_idx", "state", "S_bin", "initial_state"]:
                if c in start.columns:
                    start["start_state"] = start[c].apply(parse_state)
                    break
        else:
            start["start_state"] = start["start_state"].apply(parse_state)

        if "n_events" not in start.columns:
            for c in ["count", "n", "events", "support"]:
                if c in start.columns:
                    start["n_events"] = safe_num(start[c])
                    break

        for c in ["year", "n_events"]:
            if c in start.columns:
                start[c] = safe_num(start[c])

        if "scenario" in start.columns:
            start["scenario"] = start["scenario"].astype(str).str.lower()

        start = unique_historical(start)

    return {"yearly": yearly, "annual": annual, "start": start}


def conditional_transition_probabilities(
    annual: pd.DataFrame,
    scenario: str,
    period: Tuple[int, int],
    kernel: str,
) -> pd.DataFrame:
    if annual.empty:
        return pd.DataFrame(columns=["model", "from_state", "to_state", "prob"])

    d = annual.copy()
    d = d[
        d["kernel"].eq(kernel)
        & d["scenario"].eq(scenario)
        & d["year"].between(period[0], period[1])
    ].copy()

    if d.empty:
        return pd.DataFrame(columns=["model", "from_state", "to_state", "prob"])

    g = d.groupby(["model", "from_state", "to_state"], as_index=False)["count"].sum()
    denom = g.groupby(["model", "from_state"])["count"].transform("sum")
    g["prob"] = np.where(denom > 0, g["count"] / denom, np.nan)
    return g[["model", "from_state", "to_state", "prob"]]


def transition_change_matrix(
    annual: pd.DataFrame,
    scenario: str = "ssp585",
    kernel: str = "front",
) -> np.ndarray:
    base = conditional_transition_probabilities(annual, "historical", BASELINE, kernel)
    fut = conditional_transition_probabilities(annual, scenario, ENDCENTURY, kernel)

    if base.empty or fut.empty:
        return np.full((6, 6), np.nan)

    m = base.merge(
        fut,
        on=["model", "from_state", "to_state"],
        how="inner",
        suffixes=("_base", "_fut"),
    )

    if m.empty:
        return np.full((6, 6), np.nan)

    m["diff"] = m["prob_fut"] - m["prob_base"]

    summary = (
        m.groupby(["from_state", "to_state"], as_index=False)["diff"]
        .median()
        .rename(columns={"diff": "value"})
    )

    mat = np.full((6, 6), np.nan)

    for _, r in summary.iterrows():
        i = int(r["from_state"]) - 1
        j = int(r["to_state"]) - 1
        if 0 <= i < 6 and 0 <= j < 6:
            mat[i, j] = float(r["value"])

    return mat


def support_series_from_annual(annual: pd.DataFrame) -> pd.DataFrame:
    if annual.empty:
        return pd.DataFrame(columns=["scenario", "model", "year", "kernel", "count"])

    d = annual.copy()
    out = d.groupby(["scenario", "model", "year", "kernel"], as_index=False)["count"].sum()
    out = unique_historical(out)
    return out


def start_state_support_series(start: pd.DataFrame) -> pd.DataFrame:
    if start.empty:
        return pd.DataFrame(columns=["scenario", "model", "year", "kind", "n_events"])

    d = start.copy()
    d["kind"] = np.where(
        d["start_state"].isin(DRY_STATES),
        "dry",
        np.where(d["start_state"].isin(WET_STATES), "wet", np.nan),
    )
    d = d.dropna(subset=["kind"])

    out = d.groupby(["scenario", "model", "year", "kind"], as_index=False)["n_events"].sum()
    out = unique_historical(out)
    return out


def front_local_ratio_series(annual: pd.DataFrame) -> pd.DataFrame:
    if annual.empty:
        return pd.DataFrame(columns=["scenario", "model", "year", "ratio"])

    d = support_series_from_annual(annual)
    piv = (
        d.pivot_table(
            index=["scenario", "model", "year"],
            columns="kernel",
            values="count",
            aggfunc="sum",
        )
        .reset_index()
    )

    if "front" not in piv.columns or "local" not in piv.columns:
        return pd.DataFrame(columns=["scenario", "model", "year", "ratio"])

    piv["ratio"] = np.where(piv["local"] > 0, piv["front"] / piv["local"], np.nan)
    return piv[["scenario", "model", "year", "ratio"]]


# =============================================================================
# 9. Plot helpers
# =============================================================================

def plot_hist_future_series(
    ax,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    show_legend: bool = False,
    legend_loc: str = "upper left",
):
    if df.empty or value_col not in df.columns:
        ax.text(0.5, 0.5, "Missing data", transform=ax.transAxes, ha="center", va="center")
        clean_spines(ax)
        return

    d = df.copy()
    d["year"] = safe_num(d["year"])
    d[value_col] = safe_num(d[value_col])

    hist = d[d["scenario"].eq("historical")]
    hs = summarize_ensemble(hist, ["year"], value_col).sort_values("year")

    if not hs.empty:
        x = hs["year"].to_numpy(float)
        ax.fill_between(x, hs["q25"], hs["q75"], color="0.72", alpha=0.28, lw=0)
        ax.plot(x, rolling_mean(hs["median"]), color=SC_COLOR["historical"], lw=2.7, label="Historical")

    for sc in SCENARIOS:
        fut = d[(d["scenario"].eq(sc)) & (d["year"] >= FUT[0])]
        fs = summarize_ensemble(fut, ["year"], value_col).sort_values("year")

        if fs.empty:
            continue

        x = fs["year"].to_numpy(float)
        ax.fill_between(x, fs["q25"], fs["q75"], color=SC_COLOR[sc], alpha=0.12, lw=0)
        ax.plot(x, rolling_mean(fs["median"]), color=SC_COLOR[sc], lw=2.45, label=SC_LABEL[sc])

    ax.axvline(2015, color="0.65", lw=1.0, ls="--")
    ax.set_xlim(1950, 2100)
    ax.set_xticks([1950, 2000, 2050, 2100])
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(color="0.88", lw=0.8)
    clean_spines(ax)

    if show_legend:
        ax.legend(
            frameon=False,
            loc=legend_loc,
            ncol=2,
            fontsize=19,
            handlelength=2.2,
            columnspacing=1.1,
        )


def plot_persistence_series(ax, yearly: pd.DataFrame):
    if yearly.empty or "p11" not in yearly.columns or "kernel" not in yearly.columns:
        ax.text(0.5, 0.5, "Missing transition metrics", transform=ax.transAxes, ha="center", va="center")
        clean_spines(ax)
        return

    d = yearly.copy()
    d["p11"] = safe_num(d["p11"])
    d["year"] = safe_num(d["year"])
    d["kernel"] = d["kernel"].astype(str)

    for kernel, ls in [("front", "-"), ("local", "--")]:
        dd = d[d["kernel"].eq(kernel)].copy()

        if dd.empty:
            continue

        hist = dd[dd["scenario"].eq("historical")]
        hs = summarize_ensemble(hist, ["year"], "p11").sort_values("year")

        if not hs.empty:
            x = hs["year"].to_numpy(float)
            if kernel == "front":
                ax.fill_between(x, hs["q25"], hs["q75"], color="0.72", alpha=0.28, lw=0)
            ax.plot(x, rolling_mean(hs["median"]), color="black", lw=2.55, ls=ls)

        for sc in SCENARIOS:
            fs = summarize_ensemble(
                dd[(dd["scenario"].eq(sc)) & (dd["year"] >= FUT[0])],
                ["year"],
                "p11",
            ).sort_values("year")

            if fs.empty:
                continue

            x = fs["year"].to_numpy(float)
            if kernel == "front":
                ax.fill_between(x, fs["q25"], fs["q75"], color=SC_COLOR[sc], alpha=0.11, lw=0)
            ax.plot(x, rolling_mean(fs["median"]), color=SC_COLOR[sc], lw=2.25, ls=ls)

    ax.axvline(2015, color="0.65", lw=1.0, ls="--")
    ax.set_xlim(1950, 2100)
    ax.set_xticks([1950, 2000, 2050, 2100])
    ax.set_xlabel("Year")
    ax.set_ylabel("P(S1→S1)")
    ax.grid(color="0.88", lw=0.8)
    clean_spines(ax)

    handles = [
        Line2D([0], [0], color="black", lw=2.4, ls="-", label="Front"),
        Line2D([0], [0], color="black", lw=2.4, ls="--", label="Local"),
        Line2D([0], [0], color=SC_COLOR["ssp126"], lw=2.2, label="SSP1-2.6"),
        Line2D([0], [0], color=SC_COLOR["ssp245"], lw=2.2, label="SSP2-4.5"),
        Line2D([0], [0], color=SC_COLOR["ssp585"], lw=2.2, label="SSP5-8.5"),
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower left",
        ncol=2,
        fontsize=18.5,
        handlelength=2.2,
        columnspacing=1.0,
    )


def apply_sci_y(ax, powerlimits=(4, 4)) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(powerlimits)
    ax.yaxis.set_major_formatter(formatter)


def plot_transition_support_series(ax, annual: pd.DataFrame):
    d = support_series_from_annual(annual)

    if d.empty:
        ax.text(0.5, 0.5, "Missing support counts", transform=ax.transAxes, ha="center", va="center")
        clean_spines(ax)
        return

    for kernel, ls in [("front", "-"), ("local", "--")]:
        dd = d[d["kernel"].eq(kernel)].copy()

        if dd.empty:
            continue

        hist = summarize_ensemble(dd[dd["scenario"].eq("historical")], ["year"], "count").sort_values("year")

        if not hist.empty:
            x = hist["year"].to_numpy(float)
            ax.plot(x, rolling_mean(hist["median"]), color="black", lw=2.45, ls=ls)

        for sc in SCENARIOS:
            fut = summarize_ensemble(
                dd[(dd["scenario"].eq(sc)) & (dd["year"] >= FUT[0])],
                ["year"],
                "count",
            ).sort_values("year")

            if fut.empty:
                continue

            x = fut["year"].to_numpy(float)
            ax.plot(x, rolling_mean(fut["median"]), color=SC_COLOR[sc], lw=2.2, ls=ls)

    ax.axvline(2015, color="0.65", lw=1.0, ls="--")
    ax.set_xlim(1950, 2100)
    ax.set_xticks([1950, 2000, 2050, 2100])
    ax.set_xlabel("Year")
    ax.set_ylabel("Transitions")
    ax.grid(color="0.88", lw=0.8)
    clean_spines(ax)
    apply_sci_y(ax, powerlimits=(4, 4))

    handles = [
        Line2D([0], [0], color="black", lw=2.4, ls="-", label="Front"),
        Line2D([0], [0], color="black", lw=2.4, ls="--", label="Local"),
        Line2D([0], [0], color=SC_COLOR["ssp126"], lw=2.2, label="SSP1-2.6"),
        Line2D([0], [0], color=SC_COLOR["ssp245"], lw=2.2, label="SSP2-4.5"),
        Line2D([0], [0], color=SC_COLOR["ssp585"], lw=2.2, label="SSP5-8.5"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2, fontsize=18)


def plot_start_state_support(ax, start: pd.DataFrame):
    d = start_state_support_series(start)

    if d.empty:
        ax.text(0.5, 0.5, "Missing start-state support", transform=ax.transAxes, ha="center", va="center")
        clean_spines(ax)
        return

    for kind, ls in [("dry", "-"), ("wet", "--")]:
        dd = d[d["kind"].eq(kind)].copy()

        if dd.empty:
            continue

        hist = summarize_ensemble(dd[dd["scenario"].eq("historical")], ["year"], "n_events").sort_values("year")

        if not hist.empty:
            x = hist["year"].to_numpy(float)
            ax.plot(x, rolling_mean(hist["median"]), color="black", lw=2.45, ls=ls)

        for sc in SCENARIOS:
            fut = summarize_ensemble(
                dd[(dd["scenario"].eq(sc)) & (dd["year"] >= FUT[0])],
                ["year"],
                "n_events",
            ).sort_values("year")

            if fut.empty:
                continue

            x = fut["year"].to_numpy(float)
            ax.plot(x, rolling_mean(fut["median"]), color=SC_COLOR[sc], lw=2.2, ls=ls)

    ax.axvline(2015, color="0.65", lw=1.0, ls="--")
    ax.set_xlim(1950, 2100)
    ax.set_xticks([1950, 2000, 2050, 2100])
    ax.set_xlabel("Year")
    ax.set_ylabel("Events")
    ax.grid(color="0.88", lw=0.8)
    clean_spines(ax)
    apply_sci_y(ax, powerlimits=(3, 3))

    handles = [
        Line2D([0], [0], color="black", lw=2.4, ls="-", label="Dry starts"),
        Line2D([0], [0], color="black", lw=2.4, ls="--", label="Wet starts"),
        Line2D([0], [0], color=SC_COLOR["ssp126"], lw=2.2, label="SSP1-2.6"),
        Line2D([0], [0], color=SC_COLOR["ssp245"], lw=2.2, label="SSP2-4.5"),
        Line2D([0], [0], color=SC_COLOR["ssp585"], lw=2.2, label="SSP5-8.5"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2, fontsize=18)


def plot_support_ratio(ax, annual: pd.DataFrame):
    d = front_local_ratio_series(annual)

    if d.empty:
        ax.text(0.5, 0.5, "Missing support ratio", transform=ax.transAxes, ha="center", va="center")
        clean_spines(ax)
        return

    hist = summarize_ensemble(d[d["scenario"].eq("historical")], ["year"], "ratio").sort_values("year")

    if not hist.empty:
        x = hist["year"].to_numpy(float)
        ax.fill_between(x, hist["q25"], hist["q75"], color="0.72", alpha=0.28, lw=0)
        ax.plot(x, rolling_mean(hist["median"]), color="black", lw=2.55)

    for sc in SCENARIOS:
        fut = summarize_ensemble(
            d[(d["scenario"].eq(sc)) & (d["year"] >= FUT[0])],
            ["year"],
            "ratio",
        ).sort_values("year")

        if fut.empty:
            continue

        x = fut["year"].to_numpy(float)
        ax.fill_between(x, fut["q25"], fut["q75"], color=SC_COLOR[sc], alpha=0.10, lw=0)
        ax.plot(x, rolling_mean(fut["median"]), color=SC_COLOR[sc], lw=2.2)

    ax.axhline(1.0, color="0.65", lw=0.9, ls=":")
    ax.axvline(2015, color="0.65", lw=1.0, ls="--")
    ax.set_xlim(1950, 2100)
    ax.set_xticks([1950, 2000, 2050, 2100])
    ax.set_xlabel("Year")
    ax.set_ylabel("Front / local transitions")
    ax.grid(color="0.88", lw=0.8)
    clean_spines(ax)

    vals = pd.to_numeric(d["ratio"], errors="coerce").dropna().values
    vals = vals[np.isfinite(vals)]

    if vals.size:
        lo = np.nanpercentile(vals, 2)
        hi = np.nanpercentile(vals, 98)
        pad = 0.22 * (hi - lo) if hi > lo else 0.03

        ymin = max(0.0, lo - pad)
        ymax = hi + pad

        if ymax - ymin < 0.08:
            mid = 0.5 * (ymin + ymax)
            ymin = max(0.0, mid - 0.04)
            ymax = mid + 0.04

        ax.set_ylim(ymin, ymax)


# =============================================================================
# 10. Main Figure 5 panel d: state-year heatmap
# =============================================================================

def build_state_year_surface(
    annual_cache: pd.DataFrame,
    metric_col: str,
    scenario: str = "ssp585",
) -> pd.DataFrame:
    d = annual_cache.copy()
    d["year"] = safe_num(d["year"])
    d[metric_col] = safe_num(d[metric_col])
    d["state_idx"] = d["state_idx"].apply(parse_state)
    d["scenario"] = d["scenario"].astype(str).str.lower()

    hist = d[d["scenario"].eq("historical") & d["year"].between(*HIST)].copy()
    fut = d[d["scenario"].eq(scenario) & d["year"].between(*FUT)].copy()

    use = pd.concat([hist, fut], ignore_index=True)
    use = use[use["state_idx"].isin(STATE_ORDER)].copy()

    g = (
        use.groupby(["year", "state_idx", "model"], as_index=False)[metric_col]
        .mean()
    )

    ens = (
        g.groupby(["year", "state_idx"], as_index=False)[metric_col]
        .median()
        .rename(columns={metric_col: "value"})
    )

    return ens


def draw_state_year_heatmap(
    fig,
    spec,
    annual_cache: pd.DataFrame,
    metric_col: str,
    scenario: str,
    panel_letter: str,
):
    ax = fig.add_subplot(spec)

    surf = build_state_year_surface(annual_cache, metric_col, scenario=scenario)

    if surf.empty:
        ax.text(0.5, 0.5, "Missing state-year surface", ha="center", va="center", transform=ax.transAxes)
        clean_spines(ax)
        return

    years = np.sort(surf["year"].unique())
    states = np.array(STATE_ORDER)

    pivot = (
        surf.pivot_table(index="state_idx", columns="year", values="value", aggfunc="mean")
        .reindex(index=states, columns=years)
    )

    Z = pivot.values

    x_edges = infer_edges(years)
    y_edges = np.r_[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

    vmax = np.nanpercentile(Z[np.isfinite(Z)], 98) if np.isfinite(Z).any() else 1.0
    vmax = max(vmax, 1.0)

    im = ax.pcolormesh(
        x_edges,
        y_edges,
        Z,
        cmap="YlOrRd",
        norm=mcolors.Normalize(vmin=0, vmax=vmax),
        shading="auto",
        rasterized=True,
    )

    ax.axvline(2015, color="0.25", lw=1.0, ls="--")
    ax.set_xlim(1950, 2100)
    ax.set_xticks([1950, 2000, 2050, 2100])
    ax.set_yticks(STATE_ORDER)
    ax.set_yticklabels(STATE_LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Year")
    ax.set_ylabel("Initial state")

    ax.text(
        0.02, 0.98, SC_LABEL.get(scenario, scenario),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=22,
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.85", alpha=0.92),
    )

    cax = ax.inset_axes([1.03, 0.02, 0.050, 0.96])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both")
    cb.set_label("Event-days", fontsize=23, labelpad=8)
    cb.ax.tick_params(labelsize=18)

    clean_spines(ax)
    add_panel_label(fig, ax, panel_letter, dx=PANEL_LABEL_DX_STD, dy=PANEL_LABEL_DY_STD)


# =============================================================================
# 11. Transition heatmaps
# =============================================================================

def draw_transition_heatmaps(
    fig,
    spec,
    annual: pd.DataFrame,
    scenario: str = "ssp585",
    panel_letter: str = "f",
):
    """Panel f: SSP5-8.5 transition-probability change matrices.

    The colour bar is vertical and attached to the right side of the two
    matrices, avoiding the previous overlap with the x-axis labels.
    """
    inner = GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=spec,
        width_ratios=[1.0, 1.0, 0.055],
        wspace=0.22,
    )

    ax1 = fig.add_subplot(inner[0, 0])
    ax2 = fig.add_subplot(inner[0, 1])
    cax = fig.add_subplot(inner[0, 2])

    front = transition_change_matrix(annual, scenario=scenario, kernel="front")
    local = transition_change_matrix(annual, scenario=scenario, kernel="local")

    mats = [front, local]
    vmax = 0.0
    for m in mats:
        mm = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 0.0
        vmax = max(vmax, mm)

    vmax = max(float(vmax), 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    im = None
    for ax, mat, name in zip([ax1, ax2], mats, ["Front", "Local"]):
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(STATE_LABELS, fontsize=18)
        ax.set_yticklabels(STATE_LABELS, fontsize=18)
        ax.set_xlabel("To state", fontsize=19, labelpad=2)
        if ax is ax1:
            ax.set_ylabel("From state", fontsize=19, labelpad=2)
        else:
            ax.set_ylabel("")

        for i in range(6):
            for j in range(6):
                val = mat[i, j]
                txt = "nan" if not np.isfinite(val) else f"{val:+.2f}"
                rgba = cmap(norm(val)) if np.isfinite(val) else (1, 1, 1, 1)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "white" if lum < 0.43 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=13.8, color=color)

        ax.text(
            0.5, 1.035, name,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=22,
        )
        for sp in ax.spines.values():
            sp.set_linewidth(0.85)

    if im is not None:
        cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both")
        cb.set_label("Transition probability change", fontsize=20, labelpad=8)
        cb.ax.tick_params(labelsize=16.5)

    add_group_panel_label(fig, [ax1, ax2, cax], panel_letter, dx=PANEL_LABEL_DX_HEAT, dy=PANEL_LABEL_DY_HEAT)


# =============================================================================
# 12. Standalone CMIP6 transition-flow panel f
# =============================================================================

def fmt_support(x: float) -> str:
    """Compact support-count formatting used in the flow-panel footer."""
    if not np.isfinite(x):
        return "NA"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1000:
        return f"{x / 1000:.0f}k"
    return str(int(round(x)))


def counts_to_probability_matrix(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 6 x 6 source-target count matrix to row-normalized probabilities."""
    counts = np.asarray(counts, dtype=float)
    support = counts.sum(axis=1)
    mat = np.full((6, 6), np.nan, dtype=float)

    for i in range(6):
        if np.isfinite(support[i]) and support[i] > 0:
            mat[i, :] = counts[i, :] / support[i]

    return mat, support


def transition_probability_matrix_and_support(
    annual: pd.DataFrame,
    scenario: str,
    period: Tuple[int, int],
    kernel: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build a model-median transition-probability matrix for one scenario, period
    and kernel.

    The transition-flow panel shows the actual transition architecture, not the
    future-minus-baseline change matrix. Counts are pooled within each model and
    selected period, row-normalized by source state, and then summarized as the
    multi-model median probability for each source-target link.
    """
    empty_prob = np.full((6, 6), np.nan, dtype=float)
    empty_support = np.full(6, np.nan, dtype=float)
    empty_models = pd.DataFrame(columns=["model", "from_state", "source_support"])

    if annual.empty:
        return empty_prob, empty_support, empty_models

    required = {"scenario", "year", "kernel", "from_state", "to_state", "count"}
    missing = sorted(required.difference(annual.columns))
    if missing:
        log(f"[WARN] Transition annual table lacks columns for flow panel: {missing}")
        return empty_prob, empty_support, empty_models

    d = annual.copy()
    if "model" not in d.columns:
        d["model"] = "ensemble"

    d["scenario"] = d["scenario"].astype(str).str.lower()
    d["kernel"] = d["kernel"].astype(str).str.lower().str.strip()
    d["year"] = safe_num(d["year"])
    d["from_state"] = d["from_state"].apply(parse_state)
    d["to_state"] = d["to_state"].apply(parse_state)
    d["count"] = safe_num(d["count"])

    d = d[
        d["scenario"].eq(str(scenario).lower())
        & d["kernel"].eq(str(kernel).lower())
        & d["year"].between(period[0], period[1])
        & d["from_state"].isin(STATE_ORDER)
        & d["to_state"].isin(STATE_ORDER)
        & d["count"].ge(0)
    ].copy()

    if d.empty:
        return empty_prob, empty_support, empty_models

    prob_stack = []
    support_stack = []
    support_rows = []

    for model, gm in d.groupby("model", observed=True):
        counts = np.zeros((6, 6), dtype=float)

        gg = gm.groupby(["from_state", "to_state"], as_index=False)["count"].sum()
        for _, row in gg.iterrows():
            i = int(row["from_state"]) - 1
            j = int(row["to_state"]) - 1
            if 0 <= i < 6 and 0 <= j < 6:
                counts[i, j] += float(row["count"])

        mat, support = counts_to_probability_matrix(counts)
        prob_stack.append(mat)
        support_stack.append(support)

        for i, fs in enumerate(STATE_ORDER):
            support_rows.append({
                "model": model,
                "from_state": fs,
                "source_support": float(support[i]),
            })

    if not prob_stack:
        return empty_prob, empty_support, empty_models

    prob_arr = np.stack(prob_stack, axis=0)
    support_arr = np.stack(support_stack, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mat_median = np.nanmedian(prob_arr, axis=0)
        support_median = np.nanmedian(support_arr, axis=0)

    support_df = pd.DataFrame(support_rows)
    return mat_median, support_median, support_df


def transition_breadth_by_source(mat: np.ndarray) -> np.ndarray:
    """Expected absolute state jump for each source row."""
    arr = np.asarray(mat, dtype=float)
    targets = np.array(STATE_ORDER, dtype=float)
    out = np.full(6, np.nan, dtype=float)

    for i, fs in enumerate(STATE_ORDER):
        row = arr[i, :]
        if np.isfinite(row).any():
            out[i] = np.nansum(row * np.abs(targets - fs))

    return out


def weighted_summary(mat: np.ndarray, support: np.ndarray) -> Dict[str, float]:
    """Weighted retention and breadth summary shown at the bottom of each flow panel."""
    arr = np.asarray(mat, dtype=float)
    support = np.asarray(support, dtype=float)

    total = np.nansum(support)
    if not np.isfinite(total) or total <= 0:
        return {
            "total_support": np.nan,
            "weighted_retention": np.nan,
            "weighted_breadth": np.nan,
        }

    weights = support / total
    retention = np.diag(arr).astype(float)
    breadth = transition_breadth_by_source(arr)

    return {
        "total_support": float(total),
        "weighted_retention": float(np.nansum(retention * weights)),
        "weighted_breadth": float(np.nansum(breadth * weights)),
    }


def collect_flow_probabilities(mats: List[np.ndarray], threshold: float = FLOW_MIN_PROB) -> np.ndarray:
    vals = []
    for mat in mats:
        arr = np.asarray(mat, dtype=float)
        for i in range(6):
            for j in range(6):
                p = arr[i, j]
                if np.isfinite(p) and p >= threshold:
                    vals.append(float(p))

    if len(vals) == 0:
        return np.array([threshold], dtype=float)
    return np.array(vals, dtype=float)


def draw_curve(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
    color: str,
    alpha: float,
    zorder: float,
) -> None:
    verts = [
        (x0, y0),
        (x0 + 0.28, y0),
        (x1 - 0.28, y1),
        (x1, y1),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    path = MplPath(verts, codes)
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        lw=width,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)


def flow_line_width(p: float, pmax: float) -> float:
    """Global nonlinear line-width scaling shared by local and front kernels."""
    if not np.isfinite(pmax) or pmax <= 0:
        pmax = 1.0
    r = max(p / pmax, 0.0)
    return 0.30 + 8.40 * (r ** 1.18)


def flow_line_alpha(p: float, pmax: float) -> float:
    if not np.isfinite(pmax) or pmax <= 0:
        pmax = 1.0
    r = max(p / pmax, 0.0)
    return min(0.90, 0.18 + 0.72 * (r ** 0.85))


def _select_major_transition_links(
    mat: np.ndarray,
    threshold: float,
    top_k_per_source: int = 2,
    retention_min: float = 0.06,
) -> List[Tuple[float, int, int, int, int]]:
    """Select a sparse, legible set of major transition links.

    The flow panel is a structural summary, not a full transition matrix. The
    full matrix is already shown in panel f. To prevent the alluvial panel from
    becoming unreadable, this selector keeps (i) links above a probability
    threshold, (ii) at most the strongest few outgoing links from each source
    state, and (iii) meaningful retention links on the diagonal.
    """
    arr = np.asarray(mat, dtype=float)
    links = []

    for i, fs in enumerate(STATE_ORDER):
        row = arr[i, :]
        finite = np.where(np.isfinite(row))[0]
        if finite.size == 0:
            continue

        keep = set()

        for j in finite:
            if row[j] >= threshold:
                keep.add(int(j))

        ranked = sorted(finite, key=lambda jj: row[jj], reverse=True)
        for j in ranked[:max(1, top_k_per_source)]:
            if row[j] >= max(0.035, threshold * 0.55):
                keep.add(int(j))

        if np.isfinite(row[i]) and row[i] >= retention_min:
            keep.add(int(i))

        for j in sorted(keep):
            ts = STATE_ORDER[j]
            p = float(row[j])
            if np.isfinite(p):
                links.append((p, fs, ts, i, j))

    return sorted(links, key=lambda x: x[0])


def plot_transition_flow_panel(
    ax,
    mat: np.ndarray,
    support: np.ndarray,
    title: str,
    subtitle: str,
    pmax_global: float,
    threshold: float = FLOW_MIN_PROB,
    top_k_per_source: int = 2,
) -> None:
    """Draw a compact source-to-target transition-flow architecture panel.

    In the embedded Figure 5 panel, only major pathways are shown. This is
    deliberate: panel f carries the complete 6 x 6 transition-change matrix,
    whereas panel h is meant to expose the dominant architecture without line
    clutter or label overlap.
    """
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.92, 5.92)

    mat = np.asarray(mat, dtype=float)
    support = np.asarray(support, dtype=float)

    if not np.isfinite(mat).any():
        ax.text(
            0.5, 0.5,
            "Missing transition data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=15.0,
            color="0.35",
        )
        return

    summary = weighted_summary(mat, support)

    x_left = 0.20
    x_right = 0.80
    y_pos = {s: 5 - idx for idx, s in enumerate(STATE_ORDER)}

    ax.text(
        0.0, 1.075,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=19.0,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        0.0, 1.018,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.4,
        color="0.35",
        clip_on=False,
    )

    ax.text(x_left, 5.58, "Source", ha="center", va="bottom", fontsize=13.7, color="0.25")
    ax.text(x_right, 5.58, "Target", ha="center", va="bottom", fontsize=13.7, color="0.25")

    ax.annotate(
        "",
        xy=(0.60, 5.535),
        xytext=(0.40, 5.535),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.15,
            color="0.45",
            mutation_scale=12.5,
        ),
    )

    links = _select_major_transition_links(
        mat=mat,
        threshold=threshold,
        top_k_per_source=top_k_per_source,
        retention_min=max(0.055, threshold * 0.70),
    )

    for rank, (p, fs, ts, i, j) in enumerate(links):
        color = STATE_COLORS[fs]
        width = flow_line_width(p, pmax_global)
        alpha = flow_line_alpha(p, pmax_global)
        source_offset = (ts - fs) * 0.010
        target_offset = (fs - ts) * 0.006

        z = 2.0 + rank * 0.01
        if fs == ts:
            z += 5.0
            alpha = min(0.92, alpha + 0.05)

        draw_curve(
            ax=ax,
            x0=x_left + 0.045,
            y0=y_pos[fs] + source_offset,
            x1=x_right - 0.045,
            y1=y_pos[ts] + target_offset,
            width=width,
            color=color,
            alpha=alpha,
            zorder=z,
        )

    for s in STATE_ORDER:
        y = y_pos[s]
        color = STATE_COLORS[s]
        for x in [x_left, x_right]:
            ax.scatter(
                [x], [y],
                s=410,
                color=color,
                edgecolor="white",
                linewidth=1.05,
                zorder=30,
            )
            text_color = "white" if s in [1, 2, 5, 6] else "black"
            ax.text(
                x, y,
                f"S{s}",
                ha="center",
                va="center",
                fontsize=12.7,
                fontweight="bold",
                color=text_color,
                zorder=31,
            )

    ax.text(
        0.50,
        -0.62,
        (
            f"retention = {summary['weighted_retention']:.2f}; "
            f"breadth = {summary['weighted_breadth']:.2f}; "
            f"N = {fmt_support(summary['total_support'])}"
        ),
        ha="center",
        va="center",
        fontsize=12.8,
        color="0.32",
    )


def transition_flow_edge_table(
    mat: np.ndarray,
    support: np.ndarray,
    kernel: str,
    scenario: str,
    period: Tuple[int, int],
    flow_min_prob: float,
) -> pd.DataFrame:
    rows = []
    for i, fs in enumerate(STATE_ORDER):
        for j, ts in enumerate(STATE_ORDER):
            p = float(mat[i, j]) if np.isfinite(mat[i, j]) else np.nan
            rows.append({
                "scenario": scenario,
                "period_start": period[0],
                "period_end": period[1],
                "kernel": kernel,
                "from_state": fs,
                "to_state": ts,
                "probability": p,
                "from_state_support_median": float(support[i]) if np.isfinite(support[i]) else np.nan,
                "shown_in_flow": bool(np.isfinite(p) and p >= flow_min_prob),
            })
    return pd.DataFrame(rows)


def save_flow_figure(fig, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")
    log(f"[SAVED] {svg}")


def build_cmip6_transition_flow_panel_f(
    out_dir: Path,
    annual: pd.DataFrame,
    scenario: str = FLOW_PANEL_SCENARIO,
    period: Tuple[int, int] = FLOW_PANEL_PERIOD,
    flow_min_prob: float = FLOW_MIN_PROB,
) -> None:
    """
    Additional CMIP6 panel-f-style flow architecture figure.

    This is intentionally standalone so the original Figure 5 output is retained
    unchanged. The panel shows end-century transition architecture under the
    selected scenario using the same visual logic as the baseline Figure 5f.
    """
    ensure_dir(out_dir)

    local_mat, local_support, local_support_models = transition_probability_matrix_and_support(
        annual=annual,
        scenario=scenario,
        period=period,
        kernel="local",
    )
    front_mat, front_support, front_support_models = transition_probability_matrix_and_support(
        annual=annual,
        scenario=scenario,
        period=period,
        kernel="front",
    )

    pvals = collect_flow_probabilities([local_mat, front_mat], threshold=flow_min_prob)
    pmax_global = float(np.nanmax(pvals)) if np.isfinite(pvals).any() else flow_min_prob
    pmax_global = max(pmax_global, flow_min_prob)

    edge_df = pd.concat(
        [
            transition_flow_edge_table(local_mat, local_support, "local", scenario, period, flow_min_prob),
            transition_flow_edge_table(front_mat, front_support, "front", scenario, period, flow_min_prob),
        ],
        ignore_index=True,
    )
    edge_df.to_csv(out_dir / f"{FLOW_FIG_STEM}_edge_table.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame([
        {"scenario": scenario, "period_start": period[0], "period_end": period[1], "kernel": "local", **weighted_summary(local_mat, local_support)},
        {"scenario": scenario, "period_start": period[0], "period_end": period[1], "kernel": "front", **weighted_summary(front_mat, front_support)},
    ])
    summary_df.to_csv(out_dir / f"{FLOW_FIG_STEM}_weighted_summary.csv", index=False, encoding="utf-8-sig")

    support_models = pd.concat(
        [
            local_support_models.assign(kernel="local", scenario=scenario, period_start=period[0], period_end=period[1]),
            front_support_models.assign(kernel="front", scenario=scenario, period_start=period[0], period_end=period[1]),
        ],
        ignore_index=True,
    )
    support_models.to_csv(out_dir / f"{FLOW_FIG_STEM}_model_source_support.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(13.6, 5.65))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        wspace=0.16,
        left=0.045,
        right=0.985,
        top=0.84,
        bottom=0.18,
    )

    fig.text(
        0.015,
        0.962,
        "f",
        ha="left",
        va="top",
        fontsize=24.0,
        fontweight="bold",
    )

    scenario_label = SC_LABEL.get(scenario, scenario)
    subtitle = f"{scenario_label}, {period[0]}–{period[1]}; major pathways, P ≥ {flow_min_prob:.2f}"

    ax_local = fig.add_subplot(gs[0, 0])
    ax_front = fig.add_subplot(gs[0, 1])

    plot_transition_flow_panel(
        ax=ax_local,
        mat=local_mat,
        support=local_support,
        title="Interior/local",
        subtitle=subtitle,
        pmax_global=pmax_global,
        threshold=flow_min_prob,
    )

    plot_transition_flow_panel(
        ax=ax_front,
        mat=front_mat,
        support=front_support,
        title="Advancing front",
        subtitle=subtitle,
        pmax_global=pmax_global,
        threshold=flow_min_prob,
    )

    legend_handles = [
        Line2D(
            [0], [0],
            color=STATE_COLORS[s],
            lw=5.0,
            solid_capstyle="round",
            label=f"Source S{s}",
        )
        for s in STATE_ORDER
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.50, 0.015),
        handlelength=1.55,
        columnspacing=1.15,
        fontsize=11.0,
    )

    save_flow_figure(fig, out_dir, FLOW_FIG_STEM)


# =============================================================================
# 12. Figure builders
# =============================================================================


# =============================================================================
# 12A. Scenario fingerprint and embedded flow panel for the revised main figure
# =============================================================================

def model_transition_matrices_for_period(
    annual: pd.DataFrame,
    scenario: str,
    period: Tuple[int, int],
    kernel: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Return per-model transition-probability matrices for a given scenario,
    period and kernel. Counts are pooled within the period and row-normalized
    by source state.
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if annual.empty:
        return out

    required = {"scenario", "year", "kernel", "from_state", "to_state", "count"}
    missing = sorted(required.difference(annual.columns))
    if missing:
        log(f"[WARN] Transition table lacks columns for scenario fingerprint: {missing}")
        return out

    d = annual.copy()
    if "model" not in d.columns:
        d["model"] = "ensemble"

    d["scenario"] = d["scenario"].astype(str).str.lower()
    d["kernel"] = d["kernel"].astype(str).str.lower().str.strip()
    d["year"] = safe_num(d["year"])
    d["from_state"] = d["from_state"].apply(parse_state)
    d["to_state"] = d["to_state"].apply(parse_state)
    d["count"] = safe_num(d["count"])

    d = d[
        d["scenario"].eq(str(scenario).lower())
        & d["kernel"].eq(str(kernel).lower())
        & d["year"].between(period[0], period[1])
        & d["from_state"].isin(STATE_ORDER)
        & d["to_state"].isin(STATE_ORDER)
        & d["count"].ge(0)
    ].copy()

    if d.empty:
        return out

    for model, gm in d.groupby("model", observed=True):
        counts = np.zeros((6, 6), dtype=float)
        gg = gm.groupby(["from_state", "to_state"], as_index=False)["count"].sum()
        for _, row in gg.iterrows():
            i = int(row["from_state"]) - 1
            j = int(row["to_state"]) - 1
            if 0 <= i < 6 and 0 <= j < 6:
                counts[i, j] += float(row["count"])
        mat, support = counts_to_probability_matrix(counts)
        if np.isfinite(mat).any():
            out[str(model)] = (mat, support)

    return out


def build_transition_change_fingerprint_table(
    annual: pd.DataFrame,
    baseline: Tuple[int, int] = BASELINE,
    future: Tuple[int, int] = ENDCENTURY,
) -> pd.DataFrame:
    """
    Ensemble summary for the new panel g.

    For each SSP and kernel, the table reports the multi-model median change
    from the historical baseline to the end-century period in two compact
    architecture metrics:
      1) weighted source-state retention probability;
      2) weighted transition breadth, defined as expected absolute state jump.

    The sign_agreement column gives the fraction of models with the same sign
    as the multi-model median and is shown as a dot in panel g when >= 0.75.
    """
    rows = []

    for sc in SCENARIOS:
        for kernel in ["front", "local"]:
            base = model_transition_matrices_for_period(
                annual=annual,
                scenario="historical",
                period=baseline,
                kernel=kernel,
            )
            fut = model_transition_matrices_for_period(
                annual=annual,
                scenario=sc,
                period=future,
                kernel=kernel,
            )

            common_models = sorted(set(base).intersection(fut))
            model_rows = []

            for model in common_models:
                base_mat, base_support = base[model]
                fut_mat, fut_support = fut[model]

                base_summary = weighted_summary(base_mat, base_support)
                fut_summary = weighted_summary(fut_mat, fut_support)

                model_rows.append({
                    "scenario": sc,
                    "scenario_label": SC_LABEL.get(sc, sc),
                    "kernel": kernel,
                    "model": model,
                    "delta_retention": fut_summary["weighted_retention"] - base_summary["weighted_retention"],
                    "delta_breadth": fut_summary["weighted_breadth"] - base_summary["weighted_breadth"],
                    "baseline_retention": base_summary["weighted_retention"],
                    "future_retention": fut_summary["weighted_retention"],
                    "baseline_breadth": base_summary["weighted_breadth"],
                    "future_breadth": fut_summary["weighted_breadth"],
                    "baseline_support": base_summary["total_support"],
                    "future_support": fut_summary["total_support"],
                })

            md = pd.DataFrame(model_rows)
            if md.empty:
                for metric in ["delta_retention", "delta_breadth"]:
                    rows.append({
                        "scenario": sc,
                        "scenario_label": SC_LABEL.get(sc, sc),
                        "kernel": kernel,
                        "metric": metric,
                        "median": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                        "n_model": 0,
                        "sign_agreement": np.nan,
                    })
                continue

            for metric in ["delta_retention", "delta_breadth"]:
                vals = pd.to_numeric(md[metric], errors="coerce").dropna().to_numpy(float)
                if vals.size == 0:
                    med = q25 = q75 = agree = np.nan
                    n_model = 0
                else:
                    med = float(np.nanmedian(vals))
                    q25 = float(np.nanpercentile(vals, 25))
                    q75 = float(np.nanpercentile(vals, 75))
                    n_model = int(np.isfinite(vals).sum())
                    if med >= 0:
                        agree = float(np.mean(vals >= 0))
                    else:
                        agree = float(np.mean(vals < 0))

                rows.append({
                    "scenario": sc,
                    "scenario_label": SC_LABEL.get(sc, sc),
                    "kernel": kernel,
                    "metric": metric,
                    "median": med,
                    "q25": q25,
                    "q75": q75,
                    "n_model": n_model,
                    "sign_agreement": agree,
                })

            # Keep the model-level values in a side table for inspection.
            if not md.empty:
                rows.extend([])

    return pd.DataFrame(rows)


def _annotate_scenario_fingerprint(
    ax,
    mat: np.ndarray,
    agree: np.ndarray,
    fmt: str,
    cmap,
    norm,
    dot_threshold: float = AGREE_THR,
) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=11.0, color="0.35")
                continue

            rgba = cmap(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "white" if luminance < 0.43 else "black"
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=11.0, color=txt_color)

            if np.isfinite(agree[i, j]) and agree[i, j] >= dot_threshold:
                ax.scatter(
                    j + 0.36,
                    i - 0.34,
                    s=18,
                    color="black",
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=5,
                )


def _matrix_from_transition_subset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((6, 6), dtype=float)
    if df.empty:
        return counts_to_probability_matrix(counts)
    gg = df.groupby(["from_state", "to_state"], as_index=False)["count"].sum()
    for _, row in gg.iterrows():
        i = int(row["from_state"]) - 1
        j = int(row["to_state"]) - 1
        if 0 <= i < 6 and 0 <= j < 6:
            counts[i, j] += float(row["count"])
    return counts_to_probability_matrix(counts)


def _net_transition_tendency(mat: np.ndarray) -> np.ndarray:
    """Net wetward/dryward tendency by source state.

    Positive values mean that transitions from a source state move, on average,
    toward wetter states. Negative values mean transitions move toward drier
    states. Values are measured in expected state-jump units.
    """
    arr = np.asarray(mat, dtype=float)
    targets = np.array(STATE_ORDER, dtype=float)
    out = np.full(6, np.nan, dtype=float)
    for i, fs in enumerate(STATE_ORDER):
        row = arr[i, :]
        if np.isfinite(row).any():
            out[i] = np.nansum(row * (targets - fs))
    return out


def build_transition_profile_change_table(
    annual: pd.DataFrame,
    baseline: Tuple[int, int] = BASELINE,
    future: Tuple[int, int] = ENDCENTURY,
) -> pd.DataFrame:
    """Build model-level end-century transition-profile changes for panel g.

    The panel is not a time-series plot. For each model, kernel and scenario,
    it compares the end-century transition matrix against the model-specific
    historical baseline. Two source-state profiles are retained:

    1. retention_change:      ΔP(Si -> Si)
    2. net_tendency_change:   ΔΣj P(Si -> Sj) * (j - i)

    The second metric is a signed transition tendency: positive values indicate
    a shift toward wetter target states, and negative values indicate a shift
    toward drier target states.
    """
    cols = [
        "scenario", "scenario_label", "kernel", "model", "from_state",
        "metric", "value", "baseline_value", "future_value",
    ]
    if annual.empty:
        return pd.DataFrame(columns=cols)

    required = {"scenario", "year", "kernel", "from_state", "to_state", "count"}
    missing = sorted(required.difference(annual.columns))
    if missing:
        log(f"[WARN] Transition table lacks columns for panel g transition-profile changes: {missing}")
        return pd.DataFrame(columns=cols)

    d = annual.copy()
    if "model" not in d.columns:
        d["model"] = "ensemble"

    d["scenario"] = d["scenario"].astype(str).str.lower()
    d["kernel"] = d["kernel"].astype(str).str.lower().str.strip()
    d["model"] = d["model"].astype(str)
    d["year"] = safe_num(d["year"])
    d["from_state"] = d["from_state"].apply(parse_state)
    d["to_state"] = d["to_state"].apply(parse_state)
    d["count"] = safe_num(d["count"])
    d = d[
        d["kernel"].isin(["front", "local"])
        & d["from_state"].isin(STATE_ORDER)
        & d["to_state"].isin(STATE_ORDER)
        & d["year"].between(HIST[0], FUT[1])
        & d["count"].ge(0)
    ].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for kernel in ["front", "local"]:
        base_all = d[
            d["scenario"].eq("historical")
            & d["kernel"].eq(kernel)
            & d["year"].between(baseline[0], baseline[1])
        ].copy()
        if base_all.empty:
            continue

        for sc in SCENARIOS:
            fut_all = d[
                d["scenario"].eq(sc)
                & d["kernel"].eq(kernel)
                & d["year"].between(future[0], future[1])
            ].copy()
            if fut_all.empty:
                continue

            models = sorted(set(base_all["model"]).intersection(set(fut_all["model"])))
            for model in models:
                bmat, _ = _matrix_from_transition_subset(base_all[base_all["model"].eq(model)])
                fmat, _ = _matrix_from_transition_subset(fut_all[fut_all["model"].eq(model)])

                b_ret = np.diag(bmat).astype(float)
                f_ret = np.diag(fmat).astype(float)
                b_net = _net_transition_tendency(bmat)
                f_net = _net_transition_tendency(fmat)

                for i, fs in enumerate(STATE_ORDER):
                    metric_values = {
                        "retention_change": (b_ret[i], f_ret[i], f_ret[i] - b_ret[i]),
                        "net_tendency_change": (b_net[i], f_net[i], f_net[i] - b_net[i]),
                    }
                    for metric, (bval, fval, diff) in metric_values.items():
                        rows.append({
                            "scenario": sc,
                            "scenario_label": SC_LABEL.get(sc, sc),
                            "kernel": kernel,
                            "model": model,
                            "from_state": fs,
                            "metric": metric,
                            "value": float(diff) if np.isfinite(diff) else np.nan,
                            "baseline_value": float(bval) if np.isfinite(bval) else np.nan,
                            "future_value": float(fval) if np.isfinite(fval) else np.nan,
                        })

    return pd.DataFrame(rows, columns=cols)


def summarize_transition_profile_change(profile: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scenario", "scenario_label", "kernel", "from_state", "metric",
        "median", "q25", "q75", "n_model", "sign_agreement",
    ]
    if profile.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for keys, g in profile.groupby(["scenario", "scenario_label", "kernel", "from_state", "metric"], observed=True):
        sc, sc_label, kernel, fs, metric = keys
        vals = pd.to_numeric(g["value"], errors="coerce").dropna().to_numpy(float)
        if vals.size == 0:
            med = q25 = q75 = agree = np.nan
            n_model = 0
        else:
            med = float(np.nanmedian(vals))
            q25 = float(np.nanpercentile(vals, 25))
            q75 = float(np.nanpercentile(vals, 75))
            n_model = int(np.isfinite(vals).sum())
            agree = float(np.mean(vals >= 0)) if med >= 0 else float(np.mean(vals < 0))
        rows.append({
            "scenario": sc,
            "scenario_label": sc_label,
            "kernel": kernel,
            "from_state": int(fs),
            "metric": metric,
            "median": med,
            "q25": q25,
            "q75": q75,
            "n_model": n_model,
            "sign_agreement": agree,
        })
    return pd.DataFrame(rows, columns=cols)


def draw_scenario_transition_fingerprint(
    fig,
    spec,
    annual: pd.DataFrame,
    panel_letter: str = "g",
) -> None:
    """Panel g: source-state transition-profile changes, not time series.

    The previous version repeated a state-by-year heatmap. This version shows
    the end-century transition trend across S1-S6 source states: retention
    change and net wetward/dryward transition tendency change, separately for
    front and local kernels and for all three SSP scenarios.
    """
    ensure_dir(OUT_DIR)
    profile = build_transition_profile_change_table(annual)
    summary = summarize_transition_profile_change(profile)
    profile.to_csv(
        OUT_DIR / "Figure5g_CMIP6_model_level_transition_profile_changes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUT_DIR / "Figure5g_CMIP6_MME_transition_profile_changes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    inner = GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=spec,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.44,
        wspace=0.32,
    )

    axes = []
    metric_rows = [
        ("retention_change", "Δ retention probability"),
        ("net_tendency_change", "Δ net wetward tendency"),
    ]
    kernels = [("front", "Front"), ("local", "Local")]
    x = np.array(STATE_ORDER, dtype=float)

    for r, (metric, ylabel) in enumerate(metric_rows):
        for c, (kernel, kernel_label) in enumerate(kernels):
            ax = fig.add_subplot(inner[r, c])
            axes.append(ax)

            for sc in SCENARIOS:
                dd = summary[
                    summary["scenario"].eq(sc)
                    & summary["kernel"].eq(kernel)
                    & summary["metric"].eq(metric)
                ].copy()
                dd = dd.set_index("from_state").reindex(STATE_ORDER)

                y = pd.to_numeric(dd["median"], errors="coerce").to_numpy(float)
                q25 = pd.to_numeric(dd["q25"], errors="coerce").to_numpy(float)
                q75 = pd.to_numeric(dd["q75"], errors="coerce").to_numpy(float)
                agree = pd.to_numeric(dd["sign_agreement"], errors="coerce").to_numpy(float)

                ax.fill_between(
                    x,
                    q25,
                    q75,
                    color=SC_COLOR[sc],
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    x,
                    y,
                    color=SC_COLOR[sc],
                    lw=2.45,
                    marker="o",
                    markersize=4.6,
                    markerfacecolor="white",
                    markeredgewidth=1.15,
                    label=SC_LABEL[sc],
                    zorder=3,
                )

                mask = np.isfinite(y) & np.isfinite(agree) & (agree >= AGREE_THR)
                if mask.any():
                    ax.scatter(
                        x[mask],
                        y[mask],
                        s=18,
                        color=SC_COLOR[sc],
                        edgecolor="black",
                        linewidth=0.45,
                        zorder=4,
                    )

            ax.axhline(0.0, color="0.35", lw=0.9, ls="--", zorder=0)
            ax.set_xlim(0.72, 6.28)
            ax.set_xticks(STATE_ORDER)
            ax.set_xticklabels(STATE_LABELS, fontsize=13.6)
            ax.tick_params(axis="y", labelsize=13.2)
            ax.grid(axis="y", color="0.88", lw=0.75)
            clean_spines(ax)

            if r == 0:
                ax.set_title(kernel_label, fontsize=16.8, pad=5)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Source state", fontsize=15.0, labelpad=2)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=15.0, labelpad=5)
            else:
                ax.set_ylabel("")

    axes[0].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.03, 1.23),
        ncol=3,
        fontsize=12.6,
        handlelength=1.9,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    axes[0].text(
        0.0,
        1.075,
        f"End-century ({ENDCENTURY[0]}–{ENDCENTURY[1]}) minus {BASELINE[0]}–{BASELINE[1]}; black-edged points: model agreement ≥ {AGREE_THR:.2f}",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11.6,
        color="0.35",
        clip_on=False,
    )

    add_group_panel_label(fig, axes, panel_letter, dx=PANEL_LABEL_DX_STD, dy=PANEL_LABEL_DY_STD)

def draw_embedded_transition_flow_panel(
    fig,
    spec,
    annual: pd.DataFrame,
    scenario: str = FLOW_PANEL_SCENARIO,
    period: Tuple[int, int] = FLOW_PANEL_PERIOD,
    panel_letter: str = "h",
    flow_min_prob: float = FLOW_MIN_PROB,
) -> None:
    """Embed the CMIP6 transition-flow architecture as panel h in the main figure.

    The embedded version deliberately removes the bottom legend and keeps only
    major pathways. This prevents clipping/overlap and leaves the flow diagram
    as a structural companion to the quantitative transition profiles in panel g.
    """
    flow_min_prob = max(float(flow_min_prob), 0.08)

    local_mat, local_support, _ = transition_probability_matrix_and_support(
        annual=annual,
        scenario=scenario,
        period=period,
        kernel="local",
    )
    front_mat, front_support, _ = transition_probability_matrix_and_support(
        annual=annual,
        scenario=scenario,
        period=period,
        kernel="front",
    )

    pvals = collect_flow_probabilities([local_mat, front_mat], threshold=flow_min_prob)
    pmax_global = float(np.nanmax(pvals)) if np.isfinite(pvals).any() else flow_min_prob
    pmax_global = max(pmax_global, flow_min_prob)

    inner = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=spec,
        width_ratios=[1.0, 1.0],
        wspace=0.16,
    )

    ax_local = fig.add_subplot(inner[0, 0])
    ax_front = fig.add_subplot(inner[0, 1])

    scenario_label = SC_LABEL.get(scenario, scenario)
    subtitle = f"{scenario_label}, {period[0]}–{period[1]}; major pathways only, P ≥ {flow_min_prob:.2f}"

    plot_transition_flow_panel(
        ax=ax_local,
        mat=local_mat,
        support=local_support,
        title="Interior/local",
        subtitle=subtitle,
        pmax_global=pmax_global,
        threshold=flow_min_prob,
        top_k_per_source=2,
    )
    plot_transition_flow_panel(
        ax=ax_front,
        mat=front_mat,
        support=front_support,
        title="Advancing front",
        subtitle=subtitle,
        pmax_global=pmax_global,
        threshold=flow_min_prob,
        top_k_per_source=2,
    )

    add_group_panel_label(fig, [ax_local, ax_front], panel_letter, dx=PANEL_LABEL_DX_STD, dy=PANEL_LABEL_DY_STD)


def build_figure5(
    out_dir: Path,
    annual_cache: pd.DataFrame,
    event_col: str,
    prob_col: str,
    spatial: Dict[str, pd.DataFrame],
    transition_tables: Dict[str, pd.DataFrame],
):
    """
    Revised main Figure 5.

    Layout changes requested in this version:
      - keep panels a-c unchanged;
      - exchange the original panels d and e;
      - keep the SSP5-8.5 transition-change matrices as panel f;
      - add panel g as a three-SSP source-state transition-profile summary;
      - embed the CMIP6 transition-flow architecture as panel h.
    """
    dry_days, _, drywet_days = build_dry_wet_series(annual_cache, event_col)
    dry_prob, _, _ = build_dry_wet_series(annual_cache, prob_col)

    fig = plt.figure(figsize=(23.6, 25.2))

    gs = GridSpec(
        4, 2,
        figure=fig,
        width_ratios=[1.0, 1.03],
        height_ratios=[1.00, 1.05, 1.18, 1.72],
        hspace=0.53,
        wspace=0.32,
        left=0.085,
        right=0.985,
        top=0.982,
        bottom=0.060,
    )

    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axe = fig.add_subplot(gs[1, 1])

    plot_hist_future_series(
        axa,
        dry_days.rename(columns={"value": "dry_days"}),
        "dry_days",
        ylabel="S1–S2 event-days",
        show_legend=True,
        legend_loc="upper left",
    )
    add_panel_label(fig, axa, "a", dx=PANEL_LABEL_DX_STD, dy=PANEL_LABEL_DY_STD)

    plot_hist_future_series(
        axb,
        dry_prob.rename(columns={"value": "dry_prob"}),
        "dry_prob",
        ylabel="S1–S2 probability",
        show_legend=False,
    )
    add_panel_label(fig, axb, "b", dx=PANEL_LABEL_DX_STD, dy=PANEL_LABEL_DY_STD)

    plot_hist_future_series(
        axc,
        drywet_days.rename(columns={"contrast": "contrast"}),
        "contrast",
        ylabel="Dry–wet contrast",
        show_legend=False,
    )
    add_panel_label(fig, axc, "c", dx=PANEL_LABEL_DX_STD, dy=PANEL_LABEL_DY_STD)

    # d/e exchange requested by the user: exchange the actual plots, not
    # only the letters. Panel d is now the S1-S6 state-year surface; panel e
    # is the front/local S1-retention time series.
    draw_state_year_heatmap(
        fig=fig,
        spec=gs[2, 0],
        annual_cache=annual_cache,
        metric_col=event_col,
        scenario="ssp585",
        panel_letter="d",
    )

    plot_persistence_series(axe, transition_tables["yearly"])
    add_panel_label(fig, axe, "e", dx=PANEL_LABEL_DX_STD, dy=PANEL_LABEL_DY_STD)

    draw_transition_heatmaps(
        fig=fig,
        spec=gs[2, 1],
        annual=transition_tables["annual"],
        scenario="ssp585",
        panel_letter="f",
    )

    draw_scenario_transition_fingerprint(
        fig=fig,
        spec=gs[3, 0],
        annual=transition_tables["annual"],
        panel_letter="g",
    )

    draw_embedded_transition_flow_panel(
        fig=fig,
        spec=gs[3, 1],
        annual=transition_tables["annual"],
        scenario=FLOW_PANEL_SCENARIO,
        period=FLOW_PANEL_PERIOD,
        panel_letter="h",
        flow_min_prob=FLOW_MIN_PROB,
    )

    savefig(fig, out_dir, "Figure5_CMIP6_continuous_spatial_rebuilt_v11")

def build_supplementary_spatial_support(
    out_dir: Path,
    spatial: Dict[str, pd.DataFrame],
    transition_tables: Dict[str, pd.DataFrame],
):
    fig = plt.figure(figsize=(21.8, 13.4))

    gs = GridSpec(
        3, 3,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 0.10, 1.0],
        hspace=0.26,
        wspace=0.30,
        left=0.07,
        right=0.985,
        top=0.975,
        bottom=0.08,
    )

    vals = []
    for sc in SCENARIOS:
        if "value" in spatial[sc].columns:
            vals.extend(list(pd.to_numeric(spatial[sc]["value"], errors="coerce").dropna().values))

    vmax = np.nanpercentile(np.abs(vals), 98) if len(vals) else 3.0
    vmax = max(vmax, 1.0)

    resp_norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    resp_cmap = plt.get_cmap("RdBu_r")

    map_axes = []
    last_im = None
    for j, sc in enumerate(SCENARIOS):
        ax_map, _, im = draw_spatial_panel(
            fig=fig,
            spec=gs[0, j],
            spatial_df=spatial[sc],
            value_col="value",
            cmap=resp_cmap,
            norm=resp_norm,
            cbar_label="Dry–wet response (event-days)",
            panel_letter=chr(ord("a") + j),
            show_stipple=True,
            scenario_tag=SC_LABEL[sc],
            add_colorbar=False,
        )
        map_axes.append(ax_map)
        if im is not None:
            last_im = im

    cax_holder = fig.add_subplot(gs[1, :])
    cax_holder.axis("off")
    if last_im is not None:
        cax = cax_holder.inset_axes([0.17, 0.22, 0.66, 0.58])
        cb = fig.colorbar(last_im, cax=cax, orientation="horizontal", extend="both")
        cb.set_label("Dry–wet response (event-days)", fontsize=23, labelpad=3)
        cb.ax.tick_params(labelsize=18)

    axd = fig.add_subplot(gs[2, 0])
    plot_transition_support_series(axd, transition_tables["annual"])
    add_panel_label(fig, axd, "d", dx=PANEL_LABEL_DX_STD, dy=PANEL_LABEL_DY_STD)

    axe = fig.add_subplot(gs[2, 1])
    plot_start_state_support(axe, transition_tables["start"])
    add_panel_label(fig, axe, "e", dx=PANEL_LABEL_DX_STD, dy=PANEL_LABEL_DY_STD)

    axf = fig.add_subplot(gs[2, 2])
    plot_support_ratio(axf, transition_tables["annual"])
    add_panel_label(fig, axf, "f", dx=PANEL_LABEL_DX_STD, dy=PANEL_LABEL_DY_STD)

    savefig(fig, out_dir, "Supplementary_Fig_CMIP6_spatial_uncertainty_support_rebuilt_v9")


# =============================================================================
# 13. Main
# =============================================================================

def main():
    log("=" * 90)
    log("[INFO] Building CMIP6 editorial projection figure package (v11)")
    log(f"[INFO] CMIP6_ROOT : {CMIP6_ROOT}")
    log(f"[INFO] TS_DIR     : {TS_DIR}")
    log(f"[INFO] SPMAP_DIR  : {SPMAP_DIR}")
    log(f"[INFO] OUT_DIR    : {OUT_DIR}")
    log("=" * 90)

    ensure_dir(OUT_DIR)

    annual_cache = load_annual_state_cache(TS_DIR)
    log(f"[INFO] Annual cache loaded: {annual_cache.shape}")

    event_col, prob_col = infer_metric_columns(annual_cache)
    log(f"[INFO] Event-days column : {event_col}")
    log(f"[INFO] Probability column: {prob_col}")

    spatial = {}
    for sc in SCENARIOS:
        spatial[sc] = load_spatial_for_scenario(
            spmap_dir=SPMAP_DIR,
            scenario=sc,
            preferred_metric=event_col,
        )
        mode = spatial[sc]["spatial_mode"].iloc[0] if "spatial_mode" in spatial[sc].columns else "unknown"
        log(f"[INFO] Spatial {sc}: {spatial[sc].shape}, mode={mode}")

    transition_tables = load_transition_tables(CMIP6_ROOT)

    for k, v in transition_tables.items():
        log(f"[INFO] Transition table [{k}] shape: {v.shape}")

    build_figure5(
        out_dir=OUT_DIR,
        annual_cache=annual_cache,
        event_col=event_col,
        prob_col=prob_col,
        spatial=spatial,
        transition_tables=transition_tables,
    )

    # Additional standalone panel-h-style CMIP6 transition-flow architecture.
    # The main Figure 5 now also embeds this architecture as panel h.
    build_cmip6_transition_flow_panel_f(
        out_dir=OUT_DIR,
        annual=transition_tables["annual"],
        scenario=FLOW_PANEL_SCENARIO,
        period=FLOW_PANEL_PERIOD,
        flow_min_prob=FLOW_MIN_PROB,
    )

    build_supplementary_spatial_support(
        out_dir=OUT_DIR,
        spatial=spatial,
        transition_tables=transition_tables,
    )

    log("[DONE] All figures finished.")


if __name__ == "__main__":
    main()