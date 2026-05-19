# -*- coding: utf-8 -*-
"""
CMIP6 projection figure package | editorial v7
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
Figure5_CMIP6_continuous_spatial_rebuilt_v9.png/pdf
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

OUT_DIR = CMIP6_ROOT / "_cmip6_projection_editorial_v9"


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

PANEL_LABEL_DX_STD = 0.010
PANEL_LABEL_DY_STD = 0.004
PANEL_LABEL_DX_MAP = 0.010
PANEL_LABEL_DY_MAP = 0.004
PANEL_LABEL_DX_HEAT = 0.010
PANEL_LABEL_DY_HEAT = 0.004


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
    inner = GridSpecFromSubplotSpec(
        3, 2,
        subplot_spec=spec,
        height_ratios=[1.0, 0.08, 0.08],
        hspace=0.0,
        wspace=0.24,
    )

    ax1 = fig.add_subplot(inner[0, 0])
    ax2 = fig.add_subplot(inner[0, 1])
    spacer = fig.add_subplot(inner[1, :])
    spacer.axis("off")
    cax = fig.add_subplot(inner[2, :])

    front = transition_change_matrix(annual, scenario=scenario, kernel="front")
    local = transition_change_matrix(annual, scenario=scenario, kernel="local")

    mats = [front, local]
    vmax = 0.0
    for m in mats:
        mm = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 0.0
        vmax = max(vmax, mm)

    vmax = max(vmax, 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    for ax, mat, name in zip([ax1, ax2], mats, ["Front", "Local"]):
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(STATE_LABELS, fontsize=19)
        ax.set_yticklabels(STATE_LABELS, fontsize=19)
        ax.set_xlabel("To state", fontsize=19)
        if ax is ax1:
            ax.set_ylabel("From state", fontsize=19)

        for i in range(6):
            for j in range(6):
                val = mat[i, j]
                txt = "nan" if not np.isfinite(val) else f"{val:+.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=14.2)

        ax.text(
            0.5, 1.03, name,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=22,
        )

    add_panel_label(fig, ax1, panel_letter, dx=PANEL_LABEL_DX_HEAT, dy=PANEL_LABEL_DY_HEAT)

    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Transition probability change", fontsize=23, labelpad=4)
    cb.ax.tick_params(labelsize=17.5)


# =============================================================================
# 12. Figure builders
# =============================================================================

def build_figure5(
    out_dir: Path,
    annual_cache: pd.DataFrame,
    event_col: str,
    prob_col: str,
    spatial: Dict[str, pd.DataFrame],
    transition_tables: Dict[str, pd.DataFrame],
):
    dry_days, _, drywet_days = build_dry_wet_series(annual_cache, event_col)
    dry_prob, _, _ = build_dry_wet_series(annual_cache, prob_col)

    fig = plt.figure(figsize=(19.2, 15.2))

    gs = GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[1.0, 1.02],
        height_ratios=[1.0, 1.04, 1.18],
        hspace=0.48,
        wspace=0.34,
        left=0.10,
        right=0.98,
        top=0.975,
        bottom=0.08,
    )

    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axe = fig.add_subplot(gs[2, 0])

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

    draw_state_year_heatmap(
        fig=fig,
        spec=gs[1, 1],
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

    savefig(fig, out_dir, "Figure5_CMIP6_continuous_spatial_rebuilt_v9")


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
    log("[INFO] Building CMIP6 editorial projection figure package (v9)")
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

    build_supplementary_spatial_support(
        out_dir=OUT_DIR,
        spatial=spatial,
        transition_tables=transition_tables,
    )

    log("[DONE] All figures finished.")


if __name__ == "__main__":
    main()