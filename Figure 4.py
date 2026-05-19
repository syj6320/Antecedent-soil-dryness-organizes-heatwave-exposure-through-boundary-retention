# -*- coding: utf-8 -*-
"""
Figure 3 | Surface-energy partitioning anchors the dry-state transition mechanism
Final clean version

This script reads only two precomputed CSV files:
1) Figure3_event_level_surface_energy_anomaly_summary.csv
2) Figure3_spatial_surface_energy_anomaly_dry_wet_contrast.csv

Final layout:
Row 1:
    a  Bowen anomaly contrast
    b  Evaporative suppression contrast
    c  Sensible-heat anomaly contrast
Row 2:
    d  State gradients
       - bottom n=... labels removed
    e  Regime-space zoom
       - background grey event points darkened
    f  Dry-vs-wet front-local response summary

Outputs:
    Figure3_surface_energy_partitioning_relayout_v5_fixed_panel_d_e.png
    Figure3_surface_energy_partitioning_relayout_v5_fixed_panel_d_e.pdf
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception as e:
    raise ImportError("需要 cartopy 才能绘制空间图。请先安装 cartopy。") from e


# =============================================================================
# 1. User paths
# =============================================================================
FIG3_ROOT = Path(r"E:\第二篇数据0427\Figure3_surface_energy_partitioning_outputs")

EVENT_CSV = FIG3_ROOT / "Figure3_event_level_surface_energy_anomaly_summary.csv"
SPATIAL_CSV = FIG3_ROOT / "Figure3_spatial_surface_energy_anomaly_dry_wet_contrast.csv"

OUT_PNG = FIG3_ROOT / "Figure3_surface_energy_partitioning_relayout_v5_fixed_panel_d_e.png"
OUT_PDF = FIG3_ROOT / "Figure3_surface_energy_partitioning_relayout_v5_fixed_panel_d_e.pdf"


# =============================================================================
# 2. Figure settings
# =============================================================================
CONUS_EXTENT = [-125, -66, 24, 50]

SPATIAL_MIN_N_DRY = 10
SPATIAL_MIN_N_WET = 10
MAP_SMOOTH_SIGMA = 0.8

FONT_FAMILY = "Arial"
TITLE_FS = 24
LABEL_FS = 20
TICK_FS = 16
LEGEND_FS = 15
LETTER_FS = 26
ANNOT_FS = 14

N_BOOT = 2000
RANDOM_SEED = 42

STATE_COLORS = {
    "S1": "#8c510a",
    "S2": "#bf812d",
    "S3": "#dfc27d",
    "S4": "#80cdc1",
    "S5": "#35978f",
    "S6": "#01665e",
}

VAR_COLORS = {
    "Bowen'": "#8c510a",
    "−EF'": "#1b9e77",
    "H'": "#d95f02",
}

DRY_COLOR = "#b2182b"
WET_COLOR = "#2166ac"

# Panel e background points: these control the small grey background dots.
E_BG_COLOR = "#9a9a9a"
E_BG_ALPHA = 0.22
E_BG_SIZE = 8


# =============================================================================
# 3. Utility functions
# =============================================================================
def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df: pd.DataFrame, candidates, required=True, desc="column"):
    if isinstance(candidates, str):
        candidates = [candidates]

    col_map = {norm_name(c): c for c in df.columns}

    for cand in candidates:
        key = norm_name(cand)
        if key in col_map:
            return col_map[key]

    for cand in candidates:
        key = norm_name(cand)
        for k, v in col_map.items():
            if key in k or k in key:
                return v

    if required:
        raise KeyError(
            f"无法在表中找到 {desc}.\n"
            f"候选列名: {candidates}\n"
            f"当前表列名: {list(df.columns)}"
        )
    return None


def safe_zscore(x):
    x = pd.Series(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros(len(x))
    return (x - mu) / sd


def robust_sym_lim(values, q=0.98, floor=None):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        vmax = 1.0
    else:
        vmax = np.nanquantile(np.abs(arr), q)
    if floor is not None:
        vmax = max(vmax, floor)
    if vmax == 0:
        vmax = 1.0
    return float(vmax)


def robust_xlim(values, qlo=0.01, qhi=0.99, pad_frac=0.12):
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(arr) < 5:
        return -1.0, 1.0

    lo, hi = np.nanquantile(arr, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(arr), np.nanmax(arr)

    pad = (hi - lo) * pad_frac if hi > lo else 1.0
    return float(lo - pad), float(hi + pad)


def compute_edges(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.size < 2:
        raise ValueError("坐标数量太少，无法推算边界。")

    d = np.diff(coords) / 2.0
    edges = np.empty(coords.size + 1, dtype=float)
    edges[1:-1] = coords[:-1] + d
    edges[0] = coords[0] - d[0]
    edges[-1] = coords[-1] + d[-1]
    return edges


def smooth_2d(arr, sigma=0.8):
    if gaussian_filter is None or sigma <= 0:
        return arr

    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr).astype(float)
    data = np.where(np.isfinite(arr), arr, 0.0)

    num = gaussian_filter(data * valid, sigma=sigma, mode="nearest")
    den = gaussian_filter(valid, sigma=sigma, mode="nearest")
    out = num / np.where(den == 0, np.nan, den)
    out[den < 0.05] = np.nan
    return out


def add_panel_letter(ax, letter, x=-0.18, y=1.12):
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=LETTER_FS,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def set_mpl_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": TICK_FS,
        "axes.titlesize": TITLE_FS,
        "axes.labelsize": LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "axes.grid": True,
    })


# =============================================================================
# 4. Load and standardize data
# =============================================================================
def load_event_summary(event_csv: Path) -> pd.DataFrame:
    if not event_csv.exists():
        raise FileNotFoundError(f"找不到事件级 CSV：{event_csv}")

    df = pd.read_csv(event_csv)

    state_col = find_col(
        df,
        [
            "state_label", "initial_state", "sm_state", "soil_moisture_state",
            "s_bin_label", "sbin_label", "S_bin", "Sbin", "cc_label", "state"
        ],
        desc="state column"
    )

    def parse_state(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        m = re.search(r"([1-6])", s)
        if m:
            return int(m.group(1))
        try:
            num = int(float(s))
            if 1 <= num <= 6:
                return num
        except Exception:
            pass
        return np.nan

    df["state_num"] = df[state_col].apply(parse_state)
    df["state_lab"] = df["state_num"].apply(
        lambda x: f"S{int(x)}" if pd.notna(x) and 1 <= int(x) <= 6 else np.nan
    )

    bowen_col = find_col(
        df,
        [
            "event_mean_bowen_anom", "bowen_anom_event_mean",
            "bowen_anomaly_event_mean", "event_bowen_anom_mean",
            "mean_bowen_anom", "Bowen_anom_mean", "Bowen_anom"
        ],
        desc="event-level Bowen anomaly"
    )
    ef_col = find_col(
        df,
        [
            "event_mean_ef_anom", "ef_anom_event_mean",
            "ef_anomaly_event_mean", "event_ef_anom_mean",
            "mean_ef_anom", "EF_anom_mean", "EF_anom"
        ],
        desc="event-level EF anomaly"
    )
    h_col = find_col(
        df,
        [
            "event_mean_h_anom", "h_anom_event_mean",
            "event_mean_sensible_heat_anom", "sensible_heat_anom_event_mean",
            "mean_h_anom", "H_anom_mean", "H_anom"
        ],
        desc="event-level H anomaly"
    )
    front_col = find_col(
        df,
        [
            "front_local_drying_contrast",
            "front_local_drying_speed_contrast",
            "front_drying_contrast",
            "front_local_drying",
            "front_drying_speed"
        ],
        desc="front-local drying contrast"
    )

    out = df.copy()
    out["bowen_anom_evt"] = pd.to_numeric(out[bowen_col], errors="coerce")
    out["ef_anom_evt"] = pd.to_numeric(out[ef_col], errors="coerce")
    out["h_anom_evt"] = pd.to_numeric(out[h_col], errors="coerce")
    out["front_local_drying"] = pd.to_numeric(out[front_col], errors="coerce")

    # Evaporative suppression: dryward-positive convention.
    out["evap_supp_evt"] = -out["ef_anom_evt"]

    out["bowen_z"] = safe_zscore(out["bowen_anom_evt"])
    out["evap_supp_z"] = safe_zscore(out["evap_supp_evt"])
    out["h_z"] = safe_zscore(out["h_anom_evt"])

    out = out[out["state_num"].between(1, 6, inclusive="both")].copy()
    out["state_num"] = out["state_num"].astype(int)
    out["state_lab"] = out["state_num"].map(lambda i: f"S{i}")

    return out


def load_spatial_summary(spatial_csv: Path) -> pd.DataFrame:
    if not spatial_csv.exists():
        raise FileNotFoundError(f"找不到空间 CSV：{spatial_csv}")

    df = pd.read_csv(spatial_csv)

    lon_col = find_col(df, ["longitude", "lon", "x"], desc="longitude")
    lat_col = find_col(df, ["latitude", "lat", "y"], desc="latitude")

    bowen_col = find_col(
        df,
        [
            "dry_wet_bowen_anom", "dry_minus_wet_bowen_anom",
            "bowen_anom_contrast", "bowen_anomaly_contrast",
            "drywetbowenanom", "dryminuswetbowenanom"
        ],
        desc="spatial Bowen contrast"
    )
    ef_col = find_col(
        df,
        [
            "dry_wet_ef_anom", "dry_minus_wet_ef_anom",
            "ef_anom_contrast", "ef_anomaly_contrast",
            "drywetefanom", "dryminuswetefanom"
        ],
        desc="spatial EF contrast"
    )
    h_col = find_col(
        df,
        [
            "dry_wet_h_anom", "dry_minus_wet_h_anom",
            "h_anom_contrast", "sensible_heat_anom_contrast",
            "drywethanom", "dryminuswethanom"
        ],
        desc="spatial H contrast"
    )

    dry_n_col = find_col(
        df,
        ["dry_n", "n_dry", "ndry", "count_dry", "dry_count"],
        required=False,
        desc="dry_n"
    )
    wet_n_col = find_col(
        df,
        ["wet_n", "n_wet", "nwet", "count_wet", "wet_count"],
        required=False,
        desc="wet_n"
    )

    out = df.copy()
    out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["bowen_map"] = pd.to_numeric(out[bowen_col], errors="coerce")
    out["ef_map"] = pd.to_numeric(out[ef_col], errors="coerce")
    out["evap_supp_map"] = -out["ef_map"]
    out["h_map"] = pd.to_numeric(out[h_col], errors="coerce")

    if dry_n_col is not None:
        out["dry_n"] = pd.to_numeric(out[dry_n_col], errors="coerce")
    else:
        out["dry_n"] = np.nan

    if wet_n_col is not None:
        out["wet_n"] = pd.to_numeric(out[wet_n_col], errors="coerce")
    else:
        out["wet_n"] = np.nan

    mask = np.isfinite(out["lon"]) & np.isfinite(out["lat"])
    if out["dry_n"].notna().any():
        mask &= out["dry_n"] >= SPATIAL_MIN_N_DRY
    if out["wet_n"].notna().any():
        mask &= out["wet_n"] >= SPATIAL_MIN_N_WET

    out = out.loc[mask].copy()
    return out


# =============================================================================
# 5. Plotting helpers
# =============================================================================
def make_regular_grid(df, lon_col, lat_col, val_col):
    p = df.pivot_table(index=lat_col, columns=lon_col, values=val_col, aggfunc="mean")
    lats = p.index.to_numpy(dtype=float)
    lons = p.columns.to_numpy(dtype=float)
    z = p.to_numpy(dtype=float)

    lon_order = np.argsort(lons)
    lat_order = np.argsort(lats)
    lons = lons[lon_order]
    lats = lats[lat_order]
    z = z[np.ix_(lat_order, lon_order)]

    return lons, lats, z


def summarize_state_distribution(df, value_col):
    rows = []
    for s in range(1, 7):
        sub = df.loc[df["state_num"] == s, value_col].dropna()
        if len(sub) == 0:
            rows.append({
                "state_num": s,
                "state_lab": f"S{s}",
                "n": 0,
                "mean": np.nan,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "sd": np.nan,
                "se": np.nan,
                "ci95": np.nan,
            })
        else:
            rows.append({
                "state_num": s,
                "state_lab": f"S{s}",
                "n": len(sub),
                "mean": sub.mean(),
                "median": sub.median(),
                "q25": sub.quantile(0.25),
                "q75": sub.quantile(0.75),
                "sd": sub.std(ddof=1),
                "se": sub.std(ddof=1) / np.sqrt(len(sub)),
                "ci95": 1.96 * sub.std(ddof=1) / np.sqrt(len(sub)),
            })
    return pd.DataFrame(rows)


def bootstrap_mean_diff(x, y, n_boot=N_BOOT, seed=RANDOM_SEED):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    xb = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    yb = rng.choice(y, size=(n_boot, len(y)), replace=True).mean(axis=1)
    diff = xb - yb

    return (
        float(x.mean() - y.mean()),
        float(np.percentile(diff, 2.5)),
        float(np.percentile(diff, 97.5)),
    )


def cohens_d(x, y):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))

    if not np.isfinite(pooled) or pooled == 0:
        return np.nan

    return float((x.mean() - y.mean()) / pooled)


def bootstrap_cohens_d(x, y, n_boot=N_BOOT, seed=RANDOM_SEED + 17):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        out[i] = cohens_d(xb, yb)

    return (
        cohens_d(x, y),
        float(np.nanpercentile(out, 2.5)),
        float(np.nanpercentile(out, 97.5)),
    )


# =============================================================================
# 6. Panel plotting functions
# =============================================================================
def plot_spatial_panel(fig, ax, sdf, val_col, title, cbar_label, letter,
                       cmap="RdBu_r", q=0.98, floor=None, show_left_labels=True):
    add_panel_letter(ax, letter, x=-0.20, y=1.12)

    lons, lats, z = make_regular_grid(sdf, "lon", "lat", val_col)

    if MAP_SMOOTH_SIGMA > 0:
        z = smooth_2d(z, MAP_SMOOTH_SIGMA)

    lon_edges = compute_edges(lons)
    lat_edges = compute_edges(lats)

    vmax = robust_sym_lim(z, q=q, floor=floor)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    proj = ccrs.PlateCarree()
    pcm = ax.pcolormesh(
        lon_edges, lat_edges, z,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=proj,
        rasterized=True,
    )

    ax.set_extent(CONUS_EXTENT, crs=proj)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.65, edgecolor="0.45")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.40, edgecolor="0.55")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.30, edgecolor="0.78")

    gl = ax.gridlines(
        crs=proj,
        draw_labels=True,
        linewidth=0.30,
        color="0.88",
        alpha=1.0,
        linestyle="-"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_labels
    gl.xlabel_style = {"size": TICK_FS - 1}
    gl.ylabel_style = {"size": TICK_FS - 1}

    ax.set_title(title, pad=7, fontsize=TITLE_FS)

    cbar = fig.colorbar(
        pcm,
        ax=ax,
        orientation="horizontal",
        fraction=0.054,
        pad=0.080,
        aspect=35,
        extend="both"
    )
    cbar.set_label(cbar_label, fontsize=LABEL_FS - 1, labelpad=3)
    cbar.ax.tick_params(labelsize=TICK_FS - 1, length=2)

    return pcm


def plot_panel_d_state_gradients(ax, event_df):
    """
    Panel d.
    Final correction: bottom n=... labels are removed.
    """
    add_panel_letter(ax, "d", x=-0.14, y=1.08)

    ax.set_title("State gradients", fontsize=TITLE_FS, pad=6)
    ax.axhline(0, color="0.70", lw=0.95, ls="--")

    var_info = [
        ("bowen_z", "Bowen′", VAR_COLORS["Bowen'"]),
        ("evap_supp_z", "−EF′", VAR_COLORS["−EF'"]),
        ("h_z", "H′", VAR_COLORS["H'"]),
    ]

    states = np.arange(1, 7)

    for col, label, color in var_info:
        smry = summarize_state_distribution(event_df, col)
        x = smry["state_num"].to_numpy()
        y = smry["median"].to_numpy()
        y1 = smry["q25"].to_numpy()
        y2 = smry["q75"].to_numpy()

        ax.plot(
            x, y,
            color=color,
            lw=2.4,
            marker="o",
            ms=6.2,
            label=label,
            zorder=4
        )
        ax.vlines(
            x, y1, y2,
            color=color,
            lw=1.7,
            alpha=0.85,
            zorder=3
        )

    ax.set_xticks(states)
    ax.set_xticklabels([f"S{i}" for i in states])
    ax.set_xlabel("Initial soil-moisture state")
    ax.set_ylabel("Standardized anomaly")

    ax.legend(
        frameon=False,
        loc="upper right",
        ncol=1,
        handlelength=1.7,
        borderaxespad=0.3
    )

    ax.grid(axis="y", color="0.88", linewidth=0.75)
    ax.grid(axis="x", visible=False)


def plot_panel_e_regime_zoom(ax, event_df):
    """
    Panel e.
    Final correction: the small grey background event points are darkened.
    """
    add_panel_letter(ax, "e", x=-0.14, y=1.08)
    ax.set_title("Regime-space zoom", fontsize=TITLE_FS, pad=6)

    xcol = "evap_supp_evt"
    ycol = "h_anom_evt"

    tmp = event_df[["state_num", "state_lab", xcol, ycol]].dropna().copy()
    if tmp.empty:
        raise ValueError("Panel e data are empty: evap_supp_evt / h_anom_evt")

    bg = tmp.copy()
    if len(bg) > 3500:
        bg = bg.sample(3500, random_state=RANDOM_SEED)

    ax.scatter(
        bg[xcol], bg[ycol],
        s=E_BG_SIZE,
        color=E_BG_COLOR,
        alpha=E_BG_ALPHA,
        linewidths=0,
        rasterized=True,
        zorder=1
    )

    smry_rows = []
    for s in range(1, 7):
        sub = tmp[tmp["state_num"] == s]
        if len(sub) == 0:
            continue
        smry_rows.append({
            "state_num": s,
            "state_lab": f"S{s}",
            "x_med": sub[xcol].median(),
            "x_q25": sub[xcol].quantile(0.25),
            "x_q75": sub[xcol].quantile(0.75),
            "y_med": sub[ycol].median(),
            "y_q25": sub[ycol].quantile(0.25),
            "y_q75": sub[ycol].quantile(0.75),
            "n": len(sub),
        })

    smry = pd.DataFrame(smry_rows).sort_values("state_num")
    if smry.empty:
        raise ValueError("Panel e state summary is empty after grouping by S1-S6.")

    x_span = smry["x_q75"].max() - smry["x_q25"].min()
    y_span = smry["y_q75"].max() - smry["y_q25"].min()
    xmin = smry["x_q25"].min() - 0.22 * x_span
    xmax = smry["x_q75"].max() + 0.22 * x_span
    ymin = smry["y_q25"].min() - 0.28 * y_span
    ymax = smry["y_q75"].max() + 0.28 * y_span

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.axvline(0, color="0.72", lw=0.95, ls="--", zorder=0)
    ax.axhline(0, color="0.72", lw=0.95, ls="--", zorder=0)

    ax.plot(
        smry["x_med"], smry["y_med"],
        color="0.25",
        lw=1.6,
        alpha=0.95,
        zorder=2
    )

    for _, row in smry.iterrows():
        s_lab = row["state_lab"]
        color = STATE_COLORS[s_lab]
        x = row["x_med"]
        y = row["y_med"]
        xerr = np.array([[x - row["x_q25"]], [row["x_q75"] - x]])
        yerr = np.array([[y - row["y_q25"]], [row["y_q75"] - y]])

        ax.errorbar(
            x, y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=2.2,
            capsize=0,
            alpha=0.95,
            zorder=4
        )
        ax.scatter(
            x, y,
            s=86,
            facecolor=color,
            edgecolor="#222222",
            linewidth=1.2,
            zorder=5
        )
        ax.text(
            x + 0.015 * (xmax - xmin),
            y + 0.020 * (ymax - ymin),
            s_lab,
            color=color,
            fontsize=ANNOT_FS,
            fontweight="bold"
        )

    ax.set_xlabel("Evaporative suppression anomaly (−EF′)")
    ax.set_ylabel("H′ (W m$^{-2}$)")
    ax.grid(color="0.88", linewidth=0.75)


def plot_panel_f_dry_wet_front_summary(ax, event_df):
    """
    Panel f.
    Dry = S1-S2; Wet = S5-S6.
    """
    add_panel_letter(ax, "f", x=-0.14, y=1.08)
    ax.set_title("Front-local response", fontsize=TITLE_FS, pad=6)

    df = event_df.copy()
    df["drywet_group"] = np.where(
        df["state_num"].isin([1, 2]), "Dry",
        np.where(df["state_num"].isin([5, 6]), "Wet", "Transitional")
    )

    dat = df[df["drywet_group"].isin(["Dry", "Wet"])][
        ["drywet_group", "front_local_drying"]
    ].dropna().copy()

    if dat.empty:
        raise ValueError("Panel f has no valid Dry/Wet front-local drying data.")

    dry = dat.loc[dat["drywet_group"] == "Dry", "front_local_drying"].dropna()
    wet = dat.loc[dat["drywet_group"] == "Wet", "front_local_drying"].dropna()

    rng = np.random.default_rng(RANDOM_SEED)
    y_pos = {"Wet": 0, "Dry": 1}
    colors = {"Dry": DRY_COLOR, "Wet": WET_COLOR}

    for group in ["Dry", "Wet"]:
        vals = dat.loc[dat["drywet_group"] == group, "front_local_drying"].dropna().values
        if len(vals) == 0:
            continue

        vals_plot = vals
        if len(vals_plot) > 1800:
            vals_plot = rng.choice(vals_plot, 1800, replace=False)

        yj = y_pos[group] + rng.normal(0, 0.055, size=len(vals_plot))
        ax.scatter(
            vals_plot, yj,
            s=8,
            color=colors[group],
            alpha=0.10,
            linewidths=0,
            rasterized=True,
            zorder=1
        )

        q10, q25, q50, q75, q90 = np.nanpercentile(vals, [10, 25, 50, 75, 90])
        y = y_pos[group]
        ax.hlines(y, q10, q90, color=colors[group], lw=1.6, alpha=0.75, zorder=3)
        ax.hlines(y, q25, q75, color=colors[group], lw=7.0, alpha=0.90, zorder=4)
        ax.scatter(
            q50, y,
            s=72,
            facecolor="white",
            edgecolor=colors[group],
            linewidth=2.0,
            zorder=5
        )

    ax.axvline(0, color="0.70", lw=0.95, ls="--", zorder=0)

    mean_diff, mean_lo, mean_hi = bootstrap_mean_diff(dry, wet)
    d, _, _ = bootstrap_cohens_d(dry, wet)

    xlo, xhi = robust_xlim(dat["front_local_drying"], qlo=0.01, qhi=0.99, pad_frac=0.12)
    xlo = min(xlo, 0)
    xhi = max(xhi, 0)
    ax.set_xlim(xlo, xhi)

    ax.set_yticks([1, 0])
    ax.set_yticklabels([
        f"Dry\nS1–S2\nn={len(dry):,}",
        f"Wet\nS5–S6\nn={len(wet):,}",
    ])
    ax.set_xlabel("Front-local drying contrast")
    ax.set_ylabel("")

    ann = (
        f"Dry − wet mean = {mean_diff:.2f}\n"
        f"95% CI [{mean_lo:.2f}, {mean_hi:.2f}]\n"
        f"Std. contrast = {d:.2f}"
    )
    ax.text(
        0.98, 0.94, ann,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOT_FS - 1,
        color="0.20",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.82", lw=0.7, alpha=0.78)
    )

    ax.grid(axis="x", color="0.88", linewidth=0.75)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)


# =============================================================================
# 7. Build figure
# =============================================================================
def build_figure(event_df, spatial_df, out_png: Path, out_pdf: Path):
    set_mpl_style()
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(21.5, 10.9))
    gs = GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        height_ratios=[1.34, 1.0],
        hspace=0.25,
        wspace=0.28
    )

    axa = fig.add_subplot(gs[0, 0], projection=proj)
    axb = fig.add_subplot(gs[0, 1], projection=proj)
    axc = fig.add_subplot(gs[0, 2], projection=proj)

    plot_spatial_panel(
        fig, axa, spatial_df, "bowen_map",
        title="Bowen anomaly contrast",
        cbar_label="Dry − wet Bowen anomaly",
        letter="a",
        q=0.98,
        floor=2.0,
        show_left_labels=True
    )
    plot_spatial_panel(
        fig, axb, spatial_df, "evap_supp_map",
        title="Evaporative suppression contrast",
        cbar_label="Dry − wet evaporative suppression (−ΔEF)",
        letter="b",
        q=0.98,
        floor=0.05,
        show_left_labels=True
    )
    plot_spatial_panel(
        fig, axc, spatial_df, "h_map",
        title="Sensible-heat anomaly contrast",
        cbar_label="Dry − wet H anomaly (W m$^{-2}$)",
        letter="c",
        q=0.98,
        floor=5.0,
        show_left_labels=True
    )

    axd = fig.add_subplot(gs[1, 0])
    axe = fig.add_subplot(gs[1, 1])
    axf = fig.add_subplot(gs[1, 2])

    plot_panel_d_state_gradients(axd, event_df)
    plot_panel_e_regime_zoom(axe, event_df)
    plot_panel_f_dry_wet_front_summary(axf, event_df)

    # No super-title and no footer. Put definitions in caption/methods instead.
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.080)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=450, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=450, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 8. Main
# =============================================================================
def main():
    print("=" * 100)
    print("Figure 3 relayout v5 | clean final script")
    print("=" * 100)
    print(f"[INFO] Event CSV  : {EVENT_CSV}")
    print(f"[INFO] Spatial CSV: {SPATIAL_CSV}")
    print(f"[INFO] Output PNG : {OUT_PNG}")
    print(f"[INFO] Output PDF : {OUT_PDF}")
    print("=" * 100)

    event_df = load_event_summary(EVENT_CSV)
    spatial_df = load_spatial_summary(SPATIAL_CSV)

    print(f"[INFO] Event rows loaded  : {len(event_df):,}")
    print(f"[INFO] Spatial rows loaded: {len(spatial_df):,}")
    print("[INFO] Event state counts:")
    print(event_df["state_lab"].value_counts().sort_index())

    build_figure(event_df, spatial_df, OUT_PNG, OUT_PDF)

    print("=" * 100)
    print("[DONE] Figure 3 saved:")
    print(f"  PNG: {OUT_PNG}")
    print(f"  PDF: {OUT_PDF}")
    print("=" * 100)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
