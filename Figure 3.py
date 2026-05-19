# -*- coding: utf-8 -*-
"""
Figure 2 final v11
------------------
Fix:
1) Align panel letters a/c/e to the same left-column x position.
2) Align panel letters b/d/f to the same right-column x position.
3) Keep e/f second row lowered from v10.
4) Keep fonts from v9/v10.
5) Keep panel-specific orange contours.
6) Use cached CSV only.

Output:
    Figure2_final_v11_panel_letters_column_aligned.png
    Figure2_final_v11_panel_letters_column_aligned.pdf
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from scipy.ndimage import gaussian_filter

# =============================================================================
# 0. USER CONFIGURATION
# =============================================================================

OUT_DIR = r"E:\第二篇数据0427\Figure2_circulation_control_outputs"

MATCHED_EVENT_TABLE_CSV = os.path.join(
    OUT_DIR,
    "Figure2_main_circulation_matched_event_table.csv"
)

FRONT_LOCAL_EVENT_CSV = os.path.join(
    OUT_DIR,
    "Figure2_panel_f_front_local_drying_contrast_computed_for_matched_events.csv"
)

CACHED_WIND_COMPOSITE_CSV = os.path.join(
    OUT_DIR,
    "Figure2_revised_panel_ab_matched_wind_composite.csv"
)

CACHED_ZW_COMPOSITE_CSV = os.path.join(
    OUT_DIR,
    "Figure2_revised_panel_cd_matched_Z500_W500_composite.csv"
)

FINAL_FIG_PNG = os.path.join(
    OUT_DIR,
    "Figure2_final_v11_panel_letters_column_aligned.png"
)

FINAL_FIG_PDF = os.path.join(
    OUT_DIR,
    "Figure2_final_v11_panel_letters_column_aligned.pdf"
)

CONUS_EXTENT = (-126, -66, 24, 50)

DPI = 500
FIGSIZE = (18.6, 16.4)

# =============================================================================
# FONT
# =============================================================================

BASE_FONT = 24
TICK_FONT = 23
LABEL_FONT = 24
TITLE_FONT = 25
LEGEND_FONT = 21
PANEL_FONT = 31
CBAR_LABEL_FONT = 23
CBAR_TICK_FONT = 22
PAIRCOUNT_FONT = 22

EF_LABEL_FONT = 22
EF_ROW_TITLE_FONT = 23

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = BASE_FONT
plt.rcParams["axes.titlesize"] = TITLE_FONT
plt.rcParams["axes.labelsize"] = LABEL_FONT
plt.rcParams["xtick.labelsize"] = TICK_FONT
plt.rcParams["ytick.labelsize"] = TICK_FONT
plt.rcParams["legend.fontsize"] = LEGEND_FONT
plt.rcParams["axes.linewidth"] = 1.10
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =============================================================================
# STYLE
# =============================================================================

DRY_COLOR = "#c44e52"
WET_COLOR = "#4c72b0"
EFFECT_COLOR = "#333333"
BOUNDARY_EFFECT_COLOR = "#1b9e77"
ORANGE_CONTOUR_COLOR = "#e66101"

SIGMA_WS850 = 1.0
SIGMA_WS250 = 1.0
SIGMA_Z500 = 1.15

ASCENT_COARSEN_LON = 1.5
ASCENT_COARSEN_LAT = 1.5
SIGMA_ASCENT = 1.1

SIGMA_UV = 1.0

QUIVER_STRIDE_850 = 5
QUIVER_STRIDE_250 = 5
QUIVER_SCALE_850 = 18
QUIVER_SCALE_250 = 55

SHOW_QUIVER_KEY = False

MAP_CBAR_PAD = 0.115
MAP_CBAR_FRACTION = 0.042
MAP_CBAR_SHRINK = 0.94

# Used to define column-aligned letter x positions from e/f anchors
PANEL_DX = 0.042
PANEL_DY = 0.045

PANEL_CONTOUR_PERCENTILE = 88
PANEL_CONTOUR_LINEWIDTH = 2.0

EF_ROW_HSPACE = 1.18
EF_COL_WSPACE = 0.24

BOOTSTRAP_N = 5000
RANDOM_SEED = 42

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. BASIC UTILITIES
# =============================================================================

def log(msg):
    print(msg, flush=True)


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing columns; none of these were found: {candidates}")
    return None


def robust_sym_clim(arr, q=0.98, fallback=1.0):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return (-fallback, fallback)

    vmax = np.nanquantile(np.abs(arr), q)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = fallback

    return (-vmax, vmax)


def smooth_grid(Z, sigma):
    if sigma is None or sigma <= 0:
        return Z

    mask = np.isfinite(Z)

    if not mask.any():
        return Z

    Z0 = np.where(mask, Z, 0.0)
    W = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")
    Zs = gaussian_filter(Z0, sigma=sigma, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        out = Zs / np.where(W <= 1e-9, np.nan, W)

    out[W <= 1e-9] = np.nan

    return out


def to_grid(df, value_col):
    tmp = df[["longitude", "latitude", value_col]].dropna().copy()

    tmp["longitude"] = pd.to_numeric(tmp["longitude"], errors="coerce")
    tmp["latitude"] = pd.to_numeric(tmp["latitude"], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")

    tmp = tmp.dropna()

    lons = np.sort(np.unique(tmp["longitude"].values))
    lats = np.sort(np.unique(tmp["latitude"].values))

    grid = tmp.pivot_table(
        index="latitude",
        columns="longitude",
        values=value_col,
        aggfunc="mean"
    )

    grid = grid.reindex(index=lats, columns=lons)

    return lons, lats, grid.values


def bootstrap_mean_ci(values, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

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


# =============================================================================
# 2. CARTOPY
# =============================================================================

def import_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        return ccrs, cfeature, LongitudeFormatter, LatitudeFormatter
    except Exception as e:
        raise ImportError(
            "Cartopy is required because the figure must preserve U.S. boundary "
            "and state boundaries.\n"
            f"Original error: {repr(e)}"
        )


ccrs, cfeature, LongitudeFormatter, LatitudeFormatter = import_cartopy()


# =============================================================================
# 3. LOAD CACHED CSV ONLY
# =============================================================================

def load_cached_wind_composite():
    if not os.path.exists(CACHED_WIND_COMPOSITE_CSV):
        raise FileNotFoundError(
            f"Missing cached wind composite:\n{CACHED_WIND_COMPOSITE_CSV}\n"
            "This script will not rebuild it."
        )

    df = pd.read_csv(CACHED_WIND_COMPOSITE_CSV, low_memory=False)

    for c in df.columns:
        if c != "coord_key":
            df[c] = pd.to_numeric(df[c], errors="ignore")

    req = [
        "longitude", "latitude",
        "ws850_diff", "u850_diff", "v850_diff",
        "ws250_diff", "u250_diff", "v250_diff"
    ]

    miss = [c for c in req if c not in df.columns]

    if miss:
        raise KeyError(f"Wind composite missing columns: {miss}")

    return df


def load_cached_zw_composite():
    if not os.path.exists(CACHED_ZW_COMPOSITE_CSV):
        raise FileNotFoundError(
            f"Missing cached Z500/W500 composite:\n{CACHED_ZW_COMPOSITE_CSV}\n"
            "This script will not rebuild it."
        )

    df = pd.read_csv(CACHED_ZW_COMPOSITE_CSV, low_memory=False)

    for c in df.columns:
        if c != "coord_key":
            df[c] = pd.to_numeric(df[c], errors="ignore")

    z_col = find_col(df, ["z500_diff"], required=True)

    if "ascent_diff" not in df.columns:
        w_col = find_col(df, ["w500_diff"], required=True)
        df["ascent_diff"] = -pd.to_numeric(df[w_col], errors="coerce")

    df = df.rename(columns={z_col: "z500_diff"})

    return df


def load_matched_event_table():
    if not os.path.exists(MATCHED_EVENT_TABLE_CSV):
        raise FileNotFoundError(
            f"Missing matched-event table:\n{MATCHED_EVENT_TABLE_CSV}"
        )

    df = pd.read_csv(MATCHED_EVENT_TABLE_CSV, low_memory=False)

    if "dry_vs_wet" not in df.columns:
        if "regime" in df.columns:
            rr = df["regime"].astype(str).str.lower().str.strip()
            df["dry_vs_wet"] = np.where(
                rr.eq("dry"),
                1,
                np.where(rr.eq("wet"), 0, np.nan)
            )
        elif "start_state" in df.columns:
            ss = df["start_state"].astype(str).str.upper().str.strip()
            df["dry_vs_wet"] = np.where(
                ss.isin(["S1", "S2"]),
                1,
                np.where(ss.isin(["S5", "S6"]), 0, np.nan)
            )
        else:
            raise KeyError("Cannot infer dry_vs_wet from matched-event table.")

    numeric_cols = [
        "matched_pair_id", "dry_vs_wet",
        "event_voxels", "max_area_km2", "duration_days",
        "front_local_drying_contrast",
        "net_displacement_km", "path_length_km", "max_daily_step_km",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    need_front = (
        ("front_local_drying_contrast" not in df.columns) or
        (
            df["front_local_drying_contrast"].notna().sum() == 0
            if "front_local_drying_contrast" in df.columns else True
        )
    )

    if need_front and os.path.exists(FRONT_LOCAL_EVENT_CSV) and "event_id" in df.columns:
        fl = pd.read_csv(FRONT_LOCAL_EVENT_CSV, low_memory=False)

        if "event_id" in fl.columns:
            front_col = find_col(
                fl,
                ["front_local_drying_contrast"],
                required=False
            )

            if front_col is not None:
                fl = fl[["event_id", front_col]].drop_duplicates("event_id")
                fl = fl.rename(columns={front_col: "front_local_drying_contrast"})

                df = df.drop(
                    columns=["front_local_drying_contrast"],
                    errors="ignore"
                ).merge(
                    fl,
                    on="event_id",
                    how="left"
                )

                df["front_local_drying_contrast"] = pd.to_numeric(
                    df["front_local_drying_contrast"],
                    errors="coerce"
                )

    need = ["matched_pair_id", "dry_vs_wet", "event_voxels", "max_area_km2"]
    miss = [c for c in need if c not in df.columns]

    if miss:
        raise KeyError(f"Matched-event table missing required columns: {miss}")

    df = df[df["dry_vs_wet"].isin([0, 1])].copy()

    df["event_voxels"] = df["event_voxels"].clip(lower=1)
    df["max_area_km2"] = df["max_area_km2"].clip(lower=1)

    df["log10_event_voxels"] = np.log10(df["event_voxels"])
    df["log10_max_area_km2"] = np.log10(df["max_area_km2"])

    for raw, new in [
        ("net_displacement_km", "log1p_net_displacement_km"),
        ("path_length_km", "log1p_path_length_km"),
        ("max_daily_step_km", "log1p_max_daily_step_km"),
    ]:
        if raw in df.columns:
            df[new] = np.log1p(df[raw].clip(lower=0))

    return df


# =============================================================================
# 4. PANEL-SPECIFIC CONTOURS
# =============================================================================

def coarsen_spatial_mean(df, value_col, lon_step=1.5, lat_step=1.5):
    tmp = df[["longitude", "latitude", value_col]].dropna().copy()

    tmp["longitude"] = pd.to_numeric(tmp["longitude"], errors="coerce")
    tmp["latitude"] = pd.to_numeric(tmp["latitude"], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")

    tmp = tmp.dropna()

    tmp["lon_bin"] = np.round(tmp["longitude"] / lon_step) * lon_step
    tmp["lat_bin"] = np.round(tmp["latitude"] / lat_step) * lat_step

    out = (
        tmp.groupby(["lon_bin", "lat_bin"], as_index=False)[value_col]
        .mean()
        .rename(columns={"lon_bin": "longitude", "lat_bin": "latitude"})
    )

    return out


def prepare_panel_specific_contour(
    df,
    value_col,
    sigma=1.0,
    percentile=PANEL_CONTOUR_PERCENTILE,
    use_abs=True
):
    lons, lats, Z = to_grid(df, value_col)
    Z = smooth_grid(Z, sigma=sigma)

    if use_abs:
        Zm = np.abs(Z)
    else:
        Zm = Z.copy()

    valid = Zm[np.isfinite(Zm)]

    if valid.size == 0:
        return None

    level = np.nanpercentile(valid, percentile)

    if not np.isfinite(level):
        return None

    X, Y = np.meshgrid(lons, lats)

    return X, Y, Zm, level


# =============================================================================
# 5. PANEL LETTERS
# =============================================================================

def add_panel_letter_figure(fig, ax, letter, x_override=None, dx=PANEL_DX, dy=PANEL_DY):
    """
    Put panel letters in figure coordinates.

    If x_override is supplied, use it directly.
    This allows a/c/e and b/d/f to be column-aligned.
    """
    bbox = ax.get_position()

    if x_override is None:
        x = bbox.x0 - dx
    else:
        x = x_override

    y = bbox.y1 + dy

    fig.text(
        x,
        y,
        letter,
        fontsize=PANEL_FONT,
        fontweight="bold",
        ha="left",
        va="top"
    )


# =============================================================================
# 6. MAP DRAWING
# =============================================================================

def format_map_ax(ax, left_labels=True, bottom_labels=True):
    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())

    ax.coastlines(resolution="50m", linewidth=0.70, color="0.35")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.50, edgecolor="0.45")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.35, edgecolor="0.72")

    xticks = np.arange(-120, -69, 10)
    yticks = np.arange(25, 51, 5)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    if not bottom_labels:
        ax.set_xticklabels([])

    if not left_labels:
        ax.set_yticklabels([])

    ax.tick_params(length=3.0, width=0.9, pad=2.0, labelsize=TICK_FONT)

    ax.gridlines(
        draw_labels=False,
        linewidth=0.30,
        color="0.87",
        linestyle="-",
        alpha=1.0
    )


def add_map_colorbar(
    fig,
    mappable,
    ax,
    label,
    small_decimal=False,
    force_three_ticks=False,
    tick_values=None
):
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="horizontal",
        pad=MAP_CBAR_PAD,
        fraction=MAP_CBAR_FRACTION,
        shrink=MAP_CBAR_SHRINK,
        extend="both"
    )

    cbar.ax.tick_params(labelsize=CBAR_TICK_FONT, length=2.8, pad=2.0)
    cbar.set_label(label, fontsize=CBAR_LABEL_FONT, labelpad=3)

    if force_three_ticks and tick_values is not None:
        cbar.set_ticks(tick_values)
        cbar.formatter = FormatStrFormatter("%.3f")
        cbar.update_ticks()
    elif small_decimal:
        cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        cbar.formatter = FormatStrFormatter("%.3f")
        cbar.update_ticks()
    else:
        cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        cbar.update_ticks()

    return cbar


def draw_panel_specific_contour(ax, contour):
    if contour is None:
        return

    Xc, Yc, Zc, level = contour

    ax.contour(
        Xc,
        Yc,
        Zc,
        levels=[level],
        colors=[ORANGE_CONTOUR_COLOR],
        linewidths=PANEL_CONTOUR_LINEWIDTH,
        transform=ccrs.PlateCarree(),
        zorder=4
    )


def plot_map_panel_pcolormesh(
    fig,
    ax,
    df,
    value_col,
    title,
    cbar_label,
    sigma_fill,
    contour=None,
    quiver_cols=None,
    sigma_uv=1.0,
    quiver_stride=5,
    quiver_scale=20,
    left_labels=True,
    bottom_labels=True,
    robust_q=0.985,
    fallback=1.0
):
    format_map_ax(ax, left_labels=left_labels, bottom_labels=bottom_labels)

    lons, lats, Z = to_grid(df, value_col)
    Z = smooth_grid(Z, sigma_fill)

    Xg, Yg = np.meshgrid(lons, lats)

    vmin, vmax = robust_sym_clim(Z, q=robust_q, fallback=fallback)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    pm = ax.pcolormesh(
        Xg,
        Yg,
        Z,
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    draw_panel_specific_contour(ax, contour)

    if quiver_cols is not None:
        u_col, v_col = quiver_cols

        lons_u, lats_u, U = to_grid(df, u_col)
        lons_v, lats_v, V = to_grid(df, v_col)

        if not (
            np.array_equal(lons, lons_u) and np.array_equal(lats, lats_u)
            and np.array_equal(lons, lons_v) and np.array_equal(lats, lats_v)
        ):
            U_df = df[["longitude", "latitude", u_col]].dropna().pivot_table(
                index="latitude",
                columns="longitude",
                values=u_col,
                aggfunc="mean"
            )

            V_df = df[["longitude", "latitude", v_col]].dropna().pivot_table(
                index="latitude",
                columns="longitude",
                values=v_col,
                aggfunc="mean"
            )

            U_df = U_df.reindex(index=lats, columns=lons)
            V_df = V_df.reindex(index=lats, columns=lons)

            U = U_df.values
            V = V_df.values

        U = smooth_grid(U, sigma_uv)
        V = smooth_grid(V, sigma_uv)

        qs = slice(None, None, quiver_stride)

        Xq = Xg[qs, qs]
        Yq = Yg[qs, qs]
        Uq = U[qs, qs]
        Vq = V[qs, qs]

        mask = (
            np.isfinite(Xq)
            & np.isfinite(Yq)
            & np.isfinite(Uq)
            & np.isfinite(Vq)
        )

        Xq = Xq[mask]
        Yq = Yq[mask]
        Uq = Uq[mask]
        Vq = Vq[mask]

        qv = ax.quiver(
            Xq,
            Yq,
            Uq,
            Vq,
            transform=ccrs.PlateCarree(),
            color="0.28",
            alpha=0.78,
            pivot="mid",
            width=0.0029,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.4,
            scale=quiver_scale,
            zorder=3
        )

        if SHOW_QUIVER_KEY:
            ax.quiverkey(
                qv,
                X=0.90,
                Y=0.08,
                U=1.0,
                label="1 m s$^{-1}$",
                labelpos="E",
                coordinates="axes"
            )

    ax.set_title(title, fontsize=TITLE_FONT, pad=7)

    add_map_colorbar(
        fig=fig,
        mappable=pm,
        ax=ax,
        label=cbar_label,
        small_decimal=False
    )


def plot_map_panel_ascent_synoptic(
    fig,
    ax,
    df,
    value_col,
    title,
    cbar_label,
    contour=None,
    left_labels=True,
    bottom_labels=True,
    lon_step=1.5,
    lat_step=1.5,
    sigma=1.0
):
    format_map_ax(ax, left_labels=left_labels, bottom_labels=bottom_labels)

    coarse = coarsen_spatial_mean(
        df,
        value_col,
        lon_step=lon_step,
        lat_step=lat_step
    )

    lons, lats, Z = to_grid(coarse, value_col)
    Z = smooth_grid(Z, sigma=sigma)

    Xg, Yg = np.meshgrid(lons, lats)

    vmin, vmax = robust_sym_clim(Z, q=0.95, fallback=0.012)
    levels = np.linspace(vmin, vmax, 13)

    cf = ax.contourf(
        Xg,
        Yg,
        Z,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    if np.nanmin(Z) < 0 < np.nanmax(Z):
        ax.contour(
            Xg,
            Yg,
            Z,
            levels=[0.0],
            colors="0.45",
            linewidths=0.85,
            linestyles="-",
            transform=ccrs.PlateCarree(),
            zorder=2
        )

    draw_panel_specific_contour(ax, contour)

    ax.set_title(title, fontsize=TITLE_FONT, pad=7)

    add_map_colorbar(
        fig=fig,
        mappable=cf,
        ax=ax,
        label=cbar_label,
        small_decimal=True,
        force_three_ticks=True,
        tick_values=[vmin, 0.0, vmax]
    )


# =============================================================================
# 7. E / F PANELS
# =============================================================================

def paired_effect_from_matched(df, metric):
    wide = df.pivot_table(
        index="matched_pair_id",
        columns="dry_vs_wet",
        values=metric,
        aggfunc="first"
    )

    if 1 not in wide.columns or 0 not in wide.columns:
        raise RuntimeError(f"Cannot build paired difference for metric: {metric}")

    wide = wide.dropna(subset=[1, 0])

    dry = wide[1].values
    wet = wide[0].values
    diff = dry - wet

    mean, lo, hi = bootstrap_mean_ci(diff)

    return dry, wet, diff, mean, lo, hi, len(diff)


def violin_box_horizontal(ax, dry, wet, xlabel, row_title):
    positions = [1.0, 0.0]

    parts = ax.violinplot(
        [dry, wet],
        positions=positions,
        vert=False,
        widths=0.68,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body, color in zip(parts["bodies"], [DRY_COLOR, WET_COLOR]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)
        body.set_linewidth(1.0)

    ax.boxplot(
        [dry, wet],
        positions=positions,
        vert=False,
        widths=0.18,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="0.35", linewidth=1.0),
        medianprops=dict(color="0.15", linewidth=1.3),
        whiskerprops=dict(color="0.45", linewidth=0.9),
        capprops=dict(color="0.45", linewidth=0.9),
    )

    ax.plot(np.nanmedian(dry), 1.0, "o", ms=4.6, color=DRY_COLOR, zorder=3)
    ax.plot(np.nanmedian(wet), 0.0, "o", ms=4.6, color=WET_COLOR, zorder=3)

    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["Dry", "Wet"], fontsize=TICK_FONT)
    ax.set_xlabel(xlabel, fontsize=EF_LABEL_FONT, labelpad=5)
    ax.set_title(row_title, loc="left", fontsize=EF_ROW_TITLE_FONT, pad=5)

    ax.tick_params(axis="both", labelsize=TICK_FONT)
    ax.grid(axis="x", color="0.90", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def paired_effect_axis(ax, mean, lo, hi, xlabel, color, xlim, n_pairs=None):
    ax.axvline(0, color="0.75", lw=1.0, zorder=1)

    ax.hlines(0.5, lo, hi, color=color, lw=2.2, zorder=2)
    ax.plot(mean, 0.5, "o", color=color, ms=6.0, zorder=3)

    ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=EF_LABEL_FONT, labelpad=5)

    ax.tick_params(axis="x", labelsize=TICK_FONT)

    if n_pairs is not None:
        ax.text(
            0.98,
            0.08,
            f"Matched pairs: {n_pairs:,}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=PAIRCOUNT_FONT
        )

    ax.grid(axis="x", color="0.90", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def compute_common_xlim(stats, include_zero=True):
    vals = []

    for item in stats:
        vals.extend([item["lo"], item["hi"], item["mean"]])

    if include_zero:
        vals.append(0.0)

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return (-1.0, 1.0)

    x0 = np.nanmin(vals)
    x1 = np.nanmax(vals)

    if x0 == x1:
        pad = max(abs(x0) * 0.2, 0.1)
        return (x0 - pad, x1 + pad)

    pad = 0.13 * (x1 - x0)

    return (x0 - pad, x1 + pad)


def pick_companion_mobility_metric(df):
    candidates = [
        (
            "log1p_net_displacement_km",
            "Net displacement",
            "Displacement",
            "Pair diff.",
            BOUNDARY_EFFECT_COLOR
        ),
        (
            "log1p_path_length_km",
            "Path length",
            "Path length",
            "Pair diff.",
            BOUNDARY_EFFECT_COLOR
        ),
        (
            "log1p_max_daily_step_km",
            "Max daily step",
            "Daily step",
            "Pair diff.",
            BOUNDARY_EFFECT_COLOR
        ),
    ]

    for metric, title, xlabel_left, xlabel_right, color in candidates:
        if metric in df.columns and df[metric].notna().sum() >= 20:
            return {
                "metric": metric,
                "title": title,
                "xlabel_left": xlabel_left,
                "xlabel_right": xlabel_right,
                "color": color
            }

    return None


def build_estimation_panel(fig, subplot_spec, matched, row_specs):
    gs = GridSpecFromSubplotSpec(
        len(row_specs),
        2,
        subplot_spec=subplot_spec,
        width_ratios=[1.40, 0.82],
        hspace=EF_ROW_HSPACE,
        wspace=EF_COL_WSPACE
    )

    stats = []

    for spec in row_specs:
        dry, wet, diff, mean, lo, hi, n_pairs = paired_effect_from_matched(
            matched,
            spec["metric"]
        )

        stats.append({
            "dry": dry,
            "wet": wet,
            "diff": diff,
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "n_pairs": n_pairs,
            "spec": spec
        })

    common_xlim = compute_common_xlim(stats, include_zero=True)

    first_left_ax = None

    for i, item in enumerate(stats):
        spec = item["spec"]

        ax_l = fig.add_subplot(gs[i, 0])
        ax_r = fig.add_subplot(gs[i, 1])

        if first_left_ax is None:
            first_left_ax = ax_l

        violin_box_horizontal(
            ax_l,
            dry=item["dry"],
            wet=item["wet"],
            xlabel=spec["xlabel_left"],
            row_title=spec["title"]
        )

        paired_effect_axis(
            ax_r,
            mean=item["mean"],
            lo=item["lo"],
            hi=item["hi"],
            xlabel=spec["xlabel_right"],
            color=spec["color"],
            xlim=common_xlim,
            n_pairs=item["n_pairs"] if i == len(row_specs) - 1 else None
        )

    return first_left_ax


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    log("=" * 100)
    log("[INFO] Loading cached composites and matched event table")

    wind = load_cached_wind_composite()
    zw = load_cached_zw_composite()
    matched = load_matched_event_table()

    log(f"[INFO] Wind composite : {CACHED_WIND_COMPOSITE_CSV}")
    log(f"[INFO] ZW composite   : {CACHED_ZW_COMPOSITE_CSV}")
    log(f"[INFO] Matched table  : {MATCHED_EVENT_TABLE_CSV}")
    log(f"[INFO] Matched rows   : {len(matched):,}")

    if (
        "front_local_drying_contrast" not in matched.columns
        or matched["front_local_drying_contrast"].notna().sum() < 20
    ):
        raise RuntimeError(
            "front_local_drying_contrast is unavailable in matched-event data.\n"
            "Please ensure it exists in MATCHED_EVENT_TABLE_CSV or can be merged "
            "from FRONT_LOCAL_EVENT_CSV."
        )

    companion = pick_companion_mobility_metric(matched)

    if companion is None:
        raise RuntimeError(
            "No usable mobility metric found.\n"
            "Expected at least one of:\n"
            "  net_displacement_km\n"
            "  path_length_km\n"
            "  max_daily_step_km"
        )

    contour_a = prepare_panel_specific_contour(
        wind,
        "ws850_diff",
        sigma=SIGMA_WS850,
        percentile=88,
        use_abs=True
    )

    contour_b = prepare_panel_specific_contour(
        wind,
        "ws250_diff",
        sigma=SIGMA_WS250,
        percentile=88,
        use_abs=True
    )

    contour_c = prepare_panel_specific_contour(
        zw,
        "z500_diff",
        sigma=SIGMA_Z500,
        percentile=88,
        use_abs=True
    )

    coarse_ascent = coarsen_spatial_mean(
        zw,
        "ascent_diff",
        lon_step=ASCENT_COARSEN_LON,
        lat_step=ASCENT_COARSEN_LAT
    )

    contour_d = prepare_panel_specific_contour(
        coarse_ascent,
        "ascent_diff",
        sigma=SIGMA_ASCENT,
        percentile=84,
        use_abs=True
    )

    fig = plt.figure(figsize=FIGSIZE)

    outer = GridSpec(
        3,
        2,
        figure=fig,
        left=0.085,
        right=0.985,
        top=0.975,
        bottom=0.065,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.06, 1.06, 1.08],
        wspace=0.10,
        hspace=0.46
    )

    ax_a = fig.add_subplot(outer[0, 0], projection=ccrs.PlateCarree())
    plot_map_panel_pcolormesh(
        fig=fig,
        ax=ax_a,
        df=wind,
        value_col="ws850_diff",
        title="850-hPa wind",
        cbar_label="Wind-speed anomaly (m s$^{-1}$)",
        sigma_fill=SIGMA_WS850,
        contour=contour_a,
        quiver_cols=("u850_diff", "v850_diff"),
        sigma_uv=SIGMA_UV,
        quiver_stride=QUIVER_STRIDE_850,
        quiver_scale=QUIVER_SCALE_850,
        left_labels=True,
        bottom_labels=True,
        robust_q=0.985,
        fallback=0.45
    )

    ax_b = fig.add_subplot(outer[0, 1], projection=ccrs.PlateCarree())
    plot_map_panel_pcolormesh(
        fig=fig,
        ax=ax_b,
        df=wind,
        value_col="ws250_diff",
        title="250-hPa wind",
        cbar_label="Wind-speed anomaly (m s$^{-1}$)",
        sigma_fill=SIGMA_WS250,
        contour=contour_b,
        quiver_cols=("u250_diff", "v250_diff"),
        sigma_uv=SIGMA_UV,
        quiver_stride=QUIVER_STRIDE_250,
        quiver_scale=QUIVER_SCALE_250,
        left_labels=True,
        bottom_labels=True,
        robust_q=0.985,
        fallback=1.25
    )

    ax_c = fig.add_subplot(outer[1, 0], projection=ccrs.PlateCarree())
    plot_map_panel_pcolormesh(
        fig=fig,
        ax=ax_c,
        df=zw,
        value_col="z500_diff",
        title="Z500 height",
        cbar_label="Z500 anomaly (m)",
        sigma_fill=SIGMA_Z500,
        contour=contour_c,
        quiver_cols=None,
        left_labels=True,
        bottom_labels=True,
        robust_q=0.985,
        fallback=10.0
    )

    ax_d = fig.add_subplot(outer[1, 1], projection=ccrs.PlateCarree())
    plot_map_panel_ascent_synoptic(
        fig=fig,
        ax=ax_d,
        df=zw,
        value_col="ascent_diff",
        title="Synoptic ascent",
        cbar_label="Ascent anomaly (−W500)",
        contour=contour_d,
        left_labels=True,
        bottom_labels=True,
        lon_step=ASCENT_COARSEN_LON,
        lat_step=ASCENT_COARSEN_LAT,
        sigma=SIGMA_ASCENT
    )

    row_specs_e = [
        {
            "metric": "log10_event_voxels",
            "title": "Event voxels",
            "xlabel_left": "Voxels",
            "xlabel_right": "Pair diff.",
            "color": EFFECT_COLOR
        },
        {
            "metric": "log10_max_area_km2",
            "title": "Max area",
            "xlabel_left": "Area",
            "xlabel_right": "Pair diff.",
            "color": EFFECT_COLOR
        },
    ]

    ax_e_anchor = build_estimation_panel(
        fig=fig,
        subplot_spec=outer[2, 0],
        matched=matched,
        row_specs=row_specs_e
    )

    row_specs_f = [
        {
            "metric": "front_local_drying_contrast",
            "title": "Front drying",
            "xlabel_left": "Front drying",
            "xlabel_right": "Pair diff.",
            "color": BOUNDARY_EFFECT_COLOR
        },
        {
            "metric": companion["metric"],
            "title": companion["title"],
            "xlabel_left": companion["xlabel_left"],
            "xlabel_right": companion["xlabel_right"],
            "color": companion["color"]
        },
    ]

    ax_f_anchor = build_estimation_panel(
        fig=fig,
        subplot_spec=outer[2, 1],
        matched=matched,
        row_specs=row_specs_f
    )

    fig.canvas.draw()

    # -------------------------------------------------------------------------
    # Column-aligned panel letters
    # -------------------------------------------------------------------------
    # Use e and f as the column reference because their panel letters are already
    # visually in the correct left margin position.
    left_letter_x = ax_e_anchor.get_position().x0 - PANEL_DX
    right_letter_x = ax_f_anchor.get_position().x0 - PANEL_DX

    add_panel_letter_figure(fig, ax_a, "a", x_override=left_letter_x)
    add_panel_letter_figure(fig, ax_c, "c", x_override=left_letter_x)
    add_panel_letter_figure(fig, ax_e_anchor, "e", x_override=left_letter_x)

    add_panel_letter_figure(fig, ax_b, "b", x_override=right_letter_x)
    add_panel_letter_figure(fig, ax_d, "d", x_override=right_letter_x)
    add_panel_letter_figure(fig, ax_f_anchor, "f", x_override=right_letter_x)

    fig.savefig(FINAL_FIG_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(FINAL_FIG_PDF, dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    log("=" * 100)
    log("[DONE] Final Figure 2 saved:")
    log(FINAL_FIG_PNG)
    log(FINAL_FIG_PDF)
    log("=" * 100)


if __name__ == "__main__":
    main()