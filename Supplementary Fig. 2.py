# -*- coding: utf-8 -*-
"""
Replot lifecycle representativeness figure from existing cached tables only.

Main correction in this version:
    1. Panel b is rebuilt with nested GridSpec.
    2. Panel b heatmap uses aspect='auto' so it fills the top-right panel properly.
    3. Panel b vertical colorbar is placed in a dedicated narrow axis on the right.
    4. Panel labels are manually placed with axes coordinates, avoiding overlap with titles.
    5. No raw event CSVs are reread.
    6. No lifecycle statistics are rebuilt.

Inputs:
    E:\\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本
      \\lifecycle_representativeness_C26

Required cached files:
    event_lifecycle_summary_C26.csv
    lifecycle_trajectory_bins_C26.csv
    start_vs_lifetime_state_matrix_event_count.csv
    alternative_classification_annual_exposure.csv
    alternative_classification_summary.csv

Outputs:
    Figure_lifecycle_representativeness_C26_NatureStyle_bfixed.png
    Figure_lifecycle_representativeness_C26_NatureStyle_bfixed.pdf
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

try:
    from scipy import stats
except Exception:
    stats = None

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS
# =============================================================================

BASE_ROOT = Path(r"E:\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本")
OUT_DIR = BASE_ROOT / "lifecycle_representativeness_C26"

EVENT_SUMMARY_CSV = OUT_DIR / "event_lifecycle_summary_C26.csv"
TRAJ_CSV = OUT_DIR / "lifecycle_trajectory_bins_C26.csv"
MATRIX_CSV = OUT_DIR / "start_vs_lifetime_state_matrix_event_count.csv"
ALT_WIDE_CSV = OUT_DIR / "alternative_classification_annual_exposure.csv"
ALT_SUMMARY_CSV = OUT_DIR / "alternative_classification_summary.csv"

FIG_PNG = OUT_DIR / "Figure_lifecycle_representativeness_C26_NatureStyle_bfixed.png"
FIG_PDF = OUT_DIR / "Figure_lifecycle_representativeness_C26_NatureStyle_bfixed.pdf"


# =============================================================================
# 2. CONSTANTS
# =============================================================================

STATE_ORDER = [1, 2, 3, 4, 5, 6]
STATE_LABELS = [f"S{i}" for i in STATE_ORDER]

GROUP_ORDER = ["Dry-start", "Intermediate-start", "Wet-start"]
GROUP_LABEL = {
    "Dry-start": "Dry-start\n(S1–S2)",
    "Intermediate-start": "Intermediate\n(S3–S4)",
    "Wet-start": "Wet-start\n(S5–S6)",
}

GROUP_COLORS = {
    "Dry-start": "#8c510a",
    "Intermediate-start": "#7f7f7f",
    "Wet-start": "#01665e",
}

ALT_DEFS = [
    "first_day",
    "first_3_days",
    "lifetime_dominant",
    "lifetime_fraction_50",
    "lifetime_fraction_70",
]

ALT_LABELS = {
    "first_day": "First-day\nstate",
    "first_3_days": "First-3-day\nstate",
    "lifetime_dominant": "Lifetime\nstate",
    "lifetime_fraction_50": "Lifetime\nfraction ≥0.5",
    "lifetime_fraction_70": "Lifetime\nfraction ≥0.7",
}

DPI = 500


# =============================================================================
# 3. STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 25,
    "axes.titlesize": 30,
    "axes.labelsize": 27,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
    "legend.fontsize": 18,
    "axes.linewidth": 1.35,
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "mathtext.default": "regular",
})


# =============================================================================
# 4. HELPERS
# =============================================================================

def require_file(fp: Path):
    if not fp.exists():
        raise FileNotFoundError(
            f"Missing required cached file:\n{fp}\n\n"
            "This replot script does not reread raw event CSVs. "
            "Please run the original lifecycle script once to generate cached tables."
        )


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, letter, x=-0.16, y=1.065):
    """
    Stable panel label placement.

    Do not use automatic y-label bounding-box placement here, because heatmap
    panels with colorbars can shift and make labels overlap with titles.
    """
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=36,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
    )


def set_close_ylabel(ax, text, x=-0.078):
    ax.set_ylabel(text, labelpad=1)
    ax.yaxis.set_label_coords(x, 0.5)


def positive_values(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    return vals


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
    if p < 1e-4:
        return f"P={p:.1e}"
    return f"P={p:.3g}"


def get_nature_cmap():
    return LinearSegmentedColormap.from_list(
        "nature_teal_blue",
        ["#f7f7f0", "#d9f0d3", "#7fcdbb", "#2c7fb8", "#08306b"],
        N=256,
    )


def load_cached_tables():
    for fp in [
        EVENT_SUMMARY_CSV,
        TRAJ_CSV,
        MATRIX_CSV,
        ALT_WIDE_CSV,
        ALT_SUMMARY_CSV,
    ]:
        require_file(fp)

    ev = pd.read_csv(EVENT_SUMMARY_CSV, low_memory=False)
    traj = pd.read_csv(TRAJ_CSV, low_memory=False)
    mat = pd.read_csv(MATRIX_CSV, index_col=0)
    alt_wide = pd.read_csv(ALT_WIDE_CSV, low_memory=False)
    alt_summary = pd.read_csv(ALT_SUMMARY_CSV, low_memory=False)

    for col in [
        "year", "first_state", "first3_state", "lifetime_state",
        "object_grid_days", "duration_days",
        "dry_lifetime_fraction", "wet_lifetime_fraction",
        "dry_first_day_fraction", "dry_first3_fraction",
        "early_dry_fraction", "middle_dry_fraction", "late_dry_fraction",
    ]:
        if col in ev.columns:
            ev[col] = pd.to_numeric(ev[col], errors="coerce")

    for col in [
        "year", "event_id", "first_state", "object_grid_days",
        "duration_days", "age_bin", "age_mid",
        "dry_fraction", "wet_fraction",
    ]:
        if col in traj.columns:
            traj[col] = pd.to_numeric(traj[col], errors="coerce")

    for col in [
        "year", "dry_exposure", "wet_exposure",
        "dry_minus_wet_exposure", "dry_wet_ratio",
    ]:
        if col in alt_wide.columns:
            alt_wide[col] = pd.to_numeric(alt_wide[col], errors="coerce")

    for col in [
        "median_annual_dry_wet_ratio",
        "mean_annual_dry_wet_ratio",
        "fraction_years_dry_exposure_gt_wet",
        "trend_dry_minus_wet_per_decade",
        "trend_pvalue",
        "trend_ci95",
    ]:
        if col in alt_summary.columns:
            alt_summary[col] = pd.to_numeric(alt_summary[col], errors="coerce")

    mat.index = pd.to_numeric(mat.index, errors="coerce").astype(int)
    mat.columns = pd.to_numeric(mat.columns, errors="coerce").astype(int)
    mat = mat.reindex(index=STATE_ORDER, columns=STATE_ORDER)

    return ev, traj, mat, alt_wide, alt_summary


def dry_size_groups(ev):
    dry = ev[ev["first_group"] == "Dry-start"].copy()
    dry = dry.dropna(subset=["object_grid_days", "dry_lifetime_fraction"])
    dry = dry[dry["object_grid_days"] > 0].copy()

    q75 = dry["object_grid_days"].quantile(0.75)
    q90 = dry["object_grid_days"].quantile(0.90)
    q95 = dry["object_grid_days"].quantile(0.95)

    groups = [
        ("All dry-start", dry),
        ("Top 25%\nby size", dry[dry["object_grid_days"] >= q75]),
        ("Top 10%\nby size", dry[dry["object_grid_days"] >= q90]),
        ("Top 5%\nby size", dry[dry["object_grid_days"] >= q95]),
    ]
    return groups


def safe_kde(values, grid):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 5:
        return np.zeros_like(grid)

    if stats is not None and len(np.unique(values)) > 4:
        try:
            kde = stats.gaussian_kde(values)
            return kde(grid)
        except Exception:
            pass

    hist, edges = np.histogram(values, bins=30, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(grid, centers, hist, left=0, right=0)


def draw_vertical_raincloud(
    ax,
    values,
    xpos,
    color,
    width=0.34,
    seed=0,
    label_median=True,
    ylim=(0, 1.08),
):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= ylim[0]) & (vals <= ylim[1])]

    if len(vals) < 5:
        return np.nan

    grid = np.linspace(ylim[0], ylim[1], 260)
    dens = safe_kde(vals, grid)

    if np.nanmax(dens) > 0:
        dens = dens / np.nanmax(dens) * width
    else:
        dens = np.zeros_like(grid)

    ax.fill_betweenx(
        grid,
        xpos,
        xpos + dens,
        color=color,
        alpha=0.25,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        xpos + dens,
        grid,
        color=color,
        lw=1.9,
        zorder=2,
    )

    rng = np.random.default_rng(seed)
    vals_plot = vals
    if len(vals_plot) > 900:
        vals_plot = rng.choice(vals_plot, size=900, replace=False)

    jitter = rng.normal(loc=-0.095, scale=0.040, size=len(vals_plot))
    jitter = np.clip(jitter, -0.22, -0.030)

    ax.scatter(
        np.full(len(vals_plot), xpos) + jitter,
        vals_plot,
        s=18,
        color=color,
        alpha=0.25,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    q25, med, q75 = np.nanpercentile(vals, [25, 50, 75])

    ax.plot(
        [xpos - 0.25, xpos + 0.25],
        [med, med],
        color="black",
        lw=2.4,
        zorder=5,
    )
    ax.plot(
        [xpos, xpos],
        [q25, q75],
        color="black",
        lw=2.1,
        zorder=5,
    )
    ax.scatter(
        xpos,
        med,
        s=78,
        color=color,
        edgecolor="black",
        linewidth=0.75,
        zorder=6,
    )

    if label_median:
        ax.text(
            xpos,
            min(ylim[1] - 0.015, med + 0.085),
            f"{med:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
            color="0.15",
        )

    return med


def draw_log_ratio_raincloud(ax, values, xpos, color, width=0.34, seed=0):
    vals = positive_values(values)
    if len(vals) < 5:
        return np.nan

    z = np.log10(vals)
    z = z[np.isfinite(z)]

    if len(z) < 5:
        return np.nan

    q01, q99 = np.nanpercentile(z, [1, 99])
    pad = max((q99 - q01) * 0.20, 0.25)
    grid = np.linspace(q01 - pad, q99 + pad, 260)

    dens = safe_kde(z, grid)

    if np.nanmax(dens) > 0:
        dens = dens / np.nanmax(dens) * width
    else:
        dens = np.zeros_like(grid)

    ax.fill_betweenx(
        grid,
        xpos,
        xpos + dens,
        color=color,
        alpha=0.25,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        xpos + dens,
        grid,
        color=color,
        lw=1.9,
        zorder=2,
    )

    rng = np.random.default_rng(seed)
    jitter = rng.normal(loc=-0.095, scale=0.040, size=len(z))
    jitter = np.clip(jitter, -0.22, -0.030)

    ax.scatter(
        np.full(len(z), xpos) + jitter,
        z,
        s=22,
        color=color,
        alpha=0.26,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    q25, med, q75 = np.nanpercentile(z, [25, 50, 75])

    ax.plot(
        [xpos - 0.25, xpos + 0.25],
        [med, med],
        color="black",
        lw=2.4,
        zorder=5,
    )
    ax.plot(
        [xpos, xpos],
        [q25, q75],
        color="black",
        lw=2.1,
        zorder=5,
    )
    ax.scatter(
        xpos,
        med,
        s=78,
        color=color,
        edgecolor="black",
        linewidth=0.75,
        zorder=6,
    )

    ax.text(
        xpos,
        med + 0.15,
        f"{10 ** med:.1f}×",
        ha="center",
        va="bottom",
        fontsize=16,
        color="0.15",
    )

    return med


def ratio_tick_label_log10(v):
    raw = 10 ** v
    if np.isclose(raw, 1):
        return "1"
    if raw < 1:
        return f"{raw:.1f}"
    if raw < 10:
        return f"{raw:.0f}"
    if raw < 100:
        return f"{raw:.0f}"
    exp = int(round(np.log10(raw)))
    return rf"$10^{exp}$"


def spearman_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 10:
        return np.nan, np.nan

    if stats is not None:
        res = stats.spearmanr(x[m], y[m])
        return float(res.correlation), float(res.pvalue)

    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]), np.nan


# =============================================================================
# 5. PLOTTING
# =============================================================================

def plot_nature_style_bfixed(ev, traj, mat, alt_wide, alt_summary):
    fig = plt.figure(figsize=(28.0, 20.8))

    gs = GridSpec(
        3,
        2,
        figure=fig,
        left=0.112,
        right=0.982,
        bottom=0.082,
        top=0.948,
        hspace=0.62,
        wspace=0.34,
    )

    brown = GROUP_COLORS["Dry-start"]

    # -------------------------------------------------------------------------
    # a. Dry-start lifetime occupancy
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a", x=-0.18, y=1.055)

    groups = dry_size_groups(ev)

    for i, (name, sub) in enumerate(groups, start=1):
        vals = sub["dry_lifetime_fraction"].replace([np.inf, -np.inf], np.nan).dropna().values
        draw_vertical_raincloud(
            ax=ax,
            values=vals,
            xpos=i,
            color=brown,
            width=0.32,
            seed=100 + i,
            label_median=True,
            ylim=(0, 1.08),
        )

    ax.axhline(0.5, color="0.45", lw=1.25, ls="--")
    ax.text(
        0.985,
        0.51,
        "50%",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=15,
        color="0.35",
    )

    ax.set_xlim(0.55, len(groups) + 0.65)
    ax.set_ylim(-0.02, 1.10)
    ax.set_xticks(np.arange(1, len(groups) + 1))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_title("Dry-state occupancy over dry-start object lifetimes", pad=14)
    set_close_ylabel(ax, "Lifetime S1–S2 fraction", x=-0.075)
    ax.grid(axis="y", color="0.88", linewidth=1.0)
    clean_spines(ax)

    # -------------------------------------------------------------------------
    # b. Heatmap with proper size and dedicated vertical colorbar
    # -------------------------------------------------------------------------
    bgs = gs[0, 1].subgridspec(
        1,
        2,
        width_ratios=[32, 1.25],
        wspace=0.045,
    )

    ax = fig.add_subplot(bgs[0, 0])
    cax = fig.add_subplot(bgs[0, 1])

    # Move b far enough left and above the axis, avoiding title overlap
    add_panel_label(ax, "b", x=-0.27, y=1.055)

    arr = mat.reindex(index=STATE_ORDER, columns=STATE_ORDER).to_numpy(dtype=float)
    vmax = np.nanmax(arr)

    im = ax.imshow(
        arr,
        vmin=0,
        vmax=vmax,
        cmap=get_nature_cmap(),
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(STATE_LABELS)
    ax.set_yticklabels(STATE_LABELS)

    ax.set_xlabel("Lifetime-dominant state", labelpad=8)
    set_close_ylabel(ax, "First-day state", x=-0.095)
    ax.set_title("First-day state vs lifetime-dominant state", pad=14)

    # white grid lines between cells
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)

    for i in range(6):
        for j in range(6):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white" if val > 0.55 * vmax else "black",
                )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(
        "Row-normalized\nevent probability",
        fontsize=20,
        labelpad=10,
    )
    cbar.ax.tick_params(labelsize=18)

    # -------------------------------------------------------------------------
    # c. Lifecycle trajectory
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, "c", x=-0.18, y=1.055)

    tr = traj[traj["first_group"].isin(GROUP_ORDER)].copy()

    for group in GROUP_ORDER:
        sub = tr[tr["first_group"] == group].copy()

        if sub.empty:
            continue

        g = (
            sub.groupby("age_bin")
            .agg(
                age_mid=("age_mid", "mean"),
                median=("dry_fraction", "median"),
                q25=("dry_fraction", lambda x: np.nanpercentile(x, 25)),
                q75=("dry_fraction", lambda x: np.nanpercentile(x, 75)),
            )
            .reset_index()
        )

        ax.fill_between(
            g["age_mid"],
            g["q25"],
            g["q75"],
            color=GROUP_COLORS[group],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            g["age_mid"],
            g["median"],
            color=GROUP_COLORS[group],
            lw=3.1,
            label=GROUP_LABEL[group].replace("\n", " "),
            zorder=3,
        )

    ax.axhline(0.5, color="0.55", lw=1.25, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.04)
    ax.set_title("Dry-state occupancy through normalized event age", pad=14)
    ax.set_xlabel("Normalized event age", labelpad=8)
    set_close_ylabel(ax, "S1–S2 fraction", x=-0.075)
    ax.grid(color="0.88", linewidth=1.0)
    clean_spines(ax)

    leg = ax.legend(
        frameon=True,
        loc="upper right",
        fontsize=17,
        handlelength=2.2,
        borderpad=0.35,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.86)

    # -------------------------------------------------------------------------
    # d. Size dependence
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "d", x=-0.18, y=1.055)

    dry = ev[ev["first_group"] == "Dry-start"].copy()
    dry = dry.dropna(subset=["object_grid_days", "dry_lifetime_fraction"])
    dry = dry[dry["object_grid_days"] > 0].copy()

    try:
        dry["size_bin"] = pd.qcut(dry["object_grid_days"], q=14, duplicates="drop")
    except Exception:
        dry["size_bin"] = pd.cut(dry["object_grid_days"], bins=14)

    rb = (
        dry.groupby("size_bin")
        .agg(
            x=("object_grid_days", "median"),
            med=("dry_lifetime_fraction", "median"),
            q25=("dry_lifetime_fraction", lambda x: np.nanpercentile(x, 25)),
            q75=("dry_lifetime_fraction", lambda x: np.nanpercentile(x, 75)),
        )
        .reset_index(drop=True)
    )

    scatter = dry
    if len(scatter) > 6000:
        scatter = scatter.sample(6000, random_state=77)

    ax.scatter(
        scatter["object_grid_days"],
        scatter["dry_lifetime_fraction"],
        s=13,
        color=brown,
        alpha=0.14,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    ax.fill_between(
        rb["x"],
        rb["q25"],
        rb["q75"],
        color="0.50",
        alpha=0.22,
        linewidth=0,
        label="IQR",
        zorder=2,
    )

    ax.plot(
        rb["x"],
        rb["med"],
        color="black",
        lw=3.1,
        label="Running median",
        zorder=4,
    )

    rho, p = spearman_corr(dry["object_grid_days"], dry["dry_lifetime_fraction"])

    ax.text(
        0.04,
        0.94,
        f"Spearman ρ={rho:.2f}\n{format_p(p)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=17,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=3.0),
    )

    ax.axhline(0.5, color="0.55", lw=1.25, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.04)
    ax.set_title("Size dependence of dry-state retention", pad=14)
    ax.set_xlabel("Object grid-days", labelpad=8)
    set_close_ylabel(ax, "Lifetime S1–S2 fraction", x=-0.075)
    ax.grid(color="0.88", linewidth=1.0, which="both")
    clean_spines(ax)

    leg = ax.legend(
        frameon=True,
        loc="lower left",
        fontsize=17,
        handlelength=2.2,
        borderpad=0.35,
    )
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.86)

    # -------------------------------------------------------------------------
    # e. Alternative definitions
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    add_panel_label(ax, "e", x=-0.18, y=1.055)

    all_z = []

    for i, def_name in enumerate(ALT_DEFS, start=1):
        vals = (
            alt_wide.loc[alt_wide["definition"] == def_name, "dry_wet_ratio"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
        )
        vals = positive_values(vals)

        if len(vals):
            all_z.extend(np.log10(vals).tolist())

        draw_log_ratio_raincloud(
            ax=ax,
            values=vals,
            xpos=i,
            color=brown,
            width=0.31,
            seed=300 + i,
        )

    ax.axhline(0, color="0.55", lw=1.25, ls="--")
    ax.text(
        0.985,
        0.02,
        "ratio = 1",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=15,
        color="0.35",
    )

    if all_z:
        lo = np.nanpercentile(all_z, 1)
        hi = np.nanpercentile(all_z, 99)
        pad = max((hi - lo) * 0.18, 0.25)
        ax.set_ylim(lo - pad, hi + pad)

    raw_ticks = np.array(
        [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
        dtype=float,
    )
    zticks = np.log10(raw_ticks)
    yl = ax.get_ylim()
    keep = (zticks >= yl[0]) & (zticks <= yl[1])
    ax.set_yticks(zticks[keep])
    ax.set_yticklabels([ratio_tick_label_log10(v) for v in zticks[keep]])

    ax.set_xlim(0.55, len(ALT_DEFS) + 0.65)
    ax.set_xticks(np.arange(1, len(ALT_DEFS) + 1))
    ax.set_xticklabels([ALT_LABELS[d] for d in ALT_DEFS], rotation=17, ha="right")
    ax.set_title("Exposure hierarchy under alternative classifications", pad=14)
    set_close_ylabel(ax, "Dry-class / wet-class\nexposure ratio", x=-0.090)
    ax.grid(axis="y", color="0.88", linewidth=1.0)
    clean_spines(ax)

    # -------------------------------------------------------------------------
    # f. Trend robustness
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    add_panel_label(ax, "f", x=-0.18, y=1.055)

    summ = alt_summary.set_index("definition").reindex(ALT_DEFS).reset_index()
    y = np.arange(len(ALT_DEFS))[::-1]

    slopes = pd.to_numeric(summ["trend_dry_minus_wet_per_decade"], errors="coerce").to_numpy()
    ci = pd.to_numeric(summ["trend_ci95"], errors="coerce").to_numpy()

    xmin = np.nanmin(slopes - ci)
    xmax = np.nanmax(slopes + ci)
    xspan = xmax - xmin

    if not np.isfinite(xspan) or xspan <= 0:
        xspan = 1.0

    ax.set_xlim(min(0, xmin - 0.08 * xspan), xmax + 0.32 * xspan)

    for k, yi in enumerate(y):
        if k % 2 == 0:
            ax.axhspan(yi - 0.36, yi + 0.36, color="0.965", zorder=0)

    text_x = xmax + 0.09 * xspan

    for yi, row in zip(y, summ.itertuples(index=False)):
        slope = float(row.trend_dry_minus_wet_per_decade)
        p = float(row.trend_pvalue)
        ci95 = float(row.trend_ci95)

        if np.isfinite(ci95):
            lo = slope - ci95
            hi = slope + ci95

            ax.plot(
                [lo, hi],
                [yi, yi],
                color=brown,
                lw=7.0,
                alpha=0.18,
                solid_capstyle="round",
                zorder=1,
            )

            ax.plot(
                [lo, hi],
                [yi, yi],
                color=brown,
                lw=3.0,
                alpha=0.98,
                solid_capstyle="round",
                zorder=2,
            )

        ax.scatter(
            slope,
            yi,
            s=125,
            color=brown,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
        ax.scatter(
            slope,
            yi,
            s=132,
            facecolor="none",
            edgecolor="black",
            linewidth=0.65,
            zorder=5,
        )

        ax.text(
            text_x,
            yi,
            f"{slope:,.0f}{p_to_star(p)}",
            ha="left",
            va="center",
            fontsize=17,
            color="0.15",
        )

    ax.axvline(0, color="0.55", lw=1.25, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([ALT_LABELS[d] for d in ALT_DEFS])
    ax.set_title("Trend robustness under alternative definitions", pad=14)
    ax.set_xlabel("Dry-class − wet-class grid-days decade$^{-1}$", labelpad=8)
    ax.grid(axis="x", color="0.88", linewidth=1.0)
    clean_spines(ax)

    ax.text(
        text_x,
        len(ALT_DEFS) - 0.35,
        "Slope",
        ha="left",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        color="0.20",
    )

    fig.align_ylabels()

    fig.savefig(
        FIG_PNG,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.34,
    )
    fig.savefig(
        FIG_PDF,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.34,
    )
    plt.close(fig)

    print(f"[SAVED] {FIG_PNG}")
    print(f"[SAVED] {FIG_PDF}")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("[INFO] Replot lifecycle representativeness figure from cached tables only")
    print("[INFO] Panel b fixed: larger heatmap, dedicated vertical colorbar, no title overlap")
    print(f"[INFO] Cache directory : {OUT_DIR}")
    print(f"[INFO] Output PNG      : {FIG_PNG}")
    print(f"[INFO] Output PDF      : {FIG_PDF}")
    print("=" * 100)

    ev, traj, mat, alt_wide, alt_summary = load_cached_tables()

    print("[INFO] Cached event summary rows:", len(ev))
    print("[INFO] Cached trajectory rows   :", len(traj))
    print("[INFO] Alternative definitions  :", alt_wide["definition"].nunique())

    plot_nature_style_bfixed(ev, traj, mat, alt_wide, alt_summary)

    print("=" * 100)
    print("[DONE] Nature-style lifecycle figure regenerated.")
    print("[DONE] No raw event CSVs were reread.")
    print("=" * 100)


if __name__ == "__main__":
    main()