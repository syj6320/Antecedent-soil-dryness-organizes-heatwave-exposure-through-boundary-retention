# -*- coding: utf-8 -*-
"""
Rebuild main-text Figure 2 — Mobility and object geometry

Final panel layout:
a  Event catalogue size
b  Event size–duration structure
c  State-specific object size distribution
d  Moving fraction by initial state
e  Moving-object path length by initial state
f  Mobility–duration structure

Key revision:
- Remove low-information net-displacement panel.
- Remove low-information straightness/support-metric panel.
- Replace them with path-length distribution among moving objects
  and duration–mobility structure.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:
    stats = None

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

CACHE_DIR = Path(r"E:\第二篇数据0427\NCC_rebuilt_Figure1_and_Supplementary")
EVENT_CACHE = CACHE_DIR / "cache_rolling_event_summary.csv"

OUT_DIR = Path(r"E:\第二篇数据0427\Figure2_mobility_object_geometry_main_revised")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NAME = "Figure2_mobility_object_geometry_rebuilt_v4"


# =============================================================================
# STYLE
# =============================================================================

DPI = 500
ROLLING_WINDOW = 7
MOVING_THRESHOLD_KM = 200.0
RNG_SEED = 20260501

STATE_ORDER = [1, 2, 3, 4, 5, 6]
STATE_LABELS = [f"S{i}" for i in STATE_ORDER]

STATE_COLORS = {
    1: "#b07d3c",
    2: "#c7924b",
    3: "#d8be7a",
    4: "#8fc8c0",
    5: "#66aea8",
    6: "#4a948d",
}

GREY_LINE = "#333333"
GREY_LIGHT = "#bfbfbf"
GREY_FILL = "#d9d9d9"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18,
    "axes.titlesize": 19,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 11,
    "axes.linewidth": 1.10,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


# =============================================================================
# HELPERS
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Cannot find any of these columns: {candidates}\n"
            f"Available columns are:\n{list(df.columns)}"
        )
    return None


def add_panel_label(ax, letter: str, x: float = -0.16, y: float = 1.12, size: int = 24) -> None:
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=size,
        fontweight="bold",
        clip_on=False,
        zorder=100,
    )


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def rolling_mean(y, window: int = 7):
    return (
        pd.Series(y, dtype=float)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def lintrend_per_decade(x, y) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 8:
        return np.nan, np.nan

    x = x[m]
    y = y[m]

    if stats is not None:
        res = stats.linregress(x, y)
        return float(res.slope * 10.0), float(res.pvalue)

    coef = np.polyfit(x, y, 1)
    return float(coef[0] * 10.0), np.nan


def p_to_star(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_p_text(p):
    if not np.isfinite(p):
        return "P=NA"
    if p < 0.001:
        return "P<0.001"
    return f"P={p:.3f}"


def spearman_stat(x, y):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    m = x.notna() & y.notna()

    if m.sum() < 8:
        return np.nan, np.nan

    if stats is not None:
        rho, p = stats.spearmanr(x[m], y[m])
        return float(rho), float(p)

    xr = x[m].rank().values
    yr = y[m].rank().values
    rho = np.corrcoef(xr, yr)[0, 1]
    return float(rho), np.nan


def savefig(fig, out_dir: Path, name: str):
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"

    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")


# =============================================================================
# LOAD AND STANDARDIZE DATA
# =============================================================================

def load_event_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find event cache:\n{path}\n"
            "Please run figure1_final_revised_userfix_v3.py first."
        )

    df = pd.read_csv(path, low_memory=False)

    col_event_id = pick_col(df, ["event_uid", "event_id", "uid", "object_id"], required=False)
    col_year = pick_col(df, ["year", "start_year"])
    col_state = pick_col(df, ["start_state", "initial_state", "soil_state", "init_state"])
    col_size = pick_col(df, ["event_grid_days", "object_grid_days", "size_grid_days", "grid_days"])
    col_duration = pick_col(df, ["duration_days", "duration", "event_duration"])
    col_max_area = pick_col(df, ["max_area_gridcells", "max_area", "max_gridcells"])
    col_path = pick_col(df, ["path_length_km", "path_length", "traj_length_km"])
    col_net = pick_col(df, ["net_displacement_km", "net_displacement", "displacement_km"], required=False)
    col_moving = pick_col(df, ["moving_200km", "moving_flag", "moving", "is_moving"], required=False)

    out = pd.DataFrame()

    if col_event_id is not None:
        out["event_id"] = df[col_event_id]
    else:
        out["event_id"] = np.arange(len(df))

    out["year"] = pd.to_numeric(df[col_year], errors="coerce")
    out["state"] = pd.to_numeric(df[col_state], errors="coerce")
    out["size"] = pd.to_numeric(df[col_size], errors="coerce")
    out["duration"] = pd.to_numeric(df[col_duration], errors="coerce")
    out["max_area"] = pd.to_numeric(df[col_max_area], errors="coerce")
    out["path_km"] = pd.to_numeric(df[col_path], errors="coerce")

    if col_net is not None:
        out["net_km"] = pd.to_numeric(df[col_net], errors="coerce")
    else:
        out["net_km"] = np.nan

    if col_moving is not None:
        out["moving_200km"] = pd.to_numeric(df[col_moving], errors="coerce")
    else:
        out["moving_200km"] = np.where(out["path_km"] >= MOVING_THRESHOLD_KM, 1.0, 0.0)

    out = out[out["state"].isin(STATE_ORDER)].copy()
    out["year"] = out["year"].round().astype("Int64")
    out = out[out["year"].between(1950, 2024, inclusive="both")].copy()

    out.loc[out["size"] <= 0, "size"] = np.nan
    out.loc[out["duration"] <= 0, "duration"] = np.nan
    out.loc[out["max_area"] <= 0, "max_area"] = np.nan
    out.loc[out["path_km"] < 0, "path_km"] = np.nan
    out.loc[out["net_km"] < 0, "net_km"] = np.nan

    m = out["path_km"].notna() & out["net_km"].notna() & (out["net_km"] > out["path_km"])
    out.loc[m, "net_km"] = out.loc[m, "path_km"]

    out["moving_200km"] = np.where(out["moving_200km"] > 0, 1.0, 0.0)
    out["state"] = out["state"].astype(int)
    out["year"] = out["year"].astype(int)

    return out


# =============================================================================
# PANEL DATA
# =============================================================================

def build_panel_a_data(ev: pd.DataFrame) -> pd.DataFrame:
    d = (
        ev.groupby("year", as_index=False)["event_id"]
        .nunique()
        .rename(columns={"event_id": "n_objects"})
        .sort_values("year")
    )

    d["rolling"] = rolling_mean(d["n_objects"], ROLLING_WINDOW)
    slope, p = lintrend_per_decade(d["year"], d["n_objects"])

    coef = np.polyfit(d["year"], d["n_objects"], 1)
    d["trend"] = coef[0] * d["year"] + coef[1]
    d["slope_decade"] = slope
    d["trend_p"] = p

    return d


def build_panel_b_data(ev: pd.DataFrame):
    d = ev[["duration", "max_area"]].dropna().copy()
    d["duration_int"] = d["duration"].round().astype(int)
    d = d[d["duration_int"] >= 1].copy()

    grp = (
        d.groupby("duration_int")["max_area"]
        .agg(
            median="median",
            q25=lambda x: np.nanpercentile(x, 25),
            q75=lambda x: np.nanpercentile(x, 75),
            n="count",
        )
        .reset_index()
        .sort_values("duration_int")
    )

    grp = grp[grp["n"] >= 8].copy()

    rho, p = spearman_stat(d["duration"], np.log10(d["max_area"]))
    return d, grp, rho, p


def build_panel_c_data(ev: pd.DataFrame) -> pd.DataFrame:
    d = ev[["state", "size"]].dropna().copy()
    d["log10_size"] = np.log10(d["size"])
    return d


def build_panel_d_data(ev: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, float]]:
    d = (
        ev.groupby(["year", "state"], as_index=False)["moving_200km"]
        .mean()
        .rename(columns={"moving_200km": "moving_fraction"})
        .sort_values(["state", "year"])
    )

    out = []
    slopes = {}

    for s in STATE_ORDER:
        sub = d[d["state"] == s].copy()
        if sub.empty:
            continue

        sub["rolling"] = rolling_mean(sub["moving_fraction"], ROLLING_WINDOW)
        slope, p = lintrend_per_decade(sub["year"], sub["moving_fraction"])

        slopes[s] = {
            "slope_pp_decade": slope * 100.0 if np.isfinite(slope) else np.nan,
            "p": p,
        }
        out.append(sub)

    out = pd.concat(out, ignore_index=True)
    return out, slopes


def moving_subset(ev: pd.DataFrame) -> pd.DataFrame:
    d = ev.copy()

    # Preferred definition: moving_200km.
    m = d["moving_200km"] > 0

    # Fallback if moving_200km is not informative.
    if m.sum() < 20:
        m = d["path_km"] >= MOVING_THRESHOLD_KM

    d = d[m].copy()
    d = d[d["path_km"].notna() & (d["path_km"] > 0)].copy()

    return d


def build_panel_e_data(ev: pd.DataFrame) -> pd.DataFrame:
    d = moving_subset(ev)
    d = d[["state", "path_km"]].dropna().copy()
    d["log10_path_km"] = np.log10(d["path_km"] + 1.0)
    return d


def build_panel_f_data(ev: pd.DataFrame):
    d = moving_subset(ev)
    d = d[["state", "duration", "path_km"]].dropna().copy()
    d = d[(d["duration"] > 0) & (d["path_km"] > 0)].copy()

    d["duration_int"] = d["duration"].round().astype(int)
    d = d[d["duration_int"] >= 1].copy()

    grp = (
        d.groupby("duration_int")["path_km"]
        .agg(
            median="median",
            q25=lambda x: np.nanpercentile(x, 25),
            q75=lambda x: np.nanpercentile(x, 75),
            n="count",
        )
        .reset_index()
        .sort_values("duration_int")
    )

    grp = grp[grp["n"] >= 5].copy()

    rho, p = spearman_stat(d["duration"], np.log10(d["path_km"]))
    return d, grp, rho, p


# =============================================================================
# DRAW PANELS
# =============================================================================

def draw_panel_a(ax, d: pd.DataFrame):
    add_panel_label(ax, "a")

    slope = float(d["slope_decade"].iloc[0])
    p = float(d["trend_p"].iloc[0])

    label = f"Trend {slope:.1f} decade$^{{-1}}${p_to_star(p)}"

    ax.plot(d["year"], d["n_objects"], color=GREY_LIGHT, lw=1.0, alpha=0.9, label="Annual")
    ax.plot(d["year"], d["rolling"], color=GREY_LINE, lw=2.0, label="7-yr running mean")
    ax.plot(d["year"], d["trend"], color="#c07b1f", lw=1.6, ls="--", label=label)

    ax.set_title("Event catalogue size", pad=8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Event objects")
    ax.grid(color="0.88", lw=0.8)
    ax.set_xlim(d["year"].min(), d["year"].max())
    clean_spines(ax)
    ax.legend(frameon=False, loc="upper left")


def draw_panel_b(ax, pts: pd.DataFrame, summ: pd.DataFrame, rho: float, p: float):
    add_panel_label(ax, "b")

    ax.scatter(
        pts["duration"],
        pts["max_area"],
        s=5,
        color="0.72",
        alpha=0.35,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    if not summ.empty:
        ax.fill_between(
            summ["duration_int"].values,
            summ["q25"].values,
            summ["q75"].values,
            color="0.80",
            alpha=0.9,
            zorder=2,
            label="IQR",
        )
        ax.plot(
            summ["duration_int"].values,
            summ["median"].values,
            color=GREY_LINE,
            lw=2.0,
            marker="o",
            ms=3.0,
            zorder=3,
            label="Median",
        )

    ax.text(
        0.05, 0.95,
        f"Spearman ρ={rho:.2f}\n{format_p_text(p)}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
    )

    ax.set_title("Event size–duration structure", pad=8)
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Max area")
    ax.set_yscale("log")
    ax.set_xlim(1, max(30, int(np.nanpercentile(pts["duration"], 99))))
    ax.grid(color="0.88", lw=0.8, which="both")
    clean_spines(ax)
    ax.legend(frameon=False, loc="lower right")


def draw_panel_c(ax, d: pd.DataFrame):
    add_panel_label(ax, "c")

    data_box = [d.loc[d["state"] == s, "log10_size"].dropna().values for s in STATE_ORDER]

    bp = ax.boxplot(
        data_box,
        positions=np.arange(1, 7),
        widths=0.60,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", lw=1.4),
        boxprops=dict(linewidth=1.1),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    for patch, s in zip(bp["boxes"], STATE_ORDER):
        patch.set_facecolor(STATE_COLORS[s])
        patch.set_alpha(0.85)

    ax.set_xticks(np.arange(1, 7))
    ax.set_xticklabels(STATE_LABELS)
    ax.set_title("State-specific object size", pad=8)
    ax.set_xlabel("Initial state")
    ax.set_ylabel(r"$\log_{10}$(size)")
    ax.grid(axis="y", color="0.88", lw=0.8)
    clean_spines(ax)


def draw_panel_d(ax, d: pd.DataFrame, slopes: Dict[int, dict]):
    add_panel_label(ax, "d")

    for s in STATE_ORDER:
        sub = d[d["state"] == s].copy()
        if sub.empty:
            continue

        slope = slopes.get(s, {}).get("slope_pp_decade", np.nan)
        p = slopes.get(s, {}).get("p", np.nan)

        label = f"S{s}"
        if np.isfinite(slope):
            label = f"S{s} ({slope:+.1f} pp decade$^{{-1}}${p_to_star(p)})"

        ax.plot(
            sub["year"],
            sub["rolling"],
            color=STATE_COLORS[s],
            lw=1.8,
            label=label,
        )

    ax.set_title("Moving fraction by initial state", pad=8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction moving")
    ax.set_xlim(d["year"].min(), d["year"].max())
    ax.set_ylim(0, None)
    ax.grid(color="0.88", lw=0.8)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper left")


def draw_panel_e(ax, d: pd.DataFrame):
    add_panel_label(ax, "e")

    if d.empty:
        ax.text(0.5, 0.5, "No moving objects", transform=ax.transAxes,
                ha="center", va="center")
        clean_spines(ax)
        return

    data_box = [d.loc[d["state"] == s, "log10_path_km"].dropna().values for s in STATE_ORDER]

    bp = ax.boxplot(
        data_box,
        positions=np.arange(1, 7),
        widths=0.60,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", lw=1.4),
        boxprops=dict(linewidth=1.1),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    for patch, s in zip(bp["boxes"], STATE_ORDER):
        patch.set_facecolor(STATE_COLORS[s])
        patch.set_alpha(0.85)

    ax.set_xticks(np.arange(1, 7))
    ax.set_xticklabels(STATE_LABELS)
    ax.set_title("Moving-object path length", pad=8)
    ax.set_xlabel("Initial state")
    ax.set_ylabel(r"$\log_{10}$(path length + 1 km)")
    ax.grid(axis="y", color="0.88", lw=0.8)
    clean_spines(ax)


def draw_panel_f(ax, pts: pd.DataFrame, summ: pd.DataFrame, rho: float, p: float):
    add_panel_label(ax, "f")

    if pts.empty:
        ax.text(0.5, 0.5, "No moving objects", transform=ax.transAxes,
                ha="center", va="center")
        clean_spines(ax)
        return

    rng = np.random.default_rng(RNG_SEED)
    if len(pts) > 30000:
        pts_plot = pts.iloc[rng.choice(len(pts), 30000, replace=False)].copy()
    else:
        pts_plot = pts.copy()

    ax.scatter(
        pts_plot["duration"],
        pts_plot["path_km"],
        s=8,
        color="0.65",
        alpha=0.30,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    if not summ.empty:
        ax.fill_between(
            summ["duration_int"].values,
            summ["q25"].values,
            summ["q75"].values,
            color="0.80",
            alpha=0.9,
            zorder=2,
            label="IQR",
        )
        ax.plot(
            summ["duration_int"].values,
            summ["median"].values,
            color=GREY_LINE,
            lw=2.0,
            marker="o",
            ms=3.0,
            zorder=3,
            label="Median",
        )

    ax.text(
        0.05, 0.95,
        f"Spearman ρ={rho:.2f}\n{format_p_text(p)}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
    )

    ax.set_title("Mobility–duration structure", pad=8)
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Path length (km)")
    ax.set_yscale("log")
    ax.set_xlim(1, max(30, int(np.nanpercentile(pts["duration"], 99))))
    ax.grid(color="0.88", lw=0.8, which="both")
    clean_spines(ax)
    ax.legend(frameon=False, loc="lower right")


# =============================================================================
# MAIN
# =============================================================================

def main():
    log("=" * 88)
    log("[INFO] Building revised Figure 2")
    log(f"[INFO] EVENT_CACHE: {EVENT_CACHE}")
    log(f"[INFO] OUT_DIR    : {OUT_DIR}")
    log("=" * 88)

    ev = load_event_cache(EVENT_CACHE)

    log(f"[INFO] Event cache loaded: {ev.shape}")
    log(f"[INFO] Year range: {ev['year'].min()}–{ev['year'].max()}")
    log(f"[INFO] Moving objects: {int((ev['moving_200km'] > 0).sum())}")

    a_df = build_panel_a_data(ev)
    b_pts, b_sum, b_rho, b_p = build_panel_b_data(ev)
    c_df = build_panel_c_data(ev)
    d_df, d_slopes = build_panel_d_data(ev)
    e_df = build_panel_e_data(ev)
    f_pts, f_sum, f_rho, f_p = build_panel_f_data(ev)

    a_df.to_csv(OUT_DIR / "Figure2a_event_catalogue_size.csv", index=False)
    b_pts.to_csv(OUT_DIR / "Figure2b_size_duration_points.csv", index=False)
    b_sum.to_csv(OUT_DIR / "Figure2b_size_duration_summary.csv", index=False)
    c_df.to_csv(OUT_DIR / "Figure2c_state_object_size.csv", index=False)
    d_df.to_csv(OUT_DIR / "Figure2d_moving_fraction_by_state.csv", index=False)
    e_df.to_csv(OUT_DIR / "Figure2e_moving_object_path_length.csv", index=False)
    f_pts.to_csv(OUT_DIR / "Figure2f_mobility_duration_points.csv", index=False)
    f_sum.to_csv(OUT_DIR / "Figure2f_mobility_duration_summary.csv", index=False)

    fig = plt.figure(figsize=(18.0, 10.2))
    gs = fig.add_gridspec(
        2, 3,
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.955,
        wspace=0.30,
        hspace=0.34,
    )

    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])
    axd = fig.add_subplot(gs[1, 0])
    axe = fig.add_subplot(gs[1, 1])
    axf = fig.add_subplot(gs[1, 2])

    draw_panel_a(axa, a_df)
    draw_panel_b(axb, b_pts, b_sum, b_rho, b_p)
    draw_panel_c(axc, c_df)
    draw_panel_d(axd, d_df, d_slopes)
    draw_panel_e(axe, e_df)
    draw_panel_f(axf, f_pts, f_sum, f_rho, f_p)

    savefig(fig, OUT_DIR, OUT_NAME)

    log("=" * 88)
    log("[DONE] Revised Figure 2 completed.")
    log(f"[DONE] Output file: {OUT_DIR / (OUT_NAME + '.png')}")
    log("=" * 88)


if __name__ == "__main__":
    main()