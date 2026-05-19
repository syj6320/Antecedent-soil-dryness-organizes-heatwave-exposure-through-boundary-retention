# -*- coding: utf-8 -*-
"""
Advanced replot: connectivity sensitivity figure

Final visual corrections:
    1. Panel c y-axis tick labels are shortened using 10^4 scaling.
    2. Panel f removes the "C26 baseline" text.
    3. All fonts are enlarged by approximately 5 pt.
    4. All legend fonts are enlarged by approximately 5 pt.
    5. Panel labels a–f remain outside the axes, left of the y-axis title.
    6. No event extraction is rerun; only existing CSV outputs are read.

Inputs:
    E:\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本
      \connectivity_sensitivity_check\dry_wet_exposure_hierarchy

Outputs:
    Figure_connectivity_dry_wet_exposure_hierarchy_final_fontplus.png
    Figure_connectivity_dry_wet_exposure_hierarchy_final_fontplus.pdf
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

try:
    from scipy import stats
except Exception:
    stats = None

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS
# =============================================================================

BASE_ROOT = Path(r"E:\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本")

OUT_DIR = BASE_ROOT / "connectivity_sensitivity_check" / "dry_wet_exposure_hierarchy"

EVENT_LEVEL_CSV = OUT_DIR / "event_level_start_state_by_connectivity.csv"
ANNUAL_STATE_CSV = OUT_DIR / "annual_exposure_hierarchy_by_state.csv"
ANNUAL_GROUP_CSV = OUT_DIR / "annual_exposure_hierarchy_by_group.csv"
ANNUAL_CONTRAST_CSV = OUT_DIR / "annual_dry_wet_exposure_contrast.csv"
SUMMARY_CSV = OUT_DIR / "exposure_hierarchy_summary_by_connectivity.csv"

FIG_PNG = OUT_DIR / "Figure_connectivity_dry_wet_exposure_hierarchy_final_fontplus.png"
FIG_PDF = OUT_DIR / "Figure_connectivity_dry_wet_exposure_hierarchy_final_fontplus.pdf"


# =============================================================================
# 2. CONSTANTS
# =============================================================================

CASE_ORDER = ["C26-current", "C18-no-corner", "C6-face-only"]

STATE_ORDER = [1, 2, 3, 4, 5, 6]
STATE_LABELS = [f"S{i}" for i in STATE_ORDER]

GROUP_ORDER = ["Dry-start", "Intermediate-start", "Wet-start"]
GROUP_LABEL = {
    "Dry-start": "Dry-start\n(S1–S2)",
    "Intermediate-start": "Intermediate\n(S3–S4)",
    "Wet-start": "Wet-start\n(S5–S6)",
}

CASE_COLORS = {
    "C26-current": "#202020",
    "C18-no-corner": "#2166ac",
    "C6-face-only": "#c0392b",
}

DPI = 500
ROLLING_WINDOW = 7


# =============================================================================
# 3. STYLE
# =============================================================================
# Compared with the previous version, most visible text is enlarged by ~5 pt.

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 29,
    "axes.titlesize": 35,
    "axes.labelsize": 33,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 22,
    "axes.linewidth": 1.55,
    "xtick.major.width": 1.35,
    "ytick.major.width": 1.35,
    "xtick.major.size": 7.0,
    "ytick.major.size": 7.0,
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
            f"Missing required file:\n{fp}\n"
            "This script only replots existing CSV outputs. "
            "Please run the dry_wet_exposure_hierarchy script first."
        )


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_close_ylabel(ax, text, x=-0.075):
    ax.set_ylabel(text, labelpad=1)
    ax.yaxis.set_label_coords(x, 0.5)


def rolling_mean(y, window=ROLLING_WINDOW):
    return (
        pd.Series(y, dtype=float)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


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


def format_p_value(p):
    if not np.isfinite(p):
        return "P=NA"
    if p < 1e-4:
        return f"P={p:.1e}"
    return f"P={p:.3g}"


def lintrend_per_decade(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 8:
        return np.nan, np.nan, np.nan, np.nan

    x = x[m]
    y = y[m]

    if np.nanstd(x) == 0:
        return 0.0, np.nan, np.nan, np.nan

    if stats is not None:
        res = stats.linregress(x, y)
        slope_decade = res.slope * 10.0
        se_decade = res.stderr * 10.0 if res.stderr is not None else np.nan
        ci95 = 1.96 * se_decade if np.isfinite(se_decade) else np.nan
        return float(slope_decade), float(res.pvalue), float(res.rvalue), float(ci95)

    X = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = np.corrcoef(x, y)[0, 1]
    return float(beta[1] * 10.0), np.nan, float(r), np.nan


def positive_values(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    return vals


def to_numeric_cols(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def c_axis_formatter_1e4(x, pos):
    """
    Panel c formatter:
        0, 50000, 100000, 150000 -> 0, 5, 10, 15
    The unit is written in the y-axis title as 10^4 grid-days.
    """
    val = x / 1e4
    if abs(val) < 1e-10:
        return "0"
    if abs(val - round(val)) < 1e-8:
        return f"{int(round(val))}"
    return f"{val:.1f}"


def load_all():
    for fp in [
        EVENT_LEVEL_CSV,
        ANNUAL_STATE_CSV,
        ANNUAL_GROUP_CSV,
        ANNUAL_CONTRAST_CSV,
        SUMMARY_CSV,
    ]:
        require_file(fp)

    ev = pd.read_csv(EVENT_LEVEL_CSV, low_memory=False)
    annual_state = pd.read_csv(ANNUAL_STATE_CSV, low_memory=False)
    annual_group = pd.read_csv(ANNUAL_GROUP_CSV, low_memory=False)
    annual_contrast = pd.read_csv(ANNUAL_CONTRAST_CSV, low_memory=False)
    summary = pd.read_csv(SUMMARY_CSV, low_memory=False)

    ev = to_numeric_cols(ev, [
        "year", "start_state", "object_grid_days",
        "duration_days", "max_daily_extent",
        "dry_lifetime_fraction", "wet_lifetime_fraction",
    ])

    annual_state = to_numeric_cols(annual_state, [
        "year", "start_state", "exposure_grid_days",
        "n_events", "median_event_grid_days",
        "p95_event_grid_days", "mean_duration_days",
        "mean_max_daily_extent", "mean_dry_lifetime_fraction",
    ])

    annual_group = to_numeric_cols(annual_group, [
        "year", "exposure_grid_days",
        "n_events", "median_event_grid_days",
        "p95_event_grid_days", "mean_duration_days",
        "mean_max_daily_extent", "mean_dry_lifetime_fraction",
    ])

    annual_contrast = to_numeric_cols(annual_contrast, [
        "year", "dry_exposure", "wet_exposure",
        "intermediate_exposure", "dry_minus_wet_exposure",
        "dry_wet_ratio", "dry_fraction_of_drywet",
        "p95_dry_wet_ratio",
    ])

    summary = to_numeric_cols(summary, [
        "mean_dry_wet_ratio",
        "median_annual_dry_wet_ratio",
        "fraction_years_dry_exposure_gt_wet",
        "trend_dry_minus_wet_per_decade",
        "trend_pvalue",
        "median_p95_event_size_dry_wet_ratio",
    ])

    return ev, annual_state, annual_group, annual_contrast, summary


def draw_bullet_iqr(ax, x_med, x_q25, x_q75, y, color, label=None):
    x_med = float(x_med)
    x_q25 = float(x_q25)
    x_q75 = float(x_q75)

    if not np.isfinite(x_med) or x_med <= 0:
        return

    if not np.isfinite(x_q25) or x_q25 <= 0:
        x_q25 = x_med * 0.65
    if not np.isfinite(x_q75) or x_q75 <= 0:
        x_q75 = x_med * 1.35

    lo = min(x_q25, x_q75)
    hi = max(x_q25, x_q75)

    ax.plot(
        [lo, hi], [y, y],
        color=color,
        lw=10.5,
        alpha=0.19,
        solid_capstyle="round",
        zorder=1,
    )
    ax.plot(
        [lo, hi], [y, y],
        color=color,
        lw=3.1,
        alpha=0.98,
        solid_capstyle="round",
        zorder=2,
    )
    ax.vlines(
        [lo, hi],
        y - 0.065, y + 0.065,
        color=color,
        lw=1.9,
        alpha=0.98,
        zorder=2,
    )
    ax.scatter(
        x_med, y,
        s=112,
        color=color,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
        label=label,
    )


def draw_log_raincloud(ax, vals, xpos, color, width=0.35, seed=0):
    vals = positive_values(vals)

    if len(vals) < 5:
        return np.nan, np.nan, np.nan

    z = np.log10(vals)
    z = z[np.isfinite(z)]

    if len(z) < 5:
        return np.nan, np.nan, np.nan

    q01, q99 = np.nanpercentile(z, [1, 99])
    pad = max((q99 - q01) * 0.20, 0.20)
    grid = np.linspace(q01 - pad, q99 + pad, 260)

    if stats is not None and len(np.unique(z)) > 3:
        kde = stats.gaussian_kde(z)
        dens = kde(grid)
    else:
        hist, edges = np.histogram(z, bins=25, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dens = np.interp(grid, centers, hist, left=0, right=0)

    if np.nanmax(dens) > 0:
        dens = dens / np.nanmax(dens) * width
    else:
        dens = np.zeros_like(grid)

    ax.fill_betweenx(
        grid,
        xpos,
        xpos + dens,
        color=color,
        alpha=0.23,
        linewidth=0,
        zorder=1,
    )

    ax.plot(
        xpos + dens,
        grid,
        color=color,
        lw=2.05,
        alpha=0.98,
        zorder=2,
    )

    rng = np.random.default_rng(seed)
    jitter = rng.normal(loc=-0.105, scale=0.038, size=len(z))
    jitter = np.clip(jitter, -0.21, -0.035)

    ax.scatter(
        np.full(len(z), xpos) + jitter,
        z,
        s=34,
        color=color,
        alpha=0.34,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    q25, med, q75 = np.nanpercentile(z, [25, 50, 75])

    ax.plot(
        [xpos - 0.26, xpos + 0.26],
        [med, med],
        color="black",
        lw=2.9,
        zorder=5,
    )
    ax.plot(
        [xpos, xpos],
        [q25, q75],
        color="black",
        lw=2.6,
        zorder=5,
    )
    ax.scatter(
        xpos, med,
        s=105,
        color=color,
        edgecolor="black",
        linewidth=0.85,
        zorder=6,
    )

    return q25, med, q75


def ratio_tick_label(v):
    raw = 10 ** v

    if np.isclose(raw, 1.0):
        return "1"
    if raw < 1:
        return f"{raw:.1f}"
    if raw < 10:
        return f"{raw:.0f}"
    if raw < 100:
        return f"{raw:.0f}"
    exp = int(round(np.log10(raw)))
    return rf"$10^{exp}$"


def build_robustness_fingerprint(annual_state, annual_contrast, ev):
    rows = []

    for case in CASE_ORDER:
        ssub = annual_state[annual_state["connectivity_case"] == case].copy()
        state_med = (
            ssub.groupby("start_state")["exposure_grid_days"]
            .median()
            .reindex(STATE_ORDER)
        )
        s1 = state_med.loc[1]
        s6 = state_med.loc[6]

        s1_s6_ratio = np.nan
        if np.isfinite(s1) and np.isfinite(s6) and s6 > 0:
            s1_s6_ratio = s1 / s6

        csub = annual_contrast[annual_contrast["connectivity_case"] == case].copy()
        dry_wet_ratio = (
            csub["dry_wet_ratio"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .median()
        )

        slope, p, r, ci95 = lintrend_per_decade(
            csub["year"],
            csub["dry_minus_wet_exposure"],
        )

        esub = ev[ev["connectivity_case"] == case].copy()

        dry_vals = positive_values(
            esub.loc[esub["start_group"] == "Dry-start", "object_grid_days"].values
        )
        wet_vals = positive_values(
            esub.loc[esub["start_group"] == "Wet-start", "object_grid_days"].values
        )

        upper_tail_ratio = np.nan
        if len(dry_vals) >= 10 and len(wet_vals) >= 10:
            dry_p95 = np.nanpercentile(dry_vals, 95)
            wet_p95 = np.nanpercentile(wet_vals, 95)
            if np.isfinite(wet_p95) and wet_p95 > 0:
                upper_tail_ratio = dry_p95 / wet_p95

        rows.extend([
            {
                "connectivity_case": case,
                "metric": "S1/S6 exposure",
                "value": s1_s6_ratio,
            },
            {
                "connectivity_case": case,
                "metric": "Dry/wet exposure",
                "value": dry_wet_ratio,
            },
            {
                "connectivity_case": case,
                "metric": "Dry−wet trend",
                "value": slope,
            },
            {
                "connectivity_case": case,
                "metric": "Upper-tail dry/wet",
                "value": upper_tail_ratio,
            },
        ])

    fp = pd.DataFrame(rows)

    baseline = (
        fp[fp["connectivity_case"] == "C26-current"]
        .set_index("metric")["value"]
        .to_dict()
    )

    fp["relative_to_C26"] = np.nan
    for i, r in fp.iterrows():
        b = baseline.get(r["metric"], np.nan)
        v = r["value"]
        if np.isfinite(b) and b != 0 and np.isfinite(v):
            fp.loc[i, "relative_to_C26"] = v / b

    return fp


def add_panel_labels_left_of_ylabels(fig, axes, letters):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    for ax, letter in zip(axes, letters):
        bbox = ax.get_position()

        ylab = ax.yaxis.get_label()
        x_candidates = []

        if ylab is not None and ylab.get_text():
            ext = ylab.get_window_extent(renderer=renderer)
            p0 = inv.transform((ext.x0, ext.y0))
            x_candidates.append(p0[0])

        for tick in ax.yaxis.get_major_ticks():
            lab = tick.label1
            if lab is not None and lab.get_text():
                ext = lab.get_window_extent(renderer=renderer)
                p0 = inv.transform((ext.x0, ext.y0))
                x_candidates.append(p0[0])

        if len(x_candidates) > 0:
            x = min(x_candidates) - 0.014
        else:
            x = bbox.x0 - 0.050

        y = bbox.y1 + 0.008

        fig.text(
            x,
            y,
            letter,
            fontsize=40,
            fontweight="bold",
            ha="right",
            va="bottom",
        )


# =============================================================================
# 5. PLOTTING
# =============================================================================

def plot_figure(ev, annual_state, annual_group, annual_contrast, summary):
    fig = plt.figure(figsize=(29.5, 22.0))

    gs = GridSpec(
        3, 2,
        figure=fig,
        left=0.135,
        right=0.982,
        bottom=0.082,
        top=0.948,
        hspace=0.66,
        wspace=0.36,
    )

    axes_for_labels = []

    # -------------------------------------------------------------------------
    # a. S1-S6 exposure hierarchy
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    axes_for_labels.append(ax)

    ymin_list, ymax_list = [], []

    for case in CASE_ORDER:
        sub = annual_state[annual_state["connectivity_case"] == case].copy()

        med = (
            sub.groupby("start_state")["exposure_grid_days"]
            .median()
            .reindex(STATE_ORDER)
        )
        q25 = (
            sub.groupby("start_state")["exposure_grid_days"]
            .quantile(0.25)
            .reindex(STATE_ORDER)
        )
        q75 = (
            sub.groupby("start_state")["exposure_grid_days"]
            .quantile(0.75)
            .reindex(STATE_ORDER)
        )

        x = np.arange(1, 7)
        y = med.to_numpy(dtype=float)
        y25 = q25.to_numpy(dtype=float)
        y75 = q75.to_numpy(dtype=float)

        y[y <= 0] = np.nan
        y25[y25 <= 0] = np.nan
        y75[y75 <= 0] = np.nan

        ymin_list.extend(y25[np.isfinite(y25)].tolist())
        ymax_list.extend(y75[np.isfinite(y75)].tolist())

        ax.plot(
            x,
            y,
            marker="o",
            lw=3.5,
            ms=9.8,
            color=CASE_COLORS[case],
            label=case,
            zorder=3,
        )

        ax.fill_between(
            x,
            y25,
            y75,
            color=CASE_COLORS[case],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

    ax.set_yscale("log")
    if ymin_list and ymax_list:
        ax.set_ylim(max(np.nanmin(ymin_list) * 0.55, 1), np.nanmax(ymax_list) * 1.65)

    ax.set_xticks(np.arange(1, 7))
    ax.set_xticklabels(STATE_LABELS)
    ax.set_title("S1–S6 exposure hierarchy", pad=16)
    ax.set_xlabel("Initial soil-moisture state", labelpad=10)
    set_close_ylabel(ax, "Median annual\nobject grid-days", x=-0.070)

    ax.grid(color="0.88", linewidth=1.05, which="both")
    clean_spines(ax)
    ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=21,
        handlelength=2.2,
    )

    # -------------------------------------------------------------------------
    # b. Advanced horizontal median-IQR
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    axes_for_labels.append(ax)

    ybase = np.arange(len(GROUP_ORDER))[::-1]
    offsets = {
        "C26-current": -0.24,
        "C18-no-corner": 0.0,
        "C6-face-only": 0.24,
    }

    for i, y0 in enumerate(ybase):
        if i % 2 == 0:
            ax.axhspan(y0 - 0.43, y0 + 0.43, color="0.965", zorder=0)

    for case in CASE_ORDER:
        sub = annual_group[annual_group["connectivity_case"] == case].copy()

        med = (
            sub.groupby("start_group")["exposure_grid_days"]
            .median()
            .reindex(GROUP_ORDER)
        )
        q25 = (
            sub.groupby("start_group")["exposure_grid_days"]
            .quantile(0.25)
            .reindex(GROUP_ORDER)
        )
        q75 = (
            sub.groupby("start_group")["exposure_grid_days"]
            .quantile(0.75)
            .reindex(GROUP_ORDER)
        )

        for j, group in enumerate(GROUP_ORDER):
            label = case if j == 0 else None
            draw_bullet_iqr(
                ax=ax,
                x_med=med.loc[group],
                x_q25=q25.loc[group],
                x_q75=q75.loc[group],
                y=ybase[j] + offsets[case],
                color=CASE_COLORS[case],
                label=label,
            )

    ax.set_xscale("log")
    ax.set_yticks(ybase)
    ax.set_yticklabels([GROUP_LABEL[g] for g in GROUP_ORDER])
    ax.tick_params(axis="y", pad=8)
    ax.set_title("Dry-start vs wet-start exposure", pad=16)
    ax.set_xlabel("Median annual object grid-days", labelpad=10)
    ax.grid(axis="x", color="0.88", linewidth=1.05, which="both")
    clean_spines(ax)

    leg = ax.legend(
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(0.025, 0.980),
        fontsize=20,
        handlelength=1.6,
        borderpad=0.35,
        labelspacing=0.30,
        handletextpad=0.45,
    )
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.88)

    # -------------------------------------------------------------------------
    # c. Annual dry-minus-wet exposure contrast
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    axes_for_labels.append(ax)

    for case in CASE_ORDER:
        sub = (
            annual_contrast[annual_contrast["connectivity_case"] == case]
            .sort_values("year")
            .copy()
        )

        slope, p, r, ci95 = lintrend_per_decade(
            sub["year"],
            sub["dry_minus_wet_exposure"],
        )

        ax.plot(
            sub["year"],
            rolling_mean(sub["dry_minus_wet_exposure"]),
            color=CASE_COLORS[case],
            lw=3.35,
            label=f"{case}: {slope:.0f}{p_to_star(p)} decade$^{{-1}}$",
        )

    ax.axhline(0, color="0.55", lw=1.25, ls="--")
    ax.set_title("Annual dry-minus-wet exposure contrast", pad=16)
    ax.set_xlabel("Year", labelpad=10)

    # key correction: shorter tick labels, unit written in axis title
    set_close_ylabel(ax, "Dry-start − wet-start\nobject grid-days ($10^4$)", x=-0.082)
    ax.yaxis.set_major_formatter(FuncFormatter(c_axis_formatter_1e4))
    ax.set_yticks([0, 5e4, 1e5, 1.5e5])

    ax.grid(color="0.88", linewidth=1.05)
    clean_spines(ax)
    ax.legend(
        frameon=False,
        fontsize=20,
        loc="upper left",
        handlelength=2.2,
    )

    # -------------------------------------------------------------------------
    # d. Advanced log-ratio raincloud
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    axes_for_labels.append(ax)

    all_log_vals = []

    for i, case in enumerate(CASE_ORDER, start=1):
        vals = (
            annual_contrast.loc[
                annual_contrast["connectivity_case"] == case,
                "dry_wet_ratio",
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
        )
        vals = positive_values(vals)

        if len(vals) > 0:
            all_log_vals.extend(np.log10(vals).tolist())

        q25, med, q75 = draw_log_raincloud(
            ax=ax,
            vals=vals,
            xpos=i,
            color=CASE_COLORS[case],
            width=0.35,
            seed=100 + i,
        )

        if np.isfinite(med):
            ax.text(
                i,
                med + 0.13,
                f"{10 ** med:.1f}×",
                ha="center",
                va="bottom",
                fontsize=21,
                color="0.15",
            )

    ax.axhline(0, color="0.50", lw=1.25, ls="--", zorder=0)
    ax.text(
        0.985,
        0.020,
        "ratio = 1",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=20,
        color="0.35",
    )

    ax.set_xlim(0.45, len(CASE_ORDER) + 0.55)

    if all_log_vals:
        lo = np.nanpercentile(all_log_vals, 1)
        hi = np.nanpercentile(all_log_vals, 99)
        pad = max((hi - lo) * 0.16, 0.25)
        ax.set_ylim(lo - pad, hi + pad)

    raw_ticks = np.array(
        [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
        dtype=float,
    )
    zticks = np.log10(raw_ticks)
    yl = ax.get_ylim()
    keep = (zticks >= yl[0]) & (zticks <= yl[1])
    ax.set_yticks(zticks[keep])
    ax.set_yticklabels([ratio_tick_label(v) for v in zticks[keep]])

    ax.set_xticks(np.arange(1, len(CASE_ORDER) + 1))
    ax.set_xticklabels(CASE_ORDER, rotation=12, ha="right")
    ax.set_title("Annual dry/wet exposure ratio", pad=16)
    set_close_ylabel(ax, "Exposure ratio\n(dry-start / wet-start)", x=-0.090)

    ax.grid(axis="y", color="0.88", linewidth=1.05)
    clean_spines(ax)

    # -------------------------------------------------------------------------
    # e. Event-size CCDF
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    axes_for_labels.append(ax)

    for case in CASE_ORDER:
        for group, ls, alpha, lw in [
            ("Dry-start", "-", 0.96, 2.75),
            ("Wet-start", "--", 0.72, 2.45),
        ]:
            vals = (
                ev.loc[
                    (ev["connectivity_case"] == case) &
                    (ev["start_group"] == group),
                    "object_grid_days",
                ]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
            )

            vals = positive_values(vals)

            if len(vals) < 10:
                continue

            vals = np.sort(vals)
            ccdf = 1.0 - np.arange(1, len(vals) + 1) / len(vals)
            ccdf = np.maximum(ccdf, 1.0 / len(vals))

            ax.plot(
                vals,
                ccdf,
                lw=lw,
                ls=ls,
                color=CASE_COLORS[case],
                alpha=alpha,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Event-size upper tail", pad=16)
    ax.set_xlabel("Object grid-days", labelpad=10)
    set_close_ylabel(ax, "Exceedance probability", x=-0.078)

    ax.grid(color="0.88", linewidth=1.0, which="both")
    clean_spines(ax)

    color_handles = [
        Line2D([0], [0], color=CASE_COLORS[c], lw=3.1, label=c)
        for c in CASE_ORDER
    ]
    leg1 = ax.legend(
        handles=color_handles,
        frameon=False,
        loc="lower left",
        fontsize=20,
        handlelength=2.2,
        borderaxespad=0.3,
    )
    ax.add_artist(leg1)

    ax.text(
        0.985,
        0.965,
        "solid: dry-start\nDashed: wet-start",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=20,
        color="0.18",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=3.0),
    )

    # -------------------------------------------------------------------------
    # f. Connectivity robustness fingerprint
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    axes_for_labels.append(ax)

    fp = build_robustness_fingerprint(annual_state, annual_contrast, ev)

    metric_order = [
        "S1/S6 exposure",
        "Dry/wet exposure",
        "Dry−wet trend",
        "Upper-tail dry/wet",
    ]
    metric_labels = {
        "S1/S6 exposure": "S1/S6\nexposure",
        "Dry/wet exposure": "Dry/wet\nexposure",
        "Dry−wet trend": "Dry−wet\ntrend",
        "Upper-tail dry/wet": "Upper-tail\ndry/wet",
    }

    y_positions = np.arange(len(metric_order))[::-1]

    for k, y0 in enumerate(y_positions):
        if k % 2 == 0:
            ax.axhspan(y0 - 0.38, y0 + 0.38, color="0.965", zorder=0)

    for y0, metric in zip(y_positions, metric_order):
        sub = fp[fp["metric"] == metric].copy()

        xs = []
        for case in CASE_ORDER:
            val = sub.loc[
                sub["connectivity_case"] == case,
                "relative_to_C26",
            ]
            if len(val) == 0:
                xs.append(np.nan)
            else:
                xs.append(float(val.iloc[0]))

        xs_arr = np.asarray(xs, dtype=float)
        finite = np.isfinite(xs_arr)

        if finite.sum() >= 2:
            ax.plot(
                [np.nanmin(xs_arr), np.nanmax(xs_arr)],
                [y0, y0],
                color="0.70",
                lw=6.2,
                alpha=0.35,
                solid_capstyle="round",
                zorder=1,
            )

        for case, xval in zip(CASE_ORDER, xs):
            if not np.isfinite(xval):
                continue

            ax.scatter(
                xval,
                y0,
                s=140,
                color=CASE_COLORS[case],
                edgecolor="white",
                linewidth=1.0,
                zorder=4,
                label=case if metric == metric_order[0] else None,
            )

            if case != "C26-current":
                ax.text(
                    xval,
                    y0 + 0.18,
                    f"{xval:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=19,
                    color="0.15",
                )

    # keep the reference line but remove the "C26 baseline" text
    ax.axvline(
        1.0,
        color="0.25",
        lw=1.45,
        ls="--",
        zorder=0,
    )

    rel_vals = fp["relative_to_C26"].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(rel_vals) > 0:
        xmin = min(0.88, np.nanmin(rel_vals) - 0.05)
        xmax = max(1.12, np.nanmax(rel_vals) + 0.05)
    else:
        xmin, xmax = 0.88, 1.12

    ax.set_xlim(xmin, xmax)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([metric_labels[m] for m in metric_order])
    ax.tick_params(axis="y", pad=8)
    ax.set_ylim(-0.55, len(metric_order) - 0.45)
    ax.set_title("Connectivity robustness fingerprint", pad=16)
    ax.set_xlabel(
        "Relative effect size under alternative connectivity\n(C26-current = 1)",
        labelpad=10,
    )
    ax.grid(axis="x", color="0.88", linewidth=1.05)
    clean_spines(ax)

    leg = ax.legend(
        frameon=True,
        loc="lower right",
        fontsize=20,
        handlelength=1.3,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.45,
    )
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.88)

    # -------------------------------------------------------------------------
    # Panel labels outside y-axis titles
    # -------------------------------------------------------------------------
    add_panel_labels_left_of_ylabels(
        fig,
        axes_for_labels,
        ["a", "b", "c", "d", "e", "f"],
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
    print("[INFO] Final font-enlarged replot of connectivity sensitivity figure")
    print(f"[INFO] Input directory : {OUT_DIR}")
    print(f"[INFO] Output PNG      : {FIG_PNG}")
    print(f"[INFO] Output PDF      : {FIG_PDF}")
    print("=" * 100)

    ev, annual_state, annual_group, annual_contrast, summary = load_all()

    print("\n===== EXISTING EXPOSURE HIERARCHY SUMMARY =====")
    display_cols = [
        "connectivity_case",
        "mean_dry_wet_ratio",
        "median_annual_dry_wet_ratio",
        "fraction_years_dry_exposure_gt_wet",
        "trend_dry_minus_wet_per_decade",
        "trend_pvalue",
        "median_p95_event_size_dry_wet_ratio",
        "dry_start_hierarchy_retained",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[display_cols].to_string(index=False))

    plot_figure(ev, annual_state, annual_group, annual_contrast, summary)

    print("=" * 100)
    print("[DONE] Final font-enlarged figure regenerated. No event CSVs were rebuilt.")
    print("=" * 100)


if __name__ == "__main__":
    main()