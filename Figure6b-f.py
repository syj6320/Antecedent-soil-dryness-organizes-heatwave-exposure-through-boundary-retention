# -*- coding: utf-8 -*-
"""
Figure 5 -- template-matched compact transition-kernel layout
==============================================================

This version follows the uploaded template layout:

Top row:
    a Front transition kernel
    b Interior/local transition kernel + probability colorbar close to b
    c Front - interior/local kernel + difference colorbar close to c

Bottom row:
    d Source-state retention
    e Transition breadth
    f Two compact flow panels with legend directly below panel f

Main corrections
----------------
1. Manual axes placement replaces GridSpec to avoid unstable spacing.
2. All ordinary visible text is set to 26 pt.
3. Panel letters are set to 34 pt.
4. Heatmap cells are enlarged so 26 pt cell labels do not overlap.
5. Probability colorbar is placed close to panel b.
6. Difference colorbar is placed close to panel c.
7. Panel f legend is anchored to panel f, not to the full figure.
8. Bottom-row titles no longer collide with top-row x labels.
9. Flow-panel summary text no longer overlaps the legend.
10. Output filenames are unchanged.

Input
-----
<root>\\_result3_transition_kernel\\result3_annual_counts.csv

Required columns
----------------
year, kernel, from_state, to_state, count
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS AND CONSTANTS
# =============================================================================

DEFAULT_ROOT = Path(
    r"E:\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本"
) / "events_cc3d_with_precip_H_LE_CAPE_IVTDIV_T850_WIND_RH"

DEFAULT_OUT_BASE = Path(
    r"D:\第二篇\第二篇最终20260407版本\最终20260428\最终版本代码"
)

OUT_SUBDIR = "Figure5_baseline_front_local_transition_NCC_rebuilt_final"

STATE_ORDER = [1, 2, 3, 4, 5, 6]
STATE_LABELS = [f"S{i}" for i in STATE_ORDER]

STATE_COLORS = {
    1: "#9b5a08",
    2: "#c8872d",
    3: "#e3c77d",
    4: "#7fcdbb",
    5: "#2f9c95",
    6: "#006c5b",
}

LOCAL_COLOR = "#2166ac"
FRONT_COLOR = "#b2182b"

PROB_CMAP = LinearSegmentedColormap.from_list(
    "transition_probability",
    ["#ffffe5", "#d9f0a3", "#78c679", "#2c7fb8", "#08306b"],
    N=256,
)

DIFF_CMAP = LinearSegmentedColormap.from_list(
    "front_minus_local",
    ["#2166ac", "#67a9cf", "#f7f7f7", "#f4a582", "#b2182b"],
    N=256,
)

DPI = 500
FLOW_MIN_PROB = 0.050
N_BOOT = 800
BOOT_SEED = 123

FONT = 26
PANEL_FONT = 34

# Large canvas is necessary because the user requested 26-pt text everywhere.
# The final visual structure follows the supplied template.
FIGSIZE = (36.0, 17.2)


# =============================================================================
# 2. MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT,
    "axes.titlesize": FONT,
    "axes.labelsize": FONT,
    "xtick.labelsize": FONT,
    "ytick.labelsize": FONT,
    "legend.fontsize": FONT,
    "axes.linewidth": 1.25,
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


# =============================================================================
# 3. BASIC UTILITIES
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {desc}:\n{path}\n\n"
            "This script reads existing transition-kernel caches. "
            "Please run the transition-kernel cache-building script first."
        )


def add_panel_label(ax, letter: str, x: float = -0.18, y: float = 1.13) -> None:
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_FONT,
        fontweight="bold",
        clip_on=False,
        zorder=100,
    )


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fmt_support(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1000:
        return f"{x / 1000:.0f}k"
    return str(int(x))


def state_to_int_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return pd.to_numeric(
            s.astype(str).str.replace("S", "", regex=False),
            errors="coerce",
        )
    return pd.to_numeric(s, errors="coerce")


def save_figure(fig, out_dir: Path, name: str) -> None:
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    svg = out_dir / f"{name}.svg"

    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")
    log(f"[SAVED] {svg}")


# =============================================================================
# 4. LOAD AND PREPARE DATA
# =============================================================================

def prep_annual_counts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.rename(columns={c: str(c).strip() for c in d.columns})

    required = ["year", "kernel", "from_state", "to_state", "count"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(
            f"result3_annual_counts.csv is missing required columns: {missing}\n"
            f"Available columns: {list(d.columns)}"
        )

    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["kernel"] = d["kernel"].astype(str).str.lower().str.strip()
    d["from_state"] = state_to_int_series(d["from_state"])
    d["to_state"] = state_to_int_series(d["to_state"])
    d["count"] = pd.to_numeric(d["count"], errors="coerce")

    d = d.dropna(subset=["year", "kernel", "from_state", "to_state", "count"]).copy()
    d["year"] = d["year"].astype(int)
    d["from_state"] = d["from_state"].astype(int)
    d["to_state"] = d["to_state"].astype(int)

    d = d[
        d["kernel"].isin(["front", "local"])
        & d["from_state"].between(1, 6)
        & d["to_state"].between(1, 6)
        & (d["count"] >= 0)
    ].copy()

    if d.empty:
        raise ValueError(
            "No valid transition records remain after filtering. "
            "Check kernel/from_state/to_state/count columns."
        )

    return d


def load_annual_counts(root: Path) -> pd.DataFrame:
    fp = root / "_result3_transition_kernel" / "result3_annual_counts.csv"
    require_file(fp, "transition-kernel annual counts")

    log(f"[INFO] Reading transition counts: {fp}")
    annual = pd.read_csv(fp, low_memory=False)
    annual = prep_annual_counts(annual)

    log(f"[INFO] Valid transition rows: {len(annual):,}")
    log(
        f"[INFO] Years: {annual['year'].min()} - {annual['year'].max()}; "
        f"kernels: {sorted(annual['kernel'].unique())}"
    )

    return annual


def subset_years(
    df: pd.DataFrame,
    year_min: Optional[int],
    year_max: Optional[int],
) -> pd.DataFrame:
    d = df.copy()
    if year_min is not None:
        d = d[d["year"] >= int(year_min)]
    if year_max is not None:
        d = d[d["year"] <= int(year_max)]
    return d.copy()


# =============================================================================
# 5. MATRICES AND SUMMARY METRICS
# =============================================================================

def counts_cube_for_kernel(
    annual: pd.DataFrame,
    kernel: str,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    d = annual[annual["kernel"] == kernel].copy()
    d = subset_years(d, year_min, year_max)

    years = np.array(sorted(d["year"].unique()), dtype=int)
    if len(years) == 0:
        raise ValueError(f"No records for kernel={kernel} in selected years.")

    y_index = {y: i for i, y in enumerate(years)}
    cube = np.zeros((len(years), 6, 6), dtype=float)

    g = (
        d.groupby(["year", "from_state", "to_state"], as_index=False)["count"]
        .sum()
    )

    for _, row in g.iterrows():
        yi = y_index[int(row["year"])]
        i = int(row["from_state"]) - 1
        j = int(row["to_state"]) - 1
        cube[yi, i, j] += float(row["count"])

    return years, cube


def counts_to_prob_matrix(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    support = counts.sum(axis=1)

    mat = np.full((6, 6), np.nan, dtype=float)
    for i in range(6):
        if support[i] > 0:
            mat[i, :] = counts[i, :] / support[i]

    return mat, support


def climatology_matrix_from_cube(cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(cube, dtype=float).sum(axis=0)
    return counts_to_prob_matrix(counts)


def transition_breadth_by_source(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    out = np.full(6, np.nan, dtype=float)

    targets = np.array(STATE_ORDER, dtype=float)

    for i, fs in enumerate(STATE_ORDER):
        row = mat[i, :]
        if np.all(~np.isfinite(row)):
            continue
        out[i] = np.nansum(row * np.abs(targets - fs))

    return out


def retention_by_source(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    return np.diag(mat).astype(float)


def weighted_summary(mat: np.ndarray, support: np.ndarray) -> Dict[str, float]:
    mat = np.asarray(mat, dtype=float)
    support = np.asarray(support, dtype=float)

    total = np.nansum(support)
    if total <= 0:
        return {
            "total_support": np.nan,
            "weighted_retention": np.nan,
            "weighted_breadth": np.nan,
        }

    w = support / total
    retention = retention_by_source(mat)
    breadth = transition_breadth_by_source(mat)

    return {
        "total_support": float(total),
        "weighted_retention": float(np.nansum(retention * w)),
        "weighted_breadth": float(np.nansum(breadth * w)),
    }


def bootstrap_source_metric(
    cube: np.ndarray,
    metric: str,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    cube = np.asarray(cube, dtype=float)
    n_year = cube.shape[0]

    full_mat, _ = climatology_matrix_from_cube(cube)

    if metric == "retention":
        full_val = retention_by_source(full_mat)
    elif metric == "breadth":
        full_val = transition_breadth_by_source(full_mat)
    else:
        raise ValueError("metric must be 'retention' or 'breadth'.")

    boot_values = np.full((n_boot, 6), np.nan, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n_year, size=n_year)
        boot_counts = cube[idx, :, :].sum(axis=0)
        boot_mat, _ = counts_to_prob_matrix(boot_counts)

        if metric == "retention":
            boot_values[b, :] = retention_by_source(boot_mat)
        else:
            boot_values[b, :] = transition_breadth_by_source(boot_mat)

    lo = np.nanpercentile(boot_values, 2.5, axis=0)
    hi = np.nanpercentile(boot_values, 97.5, axis=0)

    return pd.DataFrame({
        "source_state": STATE_ORDER,
        "value": full_val,
        "lo": lo,
        "hi": hi,
    })


def edge_table_from_matrix(
    mat: np.ndarray,
    support: np.ndarray,
    kernel: str,
    flow_min_prob: float,
) -> pd.DataFrame:

    rows = []
    for i, fs in enumerate(STATE_ORDER):
        for j, ts in enumerate(STATE_ORDER):
            p = float(mat[i, j]) if np.isfinite(mat[i, j]) else np.nan
            rows.append({
                "kernel": kernel,
                "from_state": fs,
                "to_state": ts,
                "probability": p,
                "from_state_support": float(support[i]),
                "shown_in_flow": bool(np.isfinite(p) and p >= flow_min_prob),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 6. HEATMAPS
# =============================================================================

def annotate_heatmap(
    ax,
    mat: np.ndarray,
    cmap,
    norm,
    fmt: str,
    fontsize: float = FONT,
) -> None:

    arr = np.asarray(mat, dtype=float)

    for i in range(6):
        for j in range(6):
            val = arr[i, j]
            if not np.isfinite(val):
                continue

            rgba = cmap(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.42 else "black"

            ax.text(
                j, i, fmt.format(val),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )


def plot_heatmap(
    ax,
    mat: np.ndarray,
    title: str,
    cmap,
    norm,
    fmt: str,
    show_ylabel: bool = True,
):

    im = ax.imshow(
        mat,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(STATE_LABELS, fontsize=FONT)
    ax.set_yticklabels(STATE_LABELS, fontsize=FONT)

    ax.set_xlabel("Target state", fontsize=FONT, labelpad=8)
    if show_ylabel:
        ax.set_ylabel("Source state", fontsize=FONT, labelpad=10)
    else:
        ax.set_ylabel("")

    ax.set_title(title, pad=16, fontsize=FONT, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", labelsize=FONT, pad=5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    annotate_heatmap(ax, mat, cmap=cmap, norm=norm, fmt=fmt, fontsize=FONT)

    return im


# =============================================================================
# 7. ROW-WISE SUMMARY PANELS
# =============================================================================

def plot_metric_panel(
    ax,
    local_df: pd.DataFrame,
    front_df: pd.DataFrame,
    ylabel: str,
    title: str,
    ylim: Optional[Tuple[float, float]] = None,
    show_legend: bool = False,
) -> None:

    x = np.arange(1, 7)
    dx = 0.10

    for df, color, label, offset in [
        (local_df, LOCAL_COLOR, "Interior/local", -dx),
        (front_df, FRONT_COLOR, "Front", dx),
    ]:
        y = df["value"].to_numpy(dtype=float)
        lo = df["lo"].to_numpy(dtype=float)
        hi = df["hi"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])

        ax.errorbar(
            x + offset,
            y,
            yerr=yerr,
            fmt="o-",
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.0,
            linewidth=2.8,
            elinewidth=2.2,
            capsize=5.0,
            capthick=2.0,
            markersize=9.0,
            label=label,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS, fontsize=FONT)
    ax.set_xlabel("Source state", fontsize=FONT, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=FONT, labelpad=12)
    ax.set_title(title, loc="left", pad=14, fontsize=FONT, fontweight="bold")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(axis="both", labelsize=FONT, pad=5)
    ax.grid(axis="y", color="0.88", linewidth=1.2)
    ax.set_axisbelow(True)
    clean_axis(ax)

    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    if show_legend:
        ax.legend(
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.995),
            handlelength=1.45,
            borderaxespad=0.15,
            labelspacing=0.28,
            fontsize=FONT,
        )


# =============================================================================
# 8. COMPACT FLOW PANELS
# =============================================================================

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

    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
    ]

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


def collect_flow_probabilities(
    mats: List[np.ndarray],
    threshold: float = FLOW_MIN_PROB,
) -> np.ndarray:

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


def flow_line_width(p: float, pmax: float) -> float:
    if not np.isfinite(pmax) or pmax <= 0:
        pmax = 1.0

    r = max(p / pmax, 0.0)
    return 0.55 + 9.2 * (r ** 1.18)


def flow_line_alpha(p: float, pmax: float) -> float:
    if not np.isfinite(pmax) or pmax <= 0:
        pmax = 1.0

    r = max(p / pmax, 0.0)
    return min(0.88, 0.18 + 0.72 * (r ** 0.85))


def plot_flow_panel(
    ax,
    mat: np.ndarray,
    support: np.ndarray,
    title: str,
    subtitle: str,
    pmax_global: float,
    threshold: float = FLOW_MIN_PROB,
) -> None:

    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.55, 5.90)

    mat = np.asarray(mat, dtype=float)
    support = np.asarray(support, dtype=float)
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
        fontsize=FONT,
        fontweight="bold",
        clip_on=False,
    )

    ax.text(
        0.0, 1.005,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT,
        color="0.35",
        clip_on=False,
    )

    ax.text(
        x_left,
        5.48,
        "Source",
        ha="center",
        va="bottom",
        fontsize=FONT,
        color="0.25",
    )
    ax.text(
        x_right,
        5.48,
        "Target",
        ha="center",
        va="bottom",
        fontsize=FONT,
        color="0.25",
    )

    ax.annotate(
        "",
        xy=(0.60, 5.43),
        xytext=(0.40, 5.43),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.6,
            color="0.45",
            mutation_scale=18.0,
        ),
    )

    links = []
    for i, fs in enumerate(STATE_ORDER):
        for j, ts in enumerate(STATE_ORDER):
            p = float(mat[i, j])
            if not np.isfinite(p) or p < threshold:
                continue
            links.append((p, fs, ts, i, j))

    links = sorted(links, key=lambda x: x[0])

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
                s=980,
                color=color,
                edgecolor="white",
                linewidth=1.4,
                zorder=30,
            )

            text_color = "white" if s in [1, 2, 5, 6] else "black"
            ax.text(
                x, y,
                f"S{s}",
                ha="center",
                va="center",
                fontsize=FONT,
                fontweight="bold",
                color=text_color,
                zorder=31,
            )

    ax.text(
        0.50,
        -1.20,
        (
            f"retention = {summary['weighted_retention']:.2f}; "
            f"breadth = {summary['weighted_breadth']:.2f}; "
            f"N = {fmt_support(summary['total_support'])}"
        ),
        ha="center",
        va="center",
        fontsize=FONT,
        color="0.32",
        clip_on=False,
    )


# =============================================================================
# 9. MAIN FIGURE
# =============================================================================

def draw_figure(
    annual: pd.DataFrame,
    out_dir: Path,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    flow_min_prob: float = FLOW_MIN_PROB,
) -> None:

    ensure_dir(out_dir)

    _, cube_front = counts_cube_for_kernel(
        annual, "front", year_min=year_min, year_max=year_max
    )
    _, cube_local = counts_cube_for_kernel(
        annual, "local", year_min=year_min, year_max=year_max
    )

    front_mat, front_support = climatology_matrix_from_cube(cube_front)
    local_mat, local_support = climatology_matrix_from_cube(cube_local)
    diff_mat = front_mat - local_mat

    pd.DataFrame(front_mat, index=STATE_LABELS, columns=STATE_LABELS).to_csv(
        out_dir / "Figure5a_front_transition_kernel.csv",
        encoding="utf-8-sig",
    )

    pd.DataFrame(local_mat, index=STATE_LABELS, columns=STATE_LABELS).to_csv(
        out_dir / "Figure5b_interior_local_transition_kernel.csv",
        encoding="utf-8-sig",
    )

    pd.DataFrame(diff_mat, index=STATE_LABELS, columns=STATE_LABELS).to_csv(
        out_dir / "Figure5c_front_minus_interior_local_kernel.csv",
        encoding="utf-8-sig",
    )

    edge_df = pd.concat(
        [
            edge_table_from_matrix(local_mat, local_support, "local", flow_min_prob),
            edge_table_from_matrix(front_mat, front_support, "front", flow_min_prob),
        ],
        ignore_index=True,
    )
    edge_df.to_csv(
        out_dir / "Figure5_flow_edge_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = pd.DataFrame([
        {"kernel": "interior_local", **weighted_summary(local_mat, local_support)},
        {"kernel": "front", **weighted_summary(front_mat, front_support)},
    ])
    summary_df.to_csv(
        out_dir / "Figure5_weighted_summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    local_ret = bootstrap_source_metric(cube_local, "retention")
    front_ret = bootstrap_source_metric(cube_front, "retention")
    local_brd = bootstrap_source_metric(cube_local, "breadth")
    front_brd = bootstrap_source_metric(cube_front, "breadth")

    local_ret["kernel"] = "interior_local"
    front_ret["kernel"] = "front"
    local_brd["kernel"] = "interior_local"
    front_brd["kernel"] = "front"

    pd.concat([local_ret, front_ret], ignore_index=True).to_csv(
        out_dir / "Figure5d_source_state_retention_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat([local_brd, front_brd], ignore_index=True).to_csv(
        out_dir / "Figure5e_source_state_transition_breadth_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prob_vmax = np.nanmax([np.nanmax(front_mat), np.nanmax(local_mat)])
    if not np.isfinite(prob_vmax):
        prob_vmax = 1.0
    prob_vmax = max(0.80, float(np.ceil(prob_vmax * 20) / 20))
    prob_vmax = min(1.0, prob_vmax)

    prob_norm = mcolors.Normalize(vmin=0.0, vmax=prob_vmax)

    diff_lim = np.nanmax(np.abs(diff_mat))
    if not np.isfinite(diff_lim) or diff_lim <= 0:
        diff_lim = 0.10
    diff_lim = max(0.20, float(np.ceil(diff_lim * 20) / 20))

    diff_norm = mcolors.TwoSlopeNorm(
        vmin=-diff_lim,
        vcenter=0.0,
        vmax=diff_lim,
    )

    flow_probs = collect_flow_probabilities(
        [local_mat, front_mat],
        threshold=flow_min_prob,
    )
    pmax_global = float(np.nanmax(flow_probs))

    # =========================================================================
    # Manual template-matched layout
    # =========================================================================

    fig = plt.figure(figsize=FIGSIZE)

    # Top-row heatmap geometry.
    # Width and height are chosen so each heatmap is physically square.
    top_y = 0.590
    top_h = 0.360
    top_w = top_h * FIGSIZE[1] / FIGSIZE[0]

    cbar_w = 0.012
    cbar_gap = 0.010

    ax_a_pos = [0.070, top_y, top_w, top_h]
    ax_b_pos = [0.335, top_y, top_w, top_h]
    cax_prob_pos = [ax_b_pos[0] + top_w + cbar_gap, top_y, cbar_w, top_h]

    # Extra gap after the probability colorbar is necessary because the colorbar
    # tick labels and vertical label are 26 pt.
    ax_c_pos = [0.650, top_y, top_w, top_h]
    cax_diff_pos = [ax_c_pos[0] + top_w + cbar_gap, top_y, cbar_w, top_h]

    ax_a = fig.add_axes(ax_a_pos)
    ax_b = fig.add_axes(ax_b_pos)
    cax_prob = fig.add_axes(cax_prob_pos)
    ax_c = fig.add_axes(ax_c_pos)
    cax_diff = fig.add_axes(cax_diff_pos)

    add_panel_label(ax_a, "a", x=-0.18, y=1.14)
    im_prob = plot_heatmap(
        ax=ax_a,
        mat=front_mat,
        title="Front transition kernel",
        cmap=PROB_CMAP,
        norm=prob_norm,
        fmt="{:.2f}",
        show_ylabel=True,
    )

    add_panel_label(ax_b, "b", x=-0.18, y=1.14)
    plot_heatmap(
        ax=ax_b,
        mat=local_mat,
        title="Interior/local transition kernel",
        cmap=PROB_CMAP,
        norm=prob_norm,
        fmt="{:.2f}",
        show_ylabel=False,
    )

    cb_prob = fig.colorbar(
        im_prob,
        cax=cax_prob,
        orientation="vertical",
        extend="both",
    )
    cb_prob.set_label("Transition probability", fontsize=FONT, labelpad=14)
    cb_prob.ax.tick_params(labelsize=FONT, length=5.5, width=1.15, pad=5)
    cb_prob.outline.set_linewidth(1.25)

    add_panel_label(ax_c, "c", x=-0.18, y=1.14)
    im_diff = plot_heatmap(
        ax=ax_c,
        mat=diff_mat,
        title="Front − interior/local",
        cmap=DIFF_CMAP,
        norm=diff_norm,
        fmt="{:+.2f}",
        show_ylabel=True,
    )

    cb_diff = fig.colorbar(
        im_diff,
        cax=cax_diff,
        orientation="vertical",
        extend="both",
    )
    cb_diff.set_label("Probability difference", fontsize=FONT, labelpad=14)
    cb_diff.ax.tick_params(labelsize=FONT, length=5.5, width=1.15, pad=5)
    cb_diff.outline.set_linewidth(1.25)

    # Bottom-row geometry.
    bottom_y = 0.145
    bottom_h = 0.305

    ax_d = fig.add_axes([0.055, bottom_y, 0.200, bottom_h])
    add_panel_label(ax_d, "d", x=-0.22, y=1.16)
    plot_metric_panel(
        ax=ax_d,
        local_df=local_ret,
        front_df=front_ret,
        ylabel="Retention probability",
        title="Source-state retention",
        ylim=(0.0, 1.0),
        show_legend=True,
    )

    ax_e = fig.add_axes([0.305, bottom_y, 0.200, bottom_h])
    add_panel_label(ax_e, "e", x=-0.22, y=1.16)
    plot_metric_panel(
        ax=ax_e,
        local_df=local_brd,
        front_df=front_brd,
        ylabel="Expected state jump",
        title="Transition breadth",
        ylim=(0.0, None),
        show_legend=False,
    )

    # Panel f outer axis controls the panel label and the legend.
    flow_outer_pos = [0.555, bottom_y, 0.415, bottom_h]
    ax_f_outer = fig.add_axes(flow_outer_pos)
    ax_f_outer.set_axis_off()
    add_panel_label(ax_f_outer, "f", x=-0.09, y=1.16)

    # Two flow panels inside panel f.
    inner_gap = 0.035
    each_w = (flow_outer_pos[2] - inner_gap) / 2.0

    ax_f1 = fig.add_axes([
        flow_outer_pos[0],
        flow_outer_pos[1],
        each_w,
        flow_outer_pos[3],
    ])

    ax_f2 = fig.add_axes([
        flow_outer_pos[0] + each_w + inner_gap,
        flow_outer_pos[1],
        each_w,
        flow_outer_pos[3],
    ])

    plot_flow_panel(
        ax=ax_f1,
        mat=local_mat,
        support=local_support,
        title="Interior/local",
        subtitle=f"major pathways, P ≥ {flow_min_prob:.2f}",
        pmax_global=pmax_global,
        threshold=flow_min_prob,
    )

    plot_flow_panel(
        ax=ax_f2,
        mat=front_mat,
        support=front_support,
        title="Advancing front",
        subtitle=f"major pathways, P ≥ {flow_min_prob:.2f}",
        pmax_global=pmax_global,
        threshold=flow_min_prob,
    )

    legend_handles = [
        plt.Line2D(
            [0], [0],
            color=STATE_COLORS[s],
            lw=8.0,
            solid_capstyle="round",
            label=f"Source S{s}",
        )
        for s in STATE_ORDER
    ]

    # Legend fixed directly below panel f, matching the template.
    ax_f_outer.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.155),
        ncol=6,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.95,
        borderaxespad=0.0,
        fontsize=FONT,
    )

    save_figure(
        fig,
        out_dir,
        "Figure5_baseline_front_local_transition_NCC_rebuilt_final",
    )


# =============================================================================
# 10. ARGUMENTS AND MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Root directory containing _result3_transition_kernel/result3_annual_counts.csv.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory. Default is "
            "DEFAULT_OUT_BASE / Figure5_baseline_front_local_transition_NCC_rebuilt_final"
        ),
    )

    parser.add_argument(
        "--year-min",
        default=None,
        type=int,
        help="Optional baseline start year. Default uses all available years.",
    )

    parser.add_argument(
        "--year-max",
        default=None,
        type=int,
        help="Optional baseline end year. Default uses all available years.",
    )

    parser.add_argument(
        "--flow-min-prob",
        default=FLOW_MIN_PROB,
        type=float,
        help="Minimum transition probability shown in flow panels. Default: 0.05.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(args.root)

    out_dir = ensure_dir(
        Path(args.output)
        if args.output is not None
        else DEFAULT_OUT_BASE / OUT_SUBDIR
    )

    log("=" * 100)
    log("[INFO] Rebuilding Figure 5 using template-matched manual layout")
    log("[INFO] Main corrections:")
    log("[INFO]   - manual axes placement")
    log("[INFO]   - all ordinary text set to 26 pt")
    log("[INFO]   - enlarged heatmap panels for 26-pt annotations")
    log("[INFO]   - colorbars close to b and c but no longer overlapping labels")
    log("[INFO]   - panel-f legend fixed directly below panel f")
    log("[INFO]   - bottom-row titles separated from top-row x labels")
    log(f"[INFO] root          : {root}")
    log(f"[INFO] output        : {out_dir}")
    log(
        "[INFO] baseline years: "
        f"{args.year_min if args.year_min is not None else 'all'} - "
        f"{args.year_max if args.year_max is not None else 'all'}"
    )
    log(f"[INFO] flow threshold: P >= {args.flow_min_prob:.3f}")
    log("=" * 100)

    annual = load_annual_counts(root)

    draw_figure(
        annual=annual,
        out_dir=out_dir,
        year_min=args.year_min,
        year_max=args.year_max,
        flow_min_prob=float(args.flow_min_prob),
    )

    log("=" * 100)
    log("[DONE] Figure 5 finished.")
    log(f"[DONE] Output directory: {out_dir}")
    log("=" * 100)


if __name__ == "__main__":
    main()