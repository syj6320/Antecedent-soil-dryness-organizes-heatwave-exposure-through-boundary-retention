# -*- coding: utf-8 -*-
"""
Reviewer comment 5 — final v9 layout-fixed replot
=================================================

This script keeps the scientific content unchanged and only fixes layout.

Main figure:
1. Keep b/c vertical colorbar close to panel c.
2. Move "Front-local dryward" vertically below its point in panel e.
3. Keep "Front breadth" at the upper-right position.
4. Keep panel d legend at upper right.

Supplementary figure:
1. Generate Supplementary_R5_support_kernels_QC_v9_layout_fixed.png/pdf.
2. Keep transition-kernel colorbar vertical on the right with pointed ends.
3. Keep panel d legend at upper right.

No ERA5 rerun.
No raw CMIP6 reread.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS
# =============================================================================

ERA5_ROOT = Path(
    r"E:\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本"
    r"\ERA5_pseudo100km_resolution_sensitivity"
)

ERA5_DIAG_DIR = ERA5_ROOT / "3_native_vs_pseudo100_diagnostics"

OUT_DIR = ERA5_ROOT / "4_figures_R5_final_v9_layout_fixed_no_rerun"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CMIP6_COUNT_CACHE_CANDIDATES = [
    ERA5_ROOT / "4_figures_editorial_revised_with_CMIP6"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",

    ERA5_ROOT / "4_figures_editorial_revised_weighted_CMIP6"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",

    ERA5_ROOT / "4_figures_final_no_rerun_weighted"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",

    ERA5_ROOT / "4_figures_final_nature_style_no_rerun"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",

    ERA5_ROOT / "4_figures_R5_two_panel_nature_style_no_rerun"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",

    ERA5_ROOT / "4_figures_R5_final_v3_no_rerun"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",

    ERA5_ROOT / "4_figures_R5_final_v4_no_rerun"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",

    ERA5_ROOT / "4_figures_R5_final_v5_layout_fixed_no_rerun"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",

    ERA5_ROOT / "4_figures_R5_final_v6_layout_fixed_no_rerun"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",

    ERA5_ROOT / "4_figures_R5_final_v7_layout_fixed_no_rerun"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",

    ERA5_ROOT / "4_figures_R5_final_v8_layout_fixed_no_rerun"
    / "_cmip6_ssp585_transition_cache"
    / "cmip6_ssp585_transition_counts_hist_future.csv",
]

HIST_PERIOD = "historical_1985_2014"
FUTURE_PERIOD = "future_2071_2100"

DPI = 600
N_BOOT = 1000
RANDOM_SEED = 42


# =============================================================================
# 2. STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11.0,
    "axes.labelsize": 12.0,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 10.2,
    "ytick.labelsize": 10.2,
    "legend.fontsize": 9.0,
    "axes.linewidth": 0.95,
    "xtick.major.width": 0.85,
    "ytick.major.width": 0.85,
    "xtick.major.size": 3.8,
    "ytick.major.size": 3.8,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

STATE_ORDER = [1, 2, 3, 4, 5, 6]
STATE_LABELS = [f"S{i}" for i in STATE_ORDER]

BLUE = "#2166ac"
RED = "#b2182b"
GREEN = "#1b9e77"
PURPLE = "#7570b3"

METRIC_ORDER = [
    "front_S1_persistence",
    "local_S1_persistence",
    "local_memory_advantage",
    "front_dryward_tendency",
    "front_minus_local_dryward",
    "front_breadth_excess",
]

METRIC_LABEL = {
    "front_S1_persistence": "Front S1",
    "local_S1_persistence": "Local S1",
    "local_memory_advantage": "Local-front S1",
    "front_dryward_tendency": "Front dryward",
    "front_minus_local_dryward": "Front-local dryward",
    "front_breadth_excess": "Front breadth",
}

METRIC_MARKER = {
    "front_S1_persistence": "o",
    "local_S1_persistence": "s",
    "local_memory_advantage": "^",
    "front_dryward_tendency": "D",
    "front_minus_local_dryward": "P",
    "front_breadth_excess": "X",
}


# =============================================================================
# 3. BASIC HELPERS
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, label: str, x=-0.12, y=1.12, size=17):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
        zorder=30,
    )


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_resolution_names(df: pd.DataFrame) -> pd.DataFrame:
    if "resolution" in df.columns:
        df["resolution"] = df["resolution"].replace({
            "coarse": "pseudo100",
            "pseudo_100": "pseudo100",
            "pseudo-100": "pseudo100",
            "pseudo100km": "pseudo100",
        })
    return df


def standardize_period_names(df: pd.DataFrame) -> pd.DataFrame:
    if "period" not in df.columns:
        return df

    df = df.copy()
    df["period"] = df["period"].astype(str).replace({
        "historical": HIST_PERIOD,
        "hist": HIST_PERIOD,
        "1985_2014": HIST_PERIOD,
        "1985-2014": HIST_PERIOD,
        "baseline_1985_2014": HIST_PERIOD,
        "future": FUTURE_PERIOD,
        "fut": FUTURE_PERIOD,
        "2071_2100": FUTURE_PERIOD,
        "2071-2100": FUTURE_PERIOD,
        "endcentury_2071_2100": FUTURE_PERIOD,
    })
    return df


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fix_log_ticks(ax, ticks, labels):
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.tick_params(axis="x", which="major", pad=4)


# =============================================================================
# 4. LOAD EXISTING FILES ONLY
# =============================================================================

def load_era5_diagnostics() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = {
        "counts": ERA5_DIAG_DIR / "transition_counts_native_vs_pseudo100.csv",
        "support": ERA5_DIAG_DIR / "transition_support_native_vs_pseudo100.csv",
        "gradients": ERA5_DIAG_DIR / "gradient_daily_native_vs_pseudo100.csv",
        "objects": ERA5_DIAG_DIR / "object_summary_native_vs_pseudo100.csv",
    }

    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing ERA5 diagnostic files. This script does not rerun ERA5.\n"
            "Missing files:\n" + "\n".join(missing)
        )

    counts = normalize_resolution_names(pd.read_csv(files["counts"]))
    support = normalize_resolution_names(pd.read_csv(files["support"]))
    gradients = normalize_resolution_names(pd.read_csv(files["gradients"]))
    objects = normalize_resolution_names(pd.read_csv(files["objects"]))

    return counts, support, gradients, objects


def find_cmip6_counts_cache() -> Path:
    for p in CMIP6_COUNT_CACHE_CANDIDATES:
        if p.exists():
            return p

    raise FileNotFoundError(
        "CMIP6 transition-count cache was not found. This script will NOT rebuild it.\n"
        "Please add the actual cache path to CMIP6_COUNT_CACHE_CANDIDATES.\n\n"
        "Checked paths:\n" + "\n".join(str(p) for p in CMIP6_COUNT_CACHE_CANDIDATES)
    )


def load_cmip6_counts() -> pd.DataFrame:
    p = find_cmip6_counts_cache()
    log(f"[LOAD] CMIP6 transition-count cache: {p}")

    df = pd.read_csv(p)
    df = standardize_period_names(df)

    required = {"model", "period", "kernel", "from_state", "to_state", "count"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"CMIP6 cache missing columns: {sorted(missing)}\nFile: {p}"
        )

    return df


# =============================================================================
# 5. COUNT-WEIGHTED METRICS
# =============================================================================

def subset_counts(
    counts: pd.DataFrame,
    resolution: Optional[str] = None,
    kernel: Optional[str] = None,
    year: Optional[int] = None,
    model: Optional[str] = None,
    period: Optional[str] = None,
) -> pd.DataFrame:
    x = counts

    if resolution is not None and "resolution" in x.columns:
        x = x[x["resolution"] == resolution]
    if kernel is not None and "kernel" in x.columns:
        x = x[x["kernel"] == kernel]
    if year is not None and "year" in x.columns:
        x = x[x["year"] == year]
    if model is not None and "model" in x.columns:
        x = x[x["model"] == model]
    if period is not None and "period" in x.columns:
        x = x[x["period"] == period]

    return x.copy()


def matrix_and_row_counts(counts_sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.full((6, 6), np.nan)
    row_counts = np.zeros(6, dtype=float)

    if counts_sub.empty:
        return mat, row_counts

    x = counts_sub.copy()
    x["from_state"] = to_num(x["from_state"])
    x["to_state"] = to_num(x["to_state"])
    x["count"] = to_num(x["count"])

    x = x[
        x["from_state"].isin(STATE_ORDER)
        & x["to_state"].isin(STATE_ORDER)
        & x["count"].notna()
    ].copy()

    if x.empty:
        return mat, row_counts

    g = x.groupby(["from_state", "to_state"], as_index=False)["count"].sum()

    for fs in STATE_ORDER:
        row = g[g["from_state"] == fs]
        denom = row["count"].sum()
        row_counts[fs - 1] = denom

        if denom <= 0:
            continue

        for ts in STATE_ORDER:
            num = row.loc[row["to_state"] == ts, "count"].sum()
            mat[fs - 1, ts - 1] = num / denom

    return mat, row_counts


def matrix_from_counts(
    counts: pd.DataFrame,
    resolution: Optional[str],
    kernel: str,
    model: Optional[str] = None,
    period: Optional[str] = None,
) -> np.ndarray:
    sub = subset_counts(
        counts,
        resolution=resolution,
        kernel=kernel,
        model=model,
        period=period,
    )
    mat, _ = matrix_and_row_counts(sub)
    return mat


def weighted_dryward_tendency(counts_sub: pd.DataFrame) -> float:
    if counts_sub.empty:
        return np.nan

    x = counts_sub.copy()
    x["from_state"] = to_num(x["from_state"])
    x["to_state"] = to_num(x["to_state"])
    x["count"] = to_num(x["count"])

    x = x[
        x["from_state"].isin(STATE_ORDER)
        & x["to_state"].isin(STATE_ORDER)
        & x["count"].notna()
    ].copy()

    if x.empty:
        return np.nan

    total = x["count"].sum()
    if total <= 0:
        return np.nan

    dry = x.loc[x["to_state"] < x["from_state"], "count"].sum()
    wet = x.loc[x["to_state"] > x["from_state"], "count"].sum()

    return float((dry - wet) / total)


def weighted_transition_breadth(counts_sub: pd.DataFrame) -> float:
    if counts_sub.empty:
        return np.nan

    mat, row_counts = matrix_and_row_counts(counts_sub)
    total = row_counts.sum()

    if total <= 0:
        return np.nan

    ent, weights = [], []

    for i in range(6):
        if row_counts[i] <= 0:
            continue

        row = mat[i, :]
        row = row[np.isfinite(row)]
        row = row[row > 0]

        h = 0.0 if row.size <= 1 else float(-np.sum(row * np.log(row)))
        ent.append(h)
        weights.append(row_counts[i])

    if not ent:
        return np.nan

    return float(np.average(ent, weights=weights))


def metrics_from_count_subset(counts_sub: pd.DataFrame) -> Dict[str, float]:
    front_counts = subset_counts(counts_sub, kernel="front")
    local_counts = subset_counts(counts_sub, kernel="local")

    front_mat, _ = matrix_and_row_counts(front_counts)
    local_mat, _ = matrix_and_row_counts(local_counts)

    out: Dict[str, float] = {}

    out["front_S1_persistence"] = front_mat[0, 0] if np.isfinite(front_mat[0, 0]) else np.nan
    out["local_S1_persistence"] = local_mat[0, 0] if np.isfinite(local_mat[0, 0]) else np.nan

    if np.isfinite(out["front_S1_persistence"]) and np.isfinite(out["local_S1_persistence"]):
        out["local_memory_advantage"] = out["local_S1_persistence"] - out["front_S1_persistence"]
    else:
        out["local_memory_advantage"] = np.nan

    out["front_dryward_tendency"] = weighted_dryward_tendency(front_counts)
    out["local_dryward_tendency"] = weighted_dryward_tendency(local_counts)

    if np.isfinite(out["front_dryward_tendency"]) and np.isfinite(out["local_dryward_tendency"]):
        out["front_minus_local_dryward"] = out["front_dryward_tendency"] - out["local_dryward_tendency"]
    else:
        out["front_minus_local_dryward"] = np.nan

    out["front_transition_breadth"] = weighted_transition_breadth(front_counts)
    out["local_transition_breadth"] = weighted_transition_breadth(local_counts)

    if np.isfinite(out["front_transition_breadth"]) and np.isfinite(out["local_transition_breadth"]):
        out["front_breadth_excess"] = out["front_transition_breadth"] - out["local_transition_breadth"]
    else:
        out["front_breadth_excess"] = np.nan

    return out


def annual_era5_metrics(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for year in sorted(counts["year"].dropna().unique()):
        for res in ["native", "pseudo100"]:
            sub = subset_counts(counts, resolution=res, year=int(year))
            if sub.empty:
                continue

            m = metrics_from_count_subset(sub)
            m["year"] = int(year)
            m["resolution"] = res
            rows.append(m)

    return pd.DataFrame(rows)


def full_era5_metrics(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for res in ["native", "pseudo100"]:
        sub = subset_counts(counts, resolution=res)
        if sub.empty:
            continue

        m = metrics_from_count_subset(sub)
        m["resolution"] = res
        rows.append(m)

    return pd.DataFrame(rows)


def era5_bias_from_full(full_m: pd.DataFrame) -> pd.DataFrame:
    native = full_m[full_m["resolution"] == "native"].iloc[0]
    pseudo = full_m[full_m["resolution"] == "pseudo100"].iloc[0]

    rows = []

    for metric in METRIC_ORDER:
        nv = float(native[metric]) if pd.notna(native[metric]) else np.nan
        pv = float(pseudo[metric]) if pd.notna(pseudo[metric]) else np.nan

        if np.isfinite(nv) and np.isfinite(pv):
            bias = pv - nv
            retention = abs(pv) / abs(nv) if abs(nv) > 1e-12 else np.nan
        else:
            bias = np.nan
            retention = np.nan

        rows.append({
            "metric": metric,
            "native": nv,
            "pseudo100": pv,
            "bias": bias,
            "retention": retention,
        })

    return pd.DataFrame(rows)


def fast_bootstrap_era5_bias(
    annual_m: pd.DataFrame,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for metric in METRIC_ORDER:
        pivot = annual_m.pivot_table(
            index="year",
            columns="resolution",
            values=metric,
            aggfunc="mean",
        )

        if "native" not in pivot.columns or "pseudo100" not in pivot.columns:
            rows.append({
                "metric": metric,
                "bias_median": np.nan,
                "bias_q025": np.nan,
                "bias_q975": np.nan,
            })
            continue

        diff = (pivot["pseudo100"] - pivot["native"]).to_numpy(dtype=float)
        diff = diff[np.isfinite(diff)]

        if diff.size == 0:
            rows.append({
                "metric": metric,
                "bias_median": np.nan,
                "bias_q025": np.nan,
                "bias_q975": np.nan,
            })
            continue

        n = diff.size
        boot = np.empty(n_boot, dtype=float)

        for i in range(n_boot):
            boot[i] = np.nanmean(rng.choice(diff, size=n, replace=True))

        rows.append({
            "metric": metric,
            "bias_median": float(np.nanmedian(boot)),
            "bias_q025": float(np.nanquantile(boot, 0.025)),
            "bias_q975": float(np.nanquantile(boot, 0.975)),
        })

    return pd.DataFrame(rows)


def cmip6_metrics_and_signal(cmip6_counts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []

    for model in sorted(cmip6_counts["model"].dropna().unique()):
        for period in [HIST_PERIOD, FUTURE_PERIOD]:
            sub = subset_counts(cmip6_counts, model=model, period=period)
            if sub.empty:
                continue

            m = metrics_from_count_subset(sub)
            m["model"] = model
            m["period"] = period
            m["scenario"] = "ssp585"
            metric_rows.append(m)

    metrics = pd.DataFrame(metric_rows)

    signal_rows = []
    for model in sorted(metrics["model"].dropna().unique()):
        h = metrics[(metrics["model"] == model) & (metrics["period"] == HIST_PERIOD)]
        f = metrics[(metrics["model"] == model) & (metrics["period"] == FUTURE_PERIOD)]

        if h.empty or f.empty:
            continue

        h = h.iloc[0]
        f = f.iloc[0]

        row = {"model": model, "scenario": "ssp585"}
        for metric in METRIC_ORDER:
            hv = h.get(metric, np.nan)
            fv = f.get(metric, np.nan)
            row[metric] = float(fv - hv) if pd.notna(hv) and pd.notna(fv) else np.nan

        signal_rows.append(row)

    signal = pd.DataFrame(signal_rows)

    summary_rows = []
    for metric in METRIC_ORDER:
        vals = (
            pd.to_numeric(signal[metric], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
            if not signal.empty and metric in signal.columns
            else np.array([], dtype=float)
        )

        if vals.size == 0:
            summary_rows.append({
                "metric": metric,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "q05": np.nan,
                "q95": np.nan,
                "n": 0,
                "positive_agreement": np.nan,
            })
        else:
            summary_rows.append({
                "metric": metric,
                "median": float(np.nanmedian(vals)),
                "q25": float(np.nanquantile(vals, 0.25)),
                "q75": float(np.nanquantile(vals, 0.75)),
                "q05": float(np.nanquantile(vals, 0.05)),
                "q95": float(np.nanquantile(vals, 0.95)),
                "n": int(vals.size),
                "positive_agreement": float(np.mean(vals > 0)),
            })

    summary = pd.DataFrame(summary_rows)

    return metrics, signal, summary


# =============================================================================
# 6. DERIVED TABLES
# =============================================================================

def annual_gradient_summary(gradients: pd.DataFrame, domain: str = "active_heatwave_footprint") -> pd.DataFrame:
    x = gradients[gradients["domain"] == domain].copy() if "domain" in gradients.columns else gradients.copy()
    if x.empty:
        x = gradients.copy()

    return (
        x.groupby(["year", "resolution"], as_index=False)
        .agg(
            sm_gradient_per100km=("sm_gradient_per100km", "mean"),
            temp_gradient_per100km=("temp_gradient_per100km", "mean"),
            n_pairs=("n_pairs", "sum"),
        )
    )


def paired_ratio_table_from_gradients(annual_g: pd.DataFrame) -> pd.DataFrame:
    rows = []

    variables = {
        "sm_gradient_per100km": r"$|\nabla SM|$",
        "temp_gradient_per100km": r"$|\nabla T|$",
    }

    for col, label in variables.items():
        p = annual_g.pivot_table(
            index="year",
            columns="resolution",
            values=col,
            aggfunc="mean",
        )

        if "native" not in p.columns or "pseudo100" not in p.columns:
            continue

        ratio = (p["pseudo100"] / p["native"]).replace([np.inf, -np.inf], np.nan).dropna()

        for year, val in ratio.items():
            rows.append({
                "year": int(year),
                "variable": label,
                "ratio": float(val),
            })

    return pd.DataFrame(rows)


def support_ratio_table(support: pd.DataFrame) -> pd.DataFrame:
    rows = []
    support_col = pick_col(support, ["support", "count", "n", "transitions"])
    if support_col is None:
        raise RuntimeError("Cannot find support column in transition_support file.")

    for kernel, label in [
        ("front", "Front transitions"),
        ("local", "Local transitions"),
    ]:
        p = support[support["kernel"] == kernel].pivot_table(
            index="year",
            columns="resolution",
            values=support_col,
            aggfunc="sum",
        )

        if "native" not in p.columns or "pseudo100" not in p.columns:
            continue

        ratio = (p["pseudo100"] / p["native"]).replace([np.inf, -np.inf], np.nan).dropna()

        for year, val in ratio.items():
            rows.append({
                "year": int(year),
                "variable": label,
                "ratio": float(val),
            })

    return pd.DataFrame(rows)


def object_ratio_table(objects: pd.DataFrame) -> pd.DataFrame:
    rows = []

    object_grid_col = pick_col(
        objects,
        ["object_grid_days", "grid_days", "total_grid_days", "event_grid_days", "object_voxels"],
    )
    largest_col = pick_col(
        objects,
        ["largest_component_fraction", "largest_fraction", "largest_component_frac"],
    )

    variables = []
    if object_grid_col is not None:
        variables.append((object_grid_col, "Object grid-days"))
    if largest_col is not None:
        variables.append((largest_col, "Largest component"))

    for col, label in variables:
        p = objects.pivot_table(
            index="year",
            columns="resolution",
            values=col,
            aggfunc="mean",
        )

        if "native" not in p.columns or "pseudo100" not in p.columns:
            continue

        ratio = (p["pseudo100"] / p["native"]).replace([np.inf, -np.inf], np.nan).dropna()

        for year, val in ratio.items():
            rows.append({
                "year": int(year),
                "variable": label,
                "ratio": float(val),
            })

    return pd.DataFrame(rows)


def magnitude_ratio_table(
    bias: pd.DataFrame,
    boot: pd.DataFrame,
    cmip6_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    cm = {r["metric"]: r for _, r in cmip6_summary.iterrows()}

    for metric in METRIC_ORDER:
        b = bias[bias["metric"] == metric]
        if not b.empty:
            native = abs(float(b["native"].iloc[0]))
            pseudo = abs(float(b["pseudo100"].iloc[0]))

            if native > 1e-12 and np.isfinite(native) and np.isfinite(pseudo):
                rows.append({
                    "metric": metric,
                    "label": METRIC_LABEL[metric],
                    "type": "pseudo/native",
                    "ratio": pseudo / native,
                })

        eb = boot[boot["metric"] == metric]
        cs = cm.get(metric)

        if not eb.empty and cs is not None:
            era5_bias = abs(float(eb["bias_median"].iloc[0]))
            cmip6_signal = abs(float(cs["median"]))

            if cmip6_signal > 1e-12 and np.isfinite(era5_bias) and np.isfinite(cmip6_signal):
                rows.append({
                    "metric": metric,
                    "label": METRIC_LABEL[metric],
                    "type": "bias/signal",
                    "ratio": era5_bias / cmip6_signal,
                })

    return pd.DataFrame(rows)


def bias_signal_phase_table(
    boot: pd.DataFrame,
    cmip6_summary: pd.DataFrame,
) -> pd.DataFrame:
    cm = {r["metric"]: r for _, r in cmip6_summary.iterrows()}
    rows = []

    for metric in METRIC_ORDER:
        eb = boot[boot["metric"] == metric]
        cs = cm.get(metric)

        if eb.empty or cs is None:
            continue

        bias_abs = abs(float(eb["bias_median"].iloc[0]))
        bias_lo = abs(float(eb["bias_q025"].iloc[0]))
        bias_hi = abs(float(eb["bias_q975"].iloc[0]))

        signal_abs = abs(float(cs["median"]))
        signal_q25 = abs(float(cs["q25"]))
        signal_q75 = abs(float(cs["q75"]))

        if not np.isfinite(bias_abs) or not np.isfinite(signal_abs):
            continue
        if bias_abs <= 0 or signal_abs <= 0:
            continue

        rows.append({
            "metric": metric,
            "label": METRIC_LABEL[metric],
            "bias_abs": bias_abs,
            "bias_lo_abs": min(bias_lo, bias_hi),
            "bias_hi_abs": max(bias_lo, bias_hi),
            "signal_abs": signal_abs,
            "signal_lo_abs": min(signal_q25, signal_q75),
            "signal_hi_abs": max(signal_q25, signal_q75),
            "ratio": bias_abs / signal_abs if signal_abs > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 7. PLOT HELPERS
# =============================================================================

def draw_compact_ratio_distribution(
    ax,
    ratio_df: pd.DataFrame,
    order,
    xlabel,
    ylabel="",
    show_values=True,
    xlim=None,
):
    ratio_df = ratio_df.copy()
    ratio_df["variable"] = pd.Categorical(ratio_df["variable"], categories=order, ordered=True)
    ratio_df = ratio_df.sort_values("variable")

    y_positions = np.arange(len(order))[::-1]
    rng = np.random.default_rng(9)

    for yy, var in zip(y_positions, order):
        vals = ratio_df.loc[ratio_df["variable"] == var, "ratio"].dropna().to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue

        yj = yy + rng.normal(0, 0.050, size=vals.size)

        ax.scatter(
            vals,
            yj,
            s=15,
            color="0.72",
            alpha=0.42,
            edgecolor="none",
            zorder=1,
        )

        q05, q25, med, q75, q95 = np.nanquantile(vals, [0.05, 0.25, 0.50, 0.75, 0.95])

        ax.hlines(yy, q05, q95, color="0.42", lw=1.0, zorder=2)
        ax.hlines(yy, q25, q75, color="0.18", lw=2.2, zorder=3)
        ax.scatter(
            med,
            yy,
            s=52,
            color=RED,
            edgecolor="black",
            linewidth=0.40,
            zorder=4,
        )

        if show_values:
            ax.text(
                med,
                yy + 0.33,
                f"{med:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                clip_on=False,
            )

    ax.axvline(1.0, color="0.60", lw=0.85, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(order)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        all_vals = ratio_df["ratio"].dropna().to_numpy(float)
        all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
        if all_vals.size > 0:
            lo = max(0.02, np.nanquantile(all_vals, 0.01) * 0.60)
            hi = min(120, np.nanquantile(all_vals, 0.99) * 1.85)
            hi = max(hi, 1.6)
            ax.set_xlim(lo, hi)

    ax.set_ylim(-0.55, len(order) - 0.38)
    ax.grid(axis="x", color="0.88", lw=0.70, which="both")
    clean_spines(ax)


def draw_bias_signal_phase_space(ax, phase_df: pd.DataFrame):
    x = phase_df["signal_abs"].to_numpy(float)
    y = phase_df["bias_abs"].to_numpy(float)

    xmin = max(1e-4, np.nanmin(x) * 0.34)
    xmax = np.nanmax(x) * 3.05
    ymin = max(1e-4, np.nanmin(y) * 0.34)
    ymax = np.nanmax(y) * 3.05

    lo = min(xmin, ymin)
    hi = max(xmax, ymax)

    ax.set_xscale("log")
    ax.set_yscale("log")

    xx = np.logspace(np.log10(lo), np.log10(hi), 260)
    ax.fill_between(xx, xx, hi, color=RED, alpha=0.070, lw=0)
    ax.fill_between(xx, lo, xx, color=BLUE, alpha=0.052, lw=0)

    ax.plot(xx, xx, color="0.35", lw=0.95, ls="--", zorder=1)

    ax.text(
        0.10, 0.965,
        "bias > signal",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color=RED,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.78),
        zorder=8,
    )

    ax.text(
        0.94, 0.08,
        "bias < signal",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        color=BLUE,
        zorder=8,
    )

    # v9 correction:
    # Front-local dryward is now vertically below its data point.
    label_style = {
        "front_S1_persistence": {
            "xytext": (14, -22),
            "ha": "left",
            "va": "top",
        },
        "local_S1_persistence": {
            "xytext": (16, -30),
            "ha": "left",
            "va": "top",
        },
        "local_memory_advantage": {
            "xytext": (12, -58),
            "ha": "left",
            "va": "top",
        },
        "front_dryward_tendency": {
            "xytext": (24, -30),
            "ha": "left",
            "va": "top",
        },
        "front_minus_local_dryward": {
            "xytext": (0, -48),
            "ha": "center",
            "va": "top",
        },
        "front_breadth_excess": {
            "xytext": (18, 16),
            "ha": "left",
            "va": "bottom",
        },
    }

    for _, r in phase_df.iterrows():
        metric = r["metric"]
        marker = METRIC_MARKER.get(metric, "o")
        ratio = r["ratio"]
        color = RED if ratio > 1 else BLUE

        xerr_low = max(r["signal_abs"] - r["signal_lo_abs"], 0)
        xerr_high = max(r["signal_hi_abs"] - r["signal_abs"], 0)
        yerr_low = max(r["bias_abs"] - r["bias_lo_abs"], 0)
        yerr_high = max(r["bias_hi_abs"] - r["bias_abs"], 0)

        ax.errorbar(
            r["signal_abs"],
            r["bias_abs"],
            xerr=[[xerr_low], [xerr_high]],
            yerr=[[yerr_low], [yerr_high]],
            fmt=marker,
            ms=7.2,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            markeredgecolor="black",
            markeredgewidth=0.40,
            zorder=4,
        )

        cfg = label_style.get(
            metric,
            {"xytext": (12, -26), "ha": "left", "va": "top"},
        )

        ax.annotate(
            METRIC_LABEL[metric],
            xy=(r["signal_abs"], r["bias_abs"]),
            xytext=cfg["xytext"],
            textcoords="offset points",
            fontsize=7.8,
            ha=cfg["ha"],
            va=cfg["va"],
            arrowprops=dict(
                arrowstyle="-",
                lw=0.45,
                color="0.35",
                shrinkA=0,
                shrinkB=3,
            ),
            bbox=dict(
                boxstyle="round,pad=0.13",
                fc="white",
                ec="none",
                alpha=0.78,
            ),
            zorder=6,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(r"$|$CMIP6 signal$|$", labelpad=5)
    ax.set_ylabel(r"$|$ERA5 resolution bias$|$", labelpad=6)
    ax.grid(color="0.88", lw=0.70, which="both")
    clean_spines(ax)


def plot_change_matrix(ax, mat: np.ndarray, norm, annotate_threshold=0.04):
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="equal")

    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(STATE_LABELS)
    ax.set_yticklabels(STATE_LABELS)
    ax.set_xlabel("To", labelpad=4.0)
    ax.set_ylabel("From", labelpad=4.0)

    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(6):
        for j in range(6):
            v = mat[i, j]
            if np.isfinite(v) and abs(v) >= annotate_threshold:
                ax.text(
                    j, i,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color="black",
                )

    return im


def draw_kernel_matrix(
    ax,
    mat: np.ndarray,
    vmax: float,
    annotate=True,
    show_xlabel=True,
    show_ylabel=True,
):
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=vmax, aspect="equal")

    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(STATE_LABELS)
    ax.set_yticklabels(STATE_LABELS)

    ax.set_xlabel("To" if show_xlabel else "", labelpad=2.5)
    ax.set_ylabel("From" if show_ylabel else "", labelpad=2.5)

    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.70)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(6):
            for j in range(6):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(
                        j, i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color="black",
                    )

    return im


def draw_support_trends(ax, support: pd.DataFrame):
    support_col = pick_col(support, ["support", "count", "n", "transitions"])
    if support_col is None:
        raise RuntimeError("Cannot find support column in support dataframe.")

    styles = [
        ("native", "front", BLUE, "-"),
        ("native", "local", BLUE, "--"),
        ("pseudo100", "front", RED, "-"),
        ("pseudo100", "local", RED, "--"),
    ]

    for res, kernel, color, ls in styles:
        x = support[(support["resolution"] == res) & (support["kernel"] == kernel)].sort_values("year")
        if x.empty:
            continue

        y = pd.to_numeric(x[support_col], errors="coerce")
        y_smooth = y.rolling(5, center=True, min_periods=2).median()

        ax.plot(
            x["year"],
            y_smooth,
            lw=1.35,
            color=color,
            ls=ls,
            label=f"{res} {kernel}",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Transitions")
    ax.grid(color="0.88", lw=0.70, which="both")
    clean_spines(ax)

    ax.legend(
        frameon=False,
        fontsize=8.3,
        loc="upper left",
        ncol=1,
        handlelength=2.3,
        borderaxespad=0.25,
    )


def draw_magnitude_ratio_panel(ax, mag_df: pd.DataFrame):
    y_positions = np.arange(len(METRIC_ORDER))[::-1]

    for yy, metric in zip(y_positions, METRIC_ORDER):
        sub = mag_df[mag_df["metric"] == metric]

        vn_series = sub.loc[sub["type"] == "pseudo/native", "ratio"]
        vs_series = sub.loc[sub["type"] == "bias/signal", "ratio"]

        vn = float(vn_series.iloc[0]) if not vn_series.empty else np.nan
        vs = float(vs_series.iloc[0]) if not vs_series.empty else np.nan

        if np.isfinite(vn) and np.isfinite(vs):
            ax.hlines(yy, min(vn, vs), max(vn, vs), color="0.72", lw=1.2, zorder=1)

        if np.isfinite(vn):
            ax.scatter(
                vn,
                yy + 0.13,
                s=50,
                marker="o",
                color="0.30",
                edgecolor="black",
                linewidth=0.40,
                zorder=3,
                label=r"$|pseudo100|/|native|$" if yy == y_positions[0] else None,
            )
            ax.annotate(
                f"{vn:.1f}",
                xy=(vn, yy + 0.13),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.4,
            )

        if np.isfinite(vs):
            ax.scatter(
                vs,
                yy - 0.13,
                s=62,
                marker="D",
                color=RED,
                edgecolor="black",
                linewidth=0.40,
                zorder=4,
                label=r"$|bias|/|CMIP6 signal|$" if yy == y_positions[0] else None,
            )
            txt = f"{vs:.1f}" if vs < 10 else f"{vs:.0f}"
            ax.annotate(
                txt,
                xy=(vs, yy - 0.13),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8.4,
            )

    ax.axvline(1.0, color="0.60", lw=0.85, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METRIC_LABEL[m] for m in METRIC_ORDER])
    ax.set_xlabel("Magnitude ratio")
    ax.grid(axis="x", color="0.88", lw=0.70, which="both")
    clean_spines(ax)

    all_vals = mag_df["ratio"].dropna().to_numpy(float)
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    if all_vals.size > 0:
        lo = max(0.35, np.nanquantile(all_vals, 0.01) * 0.55)
        hi = min(250, np.nanquantile(all_vals, 0.99) * 1.85)
        hi = max(hi, 120)
        ax.set_xlim(lo, hi)

    ax.set_ylim(-0.65, len(METRIC_ORDER) - 0.35)

    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.985),
        fontsize=8.1,
        handletextpad=0.4,
        borderaxespad=0.2,
    )


# =============================================================================
# 8. MAIN FIGURE
# =============================================================================

def draw_main_figure(
    counts: pd.DataFrame,
    gradients: pd.DataFrame,
    boot: pd.DataFrame,
    cmip6_summary: pd.DataFrame,
    bias: pd.DataFrame,
):
    f_native = matrix_from_counts(counts, "native", "front")
    l_native = matrix_from_counts(counts, "native", "local")
    f_pseudo = matrix_from_counts(counts, "pseudo100", "front")
    l_pseudo = matrix_from_counts(counts, "pseudo100", "local")

    front_change = f_pseudo - f_native
    contrast_change = (l_pseudo - f_pseudo) - (l_native - f_native)

    vals = np.r_[
        front_change[np.isfinite(front_change)],
        contrast_change[np.isfinite(contrast_change)],
    ]
    vmax = np.nanquantile(np.abs(vals), 0.98) if vals.size else 0.1
    vmax = max(float(vmax), 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    annual_g = annual_gradient_summary(gradients)
    grad_ratio = paired_ratio_table_from_gradients(annual_g)
    phase_df = bias_signal_phase_table(boot, cmip6_summary)

    fig = plt.figure(figsize=(10.4, 7.55))

    outer = GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[1.02, 1.25],
        hspace=0.52,
    )

    top = GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=outer[0, 0],
        width_ratios=[1.18, 1.00, 1.00],
        wspace=0.58,
    )

    bottom = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[1, 0],
        width_ratios=[2.20, 1.00],
        wspace=0.36,
    )

    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    ax_d = fig.add_subplot(bottom[0, 0])
    ax_e = fig.add_subplot(bottom[0, 1])

    add_panel_label(ax_a, "a", x=-0.10, y=1.12, size=17)
    add_panel_label(ax_b, "b", x=-0.14, y=1.12, size=17)
    add_panel_label(ax_c, "c", x=-0.14, y=1.12, size=17)
    add_panel_label(ax_d, "d", x=-0.08, y=1.10, size=17)
    add_panel_label(ax_e, "e", x=-0.12, y=1.10, size=17)

    draw_compact_ratio_distribution(
        ax_a,
        grad_ratio,
        order=[r"$|\nabla SM|$", r"$|\nabla T|$"],
        xlabel="",
        ylabel="Pseudo100 / native",
        xlim=(0.16, 1.25),
    )
    fix_log_ticks(
        ax_a,
        ticks=[0.2, 0.3, 0.5, 1.0],
        labels=["0.2", "0.3", "0.5", "1.0"],
    )

    im = plot_change_matrix(ax_b, front_change, norm=norm, annotate_threshold=0.04)
    plot_change_matrix(ax_c, contrast_change, norm=norm, annotate_threshold=0.04)

    ax_b.set_title("front kernel", fontsize=11.2, pad=11)
    ax_c.set_title("front-local contrast", fontsize=11.2, pad=11)

    divider = make_axes_locatable(ax_c)
    cax_bc = divider.append_axes("right", size="5.2%", pad=0.055)

    cb = fig.colorbar(
        im,
        cax=cax_bc,
        orientation="vertical",
        extend="both",
    )
    cb.set_label("Pseudo100 − native", fontsize=10.0, labelpad=8)
    cb.ax.tick_params(labelsize=9.2, pad=3)

    y = np.arange(len(METRIC_ORDER))[::-1]
    ax_d.axvline(0, color="0.65", lw=0.85, ls="--", zorder=1)

    cm = {r["metric"]: r for _, r in cmip6_summary.iterrows()}

    e_med, e_lo, e_hi = [], [], []
    c_med, c_q25, c_q75 = [], [], []

    for metric in METRIC_ORDER:
        b = boot[boot["metric"] == metric]
        e_med.append(float(b["bias_median"].iloc[0]))
        e_lo.append(float(b["bias_q025"].iloc[0]))
        e_hi.append(float(b["bias_q975"].iloc[0]))

        r = cm[metric]
        c_med.append(float(r["median"]))
        c_q25.append(float(r["q25"]))
        c_q75.append(float(r["q75"]))

    e_med = np.asarray(e_med)
    e_lo = np.asarray(e_lo)
    e_hi = np.asarray(e_hi)
    c_med = np.asarray(c_med)
    c_q25 = np.asarray(c_q25)
    c_q75 = np.asarray(c_q75)

    ax_d.errorbar(
        e_med,
        y + 0.13,
        xerr=[e_med - e_lo, e_hi - e_med],
        fmt="o",
        ms=5.2,
        color=GREEN,
        ecolor=GREEN,
        elinewidth=1.25,
        capsize=2.6,
        label="ERA5 bias",
        zorder=4,
    )

    ax_d.errorbar(
        c_med,
        y - 0.13,
        xerr=[c_med - c_q25, c_q75 - c_med],
        fmt="D",
        ms=5.0,
        color=PURPLE,
        ecolor=PURPLE,
        elinewidth=1.25,
        capsize=2.6,
        label="CMIP6 signal",
        zorder=5,
    )

    ax_d.set_yticks(y)
    ax_d.set_yticklabels([METRIC_LABEL[m] for m in METRIC_ORDER])
    ax_d.set_xlabel("Change in diagnostic")
    ax_d.grid(axis="x", color="0.88", lw=0.70)
    ax_d.set_ylim(-0.55, len(METRIC_ORDER) - 0.45)
    clean_spines(ax_d)

    ax_d.legend(
        frameon=False,
        loc="upper right",
        fontsize=9.0,
        handletextpad=0.5,
        borderaxespad=0.3,
    )

    draw_bias_signal_phase_space(ax_e, phase_df)

    fig.subplots_adjust(
        left=0.090,
        right=0.975,
        top=0.955,
        bottom=0.105,
    )

    out_png = OUT_DIR / "Figure_R5_main_resolution_constraint_v9_layout_fixed.png"
    out_pdf = OUT_DIR / "Figure_R5_main_resolution_constraint_v9_layout_fixed.pdf"

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    log(f"[SAVED] {out_png}")
    log(f"[SAVED] {out_pdf}")


# =============================================================================
# 9. SUPPLEMENTARY FIGURE
# =============================================================================

def draw_supplementary_figure(
    counts: pd.DataFrame,
    support: pd.DataFrame,
    objects: pd.DataFrame,
    bias: pd.DataFrame,
    boot: pd.DataFrame,
    cmip6_summary: pd.DataFrame,
):
    f_native = matrix_from_counts(counts, "native", "front")
    l_native = matrix_from_counts(counts, "native", "local")
    f_pseudo = matrix_from_counts(counts, "pseudo100", "front")
    l_pseudo = matrix_from_counts(counts, "pseudo100", "local")

    mats = [f_native, l_native, f_pseudo, l_pseudo]
    vmax = float(np.nanmax(np.concatenate([m[np.isfinite(m)] for m in mats])))

    sr = support_ratio_table(support)
    orr = object_ratio_table(objects)
    ratio_df = pd.concat([sr, orr], ignore_index=True)

    mag_df = magnitude_ratio_table(bias, boot, cmip6_summary)

    fig = plt.figure(figsize=(10.1, 8.65))

    outer = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.16, 1.00],
        width_ratios=[1.14, 1.00],
        hspace=0.50,
        wspace=0.36,
    )

    gs_a = GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=outer[0, 0],
        width_ratios=[1.0, 1.0, 0.055],
        height_ratios=[1.0, 1.0],
        hspace=0.32,
        wspace=0.28,
    )

    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    ax_a3 = fig.add_subplot(gs_a[1, 0])
    ax_a4 = fig.add_subplot(gs_a[1, 1])
    cax_a = fig.add_subplot(gs_a[:, 2])

    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    add_panel_label(ax_a1, "a", x=-0.26, y=1.24, size=17)
    add_panel_label(ax_b, "b", x=-0.12, y=1.10, size=17)
    add_panel_label(ax_c, "c", x=-0.12, y=1.10, size=17)
    add_panel_label(ax_d, "d", x=-0.12, y=1.10, size=17)

    im = draw_kernel_matrix(
        ax_a1, f_native, vmax,
        show_xlabel=False,
        show_ylabel=True,
    )
    draw_kernel_matrix(
        ax_a2, l_native, vmax,
        show_xlabel=False,
        show_ylabel=False,
    )
    draw_kernel_matrix(
        ax_a3, f_pseudo, vmax,
        show_xlabel=True,
        show_ylabel=True,
    )
    draw_kernel_matrix(
        ax_a4, l_pseudo, vmax,
        show_xlabel=True,
        show_ylabel=False,
    )

    ax_a1.set_title("Front", fontsize=11.2, pad=9)
    ax_a2.set_title("Local", fontsize=11.2, pad=9)

    ax_a1.text(
        -0.57, 0.50, "Native",
        transform=ax_a1.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=10.2,
        clip_on=False,
    )
    ax_a3.text(
        -0.57, 0.50, "Pseudo100",
        transform=ax_a3.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=10.2,
        clip_on=False,
    )

    cb = fig.colorbar(
        im,
        cax=cax_a,
        orientation="vertical",
        extend="both",
    )
    cb.set_label("Transition probability", fontsize=10.0, labelpad=8)
    cb.ax.tick_params(labelsize=9.2, pad=3)

    draw_support_trends(ax_b, support)

    ratio_order = [
        "Front transitions",
        "Local transitions",
        "Object grid-days",
        "Largest component",
    ]
    draw_compact_ratio_distribution(
        ax_c,
        ratio_df,
        order=ratio_order,
        xlabel="Pseudo100 / native",
        ylabel="",
        xlim=None,
    )

    draw_magnitude_ratio_panel(ax_d, mag_df)

    fig.subplots_adjust(
        left=0.095,
        right=0.980,
        top=0.955,
        bottom=0.095,
    )

    out_png = OUT_DIR / "Supplementary_R5_support_kernels_QC_v9_layout_fixed.png"
    out_pdf = OUT_DIR / "Supplementary_R5_support_kernels_QC_v9_layout_fixed.pdf"

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    log(f"[SAVED] {out_png}")
    log(f"[SAVED] {out_pdf}")


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    log("=" * 100)
    log("[INFO] Reviewer comment 5 — final v9 layout-fixed replot")
    log("[INFO] No ERA5 rerun. No raw CMIP6 reread.")
    log("[INFO] Main figure and supplementary figure will both be generated.")
    log(f"[INFO] ERA5 diagnostics: {ERA5_DIAG_DIR}")
    log(f"[INFO] Output directory: {OUT_DIR}")
    log("=" * 100)

    counts, support, gradients, objects = load_era5_diagnostics()
    cmip6_counts = load_cmip6_counts()

    log("[BUILD] Lightweight ERA5 weighted metrics")
    annual_m = annual_era5_metrics(counts)
    full_m = full_era5_metrics(counts)
    bias = era5_bias_from_full(full_m)
    boot = fast_bootstrap_era5_bias(annual_m)

    log("[BUILD] Lightweight CMIP6 weighted metrics from existing counts")
    cmip6_metrics, cmip6_signal, cmip6_summary = cmip6_metrics_and_signal(cmip6_counts)

    annual_m.to_csv(
        OUT_DIR / "era5_weighted_annual_metrics_v9_layout_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    full_m.to_csv(
        OUT_DIR / "era5_weighted_full_metrics_v9_layout_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bias.to_csv(
        OUT_DIR / "era5_weighted_bias_v9_layout_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boot.to_csv(
        OUT_DIR / "era5_weighted_bootstrap_bias_v9_layout_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cmip6_metrics.to_csv(
        OUT_DIR / "cmip6_weighted_metrics_v9_layout_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cmip6_signal.to_csv(
        OUT_DIR / "cmip6_weighted_signal_v9_layout_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cmip6_summary.to_csv(
        OUT_DIR / "cmip6_weighted_signal_summary_v9_layout_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("[DRAW] Main figure")
    draw_main_figure(
        counts=counts,
        gradients=gradients,
        boot=boot,
        cmip6_summary=cmip6_summary,
        bias=bias,
    )

    log("[DRAW] Supplementary figure")
    draw_supplementary_figure(
        counts=counts,
        support=support,
        objects=objects,
        bias=bias,
        boot=boot,
        cmip6_summary=cmip6_summary,
    )

    log("=" * 100)
    log("[DONE]")
    log(f"[DONE] Figures saved to: {OUT_DIR}")
    log("=" * 100)


if __name__ == "__main__":
    main()