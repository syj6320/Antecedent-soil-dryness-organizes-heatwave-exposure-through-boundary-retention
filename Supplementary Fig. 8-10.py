#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nature-style clean rebuild for Figure 4, Figure 5, Supplementary Fig. 9-11.

This script follows the layout and typography logic of:
    figure1_final_revised_userfix_v3.py

Scientific restructuring:
1. Figure 4 = front/local transition-kernel mechanism only.
2. Figure 5 = transition-speed mechanism only.
3. Supplementary Fig. 9 = mobility support only.
4. Supplementary Fig. 10 = kernel climatology and sample-support diagnostics.
5. Supplementary Fig. 11 = transition-speed support diagnostics.
6. Remove duplicated spatial maps and low-information panels.
7. Use continuous annual trends, not early/late contrasts.
8. Use dedicated colorbar rows for maps and heatmaps.
9. Keep fonts close to Figure 1 final style.

Required cache inputs:
    <root>\\_result2_nature_style_raw
    <root>\\_result3_event_speed
    <root>\\_result3_transition_kernel
    <root>\\_NCC_Result2_Result3_editorial_compact_v3_continuous\\cache_result3_event_speed_nativegrid_annual.csv
    <root>\\_NCC_Result2_Result3_editorial_compact_v3_continuous\\cache_result3_transition_spatial_counts_annual.csv

Run:
    D:\\nature\\venv\\Scripts\\python.exe rebuild_Result2_Result3_editorial_compact_v5_nature_style.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
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
# 1. PATHS AND CONSTANTS
# =============================================================================

DEFAULT_ROOT = Path(
    r"E:\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本"
    r"\events_cc3d_with_precip_H_LE_CAPE_IVTDIV_T850_WIND_RH"
)

CONUS_EXTENT = (-125, -66, 24, 50.5)

STATE_ORDER = [1, 2, 3, 4, 5, 6]
STATE_LABELS = [f"S{i}" for i in STATE_ORDER]

STATE_COLORS = {
    1: "#8c510a",
    2: "#bf812d",
    3: "#dfc27d",
    4: "#80cdc1",
    5: "#35978f",
    6: "#01665e",
}

KERNEL_COLORS = {
    "front": "#D95F02",
    "local": "#1B9E77",
}

DPI = 500
ROLLING_WINDOW = 7
RANDOM_SEED = 20260430


# =============================================================================
# 2. STYLE — MATCH FIGURE 1 FINAL VERSION
# =============================================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 22,
    "axes.titlesize": 25,
    "axes.labelsize": 24,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "legend.fontsize": 18,
    "axes.linewidth": 1.35,
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


# =============================================================================
# 3. BASIC UTILITIES
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, letter: str, x: float = -0.13, y: float = 1.08, size: int = 28) -> None:
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=size,
        fontweight="bold",
        clip_on=False,
        zorder=30,
    )


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


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "P=NA"
    if p < 0.001:
        return "P<0.001"
    return f"P={p:.3f}"


def state_to_int(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return pd.to_numeric(
            s.astype(str).str.replace("S", "", regex=False),
            errors="coerce"
        )
    return pd.to_numeric(s, errors="coerce")


def rolling_mean(y, window: int = ROLLING_WINDOW):
    return (
        pd.Series(y, dtype=float)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def lintrend_per_decade(x, y) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 8:
        return np.nan, np.nan, np.nan

    x = x[m]
    y = y[m]

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0, np.nan, np.nan

    if stats is not None:
        res = stats.linregress(x, y)
        return float(res.slope * 10.0), float(res.pvalue), float(res.rvalue)

    X = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = np.corrcoef(x, y)[0, 1]
    return float(beta[1] * 10.0), np.nan, float(r)


def spearman_stat(x, y) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 8:
        return np.nan, np.nan

    if stats is not None:
        r, p = stats.spearmanr(x[m], y[m])
        return float(r), float(p)

    xr = pd.Series(x[m]).rank().values
    yr = pd.Series(y[m]).rank().values
    return float(np.corrcoef(xr, yr)[0, 1]), np.nan


def bootstrap_ci(values, func=np.nanmedian, n_boot: int = 1000, seed: int = RANDOM_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 5:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = []

    for _ in range(n_boot):
        boots.append(func(rng.choice(values, size=len(values), replace=True)))

    boots = np.asarray(boots)

    return (
        float(func(values)),
        float(np.nanpercentile(boots, 2.5)),
        float(np.nanpercentile(boots, 97.5)),
    )


def running_bin_summary(x, y, nbins: int = 18, min_n: int = 30):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < min_n:
        return pd.DataFrame(columns=["x", "median", "q25", "q75", "n"])

    edges = np.unique(np.nanquantile(x, np.linspace(0, 1, nbins + 1)))
    rows = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mm = (x >= lo) & (x <= hi)
        if mm.sum() < min_n:
            continue
        rows.append({
            "x": float(np.nanmedian(x[mm])),
            "median": float(np.nanmedian(y[mm])),
            "q25": float(np.nanpercentile(y[mm], 25)),
            "q75": float(np.nanpercentile(y[mm], 75)),
            "n": int(mm.sum()),
        })

    return pd.DataFrame(rows)


def savefig(fig, out_dir: Path, name: str) -> None:
    ensure_dir(out_dir)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")


def apply_sci_y(ax, powerlimits=(4, 4)) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(powerlimits)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=powerlimits)
    ax.yaxis.get_offset_text().set_fontsize(15)


# =============================================================================
# 4. MAP HELPERS — FIGURE 1 STYLE
# =============================================================================

def infer_edges(vals):
    vals = np.asarray(sorted(np.unique(vals)), dtype=float)

    if len(vals) == 0:
        return np.array([])

    if len(vals) == 1:
        d = 0.25
        return np.array([vals[0] - d / 2, vals[0] + d / 2])

    mid = (vals[:-1] + vals[1:]) / 2
    first = vals[0] - (mid[0] - vals[0])
    last = vals[-1] + (vals[-1] - mid[-1])

    return np.r_[first, mid, last]


def make_geo_ax(fig, spec):
    if HAS_CARTOPY:
        ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())

        ax.add_feature(
            cfeature.COASTLINE.with_scale("50m"),
            linewidth=0.55,
            edgecolor="0.55"
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            linewidth=0.45,
            edgecolor="0.55"
        )

        try:
            states = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces_lines",
                scale="50m",
                facecolor="none"
            )
            ax.add_feature(states, linewidth=0.35, edgecolor="0.70")
        except Exception:
            pass

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.45,
            color="0.80",
            alpha=0.75,
            linestyle="-"
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
    ax.set_xticklabels(["120°W", "110°W", "100°W", "90°W", "80°W", "70°W"])
    ax.set_yticks([25, 35, 45])
    ax.set_yticklabels(["25°N", "35°N", "45°N"])
    ax.set_facecolor("#f7f7f7")
    ax.grid(color="0.86", linewidth=0.55)
    clean_spines(ax)

    return ax


def plot_grid_map(
    ax,
    df: pd.DataFrame,
    value_col: str,
    cmap,
    norm=None,
    vmin=None,
    vmax=None,
    lon_col="lon",
    lat_col="lat",
    point_size: float = 5.5,
    alpha: float = 0.96,
):
    d = df[[lon_col, lat_col, value_col]].dropna().copy()

    if d.empty:
        return None

    d = d.rename(columns={lon_col: "lon", lat_col: "lat"})
    d = d.groupby(["lon", "lat"], as_index=False)[value_col].mean()

    lons = np.sort(d["lon"].unique())
    lats = np.sort(d["lat"].unique())

    transform = ccrs.PlateCarree() if HAS_CARTOPY else None

    if len(lons) * len(lats) <= len(d) * 1.30 and len(lons) > 1 and len(lats) > 1:
        pivot = d.pivot_table(index="lat", columns="lon", values=value_col)
        pivot = pivot.reindex(index=lats, columns=lons)

        X = infer_edges(lons)
        Y = infer_edges(lats)
        Z = pivot.values

        kwargs = dict(
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

        if HAS_CARTOPY:
            kwargs["transform"] = transform

        im = ax.pcolormesh(X, Y, Z, **kwargs)

    else:
        kwargs = dict(
            c=d[value_col],
            s=point_size,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=alpha,
            rasterized=True,
        )

        if HAS_CARTOPY:
            kwargs["transform"] = transform

        im = ax.scatter(d["lon"], d["lat"], **kwargs)

    return im


def horizontal_cbar(fig, cax, im, label: str, extend="neither"):
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend)
    cb.set_label(label, fontsize=20)
    cb.ax.tick_params(labelsize=18)
    return cb


# =============================================================================
# 5. LOAD CACHES
# =============================================================================

def default_dirs(root: Path) -> Dict[str, Path]:
    return {
        "result2": root / "_result2_nature_style_raw",
        "speed": root / "_result3_event_speed",
        "kernel": root / "_result3_transition_kernel",
        "v3": root / "_NCC_Result2_Result3_editorial_compact_v3_continuous",
    }


def require_file(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {desc}:\n{path}\n\n"
            "This plotting script reads existing caches only. "
            "Run the previous cache-building script once if this file is absent.\n"
        )


def load_result2(result2_dir: Path) -> Dict[str, pd.DataFrame]:
    event_csv = result2_dir / "result2_event_summary_raw.csv"
    seg_csv = result2_dir / "result2_segments_raw.csv.gz"

    require_file(event_csv, "Result 2 event summary")
    require_file(seg_csv, "Result 2 movement segment table")

    ev = pd.read_csv(event_csv, low_memory=False)
    seg = pd.read_csv(seg_csv, low_memory=False)

    if "state_index" not in ev.columns and "start_state" in ev.columns:
        ev["state_index"] = state_to_int(ev["start_state"])

    if "state_index" not in seg.columns and "start_state" in seg.columns:
        seg["state_index"] = state_to_int(seg["start_state"])

    return {"events": ev, "segments": seg}


def load_speed(speed_dir: Path) -> Dict[str, pd.DataFrame]:
    ev_csv = speed_dir / "result3_event_speed_summary.csv"
    sp_csv = speed_dir / "result3_event_speed_nativegrid.csv"

    require_file(ev_csv, "Result 3 event-speed event summary")
    require_file(sp_csv, "Result 3 event-speed spatial table")

    ev = pd.read_csv(ev_csv, low_memory=False)
    sp = pd.read_csv(sp_csv, low_memory=False)

    return {"events": ev, "spatial": sp}


def metric_from_counts(sub: pd.DataFrame) -> Dict[str, float]:
    out = {
        "p11": np.nan,
        "p21": np.nan,
        "p31": np.nan,
        "delta": np.nan,
        "support": 0.0,
    }

    if sub.empty:
        return out

    support = float(sub["count"].sum())
    out["support"] = support

    for s in [1, 2, 3]:
        row = sub[sub["from_state"] == s]
        denom = float(row["count"].sum())

        if denom > 0:
            num = float(row.loc[row["to_state"] == 1, "count"].sum())
            out["p11" if s == 1 else f"p{s}1"] = num / denom

    dry = sub.loc[sub["to_state"] < sub["from_state"], "count"].sum()
    wet = sub.loc[sub["to_state"] > sub["from_state"], "count"].sum()

    if support > 0:
        out["delta"] = float((dry - wet) / support)

    return out


def build_yearly_metrics(annual_counts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (year, kernel), sub in annual_counts.groupby(["year", "kernel"], observed=True):
        m = metric_from_counts(sub)
        m.update({"year": year, "kernel": kernel, "mode": "raw"})
        rows.append(m)

    return pd.DataFrame(rows)


def load_kernel(kernel_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "annual_counts": kernel_dir / "result3_annual_counts.csv",
        "composition_regime": kernel_dir / "result3_composition_regime.csv",
        "composition_state": kernel_dir / "result3_composition_state.csv",
        "spatial_counts": kernel_dir / "result3_spatial_counts.csv",
        "event_coev": kernel_dir / "result3_event_coevolution.csv",
        "start_counts": kernel_dir / "result3_start_counts.csv",
    }

    for key, path in files.items():
        require_file(path, f"Transition-kernel table: {key}")

    out = {key: pd.read_csv(path, low_memory=False) for key, path in files.items()}

    yearly_csv = kernel_dir / "result3_yearly_metrics.csv"

    if yearly_csv.exists():
        out["yearly"] = pd.read_csv(yearly_csv, low_memory=False)
    else:
        out["yearly"] = build_yearly_metrics(out["annual_counts"])
        out["yearly"].to_csv(yearly_csv, index=False, encoding="utf-8-sig")

    return out


def load_annual_spatial_caches(v3_dir: Path):
    speed_cache = v3_dir / "cache_result3_event_speed_nativegrid_annual.csv"
    transition_cache = v3_dir / "cache_result3_transition_spatial_counts_annual.csv"

    require_file(speed_cache, "Annual native-grid speed cache")
    require_file(transition_cache, "Annual transition spatial-count cache")

    sp_ann = pd.read_csv(speed_cache, low_memory=False)
    trans_ann = pd.read_csv(transition_cache, low_memory=False)

    return sp_ann, trans_ann


# =============================================================================
# 6. MATRIX AND TREND HELPERS
# =============================================================================

def climatology_matrix(annual_counts: pd.DataFrame, kernel: str) -> np.ndarray:
    sub = annual_counts[annual_counts["kernel"] == kernel].copy()

    mat = np.full((6, 6), np.nan)

    for s in STATE_ORDER:
        row = sub[sub["from_state"] == s].groupby("to_state")["count"].sum()
        denom = row.sum()

        if denom > 0:
            mat[s - 1, :] = [row.get(t, 0.0) / denom for t in STATE_ORDER]

    return mat


def annual_probability_trend_matrix(annual_counts: pd.DataFrame, kernel: str):
    sub = annual_counts[annual_counts["kernel"] == kernel].copy()

    mat = np.full((6, 6), np.nan)
    pmat = np.full((6, 6), np.nan)

    for s in STATE_ORDER:
        one = sub[sub["from_state"] == s].copy()

        if one.empty:
            continue

        total = one.groupby("year")["count"].sum()

        for t in STATE_ORDER:
            cell = one[one["to_state"] == t].groupby("year")["count"].sum()
            yy = pd.DataFrame({"total": total, "cell": cell}).fillna(0).reset_index()
            yy["prob"] = np.where(yy["total"] > 0, yy["cell"] / yy["total"], np.nan)

            slope, p, _ = lintrend_per_decade(yy["year"], yy["prob"])

            mat[s - 1, t - 1] = slope
            pmat[s - 1, t - 1] = p

    return mat, pmat


def plot_kernel_matrix(
    ax,
    mat,
    title,
    cmap,
    norm=None,
    vmin=None,
    vmax=None,
    pmat=None,
    fmt="{:.2f}",
    text_size=10,
):
    im = ax.imshow(
        mat,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(STATE_LABELS)
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(STATE_LABELS)

    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(title)

    for i in range(6):
        for j in range(6):
            if np.isfinite(mat[i, j]):
                star = p_to_star(pmat[i, j]) if pmat is not None else ""
                ax.text(
                    j, i,
                    fmt.format(mat[i, j]) + star,
                    ha="center",
                    va="center",
                    fontsize=text_size,
                    color="black",
                )

    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def spatial_trend_from_annual_speed(
    sp_ann: pd.DataFrame,
    kernel: str,
    value_col: str,
    min_years: int = 15,
    min_total_n: int = 50,
) -> pd.DataFrame:
    sub = sp_ann[sp_ann["kernel"] == kernel].copy()
    rows = []

    for (lon, lat), ss in sub.groupby(["lon", "lat"]):
        ss = ss.sort_values("year")

        if ss["year"].nunique() < min_years:
            continue

        if "n" in ss.columns and ss["n"].sum() < min_total_n:
            continue

        slope, p, r = lintrend_per_decade(ss["year"], ss[value_col])

        rows.append({
            "lon": lon,
            "lat": lat,
            "trend": slope,
            "p": p,
            "r": r,
            "support": ss["n"].sum() if "n" in ss.columns else np.nan,
        })

    return pd.DataFrame(rows)


def front_minus_local_speed_trend_map(sp_ann: pd.DataFrame) -> pd.DataFrame:
    front = spatial_trend_from_annual_speed(sp_ann, "front", "drying_speed")
    local = spatial_trend_from_annual_speed(sp_ann, "local", "drying_speed")

    if front.empty or local.empty:
        return pd.DataFrame()

    front = front.rename(columns={"trend": "front_trend"})
    local = local.rename(columns={"trend": "local_trend"})

    out = front[["lon", "lat", "front_trend"]].merge(
        local[["lon", "lat", "local_trend"]],
        on=["lon", "lat"],
        how="inner",
    )

    out["front_minus_local_trend"] = out["front_trend"] - out["local_trend"]

    return out


def spatial_support_trend_from_annual_transition(
    trans_ann: pd.DataFrame,
    kernel: str,
    min_years: int = 15,
) -> pd.DataFrame:
    sub = trans_ann[trans_ann["kernel"] == kernel].copy()

    yearly = (
        sub.groupby(["lon", "lat", "year"], as_index=False)["count"]
        .sum()
    )

    rows = []

    for (lon, lat), ss in yearly.groupby(["lon", "lat"]):
        ss = ss.sort_values("year")

        if ss["year"].nunique() < min_years:
            continue

        slope, p, r = lintrend_per_decade(
            ss["year"],
            np.log10(ss["count"].clip(lower=1)),
        )

        rows.append({
            "lon": lon,
            "lat": lat,
            "trend": slope,
            "p": p,
            "r": r,
            "support": ss["count"].sum(),
        })

    return pd.DataFrame(rows)


# =============================================================================
# 7. FIGURE 4 — KERNEL MECHANISM ONLY
# =============================================================================

def draw_figure4(kernel: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    annual = kernel["annual_counts"].copy()
    yearly = kernel["yearly"].copy()
    coev = kernel["event_coev"].copy()

    for col in ["year", "from_state", "to_state"]:
        if col in annual.columns:
            annual[col] = pd.to_numeric(annual[col], errors="coerce").astype(int)

    front_trend, front_p = annual_probability_trend_matrix(annual, "front")
    local_trend, local_p = annual_probability_trend_matrix(annual, "local")

    fig = plt.figure(figsize=(18.6, 10.8))
    gs = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.12],
        width_ratios=[1.05, 1.05],
        hspace=0.42,
        wspace=0.34,
    )

    # a: annual metrics
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a", x=-0.12, y=1.10)

    sub = yearly.copy()
    if "mode" in sub.columns and sub["mode"].astype(str).str.lower().eq("standardized").any():
        sub = sub[sub["mode"].astype(str).str.lower().eq("standardized")].copy()

    line_specs = [
        ("p11", "P(S1→S1)", "-"),
        ("p31", "P(S3→S1)", "--"),
        ("delta", "Drying tendency", ":"),
    ]

    for metric, label_base, ls in line_specs:
        if metric not in sub.columns:
            continue

        for kernel_name in ["front", "local"]:
            ss = sub[sub["kernel"] == kernel_name].sort_values("year")

            if ss.empty:
                continue

            slope, p, _ = lintrend_per_decade(ss["year"], ss[metric])

            ax.plot(
                ss["year"],
                rolling_mean(ss[metric]),
                color=KERNEL_COLORS[kernel_name],
                lw=2.3,
                ls=ls,
                label=f"{label_base}, {kernel_name} ({slope:.3f}{p_to_star(p)})",
            )

    ax.set_title("Continuous annual transition metrics")
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability or tendency")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=10.8, loc="upper left", handlelength=2.6)

    # b: event contrast
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "b", x=-0.12, y=1.10)

    if not coev.empty and {"front_delta_event", "local_delta_event"}.issubset(coev.columns):
        d = coev.copy()

        if {"front_support", "local_support"}.issubset(d.columns):
            d = d[(d["front_support"] >= 10) & (d["local_support"] >= 10)].copy()

        d["start_state_num"] = state_to_int(d["start_state"])
        d["front_minus_local_delta"] = d["front_delta_event"] - d["local_delta_event"]

        vals = [
            d.loc[d["start_state_num"] == s, "front_minus_local_delta"].dropna().values
            for s in STATE_ORDER
        ]

        bp = ax.boxplot(
            vals,
            positions=STATE_ORDER,
            widths=0.65,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", lw=1.35),
            boxprops=dict(linewidth=1.05),
            whiskerprops=dict(linewidth=1.05),
            capprops=dict(linewidth=1.05),
        )

        for patch, s in zip(bp["boxes"], STATE_ORDER):
            patch.set_facecolor(STATE_COLORS[s])
            patch.set_alpha(0.72)

        ax.axhline(0, color="0.60", lw=1.0, ls="--")

        if "mean_EF" in d.columns:
            rho, p = spearman_stat(d["mean_EF"], d["front_minus_local_delta"])
            ax.text(
                0.05,
                0.95,
                f"EF vs front−local\nρ={rho:.2f}, {format_p(p)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.88),
            )

    ax.set_title("Event-level front–local contrast")
    ax.set_xlabel("Initial soil-moisture state")
    ax.set_ylabel("Front − local drying tendency")
    ax.set_xticks(STATE_ORDER)
    ax.set_xticklabels(STATE_LABELS)
    ax.grid(axis="y", color="0.88", linewidth=0.9)
    clean_spines(ax)

    # c: front trend matrix
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[2, 0])
    add_panel_label(ax, "c", x=-0.12, y=1.10)

    lim = np.nanquantile(np.abs(front_trend), 0.98)
    if not np.isfinite(lim) or lim == 0:
        lim = 0.01

    im = plot_kernel_matrix(
        ax,
        front_trend,
        "Front kernel annual trend",
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
        pmat=front_p,
        fmt="{:.3f}",
        text_size=9,
    )

    horizontal_cbar(
        fig,
        cax,
        im,
        "Trend in transition probability decade$^{-1}$",
        extend="both",
    )

    # d: local trend matrix
    ax = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[2, 1])
    add_panel_label(ax, "d", x=-0.12, y=1.10)

    lim = np.nanquantile(np.abs(local_trend), 0.98)
    if not np.isfinite(lim) or lim == 0:
        lim = 0.01

    im = plot_kernel_matrix(
        ax,
        local_trend,
        "Local kernel annual trend",
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
        pmat=local_p,
        fmt="{:.3f}",
        text_size=9,
    )

    horizontal_cbar(
        fig,
        cax,
        im,
        "Trend in transition probability decade$^{-1}$",
        extend="both",
    )

    savefig(fig, out_dir, "Figure4_front_local_transition_kernel_editorial")


# =============================================================================
# 8. FIGURE 5 — TRANSITION SPEED MECHANISM ONLY
# =============================================================================

def draw_figure5(speed: Dict[str, pd.DataFrame], sp_ann: pd.DataFrame, out_dir: Path) -> None:
    ev = speed["events"].copy()
    ev["start_state"] = state_to_int(ev["start_state"])

    fig = plt.figure(figsize=(18.6, 10.6))
    gs = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.12],
        width_ratios=[1, 1],
        hspace=0.45,
        wspace=0.32,
    )

    # a
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a", x=-0.12, y=1.10)

    for kernel_name in ["front", "local"]:
        col = f"{kernel_name}_drying_speed_mean"
        tr = f"{kernel_name}_transitions"

        sub = ev[ev[tr] > 0].groupby("year", as_index=False)[col].median()
        slope, p, _ = lintrend_per_decade(sub["year"], sub[col])

        ax.plot(
            sub["year"],
            rolling_mean(sub[col]),
            color=KERNEL_COLORS[kernel_name],
            lw=2.6,
            label=f"{kernel_name.capitalize()} ({slope:.3f}{p_to_star(p)})",
        )

    ax.set_title("Annual drying-transition speed")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median drying speed")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=17)

    # b
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "b", x=-0.12, y=1.10)

    rows = []

    for s in STATE_ORDER:
        sub = ev[ev["start_state"] == s]

        for kernel_name in ["front", "local"]:
            col = f"{kernel_name}_drying_speed_mean"
            tr = f"{kernel_name}_transitions"

            vals = sub.loc[sub[tr] > 0, col]

            med, lo, hi = bootstrap_ci(
                vals,
                func=np.nanmedian,
                seed=RANDOM_SEED + s + (0 if kernel_name == "front" else 20),
            )

            rows.append({
                "state": s,
                "kernel": kernel_name,
                "med": med,
                "lo": lo,
                "hi": hi,
            })

    rr = pd.DataFrame(rows)

    for kernel_name, offset, marker in [
        ("front", -0.09, "o"),
        ("local", 0.09, "s"),
    ]:
        sub = rr[rr["kernel"] == kernel_name]

        ax.errorbar(
            sub["state"] + offset,
            sub["med"],
            yerr=[
                sub["med"] - sub["lo"],
                sub["hi"] - sub["med"],
            ],
            fmt=marker,
            color=KERNEL_COLORS[kernel_name],
            ecolor=KERNEL_COLORS[kernel_name],
            ms=7.5,
            lw=2.1,
            capsize=3,
            label=kernel_name.capitalize(),
        )

    ax.set_title("Event-level drying-speed support")
    ax.set_xlabel("Initial state")
    ax.set_ylabel("Median drying speed")
    ax.set_xticks(STATE_ORDER)
    ax.set_xticklabels(STATE_LABELS)
    ax.grid(axis="y", color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, loc="upper left")

    # c: front-minus-local trend map
    axm = make_geo_ax(fig, gs[1, :])
    cax = fig.add_subplot(gs[2, :])
    add_panel_label(axm, "c", x=-0.07, y=1.08)

    tm = front_minus_local_speed_trend_map(sp_ann)

    if not tm.empty:
        lim = np.nanquantile(np.abs(tm["front_minus_local_trend"]), 0.985)
        if not np.isfinite(lim) or lim == 0:
            lim = 0.05

        im = plot_grid_map(
            axm,
            tm,
            "front_minus_local_trend",
            cmap="RdBu_r",
            norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
            point_size=6.0,
        )

        horizontal_cbar(
            fig,
            cax,
            im,
            "Trend in front−local drying speed decade$^{-1}$",
            extend="both",
        )

    axm.set_title("Annual trend in front-minus-local drying speed")

    savefig(fig, out_dir, "Figure5_front_local_transition_speed_editorial")


# =============================================================================
# 9. SUPPLEMENTARY FIG. 9 — MOBILITY SUPPORT ONLY
# =============================================================================

def draw_supp9(result2: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    ev = result2["events"].copy()
    seg = result2["segments"].copy()

    if "state_index" not in ev.columns:
        ev["state_index"] = state_to_int(ev["start_state"])

    if "state_index" not in seg.columns and "start_state" in seg.columns:
        seg["state_index"] = state_to_int(seg["start_state"])

    fig = plt.figure(figsize=(18.6, 10.5))
    gs = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1.0, 0.12, 1.0],
        hspace=0.44,
        wspace=0.32,
    )

    # a
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a", x=-0.12, y=1.10)

    for s in STATE_ORDER:
        sub = ev[ev["state_index"] == s].groupby("year", as_index=False)["moving_flag"].mean()

        if sub.empty:
            continue

        slope, p, _ = lintrend_per_decade(sub["year"], sub["moving_flag"])

        ax.plot(
            sub["year"],
            rolling_mean(sub["moving_flag"]),
            color=STATE_COLORS[s],
            lw=2.1,
            label=f"S{s} ({slope:.3f}{p_to_star(p)})",
        )

    ax.set_title("Moving-event fraction")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction moving")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=12.5, ncol=2, loc="upper left")

    # b
    axm = make_geo_ax(fig, gs[0, 1])
    cax = fig.add_subplot(gs[1, 1])
    add_panel_label(axm, "b", x=-0.12, y=1.08)

    if not seg.empty and {"mid_lon", "mid_lat"}.issubset(seg.columns):
        sub = seg[seg["state_index"].isin([1, 2])].copy()

        if "year" in sub.columns:
            sub = sub[sub["year"] >= 2000].copy()

        if len(sub) > 90000:
            sub = sub.sample(90000, random_state=RANDOM_SEED)

        kwargs = dict(
            gridsize=55,
            extent=CONUS_EXTENT,
            mincnt=1,
            cmap="YlOrRd",
            bins="log",
        )

        if HAS_CARTOPY:
            kwargs["transform"] = ccrs.PlateCarree()

        hb = axm.hexbin(sub["mid_lon"], sub["mid_lat"], **kwargs)

        horizontal_cbar(fig, cax, hb, "Dry-start segment density")

    axm.set_title("Dry-start mobility corridors")

    # c
    ax = fig.add_subplot(gs[2, :])
    add_panel_label(ax, "c", x=-0.07, y=1.10)

    te = ev[["path_length_km", "net_displacement_km", "state_index"]].dropna().copy()
    te = te[(te["path_length_km"] > 0) & (te["net_displacement_km"] >= 0)]

    plot_d = te.sample(7000, random_state=RANDOM_SEED) if len(te) > 7000 else te

    for s in STATE_ORDER:
        ss = plot_d[plot_d["state_index"] == s]

        if not ss.empty:
            ax.scatter(
                ss["path_length_km"],
                ss["net_displacement_km"],
                s=12,
                color=STATE_COLORS[s],
                alpha=0.22,
                linewidths=0,
                rasterized=True,
            )

    max_lim = np.nanpercentile(te[["path_length_km", "net_displacement_km"]].values, 99.2)
    max_lim = max(50, max_lim)

    ax.plot([0, max_lim], [0, max_lim], ls="--", color="0.35", lw=1.3)

    rho, p = spearman_stat(te["path_length_km"], te["net_displacement_km"])

    ax.text(
        0.04,
        0.94,
        f"Spearman ρ={rho:.2f}\n{format_p(p)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.88),
    )

    ax.set_xlim(0, max_lim)
    ax.set_ylim(0, max_lim)
    ax.set_title("Trajectory efficiency")
    ax.set_xlabel("Path length (km)")
    ax.set_ylabel("Net displacement (km)")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)

    savefig(fig, out_dir, "Supplementary_Fig9_mobility_reorganization_support_editorial")


# =============================================================================
# 10. SUPPLEMENTARY FIG. 10 — KERNEL CLIMATOLOGY AND SUPPORT
# =============================================================================

def draw_supp10(kernel: Dict[str, pd.DataFrame], trans_ann: pd.DataFrame, out_dir: Path) -> None:
    annual = kernel["annual_counts"].copy()
    start = kernel["start_counts"].copy()

    for col in ["year", "from_state", "to_state"]:
        if col in annual.columns:
            annual[col] = pd.to_numeric(annual[col], errors="coerce").astype(int)

    front_clim = climatology_matrix(annual, "front")
    local_clim = climatology_matrix(annual, "local")

    fig = plt.figure(figsize=(18.6, 14.0))
    gs = GridSpec(
        5, 2,
        figure=fig,
        height_ratios=[1.0, 0.12, 1.0, 1.0, 0.12],
        hspace=0.48,
        wspace=0.32,
    )

    # a
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, "a", x=-0.12, y=1.10)

    im = plot_kernel_matrix(
        ax,
        front_clim,
        "Front kernel climatology",
        cmap="YlGnBu",
        vmin=0,
        vmax=np.nanquantile(front_clim, 0.99),
        fmt="{:.2f}",
        text_size=10,
    )

    horizontal_cbar(fig, cax, im, "Transition probability")

    # b
    ax = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "b", x=-0.12, y=1.10)

    im = plot_kernel_matrix(
        ax,
        local_clim,
        "Local kernel climatology",
        cmap="YlGnBu",
        vmin=0,
        vmax=np.nanquantile(local_clim, 0.99),
        fmt="{:.2f}",
        text_size=10,
    )

    horizontal_cbar(fig, cax, im, "Transition probability")

    # c
    ax = fig.add_subplot(gs[2, 0])
    add_panel_label(ax, "c", x=-0.12, y=1.10)

    for kernel_name in ["front", "local"]:
        sub = (
            annual[annual["kernel"] == kernel_name]
            .groupby("year", as_index=False)["count"]
            .sum()
        )

        slope, p, _ = lintrend_per_decade(
            sub["year"],
            np.log10(sub["count"].clip(lower=1)),
        )

        ax.plot(
            sub["year"],
            rolling_mean(sub["count"]),
            color=KERNEL_COLORS[kernel_name],
            lw=2.5,
            label=f"{kernel_name.capitalize()} ({slope:.3f}{p_to_star(p)})",
        )

    ax.set_title("Annual transition support")
    ax.set_xlabel("Year")
    ax.set_ylabel("Transitions")
    apply_sci_y(ax)
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, loc="upper left")

    # d
    ax = fig.add_subplot(gs[2, 1])
    add_panel_label(ax, "d", x=-0.12, y=1.10)

    st = (
        start.groupby(["year", "start_state"], observed=True)["n_events"]
        .sum()
        .reset_index()
    )

    for s in STATE_ORDER:
        sub = st[st["start_state"] == s]

        if sub.empty:
            continue

        ax.plot(
            sub["year"],
            rolling_mean(sub["n_events"]),
            color=STATE_COLORS[s],
            lw=2.1,
            label=f"S{s}",
        )

    ax.set_title("Start-state sample support")
    ax.set_xlabel("Year")
    ax.set_ylabel("Events")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=12.5, ncol=2)

    # e
    axm = make_geo_ax(fig, gs[3, :])
    cax = fig.add_subplot(gs[4, :])
    add_panel_label(axm, "e", x=-0.07, y=1.08)

    tm = spatial_support_trend_from_annual_transition(trans_ann, "front")

    if not tm.empty:
        lim = np.nanquantile(np.abs(tm["trend"]), 0.985)
        if not np.isfinite(lim) or lim == 0:
            lim = 0.05

        im = plot_grid_map(
            axm,
            tm,
            "trend",
            cmap="RdBu_r",
            norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
            point_size=6.0,
        )

        horizontal_cbar(
            fig,
            cax,
            im,
            r"Trend in $\log_{10}$(transition support) decade$^{-1}$",
            extend="both",
        )

    axm.set_title("Front-kernel support trend")

    savefig(fig, out_dir, "Supplementary_Fig10_transition_kernel_support_robustness_editorial")


# =============================================================================
# 11. SUPPLEMENTARY FIG. 11 — SPEED SUPPORT ONLY
# =============================================================================

def draw_continuous_support_panel(
    ax,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    xlabel: str,
    logx: bool = False,
    logy: bool = True,
) -> None:
    d = df[[xcol, ycol, "start_state"]].dropna().copy()
    d = d[d[ycol] > 0].copy()

    if logx:
        d = d[d[xcol] > 0].copy()
        d["_x"] = np.log10(d[xcol])
        xlabel_use = r"$\log_{10}$(" + xlabel + ")"
    else:
        d = d[d[xcol] > 0].copy()
        d["_x"] = d[xcol]
        xlabel_use = xlabel

    if logy:
        d["_y"] = np.log10(d[ycol].clip(lower=1))
        ylabel_use = r"$\log_{10}$(transition support)"
    else:
        d["_y"] = d[ycol]
        ylabel_use = "Transition support"

    plot_d = d.sample(5000, random_state=RANDOM_SEED) if len(d) > 5000 else d

    for s in STATE_ORDER:
        ss = plot_d[pd.to_numeric(plot_d["start_state"], errors="coerce") == s]

        if ss.empty:
            continue

        ax.scatter(
            ss["_x"],
            ss["_y"],
            s=10,
            color=STATE_COLORS[s],
            alpha=0.18,
            linewidths=0,
            rasterized=True,
        )

    rb = running_bin_summary(d["_x"], d["_y"], nbins=18, min_n=30)

    if not rb.empty:
        ax.plot(rb["x"], rb["median"], color="black", lw=2.3, label="Running median")
        ax.fill_between(
            rb["x"],
            rb["q25"],
            rb["q75"],
            color="black",
            alpha=0.14,
            linewidth=0,
            label="IQR",
        )

    rho, p = spearman_stat(d["_x"], d["_y"])

    ax.text(
        0.05,
        0.95,
        f"Spearman ρ={rho:.2f}\n{format_p(p)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.88),
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel_use)
    ax.set_ylabel(ylabel_use)
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=12, loc="lower right")


def draw_supp11(speed: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    ev = speed["events"].copy()
    ev["start_state"] = state_to_int(ev["start_state"])

    area_col = (
        "max_daily_area"
        if "max_daily_area" in ev.columns
        else ("mean_daily_area" if "mean_daily_area" in ev.columns else "total_voxels")
    )

    ev["total_transition_support"] = ev["front_transitions"] + ev["local_transitions"]

    fig = plt.figure(figsize=(19.2, 5.8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.34)

    # a
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a", x=-0.12, y=1.10)
    draw_continuous_support_panel(
        ax,
        ev,
        xcol="duration_days",
        ycol="total_transition_support",
        title="Transition support vs duration",
        xlabel="Duration, days",
        logx=False,
        logy=True,
    )

    # b
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "b", x=-0.12, y=1.10)
    draw_continuous_support_panel(
        ax,
        ev,
        xcol=area_col,
        ycol="total_transition_support",
        title="Transition support vs event area",
        xlabel="event area",
        logx=True,
        logy=True,
    )

    # c
    ax = fig.add_subplot(gs[0, 2])
    add_panel_label(ax, "c", x=-0.12, y=1.10)

    for kernel_name in ["front", "local"]:
        col = f"{kernel_name}_exit_hazard"
        tr = f"{kernel_name}_transitions"

        sub = ev[ev[tr] > 0].groupby("year", as_index=False)[col].median()
        slope, p, _ = lintrend_per_decade(sub["year"], sub[col])

        ax.plot(
            sub["year"],
            rolling_mean(sub[col]),
            color=KERNEL_COLORS[kernel_name],
            lw=2.6,
            label=f"{kernel_name} ({slope:.3f}{p_to_star(p)})",
        )

    ax.set_title("Annual exit hazard")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median exit hazard")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, loc="upper right")

    savefig(fig, out_dir, "Supplementary_Fig11_transition_speed_support_spatial_editorial")



# =============================================================================
# 12. EQUAL-WIDTH SINGLE-PANEL EXPORTS FOR FIG. 4, FIG. 5 AND SUPP. FIG. 9
# =============================================================================

# Match the previous Supplementary Fig. 2 single-panel canvas.
# Single-panel figures are saved without bbox_inches="tight" so their pixel
# widths remain identical and are not altered by tick-label/colorbar cropping.
SINGLE_FIGSIZE = (7.60, 5.60)
SINGLE_ADJUST = dict(left=0.22, right=0.92, bottom=0.18, top=0.84)
SINGLE_MAP_ADJUST = dict(left=0.12, right=0.96, bottom=0.16, top=0.84, hspace=0.30)
SINGLE_HEATMAP_ADJUST = dict(left=0.20, right=0.92, bottom=0.16, top=0.84, hspace=0.34)


def save_panel_figure_equal_width(fig, out_dir: Path, name: str) -> None:
    """
    Save a single-panel figure with a fixed canvas size.

    Do not use bbox_inches="tight" here. Tight bounding boxes crop each panel
    differently depending on labels, colorbars and titles, making exported
    widths inconsistent.
    """
    ensure_dir(out_dir)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"[SAVED] {png}")
    log(f"[SAVED] {pdf}")


def make_single_ax():
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    fig.subplots_adjust(**SINGLE_ADJUST)
    return fig, ax


def make_single_heatmap_axes():
    fig = plt.figure(figsize=SINGLE_FIGSIZE)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.12], hspace=0.34)
    fig.subplots_adjust(**SINGLE_HEATMAP_ADJUST)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])
    return fig, ax, cax


def make_single_map_axes():
    fig = plt.figure(figsize=SINGLE_FIGSIZE)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.12], hspace=0.30)
    fig.subplots_adjust(**SINGLE_MAP_ADJUST)
    ax = make_geo_ax(fig, gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])
    return fig, ax, cax


def draw_figure4_single_panels(kernel: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Export Figure 4a-d as separate equal-width panels, without panel letters."""
    annual = kernel["annual_counts"].copy()
    yearly = kernel["yearly"].copy()
    coev = kernel["event_coev"].copy()

    for col in ["year", "from_state", "to_state"]:
        if col in annual.columns:
            annual[col] = pd.to_numeric(annual[col], errors="coerce").astype(int)

    front_trend, front_p = annual_probability_trend_matrix(annual, "front")
    local_trend, local_p = annual_probability_trend_matrix(annual, "local")

    common_lim = np.nanquantile(np.abs(np.r_[front_trend.ravel(), local_trend.ravel()]), 0.98)
    if not np.isfinite(common_lim) or common_lim == 0:
        common_lim = 0.01
    common_norm = mcolors.TwoSlopeNorm(vmin=-common_lim, vcenter=0, vmax=common_lim)

    # Figure 4a
    fig, ax = make_single_ax()
    sub = yearly.copy()
    if "mode" in sub.columns and sub["mode"].astype(str).str.lower().eq("standardized").any():
        sub = sub[sub["mode"].astype(str).str.lower().eq("standardized")].copy()

    for metric, label_base, ls in [
        ("p11", "P(S1→S1)", "-"),
        ("p31", "P(S3→S1)", "--"),
        ("delta", "Drying tendency", ":"),
    ]:
        if metric not in sub.columns:
            continue
        for kernel_name in ["front", "local"]:
            ss = sub[sub["kernel"] == kernel_name].sort_values("year")
            if ss.empty:
                continue
            slope, p, _ = lintrend_per_decade(ss["year"], ss[metric])
            ax.plot(
                ss["year"], rolling_mean(ss[metric]),
                color=KERNEL_COLORS[kernel_name], lw=2.3, ls=ls,
                label=f"{label_base}, {kernel_name} ({slope:.3f}{p_to_star(p)})",
            )

    ax.set_title("Continuous annual transition metrics")
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability or tendency")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=9.8, loc="upper left", handlelength=2.4)
    save_panel_figure_equal_width(fig, out_dir, "Figure4a_continuous_annual_transition_metrics_equal_width")

    # Figure 4b
    fig, ax = make_single_ax()
    if not coev.empty and {"front_delta_event", "local_delta_event"}.issubset(coev.columns):
        d = coev.copy()
        if {"front_support", "local_support"}.issubset(d.columns):
            d = d[(d["front_support"] >= 10) & (d["local_support"] >= 10)].copy()

        d["start_state_num"] = state_to_int(d["start_state"])
        d["front_minus_local_delta"] = d["front_delta_event"] - d["local_delta_event"]

        vals = [
            d.loc[d["start_state_num"] == s, "front_minus_local_delta"].dropna().values
            for s in STATE_ORDER
        ]

        bp = ax.boxplot(
            vals, positions=STATE_ORDER, widths=0.65, patch_artist=True, showfliers=False,
            medianprops=dict(color="black", lw=1.35),
            boxprops=dict(linewidth=1.05),
            whiskerprops=dict(linewidth=1.05),
            capprops=dict(linewidth=1.05),
        )

        for patch, s in zip(bp["boxes"], STATE_ORDER):
            patch.set_facecolor(STATE_COLORS[s])
            patch.set_alpha(0.72)

        ax.axhline(0, color="0.60", lw=1.0, ls="--")

        if "mean_EF" in d.columns:
            rho, p = spearman_stat(d["mean_EF"], d["front_minus_local_delta"])
            ax.text(
                0.05, 0.95,
                f"EF vs front−local\nρ={rho:.2f}, {format_p(p)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.88),
            )

    ax.set_title("Event-level front–local contrast")
    ax.set_xlabel("Initial soil-moisture state")
    ax.set_ylabel("Front − local drying tendency")
    ax.set_xticks(STATE_ORDER)
    ax.set_xticklabels(STATE_LABELS)
    ax.grid(axis="y", color="0.88", linewidth=0.9)
    clean_spines(ax)
    save_panel_figure_equal_width(fig, out_dir, "Figure4b_event_level_front_local_contrast_equal_width")

    # Figure 4c
    fig, ax, cax = make_single_heatmap_axes()
    im = plot_kernel_matrix(
        ax, front_trend, "Front kernel annual trend",
        cmap="RdBu_r", norm=common_norm, pmat=front_p, fmt="{:.3f}", text_size=9,
    )
    horizontal_cbar(fig, cax, im, "Trend in transition probability decade$^{-1}$", extend="both")
    save_panel_figure_equal_width(fig, out_dir, "Figure4c_front_kernel_annual_trend_equal_width")

    # Figure 4d
    fig, ax, cax = make_single_heatmap_axes()
    im = plot_kernel_matrix(
        ax, local_trend, "Local kernel annual trend",
        cmap="RdBu_r", norm=common_norm, pmat=local_p, fmt="{:.3f}", text_size=9,
    )
    horizontal_cbar(fig, cax, im, "Trend in transition probability decade$^{-1}$", extend="both")
    save_panel_figure_equal_width(fig, out_dir, "Figure4d_local_kernel_annual_trend_equal_width")


def draw_figure5_single_panels(speed: Dict[str, pd.DataFrame], sp_ann: pd.DataFrame, out_dir: Path) -> None:
    """Export Figure 5a-c as separate equal-width panels, without panel letters."""
    ev = speed["events"].copy()
    ev["start_state"] = state_to_int(ev["start_state"])

    # Figure 5a
    fig, ax = make_single_ax()
    for kernel_name in ["front", "local"]:
        col = f"{kernel_name}_drying_speed_mean"
        tr = f"{kernel_name}_transitions"
        sub = ev[ev[tr] > 0].groupby("year", as_index=False)[col].median()
        slope, p, _ = lintrend_per_decade(sub["year"], sub[col])
        ax.plot(
            sub["year"], rolling_mean(sub[col]),
            color=KERNEL_COLORS[kernel_name], lw=2.6,
            label=f"{kernel_name.capitalize()} ({slope:.3f}{p_to_star(p)})",
        )

    ax.set_title("Annual drying-transition speed")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median drying speed")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=15)
    save_panel_figure_equal_width(fig, out_dir, "Figure5a_annual_drying_transition_speed_equal_width")

    # Figure 5b
    fig, ax = make_single_ax()
    rows = []
    for s in STATE_ORDER:
        sub = ev[ev["start_state"] == s]
        for kernel_name in ["front", "local"]:
            col = f"{kernel_name}_drying_speed_mean"
            tr = f"{kernel_name}_transitions"
            vals = sub.loc[sub[tr] > 0, col]
            med, lo, hi = bootstrap_ci(
                vals, func=np.nanmedian,
                seed=RANDOM_SEED + s + (0 if kernel_name == "front" else 20),
            )
            rows.append({"state": s, "kernel": kernel_name, "med": med, "lo": lo, "hi": hi})

    rr = pd.DataFrame(rows)
    for kernel_name, offset, marker in [("front", -0.09, "o"), ("local", 0.09, "s")]:
        sub = rr[rr["kernel"] == kernel_name]
        ax.errorbar(
            sub["state"] + offset, sub["med"],
            yerr=[sub["med"] - sub["lo"], sub["hi"] - sub["med"]],
            fmt=marker, color=KERNEL_COLORS[kernel_name], ecolor=KERNEL_COLORS[kernel_name],
            ms=7.5, lw=2.1, capsize=3, label=kernel_name.capitalize(),
        )

    ax.set_title("Event-level drying-speed support")
    ax.set_xlabel("Initial state")
    ax.set_ylabel("Median drying speed")
    ax.set_xticks(STATE_ORDER)
    ax.set_xticklabels(STATE_LABELS)
    ax.grid(axis="y", color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, loc="upper left")
    save_panel_figure_equal_width(fig, out_dir, "Figure5b_event_level_drying_speed_support_equal_width")

    # Figure 5c
    fig, axm, cax = make_single_map_axes()
    tm = front_minus_local_speed_trend_map(sp_ann)
    if not tm.empty:
        lim = np.nanquantile(np.abs(tm["front_minus_local_trend"]), 0.985)
        if not np.isfinite(lim) or lim == 0:
            lim = 0.05
        im = plot_grid_map(
            axm, tm, "front_minus_local_trend",
            cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
            point_size=6.0,
        )
        horizontal_cbar(
            fig, cax, im,
            "Trend in front−local drying speed decade$^{-1}$",
            extend="both",
        )
    else:
        axm.text(0.5, 0.5, "No valid spatial trend data", transform=axm.transAxes,
                 ha="center", va="center")

    axm.set_title("Annual trend in front-minus-local drying speed")
    save_panel_figure_equal_width(fig, out_dir, "Figure5c_front_minus_local_speed_trend_equal_width")


def draw_supp9_single_panels(result2: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Export Supplementary Fig. 9a-c as separate equal-width panels, without panel letters."""
    ev = result2["events"].copy()
    seg = result2["segments"].copy()

    if "state_index" not in ev.columns:
        ev["state_index"] = state_to_int(ev["start_state"])

    if "state_index" not in seg.columns and "start_state" in seg.columns:
        seg["state_index"] = state_to_int(seg["start_state"])

    # Supplementary Fig. 9a
    fig, ax = make_single_ax()
    for s in STATE_ORDER:
        sub = ev[ev["state_index"] == s].groupby("year", as_index=False)["moving_flag"].mean()
        if sub.empty:
            continue
        slope, p, _ = lintrend_per_decade(sub["year"], sub["moving_flag"])
        ax.plot(
            sub["year"], rolling_mean(sub["moving_flag"]),
            color=STATE_COLORS[s], lw=2.1,
            label=f"S{s} ({slope:.3f}{p_to_star(p)})",
        )

    ax.set_title("Moving-event fraction")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction moving")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    ax.legend(frameon=False, fontsize=11.8, ncol=2, loc="upper left")
    save_panel_figure_equal_width(fig, out_dir, "Supplementary_Fig9a_moving_event_fraction_equal_width")

    # Supplementary Fig. 9b
    fig, axm, cax = make_single_map_axes()
    if not seg.empty and {"mid_lon", "mid_lat"}.issubset(seg.columns):
        sub = seg[seg["state_index"].isin([1, 2])].copy()
        if "year" in sub.columns:
            sub = sub[sub["year"] >= 2000].copy()
        if len(sub) > 90000:
            sub = sub.sample(90000, random_state=RANDOM_SEED)

        kwargs = dict(gridsize=55, extent=CONUS_EXTENT, mincnt=1, cmap="YlOrRd", bins="log")
        if HAS_CARTOPY:
            kwargs["transform"] = ccrs.PlateCarree()
        hb = axm.hexbin(sub["mid_lon"], sub["mid_lat"], **kwargs)
        horizontal_cbar(fig, cax, hb, "Dry-start segment density")
    else:
        axm.text(0.5, 0.5, "No segment midpoint data", transform=axm.transAxes,
                 ha="center", va="center")

    axm.set_title("Dry-start mobility corridors")
    save_panel_figure_equal_width(fig, out_dir, "Supplementary_Fig9b_dry_start_mobility_corridors_equal_width")

    # Supplementary Fig. 9c
    fig, ax = make_single_ax()
    te = ev[["path_length_km", "net_displacement_km", "state_index"]].dropna().copy()
    te = te[(te["path_length_km"] > 0) & (te["net_displacement_km"] >= 0)]
    plot_d = te.sample(7000, random_state=RANDOM_SEED) if len(te) > 7000 else te

    for s in STATE_ORDER:
        ss = plot_d[plot_d["state_index"] == s]
        if not ss.empty:
            ax.scatter(
                ss["path_length_km"], ss["net_displacement_km"],
                s=12, color=STATE_COLORS[s], alpha=0.22, linewidths=0, rasterized=True,
            )

    max_lim = np.nanpercentile(te[["path_length_km", "net_displacement_km"]].values, 99.2)
    max_lim = max(50, max_lim)
    ax.plot([0, max_lim], [0, max_lim], ls="--", color="0.35", lw=1.3)

    rho, p = spearman_stat(te["path_length_km"], te["net_displacement_km"])
    ax.text(
        0.04, 0.94,
        f"Spearman ρ={rho:.2f}\n{format_p(p)}",
        transform=ax.transAxes, ha="left", va="top", fontsize=15,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.88),
    )

    ax.set_xlim(0, max_lim)
    ax.set_ylim(0, max_lim)
    ax.set_title("Trajectory efficiency")
    ax.set_xlabel("Path length (km)")
    ax.set_ylabel("Net displacement (km)")
    ax.grid(color="0.88", linewidth=0.9)
    clean_spines(ax)
    save_panel_figure_equal_width(fig, out_dir, "Supplementary_Fig9c_trajectory_efficiency_equal_width")


def draw_requested_equal_width_single_panels(
    result2: Dict[str, pd.DataFrame],
    speed: Dict[str, pd.DataFrame],
    kernel: Dict[str, pd.DataFrame],
    sp_ann: pd.DataFrame,
    out_dir: Path,
) -> None:
    log("[DRAW] Equal-width single panels for Figure 4")
    draw_figure4_single_panels(kernel, out_dir)

    log("[DRAW] Equal-width single panels for Figure 5")
    draw_figure5_single_panels(speed, sp_ann, out_dir)

    log("[DRAW] Equal-width single panels for Supplementary Fig. 9")
    draw_supp9_single_panels(result2, out_dir)


# =============================================================================
# 13. MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Root directory containing cached Result 2/3 outputs."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Default: <root>/_NCC_Result2_Result3_editorial_compact_v5_nature_style"
    )

    parser.add_argument("--result2-dir", default=None)
    parser.add_argument("--speed-dir", default=None)
    parser.add_argument("--kernel-dir", default=None)
    parser.add_argument("--v3-dir", default=None)

    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only export separate equal-width panels; do not regenerate combined Figure 4, Figure 5 and Supplementary Fig. 9."
    )

    parser.add_argument(
        "--with-supp10-supp11",
        action="store_true",
        help="Also regenerate Supplementary Fig. 10 and Supplementary Fig. 11 from the original script."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    dirs = default_dirs(root)

    result2_dir = Path(args.result2_dir) if args.result2_dir else dirs["result2"]
    speed_dir = Path(args.speed_dir) if args.speed_dir else dirs["speed"]
    kernel_dir = Path(args.kernel_dir) if args.kernel_dir else dirs["kernel"]
    v3_dir = Path(args.v3_dir) if args.v3_dir else dirs["v3"]

    out_dir = ensure_dir(
        Path(args.output)
        if args.output
        else root / "_NCC_Result2_Result3_editorial_compact_v5_nature_style"
    )

    log("=" * 100)
    log("[INFO] Drawing requested equal-width single panels for Figure 4, Figure 5 and Supplementary Fig. 9")
    log(f"[INFO] root       : {root}")
    log(f"[INFO] result2 dir: {result2_dir}")
    log(f"[INFO] speed dir  : {speed_dir}")
    log(f"[INFO] kernel dir : {kernel_dir}")
    log(f"[INFO] v3 cache dir: {v3_dir}")
    log(f"[INFO] output dir : {out_dir}")
    log(f"[INFO] single size: {SINGLE_FIGSIZE}; fixed-width save without bbox_inches='tight'")
    log("=" * 100)

    result2 = load_result2(result2_dir)
    speed = load_speed(speed_dir)
    kernel = load_kernel(kernel_dir)
    sp_ann, trans_ann = load_annual_spatial_caches(v3_dir)

    if not args.single_only:
        log("[DRAW] Combined Figure 4")
        draw_figure4(kernel, out_dir)

        log("[DRAW] Combined Figure 5")
        draw_figure5(speed, sp_ann, out_dir)

        log("[DRAW] Combined Supplementary Fig. 9")
        draw_supp9(result2, out_dir)

    draw_requested_equal_width_single_panels(
        result2=result2,
        speed=speed,
        kernel=kernel,
        sp_ann=sp_ann,
        out_dir=out_dir,
    )

    if args.with_supp10_supp11:
        log("[DRAW] Supplementary Fig. 10")
        draw_supp10(kernel, trans_ann, out_dir)

        log("[DRAW] Supplementary Fig. 11")
        draw_supp11(speed, out_dir)

    log("=" * 100)
    log("[DONE] Requested figures generated.")
    log(f"[DONE] Output directory: {out_dir}")
    log("=" * 100)


if __name__ == "__main__":
    main()
