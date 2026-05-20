# -*- coding: utf-8 -*-
"""
NCC-grade (NO figure sea):
MAIN Fig: migration metrics (bars) + direction rose (2 eras) under ROLL vs FIXED
ED Fig  : threshold identifiability diagnostics (event count, LCF, mean active cells/day)

Your folders:
  FIXED_ROOT:
    ...\CC3D_STATES6_UNIFIED_outputs固定的\sspXXX\<MODEL>\events_cc3d\YYYY\event_YYYY_*.csv
  ROLL_ROOT:
    ...\CC3D_STATES6_UNIFIED_outputs_ROLL30每个模型的\sspXXX\<MODEL>\events_cc3d\YYYY\event_YYYY_*.csv

Outputs (ONLY 2 figures):
  OUT_DIR/
    Fig_MAIN_CMIP6_migration_bar_rose_fixed_vs_roll.png/.pdf
    Fig_ED_CMIP6_threshold_identifiability_fixed_vs_roll.png/.pdf
    tables_ensemble_era_summary.csv
    tables_ensemble_year_summary.csv
    cache_events_both.pkl
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =========================
# USER SETTINGS
# =========================
FIXED_ROOT = Path(r"E:\第二篇的修改20251229开始修改\CC3D_STATES6_UNIFIED_outputs固定的")
ROLL_ROOT  = Path(r"E:\第二篇的修改20251229开始修改\CC3D_STATES6_UNIFIED_outputs_ROLL30每个模型的")
EVENTS_SUBDIR = "events_cc3d"
SCENARIOS = ["ssp126", "ssp245", "ssp585"]

YEAR_MIN, YEAR_MAX = 1950, 2100

# event validity
MIN_DAYS_PER_EVENT = 3

# for direction: ignore "almost-stationary" events when assigning bearing
MIN_NET_DISP_KM_FOR_DIR = 50.0

# fixed invalid if percolated
LCF_INVALID_TH = 0.80

# rose setting
ROSE_USE_IQR_FOR_ROLLING = True  # IQR envelope only for rolling (avoid line clutter)
ROSE_ERAS_MODE = "two"  # "two" (1950-74 vs 2075-2100) or "all" (NOT recommended)

# output
OUT_DIR = Path(r"E:\_figs_NCC_CMIP6_migration_bar_rose")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_EVENTS = OUT_DIR / "cache_events_both.pkl"

FONT_FAMILY = "Arial"
FONTSIZE = 13
DPI = 300


# =========================
# ERAS
# =========================
ERAS = [
    (1950, 1974, "1950–74"),
    (1975, 1999, "1975–99"),
    (2000, 2024, "2000–24"),
    (2025, 2049, "2025–49"),
    (2050, 2074, "2050–74"),
    (2075, 2100, "2075–2100"),
]

def era_index(y: int) -> int:
    for i, (a,b,_) in enumerate(ERAS):
        if a <= y <= b:
            return i
    return -1


# =========================
# CSV helpers
# =========================
def detect_sep(header_line: str) -> str:
    c = header_line.count(",")
    t = header_line.count("\t")
    s = header_line.count(";")
    if t >= c and t >= s and t > 0:
        return "\t"
    if s >= c and s >= t and s > 0:
        return ";"
    return ","

def list_models(root: Path, scenario: str) -> List[str]:
    p = root / scenario
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()])

def list_years(events_dir: Path) -> List[int]:
    if not events_dir.exists():
        return []
    yrs = []
    for x in events_dir.iterdir():
        if x.is_dir() and re.fullmatch(r"\d{4}", x.name):
            y = int(x.name)
            if YEAR_MIN <= y <= YEAR_MAX:
                yrs.append(y)
    return sorted(yrs)

def parse_event_id_from_name(fp: Path) -> int:
    m = re.search(r"event_(\d{4})_(\d+)\.csv", fp.name)
    return int(m.group(2)) if m else -1


# =========================
# Geometry
# =========================
EARTH_R_KM = 6371.0

def haversine_km(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return float(EARTH_R_KM * c)

def bearing_deg(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    b = np.arctan2(x, y)
    return float((np.rad2deg(b) + 360.0) % 360.0)


# =========================
# Read minimal columns + compute event metrics fast
# =========================
def read_event_mincols(fp: Path) -> pd.DataFrame:
    with open(fp, "r", encoding="utf-8-sig", errors="ignore") as f:
        header = f.readline()
    sep = detect_sep(header)

    usecols = {"year","doy","date","lon","lat","longitude","latitude","heat3","heat_raw","event_id"}
    df = pd.read_csv(fp, sep=sep, encoding="utf-8-sig", low_memory=False,
                     usecols=lambda c: c.strip() in usecols)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    if "lon" not in df.columns and "longitude" in df.columns:
        df["lon"] = df["longitude"]
    if "lat" not in df.columns and "latitude" in df.columns:
        df["lat"] = df["latitude"]

    for c in ["year","doy","lon","lat","heat3","heat_raw","event_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "doy" not in df.columns or df["doy"].isna().all():
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            df["doy"] = dt.dt.dayofyear.astype(float)

    if "heat3" in df.columns and df["heat3"].notna().any():
        df = df[df["heat3"] == 1]
    elif "heat_raw" in df.columns and df["heat_raw"].notna().any():
        df = df[df["heat_raw"] == 1]

    df = df[np.isfinite(df["lon"]) & np.isfinite(df["lat"]) & np.isfinite(df["doy"])]
    return df

def compute_event_metrics(fp: Path, year_hint: int) -> Dict:
    df = read_event_mincols(fp)
    if df.empty:
        return {}

    y = int(df["year"].dropna().iloc[0]) if "year" in df.columns and df["year"].notna().any() else int(year_hint)

    if "event_id" in df.columns and df["event_id"].notna().any():
        eid0 = df["event_id"].iloc[0]
        eid = int(eid0) if np.isfinite(eid0) else parse_event_id_from_name(fp)
    else:
        eid = parse_event_id_from_name(fp)

    doy = df["doy"].astype(int).values
    days = np.unique(doy)
    if days.size < MIN_DAYS_PER_EVENT:
        return {}

    latv = df["lat"].values.astype(float)
    lonv = df["lon"].values.astype(float)
    w = np.cos(np.deg2rad(latv))

    tmp = pd.DataFrame({"doy": doy, "w": w, "lw": lonv*w, "aw": latv*w})
    g = tmp.groupby("doy", sort=True)[["w","lw","aw"]].sum()

    sw = g["w"].values
    lon_c = g["lw"].values / sw
    lat_c = g["aw"].values / sw
    d_sorted = g.index.values.astype(int)

    if d_sorted.size < MIN_DAYS_PER_EVENT:
        return {}

    lon0, lat0 = float(lon_c[0]), float(lat_c[0])
    lon1, lat1 = float(lon_c[-1]), float(lat_c[-1])

    net_km = haversine_km(lon0, lat0, lon1, lat1)

    path_km = 0.0
    for i in range(1, d_sorted.size):
        path_km += haversine_km(lon_c[i-1], lat_c[i-1], lon_c[i], lat_c[i])

    if net_km >= MIN_NET_DISP_KM_FOR_DIR:
        b = bearing_deg(lon0, lat0, lon1, lat1)
        sector = int(((b + 22.5) % 360.0) // 45.0)
    else:
        b = np.nan
        sector = -1

    dlon = float(lon1 - lon0)
    dlat = float(lat1 - lat0)

    voxels = int(len(df))
    duration = int(d_sorted.size)
    mean_active_cells = float(voxels / max(duration, 1))

    return dict(
        year=y, era=era_index(y), event_id=int(eid),
        duration=duration, voxels=voxels, mean_active_cells=mean_active_cells,
        lon0=lon0, lat0=lat0, lon1=lon1, lat1=lat1,
        dlon_deg=dlon, dlat_deg=dlat,
        net_km=float(net_km), path_km=float(path_km),
        bearing_deg=float(b) if np.isfinite(b) else np.nan,
        sector=int(sector)
    )


# =========================
# Build cache (both datasets)
# =========================
def build_cache_both(force: bool) -> pd.DataFrame:
    if (not force) and CACHE_EVENTS.exists():
        print(f"[cache] load {CACHE_EVENTS}")
        return pd.read_pickle(CACHE_EVENTS)

    rows = []
    for dataset, root in [("roll", ROLL_ROOT), ("fixed", FIXED_ROOT)]:
        for ssp in SCENARIOS:
            models = list_models(root, ssp)
            if not models:
                print(f"[WARN] no models in {root}/{ssp}")
                continue

            for model in models:
                edir = root / ssp / model / EVENTS_SUBDIR
                yrs = list_years(edir)
                if not yrs:
                    print(f"[WARN] no years in {edir}")
                    continue

                for y in yrs:
                    ydir = edir / str(y)
                    files = sorted(ydir.glob("event_*.csv"))
                    if not files:
                        continue

                    total_vox = 0
                    max_vox = 0
                    year_events = []

                    for fp in files:
                        met = compute_event_metrics(fp, year_hint=y)
                        if not met:
                            continue
                        total_vox += met["voxels"]
                        max_vox = max(max_vox, met["voxels"])
                        met.update(dict(dataset=dataset, scenario=ssp, model=model))
                        year_events.append(met)

                    if year_events:
                        lcf = (max_vox / total_vox) if total_vox > 0 else np.nan
                        for r in year_events:
                            r["lcf_year"] = float(lcf) if np.isfinite(lcf) else np.nan
                        rows.extend(year_events)

                print(f"[{dataset}] scanned {ssp}/{model}: years={len(yrs)}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No events parsed. Check paths/columns (lon/lat/doy/heat3).")

    df.to_pickle(CACHE_EVENTS)
    print(f"[cache] saved {CACHE_EVENTS} rows={len(df)}")
    return df


# =========================
# Stats helpers
# =========================
def p25(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.nanpercentile(x, 25)) if x.size else np.nan
def p50(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.nanpercentile(x, 50)) if x.size else np.nan
def p75(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.nanpercentile(x, 75)) if x.size else np.nan


# =========================
# QC
# =========================
def qc_print(df: pd.DataFrame):
    for ds in ["roll","fixed"]:
        for ssp in SCENARIOS:
            sub = df[(df["dataset"]==ds) & (df["scenario"]==ssp)]
            if sub.empty:
                continue
            nk = sub["net_km"].values.astype(float)
            pk = sub["path_km"].values.astype(float)
            dlon = sub["dlon_deg"].values.astype(float)
            dlat = sub["dlat_deg"].values.astype(float)

            print(f"[QC] {ds} {ssp} | net_km p50={p50(nk):.3f}  p90={np.nanpercentile(nk[np.isfinite(nk)],90):.3f} | "
                  f"path_km p50={p50(pk):.3f}")
            print(f"     Δlon(deg) p50={p50(np.abs(dlon)):.4f} | Δlat(deg) p50={p50(np.abs(dlat)):.4f}")

            if p50(nk) < 1.0 and (p50(np.abs(dlon)) > 0.05 or p50(np.abs(dlat)) > 0.05):
                print("  [WARN] net_km median < 1 km but Δlon/Δlat not tiny. Unit/parse mismatch likely.")
            if p50(nk) < 0.1:
                print("  [WARN] net displacement extremely small. If true, 'migration' via centroid displacement may be weak.")


# =========================
# Ensemble summaries
# =========================
def summarize_era(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    rows = []
    for (dataset, ssp, model, era), g in df.groupby(["dataset","scenario","model","era"]):
        if era < 0:
            continue
        rows.append(dict(
            dataset=dataset, scenario=ssp, model=model, era=int(era),
            n_events=int(len(g)),
            net_km_med=p50(g["net_km"].values),
            path_km_med=p50(g["path_km"].values),
            lcf_med=p50(g["lcf_year"].values),
        ))
    dm = pd.DataFrame(rows)

    out = []
    for (dataset, ssp, era), g in dm.groupby(["dataset","scenario","era"]):
        out.append(dict(
            dataset=dataset, scenario=ssp, era=int(era),
            n_events_p50=p50(g["n_events"].values), n_events_p25=p25(g["n_events"].values), n_events_p75=p75(g["n_events"].values),
            net_km_p50=p50(g["net_km_med"].values), net_km_p25=p25(g["net_km_med"].values), net_km_p75=p75(g["net_km_med"].values),
            path_km_p50=p50(g["path_km_med"].values), path_km_p25=p25(g["path_km_med"].values), path_km_p75=p75(g["path_km_med"].values),
            lcf_p50=p50(g["lcf_med"].values), lcf_p25=p25(g["lcf_med"].values), lcf_p75=p75(g["lcf_med"].values),
        ))
    ens_era = pd.DataFrame(out).sort_values(["dataset","scenario","era"])

    if ROSE_ERAS_MODE == "two":
        era_keys = [0, 5]
    else:
        era_keys = list(range(len(ERAS)))

    rose = {}
    for dataset in ["roll","fixed"]:
        for ssp in SCENARIOS:
            for era in era_keys:
                sub = df[(df["dataset"]==dataset) & (df["scenario"]==ssp) & (df["era"]==era)]
                if sub.empty:
                    continue

                frac_by_model = []
                for model, gm in sub.groupby("model"):
                    sec = gm["sector"].values.astype(int)
                    sec = sec[sec >= 0]
                    if sec.size < 10:
                        continue
                    cnt = np.bincount(sec, minlength=8).astype(float)
                    frac = cnt / np.sum(cnt)
                    frac_by_model.append(frac)

                if not frac_by_model:
                    continue
                M = np.vstack(frac_by_model)
                rose[(dataset, ssp, era)] = dict(
                    p50=np.nanpercentile(M, 50, axis=0),
                    p25=np.nanpercentile(M, 25, axis=0),
                    p75=np.nanpercentile(M, 75, axis=0),
                    n_models=int(M.shape[0])
                )

    return ens_era, rose


def summarize_year(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, ssp, model, year), g in df.groupby(["dataset","scenario","model","year"]):
        rows.append(dict(
            dataset=dataset, scenario=ssp, model=model, year=int(year),
            event_count=int(len(g)),
            lcf=p50(g["lcf_year"].values),
            mean_active_cells=p50(g["mean_active_cells"].values),
        ))
    my = pd.DataFrame(rows)

    out = []
    for (dataset, ssp, year), g in my.groupby(["dataset","scenario","year"]):
        out.append(dict(
            dataset=dataset, scenario=ssp, year=int(year),
            event_count_p50=p50(g["event_count"].values), event_count_p25=p25(g["event_count"].values), event_count_p75=p75(g["event_count"].values),
            lcf_p50=p50(g["lcf"].values), lcf_p25=p25(g["lcf"].values), lcf_p75=p75(g["lcf"].values),
            mac_p50=p50(g["mean_active_cells"].values), mac_p25=p25(g["mean_active_cells"].values), mac_p75=p75(g["mean_active_cells"].values),
        ))
    return pd.DataFrame(out).sort_values(["dataset","scenario","year"])


# =========================
# Plot helpers (NO panel letters now)
# =========================
def plot_bars(ax, x, y50, y25, y75, label, face, edge, hatch=None, alpha=1.0):
    w = 0.34
    ax.bar(x, y50, width=w, color=face, edgecolor=edge, linewidth=1.2,
           hatch=hatch if hatch else None, alpha=alpha, label=label, zorder=3)
    yerr_low = np.maximum(0, y50 - y25)
    yerr_up  = np.maximum(0, y75 - y50)
    ax.errorbar(x, y50, yerr=[yerr_low, yerr_up], fmt="none",
                ecolor=edge, elinewidth=1.2, capsize=3, zorder=4)

def plot_rose(ax, rose_dict, dataset, ssp, era, style, show_iqr: bool):
    key = (dataset, ssp, era)
    if key not in rose_dict:
        return

    d = rose_dict[key]
    p50v = d["p50"]
    p25v = d["p25"]
    p75v = d["p75"]

    theta = np.deg2rad(np.arange(0, 360, 45))
    th = np.r_[theta, theta[0]]
    r50 = np.r_[p50v, p50v[0]]
    r25 = np.r_[p25v, p25v[0]]
    r75 = np.r_[p75v, p75v[0]]

    if show_iqr:
        ax.plot(th, r75, color=style["color"], lw=0.9, ls=style["ls"], alpha=0.35)
        ax.plot(th, r25, color=style["color"], lw=0.9, ls=style["ls"], alpha=0.35)

    ax.plot(th, r50, color=style["color"], lw=style["lw"], ls=style["ls"], label=style["label"])


# =========================
# Plot: MAIN (bars + rose)
# =========================
def make_main_fig(ens_era: pd.DataFrame, rose: Dict):
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.size"] = FONTSIZE

    fig = plt.figure(figsize=(16.8, 9.6), dpi=DPI)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.05], wspace=0.24, hspace=0.34)

    x = np.arange(len(ERAS))
    invalid = {ssp: np.zeros(len(ERAS), dtype=bool) for ssp in SCENARIOS}
    for ssp in SCENARIOS:
        subF = ens_era[(ens_era["dataset"]=="fixed") & (ens_era["scenario"]==ssp)].sort_values("era")
        if not subF.empty:
            lcf = subF["lcf_p50"].values
            for i in range(min(len(lcf), len(ERAS))):
                if np.isfinite(lcf[i]) and lcf[i] >= LCF_INVALID_TH:
                    invalid[ssp][i] = True

    for ci, ssp in enumerate(SCENARIOS):
        ax1 = plt.subplot(gs[0, ci]); ax1.set_title(ssp)
        ax2 = plt.subplot(gs[1, ci])
        ax3 = plt.subplot(gs[2, ci], projection="polar")

        for i, inv in enumerate(invalid[ssp]):
            if inv:
                ax1.axvspan(i-0.5, i+0.5, color="0.93", zorder=0)
                ax2.axvspan(i-0.5, i+0.5, color="0.93", zorder=0)

        def grab(ds: str):
            sub = ens_era[(ens_era["dataset"]==ds) & (ens_era["scenario"]==ssp)].sort_values("era")
            y_net50 = np.full(len(ERAS), np.nan)
            y_net25 = np.full(len(ERAS), np.nan)
            y_net75 = np.full(len(ERAS), np.nan)
            y_path50= np.full(len(ERAS), np.nan)
            y_path25= np.full(len(ERAS), np.nan)
            y_path75= np.full(len(ERAS), np.nan)
            for r in sub.itertuples(index=False):
                i = int(r.era)
                y_net50[i]=float(r.net_km_p50); y_net25[i]=float(r.net_km_p25); y_net75[i]=float(r.net_km_p75)
                y_path50[i]=float(r.path_km_p50); y_path25[i]=float(r.path_km_p25); y_path75[i]=float(r.path_km_p75)
            return y_net50,y_net25,y_net75,y_path50,y_path25,y_path75

        r_net50,r_net25,r_net75,r_path50,r_path25,r_path75 = grab("roll")
        f_net50,f_net25,f_net75,f_path50,f_path25,f_path75 = grab("fixed")

        m = invalid[ssp]
        f_net50[m]=np.nan; f_net25[m]=np.nan; f_net75[m]=np.nan
        f_path50[m]=np.nan; f_path25[m]=np.nan; f_path75[m]=np.nan

        off = 0.18
        x_roll = x - off
        x_fix  = x + off

        plot_bars(ax1, x_roll, r_net50, r_net25, r_net75, "rolling", face="white", edge="k", hatch=None, alpha=1.0)
        plot_bars(ax1, x_fix,  f_net50, f_net25, f_net75, "fixed",   face="0.85", edge="0.35", hatch="///", alpha=1.0)

        ax1.set_xticks(x); ax1.set_xticklabels([])
        ax1.set_ylabel("Net displacement  km" if ci==0 else "")
        if ci!=0: ax1.set_yticklabels([])
        ax1.axhline(0, color="0.75", lw=0.8)
        for sp in ax1.spines.values(): sp.set_linewidth(0.9)
        if ci==0:
            ax1.legend(frameon=False, loc="upper left", fontsize=FONTSIZE-1)

        plot_bars(ax2, x_roll, r_path50, r_path25, r_path75, "rolling", face="white", edge="k", hatch=None, alpha=1.0)
        plot_bars(ax2, x_fix,  f_path50, f_path25, f_path75, "fixed",   face="0.85", edge="0.35", hatch="///", alpha=1.0)

        ax2.set_xticks(x); ax2.set_xticklabels([])
        ax2.set_ylabel("Path length  km" if ci==0 else "")
        if ci!=0: ax2.set_yticklabels([])
        ax2.axhline(0, color="0.75", lw=0.8)
        for sp in ax2.spines.values(): sp.set_linewidth(0.9)

        if ROSE_ERAS_MODE == "two":
            eras_to_plot = [0, 5]
        else:
            eras_to_plot = list(range(len(ERAS)))

        ax3.set_theta_zero_location("N")
        ax3.set_theta_direction(-1)
        ax3.set_thetagrids(np.arange(0, 360, 45), labels=["N","NE","E","SE","S","SW","W","NW"])
        ax3.set_rlabel_position(90)
        ax3.grid(True, lw=0.8, color="0.75")

        rmax = 0.0
        for era in eras_to_plot:
            key = ("roll", ssp, era)
            if key in rose:
                rmax = max(rmax, float(np.nanmax(rose[key]["p75"])))
        ax3.set_ylim(0, max(0.25, min(0.5, rmax*1.15 if rmax>0 else 0.35)))

        for era in eras_to_plot:
            is_early = (era == 0)
            lab_era = ERAS[era][2]

            st_roll = dict(color="k", ls="--" if is_early else "-", lw=2.0,
                           label=f"rolling {lab_era}")
            plot_rose(ax3, rose, "roll", ssp, era, st_roll, show_iqr=ROSE_USE_IQR_FOR_ROLLING)

            if not invalid[ssp][era]:
                st_fix = dict(color="0.35", ls="--" if is_early else "-", lw=1.8,
                              label=f"fixed {lab_era}")
                plot_rose(ax3, rose, "fixed", ssp, era, st_fix, show_iqr=False)

        if ci==0:
            ax3.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.02, -0.05),
                       fontsize=FONTSIZE-2)

    fig.text(
        0.5, 0.01,
        "Bars: model-median migration metrics per era; error bars = IQR across models. "
        "Fixed-threshold eras are grey-shaded/omitted when ensemble median LCF ≥ 0.8 (percolation). "
        "Rose: bearing from start→end centroid; dashed=1950–74, solid=2075–2100.",
        ha="center", va="bottom", fontsize=FONTSIZE-1
    )

    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.09)
    out_png = OUT_DIR / "Fig_MAIN_CMIP6_migration_bar_rose_fixed_vs_roll.png"
    out_pdf = OUT_DIR / "Fig_MAIN_CMIP6_migration_bar_rose_fixed_vs_roll.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved MAIN:\n  {out_png}\n  {out_pdf}")


# =========================
# Plot: ED (identifiability)  —— removed panel letters + removed bottom caption
# =========================
def make_ed_fig(ens_year: pd.DataFrame):
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.size"] = FONTSIZE

    fig = plt.figure(figsize=(16.8, 9.6), dpi=DPI)
    gs = GridSpec(3, 3, figure=fig, wspace=0.25, hspace=0.25)

    for ci, ssp in enumerate(SCENARIOS):
        ax1 = plt.subplot(gs[0,ci]); ax1.set_title(ssp)
        ax2 = plt.subplot(gs[1,ci])
        ax3 = plt.subplot(gs[2,ci])

        def plot_series(ax, ds, ycol, ylab, ls, col, alpha_band):
            sub = ens_year[(ens_year["dataset"]==ds) & (ens_year["scenario"]==ssp)]
            if sub.empty:
                return
            x = sub["year"].values.astype(int)
            y50 = sub[f"{ycol}_p50"].values.astype(float)
            y25 = sub[f"{ycol}_p25"].values.astype(float)
            y75 = sub[f"{ycol}_p75"].values.astype(float)
            ax.fill_between(x, y25, y75, color=col, alpha=alpha_band, linewidth=0)
            ax.plot(x, y50, color=col, lw=2.0 if ds=="roll" else 1.8, ls=ls)

            ax.set_xlim(YEAR_MIN, YEAR_MAX)
            ax.set_ylabel(ylab if ci==0 else "")
            if ci!=0:
                ax.set_yticklabels([])
            for sp in ax.spines.values(): sp.set_linewidth(0.9)

        plot_series(ax1, "roll",  "event_count", "Event count per summer", "-",  "k",    0.12)
        plot_series(ax1, "fixed", "event_count", "Event count per summer", "--", "0.35", 0.10)

        plot_series(ax2, "roll",  "lcf", "Largest-component fraction", "-",  "k",    0.12)
        plot_series(ax2, "fixed", "lcf", "Largest-component fraction", "--", "0.35", 0.10)
        ax2.set_ylim(0, 1.0)

        plot_series(ax3, "roll",  "mac", "Mean active cells per day", "-",  "k",    0.12)
        plot_series(ax3, "fixed", "mac", "Mean active cells per day", "--", "0.35", 0.10)
        ax3.set_xlabel("Year")

        if ci==0:
            ax1.plot([],[], color="k", lw=2.0, ls="-", label="rolling (median ± IQR)")
            ax1.plot([],[], color="0.35", lw=1.8, ls="--", label="fixed (median ± IQR)")
            ax1.legend(frameon=False, loc="upper left", fontsize=FONTSIZE-1)

    # NOTE: removed the ED caption/fig.text entirely per your request.

    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.09)
    out_png = OUT_DIR / "Fig_ED_CMIP6_threshold_identifiability_fixed_vs_roll.png"
    out_pdf = OUT_DIR / "Fig_ED_CMIP6_threshold_identifiability_fixed_vs_roll.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ED:\n  {out_png}\n  {out_pdf}")


# =========================
# MAIN
# =========================
def main(rebuild: bool):
    df = build_cache_both(force=rebuild)
    qc_print(df)

    ens_era, rose = summarize_era(df)
    ens_year = summarize_year(df)

    ens_era.to_csv(OUT_DIR / "tables_ensemble_era_summary.csv", index=False)
    ens_year.to_csv(OUT_DIR / "tables_ensemble_year_summary.csv", index=False)

    make_main_fig(ens_era, rose)
    make_ed_fig(ens_year)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Rescan all CSVs and rebuild caches")
    args = ap.parse_args()
    main(rebuild=args.rebuild)