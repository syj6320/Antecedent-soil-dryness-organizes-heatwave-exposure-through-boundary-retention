# -*- coding: utf-8 -*-
"""
ERA5 (bucket-based) CC3D event extraction (streaming writer, low RAM peak) — FINAL FULL RUN
==========================================================================================
关键修复（保证与非-streaming版本一致）：
- Pass1 / Pass2 / Pass3 全部严格过滤 year == target_year
  否则如果 bucket/1950.csv 混入其它年份，会把不同年份相同 doy 叠在同一个 t，
  导致事件被错误合并（事件数显著变少，例如 252 -> 92）。

流程：3-pass per year
  Pass1: 扫描 doy 范围（仅 event rows & year==y）
  Pass2: 构建 3D mask A (uint8) -> cc3d labels（仅 event rows & year==y）
  Pass2b: 基于 labels 推导每个连通体的统计量（duration, extent, bbox, ...）
  Pass3: 重新分块读 bucket（仅 event rows & year==y），从 labels 取 cc_label，
         映射 event_id，并按 event_id 追加写入 event_YYYY_XXXXX.csv（无全年 df）

INPUT:
  OUT_ROOT/
    grid_index.csv
    buckets/
      1950.csv ... 2024.csv

OUTPUT:
  OUT_ROOT/events_cc3d/YYYY/event_YYYY_00001.csv ...
  OUT_ROOT/events_cc3d/YYYY/events_YYYY_summary.csv
  OUT_ROOT/events_cc3d/annual_metrics.csv
  OUT_ROOT/events_cc3d/failures_cc3d_from_buckets_streaming.txt
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import traceback
import shutil
import cc3d

# =========================
# USER SETTINGS
# =========================
OUT_ROOT = Path(r"E:\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本")
BUCKET_DIR = OUT_ROOT / "buckets"
GRID_INDEX_FP = OUT_ROOT / "grid_index.csv"
EVENTS_ROOT = OUT_ROOT / "events_cc3d"

START_YEAR = 1950
END_YEAR   = 2024

# FULL RUN
YEARS_TO_RUN = range(START_YEAR, END_YEAR + 1)

EVENT_MASK_MODE = "heat3"  # "heat3" / "DHW_sync3" / "DHW_predry3"
CONNECTIVITY = 26
MIN_DURATION_DAYS = 3

ROUND_COORD = 4
CHUNK_SIZE = 1_000_000

# 输出控制
WRITE_EVENT_CSVS = True
OVERWRITE_YEAR_DIR = True
KEEP_ALL_COLUMNS = True  # 建议正式跑：如果硬盘压力大，改 False

MIN_KEEP_COLS = [
    "date","year","month","doy",
    "grid_id","lon","lat","longitude","latitude",
    "temp_air","soil_moist",
    "T90","SM10","SM20","SM30","SM40","SM50","SM60","SM70","SM80","SM90",
    "heat_raw","dry_raw","wet90_raw","dry_lag1",
    "heat3","DHW_sync3","NDHW3","DHW_predry3","WHW90_3",
    "SM_decile","S_bin"
]

VERBOSE = True


# =========================
# Utilities
# =========================
def log(msg: str):
    if VERBOSE:
        print(msg)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def chunk_iter_csv(fp: Path, chunksize: int):
    return pd.read_csv(fp, chunksize=chunksize)

def mask_event_rows(chunk: pd.DataFrame, mask_col: str) -> pd.Series:
    return pd.to_numeric(chunk[mask_col], errors="coerce").fillna(0).astype(np.int8).eq(1)

def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), errors="coerce")

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has 'date' as YYYY-MM-DD string.
    If missing, create from year+doy.
    """
    if "date" in df.columns:
        dt = parse_date_series(df["date"])
        ok = dt.notna()
        df = df.loc[ok].copy()
        df["date"] = dt.loc[ok].dt.strftime("%Y-%m-%d")
        return df

    yy = pd.to_numeric(df["year"], errors="coerce")
    dd = pd.to_numeric(df["doy"], errors="coerce")
    ok = yy.notna() & dd.notna()
    df = df.loc[ok].copy()
    base = pd.to_datetime(yy.astype(int).astype(str) + "-01-01", errors="coerce")
    dt = base + pd.to_timedelta(dd.astype(int) - 1, unit="D")
    df["date"] = dt.dt.strftime("%Y-%m-%d")
    return df

def doy_to_date_str(year: int, doy: int) -> str:
    dt = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(int(doy) - 1, unit="D")
    return dt.strftime("%Y-%m-%d")

def build_regular_index_from_grid_index(grid_index: pd.DataFrame):
    """
    Build ix_map, iy_map, lon_vals, lat_vals using rounded lon/lat from grid_index.
    """
    gi = grid_index.copy()
    if "grid_id" not in gi.columns:
        raise ValueError("grid_index.csv missing grid_id")

    if ("lon_r" not in gi.columns) or ("lat_r" not in gi.columns):
        if ("lon" in gi.columns) and ("lat" in gi.columns):
            gi["lon_r"] = gi["lon"].astype(float).round(ROUND_COORD)
            gi["lat_r"] = gi["lat"].astype(float).round(ROUND_COORD)
        else:
            raise ValueError("grid_index.csv must contain lon/lat or lon_r/lat_r")

    gi["grid_id"] = gi["grid_id"].astype(int)
    gi["lon_r"] = gi["lon_r"].astype(float).round(ROUND_COORD)
    gi["lat_r"] = gi["lat_r"].astype(float).round(ROUND_COORD)

    lon_vals = np.sort(np.unique(gi["lon_r"].to_numpy()))
    lat_vals = np.sort(np.unique(gi["lat_r"].to_numpy()))
    nx, ny = lon_vals.size, lat_vals.size

    lon_to_ix = {float(v): int(i) for i, v in enumerate(lon_vals)}
    lat_to_iy = {float(v): int(i) for i, v in enumerate(lat_vals)}

    ngrid = int(gi["grid_id"].max()) + 1
    ix_map = np.full(ngrid, -1, dtype=np.int32)
    iy_map = np.full(ngrid, -1, dtype=np.int32)

    for r in gi.itertuples(index=False):
        gid = int(r.grid_id)
        ix_map[gid] = lon_to_ix[float(r.lon_r)]
        iy_map[gid] = lat_to_iy[float(r.lat_r)]

    log(f"[grid] regular index: nx={nx}, ny={ny}, ngrid={ngrid}")
    log(f"[grid] rough A memory (uint8, 92 days): ~{(nx*ny*92)/1024/1024:.1f} MB")
    return ix_map, iy_map, lon_vals, lat_vals, nx, ny

def _filter_year(chunk: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    强制过滤 year==目标年（防 bucket 混年导致事件合并）。
    """
    if "year" not in chunk.columns:
        raise ValueError("Bucket missing 'year' column.")
    yy = pd.to_numeric(chunk["year"], errors="coerce")
    return chunk.loc[yy.eq(int(year))].copy()

def compute_year_doy_range(bucket_fp: Path, year: int, mask_col: str):
    """
    Pass1: find dmin/dmax among event rows, strictly year==target_year.
    Return (has_rows, dmin, dmax, n_event_rows)
    """
    if not bucket_fp.exists():
        return (False, None, None, 0)

    dmin, dmax = None, None
    cnt = 0
    for chunk in chunk_iter_csv(bucket_fp, CHUNK_SIZE):
        chunk = _filter_year(chunk, year)
        if chunk.empty:
            continue
        if mask_col not in chunk.columns:
            raise ValueError(f"Missing event mask column '{mask_col}' in {bucket_fp.name}")

        m = mask_event_rows(chunk, mask_col)
        if not m.any():
            continue

        doy = pd.to_numeric(chunk.loc[m, "doy"], errors="coerce").dropna().astype(int)
        if doy.empty:
            continue

        cnt += int(doy.size)
        mn, mx = int(doy.min()), int(doy.max())
        dmin = mn if dmin is None else min(dmin, mn)
        dmax = mx if dmax is None else max(dmax, mx)

    if cnt == 0 or dmin is None or dmax is None:
        return (False, None, None, 0)
    return (True, int(dmin), int(dmax), int(cnt))

def build_A_and_labels(bucket_fp: Path, year: int, dmin: int, dmax: int,
                       ix_map: np.ndarray, iy_map: np.ndarray, nx: int, ny: int):
    """
    Pass2: build A (uint8) by chunk reading, strictly year==target_year, then cc3d labels.
    """
    nt = dmax - dmin + 1
    A = np.zeros((nt, ny, nx), dtype=np.uint8)

    for chunk in chunk_iter_csv(bucket_fp, CHUNK_SIZE):
        chunk = _filter_year(chunk, year)
        if chunk.empty:
            continue

        if EVENT_MASK_MODE not in chunk.columns:
            raise ValueError(f"Missing {EVENT_MASK_MODE} in {bucket_fp.name}")

        m = mask_event_rows(chunk, EVENT_MASK_MODE)
        if not m.any():
            continue

        sub = chunk.loc[m, ["grid_id", "doy"]].copy()
        sub["grid_id"] = pd.to_numeric(sub["grid_id"], errors="coerce")
        sub["doy"] = pd.to_numeric(sub["doy"], errors="coerce")
        sub = sub.dropna(subset=["grid_id", "doy"])
        if sub.empty:
            continue

        gid = sub["grid_id"].astype(int).to_numpy(np.int32)
        doy = sub["doy"].astype(int).to_numpy(np.int16)

        if gid.max() >= ix_map.shape[0]:
            raise ValueError(f"grid_id out of range. max_gid={gid.max()}, ix_map_size={ix_map.shape[0]}")

        ix = ix_map[gid]
        iy = iy_map[gid]
        ok = (ix >= 0) & (iy >= 0)
        if not np.any(ok):
            continue

        ix = ix[ok]
        iy = iy[ok]
        doy = doy[ok]
        t = (doy - dmin).astype(np.int16)

        ok2 = (t >= 0) & (t < nt)
        if not np.any(ok2):
            continue

        A[t[ok2], iy[ok2], ix[ok2]] = 1

    ones = int(A.sum())
    log(f"[year={year}] built A: shape=(t={nt}, y={ny}, x={nx}), ones={ones:,}, doy_range={dmin}-{dmax}")
    if ones == 0:
        return None, None, nt

    labels = cc3d.connected_components(A, connectivity=CONNECTIVITY)
    nlab = int(labels.max())
    log(f"[year={year}] cc3d labels: n_components={nlab:,}")
    if nlab == 0:
        return None, None, nt

    return A, labels, nt

def compute_label_metrics(labels: np.ndarray, dmin: int, lon_vals: np.ndarray, lat_vals: np.ndarray):
    """
    Pass2b: derive per-label metrics WITHOUT reading full rows.
    """
    nt, ny, nx = labels.shape
    nlab = int(labels.max())

    n_records = np.bincount(labels.ravel(), minlength=nlab + 1).astype(np.int64)
    n_records[0] = 0

    duration = np.zeros(nlab + 1, dtype=np.int32)
    start_t = np.full(nlab + 1, nt, dtype=np.int32)
    end_t   = np.full(nlab + 1, -1, dtype=np.int32)
    max_daily_extent = np.zeros(nlab + 1, dtype=np.int32)

    min_lon = np.full(nlab + 1, np.inf, dtype=float)
    max_lon = np.full(nlab + 1, -np.inf, dtype=float)
    min_lat = np.full(nlab + 1, np.inf, dtype=float)
    max_lat = np.full(nlab + 1, -np.inf, dtype=float)

    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)  # (ny,nx)
    lon_flat = lon2d.ravel()
    lat_flat = lat2d.ravel()

    for tt in range(nt):
        lab2d = labels[tt]
        labs = np.unique(lab2d)
        labs = labs[labs > 0]
        if labs.size > 0:
            duration[labs] += 1
            np.minimum.at(start_t, labs, tt)
            np.maximum.at(end_t, labs, tt)

        bc = np.bincount(lab2d.ravel(), minlength=nlab + 1).astype(np.int32)
        bc[0] = 0
        max_daily_extent = np.maximum(max_daily_extent, bc)

        lab_flat = lab2d.ravel()
        nz = lab_flat > 0
        if np.any(nz):
            labs_nz = lab_flat[nz]
            np.minimum.at(min_lon, labs_nz, lon_flat[nz])
            np.maximum.at(max_lon, labs_nz, lon_flat[nz])
            np.minimum.at(min_lat, labs_nz, lat_flat[nz])
            np.maximum.at(max_lat, labs_nz, lat_flat[nz])

    duration[0] = 0

    start_doy = dmin + start_t
    end_doy = dmin + end_t
    start_doy[0] = 0
    end_doy[0] = 0

    # n_grids_total: union of spatial cells across time
    L = labels.reshape(nt, -1)  # (nt, ny*nx)
    n_grids_total = np.zeros(nlab + 1, dtype=np.int32)
    for j in range(L.shape[1]):
        labs = np.unique(L[:, j])
        labs = labs[labs > 0]
        if labs.size:
            n_grids_total[labs] += 1
    n_grids_total[0] = 0

    bad = ~np.isfinite(min_lon)
    min_lon[bad] = np.nan; max_lon[bad] = np.nan
    min_lat[bad] = np.nan; max_lat[bad] = np.nan

    return {
        "nlab": nlab,
        "n_records": n_records,
        "duration": duration,
        "start_doy": start_doy.astype(np.int32),
        "end_doy": end_doy.astype(np.int32),
        "max_daily_extent": max_daily_extent.astype(np.int32),
        "n_grids_total": n_grids_total.astype(np.int32),
        "lon_min": min_lon,
        "lon_max": max_lon,
        "lat_min": min_lat,
        "lat_max": max_lat,
    }

def write_events_streaming(bucket_fp: Path, year: int, dmin: int, labels: np.ndarray,
                           ix_map: np.ndarray, iy_map: np.ndarray,
                           keep_label_mask: np.ndarray, lab2eid: np.ndarray,
                           year_dir: Path):
    """
    Pass3: chunk read -> strictly year==target_year -> assign cc_label via labels[t,iy,ix] -> append to per-event CSVs.
    """
    if not WRITE_EVENT_CSVS:
        return

    header_written = set()

    for chunk in chunk_iter_csv(bucket_fp, CHUNK_SIZE):
        chunk = _filter_year(chunk, year)
        if chunk.empty:
            continue

        if EVENT_MASK_MODE not in chunk.columns:
            raise ValueError(f"Missing {EVENT_MASK_MODE} in {bucket_fp.name}")

        m = mask_event_rows(chunk, EVENT_MASK_MODE)
        if not m.any():
            continue

        sub = chunk.loc[m].copy()

        sub["grid_id"] = pd.to_numeric(sub["grid_id"], errors="coerce")
        sub["doy"] = pd.to_numeric(sub["doy"], errors="coerce")
        sub = sub.dropna(subset=["grid_id", "doy"])
        if sub.empty:
            continue

        gid = sub["grid_id"].astype(int).to_numpy(np.int32)
        doy = sub["doy"].astype(int).to_numpy(np.int16)

        if gid.max() >= ix_map.shape[0]:
            raise ValueError(f"grid_id out of range in {bucket_fp.name}. max_gid={gid.max()}")

        ix = ix_map[gid]
        iy = iy_map[gid]
        ok = (ix >= 0) & (iy >= 0)
        if not np.any(ok):
            continue

        sub = sub.loc[ok].copy()
        ix = ix[ok]; iy = iy[ok]; doy = doy[ok]

        t = (doy - dmin).astype(np.int16)
        ok2 = (t >= 0) & (t < labels.shape[0])
        if not np.any(ok2):
            continue

        sub = sub.loc[ok2].copy()
        ix = ix[ok2]; iy = iy[ok2]; t = t[ok2]

        cc_lab = labels[t, iy, ix].astype(np.int32)
        keep = keep_label_mask[cc_lab]
        if not np.any(keep):
            continue

        sub = sub.loc[keep].copy()
        cc_lab = cc_lab[keep]
        ix = ix[keep]; iy = iy[keep]

        sub["cc_label"] = cc_lab
        sub["event_id"] = lab2eid[cc_lab].astype(np.int32)
        sub["ix"] = ix.astype(np.int32)
        sub["iy"] = iy.astype(np.int32)
        sub["year"] = int(year)

        if not KEEP_ALL_COLUMNS:
            cols = [c for c in MIN_KEEP_COLS if c in sub.columns]
            for c in ["year","doy","grid_id","event_id","cc_label","ix","iy"]:
                if c in sub.columns and c not in cols:
                    cols.append(c)
            sub = sub[cols].copy()

        # ensure date
        if ("date" not in sub.columns) and ("doy" in sub.columns):
            sub["date"] = [doy_to_date_str(year, int(d)) for d in sub["doy"].astype(int).to_numpy()]
        else:
            sub = ensure_date_column(sub)

        sort_cols = [c for c in ["event_id", "doy", "grid_id"] if c in sub.columns]
        if sort_cols:
            sub = sub.sort_values(sort_cols, kind="mergesort")

        for eid, g in sub.groupby("event_id", sort=True):
            out_fp = year_dir / f"event_{year}_{int(eid):05d}.csv"
            key = str(out_fp)
            write_header = (key not in header_written) and (not out_fp.exists())
            g.to_csv(out_fp, mode="a", index=False, header=write_header, encoding="utf-8")
            header_written.add(key)


# =========================
# Per-year pipeline
# =========================
def run_one_year(year: int, ix_map, iy_map, lon_vals, lat_vals, nx, ny, fail_log: Path):
    bucket_fp = BUCKET_DIR / f"{year}.csv"
    if not bucket_fp.exists():
        log(f"[year={year}] bucket missing -> skip")
        return None

    try:
        # Pass1
        has_rows, dmin, dmax, nrows = compute_year_doy_range(bucket_fp, year, EVENT_MASK_MODE)
        if not has_rows:
            log(f"[year={year}] no event rows -> skip")
            return {
                "year": int(year), "n_events": 0,
                "mean_duration_days": 0.0, "median_duration_days": 0.0,
                "mean_max_extent_grids": 0.0, "median_max_extent_grids": 0.0,
                "p90_max_extent_grids": 0.0, "mean_n_grids_total": 0.0,
                "event_mask_mode": EVENT_MASK_MODE, "connectivity": int(CONNECTIVITY),
                "min_duration_days": int(MIN_DURATION_DAYS),
                "doy_min_in_bucket": np.nan, "doy_max_in_bucket": np.nan
            }
        log(f"[year={year}] event rows ~ {nrows:,} | doy_range={dmin}-{dmax}")

        # Pass2
        A, labels, nt = build_A_and_labels(bucket_fp, year, dmin, dmax, ix_map, iy_map, nx, ny)
        if labels is None:
            log(f"[year={year}] labels empty -> skip")
            return None

        # Pass2b
        met = compute_label_metrics(labels, dmin, lon_vals, lat_vals)
        nlab = met["nlab"]

        keep_labels = np.where(met["duration"] >= MIN_DURATION_DAYS)[0].astype(np.int32)
        keep_labels = keep_labels[keep_labels > 0]
        if keep_labels.size == 0:
            log(f"[year={year}] no labels pass MIN_DURATION_DAYS={MIN_DURATION_DAYS}")
            return None

        keep_label_mask = np.zeros(nlab + 1, dtype=bool)
        keep_label_mask[keep_labels] = True

        stat = pd.DataFrame({
            "cc_label": keep_labels.astype(int),
            "start_doy": met["start_doy"][keep_labels].astype(int),
            "end_doy": met["end_doy"][keep_labels].astype(int),
            "duration_days": met["duration"][keep_labels].astype(int),
            "n_records": met["n_records"][keep_labels].astype(np.int64),
            "n_grids_total": met["n_grids_total"][keep_labels].astype(int),
            "max_extent_grids": met["max_daily_extent"][keep_labels].astype(int),
            "lon_min": met["lon_min"][keep_labels],
            "lon_max": met["lon_max"][keep_labels],
            "lat_min": met["lat_min"][keep_labels],
            "lat_max": met["lat_max"][keep_labels],
        })

        stat = stat.sort_values(["start_doy", "end_doy", "n_records"], ascending=[True, True, False]).reset_index(drop=True)
        stat["event_id"] = np.arange(1, len(stat) + 1, dtype=int)
        stat["start_date"] = [doy_to_date_str(year, int(d)) for d in stat["start_doy"].to_numpy()]
        stat["end_date"]   = [doy_to_date_str(year, int(d)) for d in stat["end_doy"].to_numpy()]

        lab2eid = np.zeros(nlab + 1, dtype=np.int32)
        lab2eid[stat["cc_label"].to_numpy(int)] = stat["event_id"].to_numpy(int)

        year_dir = EVENTS_ROOT / f"{year}"
        if OVERWRITE_YEAR_DIR and year_dir.exists():
            shutil.rmtree(year_dir, ignore_errors=True)
        safe_mkdir(year_dir)

        # Pass3
        write_events_streaming(bucket_fp, year, dmin, labels, ix_map, iy_map, keep_label_mask, lab2eid, year_dir)

        stat_fp = year_dir / f"events_{year}_summary.csv"
        stat.to_csv(stat_fp, index=False, encoding="utf-8")
        log(f"[year={year}] kept events={len(stat):,} | summary -> {stat_fp}")

        ann = {
            "year": int(year),
            "n_events": int(len(stat)),
            "mean_duration_days": float(stat["duration_days"].mean()) if len(stat) else 0.0,
            "median_duration_days": float(stat["duration_days"].median()) if len(stat) else 0.0,
            "mean_max_extent_grids": float(stat["max_extent_grids"].mean()) if len(stat) else 0.0,
            "median_max_extent_grids": float(stat["max_extent_grids"].median()) if len(stat) else 0.0,
            "p90_max_extent_grids": float(np.nanpercentile(stat["max_extent_grids"].values, 90)) if len(stat) else 0.0,
            "mean_n_grids_total": float(stat["n_grids_total"].mean()) if len(stat) else 0.0,
            "event_mask_mode": EVENT_MASK_MODE,
            "connectivity": int(CONNECTIVITY),
            "min_duration_days": int(MIN_DURATION_DAYS),
            "doy_min_in_bucket": int(dmin),
            "doy_max_in_bucket": int(dmax),
        }
        return ann

    except Exception as e:
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(f"{year}\t{repr(e)}\n")
            f.write(traceback.format_exc() + "\n")
        log(f"[ERROR year={year}] {repr(e)}")
        return None


# =========================
# Main
# =========================
def main():
    if not GRID_INDEX_FP.exists():
        raise FileNotFoundError(f"Missing grid_index.csv: {GRID_INDEX_FP}")
    if not BUCKET_DIR.exists():
        raise FileNotFoundError(f"Missing buckets dir: {BUCKET_DIR}")

    safe_mkdir(EVENTS_ROOT)

    fail_log = EVENTS_ROOT / "failures_cc3d_from_buckets_streaming.txt"
    with open(fail_log, "w", encoding="utf-8") as f:
        f.write("year\terror\n")

    grid_index = pd.read_csv(GRID_INDEX_FP)
    ix_map, iy_map, lon_vals, lat_vals, nx, ny = build_regular_index_from_grid_index(grid_index)

    annual_rows = []
    for y in YEARS_TO_RUN:
        ann = run_one_year(int(y), ix_map, iy_map, lon_vals, lat_vals, nx, ny, fail_log)
        if ann is not None:
            annual_rows.append(ann)

    am = pd.DataFrame(annual_rows).sort_values("year") if annual_rows else pd.DataFrame()
    out_am = EVENTS_ROOT / "annual_metrics.csv"
    am.to_csv(out_am, index=False, encoding="utf-8")

    print("\nALL DONE.")
    print("events root :", EVENTS_ROOT)
    print("annual      :", out_am)
    print("fail log    :", fail_log)


if __name__ == "__main__":
    main()