# -*- coding: utf-8 -*-
"""
ERA5 (bucket-based) CC3D event extraction (streaming writer, low RAM peak) — SPEEDUP + FIXED (pandas dtype)
========================================================================================================
修复：
- 不在 read_csv(dtype=...) 使用 pandas 扩展 dtype(Int64/Int8)，避免 TypeError: Int64Dtype()
- chunk 内用 pd.to_numeric(errors="coerce") 强制转换，保证鲁棒

提速（不改变输出内容/事件定义/排序规则）：
1) n_grids_total 精确快速：对 (label, space_id) 编码 unique -> bincount(label)
2) bbox 用 lon_flat/lat_flat 预先构建，避免每年 meshgrid
3) Pass1/Pass2 用 usecols 只读必要列
4) Pass3 写 event 文件避免 groupby：排序后按 event_id 切段写

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
import re
from collections import OrderedDict

# =========================
# USER SETTINGS
# =========================
OUT_ROOT = Path(r"E:\temp_events_ERA5_S1S6_NatureROLL30滚动的所有数据版本")
BUCKET_DIR = OUT_ROOT / "buckets"
GRID_INDEX_FP = OUT_ROOT / "grid_index.csv"
EVENTS_ROOT = OUT_ROOT / "events_cc3d222"

START_YEAR = 1950
END_YEAR   = 2024
YEARS_TO_RUN = range(START_YEAR, END_YEAR + 1)

EVENT_MASK_MODE = "heat3"  # "heat3" / "DHW_sync3" / "DHW_predry3"
CONNECTIVITY = 26
MIN_DURATION_DAYS = 3

ROUND_COORD = 4
CHUNK_SIZE = 1_000_000  # 内存足够可以调大：2_000_000 会更快

# 输出控制
WRITE_EVENT_CSVS = True
OVERWRITE_YEAR_DIR = True
KEEP_ALL_COLUMNS = True  # 若硬盘压力大可改 False（但你要求不改变内容 -> 建议保持 True）

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

# read_csv 常用参数（不传 dtype，最稳）
READ_KW = dict(engine="c", low_memory=False, memory_map=True)

# 为防止事件多时打开文件句柄爆掉（Windows 常见上限较低）
MAX_OPEN_FILES = 128


# =========================
# Utilities
# =========================
def log(msg: str):
    if VERBOSE:
        print(msg)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_csv_chunks(fp: Path, chunksize: int, usecols=None):
    return pd.read_csv(fp, chunksize=chunksize, usecols=usecols, **READ_KW)

def to_num(a, default=np.nan):
    # 返回 numpy float array
    out = pd.to_numeric(a, errors="coerce").to_numpy()
    if default is not np.nan:
        out = np.where(np.isfinite(out), out, default)
    return out

def year_mask(chunk: pd.DataFrame, year: int) -> np.ndarray:
    yy = to_num(chunk["year"])
    return yy == float(int(year))

def event_mask(chunk: pd.DataFrame, mask_col: str) -> np.ndarray:
    mm = to_num(chunk[mask_col], default=0.0)
    return mm.astype(np.int8) == 1

_date_iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def ensure_date_column_fast(sub: pd.DataFrame) -> pd.DataFrame:
    """
    等价于你原来的 ensure_date_column：
    - 若 date 存在：解析，无法解析的行删除；最后 YYYY-MM-DD
    - 若 date 不存在：用 year+doy 生成
    """
    if "date" in sub.columns:
        s = sub["date"].astype(str)
        is_iso = s.str.match(_date_iso_re)
        if is_iso.all():
            return sub

        dt = pd.to_datetime(s.where(~is_iso, other=np.nan), errors="coerce")
        dt_iso = pd.to_datetime(s.where(is_iso, other=np.nan), errors="coerce")
        dt = dt.fillna(dt_iso)

        ok = dt.notna()
        sub = sub.loc[ok].copy()
        sub.loc[:, "date"] = dt.loc[ok].dt.strftime("%Y-%m-%d")
        return sub

    yy = pd.to_numeric(sub["year"], errors="coerce")
    dd = pd.to_numeric(sub["doy"], errors="coerce")
    ok = yy.notna() & dd.notna()
    sub = sub.loc[ok].copy()
    base = pd.to_datetime(yy.astype(int).astype(str) + "-01-01", errors="coerce")
    dt = base + pd.to_timedelta(dd.astype(int) - 1, unit="D")
    sub.loc[:, "date"] = dt.dt.strftime("%Y-%m-%d")
    return sub

def doy_to_date_str(year: int, doy: int) -> str:
    dt = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(int(doy) - 1, unit="D")
    return dt.strftime("%Y-%m-%d")

def build_regular_index_from_grid_index(grid_index: pd.DataFrame):
    """
    Build ix_map, iy_map, lon_vals, lat_vals, lon_flat, lat_flat, nx, ny.
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

    gi["grid_id"] = pd.to_numeric(gi["grid_id"], errors="coerce").astype(int)
    gi["lon_r"] = pd.to_numeric(gi["lon_r"], errors="coerce").astype(float).round(ROUND_COORD)
    gi["lat_r"] = pd.to_numeric(gi["lat_r"], errors="coerce").astype(float).round(ROUND_COORD)

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

    # 预构建 flat lon/lat（ny*nx）
    lon_flat = np.tile(lon_vals.astype(float), ny)          # (ny*nx,)
    lat_flat = np.repeat(lat_vals.astype(float), nx)        # (ny*nx,)

    log(f"[grid] regular index: nx={nx}, ny={ny}, ngrid={ngrid}")
    log(f"[grid] rough A memory (uint8, 92 days): ~{(nx*ny*92)/1024/1024:.1f} MB")
    return ix_map, iy_map, lon_vals, lat_vals, lon_flat, lat_flat, nx, ny


# =========================
# Pass1: doy range
# =========================
def compute_year_doy_range(bucket_fp: Path, year: int, mask_col: str):
    """
    Find dmin/dmax among event rows, strictly year==target_year.
    Return (has_rows, dmin, dmax, n_event_rows)
    """
    if not bucket_fp.exists():
        return (False, None, None, 0)

    usecols = ["year", mask_col, "doy"]

    dmin, dmax = None, None
    cnt = 0

    for chunk in read_csv_chunks(bucket_fp, CHUNK_SIZE, usecols=usecols):
        if chunk.empty:
            continue
        ym = year_mask(chunk, year)
        if not ym.any():
            continue
        em = event_mask(chunk, mask_col)
        m = ym & em
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


# =========================
# Pass2: build A + cc3d labels
# =========================
def build_A_and_labels(bucket_fp: Path, year: int, dmin: int, dmax: int,
                       ix_map: np.ndarray, iy_map: np.ndarray, nx: int, ny: int):
    nt = dmax - dmin + 1
    A = np.zeros((nt, ny, nx), dtype=np.uint8)

    usecols = ["year", EVENT_MASK_MODE, "grid_id", "doy"]

    for chunk in read_csv_chunks(bucket_fp, CHUNK_SIZE, usecols=usecols):
        if chunk.empty:
            continue
        ym = year_mask(chunk, year)
        if not ym.any():
            continue
        em = event_mask(chunk, EVENT_MASK_MODE)
        m = ym & em
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
        if not ok.any():
            continue

        ix = ix[ok]
        iy = iy[ok]
        doy = doy[ok]

        t = (doy - dmin).astype(np.int16)
        ok2 = (t >= 0) & (t < nt)
        if not ok2.any():
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


# =========================
# Pass2b: metrics from labels (FAST exact)
# =========================
def compute_n_grids_total_fast(labels: np.ndarray, nlab: int) -> np.ndarray:
    """
    精确计算每个 label 的 n_grids_total（跨时间的空间并集格点数）
    方法：取所有非0体素 -> (label, space_id=y*nx+x) 编码 -> unique -> bincount(label)
    """
    nt, ny, nx = labels.shape
    t, y, x = np.nonzero(labels)
    out = np.zeros(nlab + 1, dtype=np.int32)
    if t.size == 0:
        return out

    lab = labels[t, y, x].astype(np.int64)
    s = (y.astype(np.int64) * nx + x.astype(np.int64))
    key = lab * (nx * ny) + s
    key = np.unique(key)
    lab_u = (key // (nx * ny)).astype(np.int64)

    out = np.bincount(lab_u, minlength=nlab + 1).astype(np.int32)
    out[0] = 0
    return out

def compute_label_metrics_fast(labels: np.ndarray, dmin: int,
                              lon_flat: np.ndarray, lat_flat: np.ndarray,
                              nx: int, ny: int):
    nt, ny0, nx0 = labels.shape
    assert ny0 == ny and nx0 == nx

    nlab = int(labels.max())

    # n_records: voxel count
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

    for tt in range(nt):
        lab2d = labels[tt]  # (ny,nx)

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
    end_doy   = dmin + end_t
    start_doy[0] = 0
    end_doy[0] = 0

    n_grids_total = compute_n_grids_total_fast(labels, nlab)

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


# =========================
# Pass3: streaming write per-event CSVs (FAST, no groupby) + file handle cap
# =========================
def prepare_output_columns(sample_fp: Path):
    cols = pd.read_csv(sample_fp, nrows=0, **READ_KW).columns.tolist()

    extra = []
    for c in ["cc_label", "event_id", "ix", "iy"]:
        if c not in cols:
            extra.append(c)
    if "date" not in cols:
        extra.append("date")

    out_cols = cols + extra
    return cols, out_cols

class FileHandlePool:
    def __init__(self, max_open: int):
        self.max_open = int(max_open)
        self.pool = OrderedDict()  # eid -> (fp, handle)

    def get(self, eid: int, fp: Path, header_line: str):
        # 触碰更新 LRU
        if eid in self.pool:
            fh = self.pool.pop(eid)[1]
            self.pool[eid] = (fp, fh)
            return fh

        # 超限：关掉最旧的
        while len(self.pool) >= self.max_open:
            _, (_, oldfh) = self.pool.popitem(last=False)
            try:
                oldfh.close()
            except Exception:
                pass

        fp.parent.mkdir(parents=True, exist_ok=True)
        fh = open(fp, "a", encoding="utf-8", newline="")
        # 空文件写 header
        if fp.exists() and fp.stat().st_size == 0:
            fh.write(header_line)
        self.pool[eid] = (fp, fh)
        return fh

    def close_all(self):
        for _, fh in self.pool.values():
            try:
                fh.close()
            except Exception:
                pass
        self.pool.clear()

def write_events_streaming_fast(bucket_fp: Path, year: int, dmin: int, labels: np.ndarray,
                                ix_map: np.ndarray, iy_map: np.ndarray,
                                keep_label_mask: np.ndarray, lab2eid: np.ndarray,
                                year_dir: Path,
                                out_cols: list[str]):
    if not WRITE_EVENT_CSVS:
        return

    header_line = ",".join(out_cols) + "\n"
    pool = FileHandlePool(MAX_OPEN_FILES)

    for chunk in read_csv_chunks(bucket_fp, CHUNK_SIZE, usecols=None):
        if chunk.empty:
            continue

        ym = year_mask(chunk, year)
        if not ym.any():
            continue
        if EVENT_MASK_MODE not in chunk.columns:
            raise ValueError(f"Missing {EVENT_MASK_MODE} in {bucket_fp.name}")
        em = event_mask(chunk, EVENT_MASK_MODE)
        m = ym & em
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
        if not ok.any():
            continue

        sub = sub.loc[ok].copy()
        ix = ix[ok]; iy = iy[ok]; doy = doy[ok]

        t = (doy - dmin).astype(np.int16)
        ok2 = (t >= 0) & (t < labels.shape[0])
        if not ok2.any():
            continue

        sub = sub.loc[ok2].copy()
        ix = ix[ok2]; iy = iy[ok2]; t = t[ok2]

        cc_lab = labels[t, iy, ix].astype(np.int32)
        keep = keep_label_mask[cc_lab]
        if not keep.any():
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

        sub = ensure_date_column_fast(sub)
        if sub.empty:
            continue

        sort_cols = [c for c in ["event_id", "doy", "grid_id"] if c in sub.columns]
        if sort_cols:
            sub = sub.sort_values(sort_cols, kind="mergesort")

        # 固定输出列顺序（保证稳定，不改变内容）
        if KEEP_ALL_COLUMNS:
            for c in out_cols:
                if c not in sub.columns:
                    sub[c] = np.nan
            sub = sub[out_cols]

        e = sub["event_id"].to_numpy()
        if e.size == 0:
            continue

        # event_id 已排序，相同 eid 连续；按边界切段写
        cuts = np.flatnonzero(np.diff(e)) + 1
        bounds = np.concatenate(([0], cuts, [e.size]))

        for i in range(bounds.size - 1):
            a, b = int(bounds[i]), int(bounds[i+1])
            eid = int(e[a])

            out_fp = year_dir / f"event_{year}_{eid:05d}.csv"
            fh = pool.get(eid, out_fp, header_line)

            g = sub.iloc[a:b]
            # 注意：对 file-handle 写时不要传 mode
            g.to_csv(fh, index=False, header=False, encoding="utf-8")

    pool.close_all()


# =========================
# Per-year pipeline
# =========================
def run_one_year(year: int, ix_map, iy_map, lon_flat, lat_flat, nx, ny,
                 fail_log: Path, out_cols: list[str]):
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
        _, labels, _ = build_A_and_labels(bucket_fp, year, dmin, dmax, ix_map, iy_map, nx, ny)
        if labels is None:
            log(f"[year={year}] labels empty -> skip")
            return None

        # Pass2b (FAST exact)
        met = compute_label_metrics_fast(labels, dmin, lon_flat, lat_flat, nx, ny)
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

        # event_id 规则不变
        stat = stat.sort_values(["start_doy", "end_doy", "n_records"],
                                ascending=[True, True, False]).reset_index(drop=True)
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
        write_events_streaming_fast(bucket_fp, year, dmin, labels, ix_map, iy_map,
                                    keep_label_mask, lab2eid, year_dir, out_cols)

        # summary
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

    # grid mapping
    grid_index = pd.read_csv(GRID_INDEX_FP, **READ_KW)
    ix_map, iy_map, lon_vals, lat_vals, lon_flat, lat_flat, nx, ny = build_regular_index_from_grid_index(grid_index)

    # 用任一存在的 bucket 定义输出列顺序（bucket列 + 新增列）
    sample_fp = None
    for y in YEARS_TO_RUN:
        fp = BUCKET_DIR / f"{y}.csv"
        if fp.exists():
            sample_fp = fp
            break
    if sample_fp is None:
        raise FileNotFoundError("No bucket csv found in buckets/")

    _, out_cols = prepare_output_columns(sample_fp)
    log(f"[cols] out_cols={len(out_cols)}")

    annual_rows = []
    for y in YEARS_TO_RUN:
        ann = run_one_year(int(y), ix_map, iy_map, lon_flat, lat_flat, nx, ny, fail_log, out_cols)
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