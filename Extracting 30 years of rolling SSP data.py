# -*- coding: utf-8 -*-
"""
Unified STATES6 + CC3D on SCENARIO MME (calendar-robust) + ROLLING 30-YR T90(doy±window)
======================================================================================

CHANGES vs your original:
1) T90_by_doy is no longer from fixed BASELINE_START/END.
2) For each year y, compute rolling (trailing) 30-year climatology:
      base_years = [y-29, y]  (clamped at START_YEAR at early edge)
   then compute per-grid daily T90 using doy±DOY_WINDOW within that rolling base.
3) Everything else (heat3 definition, CC3D extraction, outputs) unchanged.

Notes:
- Rolling mode is trailing (no future leakage).
- Early years edge handling:
    ROLLING_EDGE_MODE="clamp"  -> use earliest available 30-yr window (START..START+29)
    ROLLING_EDGE_MODE="skip"   -> skip years with <30yr history
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
import cc3d
import traceback


# =========================
# USER SETTINGS
# =========================
SCENARIO_DIRS = {
    "ssp126": Path(r"E:\第二篇的修改20251229开始修改\ssp126\ssp126REGRID_100km\每模型合并后"),
    "ssp245": Path(r"E:\第二篇的修改20251229开始修改\ssp245\ssp245REGRID_100km\每模型合并后"),
    "ssp585": Path(r"E:\第二篇的修改20251229开始修改\ssp585\REGRID_100km\每模型合并后"),
}

OUT_ROOT = Path(r"E:\第二篇的修改20251229开始修改\CC3D_STATES6_MME_outputs滚动30年")

START_YEAR, END_YEAR = 1950, 2100
DOY_WINDOW = 5
CONSEC_DAYS = 3
SEASON_MONTHS = (6, 7, 8)

EVENT_MASK_MODE = "heat3"   # "heat3" / "DHW_sync3" / "DHW_predry3"

CONNECTIVITY = 26
MIN_DURATION_DAYS = 3
ROUND_COORD = 4

TAS_CANDS = ["tas", "tasmean", "t2m", "temperature_2m"]
SM_CANDS = ["mrsos", "mrso", "soil_moist", "sm", "volumetric_soil_water_layer_1"]
VERT_DIMS_CANDS = ["depth", "lev", "level", "layer", "soil_layer", "levgrnd", "sdepth", "z", "nz"]

# ---- Rolling threshold config ----
ROLLING_YEARS = 30
ROLLING_MODE = "trailing"        # keep; (centered would use future)
ROLLING_EDGE_MODE = "clamp"      # "clamp" or "skip"
# ----------------------------------

VERBOSE = True


# =========================
# Helpers
# =========================
def log(msg: str):
    if VERBOSE:
        print(msg)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sniff_nc_signature(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "HDF5"
    if head[:3] == b"CDF":
        return "CDF"
    return "UNKNOWN"

def open_dataset_safe(path: Path) -> xr.Dataset:
    sig = sniff_nc_signature(path)
    if sig == "HDF5":
        return xr.open_dataset(str(path), engine="h5netcdf", cache=False, decode_times=True, use_cftime=True)
    if sig == "CDF":
        return xr.open_dataset(str(path), engine="scipy", cache=False, decode_times=True, use_cftime=True)
    raise RuntimeError(f"[OPEN FAIL] Unknown netCDF signature: {path}")

def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename["longitude"] = "lon"
    if "latitude" in ds.dims and "lat" not in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims and "lon" not in ds.dims:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    return ds

def find_var(ds: xr.Dataset, candidates):
    for c in candidates:
        if c in ds.data_vars:
            return c
    for v in ds.data_vars:
        dims = set(ds[v].dims)
        if "time" in dims and ("lat" in dims or "latitude" in dims) and ("lon" in dims or "longitude" in dims):
            return v
    return list(ds.data_vars)[0]

def model_name_from_filename(fp: Path):
    m = re.match(r"(.+?)_(tas|mrsos)_\d{4}_\d{4}_merged\.nc$", fp.name)
    if m:
        return m.group(1)
    x = re.split(r"_tas_|_mrsos_", fp.name, maxsplit=1)
    return x[0]

def discover_model_pairs(scen_dir: Path):
    tas_files = sorted(scen_dir.glob("*_tas_*_merged.nc"))
    sm_files = sorted(scen_dir.glob("*_mrsos_*_merged.nc"))
    tas_map = {model_name_from_filename(p): p for p in tas_files}
    sm_map = {model_name_from_filename(p): p for p in sm_files}
    models = sorted(set(tas_map.keys()) & set(sm_map.keys()))
    return [(m, tas_map[m], sm_map[m]) for m in models]


# =========================
# Array shape normalization
# =========================
def choose_vertical_dim(da: xr.DataArray):
    lower_cands = [x.lower() for x in VERT_DIMS_CANDS]
    for d in da.dims:
        if d.lower() in lower_cands:
            return d
    for d in da.dims:
        if d.lower() not in ("time", "lat", "lon", "latitude", "longitude"):
            return d
    return None

def drop_noncore_coords(da: xr.DataArray) -> xr.DataArray:
    keep = set(da.dims) | {"time", "lat", "lon"}
    drops = [c for c in da.coords if c not in keep]
    if drops:
        da = da.drop_vars(drops, errors="ignore")
    return da

def ensure_3d_time_lat_lon(da: xr.DataArray, var_role: str) -> xr.DataArray:
    rename = {}
    for d in da.dims:
        if d == "latitude":
            rename["latitude"] = "lat"
        elif d == "longitude":
            rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

    if var_role == "sm":
        vdim = choose_vertical_dim(da)
        if vdim is not None and vdim not in ("time", "lat", "lon"):
            da = da.isel({vdim: 0}, drop=True)  # top layer

    req = ["time", "lat", "lon"]
    for r in req:
        if r not in da.dims:
            raise ValueError(f"Missing required dim {r}; dims={da.dims}")

    extra = [d for d in da.dims if d not in req]
    for d in extra:
        da = da.isel({d: 0}, drop=True)

    da = da.transpose("time", "lat", "lon")
    da = drop_noncore_coords(da)
    return da


# =========================
# Calendar-robust time key
# =========================
def to_tkey_da(da: xr.DataArray) -> xr.DataArray:
    t = da["time"].values
    years = np.array([tt.year for tt in t], dtype=int)
    months = np.array([tt.month for tt in t], dtype=int)
    days = np.array([tt.day for tt in t], dtype=int)

    tkey = np.array([f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)], dtype=object)

    mask = (years >= START_YEAR) & (years <= END_YEAR)
    da2 = da.isel(time=np.where(mask)[0])

    tkey2 = tkey[mask]
    _, idx_unique = np.unique(tkey2, return_index=True)
    idx_unique = np.sort(idx_unique)
    da2 = da2.isel(time=idx_unique)
    tkey2 = tkey2[idx_unique]

    da2 = da2.assign_coords(tkey=("time", tkey2)).swap_dims({"time": "tkey"}).drop_vars("time")
    return da2

def from_tkey_to_time(da_tkey: xr.DataArray) -> xr.DataArray:
    tkey = da_tkey["tkey"].values.astype(str)
    ts = pd.to_datetime(tkey, errors="coerce")
    valid = ~pd.isna(ts)
    if valid.sum() == 0:
        return da_tkey.isel(tkey=slice(0, 0)).rename({"tkey": "time"}).assign_coords(time=("time", pd.to_datetime([])))

    da2 = da_tkey.isel(tkey=np.where(valid)[0])
    ts2 = ts[valid]

    da2 = da2.rename({"tkey": "time"})
    da2 = da2.assign_coords(time=("time", ts2.values))
    da2 = da2.sortby("time")
    return da2


# =========================
# STATES6 core
# =========================
def select_jja(da: xr.DataArray) -> xr.DataArray:
    return da.sel(time=da["time"].dt.month.isin(SEASON_MONTHS))

def heat3_from_heatraw(heat_raw: xr.DataArray) -> xr.DataArray:
    def _one_year(x):
        end = (x.rolling(time=CONSEC_DAYS, min_periods=CONSEC_DAYS).sum() >= CONSEC_DAYS)
        end = end.fillna(False).astype(bool)
        shift1 = end.shift(time=1, fill_value=False)
        shift2 = end.shift(time=2, fill_value=False)
        return (end | shift1 | shift2)
    return heat_raw.groupby("time.year").map(_one_year)


# =========================
# Rolling 30-year daily thresholds (doy ± window)
# =========================
def _get_base_years_for_y(y: int):
    """
    trailing rolling window of length ROLLING_YEARS.
    If not enough history:
      - clamp: use [START_YEAR, START_YEAR+ROLLING_YEARS-1]
      - skip: return None
    """
    if ROLLING_MODE != "trailing":
        raise ValueError("This script uses trailing rolling by design.")

    y0 = y - (ROLLING_YEARS - 1)
    y1 = y

    if y0 < START_YEAR:
        if ROLLING_EDGE_MODE == "skip":
            return None
        elif ROLLING_EDGE_MODE == "clamp":
            y0 = START_YEAR
            y1 = START_YEAR + (ROLLING_YEARS - 1)
        else:
            raise ValueError(f"Unknown ROLLING_EDGE_MODE={ROLLING_EDGE_MODE}")

    # also guard end beyond END_YEAR (shouldn't for trailing)
    if y1 > END_YEAR:
        y1 = END_YEAR
        y0 = max(START_YEAR, y1 - (ROLLING_YEARS - 1))

    return int(y0), int(y1)

def compute_T90_for_year_from_base(tas_jja_all: xr.DataArray, y: int, doy_window: int) -> xr.DataArray:
    """
    Return T90_t for the specific year y (dims: time, lat, lon) for JJA days of that year,
    computed from rolling base years using doy±window.
    """
    base = _get_base_years_for_y(y)
    if base is None:
        return None
    y0, y1 = base

    tas_base = tas_jja_all.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    if tas_base.time.size < 100:
        raise RuntimeError(f"Rolling base too short for year {y}: base {y0}-{y1} has {tas_base.time.size} days")

    doy_base = tas_base["time"].dt.dayofyear
    doys = np.unique(doy_base.values)
    doys = np.array(sorted([int(d) for d in doys]))
    doy_set = set(doys.tolist())

    # compute T90(doy) on base
    T90_list = []
    for d in doys:
        win = [dd for dd in range(int(d) - doy_window, int(d) + doy_window + 1) if dd in doy_set]
        mask = doy_base.isin(win)
        tas_sub = tas_base.where(mask, drop=True)
        T90_d = tas_sub.quantile(0.9, dim="time", skipna=True)
        T90_list.append(T90_d.expand_dims(doy=[int(d)]))
    T90_by_doy = xr.concat(T90_list, dim="doy")

    # map to this year's JJA time axis
    tas_y = tas_jja_all.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
    if tas_y.time.size == 0:
        return None
    doy_y = tas_y["time"].dt.dayofyear.astype(int)
    T90_t = T90_by_doy.sel(doy=doy_y)
    return T90_t


# =========================
# Build scenario MME (calendar robust)
# =========================
def build_scenario_mme(scenario: str, scen_dir: Path, fail_log: Path):
    pairs = discover_model_pairs(scen_dir)
    if len(pairs) == 0:
        log(f"[WARN] {scenario}: no matched model pairs.")
        return None, None, []

    log(f"[{scenario}] matched model pairs: {len(pairs)}")

    tas_list, sm_list, used_models = [], [], []

    for model, tas_fp, sm_fp in pairs:
        ds_t, ds_s = None, None
        try:
            ds_t = standardize_latlon(open_dataset_safe(tas_fp))
            ds_s = standardize_latlon(open_dataset_safe(sm_fp))

            tas_var = find_var(ds_t, TAS_CANDS)
            sm_var = find_var(ds_s, SM_CANDS)

            tas = ensure_3d_time_lat_lon(ds_t[tas_var], var_role="tas")
            sm = ensure_3d_time_lat_lon(ds_s[sm_var], var_role="sm")

            tas = tas.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
            sm = sm.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))

            tas, sm = xr.align(tas, sm, join="inner")
            if tas.time.size == 0:
                raise RuntimeError("Empty after tas-sm inner align")

            tas_k = to_tkey_da(tas)
            sm_k = to_tkey_da(sm)

            tas_k, sm_k = xr.align(tas_k, sm_k, join="inner")
            if tas_k.sizes.get("tkey", 0) == 0:
                raise RuntimeError("Empty after tas-sm tkey align")

            tas_list.append(tas_k.expand_dims(model=[model]))
            sm_list.append(sm_k.expand_dims(model=[model]))
            used_models.append(model)

            log(f"[{scenario}] + model {model} | tkey={tas_k.sizes['tkey']} lat={tas_k.sizes['lat']} lon={tas_k.sizes['lon']}")

        except Exception as e:
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{scenario}\t{model}\tMME_BUILD\t{repr(e)}\n")
            log(f"[{scenario}] skip model {model}: {repr(e)}")
        finally:
            try:
                if ds_t is not None: ds_t.close()
            except Exception:
                pass
            try:
                if ds_s is not None: ds_s.close()
            except Exception:
                pass

    if len(tas_list) == 0:
        return None, None, []

    tas_all = xr.concat(tas_list, dim="model", join="inner", coords="minimal", compat="override")
    sm_all = xr.concat(sm_list, dim="model", join="inner", coords="minimal", compat="override")

    if tas_all.sizes.get("tkey", 0) == 0 or sm_all.sizes.get("tkey", 0) == 0:
        raise RuntimeError(f"[{scenario}] MME concat produced zero tkey. Check model calendar/time coverage.")

    tas_mme_k = tas_all.mean(dim="model", skipna=True)
    sm_mme_k = sm_all.mean(dim="model", skipna=True)

    tas_mme = from_tkey_to_time(tas_mme_k)
    sm_mme = from_tkey_to_time(sm_mme_k)

    tas_mme, sm_mme = xr.align(tas_mme, sm_mme, join="inner")
    if tas_mme.sizes.get("time", 0) == 0:
        raise RuntimeError(f"[{scenario}] MME after back-convert has zero time.")

    log(f"[{scenario}] MME built from {len(used_models)} models; shape={tuple(tas_mme.shape)}")
    return tas_mme, sm_mme, used_models


# =========================
# Extract events on one MME
# =========================
def extract_events_on_mme(scenario: str, tas: xr.DataArray, sm: xr.DataArray, used_models, fail_log: Path):
    label_name = "MME"
    model_out_root = OUT_ROOT / scenario / label_name
    events_root = model_out_root / "events_cc3d"
    safe_mkdir(events_root)

    pd.DataFrame({"model_used_in_mme": used_models}).to_csv(
        model_out_root / "mme_models_used.csv", index=False, encoding="utf-8"
    )

    try:
        tas_jja_all = select_jja(tas)
        sm_jja_all = select_jja(sm)
        if tas_jja_all.time.size == 0:
            log(f"[{scenario}] no JJA data in MME.")
            return

        # pre-pack numpy arrays once (tas/sm only); thresholds per-year computed on the fly
        tvals_all = tas_jja_all["time"].values
        years_all = pd.DatetimeIndex(tvals_all).year.values.astype(np.int32)
        doys_all = pd.DatetimeIndex(tvals_all).dayofyear.values.astype(np.int16)
        months_all = pd.DatetimeIndex(tvals_all).month.values.astype(np.int8)
        dates_str_all = pd.DatetimeIndex(tvals_all).strftime("%Y-%m-%d").values.astype(object)

        tas_np_all = tas_jja_all.values.astype(np.float32)
        sm_np_all = sm_jja_all.values.astype(np.float32)

        lat1d = np.asarray(tas_jja_all["lat"].values, dtype=float)
        lon1d = np.asarray(tas_jja_all["lon"].values, dtype=float)
        nx, ny = lon1d.size, lat1d.size
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)

        gi_fp = model_out_root / "grid_index.csv"
        if not gi_fp.exists():
            iy0, ix0 = np.indices((ny, nx))
            grid_id = iy0.ravel().astype(np.int64) * np.int64(nx) + ix0.ravel().astype(np.int64)
            gi = pd.DataFrame({
                "grid_id": grid_id,
                "ix": ix0.ravel().astype(int),
                "iy": iy0.ravel().astype(int),
                "lon": np.round(lon2d.ravel(), ROUND_COORD),
                "lat": np.round(lat2d.ravel(), ROUND_COORD),
                "lon_r": np.round(lon2d.ravel(), ROUND_COORD),
                "lat_r": np.round(lat2d.ravel(), ROUND_COORD),
            })
            gi.to_csv(gi_fp, index=False, encoding="utf-8")

        # yearly cc3d
        for y in range(START_YEAR, END_YEAR + 1):
            try:
                idx = np.where(years_all == y)[0]
                if len(idx) == 0:
                    continue

                # ---- rolling thresholds for this year y ----
                T90_t = compute_T90_for_year_from_base(tas_jja_all, y, DOY_WINDOW)
                if T90_t is None or T90_t.time.size == 0:
                    if ROLLING_EDGE_MODE == "skip":
                        continue
                    else:
                        raise RuntimeError(f"T90_t empty for year {y}")

                # align T90_t time with this year's tas subset (JJA days)
                tas_y = tas_jja_all.isel(time=idx)
                sm_y = sm_jja_all.isel(time=idx)
                T90_y = T90_t  # already on year y time axis in JJA

                tas_y, sm_y, T90_y = xr.align(tas_y, sm_y, T90_y, join="inner")
                if tas_y.time.size == 0:
                    continue

                # ---- heat3 definition unchanged ----
                heat_raw = (tas_y > T90_y)
                heat3 = heat3_from_heatraw(heat_raw)

                # NOTE: you previously also computed dry-related masks for DHW; keep structure,
                # but since this request only asked rolling T90, we leave SM thresholds as-is
                # (not used when EVENT_MASK_MODE="heat3").
                # If you later want rolling SM10 too, we can extend similarly.

                mode = EVENT_MASK_MODE.strip()
                if mode == "heat3":
                    event_mask = heat3
                else:
                    # keep placeholders consistent with your old structure
                    raise ValueError(
                        f"EVENT_MASK_MODE={EVENT_MASK_MODE} requires SM rolling too. "
                        f"Current script only rolls T90. Use heat3 or ask to extend SM rolling."
                    )

                event_mask_np = event_mask.values.astype(np.uint8)

                A = event_mask_np  # (time, lat, lon) for this year
                if int(A.sum()) == 0:
                    continue

                labels = cc3d.connected_components(A, connectivity=CONNECTIVITY)
                if int(labels.max()) == 0:
                    continue

                tz, iy, ix = np.where(A == 1)
                cc_lab = labels[tz, iy, ix].astype(np.int32)
                keep = cc_lab > 0
                tz, iy, ix, cc_lab = tz[keep], iy[keep], ix[keep], cc_lab[keep]
                if len(cc_lab) == 0:
                    continue

                # map local tz to global time index for saving date/year/doy
                # We need the global indices of the selected year subset after align:
                # use the aligned time coordinate to find positions in the "all" arrays.
                # simplest robust: build a map from date string to global index for this year.
                dates_y = pd.DatetimeIndex(tas_y["time"].values).strftime("%Y-%m-%d").values.astype(object)
                # global positions for this year in the "all" arrays
                # idx_all is the original indices, but align may have dropped some; so remap by date.
                # build dict date->global_index
                date_to_glb = {dates_str_all[i]: i for i in idx}
                i_glb = np.array([date_to_glb[dates_y[t]] for t in tz], dtype=np.int64)

                doy_ev = doys_all[i_glb]
                tdf = pd.DataFrame({"cc_label": cc_lab, "doy": doy_ev})
                dur = tdf.groupby("cc_label")["doy"].nunique()
                keep_labels = dur.index[dur.values >= MIN_DURATION_DAYS].to_numpy(dtype=np.int32)
                if len(keep_labels) == 0:
                    continue

                kset = set(keep_labels.tolist())
                m = np.array([lab in kset for lab in cc_lab], dtype=bool)
                tz, iy, ix, cc_lab, i_glb = tz[m], iy[m], ix[m], cc_lab[m], i_glb[m]
                if len(cc_lab) == 0:
                    continue

                grid_id = iy.astype(np.int64) * np.int64(nx) + ix.astype(np.int64)

                df = pd.DataFrame({
                    "date": dates_str_all[i_glb],
                    "year": years_all[i_glb].astype(int),
                    "month": months_all[i_glb].astype(int),
                    "doy": doys_all[i_glb].astype(int),

                    "grid_id": grid_id,
                    "ix": ix.astype(int),
                    "iy": iy.astype(int),

                    "lon": np.round(lon2d[iy, ix].astype(float), ROUND_COORD),
                    "lat": np.round(lat2d[iy, ix].astype(float), ROUND_COORD),
                    "longitude": np.round(lon2d[iy, ix].astype(float), ROUND_COORD),
                    "latitude": np.round(lat2d[iy, ix].astype(float), ROUND_COORD),

                    "temp_air": tas_np_all[i_glb, iy, ix].astype(np.float32),
                    "soil_moist": sm_np_all[i_glb, iy, ix].astype(np.float32),

                    # store rolling T90 used for this day/grid
                    "T90": np.asarray(T90_y.values, dtype=np.float32)[tz, iy, ix],

                    # keep original fields (placeholders) for compatibility
                    "heat_raw": (tas_np_all[i_glb, iy, ix] > np.asarray(T90_y.values, dtype=np.float32)[tz, iy, ix]).astype(np.int8),
                    "heat3": np.ones_like(cc_lab, dtype=np.int8),  # will be 1 because A==1 positions
                    "cc_label": cc_lab.astype(int),
                })

                stat = df.groupby("cc_label").agg(
                    start_doy=("doy", "min"),
                    end_doy=("doy", "max"),
                    duration_days=("doy", "nunique"),
                    n_records=("doy", "size"),
                    n_grids=("grid_id", "nunique"),
                    lon_min=("lon", "min"),
                    lon_max=("lon", "max"),
                    lat_min=("lat", "min"),
                    lat_max=("lat", "max"),
                ).reset_index()

                stat = stat.sort_values(["start_doy", "end_doy", "n_records"], ascending=[True, True, False]).reset_index(drop=True)
                stat["event_id"] = np.arange(1, len(stat) + 1, dtype=int)

                lab2eid = dict(zip(stat["cc_label"].astype(int), stat["event_id"].astype(int)))
                df["event_id"] = df["cc_label"].map(lab2eid).astype(int)

                tmpd = df[["event_id", "doy", "date"]].drop_duplicates(["event_id", "doy"])
                start_date = tmpd.sort_values(["event_id", "doy"]).groupby("event_id")["date"].first().rename("start_date")
                end_date = tmpd.sort_values(["event_id", "doy"]).groupby("event_id")["date"].last().rename("end_date")
                stat = stat.merge(start_date, on="event_id", how="left").merge(end_date, on="event_id", how="left")

                ydir = events_root / f"{y}"
                safe_mkdir(ydir)
                for eid, g in df.groupby("event_id", sort=True):
                    out_fp = ydir / f"event_{y}_{int(eid):05d}.csv"
                    g.sort_values(["doy", "grid_id"], kind="mergesort").to_csv(out_fp, index=False, encoding="utf-8")
                stat.to_csv(ydir / f"events_{y}_summary.csv", index=False, encoding="utf-8")

                log(f"[{scenario}/MME/{y}] events={len(stat)} (mode={EVENT_MASK_MODE}) | rolling={ROLLING_YEARS}yr")

            except Exception as ey:
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(f"{scenario}\tMME\tYEAR_{y}\t{repr(ey)}\n")
                log(f"[YEAR FAIL] {scenario}/MME/{y}: {repr(ey)}")

        log(f"[DONE] {scenario}/MME extraction done.")

    except Exception as e:
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(f"{scenario}\tMME\tPROCESS\t{repr(e)}\n")
            f.write(traceback.format_exc() + "\n")
        log(f"[PROCESS FAIL] {scenario}/MME: {repr(e)}")


# =========================
# Main
# =========================
def main():
    safe_mkdir(OUT_ROOT)
    fail_log = OUT_ROOT / "failures_cc3d_states6_mme_rollingT90.txt"
    with open(fail_log, "w", encoding="utf-8") as f:
        f.write("scenario\tmodel_or_mme\tstage\terror\n")

    for scen, in_dir in SCENARIO_DIRS.items():
        if not in_dir.exists():
            log(f"[WARN] missing scenario dir: {in_dir}")
            continue

        log(f"\n######## {scen}: build MME then extract CC3D (rolling T90) ########")
        try:
            tas_mme, sm_mme, used_models = build_scenario_mme(scen, in_dir, fail_log)
            if tas_mme is None or sm_mme is None or len(used_models) == 0:
                log(f"[WARN] {scen}: MME build failed or no valid models.")
                continue

            log(f"[{scen}] used models ({len(used_models)}): {', '.join(used_models)}")
            extract_events_on_mme(scen, tas_mme, sm_mme, used_models, fail_log)

        except Exception as e:
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{scen}\tMME\tMAIN_SCENARIO\t{repr(e)}\n")
                f.write(traceback.format_exc() + "\n")
            log(f"[SCENARIO FAIL] {scen}: {repr(e)}")

    print("\nALL DONE.")
    print("Output root:", OUT_ROOT)
    print("Failure log:", fail_log)


if __name__ == "__main__":
    main()