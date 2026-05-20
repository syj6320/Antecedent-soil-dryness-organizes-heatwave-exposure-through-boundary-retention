# -*- coding: utf-8 -*-
"""
Unified STATES6 + CC3D (26-neighborhood) event extraction for CMIP6 tas/mrsos
===============================================================================

Definition aligned with your STATES6 pipeline:
- Baseline: 1981-2010
- Season: JJA (6,7,8)
- Thresholds: DOY ± 5-day window quantiles
- heat_raw = tas > T90
- heat3 = all days belonging to runs with length >= 3
- SM bins by SM10/30/50/70/90 -> S1..S6
- Optional event mask:
    * heat3
    * DHW_sync3 = heat3 & (SM<SM10)
    * DHW_predry3 = heat3 & (dry_lag1)

CC3D extraction:
- A[t, y, x] from selected event mask
- connected-components-3d connectivity=26
- filter by unique-day duration >= MIN_DURATION_DAYS
- output per-event CSV + yearly summary

Input dirs:
- ssp126, ssp245, ssp585 model merged nc files:
    <MODEL>_tas_1950_2100_merged.nc
    <MODEL>_mrsos_1950_2100_merged.nc

Outputs:
OUT_ROOT/
  ssp126/<MODEL>/events_cc3d/<YYYY>/event_YYYY_00001.csv ...
  ssp126/<MODEL>/events_cc3d/<YYYY>/events_YYYY_summary.csv
  ... (ssp245, ssp585 similarly)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
import cc3d
import traceback
import math


# =========================
# USER SETTINGS
# =========================
SCENARIO_DIRS = {
    "ssp585": Path(r"E:\第二篇的修改20251229开始修改\ssp585\REGRID_100km\每模型合并后"),
}

OUT_ROOT = Path(r"E:\第二篇的修改20251229开始修改\CC3D_STATES6_UNIFIED_outputs")

# analysis period in merged files
START_YEAR, END_YEAR = 1950, 2100

# STATES6-consistent settings
BASELINE_START, BASELINE_END = 1981, 2010
DOY_WINDOW = 5
CONSEC_DAYS = 3
SEASON_MONTHS = (6, 7, 8)

# event extraction mode
# "heat3" / "DHW_sync3" / "DHW_predry3"
EVENT_MASK_MODE = "heat3"

# CC3D settings
CONNECTIVITY = 26
MIN_DURATION_DAYS = 3

# io/var
ROUND_COORD = 4
TAS_CANDS = ["tas", "tasmean", "t2m", "temperature_2m"]
SM_CANDS = ["mrsos", "mrso", "soil_moist", "sm", "volumetric_soil_water_layer_1"]

VERBOSE = True


# =========================
# General helpers
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
    # fallback: first 3D variable with time/lat/lon
    for v in ds.data_vars:
        dims = set(ds[v].dims)
        if "time" in dims and "lat" in dims and "lon" in dims:
            return v
    # last fallback
    return list(ds.data_vars)[0]

def select_jja(da: xr.DataArray) -> xr.DataArray:
    return da.sel(time=da["time"].dt.month.isin(SEASON_MONTHS))

def model_name_from_filename(fp: Path):
    # <MODEL>_tas_1950_2100_merged.nc
    m = re.match(r"(.+?)_(tas|mrsos)_\d{4}_\d{4}_merged\.nc$", fp.name)
    if m:
        return m.group(1)
    # fallback robust split
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
# STATES6 core definitions
# =========================
def heat3_from_heatraw(heat_raw: xr.DataArray) -> xr.DataArray:
    """
    heat3 = 1 for all days belonging to any run of heat_raw with length >= CONSEC_DAYS.
    Groupby year to prevent crossing year boundary.
    """
    def _one_year(x):
        end = (x.rolling(time=CONSEC_DAYS, min_periods=CONSEC_DAYS).sum() >= CONSEC_DAYS)
        end = end.fillna(False).astype(bool)
        # for CONSEC_DAYS=3: include terminal day and two previous days
        shift1 = end.shift(time=1, fill_value=False)
        shift2 = end.shift(time=2, fill_value=False)
        return (end | shift1 | shift2)

    return heat_raw.groupby("time.year").map(_one_year)

def compute_thresholds_doywindow(tas_base: xr.DataArray, sm_base: xr.DataArray, doy_window: int):
    """
    Return:
      T90_by_doy[doy,lat,lon]
      SMp_by_doy[doy,p,lat,lon], p in {10,30,50,70,90}
    """
    doy = tas_base["time"].dt.dayofyear
    doys = np.unique(doy.values)
    doys = np.array(sorted([int(d) for d in doys]))
    doy_set = set(doys.tolist())

    p_list = [10, 30, 50, 70, 90]
    q_list = [p / 100.0 for p in p_list]

    T90_list = []
    SMp_list = []

    for d in doys:
        win = [dd for dd in range(int(d) - doy_window, int(d) + doy_window + 1) if dd in doy_set]
        mask = doy.isin(win)

        tas_sub = tas_base.where(mask, drop=True)
        sm_sub = sm_base.where(mask, drop=True)

        T90_d = tas_sub.quantile(0.9, dim="time", skipna=True)
        SMq_d = sm_sub.quantile(q_list, dim="time", skipna=True)
        SMq_d = SMq_d.assign_coords(quantile=np.array(p_list, dtype=int)).rename({"quantile": "p"})

        T90_list.append(T90_d.expand_dims(doy=[int(d)]))
        SMp_list.append(SMq_d.expand_dims(doy=[int(d)]))

    T90 = xr.concat(T90_list, dim="doy")
    SMp = xr.concat(SMp_list, dim="doy")
    return T90, SMp

def assign_sbin(sm_val, sm10, sm30, sm50, sm70, sm90):
    # strict STATES6 bins:
    # S1: SM < SM10
    # S2: SM10 <= SM < SM30
    # S3: SM30 <= SM < SM50
    # S4: SM50 <= SM < SM70
    # S5: SM70 <= SM <= SM90
    # S6: SM > SM90
    if np.isnan(sm_val) or np.isnan(sm10) or np.isnan(sm30) or np.isnan(sm50) or np.isnan(sm70) or np.isnan(sm90):
        return np.nan
    if sm_val < sm10:
        return 1
    if sm_val < sm30:
        return 2
    if sm_val < sm50:
        return 3
    if sm_val < sm70:
        return 4
    if sm_val <= sm90:
        return 5
    return 6

def sm_to_decile(sm_val, q10, q20, q30, q40, q50, q60, q70, q80, q90):
    # optional decile output (1 driest, 10 wettest)
    qs = [q10, q20, q30, q40, q50, q60, q70, q80, q90]
    if any(np.isnan(v) for v in [sm_val] + qs):
        return np.nan
    if sm_val < q10:
        return 1
    if sm_val < q20:
        return 2
    if sm_val < q30:
        return 3
    if sm_val < q40:
        return 4
    if sm_val < q50:
        return 5
    if sm_val < q60:
        return 6
    if sm_val < q70:
        return 7
    if sm_val < q80:
        return 8
    if sm_val < q90:
        return 9
    return 10


# =========================
# Core extraction for one model-scenario
# =========================
def process_one_model_scenario(scenario: str, model: str, tas_fp: Path, sm_fp: Path, fail_log: Path):
    log(f"\n=== [{scenario}] {model} ===")
    log(f"tas   : {tas_fp.name}")
    log(f"mrsos : {sm_fp.name}")

    try:
        ds_t = standardize_latlon(open_dataset_safe(tas_fp))
        ds_s = standardize_latlon(open_dataset_safe(sm_fp))
    except Exception as e:
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(f"{scenario}\t{model}\tOPEN\t{repr(e)}\n")
        log(f"[OPEN FAIL] {repr(e)}")
        return

    try:
        tas_var = find_var(ds_t, TAS_CANDS)
        sm_var = find_var(ds_s, SM_CANDS)

        tas = ds_t[tas_var]
        sm = ds_s[sm_var]

        # align same time/lat/lon
        tas, sm = xr.align(tas, sm, join="inner")
        tas = tas.transpose("time", "lat", "lon")
        sm = sm.transpose("time", "lat", "lon")

        # analysis period
        tas = tas.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
        sm = sm.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
        if tas.time.size == 0:
            log("[SKIP] no data in analysis period.")
            return

        # JJA only
        tas_jja = select_jja(tas)
        sm_jja = select_jja(sm)
        if tas_jja.time.size == 0:
            log("[SKIP] no JJA data.")
            return

        # baseline for thresholds
        tas_base = tas_jja.sel(time=slice(f"{BASELINE_START}-01-01", f"{BASELINE_END}-12-31"))
        sm_base = sm_jja.sel(time=slice(f"{BASELINE_START}-01-01", f"{BASELINE_END}-12-31"))
        if tas_base.time.size < 100:
            raise RuntimeError(
                f"baseline JJA too short for {scenario}/{model}: {tas_base.time.size} days"
            )

        log(f"[{scenario}/{model}] computing thresholds (baseline {BASELINE_START}-{BASELINE_END}, DOY±{DOY_WINDOW}) ...")
        T90_by_doy, SMp_by_doy = compute_thresholds_doywindow(tas_base, sm_base, DOY_WINDOW)

        # also build decile thresholds for output SM_decile
        p_dec = [10,20,30,40,50,60,70,80,90]
        q_dec = [p/100 for p in p_dec]
        # compute decile thresholds by doy using same window logic
        doy_base = tas_base["time"].dt.dayofyear
        doys = np.unique(doy_base.values)
        doys = np.array(sorted([int(d) for d in doys]))
        doy_set = set(doys.tolist())
        SMdec_list = []
        for d in doys:
            win = [dd for dd in range(int(d)-DOY_WINDOW, int(d)+DOY_WINDOW+1) if dd in doy_set]
            m = doy_base.isin(win)
            sm_sub = sm_base.where(m, drop=True)
            qv = sm_sub.quantile(q_dec, dim="time", skipna=True)
            qv = qv.assign_coords(quantile=np.array(p_dec, dtype=int)).rename({"quantile":"p"})
            SMdec_list.append(qv.expand_dims(doy=[int(d)]))
        SMdec_by_doy = xr.concat(SMdec_list, dim="doy")

        # map thresholds to all JJA time
        doy_all = tas_jja["time"].dt.dayofyear.astype(int)
        T90_t = T90_by_doy.sel(doy=doy_all)
        SM10_t = SMp_by_doy.sel(doy=doy_all, p=10)
        SM30_t = SMp_by_doy.sel(doy=doy_all, p=30)
        SM50_t = SMp_by_doy.sel(doy=doy_all, p=50)
        SM70_t = SMp_by_doy.sel(doy=doy_all, p=70)
        SM90_t = SMp_by_doy.sel(doy=doy_all, p=90)

        SM10d = SMdec_by_doy.sel(doy=doy_all, p=10)
        SM20d = SMdec_by_doy.sel(doy=doy_all, p=20)
        SM30d = SMdec_by_doy.sel(doy=doy_all, p=30)
        SM40d = SMdec_by_doy.sel(doy=doy_all, p=40)
        SM50d = SMdec_by_doy.sel(doy=doy_all, p=50)
        SM60d = SMdec_by_doy.sel(doy=doy_all, p=60)
        SM70d = SMdec_by_doy.sel(doy=doy_all, p=70)
        SM80d = SMdec_by_doy.sel(doy=doy_all, p=80)
        SM90d = SMdec_by_doy.sel(doy=doy_all, p=90)

        # flags
        heat_raw = (tas_jja > T90_t)
        heat3 = heat3_from_heatraw(heat_raw)

        dry_raw = (sm_jja < SM10_t)
        dry_lag1 = dry_raw.shift(time=1, fill_value=False)

        DHW_sync3 = heat3 & dry_raw
        DHW_predry3 = heat3 & dry_lag1
        NDHW3 = heat3 & (~dry_raw)

        mode = EVENT_MASK_MODE.strip()
        if mode == "heat3":
            event_mask = heat3
        elif mode == "DHW_sync3":
            event_mask = DHW_sync3
        elif mode == "DHW_predry3":
            event_mask = DHW_predry3
        else:
            raise ValueError(f"Unknown EVENT_MASK_MODE={EVENT_MASK_MODE}")

        # pre-extract numpy arrays for speed
        tvals = tas_jja["time"].values
        years = np.array([t.year for t in tvals], dtype=np.int32)
        doys_all = np.array([t.timetuple().tm_yday for t in tvals], dtype=np.int16)
        months = np.array([t.month for t in tvals], dtype=np.int8)
        dates_str = np.array([t.strftime("%Y-%m-%d") for t in tvals], dtype=object)

        tas_np = tas_jja.values.astype(np.float32)
        sm_np = sm_jja.values.astype(np.float32)

        T90_np = T90_t.values.astype(np.float32)
        SM10_np = SM10_t.values.astype(np.float32)
        SM30_np = SM30_t.values.astype(np.float32)
        SM50_np = SM50_t.values.astype(np.float32)
        SM70_np = SM70_t.values.astype(np.float32)
        SM90_np = SM90_t.values.astype(np.float32)

        SM10d_np = SM10d.values.astype(np.float32)
        SM20d_np = SM20d.values.astype(np.float32)
        SM30d_np = SM30d.values.astype(np.float32)
        SM40d_np = SM40d.values.astype(np.float32)
        SM50d_np = SM50d.values.astype(np.float32)
        SM60d_np = SM60d.values.astype(np.float32)
        SM70d_np = SM70d.values.astype(np.float32)
        SM80d_np = SM80d.values.astype(np.float32)
        SM90d_np = SM90d.values.astype(np.float32)

        heat_raw_np = heat_raw.values.astype(np.int8)
        dry_raw_np = dry_raw.values.astype(np.int8)
        dry_lag1_np = dry_lag1.values.astype(np.int8)
        heat3_np = heat3.values.astype(np.int8)
        dhw_sync_np = DHW_sync3.values.astype(np.int8)
        ndhw_np = NDHW3.values.astype(np.int8)
        dhw_predry_np = DHW_predry3.values.astype(np.int8)
        event_mask_np = event_mask.values.astype(np.uint8)

        lat1d = np.asarray(tas_jja["lat"].values, dtype=float)
        lon1d = np.asarray(tas_jja["lon"].values, dtype=float)
        nx = lon1d.size
        ny = lat1d.size
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)

        # output roots
        model_out_root = OUT_ROOT / scenario / model
        events_root = model_out_root / "events_cc3d"
        safe_mkdir(events_root)

        # write grid index once
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
            safe_mkdir(gi_fp.parent)
            gi.to_csv(gi_fp, index=False, encoding="utf-8")

        # per-year CC3D extraction
        for y in range(START_YEAR, END_YEAR + 1):
            try:
                idx = np.where(years == y)[0]
                if len(idx) == 0:
                    continue

                A = event_mask_np[idx, :, :]
                if int(A.sum()) == 0:
                    continue

                labels = cc3d.connected_components(A, connectivity=CONNECTIVITY)
                nlab = int(labels.max())
                if nlab == 0:
                    continue

                tz, iy, ix = np.where(A == 1)
                cc_lab = labels[tz, iy, ix].astype(np.int32)
                keep = cc_lab > 0
                tz, iy, ix, cc_lab = tz[keep], iy[keep], ix[keep], cc_lab[keep]
                if len(cc_lab) == 0:
                    continue

                # duration filter on unique doy
                doy_ev = doys_all[idx][tz]
                tdf = pd.DataFrame({"cc_label": cc_lab, "doy": doy_ev})
                dur = tdf.groupby("cc_label")["doy"].nunique()
                keep_labels = dur.index[dur.values >= MIN_DURATION_DAYS].to_numpy(dtype=np.int32)
                if len(keep_labels) == 0:
                    continue

                kset = set(keep_labels.tolist())
                m = np.array([lab in kset for lab in cc_lab], dtype=bool)
                tz, iy, ix, cc_lab = tz[m], iy[m], ix[m], cc_lab[m]

                if len(cc_lab) == 0:
                    continue

                # local arrays
                i_glb = idx[tz]

                sm_v = sm_np[i_glb, iy, ix]
                sm10_v = SM10_np[i_glb, iy, ix]
                sm30_v = SM30_np[i_glb, iy, ix]
                sm50_v = SM50_np[i_glb, iy, ix]
                sm70_v = SM70_np[i_glb, iy, ix]
                sm90_v = SM90_np[i_glb, iy, ix]

                # SM_decile and S_bin
                sm_dec = np.array([
                    sm_to_decile(
                        sm_v[k],
                        SM10d_np[i_glb[k], iy[k], ix[k]],
                        SM20d_np[i_glb[k], iy[k], ix[k]],
                        SM30d_np[i_glb[k], iy[k], ix[k]],
                        SM40d_np[i_glb[k], iy[k], ix[k]],
                        SM50d_np[i_glb[k], iy[k], ix[k]],
                        SM60d_np[i_glb[k], iy[k], ix[k]],
                        SM70d_np[i_glb[k], iy[k], ix[k]],
                        SM80d_np[i_glb[k], iy[k], ix[k]],
                        SM90d_np[i_glb[k], iy[k], ix[k]],
                    ) for k in range(len(i_glb))
                ], dtype=float)

                s_bin = np.array([
                    assign_sbin(sm_v[k], sm10_v[k], sm30_v[k], sm50_v[k], sm70_v[k], sm90_v[k])
                    for k in range(len(sm_v))
                ], dtype=float)

                # grid_id
                grid_id = iy.astype(np.int64) * np.int64(nx) + ix.astype(np.int64)

                # build dataframe
                df = pd.DataFrame({
                    "date": dates_str[i_glb],
                    "year": y,
                    "month": months[i_glb].astype(int),
                    "doy": doys_all[i_glb].astype(int),

                    "grid_id": grid_id,
                    "ix": ix.astype(int),
                    "iy": iy.astype(int),

                    "lon": np.round(lon2d[iy, ix].astype(float), ROUND_COORD),
                    "lat": np.round(lat2d[iy, ix].astype(float), ROUND_COORD),
                    "longitude": np.round(lon2d[iy, ix].astype(float), ROUND_COORD),
                    "latitude": np.round(lat2d[iy, ix].astype(float), ROUND_COORD),

                    "temp_air": tas_np[i_glb, iy, ix].astype(np.float32),
                    "soil_moist": sm_v.astype(np.float32),

                    "T90": T90_np[i_glb, iy, ix].astype(np.float32),
                    "SM10": sm10_v.astype(np.float32),
                    "SM30": sm30_v.astype(np.float32),
                    "SM50": sm50_v.astype(np.float32),
                    "SM70": sm70_v.astype(np.float32),
                    "SM90": sm90_v.astype(np.float32),

                    # decile thresholds
                    "SM20": SM20d_np[i_glb, iy, ix].astype(np.float32),
                    "SM40": SM40d_np[i_glb, iy, ix].astype(np.float32),
                    "SM60": SM60d_np[i_glb, iy, ix].astype(np.float32),
                    "SM80": SM80d_np[i_glb, iy, ix].astype(np.float32),

                    # compatibility fields
                    "T90_old": T90_np[i_glb, iy, ix].astype(np.float32),
                    "SM10_old": sm10_v.astype(np.float32),
                    "SM90_old": sm90_v.astype(np.float32),

                    "heat_raw": heat_raw_np[i_glb, iy, ix].astype(np.int8),
                    "dry_raw": dry_raw_np[i_glb, iy, ix].astype(np.int8),
                    "wet90_raw": (sm_np[i_glb, iy, ix] > SM90_np[i_glb, iy, ix]).astype(np.int8),
                    "dry_lag1": dry_lag1_np[i_glb, iy, ix].astype(np.int8),

                    "heat3": heat3_np[i_glb, iy, ix].astype(np.int8),
                    "DHW_sync3": dhw_sync_np[i_glb, iy, ix].astype(np.int8),
                    "NDHW3": ndhw_np[i_glb, iy, ix].astype(np.int8),
                    "DHW_predry3": dhw_predry_np[i_glb, iy, ix].astype(np.int8),
                    "WHW90_3": ((heat3_np[i_glb, iy, ix] == 1) & (sm_np[i_glb, iy, ix] > SM90_np[i_glb, iy, ix])).astype(np.int8),

                    "SM_decile": sm_dec,
                    "S_bin": s_bin,

                    "cc_label": cc_lab.astype(int),
                })

                # drop invalid bins if any
                df = df.dropna(subset=["S_bin"]).copy()
                if df.empty:
                    continue
                df["S_bin"] = df["S_bin"].astype(int)
                df["SM_decile"] = df["SM_decile"].round().astype("Int64")

                # stable event_id by temporal order
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

                stat = stat.sort_values(
                    ["start_doy", "end_doy", "n_records"],
                    ascending=[True, True, False]
                ).reset_index(drop=True)
                stat["event_id"] = np.arange(1, len(stat) + 1, dtype=int)

                lab2eid = dict(zip(stat["cc_label"].astype(int), stat["event_id"].astype(int)))
                df["event_id"] = df["cc_label"].map(lab2eid).astype(int)

                # start/end date
                tmpd = df[["event_id", "doy", "date"]].drop_duplicates(["event_id", "doy"])
                start_date = tmpd.sort_values(["event_id", "doy"]).groupby("event_id")["date"].first().rename("start_date")
                end_date = tmpd.sort_values(["event_id", "doy"]).groupby("event_id")["date"].last().rename("end_date")
                stat = stat.merge(start_date, on="event_id", how="left")
                stat = stat.merge(end_date, on="event_id", how="left")

                # output
                ydir = events_root / f"{y}"
                safe_mkdir(ydir)

                for eid, g in df.groupby("event_id", sort=True):
                    out_fp = ydir / f"event_{y}_{int(eid):05d}.csv"
                    g.sort_values(["doy", "grid_id"], kind="mergesort").to_csv(out_fp, index=False, encoding="utf-8")

                stat_fp = ydir / f"events_{y}_summary.csv"
                stat.to_csv(stat_fp, index=False, encoding="utf-8")

                log(f"[{scenario}/{model}/{y}] events={len(stat)} (mode={EVENT_MASK_MODE})")

            except Exception as ey:
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(f"{scenario}\t{model}\tYEAR_{y}\t{repr(ey)}\n")
                log(f"[YEAR FAIL] {scenario}/{model}/{y}: {repr(ey)}")

        log(f"[DONE] {scenario}/{model}")

    except Exception as e:
        with open(fail_log, "a", encoding="utf-8") as f:
            f.write(f"{scenario}\t{model}\tPROCESS\t{repr(e)}\n")
            f.write(traceback.format_exc() + "\n")
        log(f"[PROCESS FAIL] {scenario}/{model}: {repr(e)}")
    finally:
        try:
            ds_t.close()
        except Exception:
            pass
        try:
            ds_s.close()
        except Exception:
            pass


# =========================
# Main
# =========================
def main():
    safe_mkdir(OUT_ROOT)
    fail_log = OUT_ROOT / "failures_cc3d_states6_unified.txt"
    with open(fail_log, "w", encoding="utf-8") as f:
        f.write("scenario\tmodel\tstage\terror\n")

    for scen, in_dir in SCENARIO_DIRS.items():
        if not in_dir.exists():
            log(f"[WARN] missing scenario dir: {in_dir}")
            continue

        pairs = discover_model_pairs(in_dir)
        if len(pairs) == 0:
            log(f"[WARN] no tas/mrsos pairs in: {in_dir}")
            continue

        log(f"\n######## Scenario {scen}: {len(pairs)} model pairs ########")
        for model, tas_fp, sm_fp in pairs:
            process_one_model_scenario(scen, model, tas_fp, sm_fp, fail_log)

    print("\nALL DONE.")
    print("Output root:", OUT_ROOT)
    print("Failure log:", fail_log)


if __name__ == "__main__":
    main()
