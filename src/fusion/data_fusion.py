#!/usr/bin/env python3
"""
Phase 4.1 - Fusion: Station-Time Master Table from All Preprocessed Sources
----------------------------------------------------------------------------
This is the sole Phase 4 script and the final data preparation step
in the pipeline. It reads all five preprocessed source directories
(SuperMAG, OMNI, GOES XRS, GOES MAG, Swarm CHAOS), joins them into
a unified station-time master table using leakage-safe backward as-of
joins, appends per-source freshness and missingness diagnostics, and
writes one Parquet file per month to
data/fused/station_time_master/. All six Phase 3 preprocessing
scripts must complete before this script can run.

Purpose
-------
The fusion stage assembles the station-time master table that serves
as the primary modeling input. Each row in the output corresponds to
one SuperMAG ground station at one UTC minute. The SuperMAG base
table provides the target-side magnetic field perturbation features.
All other sources (solar wind, X-ray flux, geostationary magnetic
field, low-Earth-orbit CHAOS residuals) are joined to it as
explanatory features.

The join design is explicitly leakage-safe: all non-SuperMAG sources
are attached using pd.merge_asof with direction="backward". For each
SuperMAG station-minute row, the join finds the most recent
observation in the source that occurred at or before the row's
timestamp and within the configured lookback tolerance. No future
source data is ever attached to a given row. This ensures that the
fused table can be used for time-series forecasting without leaking
future information.

Three GIC-relevant features are derived from the SuperMAG base before
any join:
- horizontal_perturbation_nt: np.hypot(dbn_nt, dbe_nt), the
  horizontal magnetic field perturbation magnitude per station-minute
- abs_dbdt_nt_per_min: per-station time derivative of
  horizontal_perturbation_nt, computed as the absolute 1-minute diff
- gic_proxy: equal to abs_dbdt_nt_per_min (float32), a proxy for
  geomagnetically induced current magnitude

Multi-satellite sources (GOES XRS, GOES MAG, Swarm) are reduced to
a single per-timestamp row before the join by groupby aggregation:
flux and field components are averaged across satellites, validity
flags are reduced by max, satellite counts are recorded via nunique.
Swarm n_valid_1min_points is summed across satellites. All aggregated
columns carry a source prefix (xrs_, gmag_, swarm_).

A configurable lookback window controls how far backward each join is
allowed to search:
- --source-lookback-min (default 180): applied to OMNI and GOES joins
- --swarm-lookback-min  (default 180): applied to the Swarm join

To support backward matches near the start of a month, each source
is pre-loaded over a window of [month_start - max_lookback, month_end]
so that observations from the previous month are available for matches
near the month boundary.

After each join, two diagnostic columns are appended via
add_freshness_and_missing(): {source}_freshness_min (float32, seconds
between the SuperMAG row and the matched source row divided by 60)
and {source}_missing (boolean, True when no match was found within
tolerance). These allow downstream models to condition on data
availability and recency.

WORK FLOW
---------
Step 1: Parse CLI arguments - five source root directories, output
        root, optional --months list, optional date range, independent
        lookback tolerances for OMNI/GOES and Swarm, and --force flag.
Step 2: Discover all available YYYYMM values from the SuperMAG
        preprocessed root (or use --months if provided). Raise
        FileNotFoundError if none are found.
Step 3: For each YYYYMM, run fuse_one_month():
  Step 3a: Skip existing output unless --force is set.
  Step 3b: Load the SuperMAG base via load_supermag_month(): rglob
           all supermag_*_{YYYYMM}.parquet files, concatenate all
           stations, normalize timestamps to UTC, deduplicate on
           (station, timestamp) keeping last, and compute
           horizontal_perturbation_nt, abs_dbdt_nt_per_min,
           gic_proxy.
  Step 3c: Apply optional global date filter to the base table.
  Step 3d: Compute the pre-load window: [month_start - max_lookback,
           month_end] where max_lookback = max(source_lookback_min,
           swarm_lookback_min).
  Step 3e: Load each non-SuperMAG source over the pre-load window
           via its load_*_window() function. Multi-satellite sources
           are aggregated to a single per-timestamp row. Each source
           timestamp column is renamed to {source}_timestamp to
           prevent collision during merge_asof.
  Step 3f: Apply backward_asof_join() sequentially for OMNI, GOES
           XRS, GOES MAG, and Swarm. After each join, append
           {source}_freshness_min and {source}_missing columns via
           add_freshness_and_missing().
  Step 3g: Append year (Int16), month (Int8), and yyyymm (string)
           partition columns.
  Step 3h: Sort by (station, timestamp), deduplicate on
           (station, timestamp) keeping last, and write to
           data/fused/station_time_master/{YYYY}/
           station_time_master_{YYYYMM}.parquet.

INPUT DATA
----------
- data/preprocessed/supermag/**/supermag_*_{YYYYMM}.parquet
  Phase 3.6 output. Base table; drives row count and month coverage.
- data/preprocessed/omni/**/{YYYYMM}*.parquet
  Phase 3.3 output. 1-minute solar wind and IMF observations.
- data/preprocessed/goes_xrs/**/{YYYYMM}*.parquet
  Phase 3.4 output. Per-satellite GOES XRS flux at 1-minute cadence.
- data/preprocessed/goes_mag/**/{YYYYMM}*.parquet
  Phase 3.5 output. Per-satellite GOES MAG vectors at 1-minute cadence.
- data/preprocessed/swarm_chaos/**/swarm_*_{YYYYMM}_1min_chaos.parquet
  Phase 3.2 output. Per-satellite Swarm CHAOS residuals at 1-minute
  cadence.
- CLI optional: --supermag-root        (default: data/preprocessed/supermag)
               --omni-root            (default: data/preprocessed/omni)
               --goes-xrs-root        (default: data/preprocessed/goes_xrs)
               --goes-mag-root        (default: data/preprocessed/goes_mag)
               --swarm-root           (default: data/preprocessed/swarm_chaos)
               --output-root          (default: data/fused/station_time_master)
               --months               (default: all discovered; YYYYMM list)
               --start-date           (default: no lower bound; YYYY-MM-DD)
               --end-date             (default: no upper bound; YYYY-MM-DD)
               --source-lookback-min  (default: 180; for OMNI and GOES)
               --swarm-lookback-min   (default: 180; for Swarm)
               --force                (overwrite existing output files)

OUTPUT DATA
-----------
- data/fused/station_time_master/{YYYY}/station_time_master_{YYYYMM}.parquet
  One Parquet file per month. Each row is one SuperMAG station at
  one UTC minute. Schema includes:
  - All SuperMAG columns from Phase 3.6 plus horizontal_perturbation_nt,
    abs_dbdt_nt_per_min, gic_proxy
  - OMNI columns (bt_nT, bx_gse_gsm_nT, by_gsm_nT, bz_gsm_nT,
    speed_km_s, vx/vy/vz_gse_km_s, proton_density_n_cc,
    flow_pressure_nPa, percent_interpolation, timeshift_sec,
    imf_valid, plasma_valid, velocity_vector_valid)
    plus omni_timestamp, omni_freshness_min, omni_missing
  - GOES XRS aggregated columns (xrs_sat_count, xrs_xrsa_flux,
    xrs_xrsb_flux, xrs_flux_ratio, xrs_any_valid,
    xrs_has_electron_correction)
    plus goes_xrs_timestamp, goes_xrs_freshness_min, goes_xrs_missing
  - GOES MAG aggregated columns (gmag_sat_count, gmag_b_gsm_{x,y,z},
    gmag_b_vdh_{x,y,z}, gmag_any_valid)
    plus goes_mag_timestamp, goes_mag_freshness_min, goes_mag_missing
  - Swarm aggregated columns (swarm_sat_count,
    swarm_residual_total_b_{north,east,center}_nT,
    swarm_residual_internal_b_{north,east,center}_nT,
    swarm_n_valid_1min_points)
    plus swarm_timestamp, swarm_freshness_min, swarm_missing
  - Partition columns: year (Int16), month (Int8), yyyymm (string)

NEXT PIPELINE PHASE
-------------------
- Phase 4.1 is the final data preparation step. Its output is the
  primary input to all downstream modeling, feature engineering,
  and analysis scripts.
- All six Phase 3 scripts must complete before this script can run.
- The critical intra-pipeline ordering constraint is: Phase 3.1
  (swarm_1min.py) must complete before Phase 3.2 (swarm_chaos.py),
  which must complete before this script can use swarm_chaos output.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SUPERMAG_ROOT = Path("data/preprocessed/supermag")
DEFAULT_OMNI_ROOT = Path("data/preprocessed/omni")
DEFAULT_GOES_XRS_ROOT = Path("data/preprocessed/goes_xrs")
DEFAULT_GOES_MAG_ROOT = Path("data/preprocessed/goes_mag")
DEFAULT_SWARM_ROOT = Path("data/preprocessed/swarm_chaos")
DEFAULT_OUTPUT_ROOT = Path("data/fused/station_time_master")

SUPERMAG_FILE_RE = re.compile(r"supermag_[A-Z0-9]{3}_(\d{6})\.parquet$")
OMNI_FILE_RE = re.compile(r"omni_(\d{6})\.parquet$")
GOES_FILE_RE = re.compile(r".*_(\d{6})\.parquet$")
SWARM_FILE_RE = re.compile(r"swarm_[A-Z]_(\d{6})_1min_chaos\.parquet$")

OMNI_COLS = [
    "bt_nT",
    "bx_gse_gsm_nT",
    "by_gsm_nT",
    "bz_gsm_nT",
    "speed_km_s",
    "vx_gse_km_s",
    "vy_gse_km_s",
    "vz_gse_km_s",
    "proton_density_n_cc",
    "flow_pressure_nPa",
    "percent_interpolation",
    "timeshift_sec",
    "imf_valid",
    "plasma_valid",
    "velocity_vector_valid",
]

XRS_COLS = [
    "xrsa_flux",
    "xrsb_flux",
    "xray_flux_ratio",
    "xrsa_valid",
    "xrsb_valid",
    "has_electron_correction",
]

GOES_MAG_COLS = [
    "b_gsm_x",
    "b_gsm_y",
    "b_gsm_z",
    "b_vdh_x",
    "b_vdh_y",
    "b_vdh_z",
    "mag_valid",
]

SWARM_COLS = [
    "residual_total_b_north_nT",
    "residual_total_b_east_nT",
    "residual_total_b_center_nT",
    "residual_internal_b_north_nT",
    "residual_internal_b_east_nT",
    "residual_internal_b_center_nT",
    "n_valid_1min_points",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe station-time fusion tables using SuperMAG as base "
            "and backward/as-of joins for OMNI, GOES, and Swarm."
        )
    )
    parser.add_argument("--supermag-root", type=Path, default=DEFAULT_SUPERMAG_ROOT)
    parser.add_argument("--omni-root", type=Path, default=DEFAULT_OMNI_ROOT)
    parser.add_argument("--goes-xrs-root", type=Path, default=DEFAULT_GOES_XRS_ROOT)
    parser.add_argument("--goes-mag-root", type=Path, default=DEFAULT_GOES_MAG_ROOT)
    parser.add_argument("--swarm-root", type=Path, default=DEFAULT_SWARM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="Optional YYYYMM filters. If omitted, process all available SuperMAG months.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Optional inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="Optional inclusive end date YYYY-MM-DD")
    parser.add_argument(
        "--source-lookback-min",
        type=int,
        default=180,
        help="Max backward lookback (minutes) for OMNI and GOES as-of joins.",
    )
    parser.add_argument(
        "--swarm-lookback-min",
        type=int,
        default=180,
        help="Max backward lookback (minutes) for Swarm as-of joins.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing fused monthly outputs.")
    return parser.parse_args()


def extract_yyyymm(path: Path, pattern: re.Pattern[str]) -> str | None:
    match = pattern.match(path.name)
    if not match:
        return None
    return match.group(1)


def month_start_end_utc(yyyymm: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{yyyymm[:4]}-{yyyymm[4:6]}-01", tz="UTC")
    end = (start + pd.offsets.MonthBegin(1)) - pd.Timedelta(minutes=1)
    return start, end


def month_sequence_between(start: pd.Timestamp, end: pd.Timestamp) -> set[str]:
    out: set[str] = set()
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while cur <= last:
        out.add(cur.strftime("%Y%m"))
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def load_supermag_month(supermag_root: Path, yyyymm: str) -> pd.DataFrame:
    files = sorted(supermag_root.rglob(f"supermag_*_{yyyymm}.parquet"))
    if not files:
        raise FileNotFoundError(f"No SuperMAG files found for month {yyyymm} under {supermag_root}")

    dfs = [pd.read_parquet(path) for path in files]
    out = pd.concat(dfs, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "station"]).copy()
    out["station"] = out["station"].astype(str).str.upper()
    out = out.sort_values(["station", "timestamp"])
    out = out.drop_duplicates(subset=["station", "timestamp"], keep="last").reset_index(drop=True)

    h = np.hypot(pd.to_numeric(out["dbn_nt"], errors="coerce"), pd.to_numeric(out["dbe_nt"], errors="coerce"))
    out["horizontal_perturbation_nt"] = h.astype("float32")
    out["abs_dbdt_nt_per_min"] = out.groupby("station")["horizontal_perturbation_nt"].diff().abs().astype("float32")
    out["gic_proxy"] = out["abs_dbdt_nt_per_min"].astype("float32")
    return out


def select_files_for_window(
    root: Path,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    pattern: re.Pattern[str],
) -> list[Path]:
    months = month_sequence_between(window_start, window_end)
    out: list[Path] = []
    for path in sorted(root.rglob("*.parquet")):
        yyyymm = extract_yyyymm(path, pattern)
        if yyyymm and yyyymm in months:
            out.append(path)
    return out


def load_omni_window(root: Path, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    files = select_files_for_window(root, window_start, window_end, OMNI_FILE_RE)
    if not files:
        return pd.DataFrame(columns=["omni_timestamp"])

    dfs = []
    for path in files:
        df = pd.read_parquet(path)
        keep = ["timestamp"] + [c for c in OMNI_COLS if c in df.columns]
        df = df[keep].copy()
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out = out[(out["timestamp"] >= window_start) & (out["timestamp"] <= window_end)]
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    out = out.rename(columns={"timestamp": "omni_timestamp"}).reset_index(drop=True)
    return out


def load_goes_xrs_window(root: Path, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    files = select_files_for_window(root, window_start, window_end, GOES_FILE_RE)
    if not files:
        return pd.DataFrame(columns=["goes_xrs_timestamp"])

    dfs = []
    for path in files:
        df = pd.read_parquet(path)
        keep = ["timestamp", "sat_id"] + [c for c in XRS_COLS if c in df.columns]
        df = df[keep].copy()
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw = raw[(raw["timestamp"] >= window_start) & (raw["timestamp"] <= window_end)]
    raw = raw.sort_values(["timestamp", "sat_id"]).drop_duplicates(subset=["timestamp", "sat_id"], keep="last")

    grouped = raw.groupby("timestamp", as_index=False).agg(
        xrs_sat_count=("sat_id", "nunique"),
        xrs_xrsa_flux=("xrsa_flux", "mean"),
        xrs_xrsb_flux=("xrsb_flux", "mean"),
        xrs_flux_ratio=("xray_flux_ratio", "mean"),
        xrs_any_valid=("xrsa_valid", "max"),
        xrs_has_electron_correction=("has_electron_correction", "max"),
    )
    grouped = grouped.rename(columns={"timestamp": "goes_xrs_timestamp"}).sort_values("goes_xrs_timestamp")
    return grouped.reset_index(drop=True)


def load_goes_mag_window(root: Path, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    files = select_files_for_window(root, window_start, window_end, GOES_FILE_RE)
    if not files:
        return pd.DataFrame(columns=["goes_mag_timestamp"])

    dfs = []
    for path in files:
        df = pd.read_parquet(path)
        keep = ["timestamp", "sat_id"] + [c for c in GOES_MAG_COLS if c in df.columns]
        df = df[keep].copy()
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw = raw[(raw["timestamp"] >= window_start) & (raw["timestamp"] <= window_end)]
    raw = raw.sort_values(["timestamp", "sat_id"]).drop_duplicates(subset=["timestamp", "sat_id"], keep="last")

    grouped = raw.groupby("timestamp", as_index=False).agg(
        gmag_sat_count=("sat_id", "nunique"),
        gmag_b_gsm_x=("b_gsm_x", "mean"),
        gmag_b_gsm_y=("b_gsm_y", "mean"),
        gmag_b_gsm_z=("b_gsm_z", "mean"),
        gmag_b_vdh_x=("b_vdh_x", "mean"),
        gmag_b_vdh_y=("b_vdh_y", "mean"),
        gmag_b_vdh_z=("b_vdh_z", "mean"),
        gmag_any_valid=("mag_valid", "max"),
    )
    grouped = grouped.rename(columns={"timestamp": "goes_mag_timestamp"}).sort_values("goes_mag_timestamp")
    return grouped.reset_index(drop=True)


def load_swarm_window(root: Path, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    files = select_files_for_window(root, window_start, window_end, SWARM_FILE_RE)
    if not files:
        return pd.DataFrame(columns=["swarm_timestamp"])

    dfs = []
    for path in files:
        df = pd.read_parquet(path)
        keep = ["timestamp", "satellite"] + [c for c in SWARM_COLS if c in df.columns]
        df = df[keep].copy()
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw = raw[(raw["timestamp"] >= window_start) & (raw["timestamp"] <= window_end)]
    raw = raw.sort_values(["timestamp", "satellite"]).drop_duplicates(
        subset=["timestamp", "satellite"], keep="last"
    )

    grouped = raw.groupby("timestamp", as_index=False).agg(
        swarm_sat_count=("satellite", "nunique"),
        swarm_residual_total_b_north_nT=("residual_total_b_north_nT", "mean"),
        swarm_residual_total_b_east_nT=("residual_total_b_east_nT", "mean"),
        swarm_residual_total_b_center_nT=("residual_total_b_center_nT", "mean"),
        swarm_residual_internal_b_north_nT=("residual_internal_b_north_nT", "mean"),
        swarm_residual_internal_b_east_nT=("residual_internal_b_east_nT", "mean"),
        swarm_residual_internal_b_center_nT=("residual_internal_b_center_nT", "mean"),
        swarm_n_valid_1min_points=("n_valid_1min_points", "sum"),
    )
    grouped = grouped.rename(columns={"timestamp": "swarm_timestamp"}).sort_values("swarm_timestamp")
    return grouped.reset_index(drop=True)


def backward_asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_key: str,
    right_key: str,
    tolerance_min: int,
) -> pd.DataFrame:
    left_sorted = left.sort_values(left_key).reset_index(drop=True)
    right_sorted = right.sort_values(right_key).reset_index(drop=True)

    if right_sorted.empty:
        out = left_sorted.copy()
        out[right_key] = pd.NaT
        return out

    out = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=left_key,
        right_on=right_key,
        direction="backward",
        tolerance=pd.Timedelta(minutes=tolerance_min),
    )
    return out


def add_freshness_and_missing(df: pd.DataFrame, source_name: str, source_ts_col: str) -> pd.DataFrame:
    freshness_col = f"{source_name}_freshness_min"
    missing_col = f"{source_name}_missing"
    df[freshness_col] = (
        (df["timestamp"] - df[source_ts_col]).dt.total_seconds() / 60.0
    ).astype("float32")
    df[missing_col] = df[source_ts_col].isna().astype("boolean")
    return df


def discover_supermag_months(supermag_root: Path) -> list[str]:
    months = set()
    for path in supermag_root.rglob("supermag_*.parquet"):
        yyyymm = extract_yyyymm(path, SUPERMAG_FILE_RE)
        if yyyymm:
            months.add(yyyymm)
    return sorted(months)


def apply_global_date_filter(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df["timestamp"] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        df = df[df["timestamp"] <= end_ts]
    return df


def fuse_one_month(args: argparse.Namespace, yyyymm: str) -> Path | None:
    out_year = yyyymm[:4]
    out_dir = args.output_root / out_year
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"station_time_master_{yyyymm}.parquet"
    if out_path.exists() and not args.force:
        print(f"skip (exists): {out_path}")
        return out_path

    base = load_supermag_month(args.supermag_root, yyyymm)
    base = apply_global_date_filter(base, args.start_date, args.end_date)
    if base.empty:
        print(f"skip (empty after date filter): month={yyyymm}")
        return None

    month_start, month_end = month_start_end_utc(yyyymm)
    max_lb = max(args.source_lookback_min, args.swarm_lookback_min)
    window_start = month_start - pd.Timedelta(minutes=max_lb)
    window_end = month_end

    omni = load_omni_window(args.omni_root, window_start, window_end)
    goes_xrs = load_goes_xrs_window(args.goes_xrs_root, window_start, window_end)
    goes_mag = load_goes_mag_window(args.goes_mag_root, window_start, window_end)
    swarm = load_swarm_window(args.swarm_root, window_start, window_end)

    fused = backward_asof_join(base, omni, "timestamp", "omni_timestamp", args.source_lookback_min)
    fused = add_freshness_and_missing(fused, "omni", "omni_timestamp")

    fused = backward_asof_join(
        fused, goes_xrs, "timestamp", "goes_xrs_timestamp", args.source_lookback_min
    )
    fused = add_freshness_and_missing(fused, "goes_xrs", "goes_xrs_timestamp")

    fused = backward_asof_join(
        fused, goes_mag, "timestamp", "goes_mag_timestamp", args.source_lookback_min
    )
    fused = add_freshness_and_missing(fused, "goes_mag", "goes_mag_timestamp")

    fused = backward_asof_join(fused, swarm, "timestamp", "swarm_timestamp", args.swarm_lookback_min)
    fused = add_freshness_and_missing(fused, "swarm", "swarm_timestamp")

    fused["year"] = fused["timestamp"].dt.year.astype("Int16")
    fused["month"] = fused["timestamp"].dt.month.astype("Int8")
    fused["yyyymm"] = fused["timestamp"].dt.strftime("%Y%m")

    fused = fused.sort_values(["station", "timestamp"]).reset_index(drop=True)
    fused = fused.drop_duplicates(subset=["station", "timestamp"], keep="last")

    fused.to_parquet(out_path, index=False)
    print(
        f"wrote {out_path} | rows={len(fused)} | stations={fused['station'].nunique()} | "
        f"range={fused['timestamp'].min()} -> {fused['timestamp'].max()}"
    )
    return out_path


def main() -> None:
    args = parse_args()
    months = args.months if args.months else discover_supermag_months(args.supermag_root)
    if not months:
        raise FileNotFoundError(f"No SuperMAG monthly files found under {args.supermag_root}")

    written = 0
    for yyyymm in months:
        try:
            out = fuse_one_month(args, yyyymm)
            if out is not None:
                written += 1
        except Exception as exc:
            print(f"failed month={yyyymm}: {exc}")

    print(f"\nDone. Wrote {written} fused monthly files.")


if __name__ == "__main__":
    main()
