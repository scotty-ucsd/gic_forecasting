#!/usr/bin/env python3
"""
Phase 5.1 - Clean Fused: Station-Time Master to Interim Cleaned Parquet
------------------------------------------------------------------------
This is the sole Phase 5 script and the only script in src/clean/fused/.
It reads the monthly station-time master Parquet files produced by
Phase 4.1 (src/fusion/data_fusion.py), enforces a canonical
source-prefixed column schema via RENAME_MAP, ensures a gapless
1-minute time grid per station, strips join artifacts and redundant
columns, and writes cleaned monthly Parquet files to
data/fused/interim/. This script does NOT create classification
targets; the output filename suffix "_plus_target" reserves space
for a downstream target-generation script to append target columns.

Purpose
-------
The Phase 4.1 fusion output carries two schema issues that must be
resolved before modeling:

1. Mixed naming conventions: Phase 4.1 columns arrive under a mix
   of naming schemes (bare column names from SuperMAG, prefixed
   aggregation names from GOES such as "gmag_b_gsm_x", freshness
   columns as "goes_xrs_freshness_min", etc.). RENAME_MAP enforces
   a strict {source}_feature naming convention across all columns:
   SuperMAG columns gain a "supermag_" prefix, OMNI columns gain an
   "omni_" prefix, GOES MAG columns are renamed from "gmag_*" to
   "goes_mag_*", and XRS freshness/missing columns are renamed from
   "goes_xrs_*" to "xrs_*". Two historical naming variants for GOES
   MAG vectors are both mapped to the canonical "goes_mag_*" names.

2. Join artifacts: The as-of join in Phase 4.1 leaves four source
   timestamp columns (omni_timestamp, goes_xrs_timestamp,
   goes_mag_timestamp, swarm_timestamp), the "yyyymm" redundant
   temporal column, and (after buffer processing) an "_is_buffer"
   flag column. These are all dropped by clean_schema().

Additionally, resolve_supermag_frame() verifies that the three
canonical preferred-value columns (supermag_dbn_nt, supermag_dbe_nt,
supermag_dbz_nt) are present, then drops all raw per-frame component
columns (anything ending in "_nez_nt" or "_geo_nt"). These per-frame
columns are redundant once Phase 3.6 has resolved the preferred frame.

The script also enforces a gapless per-station 1-minute time grid via
_enforce_uniform_grid_segment(), which inserts NaN-filled rows for any
missing minutes and emits runtime warnings (not errors) for each gap.
Timestamps are stripped of timezone metadata to produce naive UTC
(tz_convert(None)) to prevent tz-aware/tz-naive comparison failures
downstream.

Month-edge buffer loading
-------------------------
To support continuity checks at the end of each month, load_with_buffer()
optionally reads up to --lookahead-min minutes (default 120) from the
start of the next month's file. However, the buffer rows are NOT
written to the output. They are loaded and grid-enforced solely to
ensure that the station grid at the month boundary is well-formed,
and are then filtered out (combined[combined["_is_buffer"] == False])
before the cleaned DataFrame is returned. The final output contains
only rows belonging to the current month. If no next-month file
exists (e.g. the last available month in the dataset), the current
month is processed without a buffer and a coverage note is logged.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory, output directory,
        --lookahead-min (default 120), and --force flag.
Step 2: Discover all station_time_master_*.parquet files under
        --input-dir (recursively). Raise FileNotFoundError if none.
Step 3: For each monthly file, skip if output exists and --force
        is not set.
Step 4: Probe the file after _rename_columns() to confirm required
        columns (station, timestamp, supermag_dbn_nt, supermag_dbe_nt,
        supermag_dbz_nt) are present. Raise ValueError if not.
Step 5: Call load_with_buffer():
        - Load and rename current month, coerce timestamps to naive
          UTC, deduplicate on (station, timestamp), clip to
          [month_start, month_end].
        - Enforce uniform 1-minute grid per station for the current
          month, marking all rows _is_buffer=False.
        - If next-month file exists, load its first --lookahead-min
          minutes, enforce uniform grid marking rows _is_buffer=True.
        - Concatenate current and buffer, deduplicate, then filter
          to _is_buffer=False only.
Step 6: Call clean_schema():
        - Drop JOIN_TIMESTAMPS (four source timestamp columns).
        - Drop REDUNDANT_TEMPORAL (yyyymm).
        - Drop DROP_IF_PRESENT (_is_buffer).
        - Call resolve_supermag_frame(): verify canonical SuperMAG
          preferred-value columns exist, drop all *_nez_nt and
          *_geo_nt raw frame columns.
Step 7: Write to
        data/fused/interim/{YYYY}/station_time_reduced_plus_target_{YYYYMM}.parquet
        using engine="pyarrow", compression="snappy".

INPUT DATA
----------
- data/fused/station_time_master/{YYYY}/station_time_master_{YYYYMM}.parquet
  Monthly station-time master Parquet files produced by Phase 4.1.
  All source columns present; mixed naming conventions; tz-aware
  timestamps; join timestamp columns present.
- CLI optional: --input-dir     (default: data/fused/station_time_master)
               --output-dir    (default: data/fused/interim)
               --lookahead-min (default: 120)
               --force         (overwrite existing output files)

OUTPUT DATA
-----------
- data/fused/interim/{YYYY}/station_time_reduced_plus_target_{YYYYMM}.parquet
  One Parquet file per month. Current-month rows only (no buffer).
  Timestamps are naive UTC. Schema enforces strict {source}_feature
  naming via RENAME_MAP. Raw per-frame SuperMAG component columns
  dropped; preferred-frame supermag_dbn_nt/dbe_nt/dbz_nt retained.
  Join timestamp columns, yyyymm, and _is_buffer columns dropped.
  Written with pyarrow engine and snappy compression.
  The "_plus_target" filename suffix reserves space for a downstream
  target-generation script; this script does not write any target
  columns.

NEXT PIPELINE PHASE
-------------------
- Phase 5.1 depends only on Phase 4.1 (src/fusion/data_fusion.py)
  completing first.
- This is the sole Phase 5 script and the final cleaning step in
  the pipeline.
- The output of this script is the primary input for downstream
  target generation and modeling. A subsequent target-generation
  script is expected to append classification or regression target
  columns to these files, leveraging the "_plus_target" filename
  reservation.
"""


from __future__ import annotations

import argparse
import calendar
import warnings
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_LOOKAHEAD_MIN: int = 120

JOIN_TIMESTAMPS: tuple[str, ...] = (
    "omni_timestamp",
    "goes_xrs_timestamp",
    "goes_mag_timestamp",
    "swarm_timestamp",
)

REDUNDANT_TEMPORAL: tuple[str, ...] = ("yyyymm",)

DROP_IF_PRESENT: tuple[str, ...] = (
    "_is_buffer",
)

RENAME_MAP: dict[str, str] = {
    "has_observation": "supermag_has_observation",
    "dbn_nt": "supermag_dbn_nt",
    "dbe_nt": "supermag_dbe_nt",
    "dbz_nt": "supermag_dbz_nt",
    "preferred_frame": "supermag_preferred_frame",
    "supermag_valid": "supermag_valid",
    "nez_valid": "supermag_nez_valid",
    "geo_valid": "supermag_geo_valid",
    "dbn_missing": "supermag_dbn_missing",
    "dbe_missing": "supermag_dbe_missing",
    "dbz_missing": "supermag_dbz_missing",
    "ext_sec": "supermag_ext_sec",
    "glon_deg": "supermag_glon_deg",
    "glat_deg": "supermag_glat_deg",
    "mlt_hour": "supermag_mlt_hour",
    "mcolat_deg": "supermag_mcolat_deg",
    "decl_deg": "supermag_decl_deg",
    "sza_deg": "supermag_sza_deg",
    "horizontal_perturbation_nt": "supermag_horizontal_perturbation_nt",
    "abs_dbdt_nt_per_min": "supermag_abs_dbdt_nt_per_min",
    "gic_proxy": "supermag_gic_proxy",
    "bt_nT": "omni_bt_nT",
    "bx_gse_gsm_nT": "omni_bx_gse_gsm_nT",
    "by_gsm_nT": "omni_by_gsm_nT",
    "bz_gsm_nT": "omni_bz_gsm_nT",
    "speed_km_s": "omni_speed_km_s",
    "vx_gse_km_s": "omni_vx_gse_km_s",
    "vy_gse_km_s": "omni_vy_gse_km_s",
    "vz_gse_km_s": "omni_vz_gse_km_s",
    "proton_density_n_cc": "omni_proton_density_n_cc",
    "flow_pressure_nPa": "omni_flow_pressure_nPa",
    "percent_interpolation": "omni_percent_interpolation",
    "timeshift_sec": "omni_timeshift_sec",
    "imf_valid": "omni_imf_valid",
    "plasma_valid": "omni_plasma_valid",
    "velocity_vector_valid": "omni_velocity_vector_valid",
    "goes_xrs_freshness_min": "xrs_freshness_min",
    "goes_xrs_missing": "xrs_missing",
    "gmag_sat_count": "goes_mag_sat_count",
    "gmag_any_valid": "goes_mag_any_valid",
    "gmag_b_gsm_x": "goes_mag_bx_gsm_nT",
    "gmag_b_gsm_y": "goes_mag_by_gsm_nT",
    "gmag_b_gsm_z": "goes_mag_bz_gsm_nT",
    "gmag_b_vdh_x": "goes_mag_bx_vdh_nT",
    "gmag_b_vdh_y": "goes_mag_by_vdh_nT",
    "gmag_b_vdh_z": "goes_mag_bz_vdh_nT",
    "goes_bx_gsm_nT": "goes_mag_bx_gsm_nT",
    "goes_by_gsm_nT": "goes_mag_by_gsm_nT",
    "goes_bz_gsm_nT": "goes_mag_bz_gsm_nT",
    "goes_bx_vdh_nT": "goes_mag_bx_vdh_nT",
    "goes_by_vdh_nT": "goes_mag_by_vdh_nT",
    "goes_bz_vdh_nT": "goes_mag_bz_vdh_nT",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_yyyymm_from_path(parquet_path: Path) -> tuple[int, int]:
    yyyymm = parquet_path.stem.split("_")[-1]
    return int(yyyymm[:4]), int(yyyymm[4:])


def _month_bounds_naive_utc(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1, hour=0, minute=0)
    last_day = calendar.monthrange(year, month)[1]
    end = pd.Timestamp(year=year, month=month, day=last_day, hour=23, minute=59)
    return start, end


def _next_month_path(current_path: Path) -> Path | None:
    year, month = _parse_yyyymm_from_path(current_path)
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1

    next_yyyymm = f"{next_year}{next_month:02d}"
    next_path = (
        current_path.parent.parent
        / str(next_year)
        / f"station_time_master_{next_yyyymm}.parquet"
    )
    return next_path if next_path.exists() else None


def _coerce_and_dedup_station_time(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "station" not in df.columns or "timestamp" not in df.columns:
        raise ValueError(f"{source_name}: missing required columns 'station' and/or 'timestamp'.")

    out = df.copy()

    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    bad_ts = int(ts.isna().sum())
    if bad_ts > 0:
        warnings.warn(
            f"{source_name}: dropping {bad_ts} rows with invalid timestamps.",
            stacklevel=2,
        )
    out["timestamp"] = ts.dt.tz_convert(None)
    out = out.dropna(subset=["timestamp"])

    out["station"] = out["station"].astype(str)
    out = out.sort_values(["station", "timestamp"]).reset_index(drop=True)

    dup_mask = out.duplicated(subset=["station", "timestamp"], keep="last")
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        warnings.warn(
            f"{source_name}: dropping {n_dup} duplicate (station, timestamp) rows; keeping last.",
            stacklevel=2,
        )
        out = out.loc[~dup_mask].copy()

    return out.reset_index(drop=True)


def _enforce_uniform_grid_segment(
    df: pd.DataFrame,
    *,
    segment_name: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    is_buffer_value: bool,
) -> pd.DataFrame:
    df = _coerce_and_dedup_station_time(df, segment_name)

    parts: list[pd.DataFrame] = []
    for station, grp in df.groupby("station", sort=False):
        grp = grp.sort_values("timestamp").copy()

        full_index = pd.date_range(start_ts, end_ts, freq="1min", name="timestamp")
        n_gap = len(full_index) - len(grp)

        if n_gap > 0:
            warnings.warn(
                f"Station {station}: inserted {n_gap} gap rows in {segment_name} "
                f"for {start_ts.strftime('%Y-%m-%d %H:%M')}..{end_ts.strftime('%Y-%m-%d %H:%M')}.",
                stacklevel=2,
            )

        grp = grp.set_index("timestamp").reindex(full_index)
        grp["station"] = station
        grp["_is_buffer"] = pd.Series(is_buffer_value, index=grp.index, dtype="boolean")
        parts.append(grp.reset_index())

    if not parts:
        out = df.iloc[0:0].copy()
        out["_is_buffer"] = pd.Series([], dtype="boolean")
        return out

    out = pd.concat(parts, ignore_index=True)
    if "index" in out.columns:
        out = out.rename(columns={"index": "timestamp"})
    return out


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})


# ---------------------------------------------------------------------------
# Loading with bounded next-month buffer
# ---------------------------------------------------------------------------

def load_with_buffer(
    parquet_path: Path,
    lookahead_min: int,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    year, month = _parse_yyyymm_from_path(parquet_path)
    month_start, month_end = _month_bounds_naive_utc(year, month)

    df_current = pd.read_parquet(parquet_path)
    df_current = _rename_columns(df_current)
    df_current = _coerce_and_dedup_station_time(df_current, parquet_path.name)
    df_current = df_current[
        (df_current["timestamp"] >= month_start) &
        (df_current["timestamp"] <= month_end)
    ].copy()

    current_grid_parts: list[pd.DataFrame] = []
    stations = list(df_current["station"].drop_duplicates())

    for station in stations:
        grp = df_current[df_current["station"] == station].copy()
        current_grid_parts.append(
            _enforce_uniform_grid_segment(
                grp,
                segment_name=f"{parquet_path.name}:current_month",
                start_ts=month_start,
                end_ts=month_end,
                is_buffer_value=False,
            )
        )

    current_grid = (
        pd.concat(current_grid_parts, ignore_index=True)
        if current_grid_parts else df_current.iloc[0:0].copy()
    )

    next_path = _next_month_path(parquet_path)
    if next_path is None:
        return current_grid, None

    next_year, next_month = _parse_yyyymm_from_path(next_path)
    next_month_start, _ = _month_bounds_naive_utc(next_year, next_month)
    buffer_end = next_month_start + pd.Timedelta(minutes=lookahead_min - 1)

    df_next = pd.read_parquet(next_path)
    df_next = _rename_columns(df_next)
    df_next = _coerce_and_dedup_station_time(df_next, next_path.name)
    df_next = df_next[
        (df_next["timestamp"] >= next_month_start) &
        (df_next["timestamp"] <= buffer_end)
    ].copy()

    buffer_parts: list[pd.DataFrame] = []
    for station in stations:
        grp = df_next[df_next["station"] == station].copy()
        if grp.empty:
            grp = df_current[df_current["station"] == station].iloc[0:0].copy()

        buffer_parts.append(
            _enforce_uniform_grid_segment(
                grp,
                segment_name=f"{next_path.name}:lookahead_buffer",
                start_ts=next_month_start,
                end_ts=buffer_end,
                is_buffer_value=True,
            )
        )

    buffer_grid = (
        pd.concat(buffer_parts, ignore_index=True)
        if buffer_parts else df_next.iloc[0:0].copy()
    )

    combined = pd.concat([current_grid, buffer_grid], ignore_index=True)
    combined = _coerce_and_dedup_station_time(combined, f"{parquet_path.name}+buffer_combined")
    combined["_is_buffer"] = combined["_is_buffer"].astype("boolean")

    if "_is_buffer" in combined.columns:
        combined = combined[combined["_is_buffer"] == False].copy()

    return combined, next_month_start


# ---------------------------------------------------------------------------
# Schema cleanup
# ---------------------------------------------------------------------------

def resolve_supermag_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "supermag_dbn_nt",
        "supermag_dbe_nt",
        "supermag_dbz_nt",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Canonical SuperMAG columns missing: {missing}. "
            "Check preferred_frame resolution in src/fusion/data_fusion.py."
        )

    frame_cols = [c for c in df.columns if c.endswith("_nez_nt") or c.endswith("_geo_nt")]
    return df.drop(columns=frame_cols, errors="ignore")


def clean_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = (
        list(JOIN_TIMESTAMPS)
        + list(REDUNDANT_TEMPORAL)
        + [c for c in DROP_IF_PRESENT if c in df.columns]
    )

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    df = resolve_supermag_frame(df)
    return df


# ---------------------------------------------------------------------------
# Per-month orchestration
# ---------------------------------------------------------------------------

def process_month(
    parquet_path: Path,
    lookahead_min: int = MAX_LOOKAHEAD_MIN,
) -> pd.DataFrame:
    probe = pd.read_parquet(parquet_path)
    probe = _rename_columns(probe)

    required_cols = {"station", "timestamp", "supermag_dbn_nt", "supermag_dbe_nt", "supermag_dbz_nt"}
    missing = required_cols - set(probe.columns)
    if missing:
        raise ValueError(
            f"{parquet_path.name}: missing required columns after rename: {sorted(missing)}"
        )

    df, _ = load_with_buffer(parquet_path, lookahead_min)
    df = clean_schema(df)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def convert_all(
    input_root: Path,
    output_root: Path,
    force: bool = False,
    lookahead_min: int = MAX_LOOKAHEAD_MIN,
) -> None:
    parquet_files = sorted(input_root.rglob("station_time_master_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No station_time_master_*.parquet files found under {input_root}"
        )

    for parquet_path in parquet_files:
        yyyymm = parquet_path.stem.split("_")[-1]
        yyyy = yyyymm[:4]
        out_path = output_root / yyyy / f"station_time_reduced_plus_target_{yyyymm}.parquet"

        if out_path.exists() and not force:
            print(f"[skip]  {out_path.name}")
            continue

        try:
            df = process_month(
                parquet_path=parquet_path,
                lookahead_min=lookahead_min,
            )
        except Exception as exc:
            print(f"[ERROR] {parquet_path.name}: {exc}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

        print(
            f"[ok]    {out_path.name} | "
            f"rows={len(df):>8,} | "
            f"stations={df['station'].nunique()} | "
            f"cols={len(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean fused station-time monthly parquets by enforcing a canonical "
            "source-prefixed schema, time-safe station/timestamp indexing, "
            "bounded month-edge continuity handling, and downstream-ready column cleanup."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/fused/station_time_master"),
        help="Root of fused station_time_master parquets (YYYY/ subdirs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fused/interim"),
        help="Root for cleaned fused output parquets (YYYY/ subdirs created).",
    )
    parser.add_argument(
        "--lookahead-min",
        type=int,
        default=MAX_LOOKAHEAD_MIN,
        help="Bounded next-month lookahead in minutes for month-edge continuity checks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output parquet files.",
    )
    args = parser.parse_args()

    convert_all(
        input_root=args.input_dir,
        output_root=args.output_dir,
        force=args.force,
        lookahead_min=args.lookahead_min,
    )


if __name__ == "__main__":
    main()

