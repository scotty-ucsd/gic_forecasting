#!/usr/bin/env python3
"""
Phase 3.6 - Preprocess: SuperMAG Interim Parquet to Preprocessed Parquet
-------------------------------------------------------------------------
This is the sixth and final Phase 3 preprocessing script. It reads
monthly interim SuperMAG Parquet files produced by Phase 2.5
(src/clean/interim/supermag.py), enforces a UTC-aware gapless
1-minute time index, resolves dual magnetic coordinate frame
availability into preferred-frame scalar columns, appends per-frame
and per-component validity flags, standardizes column types, and
writes preprocessed monthly Parquet files to
data/preprocessed/supermag/. This script completes Phase 3 and
enables Phase 4 (src/fusion/) to begin once all other Phase 3.x
scripts have also finished.

Purpose
-------
SuperMAG provides magnetic field perturbations in two coordinate
frames per station: NEZ (local geomagnetic North-East-Z) and GEO
(geographic North-East-Z). Not all stations or time periods have
both frames available. This script resolves frame availability into
three unified preferred-value columns (dbn_nt, dbe_nt, dbz_nt) by
selecting NEZ values when all three NEZ components are non-null, and
falling back to GEO values otherwise. The selection is recorded in
the "preferred_frame" string column ("nez", "geo", or null).
Both raw frame columns are preserved in the output so downstream
scripts can override the preference if needed.

Three boolean validity columns are appended:
- nez_valid: all three NEZ components are non-null
- geo_valid: all three GEO components are non-null
- supermag_valid: nez_valid OR geo_valid (at least one frame usable)

Three per-component missingness columns (dbn_missing, dbe_missing,
dbz_missing) reflect missingness in the preferred-value columns.

Unlike Phase 3.4 and 3.5, the 1-minute reindexing in this script
produces a UTC timezone-aware index (pd.date_range with tz="UTC")
and is controlled by the --reindex-minutely flag (default: True).
When reindexing is applied, a "has_observation" boolean column
distinguishes rows that originated from the source data (True) from
rows inserted to fill gaps (False). When --no-reindex-minutely is
set, has_observation is set to True for all rows without reindexing.

This script also includes a --force flag and optional date range
filter not present in Phase 3.4 and 3.5.

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, optional
        station filter list, --reindex-minutely flag (default True),
        optional date range, and optional --force flag.
Step 2: Recursively discover all supermag_*.parquet files under
        --input-root, optionally filtered by station code inferred
        from the two-levels-up directory name (uppercased).
Step 3: For each file, parse station code and YYYYMM from the
        filename via FILE_RE. Skip files with unexpected names.
        Skip existing output files unless --force is set.
Step 4: Apply ensure_station(): normalize station to uppercase from
        the "station" column, then "source_station" column, then
        the filename hint.
Step 5: Normalize timestamp to UTC-aware Timestamp, drop rows with
        unparseable timestamps, apply optional date range filter.
        Skip the file if the result is empty.
Step 6: Drop source_url and source_file provenance columns if present.
        Retain only timestamp, station, COMPONENT_COLS, and META_COLS.
        Deduplicate on (station, timestamp) keeping the last occurrence.
Step 7: If --reindex-minutely (default), apply minute_reindex():
        construct a UTC-aware gapless 1-minute pd.date_range, reindex,
        forward/back fill station, and set has_observation=True only
        for rows present in the source data. Otherwise set
        has_observation=True for all rows without reindexing.
Step 8: Apply add_validity_flags(): ensure all COMPONENT_COLS and
        META_COLS exist (fill with NaN if absent), compute nez_valid,
        geo_valid, supermag_valid, derive preferred-frame dbn_nt/
        dbe_nt/dbz_nt and preferred_frame label, and add per-component
        missingness flags.
Step 9: Apply standardize_dtypes(): field and meta columns to float32,
        boolean columns to pandas nullable boolean, preferred_frame
        to pandas string.
Step 10: Enforce output column order and write to
         data/preprocessed/supermag/{STATION}/{YYYY}/
         supermag_{STATION}_{YYYYMM}.parquet.

INPUT DATA
----------
- data/interim/supermag/{STATION}/{YYYY}/supermag_{STATION}_{YYYYMM}.parquet
  Monthly interim SuperMAG Parquet files produced by Phase 2.5.
  Expected COMPONENT_COLS: dbn_nez_nt, dbe_nez_nt, dbz_nez_nt,
  dbn_geo_nt, dbe_geo_nt, dbz_geo_nt.
  Expected META_COLS: ext_sec, glon_deg, glat_deg, mlt_hour,
  mcolat_deg, decl_deg, sza_deg.
- CLI optional: --input-root        (default: data/interim/supermag)
               --output-root       (default: data/preprocessed/supermag)
               --stations          (default: all found; e.g. YKC MEA
                                    SOD ABK)
               --reindex-minutely  (default: True; use
                                    --no-reindex-minutely to disable)
               --start-date        (default: no lower bound; YYYY-MM-DD)
               --end-date          (default: no upper bound; YYYY-MM-DD)
               --force             (overwrite existing output files)

OUTPUT DATA
-----------
- data/preprocessed/supermag/{STATION}/{YYYY}/supermag_{STATION}_{YYYYMM}.parquet
  One Parquet file per station per month. Output column order:
  timestamp (UTC-aware), station, has_observation (boolean),
  dbn_nt/dbe_nt/dbz_nt (float32, preferred frame values),
  preferred_frame (string), supermag_valid/nez_valid/geo_valid
  (boolean), dbn_missing/dbe_missing/dbz_missing (boolean),
  dbn_nez_nt/dbe_nez_nt/dbz_nez_nt (float32),
  dbn_geo_nt/dbe_geo_nt/dbz_geo_nt (float32),
  ext_sec, glon_deg, glat_deg, mlt_hour, mcolat_deg, decl_deg,
  sza_deg (float32).

NEXT PIPELINE PHASE
-------------------
- Phase 3.6 depends only on Phase 2.5 (src/clean/interim/supermag.py)
  completing first. It has no dependency on Phase 3.1 through 3.5
  and can run in parallel with other Phase 3.x scripts.
- This is the final Phase 3 script. Once all six Phase 3.x scripts
  have completed (swarm_1min, swarm_chaos, omni, goes_xrs, goes_mag,
  supermag), all preprocessed sources are ready for Phase 4.
- Move on to Phase 4 (src/fusion/data_fusion.py) once all Phase 3.x
  scripts have finished.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_ROOT = Path("data/interim/supermag")
DEFAULT_OUTPUT_ROOT = Path("data/preprocessed/supermag")

FILE_RE = re.compile(r"supermag_([A-Z0-9]{3})_(\d{6})\.parquet$")

COMPONENT_COLS = [
    "dbn_nez_nt",
    "dbe_nez_nt",
    "dbz_nez_nt",
    "dbn_geo_nt",
    "dbe_geo_nt",
    "dbz_geo_nt",
]

META_COLS = [
    "ext_sec",
    "glon_deg",
    "glat_deg",
    "mlt_hour",
    "mcolat_deg",
    "decl_deg",
    "sza_deg",
]

DROP_COLUMNS_IF_PRESENT = [
    "source_url",
    "source_file",
]


def collect_files(root: Path, stations: list[str] | None) -> list[Path]:
    files = sorted(root.rglob("supermag_*.parquet"))
    if stations:
        keep = {s.upper() for s in stations}
        files = [p for p in files if p.parent.parent.name.upper() in keep]
    return files


def parse_station_yyyymm(path: Path) -> tuple[str, str] | None:
    match = FILE_RE.match(path.name)
    if not match:
        return None
    return match.group(1), match.group(2)


def ensure_station(df: pd.DataFrame, station_hint: str) -> pd.DataFrame:
    if "station" in df.columns:
        df["station"] = df["station"].astype(str).str.upper()
        return df
    if "source_station" in df.columns:
        df["station"] = df["source_station"].astype(str).str.upper()
        return df
    df["station"] = station_hint.upper()
    return df


def minute_reindex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["has_observation"] = True
    full_index = pd.date_range(
        df["timestamp"].min().floor("min"),
        df["timestamp"].max().floor("min"),
        freq="1min",
        tz="UTC",
    )
    out = df.set_index("timestamp").reindex(full_index)
    out.index.name = "timestamp"
    out = out.reset_index()
    out["station"] = out["station"].ffill().bfill()
    out["has_observation"] = out["has_observation"].eq(True)
    return out


def add_validity_flags(df: pd.DataFrame) -> pd.DataFrame:
    for col in COMPONENT_COLS + META_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df["nez_valid"] = df[["dbn_nez_nt", "dbe_nez_nt", "dbz_nez_nt"]].notna().all(axis=1)
    df["geo_valid"] = df[["dbn_geo_nt", "dbe_geo_nt", "dbz_geo_nt"]].notna().all(axis=1)
    df["supermag_valid"] = df["nez_valid"] | df["geo_valid"]

    # Prefer NEZ components for downstream modeling while preserving both frames.
    df["dbn_nt"] = np.where(df["nez_valid"], df["dbn_nez_nt"], df["dbn_geo_nt"])
    df["dbe_nt"] = np.where(df["nez_valid"], df["dbe_nez_nt"], df["dbe_geo_nt"])
    df["dbz_nt"] = np.where(df["nez_valid"], df["dbz_nez_nt"], df["dbz_geo_nt"])

    df["preferred_frame"] = np.where(
        df["nez_valid"],
        "nez",
        np.where(df["geo_valid"], "geo", pd.NA),
    )

    df["dbn_missing"] = df["dbn_nt"].isna()
    df["dbe_missing"] = df["dbe_nt"].isna()
    df["dbz_missing"] = df["dbz_nt"].isna()
    return df


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = COMPONENT_COLS + META_COLS + ["dbn_nt", "dbe_nt", "dbz_nt"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    bool_cols = [
        "has_observation",
        "nez_valid",
        "geo_valid",
        "supermag_valid",
        "dbn_missing",
        "dbe_missing",
        "dbz_missing",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    if "preferred_frame" in df.columns:
        df["preferred_frame"] = df["preferred_frame"].astype("string")
    return df


def apply_date_filter(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df["timestamp"] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        df = df[df["timestamp"] <= end_ts]
    return df


def preprocess_file(
    path: Path,
    output_root: Path,
    *,
    reindex_minutely: bool,
    start_date: str | None,
    end_date: str | None,
    force: bool,
) -> Path | None:
    parsed = parse_station_yyyymm(path)
    if not parsed:
        print(f"skip (unexpected filename): {path}")
        return None
    station_hint, yyyymm = parsed
    year = yyyymm[:4]

    out_dir = output_root / station_hint / year
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"supermag_{station_hint}_{yyyymm}.parquet"
    if out_path.exists() and not force:
        print(f"skip (exists): {out_path}")
        return out_path

    df = pd.read_parquet(path).copy()
    df = ensure_station(df, station_hint)

    if "timestamp" not in df.columns:
        raise ValueError(f"missing timestamp column in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = apply_date_filter(df, start_date, end_date)
    if df.empty:
        print(f"skip (empty after date filter): {path}")
        return None

    for col in DROP_COLUMNS_IF_PRESENT:
        if col in df.columns:
            df = df.drop(columns=col)

    keep = ["timestamp", "station"] + [c for c in COMPONENT_COLS + META_COLS if c in df.columns]
    df = df[keep]
    df = df.sort_values("timestamp").drop_duplicates(subset=["station", "timestamp"], keep="last")

    if reindex_minutely:
        df = minute_reindex(df)
    else:
        df["has_observation"] = True

    df = add_validity_flags(df)
    df = standardize_dtypes(df)

    ordered = [
        "timestamp",
        "station",
        "has_observation",
        "dbn_nt",
        "dbe_nt",
        "dbz_nt",
        "preferred_frame",
        "supermag_valid",
        "nez_valid",
        "geo_valid",
        "dbn_missing",
        "dbe_missing",
        "dbz_missing",
        "dbn_nez_nt",
        "dbe_nez_nt",
        "dbz_nez_nt",
        "dbn_geo_nt",
        "dbe_geo_nt",
        "dbz_geo_nt",
        "ext_sec",
        "glon_deg",
        "glat_deg",
        "mlt_hour",
        "mcolat_deg",
        "decl_deg",
        "sza_deg",
    ]
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered].reset_index(drop=True)

    df.to_parquet(out_path, index=False)
    print(
        f"wrote {out_path} | rows={len(df)} | "
        f"start={df['timestamp'].min()} | end={df['timestamp'].max()}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess interim SuperMAG parquet files.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Optional station filters like YKC MEA SOD ABK",
    )
    parser.add_argument(
        "--reindex-minutely",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reindex each station-month to 1-minute cadence over observed bounds.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Optional inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="Optional inclusive end date YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    files = collect_files(args.input_root, args.stations)
    if not files:
        print("No interim SuperMAG parquet files found.")
        return

    written = 0
    for path in files:
        try:
            out = preprocess_file(
                path,
                args.output_root,
                reindex_minutely=args.reindex_minutely,
                start_date=args.start_date,
                end_date=args.end_date,
                force=args.force,
            )
            if out is not None:
                written += 1
        except Exception as exc:
            print(f"failed preprocessing {path}: {exc}")

    print(f"\nDone. Wrote {written} files.")


if __name__ == "__main__":
    main()
