#!/usr/bin/env python3
"""
Phase 3.4 - Preprocess: GOES XRS Interim Parquet to Preprocessed Parquet
-------------------------------------------------------------------------
This is the fourth Phase 3 preprocessing script. It reads monthly
interim GOES XRS Parquet files produced by Phase 2.1
(src/clean/interim/goes_xrs.py), enforces a gapless 1-minute time
index, appends generation-aware validity flags and a derived flux
ratio column, standardizes column types, and writes preprocessed
monthly Parquet files to data/preprocessed/goes_xrs/. This script
has no dependency on Phase 3.1, 3.2, or 3.3 and can run in parallel
with other Phase 3.x scripts.

Purpose
-------
The interim GOES XRS data from Phase 2.1 contains only the timestamps
actually present in the source NetCDF files. Gaps in the 1-minute
record are represented as absent rows rather than NaN-filled rows.
This script enforces a complete 1-minute time index via minute_reindex(),
which constructs a full pd.date_range from the first to the last
minute in the file and reindexes the DataFrame against it, introducing
NaN-filled rows for every missing minute. This ensures a gapless
regular time grid for all downstream consumers.

The script is generation-aware: it detects whether a file comes from
a legacy satellite (goes15) or a GOES-R satellite (goes16/17/18) by
checking for the presence of GOESR_OPTIONAL_COLUMNS, and assigns the
result to an "xrs_source" label column. This label is then used to
apply the correct flag acceptance criteria when computing validity
masks:
- Legacy: xrsa_flag and xrsb_flag in {0, 1}
- GOES-R: xrsa_flag and xrsb_flag in {0, 1, 2}

Both validity columns additionally require the corresponding flux
value to be non-null. A derived "xray_flux_ratio" column (xrsb_flux /
xrsa_flux) is computed when both channels are positive. The
has_electron_correction boolean column indicates whether a GOES-R
electron correction was applied to a given row.

Provenance columns from the interim stage (source_file,
source_file_type, satellite) are dropped before output. This script
does not apply a --force flag; existing output files for a given
satellite/month are unconditionally overwritten.

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, and optional
        satellite filter list.
Step 2: Recursively discover all Parquet files under --input-root,
        optionally filtered by satellite name inferred from the
        two-levels-up directory name.
Step 3: For each file, apply ensure_sat_id() to normalize the sat_id
        column: fall back to "satellite" column, then to the directory
        name if neither column exists.
Step 4: Drop interim-only provenance columns (source_file,
        source_file_type, satellite) and parse the timestamp column.
Step 5: Apply minute_reindex(): construct a complete 1-minute
        pd.date_range across the file's time span and reindex the
        DataFrame, introducing NaN rows for missing minutes.
        Forward/back fill sat_id over introduced NaN rows.
Step 6: Assign the xrs_source label ("goesr" or "legacy") via
        build_source_label() based on presence of GOESR_OPTIONAL_COLUMNS.
Step 7: Compute generation-aware xrsa_valid and xrsb_valid boolean
        columns using flag acceptance sets appropriate to each
        generation, combined with a non-null flux requirement.
Step 8: Compute has_electron_correction boolean (GOES-R only) and
        xray_flux_ratio = xrsb_flux / xrsa_flux (when both are positive).
Step 9: Enforce ordered output column list, apply standardize_dtypes()
        (flux -> float32, flags -> Int64, validity -> boolean), and
        write to data/preprocessed/goes_xrs/{sat_id}/{YYYY}/{sat_id}_{YYYYMM}.parquet.

INPUT DATA
----------
- data/interim/goes_xrs/{sat_id}/{YYYY}/*.parquet
  Monthly interim GOES XRS Parquet files produced by Phase 2.1.
  Expected to contain timestamp, satellite or sat_id, xrsa_flux,
  xrsb_flux, xrsa_flag, xrsb_flag, xrsa_num, xrsb_num, and
  optionally GOESR_OPTIONAL_COLUMNS (xrsa_flux_electrons,
  xrsb_flux_electrons, electron_correction_flag).
- CLI optional: --input-root  (default: data/interim/goes_xrs)
               --output-root (default: data/preprocessed/goes_xrs)
               --satellites  (default: all found; e.g. goes15 goes16
                              goes17 goes18)

OUTPUT DATA
-----------
- data/preprocessed/goes_xrs/{sat_id}/{YYYY}/{sat_id}_{YYYYMM}.parquet
  One Parquet file per satellite per month with a gapless 1-minute
  time index. Output column order: timestamp, sat_id, xrs_source,
  xrsa_flux (float32), xrsb_flux (float32), xray_flux_ratio (float32),
  xrsa_flag (Int64), xrsb_flag (Int64), xrsa_num (Int64),
  xrsb_num (Int64), xrsa_valid (boolean), xrsb_valid (boolean),
  and for GOES-R files: xrsa_flux_electrons, xrsb_flux_electrons,
  electron_correction_flag, has_electron_correction (boolean).

NEXT PIPELINE PHASE
-------------------
- Phase 3.4 depends only on Phase 2.1 (src/clean/interim/goes_xrs.py)
  completing first. It has no dependency on Phase 3.1, 3.2, or 3.3
  and can run in parallel with other Phase 3.x scripts.
- After this script completes, GOES XRS preprocessed data is ready
  for Phase 4 (src/fusion/data_fusion.py).
- All Phase 3.x preprocessing scripts must complete before Phase 4
  can begin, since the fusion stage requires all sources to be present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_ROOT = Path("data/interim/goes_xrs")
DEFAULT_OUTPUT_ROOT = Path("data/preprocessed/goes_xrs")


CORE_COLUMNS = [
    "timestamp",
    "sat_id",
    "xrsa_flux",
    "xrsb_flux",
    "xrsa_flag",
    "xrsb_flag",
    "xrsa_num",
    "xrsb_num",
]

GOESR_OPTIONAL_COLUMNS = [
    "xrsa_flux_electrons",
    "xrsb_flux_electrons",
    "electron_correction_flag",
]

DROP_COLUMNS_IF_PRESENT = [
    "source_file",
    "source_file_type",
    "satellite",
]


def collect_files(root: Path, satellites: list[str] | None) -> list[Path]:
    files = sorted(root.rglob("*.parquet"))
    if satellites:
        sat_set = set(satellites)
        files = [p for p in files if p.parent.parent.name in sat_set]
    return files


def ensure_sat_id(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "sat_id" not in df.columns:
        if "satellite" in df.columns:
            df["sat_id"] = df["satellite"]
        else:
            df["sat_id"] = path.parent.parent.name
    return df


def minute_reindex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    full_index = pd.date_range(ts.min().floor("min"), ts.max().floor("min"), freq="1min")

    df = df.set_index("timestamp")
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    df = df.reset_index()

    if "sat_id" in df.columns:
        df["sat_id"] = df["sat_id"].ffill().bfill()

    return df


def build_source_label(df: pd.DataFrame) -> pd.Series:
    has_goesr_fields = any(col in df.columns for col in GOESR_OPTIONAL_COLUMNS)
    label = "goesr" if has_goesr_fields else "legacy"
    return pd.Series(label, index=df.index, dtype="object")


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = [
        "xrsa_flux",
        "xrsb_flux",
        "xrsa_flux_electrons",
        "xrsb_flux_electrons",
        "xray_flux_ratio",
    ]
    int_cols = [
        "xrsa_flag",
        "xrsb_flag",
        "xrsa_num",
        "xrsb_num",
        "electron_correction_flag",
    ]
    bool_cols = [
        "xrsa_valid",
        "xrsb_valid",
        "has_electron_correction",
    ]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.array(pd.to_numeric(df[col], errors="coerce"), dtype="Int64")

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    return df


def preprocess_file(path: Path, output_root: Path) -> Path:
    df = pd.read_parquet(path)
    df = ensure_sat_id(df, path)

    for col in DROP_COLUMNS_IF_PRESENT:
        if col in df.columns:
            df = df.drop(columns=col)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = minute_reindex(df)

    df["xrs_source"] = build_source_label(df)

    keep_cols = CORE_COLUMNS + [c for c in GOESR_OPTIONAL_COLUMNS if c in df.columns]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[[c for c in keep_cols if c in df.columns] + ["xrs_source"]]

    for required in ["xrsa_flux", "xrsb_flux", "xrsa_flag", "xrsb_flag", "xrsa_num", "xrsb_num"]:
        if required not in df.columns:
            df[required] = pd.NA

    is_goesr = df["xrs_source"].eq("goesr")

    xrsa_flag_num = pd.to_numeric(df["xrsa_flag"], errors="coerce")
    xrsb_flag_num = pd.to_numeric(df["xrsb_flag"], errors="coerce")

    legacy_ok_a = xrsa_flag_num.isin([0, 1])
    legacy_ok_b = xrsb_flag_num.isin([0, 1])

    goesr_ok_a = xrsa_flag_num.isin([0, 1, 2])
    goesr_ok_b = xrsb_flag_num.isin([0, 1, 2])

    df["xrsa_valid"] = np.where(is_goesr, goesr_ok_a, legacy_ok_a) & df["xrsa_flux"].notna()
    df["xrsb_valid"] = np.where(is_goesr, goesr_ok_b, legacy_ok_b) & df["xrsb_flux"].notna()

    if "electron_correction_flag" in df.columns:
        ecf = pd.to_numeric(df["electron_correction_flag"], errors="coerce")
        df["has_electron_correction"] = ecf.notna()
    else:
        df["has_electron_correction"] = False

    a = pd.to_numeric(df["xrsa_flux"], errors="coerce")
    b = pd.to_numeric(df["xrsb_flux"], errors="coerce")
    df["xray_flux_ratio"] = np.where((a > 0) & (b > 0), b / a, np.nan)

    ordered_cols = [
        "timestamp",
        "sat_id",
        "xrs_source",
        "xrsa_flux",
        "xrsb_flux",
        "xray_flux_ratio",
        "xrsa_flag",
        "xrsb_flag",
        "xrsa_num",
        "xrsb_num",
        "xrsa_valid",
        "xrsb_valid",
    ]

    for col in GOESR_OPTIONAL_COLUMNS:
        if col in df.columns:
            ordered_cols.append(col)

    ordered_cols.append("has_electron_correction")
    df = df[ordered_cols]

    df = standardize_dtypes(df)

    sat_id = str(df["sat_id"].dropna().iloc[0])
    month_start = pd.to_datetime(df["timestamp"].dropna().iloc[0])
    year = f"{month_start.year:04d}"
    yyyymm = f"{month_start.year:04d}{month_start.month:02d}"

    outdir = output_root / sat_id / year
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{sat_id}_{yyyymm}.parquet"
    df.to_parquet(outpath, index=False)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess interim GOES XRS parquet files.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--satellites", nargs="*", default=None, help="Optional sat filters like goes15 goes16")
    args = parser.parse_args()

    files = collect_files(args.input_root, args.satellites)
    if not files:
        print("No interim GOES XRS parquet files found.")
        return

    outputs = []
    for path in files:
        try:
            outpath = preprocess_file(path, args.output_root)
            outputs.append(outpath)
            print(f"wrote {outpath}")
        except Exception as e:
            print(f"failed preprocessing {path}: {e}")

    print(f"\nDone. Wrote {len(outputs)} files.")


if __name__ == "__main__":
    main()
