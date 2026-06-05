#!/usr/bin/env python3
"""
Phase 2.3 - Clean Interim: OMNI Solar Wind .lst Files to Monthly Parquet
-------------------------------------------------------------------------
This is the third Phase 2 cleaning script. It reads manually staged
yearly OMNI 1-minute solar wind .lst files, reconstructs a full
datetime index from their year/doy/hour/minute columns, coerces all
physical variables to numeric types, and writes cleaned monthly
Parquet files partitioned by year. The output of this script is
required before OMNI preprocessing in Phase 3.x can begin.

Purpose
-------
OMNI provides 1-minute resolution solar wind and interplanetary
magnetic field (IMF) measurements propagated to the Earth bow shock
nose, making it the primary upstream solar wind driver input for the
ML pipeline. Key physical variables include IMF components (Bt, Bx,
By, Bz in GSM), solar wind speed components, proton density,
proton temperature, and flow pressure.

Unlike all other Phase 2 scripts, OMNI has no corresponding Phase 1
download script in this repository. Raw .lst files must be manually
downloaded from the NASA OMNIWeb interface at:
  https://omniweb.gsfc.nasa.gov/form/omni_min_def.html
and placed under data/raw/omni/ with filenames matching the pattern
omni_min_YYYY.lst before this script can run. The script raises
FileNotFoundError immediately if no matching files are found.

This script performs format conversion only: it does not interpolate,
resample, or merge across sources. Each yearly .lst file is treated
as a self-contained non-overlapping input, so no deduplication is
applied.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory and output directory.
Step 2: Glob all omni_min_*.lst files in --input-dir; raise
        FileNotFoundError if none are found.
Step 3: For each .lst file, read all columns as strings using
        whitespace delimiter and OMNI_COLUMNS as the header.
        Filter out any rows where the year field is not a 4-digit
        integer string (malformed row guard).
Step 4: Coerce year, doy, hour, minute to nullable Int64 and all
        remaining physical columns to float64 via pd.to_numeric
        with errors="coerce".
Step 5: Reconstruct a datetime timestamp arithmetically from
        year + (doy - 1 days) + hour + minute. Append a formatted
        "date" string column (YYYY-MM-DDTHH:MM) and a "yyyymm"
        grouping key column.
Step 6: Partition the DataFrame by yyyymm, drop the yyyymm column,
        and write one Parquet file per month to
        data/interim/omni/{YYYY}/omni_{YYYYMM}.parquet.

INPUT DATA
----------
- data/raw/omni/omni_min_*.lst
  Yearly OMNI 1-minute solar wind files in fixed-width whitespace-
  delimited text format, manually staged from NASA OMNIWeb.
  No automated download script exists for this source.
  Expected filename pattern: omni_min_YYYY.lst (e.g. omni_min_2015.lst)
- CLI optional: --input-dir  (default: data/raw/omni)
               --output-dir (default: data/interim/omni)

OUTPUT DATA
-----------
- data/interim/omni/{YYYY}/omni_{YYYYMM}.parquet
  One Parquet file per month. Schema includes all OMNI_COLUMNS
  (year, doy, hour, minute, spacecraft IDs, interpolation metadata,
  IMF components, solar wind velocity components, proton density,
  proton temperature, flow pressure) plus derived columns datetime
  and date. Physical columns are typed as float64; time index
  columns as nullable Int64.

NEXT PIPELINE PHASE
-------------------
- Phase 2.3 has no dependency on any other Phase 2 script and can
  run independently once raw OMNI .lst files are manually staged
  under data/raw/omni/.
- After this script completes, the OMNI interim data is ready for
  Phase 3.x preprocessing (src/preprocess/).
- All other Phase 2.x interim cleaning scripts should also complete
  before moving on to Phase 3.x, since Phase 4 (fusion) requires
  all preprocessed sources to be present.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

OMNI_COLUMNS = [
    "year",
    "doy",
    "hour",
    "minute",
    "imf_spacecraft_id",
    "plasma_spacecraft_id",
    "n_points_imf_avg",
    "n_points_plasma_avg",
    "percent_interpolation",
    "timeshift_sec",
    "time_between_obs_sec",
    "bt_nT",
    "bx_gse_gsm_nT",
    "by_gsm_nT",
    "bz_gsm_nT",
    "speed_km_s",
    "vx_gse_km_s",
    "vy_gse_km_s",
    "vz_gse_km_s",
    "proton_density_n_cc",
    "proton_temperature_K",
    "flow_pressure_nPa",
]


def read_omni_lst(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=OMNI_COLUMNS,
        engine="python",
        dtype=str,
    )

    df = df[df["year"].str.fullmatch(r"\d{4}", na=False)].copy()

    int_cols = ["year", "doy", "hour", "minute"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    float_cols = [c for c in OMNI_COLUMNS if c not in int_cols]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    base = pd.to_datetime(df["year"].astype(str), format="%Y", errors="coerce")
    df["datetime"] = (
        base
        + pd.to_timedelta(df["doy"] - 1, unit="D")
        + pd.to_timedelta(df["hour"], unit="h")
        + pd.to_timedelta(df["minute"], unit="m")
    )
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M")
    df["yyyymm"] = df["datetime"].dt.strftime("%Y%m")

    return df


def write_monthly_parquets(df: pd.DataFrame, output_root: Path) -> None:
    if df.empty:
        return

    year = int(df["year"].dropna().iloc[0])
    year_dir = output_root / f"{year}"
    year_dir.mkdir(parents=True, exist_ok=True)

    for yyyymm, group in df.groupby("yyyymm", sort=True):
        out_path = year_dir / f"omni_{yyyymm}.parquet"
        group = (
            group.drop(columns=["yyyymm"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        group.to_parquet(out_path, index=False)


def convert_all(input_dir: Path, output_dir: Path) -> None:
    files = sorted(input_dir.glob("omni_min_*.lst"))
    if not files:
        raise FileNotFoundError(f"No omni_min_*.lst files found in {input_dir}")

    for path in files:
        df = read_omni_lst(path)
        write_monthly_parquets(df, output_dir)
        print(f"Converted {path} -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert yearly OMNI .lst files into monthly parquet files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/omni"),
        help="Directory containing omni_min_YYYY.lst files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/omni"),
        help="Directory to write monthly parquet files",
    )
    args = parser.parse_args()

    convert_all(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
