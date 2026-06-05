#!/usr/bin/env python3
"""
Phase 2.5 - Clean Interim: SuperMAG Raw Daily Parquet to Interim Parquet
-------------------------------------------------------------------------
This is the fifth and final Phase 2 cleaning script. It reads raw
daily SuperMAG Parquet files produced by Phase 1.4
(src/download/supermag.py), normalizes their schema into a fixed
column order, flattens nested magnetic perturbation dicts into scalar
columns, resolves multiple possible timestamp and station column name
variants, deduplicates on (station, timestamp), and writes either
cleaned daily or monthly Parquet files. The output of this script
completes Phase 2 and enables Phase 3.x preprocessing to begin across
all sources.

Purpose
-------
SuperMAG raw Parquet files present two structural challenges that no
other Phase 2 script faces. First, the timestamp column may arrive
under any of four names depending on API version or schema drift:
"tval" (Unix epoch int), "timestamp", "Timestamp", or "time". The
script attempts each in order and raises ValueError if none is found.
Second, the magnetic perturbation components N, E, Z arrive as nested
dicts (or stringified dicts) keyed by coordinate frame: "nez" for the
local geomagnetic NEZ frame and "geo" for the geographic frame. The
flatten_component() function uses ast.literal_eval to handle
stringified dict values, extracting scalar nT values into six output
columns: dbn_nez_nt, dbe_nez_nt, dbz_nez_nt, dbn_geo_nt, dbe_geo_nt,
dbz_geo_nt.

SuperMAG is a multi-station source, so deduplication is applied on
(station, timestamp) rather than timestamp alone, which is the key
distinction from the Swarm cleaning script (Phase 2.4). Empty input
files (0 rows, 0 columns) are handled gracefully by returning an
empty DataFrame with the correct COLUMN_ORDER schema rather than
raising an error.

Like Phase 2.4, this script supports two output modes via --clean-type:
"monthly" (default) concatenates daily files per (station, year, month)
and writes one Parquet file per group; "daily" mirrors the raw directory
structure without concatenation.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory, output directory,
        --clean-type (monthly or daily), and optional --force flag.
Step 2: Recursively discover all supermag_*.parquet files under
        --input-dir. Raise FileNotFoundError if none are found.
Step 3 (both modes): For each raw daily file, apply clean_one_file():
        - Detect timestamp column from candidates: tval (Unix epoch
          seconds -> UTC datetime), timestamp, Timestamp, or time.
          Drop rows with unparseable timestamps.
        - Detect station column from: iaga or source_station.
        - Map scalar coordinate and metadata fields (glon, glat, mlt,
          mcolat, decl, sza, ext) to pipeline column names, filling
          with pd.NA if absent.
        - Flatten N/E/Z magnetic perturbation nested dicts into six
          scalar columns (NEZ and GEO frames) via flatten_component()
          and ast.literal_eval for stringified dict handling.
        - Append source_station, source_url (pd.NA if absent), and
          source_file provenance columns.
        - Sort by timestamp, deduplicate on (station, timestamp),
          and enforce COLUMN_ORDER.
Step 4a (daily mode): Write each cleaned file to the same relative
        path under --output-dir. Skip existing files unless --force.
Step 4b (monthly mode): Group files by (station, year, month) parsed
        via FILENAME_RE. Concatenate all valid daily frames, apply a
        second deduplication on (station, timestamp) across the
        combined frame, enforce COLUMN_ORDER, and write one Parquet
        file per group. Skip existing files unless --force.

INPUT DATA
----------
- data/raw/supermag/**/supermag_*.parquet
  Raw daily SuperMAG Parquet files produced by Phase 1.4.
  Expected filename pattern: supermag_{STATION}_{YYYYMMDD}.parquet
  Station codes are 3-character IAGA identifiers (e.g. YKC, MEA).
  Timestamp column name may vary: tval, timestamp, Timestamp, or time.
  Magnetic components N, E, Z may be nested dicts or stringified dicts
  with keys "nez" and "geo".
- CLI optional: --input-dir   (default: data/raw/supermag)
               --output-dir  (default: data/interim/supermag)
               --clean-type  (default: monthly; choices: monthly, daily)
               --force       (overwrite existing output files)

OUTPUT DATA
-----------
- Monthly mode (default):
  data/interim/supermag/{STATION}/{YYYY}/supermag_{STATION}_{YYYYMM}.parquet
  One Parquet file per station per month.
- Daily mode:
  data/interim/supermag/{STATION}/{YYYY}/{MM}/supermag_{STATION}_{YYYYMMDD}.parquet
  Mirrors raw input directory structure under --output-dir.
- Fixed output schema (COLUMN_ORDER, 18 columns): timestamp, station,
  ext_sec, glon_deg, glat_deg, mlt_hour, mcolat_deg, decl_deg,
  sza_deg, dbn_nez_nt, dbe_nez_nt, dbz_nez_nt, dbn_geo_nt,
  dbe_geo_nt, dbz_geo_nt, source_station, source_url, source_file.

NEXT PIPELINE PHASE
-------------------
- Phase 2.5 has no dependency on any other Phase 2 script and can
  run independently once Phase 1.4 (src/download/supermag.py) has
  completed.
- Run with --clean-type monthly (the default) to produce the output
  expected by Phase 3.x preprocessing.
- This is the final Phase 2 script. Once all five Phase 2.x scripts
  have completed (goes_xrs, goes_mag, omni, swarm, supermag), all
  interim sources are ready and Phase 3.x preprocessing can begin.
- Reminder: in Phase 3.x, swarm_1min.py must complete before
  swarm_chaos.py can run. All other Phase 3.x scripts are independent.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
import re
import pandas as pd


FILENAME_RE = re.compile(r"supermag_([A-Z0-9]{3})_(\d{8})\.parquet$")


COLUMN_ORDER = [
    "timestamp",
    "station",
    "ext_sec",
    "glon_deg",
    "glat_deg",
    "mlt_hour",
    "mcolat_deg",
    "decl_deg",
    "sza_deg",
    "dbn_nez_nt",
    "dbe_nez_nt",
    "dbz_nez_nt",
    "dbn_geo_nt",
    "dbe_geo_nt",
    "dbz_geo_nt",
    "source_station",
    "source_url",
    "source_file",
]


def parse_nested_value(value, key: str):
    if isinstance(value, dict):
        return value.get(key)

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed.get(key)
        except Exception:
            return None

    return None


def flatten_component(df: pd.DataFrame, column: str, key: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    return df[column].map(lambda x: parse_nested_value(x, key))


def clean_one_file(infile: Path) -> pd.DataFrame:
    df = pd.read_parquet(infile).copy()
    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(columns=COLUMN_ORDER)

    if "tval" in df.columns:
        df["timestamp"] = pd.to_datetime(df["tval"], unit="s", utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif "Timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        raise ValueError(f"missing tval/timestamp column in {infile}")

    df = df.dropna(subset=["timestamp"])

    if "iaga" in df.columns:
        df["station"] = df["iaga"].astype(str)
    elif "source_station" in df.columns:
        df["station"] = df["source_station"].astype(str)
    else:
        raise ValueError(f"missing station column in {infile}")

    df["ext_sec"] = df["ext"] if "ext" in df.columns else pd.NA

    df["glon_deg"] = df["glon"] if "glon" in df.columns else pd.NA
    df["glat_deg"] = df["glat"] if "glat" in df.columns else pd.NA
    df["mlt_hour"] = df["mlt"] if "mlt" in df.columns else pd.NA
    df["mcolat_deg"] = df["mcolat"] if "mcolat" in df.columns else pd.NA
    df["decl_deg"] = df["decl"] if "decl" in df.columns else pd.NA
    df["sza_deg"] = df["sza"] if "sza" in df.columns else pd.NA

    df["dbn_nez_nt"] = flatten_component(df, "N", "nez")
    df["dbe_nez_nt"] = flatten_component(df, "E", "nez")
    df["dbz_nez_nt"] = flatten_component(df, "Z", "nez")

    df["dbn_geo_nt"] = flatten_component(df, "N", "geo")
    df["dbe_geo_nt"] = flatten_component(df, "E", "geo")
    df["dbz_geo_nt"] = flatten_component(df, "Z", "geo")

    if "source_station" not in df.columns:
        df["source_station"] = df["station"]

    if "source_url" not in df.columns:
        df["source_url"] = pd.NA

    df["source_file"] = str(infile)

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["station", "timestamp"], keep="first")

    return df[COLUMN_ORDER].reset_index(drop=True)


def parse_file_info(infile: Path) -> tuple[str, str, str]:
    m = FILENAME_RE.match(infile.name)
    if not m:
        raise ValueError(f"unexpected filename format: {infile.name}")

    station = m.group(1)
    yyyymmdd = m.group(2)
    year = yyyymmdd[:4]
    month = yyyymmdd[4:6]

    return station, year, month


def daily_output_path(infile: Path, input_dir: Path, output_dir: Path) -> Path:
    rel = infile.relative_to(input_dir)
    return output_dir / rel


def monthly_output_path(output_dir: Path, station: str, year: str, month: str) -> Path:
    out_dir = output_dir / station / year
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"supermag_{station}_{year}{month}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean raw SuperMAG parquet files into interim daily or monthly parquet files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/supermag"),
        help="Root directory containing raw daily SuperMAG parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/supermag"),
        help="Root directory for cleaned interim SuperMAG parquet files.",
    )
    parser.add_argument(
        "--clean-type",
        default="monthly",
        choices=["monthly", "daily"],
        help="Output cleaned daily files or monthly concatenated files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, overwrite existing cleaned output files.",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("supermag_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {args.input_dir}")

    if args.clean_type == "daily":
        for infile in files:
            try:
                outfile = daily_output_path(infile, args.input_dir, args.output_dir)

                if outfile.exists() and not args.force:
                    print(f"skip existing {outfile}")
                    continue

                df = clean_one_file(infile)
                outfile.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(outfile, index=False)

                print(f"cleaned daily {infile} -> {outfile} rows={len(df)}")

            except Exception as e:
                print(f"failed {infile} error={e}")

    elif args.clean_type == "monthly":
        grouped: dict[tuple[str, str, str], list[Path]] = {}

        for infile in files:
            try:
                station, year, month = parse_file_info(infile)
                grouped.setdefault((station, year, month), []).append(infile)
            except Exception as e:
                print(f"failed {infile} error={e}")

        for (station, year, month), month_files in grouped.items():
            try:
                outfile = monthly_output_path(args.output_dir, station, year, month)

                if outfile.exists() and not args.force:
                    print(f"skip existing {outfile}")
                    continue

                dfs: list[pd.DataFrame] = []
                for path in sorted(month_files):
                    try:
                        cleaned = clean_one_file(path)
                        if not cleaned.empty:
                            dfs.append(cleaned)
                    except Exception as file_error:
                        print(f"failed file {path} error={file_error}")

                if not dfs:
                    print(f"skip monthly station={station} year={year} month={month} (no valid daily files)")
                    continue

                out = pd.concat(dfs, ignore_index=True)

                out = out.sort_values("timestamp")
                out = out.drop_duplicates(subset=["station", "timestamp"], keep="first")
                out = out[COLUMN_ORDER].reset_index(drop=True)

                outfile.parent.mkdir(parents=True, exist_ok=True)
                out.to_parquet(outfile, index=False)

                print(f"wrote monthly {outfile} rows={len(out)}")

            except Exception as e:
                print(
                    f"failed monthly station={station} year={year} month={month} error={e}"
                )


if __name__ == "__main__":
    main()
