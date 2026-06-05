#!/usr/bin/env python3
"""
Phase 1.2 - Download: GOES Magnetometer (MAG) Raw Data
-------------------------------------------------------
This is a Phase 1 download script. It fetches raw GOES onboard
magnetometer data in NetCDF (.nc) format from remote NOAA/NCEI HTTP
directories and writes the files to a local raw data directory. It
must run before any GOES MAG cleaning or preprocessing steps.

Purpose
-------
GOES MAG measures the geomagnetic field in the geostationary orbit
environment, providing a space-based proxy for solar wind driven
magnetospheric disturbance. The raw .nc files downloaded here are
the direct upstream dependency for the GOES MAG interim cleaning
stage (Phase 2.x). Unlike the XRS download (Phase 1.1), the remote
source organizes files by year and month, so this script traverses
month-by-month across the requested date range.

WORK FLOW
---------
Step 1: Parse CLI arguments - satellite list, date range, output dir,
        and optional --force flag to re-download existing files.
Step 2: For each requested satellite, look up the remote base URL from
        GOES_SOURCES in src/download/goes_config.py under the "mag" key.
Step 3: Iterate over each calendar month in the requested date range
        using iter_months() and construct a per-month remote URL of
        the form {base_url}/{YYYY}/{MM}/.
Step 4: Fetch the directory listing for each month URL using curl and
        extract all .nc filenames from the HTML href attributes.
Step 5: Filter filenames to those whose embedded daily date field
        (pattern d{YYYYMMDD}) falls within the requested date range.
Step 6: Download each matched .nc file via curl into a per-satellite,
        per-year, per-month subdirectory, skipping files that already
        exist unless --force is set.

INPUT DATA
----------
- Remote: NOAA/NCEI HTTP directories organized as {base_url}/{YYYY}/{MM}/
  specified in src/download/goes_config.py under
  GOES_SOURCES[sat]["mag"]["base_url"]
- Config: src/download/goes_config.py (GOES_SOURCES dict)
- CLI required: --start-date YYYY-MM-DD, --end-date YYYY-MM-DD
- CLI optional: --satellites (default: goes15 goes16 goes17 goes18)
               --output-dir  (default: data/raw/goes_mag)
               --force       (re-download files that already exist)

OUTPUT DATA
-----------
- data/raw/goes_mag/{satellite}/{YYYY}/{MM}/*.nc
  One subdirectory tree per satellite, partitioned by year and month.
  Each file is a raw GOES MAG NetCDF file as served by NOAA/NCEI.
  No transformation is applied.

NEXT PIPELINE PHASE
-------------------
- This script has no dependency on any other download script and can
  run in parallel with other Phase 1.x download scripts.
- After all required satellites are downloaded, move on to Phase 2.x
  (src/clean/interim/) to clean the raw GOES MAG .nc files.
- The clean/interim script expects files under data/raw/goes_mag/.
"""

from __future__ import annotations

import argparse
import calendar
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

# note: this assumes you are running the script from parent dir of src
sys.path.append(os.getcwd())

from src.download.goes_config import GOES_SOURCES

DAILY_RE = re.compile(r"d(\d{8})")


def iter_months(start_date: date, end_date: date):
    y, m = start_date.year, start_date.month
    while (y, m) <= (end_date.year, end_date.month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def fetch_nc_names(url: str) -> list[str]:
    result = subprocess.run(
        ["curl", "-fsSL", url],
        check=True,
        capture_output=True,
        text=True,
    )
    html = result.stdout
    return sorted(set(re.findall(r'href="([^"]+\.nc)"', html)))


def in_range(filename: str, start_date: date, end_date: date) -> bool:
    m = DAILY_RE.search(filename)
    if not m:
        return False
    d = m.group(1)
    day = date.fromisoformat(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
    return start_date <= day <= end_date


def download(url: str, outdir: Path, force: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / url.rsplit("/", 1)[-1]

    if outfile.exists() and not force:
        print(f"skip existing {outfile}")
        return

    subprocess.run(["curl", "-fL", url, "-o", str(outfile)], check=True)
    print(f"downloaded {url} -> {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GOES MAG .nc files for a date range."
    )
    parser.add_argument(
        "--satellites", nargs="+", default=["goes15", "goes16", "goes17", "goes18"]
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/goes_mag"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    for sat in args.satellites:
        if sat not in GOES_SOURCES or "mag" not in GOES_SOURCES[sat]:
            print(f"skip {sat}: no mag config")
            continue

        base_url = GOES_SOURCES[sat]["mag"]["base_url"]

        try:
            for year, month in iter_months(start_date, end_date):
                month_url = f"{base_url.rstrip('/')}/{year:04d}/{month:02d}/"
                try:
                    names = fetch_nc_names(month_url)
                except subprocess.CalledProcessError:
                    continue

                matched = [
                    name for name in names if in_range(name, start_date, end_date)
                ]
                for name in matched:
                    url = month_url + name
                    # Update outdir to include sat, year, and month subdirectories
                    sat_outdir = args.output_dir / sat / f"{year:04d}" / f"{month:02d}"
                    download(url, sat_outdir, args.force)

        except Exception as e:
            print(f"failed satellite={sat} error={e}")


if __name__ == "__main__":
    main()
