#!/usr/bin/env python3
"""
Phase 1.1 - Download: GOES X-Ray Sensor (XRS) Raw Data
-------------------------------------------------------
This is the first phase of the pipeline. It fetches raw GOES XRS solar
X-ray flux data in NetCDF (.nc) format from remote NOAA/NCEI HTTP
directories and writes the files to a local raw data directory. This
script must run before any GOES XRS cleaning or preprocessing steps.

Purpose
-------
GOES XRS measures solar X-ray flux in two bands and is the primary
source for detecting and characterizing solar flare events. The raw
.nc files downloaded here are the direct upstream dependency for the
GOES XRS interim cleaning stage (Phase 2.x). Without these files, no
solar flare proxy features can be built for the ML dataset.

WORK FLOW
---------
Step 1: Parse CLI arguments - satellite list, date range, output dir,
        and optional --force flag to re-download existing files.
Step 2: For each requested satellite, look up the remote base URL from
        GOES_SOURCES in src/download/goes_config.py.
Step 3: Fetch the directory listing at the base URL using curl and
        extract all .nc filenames from the HTML href attributes.
Step 4: Filter the .nc filenames to those whose embedded date range
        (mission-span _sYYYYMMDD_eYYYYMMDD_ or yearly _yYYYY_ pattern)
        intersects the requested start/end date window.
Step 5: Download each matched .nc file via curl into a per-satellite
        subdirectory under the output directory, skipping files that
        already exist unless --force is set.

INPUT DATA
----------
- Remote: NOAA/NCEI HTTP directories specified in
  src/download/goes_config.py under GOES_SOURCES[sat]["xray"]["base_url"]
- Config: src/download/goes_config.py (GOES_SOURCES dict)
- CLI required: --start-date YYYY-MM-DD, --end-date YYYY-MM-DD
- CLI optional: --satellites (default: goes15 goes16 goes17 goes18)
               --output-dir  (default: data/raw/goes_xrs)
               --force       (re-download files that already exist)

OUTPUT DATA
-----------
- data/raw/goes_xrs/{satellite}/*.nc
  One subdirectory per satellite; each file is a raw GOES XRS NetCDF
  file as served by NOAA/NCEI. No transformation is applied.

NEXT PIPELINE PHASE
-------------------
- This script has no dependency on any other download script and can
  run in parallel with other Phase 1.x download scripts.
- After all required satellites are downloaded, move on to Phase 2.x
  (src/clean/interim/) to clean the raw GOES XRS .nc files.
- The clean/interim script expects files under data/raw/goes_xrs/.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import subprocess
from datetime import date
import sys
import os
# note: this assumes you are running the script from parent dir of src
sys.path.append(os.getcwd())

from src.download.goes_config import GOES_SOURCES


MISSION_SPAN_RE = re.compile(r"_s(\d{8})_e(\d{8})_")
YEARLY_RE = re.compile(r"_y(\d{4})_")


def fetch_nc_names(url: str) -> list[str]:
    result = subprocess.run(
        ["curl", "-fsSL", url],
        check=True,
        capture_output=True,
        text=True,
    )
    html = result.stdout
    return sorted(set(re.findall(r'href="([^"]+\.nc)"', html)))


def intersects_file_range(filename: str, start_date: date, end_date: date) -> bool:
    m = MISSION_SPAN_RE.search(filename)
    if m:
        s = date.fromisoformat(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}")
        e = date.fromisoformat(f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:8]}")
        return not (e < start_date or s > end_date)

    m = YEARLY_RE.search(filename)
    if m:
        year = int(m.group(1))
        return start_date.year <= year <= end_date.year

    return False


def download(url: str, outdir: Path, force: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / url.rsplit("/", 1)[-1]

    if outfile.exists() and not force:
        print(f"skip existing {outfile}")
        return

    subprocess.run(["curl", "-fL", url, "-o", str(outfile)], check=True)
    print(f"downloaded {url} -> {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GOES XRS .nc files for a date range.")
    parser.add_argument("--satellites", nargs="+", default=["goes15", "goes16", "goes17", "goes18"])
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/goes_xrs"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    for sat in args.satellites:
        if sat not in GOES_SOURCES or "xray" not in GOES_SOURCES[sat]:
            print(f"skip {sat}: no xray config")
            continue

        base_url = GOES_SOURCES[sat]["xray"]["base_url"]

        try:
            names = fetch_nc_names(base_url)
            matched = [name for name in names if intersects_file_range(name, start_date, end_date)]

            if not matched:
                print(f"no xray matches for {sat}")
                continue

            for name in matched:
                url = base_url.rstrip("/") + "/" + name
                download(url, args.output_dir / sat, args.force)

        except Exception as e:
            print(f"failed satellite={sat} error={e}")


if __name__ == "__main__":
    main()
