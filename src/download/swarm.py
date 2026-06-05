#!/usr/bin/env python3
"""
Phase 1.3 - Download: Swarm MAGx LR 1B Raw Data
------------------------------------------------
This is a Phase 1 download script. It fetches low-resolution Swarm
magnetometer data (MAGx_LR_1B) from the ESA VirES API and writes
one Parquet file per satellite per day to a local raw data directory.
It must run before any Swarm cleaning or preprocessing steps.

Purpose
-------
Swarm is a constellation of three ESA low-Earth-orbit satellites
(A, B, C) carrying fluxgate and absolute scalar magnetometers. The
MAGx_LR_1B product provides 1 Hz magnetic field measurements in the
NEC (North-East-Centre) frame along with quality flags and orbital
ancillary data. These raw daily Parquet files are the upstream
dependency for the Swarm interim cleaning stage (Phase 2.x) and,
subsequently, the two-step Swarm preprocessing stage (Phase 3.x),
where swarm_1min.py must run before swarm_chaos.py.

Unlike the GOES download scripts (Phase 1.1 and 1.2), this script
does not use curl to fetch files from an HTTP directory. It queries
the VirES API directly via the viresclient library, retrieves data
as a DataFrame, decomposes the B_NEC vector field into scalar columns
B_N, B_E, B_C in memory, appends a source_collection label, and
writes the result to Parquet. No --force flag is available; existing
daily files are skipped automatically.

WORK FLOW
---------
Step 1: Parse CLI arguments - satellite list, date range, and
        output directory.
Step 2: For each satellite (A, B, or C) and each day in the date
        range, construct the output file path and skip if it already
        exists.
Step 3: Submit a VirES API request via SwarmRequest for the
        corresponding MAGx_LR_1B collection, requesting measurements
        F, B_NEC, Flags_F, Flags_B and auxiliaries Latitude,
        Longitude, Radius, Spacecraft over a 24-hour window.
Step 4: Convert the API response to a DataFrame, decompose the
        B_NEC column into scalar columns B_N, B_E, B_C, and append
        a source_collection column identifying the collection name.
Step 5: Write the resulting DataFrame to a Parquet file under
        data/raw/swarm/{sat}/{YYYY}/{MM}/.

INPUT DATA
----------
- Remote: ESA VirES API (viresclient.SwarmRequest)
  Collections: SW_OPER_MAGA_LR_1B, SW_OPER_MAGB_LR_1B,
               SW_OPER_MAGC_LR_1B
- No local input files required
- CLI required: --start-date YYYY-MM-DD, --end-date YYYY-MM-DD
- CLI optional: --satellites (default: A B C, choices: A B C)
               --output-dir  (default: data/raw/swarm)

OUTPUT DATA
-----------
- data/raw/swarm/{sat}/{YYYY}/{MM}/swarm_{sat}_{YYYYMMDD}.parquet
  One file per satellite per day. Each file is a Parquet DataFrame
  with columns: Timestamp, F, B_N, B_E, B_C, Flags_F, Flags_B,
  Latitude, Longitude, Radius, Spacecraft, source_collection.

NEXT PIPELINE PHASE
-------------------
- This script has no dependency on any other download script and can
  run in parallel with other Phase 1.x download scripts.
- After all required satellite-days are downloaded, move on to
  Phase 2.x (src/clean/interim/) to clean the raw Swarm Parquet files.
- The clean/interim script expects files under data/raw/swarm/.
- Note the downstream ordering constraint: in Phase 3.x, the Swarm
  preprocessing script swarm_1min.py must complete before
  swarm_chaos.py can run.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
from viresclient import SwarmRequest


COLLECTIONS = {
    "A": "SW_OPER_MAGA_LR_1B",
    "B": "SW_OPER_MAGB_LR_1B",
    "C": "SW_OPER_MAGC_LR_1B",
}


MEASUREMENTS = ["F", "B_NEC", "Flags_F", "Flags_B"]
AUXILIARIES = ["Latitude", "Longitude", "Radius", "Spacecraft"]


def date_range(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    return list(pd.date_range(start, end, freq="D"))


def fetch_day(sat: str, day: pd.Timestamp) -> pd.DataFrame:
    request = SwarmRequest()
    request.set_collection(COLLECTIONS[sat])
    request.set_products(
        measurements=MEASUREMENTS,
        auxiliaries=AUXILIARIES,
    )

    start_time = day.strftime("%Y-%m-%dT00:00:00Z")
    end_time = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    data = request.get_between(
        start_time=start_time,
        end_time=end_time,
        asynchronous=False,
    )

    df = data.as_dataframe().reset_index()

    if "B_NEC" in df.columns:
        b = pd.DataFrame(df["B_NEC"].tolist(), columns=["B_N", "B_E", "B_C"])
        df = pd.concat([df.drop(columns=["B_NEC"]), b], axis=1)

    df["source_collection"] = COLLECTIONS[sat]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily Swarm MAGx_LR_1B parquet files from VirES.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--satellites", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/swarm"))
    args = parser.parse_args()

    for sat in args.satellites:
        for day in date_range(args.start_date, args.end_date):
            out_dir = args.output_dir / sat / f"{day:%Y}" / f"{day:%m}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"swarm_{sat}_{day:%Y%m%d}.parquet"

            if out_file.exists():
                continue

            try:
                df = fetch_day(sat, day)
                df.to_parquet(out_file, index=False)
                print(f"saved {out_file} rows={len(df)}")
            except Exception as e:
                print(f"failed sat={sat} day={day:%Y-%m-%d} error={e}")


if __name__ == "__main__":
    main()
