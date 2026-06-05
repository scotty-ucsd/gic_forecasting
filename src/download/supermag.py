#!/usr/bin/env python3
"""
Phase 1.4 - Download: SuperMAG Ground Station Raw Data
-------------------------------------------------------
This is the final Phase 1 download script. It fetches per-station
daily ground magnetometer data from the SuperMAG JSON REST API and
writes one Parquet file per station per day to a local raw data
directory. It must run before any SuperMAG cleaning or preprocessing
steps, and requires a registered SuperMAG user ID for API access.

Purpose
-------
SuperMAG is a global network of ground-based magnetometer stations
that measure perturbations in the horizontal magnetic field components
(B_N, B_E) caused by geomagnetic disturbances such as substorms and
geomagnetic storms. These per-station daily records are the primary
upstream source for the station-level targets that the pipeline
ultimately labels as geomagnetic disturbance events (Phase 6.x).

Unlike the GOES download scripts (Phase 1.1 and 1.2), this script
does not use curl, and unlike the Swarm script (Phase 1.3), it does
not use a dedicated client library. It queries the SuperMAG REST API
directly via urllib, processes JSON responses into DataFrames, and
applies a polite rate-limit sleep and retry policy with backoff to
handle transient API errors. Failures are accumulated rather than
halting the run, and a summary is printed at the end.

The --flags argument controls which SuperMAG data families are
appended as query parameters. The default value "all" expands to
mlt, mag, geo, decl, and sza. The baseline=all flag is explicitly
dropped by flags_to_query() and not forwarded to the API.

WORK FLOW
---------
Step 1: Parse CLI arguments - station list, date range, userid,
        flags, output directory, and optional --force flag.
        Raise ValueError immediately if userid is not provided via
        --userid or the SUPERMAG_USERID environment variable.
Step 2: Partition the full date range into calendar month chunks
        using month_chunks() to manage request volume.
Step 3: For each month chunk, iterate over each station and each day.
        Skip existing output files unless --force is set.
Step 4: For each station/day, construct the SuperMAG API request URL
        using build_url() with the decoded flags, and fetch the JSON
        response via fetch_day_with_retries(), which applies up to
        N_RETRIES + 1 total attempts with exponential backoff on
        retryable HTTP and network errors.
Step 5: Parse the JSON response into a DataFrame, append
        source_station and source_url columns, and write to Parquet.
Step 6: Sleep REQUEST_SLEEP_SECONDS after each day fetch to respect
        API rate limits. Collect failures without halting the run and
        print a summary on completion.

INPUT DATA
----------
- Remote: SuperMAG JSON REST API
  BASE_URL: https://supermag.jhuapl.edu/services/data-api.php
- Auth: --userid or SUPERMAG_USERID environment variable (required)
- CLI required: --start-date YYYY-MM-DD, --end-date YYYY-MM-DD
- CLI optional: --stations   (default: YKC MEA SOD ABK)
               --userid     (default: SUPERMAG_USERID env var)
               --flags      (default: "all"; expands to mlt mag geo
                             decl sza; baseline=all is dropped)
               --output-dir (default: data/raw/supermag)
               --force      (re-download files that already exist)

OUTPUT DATA
-----------
- data/raw/supermag/{STATION}/{YYYY}/{MM}/supermag_{STATION}_{YYYYMMDD}.parquet
  One file per station per day. Each file is a Parquet DataFrame
  containing magnetic field components, coordinate fields, quality
  flags, and appended columns source_station and source_url.

NEXT PIPELINE PHASE
-------------------
- This script has no dependency on any other download script and can
  run in parallel with other Phase 1.x download scripts.
- OMNI solar wind data (Phase 1.x) is not downloaded by a script in
  this repository; it is retrieved manually from:
  https://omniweb.gsfc.nasa.gov/form/omni_min_def.html
  and must be placed under the expected raw data directory before
  Phase 2.x cleaning begins.
- After all required station-days are downloaded and OMNI data is
  manually staged, move on to Phase 2.x (src/clean/interim/) to
  clean all raw source data including SuperMAG Parquet files.
- The clean/interim script expects files under data/raw/supermag/.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd


BASE_URL = "https://supermag.jhuapl.edu/services/data-api.php"
DEFAULT_STATIONS = ["YKC", "MEA", "SOD", "ABK"]

REQUEST_TIMEOUT_SECONDS = 120
REQUEST_SLEEP_SECONDS = 2.0

N_RETRIES = 1
RETRY_BACKOFF_SECONDS = 10.0


def date_range(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if end < start:
        raise ValueError(f"end date {end:%Y-%m-%d} is before start date {start:%Y-%m-%d}")

    return list(pd.date_range(start, end, freq="D"))


def month_chunks(start_date: str, end_date: str):
    """
    Yield monthly chunks between start_date and end_date.

    Example:
        2014-01-15 to 2014-03-10 yields:
            2014-01-15 to 2014-01-31
            2014-02-01 to 2014-02-28
            2014-03-01 to 2014-03-10
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if end < start:
        raise ValueError(f"end date {end:%Y-%m-%d} is before start date {start:%Y-%m-%d}")

    current = start.replace(day=1)

    while current <= end:
        month_start = current
        month_end = current + pd.offsets.MonthEnd(0)

        chunk_start = max(start, month_start)
        chunk_end = min(end, month_end)

        yield chunk_start, chunk_end

        current = current + pd.offsets.MonthBegin(1)


def flags_to_query(flags: str) -> str:
    parts = []

    for flag in flags.split(","):
        flag = flag.strip().lower()
        if not flag:
            continue

        if flag == "all":
            parts.extend(["mlt", "mag", "geo", "decl", "sza"])
        elif flag == "baseline=all":
            continue
        else:
            parts.append(flag)

    return "".join(f"&{part}" for part in parts)


def build_url(userid: str, station: str, day: pd.Timestamp, flags: str) -> str:
    start = urllib.parse.quote(day.strftime("%Y-%m-%dT00:00"), safe="")
    logon = urllib.parse.quote(userid, safe="")
    station = urllib.parse.quote(station.upper(), safe="")

    return (
        f"{BASE_URL}?python&nohead"
        f"&start={start}"
        f"&logon={logon}"
        f"&extent={86400:012d}"
        f"{flags_to_query(flags)}"
        f"&station={station}"
    )


def fetch_day(userid: str, station: str, day: pd.Timestamp, flags: str) -> pd.DataFrame:
    url = build_url(userid, station, day, flags)

    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        text = response.read().decode("utf-8").strip()

    if not text or text == "OK":
        return pd.DataFrame()

    if text.startswith("ERROR"):
        raise RuntimeError(text)

    data = json.loads(text)
    df = pd.DataFrame(data)

    df["source_station"] = station.upper()
    df["source_url"] = url

    return df


def fetch_day_with_retries(
    userid: str,
    station: str,
    day: pd.Timestamp,
    flags: str,
) -> pd.DataFrame:
    """
    Fetch one station/day with retry handling.

    N_RETRIES means retries after the initial attempt.
    So N_RETRIES = 3 gives up to 4 total attempts.
    """
    total_attempts = N_RETRIES + 1
    last_error: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        try:
            return fetch_day(userid, station, day, flags)

        except RuntimeError:
            # SuperMAG returned an explicit ERROR.
            # Usually this will not be fixed by retrying.
            raise

        except urllib.error.HTTPError as e:
            last_error = e

            # Retry only common temporary/server-side HTTP errors.
            if e.code not in {429, 500, 502, 503, 504}:
                raise

        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            ConnectionResetError,
            json.JSONDecodeError,
            OSError,
        ) as e:
            last_error = e

        if attempt >= total_attempts:
            break

        wait = RETRY_BACKOFF_SECONDS * attempt
        print(
            f"retry station={station.upper()} day={day:%Y-%m-%d} "
            f"attempt={attempt}/{total_attempts} wait={wait:.1f}s error={last_error}"
        )
        time.sleep(wait)

    raise RuntimeError(
        f"failed after {total_attempts} attempts "
        f"station={station.upper()} day={day:%Y-%m-%d} error={last_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily SuperMAG station data.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    parser.add_argument("--userid", default=os.environ.get("SUPERMAG_USERID"))
    parser.add_argument("--flags", default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/supermag"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.userid:
        raise ValueError("missing --userid or SUPERMAG_USERID environment variable")

    failures = []

    for chunk_start, chunk_end in month_chunks(args.start_date, args.end_date):
        print(f"\nstarting chunk {chunk_start:%Y-%m-%d} to {chunk_end:%Y-%m-%d}")

        for station in args.stations:
            station = station.upper()

            for day in date_range(chunk_start, chunk_end):
                out_dir = args.output_dir / station / f"{day:%Y}" / f"{day:%m}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"supermag_{station}_{day:%Y%m%d}.parquet"

                if out_file.exists() and not args.force:
                    print(f"skip existing {out_file}")
                    continue

                try:
                    df = fetch_day_with_retries(args.userid, station, day, args.flags)
                    df.to_parquet(out_file, index=False)
                    print(f"saved {out_file} rows={len(df)}")

                except Exception as e:
                    print(f"failed station={station} day={day:%Y-%m-%d} error={e}")
                    failures.append((station, day.strftime("%Y-%m-%d"), str(e)))

                time.sleep(REQUEST_SLEEP_SECONDS)

    if failures:
        print("\ncompleted with failures:")
        for station, day, error in failures:
            print(f"  station={station} day={day} error={error}")
    else:
        print("\ncompleted with no failures")


if __name__ == "__main__":
    main()
