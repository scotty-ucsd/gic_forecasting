src/download/omni.md

---

src/download/goes_mag.py
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

---

src/download/goes_config.py
"""
Phase 1 Support - Config: GOES Source URL Registry
---------------------------------------------------
This is a shared support module for the Phase 1 GOES download scripts.
It is not a pipeline entrypoint and does not execute independently.
It exports the GOES_SOURCES dict, which is imported by goes_xrs.py
(Phase 1.1) and goes_mag.py (Phase 1.2) to resolve per-satellite
remote base URLs and filename patterns.

Purpose
-------
Centralizes all GOES satellite remote URL configuration in one place
so that goes_xrs.py and goes_mag.py do not hardcode source paths.
Covers four satellites across two hardware generations: goes15
(legacy) and goes16, goes17, goes18 (GOES-R series). Each satellite
entry defines base URLs and filename patterns for both the XRS and
MAG instrument types.

WORK FLOW
---------
Step 1: Imported by goes_xrs.py or goes_mag.py at module load time.
Step 2: Caller accesses GOES_SOURCES[sat]["xray"] or
        GOES_SOURCES[sat]["mag"] to retrieve base_url and filename
        pattern for the requested satellite and instrument.

INPUT DATA
----------
- None. This module is a static configuration definition.

OUTPUT DATA
-----------
- Exports: GOES_SOURCES (dict)
  Keyed by satellite name: goes15, goes16, goes17, goes18.
  Each entry contains "xray" and "mag" sub-dicts with fields:
    - base_url: remote NOAA/NCEI HTTP directory root
    - mode: one of "yearly", "mission_span", "daily_hierarchical"
    - filename or filename_hint: expected .nc filename pattern

NEXT PIPELINE PHASE
-------------------
- This file has no next phase of its own.
- It must be present and correct before running Phase 1.1
  (goes_xrs.py) or Phase 1.2 (goes_mag.py).
- If a satellite or instrument key is missing from GOES_SOURCES,
  the corresponding download script will skip that satellite silently.
"""

---

src/download/swarm.py
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

---

src/download/supermag.py
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
    """
    Yield monthly chunks between start_date and end_date.

    Example:
        2014-01-15 to 2014-03-10 yields:
            2014-01-15 to 2014-01-31
            2014-02-01 to 2014-02-28
            2014-03-01 to 2014-03-10
    """
    """
    Fetch one station/day with retry handling.

    N_RETRIES means retries after the initial attempt.
    So N_RETRIES = 3 gives up to 4 total attempts.
    """

---

src/download/goes_xrs.py
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

---

