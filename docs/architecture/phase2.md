src/clean/interim/goes_mag.py
"""
Phase 2.2 - Clean Interim: GOES MAG Raw NetCDF to Monthly Parquet
------------------------------------------------------------------
This is the second Phase 2 cleaning script. It reads raw GOES MAG
NetCDF files produced by Phase 1.2 (src/download/goes_mag.py),
normalizes their structure into a consistent 1-minute tabular schema,
decomposes multi-dimensional vector field variables into scalar
columns, and writes cleaned monthly Parquet files partitioned by
satellite and year. The output of this script is required before
GOES MAG preprocessing in Phase 3.x can begin.

Purpose
-------
GOES MAG data is distributed in two incompatible formats depending
on satellite generation. GOES-R satellites (goes16, goes17, goes18)
provide "magn-l2-avg1m" files that are already at 1-minute cadence.
goes15 (legacy) provides "magn-l2-hires" files at a sub-minute
high-resolution cadence with inner-boom and outer-boom sensor columns
and a separate orbital position time coordinate (time_orbit) that is
not aligned to the main measurement time axis.

This script handles both formats through separate code paths:
- GOES-R files are opened directly and their vector variables
  (b_eci, b_epn, b_gse, b_gsm, b_brf, b_vdh, orbit_llr_geo) are
  decomposed into x/y/z or lat/lon/radius scalar columns.
- Legacy hires files are opened into a main measurement DataFrame
  and a separate orbit DataFrame, each resampled independently to
  1-minute means via resample_legacy_to_1min(), then joined. Flag
  and method columns are rounded to the nearest integer before
  being cast to nullable Int64, since resampled means of integer
  flags are floating-point.

The net result is a unified 1-minute schema across all satellites
and generations, with no gap-filling or interpolation applied beyond
the resampling of legacy hires data.

WORK FLOW
---------
Step 1: Parse CLI arguments - raw input directory, output directory,
        optional satellite filter list, and optional date range.
Step 2: Recursively discover all .nc files under --raw-dir, optionally
        filtered to the requested satellite subset.
Step 3: For each .nc file, detect source type from the filename
        (goesr_1min or legacy_hires). Skip files with unknown type.
Step 4a: If goesr_1min, open with xarray and extract b_total_nT,
         quality flags, DQF, and vector variables into scalar columns
         using add_vector_columns(). Cast flag columns to Int64.
Step 4b: If legacy_hires, open with xarray, extract main measurement
         and orbit DataFrames separately, resample both to 1-minute
         means, join on the 1-minute timestamp index, and round-cast
         flag and method columns to nullable Int64.
Step 5: Apply the optional date range filter to each per-file
         DataFrame before accumulating into a per-satellite frame list.
Step 6: For each satellite, concatenate all per-file frames and
         drop duplicate (timestamp, sat_id) pairs keeping the first
         occurrence after sorting by (timestamp, source_file).
Step 7: Partition the merged, sorted DataFrame by sat_id, year, and
         month, and write one Parquet file per partition to the output
         directory.

INPUT DATA
----------
- data/raw/goes_mag/**/*.nc
  Raw GOES MAG NetCDF files produced by Phase 1.2.
  Expected to be organized in per-satellite/year/month subdirectories
  but discovered recursively regardless of depth.
  Two file types are supported:
    - magn-l2-avg1m: GOES-R 1-minute averaged MAG (goes16/17/18)
    - magn-l2-hires: goes15 legacy high-resolution MAG
- CLI optional: --raw-dir    (default: data/raw/goes_mag)
               --output-dir (default: data/interim/goes_mag)
               --satellites (default: all found; e.g. goes15 goes16
                             goes17 goes18)
               --start-date (default: no lower bound; YYYY-MM-DD)
               --end-date   (default: no upper bound; YYYY-MM-DD)

OUTPUT DATA
-----------
- data/interim/goes_mag/{sat_id}/{YYYY}/{sat_id}_{YYYY}{MM}.parquet
  One Parquet file per satellite per month at 1-minute cadence.
  Schema includes timestamp, sat_id, source_file, source_type,
  b_total_nT, quality flags, and decomposed vector field scalar
  columns (e.g. b_epn_x, b_epn_y, b_epn_z, orbit_llr_geo_latitude_deg,
  etc.). Flag and method columns are typed as nullable Int64.

NEXT PIPELINE PHASE
-------------------
- Phase 2.2 has no dependency on any other Phase 2 script and can
  run independently once Phase 1.2 (src/download/goes_mag.py) has
  completed.
- After this script completes, the GOES MAG interim data is ready
  for Phase 3.x preprocessing (src/preprocess/).
- All other Phase 2.x interim cleaning scripts should also complete
  before moving on to Phase 3.x, since Phase 4 (fusion) requires
  all preprocessed sources to be present.
"""

---

src/clean/interim/omni.py
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

---

src/clean/interim/swarm.py
"""
Phase 2.4 - Clean Interim: Swarm Raw Daily Parquet to Interim Parquet
----------------------------------------------------------------------
This is the fourth Phase 2 cleaning script. It reads raw daily Swarm
Parquet files produced by Phase 1.3 (src/download/swarm.py), renames
columns from VirES API PascalCase to lowercase snake_case with units,
normalizes timestamps to UTC, deduplicates on timestamp, enforces a
fixed output column schema, and writes either cleaned daily files or
concatenated monthly Parquet files depending on the --clean-type
argument. The output of this script is required before Swarm
preprocessing in Phase 3.x can begin.

Purpose
-------
The raw Swarm Parquet files produced by Phase 1.3 use the VirES API
column naming convention (e.g. Timestamp, Latitude, B_NEC decomposed
into B_N, B_E, B_C). This script standardizes column names to the
pipeline convention (e.g. timestamp, latitude_deg, b_north_nT) and
enforces a fixed COLUMN_ORDER schema so that all downstream consumers
receive a predictable schema regardless of satellite.

The script offers two output modes:
- "monthly" (default): groups daily files by (satellite, year, month),
  concatenates them, deduplicates on timestamp, and writes one Parquet
  file per month. This is the mode expected by Phase 3.x.
- "daily": cleans each raw file in place, preserving the per-satellite
  year/month directory structure, without concatenation. This mode is
  useful for incremental or partial re-runs.

No resampling, interpolation, or gap-filling is applied. Per-file
cleaning is limited to column renaming, UTC timestamp normalization,
dropping rows with unparseable timestamps, and deduplication.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory, output directory,
        --clean-type (monthly or daily), and optional --force flag.
Step 2: Recursively discover all swarm_*.parquet files under
        --input-dir. Raise FileNotFoundError if none are found.
Step 3 (both modes): For each raw daily file, apply clean_one_file():
        rename columns via COLUMN_RENAME_MAP, normalize timestamp to
        UTC (utc=True to prevent tz-naive/tz-aware comparison errors),
        drop rows with unparseable timestamps, cast satellite to str,
        deduplicate on timestamp, and enforce COLUMN_ORDER.
Step 4a (daily mode): Write each cleaned file to the same relative
        path under --output-dir, mirroring the raw directory structure.
        Skip existing files unless --force is set.
Step 4b (monthly mode): Group cleaned daily files by (sat, year, month)
        parsed from the filename via FILENAME_RE. Concatenate all daily
        frames for each month, apply a second deduplication on timestamp
        across the combined frame, enforce COLUMN_ORDER, and write one
        Parquet file per month. Skip existing files unless --force is set.

INPUT DATA
----------
- data/raw/swarm/**/swarm_*.parquet
  Raw daily Swarm Parquet files produced by Phase 1.3.
  Expected filename pattern: swarm_{A|B|C}_{YYYYMMDD}.parquet
  Expected columns (VirES API names): Timestamp, Spacecraft,
  Latitude, Longitude, Radius, F, Flags_F, Flags_B, B_N, B_E, B_C,
  source_collection.
- CLI optional: --input-dir   (default: data/raw/swarm)
               --output-dir  (default: data/interim/swarm)
               --clean-type  (default: monthly; choices: monthly, daily)
               --force       (overwrite existing output files)

OUTPUT DATA
-----------
- Monthly mode (default):
  data/interim/swarm/{sat}/{YYYY}/swarm_{sat}_{YYYYMM}.parquet
  One Parquet file per satellite per month.
- Daily mode:
  data/interim/swarm/{sat}/{YYYY}/{MM}/swarm_{sat}_{YYYYMMDD}.parquet
  Mirrors raw input directory structure under --output-dir.
- Fixed output schema (COLUMN_ORDER): timestamp, satellite,
  latitude_deg, longitude_deg, radius_m, f_nT, flags_f, flags_b,
  b_north_nT, b_east_nT, b_center_nT, source_collection.

NEXT PIPELINE PHASE
-------------------
- Phase 2.4 has no dependency on any other Phase 2 script and can
  run independently once Phase 1.3 (src/download/swarm.py) has
  completed.
- Run with --clean-type monthly (the default) to produce the output
  expected by Phase 3.x preprocessing.
- After this script completes, the Swarm interim data is ready for
  Phase 3.x preprocessing (src/preprocess/).
- Note the critical ordering constraint in Phase 3.x: the Swarm
  preprocessing script swarm_1min.py must complete before
  swarm_chaos.py can run.
- All other Phase 2.x interim cleaning scripts should also complete
  before moving on to Phase 3.x, since Phase 4 (fusion) requires
  all preprocessed sources to be present.
"""

---

src/clean/interim/supermag.py
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

---

src/clean/interim/goes_xrs.py
"""
Phase 2.1 - Clean Interim: GOES XRS Raw NetCDF to Monthly Parquet
------------------------------------------------------------------
This is the first Phase 2 cleaning script. It reads raw GOES XRS
NetCDF files produced by Phase 1.1 (src/download/goes_xrs.py),
normalizes their structure into a consistent tabular schema, resolves
duplicate timestamps across overlapping source file types, and writes
cleaned monthly Parquet files partitioned by satellite and year. The
output of this script is required before GOES XRS preprocessing in
Phase 3.x can begin.

Purpose
-------
Raw GOES XRS data is distributed in two distinct file type formats:
"yearly" files (goes15, legacy) and "mission_span" files (goes16,
goes17, goes18, GOES-R series). These formats carry different column
schemas - GOES-R files include extended dual-channel fields, AU
correction factors, and attitude flags that are absent from goes15
files. Because satellite coverage can overlap across file types for
the same time period, this script implements a priority-based
deduplication strategy: for any duplicate (timestamp, satellite)
pair, "yearly" records take priority over "mission_span" records,
which take priority over all other file types.

The script does not interpolate, impute, or gap-fill any values.
Its role is purely structural: unify the NetCDF schema into a flat
DataFrame, cast flag and count columns to nullable Int64 to preserve
the distinction between a missing value and a zero, strip any
timezone information from timestamps, and partition the result into
monthly Parquet files for efficient downstream access.

WORK FLOW
---------
Step 1: Parse CLI arguments - raw input directory, output directory,
        optional satellite filter list, and optional date range.
Step 2: Recursively discover all .nc files under --raw-dir, optionally
        filtered to the requested satellite subset.
Step 3: For each .nc file, open it with xarray (decode_cf=True,
        mask_and_scale=True, decode_times=True), extract the time
        coordinate as a UTC-naive timestamp Series, infer satellite
        name and file type from the filename, and extract all present
        COMMON_COLUMNS and OPTIONAL_COLUMNS variables into a flat
        DataFrame. Cast flag and count columns to nullable Int64.
Step 4: Apply the optional date range filter to each per-file
        DataFrame before accumulating into a per-satellite frame list.
Step 5: For each satellite, concatenate all per-file frames and
        resolve duplicate (timestamp, satellite) pairs by preferring
        higher-priority source file types (yearly > mission_span >
        other) via merge_prefer_higher_priority().
Step 6: Partition the deduplicated, sorted DataFrame by satellite,
        year, and month, and write one Parquet file per partition to
        the output directory.

INPUT DATA
----------
- data/raw/goes_xrs/**/*.nc
  Raw GOES XRS NetCDF files produced by Phase 1.1.
  Expected to be organized in per-satellite subdirectories but
  discovered recursively regardless of depth.
- CLI optional: --raw-dir    (default: data/raw/goes_xrs)
               --output-dir (default: data/interim/goes_xrs)
               --satellites (default: all found; choices: goes15
                             goes16 goes17 goes18)
               --start-date (default: no lower bound; YYYY-MM-DD)
               --end-date   (default: no upper bound; YYYY-MM-DD)

OUTPUT DATA
-----------
- data/interim/goes_xrs/{satellite}/{YYYY}/{satellite}_{YYYY}{MM}.parquet
  One Parquet file per satellite per month. Schema includes
  COMMON_COLUMNS (timestamp, satellite, source_file,
  source_file_type, xrsa_flux, xrsb_flux, flux flags, and counts)
  plus whichever OPTIONAL_COLUMNS were present in the source files
  (extended channel fields, au_factor, roll_angle, attitude flags).
  Flag and count columns are typed as nullable Int64.

NEXT PIPELINE PHASE
-------------------
- Phase 2.1 has no dependency on any other Phase 2 script and can
  run independently once Phase 1.1 (src/download/goes_xrs.py) 
"""

---

