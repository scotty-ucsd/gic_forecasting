src/clean/fused/clean_fused_station_times.py
"""
Phase 5.1 - Clean Fused: Station-Time Master to Interim Cleaned Parquet
------------------------------------------------------------------------
This is the sole Phase 5 script and the only script in src/clean/fused/.
It reads the monthly station-time master Parquet files produced by
Phase 4.1 (src/fusion/data_fusion.py), enforces a canonical
source-prefixed column schema via RENAME_MAP, ensures a gapless
1-minute time grid per station, strips join artifacts and redundant
columns, and writes cleaned monthly Parquet files to
data/fused/interim/. This script does NOT create classification
targets; the output filename suffix "_plus_target" reserves space
for a downstream target-generation script to append target columns.

Purpose
-------
The Phase 4.1 fusion output carries two schema issues that must be
resolved before modeling:

1. Mixed naming conventions: Phase 4.1 columns arrive under a mix
   of naming schemes (bare column names from SuperMAG, prefixed
   aggregation names from GOES such as "gmag_b_gsm_x", freshness
   columns as "goes_xrs_freshness_min", etc.). RENAME_MAP enforces
   a strict {source}_feature naming convention across all columns:
   SuperMAG columns gain a "supermag_" prefix, OMNI columns gain an
   "omni_" prefix, GOES MAG columns are renamed from "gmag_*" to
   "goes_mag_*", and XRS freshness/missing columns are renamed from
   "goes_xrs_*" to "xrs_*". Two historical naming variants for GOES
   MAG vectors are both mapped to the canonical "goes_mag_*" names.

2. Join artifacts: The as-of join in Phase 4.1 leaves four source
   timestamp columns (omni_timestamp, goes_xrs_timestamp,
   goes_mag_timestamp, swarm_timestamp), the "yyyymm" redundant
   temporal column, and (after buffer processing) an "_is_buffer"
   flag column. These are all dropped by clean_schema().

Additionally, resolve_supermag_frame() verifies that the three
canonical preferred-value columns (supermag_dbn_nt, supermag_dbe_nt,
supermag_dbz_nt) are present, then drops all raw per-frame component
columns (anything ending in "_nez_nt" or "_geo_nt"). These per-frame
columns are redundant once Phase 3.6 has resolved the preferred frame.

The script also enforces a gapless per-station 1-minute time grid via
_enforce_uniform_grid_segment(), which inserts NaN-filled rows for any
missing minutes and emits runtime warnings (not errors) for each gap.
Timestamps are stripped of timezone metadata to produce naive UTC
(tz_convert(None)) to prevent tz-aware/tz-naive comparison failures
downstream.

Month-edge buffer loading
-------------------------
To support continuity checks at the end of each month, load_with_buffer()
optionally reads up to --lookahead-min minutes (default 120) from the
start of the next month's file. However, the buffer rows are NOT
written to the output. They are loaded and grid-enforced solely to
ensure that the station grid at the month boundary is well-formed,
and are then filtered out (combined[combined["_is_buffer"] == False])
before the cleaned DataFrame is returned. The final output contains
only rows belonging to the current month. If no next-month file
exists (e.g. the last available month in the dataset), the current
month is processed without a buffer and a coverage note is logged.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory, output directory,
        --lookahead-min (default 120), and --force flag.
Step 2: Discover all station_time_master_*.parquet files under
        --input-dir (recursively). Raise FileNotFoundError if none.
Step 3: For each monthly file, skip if output exists and --force
        is not set.
Step 4: Probe the file after _rename_columns() to confirm required
        columns (station, timestamp, supermag_dbn_nt, supermag_dbe_nt,
        supermag_dbz_nt) are present. Raise ValueError if not.
Step 5: Call load_with_buffer():
        - Load and rename current month, coerce timestamps to naive
          UTC, deduplicate on (station, timestamp), clip to
          [month_start, month_end].
        - Enforce uniform 1-minute grid per station for the current
          month, marking all rows _is_buffer=False.
        - If next-month file exists, load its first --lookahead-min
          minutes, enforce uniform grid marking rows _is_buffer=True.
        - Concatenate current and buffer, deduplicate, then filter
          to _is_buffer=False only.
Step 6: Call clean_schema():
        - Drop JOIN_TIMESTAMPS (four source timestamp columns).
        - Drop REDUNDANT_TEMPORAL (yyyymm).
        - Drop DROP_IF_PRESENT (_is_buffer).
        - Call resolve_supermag_frame(): verify canonical SuperMAG
          preferred-value columns exist, drop all *_nez_nt and
          *_geo_nt raw frame columns.
Step 7: Write to
        data/fused/interim/{YYYY}/station_time_reduced_plus_target_{YYYYMM}.parquet
        using engine="pyarrow", compression="snappy".

INPUT DATA
----------
- data/fused/station_time_master/{YYYY}/station_time_master_{YYYYMM}.parquet
  Monthly station-time master Parquet files produced by Phase 4.1.
  All source columns present; mixed naming conventions; tz-aware
  timestamps; join timestamp columns present.
- CLI optional: --input-dir     (default: data/fused/station_time_master)
               --output-dir    (default: data/fused/interim)
               --lookahead-min (default: 120)
               --force         (overwrite existing output files)

OUTPUT DATA
-----------
- data/fused/interim/{YYYY}/station_time_reduced_plus_target_{YYYYMM}.parquet
  One Parquet file per month. Current-month rows only (no buffer).
  Timestamps are naive UTC. Schema enforces strict {source}_feature
  naming via RENAME_MAP. Raw per-frame SuperMAG component columns
  dropped; preferred-frame supermag_dbn_nt/dbe_nt/dbz_nt retained.
  Join timestamp columns, yyyymm, and _is_buffer columns dropped.
  Written with pyarrow engine and snappy compression.
  The "_plus_target" filename suffix reserves space for a downstream
  target-generation script; this script does not write any target
  columns.

NEXT PIPELINE PHASE
-------------------
- Phase 5.1 depends only on Phase 4.1 (src/fusion/data_fusion.py)
  completing first.
- This is the sole Phase 5 script and the final cleaning step in
  the pipeline.
- The output of this script is the primary input for downstream
  target generation and modeling. A subsequent target-generation
  script is expected to append classification or regression target
  columns to these files, leveraging the "_plus_target" filename
  reservation.
"""

---

