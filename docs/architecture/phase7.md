"""
Phase 7.1 - Features: Build Standardized Feature Table from Labeled Data
-------------------------------------------------------------------------
This is the sole Phase 7 script. It reads monthly labeled fused
Parquet files produced by Phase 6.2 (src/targets/attach_station_targets.py),
renames all source columns into a strict four-family prefix schema
(sun_, l1_, geo_, leo_), computes 20 derived engineered features
(5 per source family), and writes standardized monthly feature
Parquet files to data/preprocessed/fused/standardized/. After all
months are processed, a metadata JSON summarizing the feature schema
is written to the same root. This script replaces an earlier
multi-regime feature-building design; the filename retains the
"regimes" label for historical continuity.

Purpose
-------
The labeled fused data from Phase 6.2 carries source-prefixed column
names from Phase 5.1 (supermag_*, omni_*, goes_mag_*, xrs_*, swarm_*)
that do not yet reflect the four domain-oriented source families used
in modeling. This script applies a second rename layer to produce
a consistent physical-domain naming scheme:

  sun_   <- xrs_*       (GOES X-ray flux; solar activity)
  l1_    <- omni_*      (OMNI L1 solar wind and IMF)
  geo_   <- goes_mag_*  (GOES MAG geostationary magnetic field)
  leo_   <- swarm_*     (Swarm CHAOS LEO residuals)

Ground station columns (supermag_*) and context columns
(glon_deg, glat_deg, mlt_hour, mcolat_deg, decl_deg, sza_deg)
are preserved without renaming.

Any source column absent from a given monthly file is silently
skipped rather than raising an error. Any BASE_FEATURES column
that is absent after renaming is filled with null Float64 to ensure
a consistent output schema across all months.

Derived features (20 total, 5 per source family)
-------------------------------------------------
All derived features are backward-looking or contemporaneous only.
No forward-looking transforms are applied. Shift-based time
differences use .over(station) to compute per-station windows.

Sun (5):
- sun_log_flux_ratio:       log(xrsb + 1e-12) - log(xrsa + 1e-12)
- sun_xrsb_diff_10m:        xrsb - xrsb.shift(10) per station
- sun_xrsa_diff_10m:        xrsa - xrsa.shift(10) per station
- sun_xrs_sum_flux:         xrsb + xrsa
- sun_xrs_hardness_proxy:   xrsb / (xrsa + 1e-12)

L1 (5):
- l1_clock_angle_rad:           arctan2(By_gsm, Bz_gsm)
- l1_newell_like:               speed^(4/3) * Byz^(2/3) *
                                 sin(|clock_angle|/2)^(8/3)
                                 (approximation of Newell coupling function)
- l1_dynamic_pressure_speed:    flow_pressure * speed
- l1_bmag_pressure_ratio:       Bt / (flow_pressure + 1e-12)
- l1_electric_field_proxy:      speed * |Bz_gsm|

GEO (5):
- geo_bz_minus_vdh:             Bz_gsm - Bz_vdh
- geo_b_transverse_gsm:         sqrt(Bx_gsm^2 + By_gsm^2)
- geo_b_transverse_vdh_proxy:   |Bz_gsm| + |Bz_vdh|
- geo_bz_gsm_diff_15m:          Bz_gsm - Bz_gsm.shift(15) per station
- geo_bz_vdh_diff_15m:          Bz_vdh - Bz_vdh.shift(15) per station

LEO (5):
- leo_fac_cond_proxy:                   north_residual / (1 + |sza_deg|)
- leo_residual_horizontal_mag:          sqrt(north^2 + east^2)
- leo_residual_vertical_to_horizontal_ratio: |center| / (horizontal + 1e-12)
- leo_north_diff_10m:                   north - north.shift(10) per station
- leo_freshness_weighted_residual:      north / (1 + |freshness_min|)

Each derived feature group is only computed when all required base
columns are present in the schema. Groups with missing dependencies
are silently skipped; the output column will be absent for those
months.

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, optional
        --months filter, --force flag.
Step 2: Discover all station_time_labeled_*.parquet files under
        --input-root, optionally filtered by --months.
Step 3: For each monthly file, call build_month():
  Step 3a: Skip if output exists and --force is not set.
  Step 3b: Scan the file via scan_base_month(): validate required
           columns (station, timestamp, all six TARGET_COLS),
           select IDENTITY_COLS + TARGET_COLS + CONTEXT_COLS +
           GROUND_KEEP_COLS + rename-map source columns, apply
           source-family renames via RENAME_MAP, fill absent
           BASE_FEATURES with null Float64.
  Step 3c: Add time context columns (year Int16, month YYYY-MM,
           yyyymm YYYYMM) if not already present.
  Step 3d: Compute 20 derived features via add_derived_features()
           using conditional schema checks per feature group.
  Step 3e: Select final ordered columns via choose_final_columns():
           metadata -> context -> ground -> base features -> derived
           features -> target columns (deduplication preserving order).
  Step 3f: Collect summary stats (rows, stations) via streaming,
           then write via lf.sink_parquet() to
           data/preprocessed/fused/standardized/{YYYY}/
           station_time_features_{YYYYMM}.parquet.
Step 4: After all months, call write_metadata() once to write
        data/preprocessed/fused/standardized/metadata/feature_columns.json
        with schema_name, description, target_columns, base_features,
        per-family derived lists, and selected_columns from the last
        written file.

INPUT DATA
----------
- data/fused/labeled/{YYYY}/station_time_labeled_{YYYYMM}.parquet
  Phase 6.2 output. Required columns: station, timestamp,
  and all six target_geq_* columns.
- CLI optional: --input-root  (default: data/fused/labeled)
               --output-root (default: data/preprocessed/fused/standardized)
               --months      (default: all found)
               --force       (overwrite existing output files)

OUTPUT DATA
-----------
- data/preprocessed/fused/standardized/{YYYY}/station_time_features_{YYYYMM}.parquet
  One Parquet file per month. Column order: timestamp, station,
  year, month, yyyymm, context columns (supermag position/time),
  ground columns (supermag_dbn/dbe/dbz_nt, missingness flags,
  has_observation), 21 base feature columns (across sun/l1/geo/leo),
  up to 20 derived feature columns (across sun/l1/geo/leo), 6 target
  columns. Written via sink_parquet() (Polars streaming write).
- data/preprocessed/fused/standardized/metadata/feature_columns.json
  Written once after all months complete. Fields: schema_name,
  description, target_columns, base_features, sun_derived,
  l1_derived, geo_derived, leo_derived, selected_columns.

NEXT PIPELINE PHASE
-------------------
- Phase 7.1 depends on Phase 6.2 (src/targets/attach_station_targets.py)
  completing first for all months to be processed.
- The second script in src/features/ (evaluate_feature_importance.py)
  is NOT part of Phase 7. It depends on Phase 8 model outputs and
  runs after Phase 8 completes.
- The output of this script (data/preprocessed/fused/standardized/)
  is the primary input for Phase 8 model training.
"""
