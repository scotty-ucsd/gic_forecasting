src/preprocess/goes_mag.py
"""
Phase 3.5 - Preprocess: GOES MAG Interim Parquet to Preprocessed Parquet
-------------------------------------------------------------------------
This is the fifth Phase 3 preprocessing script. It reads monthly
interim GOES MAG Parquet files produced by Phase 2.2
(src/clean/interim/goes_mag.py), enforces a gapless 1-minute time
index, normalizes orbital position column naming variants, harmonizes
generation-specific quality column names, applies physical bounds
validation to orbital position values, appends a generation label
and a composite validity flag, standardizes column types, and writes
preprocessed monthly Parquet files to data/preprocessed/goes_mag/.
This script has no dependency on any other Phase 3 script and can
run in parallel with Phase 3.1 through 3.4.

Purpose
-------
GOES MAG interim data carries structural differences between the
legacy goes15 format and the GOES-R format (goes16/17/18) that must
be resolved before fusion. This script resolves three categories of
cross-generation schema inconsistency:

1. Orbit column naming: GOES-R files carry orbit position under
   "orbit_{lon/lat/radius}" names. Legacy files use the longer
   "orbit_llr_geo_{lon/lat/radius}" variant. pick_orbit_columns()
   promotes the llr_geo names to the canonical short names when the
   canonical name is absent, ensuring a consistent orbit column set
   regardless of generation.

2. Quality column naming: GOES-R files use "b_quality" or "dqf";
   legacy files use "quality". harmonize_quality() maps whichever
   is present to a single "quality_summary" column typed as Int64,
   or assigns pd.NA if neither is found.

3. Orbital position bounds: sanitize_orbit() applies conservative
   physical range checks and nulls out-of-range values: radius must
   fall within the GEO sanity band [4.0e7, 4.3e7] meters; longitude
   within [-180, 180] degrees; latitude within [-90, 90] degrees.
   Values outside these bounds are set to NaN rather than raising.

Like Phase 3.4 (goes_xrs.py), this script applies minute_reindex()
to construct a gapless 1-minute pd.date_range across each file and
forward/back fills sat_id over the introduced NaN rows. The mag_valid
boolean column is True for any row where at least one of the six core
GSM/VDH vector components or b_total_nT is non-null. Provenance
columns source_file and source_type are dropped before output. No
--force flag is present; existing output files are unconditionally
overwritten.

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, and optional
        satellite filter list.
Step 2: Recursively discover all Parquet files under --input-root,
        optionally filtered by satellite name inferred from the
        two-levels-up directory name.
Step 3: For each file, apply ensure_sat_id(): assign sat_id from the
        directory name if the column is absent.
Step 4: Drop interim-only provenance columns (source_file,
        source_type) and parse the timestamp column.
Step 5: Apply minute_reindex(): construct a complete 1-minute
        pd.date_range and reindex, introducing NaN rows for missing
        minutes. Forward/back fill sat_id over introduced NaN rows.
Step 6: Apply pick_orbit_columns(): promote orbit_llr_geo_* column
        names to the canonical orbit_* names if the canonical name
        is absent.
Step 7: Apply harmonize_quality(): map b_quality (GOES-R) or quality
        (legacy) to a unified quality_summary column.
Step 8: Assign the mag_source label ("goesr" or "legacy") via
        build_mag_source() based on presence of b_quality or dqf.
Step 9: Retain CORE_VECTOR_COLS and present OPTIONAL_COLS; fill any
        absent CORE_VECTOR_COLS with np.nan.
Step 10: Apply sanitize_orbit(): null orbit radius, longitude, and
         latitude values that fall outside conservative physical bounds.
Step 11: Compute mag_valid boolean: True if any core vector component
         or b_total_nT is non-null for the row.
Step 12: Enforce ordered output column list, drop entirely-null non-
         protected columns, apply standardize_dtypes() (field vectors
         and orbit -> float32, quality/count columns -> Int64,
         mag_valid -> boolean), and write to
         data/preprocessed/goes_mag/{sat_id}/{YYYY}/{sat_id}_{YYYYMM}.parquet.

INPUT DATA
----------
- data/interim/goes_mag/{sat_id}/{YYYY}/*.parquet
  Monthly interim GOES MAG Parquet files produced by Phase 2.2.
  Expected CORE_VECTOR_COLS: b_gsm_x, b_gsm_y, b_gsm_z,
  b_vdh_x, b_vdh_y, b_vdh_z.
  Optional: b_total_nT, num_points, b_quality or quality, method,
  dqf, orbit position columns in either naming variant.
- CLI optional: --input-root  (default: data/interim/goes_mag)
               --output-root (default: data/preprocessed/goes_mag)
               --satellites  (default: all found; e.g. goes15 goes16
                              goes17 goes18)

OUTPUT DATA
-----------
- data/preprocessed/goes_mag/{sat_id}/{YYYY}/{sat_id}_{YYYYMM}.parquet
  One Parquet file per satellite per month with a gapless 1-minute
  time index. Output column order: timestamp, sat_id, mag_source,
  b_total_nT (float32), b_gsm_x/y/z (float32), b_vdh_x/y/z (float32),
  quality_summary (Int64), num_points (Int64), method (Int64),
  dqf (Int64), orbit_longitude_deg (float32),
  orbit_latitude_deg (float32), orbit_radius_m (float32),
  mag_valid (boolean). Entirely-null non-protected columns are dropped.

NEXT PIPELINE PHASE
-------------------
- Phase 3.5 depends only on Phase 2.2 (src/clean/interim/goes_mag.py)
  completing first. It has no dependency on Phase 3.1 through 3.4
  and can run in parallel with other Phase 3.x scripts.
- After this script completes, GOES MAG preprocessed data is ready
  for Phase 4 (src/fusion/data_fusion.py).
- All Phase 3.x preprocessing scripts must complete before Phase 4
  can begin, since the fusion stage requires all sources to be present.
"""

---

src/preprocess/omni.py
"""
Phase 3.3 - Preprocess: OMNI Interim Parquet to Preprocessed Parquet
---------------------------------------------------------------------
This is the third Phase 3 preprocessing script. It reads monthly
OMNI Parquet files produced by Phase 2.3 (src/clean/interim/omni.py),
replaces OMNI-specific fill sentinel values with NaN, appends boolean
data validity mask columns, standardizes column types and ordering,
and writes cleaned monthly Parquet files to data/preprocessed/omni/.
This script has no dependency on Phase 3.1 or 3.2 and can run in
parallel with the Swarm preprocessing scripts.

Purpose
-------
OMNI 1-minute solar wind data uses sentinel fill values to indicate
missing measurements rather than NaN or null. For example, IMF total
field Bt uses 999.9 and 9999.99, proton temperature uses 9999999.0,
and spacecraft IDs use 99 and 999. These sentinels are defined in the
OMNIWeb format specification and must be replaced with NaN before the
data can be used in any numeric computation or joined with other
sources. This is the primary purpose of this script and the key
transformation not performed in Phase 2.3.

After fill replacement, five composite boolean validity mask columns
are added by add_validity_masks():
- imf_valid: all four IMF components (Bt, Bx, By, Bz) are non-null
- plasma_valid: solar wind speed, proton density, and flow pressure
  are all non-null
- velocity_vector_valid: all three GSE velocity components non-null
- timeshift_valid: propagation timeshift_sec is non-null
- low_interpolation: percent_interpolation is at or below 20 percent

These masks allow downstream scripts to filter on measurement quality
without repeating the fill-value logic. All mask columns are typed as
pandas nullable boolean.

The interim "datetime" column (reconstructed from year/doy/hour/minute
in Phase 2.3) is renamed to "timestamp" and converted to UTC-aware
pandas Timestamp. The raw time-index columns (year, doy, hour, minute,
yyyymm) are dropped since they are redundant with timestamp. Columns
that are entirely null in a given file and are not in PROTECTED_COLUMNS
are removed by drop_all_null_nonprotected() to guard against schema
drift across OMNI annual files.

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, optional date
        range, and optional --force flag.
Step 2: Discover all omni_*.parquet files one directory level below
        --input-root (i.e. data/interim/omni/{YYYY}/omni_*.parquet).
        Raise FileNotFoundError if none are found.
Step 3: For each file, append source_file and source_year_dir
        provenance columns, then apply standardize_columns():
        rename "datetime" -> "timestamp" via RENAME_MAP, convert to
        UTC-aware Timestamp, drop rows with unparseable timestamps,
        drop raw time-index columns, ensure "date" string column exists.
Step 4: Apply optional date range filter (UTC-aware comparison).
        Skip the file if the result is empty.
Step 5: Apply replace_fill_values(): for each physical column defined
        in OMNI_FILL_VALUES, coerce to numeric and replace each known
        sentinel value with np.nan.
Step 6: Apply add_validity_masks(): compute five composite boolean
        validity columns (imf_valid, plasma_valid,
        velocity_vector_valid, timeshift_valid, low_interpolation).
Step 7: Apply standardize_dtypes(): cast spacecraft ID and averaging
        count columns to nullable Int64; cast physical measurement
        columns to float64; cast validity mask columns to pandas
        nullable boolean.
Step 8: Apply drop_all_null_nonprotected(): drop any non-protected
        column that is entirely null in this file.
Step 9: Sort by timestamp, deduplicate on timestamp keeping the last
        occurrence, reorder columns with PROTECTED_COLUMNS first, and
        write to data/preprocessed/omni/{YYYY}/omni_{YYYYMM}.parquet.
        Skip existing files unless --force is set.

INPUT DATA
----------
- data/interim/omni/{YYYY}/omni_{YYYYMM}.parquet
  Monthly OMNI Parquet files produced by Phase 2.3.
  Contains raw OMNI columns including year, doy, hour, minute,
  physical measurements, and a "datetime" column.
- CLI optional: --input-root  (default: data/interim/omni)
               --output-root (default: data/preprocessed/omni)
               --start-date  (default: no lower bound; YYYY-MM-DD)
               --end-date    (default: no upper bound; YYYY-MM-DD)
               --force       (overwrite existing output files)

OUTPUT DATA
-----------
- data/preprocessed/omni/{YYYY}/omni_{YYYYMM}.parquet
  One Parquet file per month. Schema leads with PROTECTED_COLUMNS
  (timestamp, date, source_file, source_year_dir, bt_nT,
  bx_gse_gsm_nT, by_gsm_nT, bz_gsm_nT, speed_km_s, vx_gse_km_s,
  vy_gse_km_s, vz_gse_km_s, proton_density_n_cc,
  proton_temperature_K, flow_pressure_nPa, percent_interpolation,
  timeshift_sec, time_between_obs_sec, n_points_imf_avg,
  n_points_plasma_avg, imf_spacecraft_id, plasma_spacecraft_id),
  followed by the five boolean validity mask columns, then any
  remaining non-null columns. Fill sentinels replaced with NaN.
  Physical columns typed as float64; ID/count columns as Int64;
  validity masks as pandas nullable boolean.

NEXT PIPELINE PHASE
-------------------
- Phase 3.3 depends only on Phase 2.3 (src/clean/interim/omni.py)
  completing first. It has no dependency on Phase 3.1 or 3.2 and
  can run in parallel with the Swarm preprocessing scripts.
- After this script completes, OMNI preprocessed data is ready for
  Phase 4 (src/fusion/data_fusion.py).
- All Phase 3.x preprocessing scripts must complete before Phase 4
  can begin, since the fusion stage requires all sources to be present.
"""

---

src/preprocess/swarm_1min.py
"""
Phase 3.1 - Preprocess: Swarm Interim to 1-Minute Aggregated Parquet
---------------------------------------------------------------------
This is the first Phase 3 preprocessing script and must run before
Phase 3.2 (swarm_chaos.py). It reads monthly cleaned Swarm Parquet
files produced by Phase 2.4 (src/clean/interim/swarm.py), aggregates
the 1 Hz measurements to 1-minute bins, and writes per-month 1-minute
Parquet files to data/preprocessed/swarm_1min/. These files are the
required input for swarm_chaos.py (Phase 3.2).

Purpose
-------
The Swarm MAGx_LR_1B product is nominally sampled at 1 Hz. The
downstream fusion stage (Phase 4) operates on a common 1-minute
time grid shared across all data sources. This script performs the
temporal aggregation from 1 Hz to 1-minute cadence and produces
per-bin diagnostic columns (valid counts, missing fractions, flag
maxima) so that data quality is traceable through the pipeline.

The aggregation uses dt.floor("min") to bin timestamps, then groupby
aggregation. Longitude is aggregated using a circular mean (sin/cos
decomposition followed by arctan2 reconstruction) to correctly handle
the 0/360 degree wraparound that an arithmetic mean would corrupt.
Flag columns (flags_f, flags_b) are reduced by max within each bin,
preserving the worst quality indicator present. All three B-component
valid counts are tracked independently, and a combined indicator
n_valid_1min_points counts bins where all three B components are
simultaneously non-null.

This script does not apply CHAOS model corrections. Its output is
intentionally labeled "pre-CHAOS" and is consumed exclusively by
swarm_chaos.py (Phase 3.2), which applies the CHAOS field model
subtraction to isolate the external field perturbation.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory, output directory,
        and optional --force flag.
Step 2: Recursively discover monthly Swarm Parquet files under
        --input-dir using is_monthly_file() to filter out any daily
        files (requires parent dir to be a 4-digit year and filename
        to contain exactly 2 underscores).
        Raise FileNotFoundError if no matching files are found.
Step 3: For each monthly file, apply aggregate_month():
        - Parse and floor timestamps to 1-minute bins.
        - Coerce all numeric columns to float64.
        - Decompose longitude into sin/cos components for circular
          mean aggregation.
        - Compute per-bin validity indicators: b_all_valid (all three
          B components non-null), f_is_missing, b_any_missing.
        - Group by 1-minute bin and aggregate: mean for position and
          field values, count for valid-point tallies, sum for
          n_valid_1min_points, max for flag columns, first for
          satellite and source_collection, size for n_rows_1min_total,
          mean for missing fraction diagnostics.
        - Reconstruct longitude_deg_mean from arctan2(sin, cos).
        - Cast count columns to int16 and flag columns to Int64.
        - Enforce OUTPUT_COLUMNS column order.
Step 4: Write the aggregated monthly DataFrame to
        data/preprocessed/swarm_1min/{sat}/{YYYY}/{original_name}_1min.parquet.
        Skip existing files unless --force is set.

INPUT DATA
----------
- data/interim/swarm/{sat}/{YYYY}/swarm_{sat}_{YYYYMM}.parquet
  Monthly cleaned Swarm Parquet files produced by Phase 2.4.
  Expected columns: timestamp, satellite, latitude_deg,
  longitude_deg, radius_m, f_nT, flags_f, flags_b, b_north_nT,
  b_east_nT, b_center_nT, source_collection.
- CLI optional: --input-dir  (default: data/interim/swarm)
               --output-dir (default: data/preprocessed/swarm_1min)
               --force      (overwrite existing output files)

OUTPUT DATA
-----------
- data/preprocessed/swarm_1min/{sat}/{YYYY}/swarm_{sat}_{YYYYMM}_1min.parquet
  One Parquet file per satellite per month at 1-minute cadence.
  Output schema (OUTPUT_COLUMNS, 20 columns): timestamp, satellite,
  latitude_deg_mean, longitude_deg_mean (circular mean), radius_m_mean,
  f_nT_mean, b_north_nT_mean, b_east_nT_mean, b_center_nT_mean,
  n_valid_f_1min, n_valid_b_north_1min, n_valid_b_east_1min,
  n_valid_b_center_1min, n_valid_1min_points, flags_f_max,
  flags_b_max, source_collection, n_rows_1min_total,
  f_missing_frac_1min, b_missing_frac_1min.

NEXT PIPELINE PHASE
-------------------
- Phase 3.1 depends on Phase 2.4 (src/clean/interim/swarm.py)
  completing with --clean-type monthly before it can run.
- Phase 3.2 (src/preprocess/swarm_chaos.py) depends on Phase 3.1
  completing and expects files under data/preprocessed/swarm_1min/.
  Do not run swarm_chaos.py until this script has finished.
- After Phase 3.2 completes, both Swarm preprocessing steps are done
  and Swarm data is ready for Phase 4 (src/fusion/).
"""

---

src/preprocess/supermag.py
"""
Phase 3.6 - Preprocess: SuperMAG Interim Parquet to Preprocessed Parquet
-------------------------------------------------------------------------
This is the sixth and final Phase 3 preprocessing script. It reads
monthly interim SuperMAG Parquet files produced by Phase 2.5
(src/clean/interim/supermag.py), enforces a UTC-aware gapless
1-minute time index, resolves dual magnetic coordinate frame
availability into preferred-frame scalar columns, appends per-frame
and per-component validity flags, standardizes column types, and
writes preprocessed monthly Parquet files to
data/preprocessed/supermag/. This script completes Phase 3 and
enables Phase 4 (src/fusion/) to begin once all other Phase 3.x
scripts have also finished.

Purpose
-------
SuperMAG provides magnetic field perturbations in two coordinate
frames per station: NEZ (local geomagnetic North-East-Z) and GEO
(geographic North-East-Z). Not all stations or time periods have
both frames available. This script resolves frame availability into
three unified preferred-value columns (dbn_nt, dbe_nt, dbz_nt) by
selecting NEZ values when all three NEZ components are non-null, and
falling back to GEO values otherwise. The selection is recorded in
the "preferred_frame" string column ("nez", "geo", or null).
Both raw frame columns are preserved in the output so downstream
scripts can override the preference if needed.

Three boolean validity columns are appended:
- nez_valid: all three NEZ components are non-null
- geo_valid: all three GEO components are non-null
- supermag_valid: nez_valid OR geo_valid (at least one frame usable)

Three per-component missingness columns (dbn_missing, dbe_missing,
dbz_missing) reflect missingness in the preferred-value columns.

Unlike Phase 3.4 and 3.5, the 1-minute reindexing in this script
produces a UTC timezone-aware index (pd.date_range with tz="UTC")
and is controlled by the --reindex-minutely flag (default: True).
When reindexing is applied, a "has_observation" boolean column
distinguishes rows that originated from the source data (True) from
rows inserted to fill gaps (False). When --no-reindex-minutely is
set, has_observation is set to True for all rows without reindexing.

This script also includes a --force flag and optional date range
filter not present in Phase 3.4 and 3.5.

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, optional
        station filter list, --reindex-minutely flag (default True),
        optional date range, and optional --force flag.
Step 2: Recursively discover all supermag_*.parquet files under
        --input-root, optionally filtered by station code inferred
        from the two-levels-up directory name (uppercased).
Step 3: For each file, parse station code and YYYYMM from the
        filename via FILE_RE. Skip files with unexpected names.
        Skip existing output files unless --force is set.
Step 4: Apply ensure_station(): normalize station to uppercase from
        the "station" column, then "source_station" column, then
        the filename hint.
Step 5: Normalize timestamp to UTC-aware Timestamp, drop rows with
        unparseable timestamps, apply optional date range filter.
        Skip the file if the result is empty.
Step 6: Drop source_url and source_file provenance columns if present.
        Retain only timestamp, station, COMPONENT_COLS, and META_COLS.
        Deduplicate on (station, timestamp) keeping the last occurrence.
Step 7: If --reindex-minutely (default), apply minute_reindex():
        construct a UTC-aware gapless 1-minute pd.date_range, reindex,
        forward/back fill station, and set has_observation=True only
        for rows present in the source data. Otherwise set
        has_observation=True for all rows without reindexing.
Step 8: Apply add_validity_flags(): ensure all COMPONENT_COLS and
        META_COLS exist (fill with NaN if absent), compute nez_valid,
        geo_valid, supermag_valid, derive preferred-frame dbn_nt/
        dbe_nt/dbz_nt and preferred_frame label, and add per-component
        missingness flags.
Step 9: Apply standardize_dtypes(): field and meta columns to float32,
        boolean columns to pandas nullable boolean, preferred_frame
        to pandas string.
Step 10: Enforce output column order and write to
         data/preprocessed/supermag/{STATION}/{YYYY}/
         supermag_{STATION}_{YYYYMM}.parquet.

INPUT DATA
----------
- data/interim/supermag/{STATION}/{YYYY}/supermag_{STATION}_{YYYYMM}.parquet
  Monthly interim SuperMAG Parquet files produced by Phase 2.5.
  Expected COMPONENT_COLS: dbn_nez_nt, dbe_nez_nt, dbz_nez_nt,
  dbn_geo_nt, dbe_geo_nt, dbz_geo_nt.
  Expected META_COLS: ext_sec, glon_deg, glat_deg, mlt_hour,
  mcolat_deg, decl_deg, sza_deg.
- CLI optional: --input-root        (default: data/interim/supermag)
               --output-root       (default: data/preprocessed/supermag)
               --stations          (default: all found; e.g. YKC MEA
                                    SOD ABK)
               --reindex-minutely  (default: True; use
                                    --no-reindex-minutely to disable)
               --start-date        (default: no lower bound; YYYY-MM-DD)
               --end-date          (default: no upper bound; YYYY-MM-DD)
               --force             (overwrite existing output files)

OUTPUT DATA
-----------
- data/preprocessed/supermag/{STATION}/{YYYY}/supermag_{STATION}_{YYYYMM}.parquet
  One Parquet file per station per month. Output column order:
  timestamp (UTC-aware), station, has_observation (boolean),
  dbn_nt/dbe_nt/dbz_nt (float32, preferred frame values),
  preferred_frame (string), supermag_valid/nez_valid/geo_valid
  (boolean), dbn_missing/dbe_missing/dbz_missing (boolean),
  dbn_nez_nt/dbe_nez_nt/dbz_nez_nt (float32),
  dbn_geo_nt/dbe_geo_nt/dbz_geo_nt (float32),
  ext_sec, glon_deg, glat_deg, mlt_hour, mcolat_deg, decl_deg,
  sza_deg (float32).

NEXT PIPELINE PHASE
-------------------
- Phase 3.6 depends only on Phase 2.5 (src/clean/interim/supermag.py)
  completing first. It has no dependency on Phase 3.1 through 3.5
  and can run in parallel with other Phase 3.x scripts.
- This is the final Phase 3 script. Once all six Phase 3.x scripts
  have completed (swarm_1min, swarm_chaos, omni, goes_xrs, goes_mag,
  supermag), all preprocessed sources are ready for Phase 4.
- Move on to Phase 4 (src/fusion/data_fusion.py) once all Phase 3.x
  scripts have finished.
"""

---

src/preprocess/swarm_chaos.py
"""
Phase 3.2 - Preprocess: Swarm 1-Minute Parquet with CHAOS Model Fields
-----------------------------------------------------------------------
This is the second and final Phase 3 preprocessing script for Swarm
data. It must not run until Phase 3.1 (swarm_1min.py) has completed.
It reads 1-minute aggregated Swarm Parquet files produced by Phase 3.1,
evaluates the CHAOS geomagnetic field model at each observation point
using chaosmagpy, appends synthesized field components and residuals
as new columns, and writes CHAOS-enhanced monthly Parquet files to
data/preprocessed/swarm_chaos/. The output of this script is the
final preprocessed Swarm artifact consumed by Phase 4 (src/fusion/).

Purpose
-------
Raw Swarm magnetometer measurements contain contributions from
multiple field sources: the core field, lithospheric field, and
external magnetospheric field. To isolate the external field
perturbations that are most physically relevant to geomagnetic
disturbance prediction, this script evaluates the CHAOS-8 model
(Finlay et al.) at each 1-minute Swarm observation point and
computes the difference between the observed field and the modeled
field.

The CHAOS model is decomposed into four synthesized components via
chaosmagpy:
- Time-dependent internal field (synth_values_tdep): core field
- Static internal field (synth_values_static): lithospheric field
- GSM external field (synth_values_gsm): magnetospheric ring current
  and tail contributions in GSM coordinates
- SM external field (synth_values_sm): symmetric ring current in SM
  coordinates

These are combined as:
  chaos_internal = tdep + static
  chaos_external = GSM + SM
  chaos_total    = internal + external

The CHAOS model evaluates field values in spherical coordinates
(Br, Btheta, Bphi). These are converted to the local NEC frame
(North, East, Centre) using sph_to_local_nec():
  B_north  = -Btheta
  B_east   =  Bphi
  B_center = -Br

Two residual families are computed per NEC component:
  residual_total    = observed - chaos_total
  residual_internal = observed - chaos_internal

Rows missing latitude, longitude, or radius are dropped before
model evaluation since all three are required by chaosmagpy.
Timestamps are stripped of timezone metadata before conversion to
MJD2000, as chaosmagpy requires timezone-naive numpy datetime64.

The RC index file (Ring Current index) is required by chaosmagpy
for the external field synthesis. It can be auto-downloaded via
--download-rc-if-missing if not already present locally.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory, output directory,
        CHAOS .mat file path, RC index HDF5 path, optional
        --download-rc-if-missing flag, and optional --force flag.
Step 2: Discover all swarm_*_1min.parquet files under --input-dir.
        Raise FileNotFoundError if none are found.
Step 3: Ensure the RC index file is present via ensure_rc_index().
        If absent and --download-rc-if-missing is set, download it
        via cp.data_utils.save_RC_h5file(). Otherwise raise
        FileNotFoundError. Register the RC index path with
        chaosmagpy via cp.basicConfig["file.RC_index"].
Step 4: Load the CHAOS model from the .mat file using
        cp.load_CHAOS_matfile().
Step 5: For each 1-minute input file, apply process_file():
        - Normalize timestamps to UTC, then strip timezone metadata
          to produce timezone-naive UTC numpy datetime64 values.
        - Drop rows with null latitude, longitude, or radius.
        - Convert timestamps to MJD2000 via cp.data_utils.mjd2000().
        - Convert radius from meters to kilometers.
        - Convert latitude to colatitude (theta = 90 - latitude).
        - Evaluate CHAOS components via compute_chaos_components():
          call synth_values_tdep, synth_values_static,
          synth_values_gsm, and synth_values_sm; combine into
          internal, external, and total spherical field vectors;
          convert each to NEC frame via sph_to_local_nec().
        - Append 9 chaos field columns and 6 residual columns.
        - Enforce OUTPUT_BASE_COLUMNS + OUTPUT_CHAOS_COLUMNS order.
Step 6: Write the output to
        data/preprocessed/swarm_chaos/{sat}/{YYYY}/{name}_1min_chaos.parquet.
        Skip existing files unless --force is set.

INPUT DATA
----------
- data/preprocessed/swarm_1min/{sat}/{YYYY}/swarm_{sat}_{YYYYMM}_1min.parquet
  1-minute aggregated Swarm Parquet files produced by Phase 3.1.
  Required columns: timestamp, latitude_deg_mean, longitude_deg_mean,
  radius_m_mean, b_north_nT_mean, b_east_nT_mean, b_center_nT_mean.
- models/chaos/CHAOS-8.5.mat
  CHAOS-8 model coefficient file. Required. Not downloaded automatically.
  Obtain from: http://www.spacecenter.dk/files/magnetic-models/CHAOS-8/
- models/chaos/RC_index.h5
  Ring Current index HDF5 file required by chaosmagpy for external
  field synthesis. Can be downloaded automatically via
  --download-rc-if-missing if absent.
- CLI optional: --input-dir             (default: data/preprocessed/swarm_1min)
               --output-dir            (default: data/preprocessed/swarm_chaos)
               --chaos-mat-file        (default: models/chaos/CHAOS-8.5.mat)
               --rc-index-file         (default: models/chaos/RC_index.h5)
               --download-rc-if-missing (flag; auto-download RC index)
               --force                 (overwrite existing output files)

OUTPUT DATA
-----------
- data/preprocessed/swarm_chaos/{sat}/{YYYY}/swarm_{sat}_{YYYYMM}_1min_chaos.parquet
  One Parquet file per satellite per month. Schema is
  OUTPUT_BASE_COLUMNS (20 columns from Phase 3.1) plus
  OUTPUT_CHAOS_COLUMNS (15 new columns):
    chaos_total_b_{north,east,center}_nT    (3 cols)
    chaos_internal_b_{north,east,center}_nT (3 cols)
    chaos_external_b_{north,east,center}_nT (3 cols)
    residual_total_b_{north,east,center}_nT (3 cols)
    residual_internal_b_{north,east,center}_nT (3 cols)
  Total: 35 columns.

NEXT PIPELINE PHASE
-------------------
- Phase 3.2 strictly depends on Phase 3.1 (swarm_1min.py) completing
  first. Do not run this script until swarm_1min.py has finished.
- This is the final preprocessing step for Swarm data.
- After Phase 3.2 completes, all Phase 3.x preprocessing is done and
  all sources (GOES XRS, GOES MAG, OMNI, Swarm, SuperMAG) are ready
  for Phase 4 (src/fusion/data_fusion.py).
"""

---

src/preprocess/goes_xrs.py
"""
Phase 3.4 - Preprocess: GOES XRS Interim Parquet to Preprocessed Parquet
-------------------------------------------------------------------------
This is the fourth Phase 3 preprocessing script. It reads monthly
interim GOES XRS Parquet files produced by Phase 2.1
(src/clean/interim/goes_xrs.py), enforces a gapless 1-minute time
index, appends generation-aware validity flags and a derived flux
ratio column, standardizes column types, and writes preprocessed
monthly Parquet files to data/preprocessed/goes_xrs/. This script
has no dependency on Phase 3.1, 3.2, or 3.3 and can run in parallel
with other Phase 3.x scripts.

Purpose
-------
The interim GOES XRS data from Phase 2.1 contains only the timestamps
actually present in the source NetCDF files. Gaps in the 1-minute
record are represented as absent rows rather than NaN-filled rows.
This script enforces a complete 1-minute time index via minute_reindex(),
which constructs a full pd.date_range from the first to the last
minute in the file and reindexes the DataFrame against it, introducing
NaN-filled rows for every missing minute. This ensures a gapless
regular time grid for all downstream consumers.

The script is generation-aware: it detects whether a file comes from
a legacy satellite (goes15) or a GOES-R satellite (goes16/17/18) by
checking for the presence of GOESR_OPTIONAL_COLUMNS, and assigns the
result to an "xrs_source" label column. This label is then used to
apply the correct flag acceptance criteria when computing validity
masks:
- Legacy: xrsa_flag and xrsb_flag in {0, 1}
- GOES-R: xrsa_flag and xrsb_flag in {0, 1, 2}

Both validity columns additionally require the corresponding flux
value to be non-null. A derived "xray_flux_ratio" column (xrsb_flux /
xrsa_flux) is computed when both channels are positive. The
has_electron_correction boolean column indicates whether a GOES-R
electron correction was applied to a given row.

Provenance columns from the interim stage (source_file,
source_file_type, satellite) are dropped before output. This script
does not apply a --force flag; existing output files for a given
satellite/month are unconditionally overwritten.

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, and optional
        satellite filter list.
Step 2: Recursively discover all Parquet files under --input-root,
        optionally filtered by satellite name inferred from the
        two-levels-up directory name.
Step 3: For each file, apply ensure_sat_id() to normalize the sat_id
        column: fall back to "satellite" column, then to the directory
        name if neither column exists.
Step 4: Drop interim-only provenance columns (source_file,
        source_file_type, satellite) and parse the timestamp column.
Step 5: Apply minute_reindex(): construct a complete 1-minute
        pd.date_range across the file's time span and reindex the
        DataFrame, introducing NaN rows for missing minutes.
        Forward/back fill sat_id over introduced NaN rows.
Step 6: Assign the xrs_source label ("goesr" or "legacy") via
        build_source_label() based on presence of GOESR_OPTIONAL_COLUMNS.
Step 7: Compute generation-aware xrsa_valid and xrsb_valid boolean
        columns using flag acceptance sets appropriate to each
        generation, combined with a non-null flux requirement.
Step 8: Compute has_electron_correction boolean (GOES-R only) and
        xray_flux_ratio = xrsb_flux / xrsa_flux (when both are positive).
Step 9: Enforce ordered output column list, apply standardize_dtypes()
        (flux -> float32, flags -> Int64, validity -> boolean), and
        write to data/preprocessed/goes_xrs/{sat_id}/{YYYY}/{sat_id}_{YYYYMM}.parquet.

INPUT DATA
----------
- data/interim/goes_xrs/{sat_id}/{YYYY}/*.parquet
  Monthly interim GOES XRS Parquet files produced by Phase 2.1.
  Expected to contain timestamp, satellite or sat_id, xrsa_flux,
  xrsb_flux, xrsa_flag, xrsb_flag, xrsa_num, xrsb_num, and
  optionally GOESR_OPTIONAL_COLUMNS (xrsa_flux_electrons,
  xrsb_flux_electrons, electron_correction_flag).
- CLI optional: --input-root  (default: data/interim/goes_xrs)
               --output-root (default: data/preprocessed/goes_xrs)
               --satellites  (default: all found; e.g. goes15 goes16
                              goes17 goes18)

OUTPUT DATA
-----------
- data/preprocessed/goes_xrs/{sat_id}/{YYYY}/{sat_id}_{YYYYMM}.parquet
  One Parquet file per satellite per month with a gapless 1-minute
  time index. Output column order: timestamp, sat_id, xrs_source,
  xrsa_flux (float32), xrsb_flux (float32), xray_flux_ratio (float32),
  xrsa_flag (Int64), xrsb_flag (Int64), xrsa_num (Int64),
  xrsb_num (Int64), xrsa_valid (boolean), xrsb_valid (boolean),
  and for GOES-R files: xrsa_flux_electrons, xrsb_flux_electrons,
  electron_correction_flag, has_electron_correction (boolean).

NEXT PIPELINE PHASE
-------------------
- Phase 3.4 depends only on Phase 2.1 (src/clean/interim/goes_xrs.py)
  completing first. It has no dependency on Phase 3.1, 3.2, or 3.3
  and can run in parallel with other Phase 3.x scripts.
- After this script completes, GOES XRS preprocessed data is ready
  for Phase 4 (src/fusion/data_fusion.py).
- All Phase 3.x preprocessing scripts must complete before Phase 4
  can begin, since the fusion stage requires all sources to be present.
"""

---

