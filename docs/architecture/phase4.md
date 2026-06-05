src/fusion/data_fusion.py
"""
Phase 4.1 - Fusion: Station-Time Master Table from All Preprocessed Sources
----------------------------------------------------------------------------
This is the sole Phase 4 script and the final data preparation step
in the pipeline. It reads all five preprocessed source directories
(SuperMAG, OMNI, GOES XRS, GOES MAG, Swarm CHAOS), joins them into
a unified station-time master table using leakage-safe backward as-of
joins, appends per-source freshness and missingness diagnostics, and
writes one Parquet file per month to
data/fused/station_time_master/. All six Phase 3 preprocessing
scripts must complete before this script can run.

Purpose
-------
The fusion stage assembles the station-time master table that serves
as the primary modeling input. Each row in the output corresponds to
one SuperMAG ground station at one UTC minute. The SuperMAG base
table provides the target-side magnetic field perturbation features.
All other sources (solar wind, X-ray flux, geostationary magnetic
field, low-Earth-orbit CHAOS residuals) are joined to it as
explanatory features.

The join design is explicitly leakage-safe: all non-SuperMAG sources
are attached using pd.merge_asof with direction="backward". For each
SuperMAG station-minute row, the join finds the most recent
observation in the source that occurred at or before the row's
timestamp and within the configured lookback tolerance. No future
source data is ever attached to a given row. This ensures that the
fused table can be used for time-series forecasting without leaking
future information.

Three GIC-relevant features are derived from the SuperMAG base before
any join:
- horizontal_perturbation_nt: np.hypot(dbn_nt, dbe_nt), the
  horizontal magnetic field perturbation magnitude per station-minute
- abs_dbdt_nt_per_min: per-station time derivative of
  horizontal_perturbation_nt, computed as the absolute 1-minute diff
- gic_proxy: equal to abs_dbdt_nt_per_min (float32), a proxy for
  geomagnetically induced current magnitude

Multi-satellite sources (GOES XRS, GOES MAG, Swarm) are reduced to
a single per-timestamp row before the join by groupby aggregation:
flux and field components are averaged across satellites, validity
flags are reduced by max, satellite counts are recorded via nunique.
Swarm n_valid_1min_points is summed across satellites. All aggregated
columns carry a source prefix (xrs_, gmag_, swarm_).

A configurable lookback window controls how far backward each join is
allowed to search:
- --source-lookback-min (default 180): applied to OMNI and GOES joins
- --swarm-lookback-min  (default 180): applied to the Swarm join

To support backward matches near the start of a month, each source
is pre-loaded over a window of [month_start - max_lookback, month_end]
so that observations from the previous month are available for matches
near the month boundary.

After each join, two diagnostic columns are appended via
add_freshness_and_missing(): {source}_freshness_min (float32, seconds
between the SuperMAG row and the matched source row divided by 60)
and {source}_missing (boolean, True when no match was found within
tolerance). These allow downstream models to condition on data
availability and recency.

WORK FLOW
---------
Step 1: Parse CLI arguments - five source root directories, output
        root, optional --months list, optional date range, independent
        lookback tolerances for OMNI/GOES and Swarm, and --force flag.
Step 2: Discover all available YYYYMM values from the SuperMAG
        preprocessed root (or use --months if provided). Raise
        FileNotFoundError if none are found.
Step 3: For each YYYYMM, run fuse_one_month():
  Step 3a: Skip existing output unless --force is set.
  Step 3b: Load the SuperMAG base via load_supermag_month(): rglob
           all supermag_*_{YYYYMM}.parquet files, concatenate all
           stations, normalize timestamps to UTC, deduplicate on
           (station, timestamp) keeping last, and compute
           horizontal_perturbation_nt, abs_dbdt_nt_per_min,
           gic_proxy.
  Step 3c: Apply optional global date filter to the base table.
  Step 3d: Compute the pre-load window: [month_start - max_lookback,
           month_end] where max_lookback = max(source_lookback_min,
           swarm_lookback_min).
  Step 3e: Load each non-SuperMAG source over the pre-load window
           via its load_*_window() function. Multi-satellite sources
           are aggregated to a single per-timestamp row. Each source
           timestamp column is renamed to {source}_timestamp to
           prevent collision during merge_asof.
  Step 3f: Apply backward_asof_join() sequentially for OMNI, GOES
           XRS, GOES MAG, and Swarm. After each join, append
           {source}_freshness_min and {source}_missing columns via
           add_freshness_and_missing().
  Step 3g: Append year (Int16), month (Int8), and yyyymm (string)
           partition columns.
  Step 3h: Sort by (station, timestamp), deduplicate on
           (station, timestamp) keeping last, and write to
           data/fused/station_time_master/{YYYY}/
           station_time_master_{YYYYMM}.parquet.

INPUT DATA
----------
- data/preprocessed/supermag/**/supermag_*_{YYYYMM}.parquet
  Phase 3.6 output. Base table; drives row count and month coverage.
- data/preprocessed/omni/**/{YYYYMM}*.parquet
  Phase 3.3 output. 1-minute solar wind and IMF observations.
- data/preprocessed/goes_xrs/**/{YYYYMM}*.parquet
  Phase 3.4 output. Per-satellite GOES XRS flux at 1-minute cadence.
- data/preprocessed/goes_mag/**/{YYYYMM}*.parquet
  Phase 3.5 output. Per-satellite GOES MAG vectors at 1-minute cadence.
- data/preprocessed/swarm_chaos/**/swarm_*_{YYYYMM}_1min_chaos.parquet
  Phase 3.2 output. Per-satellite Swarm CHAOS residuals at 1-minute
  cadence.
- CLI optional: --supermag-root        (default: data/preprocessed/supermag)
               --omni-root            (default: data/preprocessed/omni)
               --goes-xrs-root        (default: data/preprocessed/goes_xrs)
               --goes-mag-root        (default: data/preprocessed/goes_mag)
               --swarm-root           (default: data/preprocessed/swarm_chaos)
               --output-root          (default: data/fused/station_time_master)
               --months               (default: all discovered; YYYYMM list)
               --start-date           (default: no lower bound; YYYY-MM-DD)
               --end-date             (default: no upper bound; YYYY-MM-DD)
               --source-lookback-min  (default: 180; for OMNI and GOES)
               --swarm-lookback-min   (default: 180; for Swarm)
               --force                (overwrite existing output files)

OUTPUT DATA
-----------
- data/fused/station_time_master/{YYYY}/station_time_master_{YYYYMM}.parquet
  One Parquet file per month. Each row is one SuperMAG station at
  one UTC minute. Schema includes:
  - All SuperMAG columns from Phase 3.6 plus horizontal_perturbation_nt,
    abs_dbdt_nt_per_min, gic_proxy
  - OMNI columns (bt_nT, bx_gse_gsm_nT, by_gsm_nT, bz_gsm_nT,
    speed_km_s, vx/vy/vz_gse_km_s, proton_density_n_cc,
    flow_pressure_nPa, percent_interpolation, timeshift_sec,
    imf_valid, plasma_valid, velocity_vector_valid)
    plus omni_timestamp, omni_freshness_min, omni_missing
  - GOES XRS aggregated columns (xrs_sat_count, xrs_xrsa_flux,
    xrs_xrsb_flux, xrs_flux_ratio, xrs_any_valid,
    xrs_has_electron_correction)
    plus goes_xrs_timestamp, goes_xrs_freshness_min, goes_xrs_missing
  - GOES MAG aggregated columns (gmag_sat_count, gmag_b_gsm_{x,y,z},
    gmag_b_vdh_{x,y,z}, gmag_any_valid)
    plus goes_mag_timestamp, goes_mag_freshness_min, goes_mag_missing
  - Swarm aggregated columns (swarm_sat_count,
    swarm_residual_total_b_{north,east,center}_nT,
    swarm_residual_internal_b_{north,east,center}_nT,
    swarm_n_valid_1min_points)
    plus swarm_timestamp, swarm_freshness_min, swarm_missing
  - Partition columns: year (Int16), month (Int8), yyyymm (string)

NEXT PIPELINE PHASE
-------------------
- Phase 4.1 is the final data preparation step. Its output is the
  primary input to all downstream modeling, feature engineering,
  and analysis scripts.
- All six Phase 3 scripts must complete before this script can run.
- The critical intra-pipeline ordering constraint is: Phase 3.1
  (swarm_1min.py) must complete before Phase 3.2 (swarm_chaos.py),
  which must complete before this script can use swarm_chaos output.
"""

---

