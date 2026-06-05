#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

DEFAULT_INPUT_ROOT = Path("data/fused/labeled")
DEFAULT_OUTPUT_ROOT = Path("data/preprocessed/fused/standardized")

STATION_COL = "station"
TIMESTAMP_COL = "timestamp"

TARGET_COLS = [
    "target_geq_p95_30m",
    "target_geq_p99_30m",
    "target_geq_p95_60m",
    "target_geq_p99_60m",
    "target_geq_p95_120m",
    "target_geq_p99_120m",
]

IDENTITY_COLS = [
    TIMESTAMP_COL,
    STATION_COL,
]

CONTEXT_COLS = [
    "supermag_glon_deg",
    "supermag_glat_deg",
    "supermag_mlt_hour",
    "supermag_mcolat_deg",
    "supermag_decl_deg",
    "supermag_sza_deg",
]

GROUND_KEEP_COLS = [
    "supermag_dbn_nt",
    "supermag_dbe_nt",
    "supermag_dbz_nt",
    "supermag_dbn_missing",
    "supermag_dbe_missing",
    "supermag_dbz_missing",
    "supermag_has_observation",
]

SUN_BASE_MAP = {
    "xrs_xrsa_flux": "sun_xrsa_flux",
    "xrs_xrsb_flux": "sun_xrsb_flux",
    "xrs_missing": "sun_missing",
    "xrs_freshness_min": "sun_freshness_min",
}

L1_BASE_MAP = {
    "omni_bt_nT": "l1_bt_nT",
    "omni_bz_gsm_nT": "l1_bz_gsm_nT",
    "omni_speed_km_s": "l1_speed_km_s",
    "omni_flow_pressure_nPa": "l1_flow_pressure_nPa",
    "omni_proton_density_n_cc": "l1_proton_density_n_cc",
    "omni_by_gsm_nT": "l1_by_gsm_nT",
    "omni_bx_gse_gsm_nT": "l1_bx_gse_gsm_nT",
    "omni_timeshift_sec": "l1_timeshift_sec",
    "omni_imf_valid": "l1_imf_valid",
    "omni_plasma_valid": "l1_plasma_valid",
    "omni_velocity_vector_valid": "l1_velocity_vector_valid",
    "omni_percent_interpolation": "l1_percent_interpolation",
}

GEO_BASE_MAP = {
    "goes_mag_bz_gsm_nT": "geo_bz_gsm_nT",
    "goes_mag_bz_vdh_nT": "geo_bz_vdh_nT",
    "goes_mag_bx_gsm_nT": "geo_bx_gsm_nT",
    "goes_mag_by_gsm_nT": "geo_by_gsm_nT",
    "goes_mag_any_valid": "geo_any_valid",
    "goes_mag_sat_count": "geo_sat_count",
}

LEO_BASE_MAP = {
    "swarm_residual_internal_b_north_nT": "leo_residual_internal_b_north_nT",
    "swarm_residual_internal_b_east_nT": "leo_residual_internal_b_east_nT",
    "swarm_residual_internal_b_center_nT": "leo_residual_internal_b_center_nT",
    "swarm_n_valid_1min_points": "leo_n_valid_1min_points",
    "swarm_freshness_min": "leo_freshness_min",
    "swarm_missing": "leo_missing",
}

BASE_FEATURES = [
    "sun_xrsa_flux",
    "sun_xrsb_flux",
    "sun_missing",
    "sun_freshness_min",
    "l1_bt_nT",
    "l1_bz_gsm_nT",
    "l1_speed_km_s",
    "l1_flow_pressure_nPa",
    "l1_proton_density_n_cc",
    "l1_by_gsm_nT",
    "geo_bz_gsm_nT",
    "geo_bz_vdh_nT",
    "geo_bx_gsm_nT",
    "geo_by_gsm_nT",
    "geo_any_valid",
    "leo_residual_internal_b_north_nT",
    "leo_residual_internal_b_east_nT",
    "leo_residual_internal_b_center_nT",
    "leo_n_valid_1min_points",
    "leo_freshness_min",
    "leo_missing",
]

SUN_DERIVED = [
    "sun_log_flux_ratio",
    "sun_xrsb_diff_10m",
    "sun_xrsa_diff_10m",
    "sun_xrs_sum_flux",
    "sun_xrs_hardness_proxy",
]

L1_DERIVED = [
    "l1_clock_angle_rad",
    "l1_newell_like",
    "l1_dynamic_pressure_speed",
    "l1_bmag_pressure_ratio",
    "l1_electric_field_proxy",
]

GEO_DERIVED = [
    "geo_bz_minus_vdh",
    "geo_b_transverse_gsm",
    "geo_b_transverse_vdh_proxy",
    "geo_bz_gsm_diff_15m",
    "geo_bz_vdh_diff_15m",
]

LEO_DERIVED = [
    "leo_fac_cond_proxy",
    "leo_residual_horizontal_mag",
    "leo_residual_vertical_to_horizontal_ratio",
    "leo_north_diff_10m",
    "leo_freshness_weighted_residual",
]

ALL_DERIVED = SUN_DERIVED + L1_DERIVED + GEO_DERIVED + LEO_DERIVED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single standardized compact feature table from labeled fused monthly parquet files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root of labeled fused parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for standardized feature outputs.",
    )
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="Optional YYYYMM filters.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing monthly outputs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_input_files(root: Path, months: list[str] | None) -> list[Path]:
    files = sorted(root.rglob("station_time_labeled_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No station_time_labeled_*.parquet files found under {root}")

    if months:
        keep = set(months)
        files = [p for p in files if p.stem.split("_")[-1] in keep]

    if not files:
        raise FileNotFoundError("No labeled parquet files matched the requested months.")

    return files


def build_base_rename_map(available: set[str]) -> dict[str, str]:
    raw_map = {}
    raw_map.update(SUN_BASE_MAP)
    raw_map.update(L1_BASE_MAP)
    raw_map.update(GEO_BASE_MAP)
    raw_map.update(LEO_BASE_MAP)
    return {src: dst for src, dst in raw_map.items() if src in available}


def required_output_base_cols() -> list[str]:
    return BASE_FEATURES


def scan_base_month(path: Path) -> pl.LazyFrame:
    lf = pl.scan_parquet(str(path))
    available = set(lf.collect_schema().names())

    required = {STATION_COL, TIMESTAMP_COL, *TARGET_COLS}
    missing = required - available
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {sorted(missing)}")

    rename_map = build_base_rename_map(available)

    keep_cols = (
        IDENTITY_COLS
        + TARGET_COLS
        + [c for c in CONTEXT_COLS if c in available]
        + [c for c in GROUND_KEEP_COLS if c in available]
        + list(rename_map.keys())
    )
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in available]))

    lf = lf.select(keep_cols).with_columns(
        [
            pl.col(STATION_COL).cast(pl.Utf8, strict=False).alias(STATION_COL),
            pl.col(TIMESTAMP_COL)
            .cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
            .alias(TIMESTAMP_COL),
        ]
    )

    if rename_map:
        lf = lf.rename(rename_map)

    schema_after = set(lf.collect_schema().names())
    missing_base = set(required_output_base_cols()) - schema_after
    for col in sorted(missing_base):
        lf = lf.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return lf.sort([STATION_COL, TIMESTAMP_COL])


def add_time_context(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = set(lf.collect_schema().names())
    exprs = []

    if "year" not in schema:
        exprs.append(pl.col(TIMESTAMP_COL).dt.year().cast(pl.Int16).alias("year"))
    if "month" not in schema:
        exprs.append(pl.col(TIMESTAMP_COL).dt.strftime("%Y-%m").alias("month"))
    if "yyyymm" not in schema:
        exprs.append(pl.col(TIMESTAMP_COL).dt.strftime("%Y%m").alias("yyyymm"))

    if exprs:
        lf = lf.with_columns(exprs)
    return lf


def add_derived_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = set(lf.collect_schema().names())
    exprs: list[pl.Expr] = []

    if {"sun_xrsb_flux", "sun_xrsa_flux"}.issubset(schema):
        exprs.extend(
            [
                (
                    (pl.col("sun_xrsb_flux").abs() + 1e-12).log()
                    - (pl.col("sun_xrsa_flux").abs() + 1e-12).log()
                ).alias("sun_log_flux_ratio"),
                (
                    pl.col("sun_xrsb_flux") - pl.col("sun_xrsb_flux").shift(10).over(STATION_COL)
                ).alias("sun_xrsb_diff_10m"),
                (
                    pl.col("sun_xrsa_flux") - pl.col("sun_xrsa_flux").shift(10).over(STATION_COL)
                ).alias("sun_xrsa_diff_10m"),
                (
                    pl.col("sun_xrsb_flux") + pl.col("sun_xrsa_flux")
                ).alias("sun_xrs_sum_flux"),
                (
                    pl.col("sun_xrsb_flux") / (pl.col("sun_xrsa_flux").abs() + 1e-12)
                ).alias("sun_xrs_hardness_proxy"),
            ]
        )

    if {"l1_by_gsm_nT", "l1_bz_gsm_nT", "l1_speed_km_s", "l1_flow_pressure_nPa", "l1_bt_nT"}.issubset(schema):
        bt_yz = ((pl.col("l1_by_gsm_nT") ** 2 + pl.col("l1_bz_gsm_nT") ** 2) ** 0.5)
        clock_half = pl.arctan2(pl.col("l1_by_gsm_nT"), pl.col("l1_bz_gsm_nT")).abs() / 2.0
        exprs.extend(
            [
                pl.arctan2(pl.col("l1_by_gsm_nT"), pl.col("l1_bz_gsm_nT")).alias("l1_clock_angle_rad"),
                (
                    (pl.col("l1_speed_km_s").clip(lower_bound=0.0) ** (4.0 / 3.0))
                    * (bt_yz.clip(lower_bound=0.0) ** (2.0 / 3.0))
                    * (clock_half.sin().abs() ** (8.0 / 3.0))
                ).alias("l1_newell_like"),
                (
                    pl.col("l1_flow_pressure_nPa") * pl.col("l1_speed_km_s")
                ).alias("l1_dynamic_pressure_speed"),
                (
                    pl.col("l1_bt_nT") / (pl.col("l1_flow_pressure_nPa").abs() + 1e-12)
                ).alias("l1_bmag_pressure_ratio"),
                (
                    pl.col("l1_speed_km_s") * pl.col("l1_bz_gsm_nT").abs()
                ).alias("l1_electric_field_proxy"),
            ]
        )

    if {"geo_bz_gsm_nT", "geo_bz_vdh_nT", "geo_bx_gsm_nT", "geo_by_gsm_nT"}.issubset(schema):
        exprs.extend(
            [
                (
                    pl.col("geo_bz_gsm_nT") - pl.col("geo_bz_vdh_nT")
                ).alias("geo_bz_minus_vdh"),
                (
                    ((pl.col("geo_bx_gsm_nT") ** 2) + (pl.col("geo_by_gsm_nT") ** 2)) ** 0.5
                ).alias("geo_b_transverse_gsm"),
                (
                    pl.col("geo_bz_gsm_nT").abs() + pl.col("geo_bz_vdh_nT").abs()
                ).alias("geo_b_transverse_vdh_proxy"),
                (
                    pl.col("geo_bz_gsm_nT") - pl.col("geo_bz_gsm_nT").shift(15).over(STATION_COL)
                ).alias("geo_bz_gsm_diff_15m"),
                (
                    pl.col("geo_bz_vdh_nT") - pl.col("geo_bz_vdh_nT").shift(15).over(STATION_COL)
                ).alias("geo_bz_vdh_diff_15m"),
            ]
        )

    if {
        "leo_residual_internal_b_north_nT",
        "leo_residual_internal_b_east_nT",
        "leo_residual_internal_b_center_nT",
        "leo_freshness_min",
        "supermag_sza_deg",
    }.issubset(schema):
        horiz = (
            (
                pl.col("leo_residual_internal_b_north_nT") ** 2
                + pl.col("leo_residual_internal_b_east_nT") ** 2
            ) ** 0.5
        )
        exprs.extend(
            [
                (
                    pl.col("leo_residual_internal_b_north_nT")
                    * (1.0 / (1.0 + pl.col("supermag_sza_deg").abs()))
                ).alias("leo_fac_cond_proxy"),
                horiz.alias("leo_residual_horizontal_mag"),
                (
                    pl.col("leo_residual_internal_b_center_nT").abs() / (horiz.abs() + 1e-12)
                ).alias("leo_residual_vertical_to_horizontal_ratio"),
                (
                    pl.col("leo_residual_internal_b_north_nT")
                    - pl.col("leo_residual_internal_b_north_nT").shift(10).over(STATION_COL)
                ).alias("leo_north_diff_10m"),
                (
                    pl.col("leo_residual_internal_b_north_nT")
                    / (1.0 + pl.col("leo_freshness_min").abs())
                ).alias("leo_freshness_weighted_residual"),
            ]
        )

    if exprs:
        lf = lf.with_columns(exprs)

    return lf


def choose_final_columns(lf: pl.LazyFrame) -> list[str]:
    schema = set(lf.collect_schema().names())

    metadata_cols = [
        TIMESTAMP_COL,
        STATION_COL,
        "year",
        "month",
        "yyyymm",
        *[c for c in CONTEXT_COLS if c in schema],
        *[c for c in GROUND_KEEP_COLS if c in schema],
    ]

    keep: list[str] = []
    keep.extend([c for c in metadata_cols if c in schema])
    keep.extend([c for c in BASE_FEATURES if c in schema])
    keep.extend([c for c in ALL_DERIVED if c in schema])
    keep.extend([c for c in TARGET_COLS if c in schema])

    seen = set()
    ordered: list[str] = []
    for c in keep:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    return ordered


def write_metadata(output_root: Path, selected_columns: list[str]) -> None:
    meta_dir = output_root / "metadata"
    ensure_dir(meta_dir)

    payload = {
        "schema_name": "standardized",
        "description": "Single standardized compact feature schema with 5 engineered candidates per source family.",
        "target_columns": TARGET_COLS,
        "base_features": BASE_FEATURES,
        "sun_derived": SUN_DERIVED,
        "l1_derived": L1_DERIVED,
        "geo_derived": GEO_DERIVED,
        "leo_derived": LEO_DERIVED,
        "selected_columns": selected_columns,
    }
    (meta_dir / "feature_columns.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


def build_month(
    input_path: Path,
    output_root: Path,
    force: bool,
) -> tuple[str, int, int, int]:
    yyyymm = input_path.stem.split("_")[-1]
    yyyy = yyyymm[:4]
    out_path = output_root / yyyy / f"station_time_features_{yyyymm}.parquet"

    if out_path.exists() and not force:
        return yyyymm, -1, -1, -1

    lf = scan_base_month(input_path)
    lf = add_time_context(lf)
    lf = add_derived_features(lf)

    final_cols = choose_final_columns(lf)
    lf = lf.select(final_cols).sort([STATION_COL, TIMESTAMP_COL])

    ensure_dir(out_path.parent)

    summary = (
        lf.select(
            [
                pl.len().alias("rows"),
                pl.col(STATION_COL).n_unique().alias("stations"),
            ]
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )

    lf.sink_parquet(str(out_path))
    return yyyymm, int(summary["rows"]), int(summary["stations"]), len(final_cols)


def main() -> None:
    args = parse_args()

    input_files = discover_input_files(args.input_root, args.months)
    written_cols: list[str] | None = None

    for input_path in input_files:
        yyyymm, rows, stations, ncols = build_month(
            input_path=input_path,
            output_root=args.output_root,
            force=args.force,
        )

        if rows < 0:
            print(f"[skip]  station_time_features_{yyyymm}.parquet")
            continue

        sample_out = args.output_root / yyyymm[:4] / f"station_time_features_{yyyymm}.parquet"
        written_cols = pl.scan_parquet(str(sample_out)).collect_schema().names()

        print(
            f"[ok]    station_time_features_{yyyymm}.parquet | "
            f"rows={rows:>8,} | stations={stations} | cols={ncols}"
        )

    if written_cols is not None:
        write_metadata(
            output_root=args.output_root,
            selected_columns=written_cols,
        )

        print()
        print("Done: standardized")
        print(f"Output root: {args.output_root}")
        print("Derived features kept: 5 sun, 5 l1, 5 geo, 5 leo")


if __name__ == "__main__":
    main()


