#!/usr/bin/env python3
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

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import chaosmagpy as cp


OUTPUT_BASE_COLUMNS = [
    "timestamp",
    "satellite",
    "latitude_deg_mean",
    "longitude_deg_mean",
    "radius_m_mean",
    "f_nT_mean",
    "b_north_nT_mean",
    "b_east_nT_mean",
    "b_center_nT_mean",
    "n_valid_f_1min",
    "n_valid_b_north_1min",
    "n_valid_b_east_1min",
    "n_valid_b_center_1min",
    "n_valid_1min_points",
    "flags_f_max",
    "flags_b_max",
    "source_collection",
    "n_rows_1min_total",
    "f_missing_frac_1min",
    "b_missing_frac_1min",
]

OUTPUT_CHAOS_COLUMNS = [
    "chaos_total_b_north_nT",
    "chaos_total_b_east_nT",
    "chaos_total_b_center_nT",
    "chaos_internal_b_north_nT",
    "chaos_internal_b_east_nT",
    "chaos_internal_b_center_nT",
    "chaos_external_b_north_nT",
    "chaos_external_b_east_nT",
    "chaos_external_b_center_nT",
    "residual_total_b_north_nT",
    "residual_total_b_east_nT",
    "residual_total_b_center_nT",
    "residual_internal_b_north_nT",
    "residual_internal_b_east_nT",
    "residual_internal_b_center_nT",
]


def sph_to_local_nec(br: np.ndarray, btheta: np.ndarray, bphi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b_north = -btheta
    b_east = bphi
    b_center = -br
    return b_north, b_east, b_center


def ensure_rc_index(rc_index_file: Path, download_if_missing: bool) -> None:
    if rc_index_file.exists():
        return

    if not download_if_missing:
        raise FileNotFoundError(
            f"RC index file not found: {rc_index_file}. "
            f"Provide --rc-index-file or use --download-rc-if-missing."
        )

    rc_index_file.parent.mkdir(parents=True, exist_ok=True)
    cp.data_utils.save_RC_h5file(str(rc_index_file), read_from=None)


def compute_chaos_components(model, time_mjd2000, radius_km, theta_deg, phi_deg):
    br_tdep, bt_tdep, bp_tdep = model.synth_values_tdep(time_mjd2000, radius_km, theta_deg, phi_deg)
    br_static, bt_static, bp_static = model.synth_values_static(radius_km, theta_deg, phi_deg)
    br_gsm, bt_gsm, bp_gsm = model.synth_values_gsm(time_mjd2000, radius_km, theta_deg, phi_deg)
    br_sm, bt_sm, bp_sm = model.synth_values_sm(time_mjd2000, radius_km, theta_deg, phi_deg)

    br_internal = br_tdep + br_static
    bt_internal = bt_tdep + bt_static
    bp_internal = bp_tdep + bp_static

    br_external = br_gsm + br_sm
    bt_external = bt_gsm + bt_sm
    bp_external = bp_gsm + bp_sm

    br_total = br_internal + br_external
    bt_total = bt_internal + bt_external
    bp_total = bp_internal + bp_external

    return {
        "total": sph_to_local_nec(br_total, bt_total, bp_total),
        "internal": sph_to_local_nec(br_internal, bt_internal, bp_internal),
        "external": sph_to_local_nec(br_external, bt_external, bp_external),
    }


def process_file(infile: Path, model) -> pd.DataFrame:
    df = pd.read_parquet(infile).copy()
    # ChaosMagPy works on timezone-naive numpy datetime64 values.
    # Normalize to UTC first, then drop tz metadata explicitly.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "latitude_deg_mean", "longitude_deg_mean", "radius_m_mean"])

    ts_naive_utc = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    time_mjd2000 = cp.data_utils.mjd2000(ts_naive_utc.to_numpy())
    radius_km = df["radius_m_mean"].to_numpy(dtype=float) / 1000.0
    theta_deg = 90.0 - df["latitude_deg_mean"].to_numpy(dtype=float)
    phi_deg = df["longitude_deg_mean"].to_numpy(dtype=float)

    comps = compute_chaos_components(model, time_mjd2000, radius_km, theta_deg, phi_deg)

    total_n, total_e, total_c = comps["total"]
    internal_n, internal_e, internal_c = comps["internal"]
    external_n, external_e, external_c = comps["external"]

    df["chaos_total_b_north_nT"] = total_n
    df["chaos_total_b_east_nT"] = total_e
    df["chaos_total_b_center_nT"] = total_c

    df["chaos_internal_b_north_nT"] = internal_n
    df["chaos_internal_b_east_nT"] = internal_e
    df["chaos_internal_b_center_nT"] = internal_c

    df["chaos_external_b_north_nT"] = external_n
    df["chaos_external_b_east_nT"] = external_e
    df["chaos_external_b_center_nT"] = external_c

    df["residual_total_b_north_nT"] = df["b_north_nT_mean"] - df["chaos_total_b_north_nT"]
    df["residual_total_b_east_nT"] = df["b_east_nT_mean"] - df["chaos_total_b_east_nT"]
    df["residual_total_b_center_nT"] = df["b_center_nT_mean"] - df["chaos_total_b_center_nT"]

    df["residual_internal_b_north_nT"] = df["b_north_nT_mean"] - df["chaos_internal_b_north_nT"]
    df["residual_internal_b_east_nT"] = df["b_east_nT_mean"] - df["chaos_internal_b_east_nT"]
    df["residual_internal_b_center_nT"] = df["b_center_nT_mean"] - df["chaos_internal_b_center_nT"]

    keep_cols = OUTPUT_BASE_COLUMNS + OUTPUT_CHAOS_COLUMNS
    return df[keep_cols].sort_values("timestamp").reset_index(drop=True)


def output_path(infile: Path, output_dir: Path) -> Path:
    sat = infile.parent.parent.name
    year = infile.parent.name
    out_dir = output_dir / sat / year
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = infile.name.replace("_1min.parquet", "_1min_chaos.parquet")
    return out_dir / out_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CHAOS model and residual components for 1-minute Swarm parquet files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/preprocessed/swarm_1min"),
        help="Directory containing 1-minute pre-CHAOS Swarm parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preprocessed/swarm_chaos"),
        help="Directory for CHAOS-enhanced Swarm parquet files",
    )
    parser.add_argument(
        "--chaos-mat-file",
        type=Path,
        default=Path("models/chaos/CHAOS-8.5.mat"),
        help="Path to CHAOS model .mat file",
    )
    parser.add_argument(
        "--rc-index-file",
        type=Path,
        default=Path("models/chaos/RC_index.h5"),
        help="Path to RC index HDF5 file used by ChaosMagPy",
    )
    parser.add_argument(
        "--download-rc-if-missing",
        action="store_true",
        help="Download the latest RC index if the file does not exist.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("swarm_*_1min.parquet"))
    if not files:
        raise FileNotFoundError(f"No 1-minute Swarm parquet files found under {args.input_dir}")

    ensure_rc_index(args.rc_index_file, args.download_rc_if_missing)
    cp.basicConfig["file.RC_index"] = str(args.rc_index_file)

    model = cp.load_CHAOS_matfile(str(args.chaos_mat_file))

    for infile in files:
        try:
            outfile = output_path(infile, args.output_dir)
            if outfile.exists() and not args.force:
                print(f"skip existing {outfile}")
                continue

            out = process_file(infile, model)
            out.to_parquet(outfile, index=False)
            print(f"wrote {outfile} rows={len(out)}")
        except Exception as e:
            print(f"failed {infile} error={e}")


if __name__ == "__main__":
    main()
