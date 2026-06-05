#!/usr/bin/env python3
"""Build compact smoke-test raw subsets for OMNI, Swarm, GOES MAG, and GOES XRS."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import shutil

import pandas as pd

SWARM_RE = re.compile(r"^swarm_([ABC])_(\d{8})\.parquet$")
GOES_MAG_DAY_RE = re.compile(r"_d(\d{8})_")
GOES_XRS_SPAN_RE = re.compile(r"_s(\d{8})_e(\d{8})_")
GOES_XRS_YEAR_RE = re.compile(r"_y(\d{4})_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate smoke-test raw subsets for OMNI, Swarm, GOES MAG, and GOES XRS."
    )
    parser.add_argument("--start-date", type=str, required=True, help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, required=True, help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/raw"),
        help="Production raw root to read from. Default is data/raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/test/raw"),
        help="Target output directory. Default is data/test/raw",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing subset outputs.")
    return parser.parse_args()


def normalize_date(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def ensure_source_tree(input_root: Path) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input raw root does not exist: {input_root}")
    required_dirs = ["omni", "swarm", "goes_mag", "goes_xrs"]
    for rel in required_dirs:
        path = input_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Required source directory missing: {path}")


def in_window(ts: pd.Timestamp, start_date: pd.Timestamp, end_date: pd.Timestamp) -> bool:
    return start_date <= ts <= end_date


def copy_file(src: Path, dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        raise FileExistsError(f"Output exists and --force not set: {dst}")
    shutil.copy2(src, dst)


def subset_omni(
    input_root: Path,
    output_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    force: bool,
) -> int:
    omni_in = input_root / "omni"
    omni_out = output_root / "omni"
    omni_out.mkdir(parents=True, exist_ok=True)

    source_files = sorted(omni_in.glob("omni_min_*.lst"))
    if not source_files:
        raise FileNotFoundError(f"No OMNI yearly files found in {omni_in}")

    by_year: dict[int, list[str]] = defaultdict(list)
    for src in source_files:
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 4:
                    continue
                try:
                    year = int(parts[0])
                    doy = int(parts[1])
                    hour = int(parts[2])
                    minute = int(parts[3])
                except ValueError:
                    continue

                ts = (
                    pd.Timestamp(year=year, month=1, day=1)
                    + pd.Timedelta(days=doy - 1, hours=hour, minutes=minute)
                )

                if in_window(ts, start_date, end_date):
                    by_year[year].append(line if line.endswith("\n") else f"{line}\n")

    if not by_year:
        raise FileNotFoundError("No OMNI rows overlap the requested smoke-test window.")

    written = 0
    for year, lines in sorted(by_year.items()):
        out_path = omni_out / f"omni_min_{year}.lst"
        if out_path.exists() and not force:
            raise FileExistsError(f"Output exists and --force not set: {out_path}")
        out_path.write_text("".join(lines), encoding="utf-8")
        print(f"[ok] wrote {out_path} rows={len(lines)}")
        written += 1
    return written


def subset_swarm(
    input_root: Path,
    output_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    force: bool,
) -> tuple[int, set[str]]:
    src_root = input_root / "swarm"
    written = 0
    satellites_found: set[str] = set()

    for src in sorted(src_root.rglob("swarm_*.parquet")):
        match = SWARM_RE.match(src.name)
        if not match:
            continue
        sat = match.group(1)
        day = pd.Timestamp(match.group(2))
        if not in_window(day, start_date, end_date):
            continue
        rel = src.relative_to(input_root)
        dst = output_root / rel
        copy_file(src, dst, force=force)
        satellites_found.add(sat)
        written += 1
        print(f"[ok] copied {src} -> {dst}")

    missing = {"A", "B", "C"} - satellites_found
    if missing:
        raise FileNotFoundError(f"Missing Swarm subset files for satellites: {sorted(missing)}")
    return written, satellites_found


def subset_goes_mag(
    input_root: Path,
    output_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    force: bool,
) -> tuple[int, set[str]]:
    src_root = input_root / "goes_mag"
    written = 0
    satellites_found: set[str] = set()

    for src in sorted(src_root.rglob("*.nc")):
        match = GOES_MAG_DAY_RE.search(src.name)
        if not match:
            continue
        day = pd.Timestamp(match.group(1))
        if not in_window(day, start_date, end_date):
            continue
        rel = src.relative_to(input_root)
        dst = output_root / rel
        copy_file(src, dst, force=force)
        written += 1
        satellites_found.add(src.parent.parent.parent.name)
        print(f"[ok] copied {src} -> {dst}")

    if not satellites_found:
        raise FileNotFoundError(
            "No GOES MAG files overlap the requested smoke-test window."
        )
    return written, satellites_found


def subset_goes_xrs(
    input_root: Path,
    output_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    force: bool,
) -> tuple[int, set[str]]:
    src_root = input_root / "goes_xrs"
    written = 0
    satellites_found: set[str] = set()

    for src in sorted(src_root.rglob("*.nc")):
        keep = False
        span_match = GOES_XRS_SPAN_RE.search(src.name)
        year_match = GOES_XRS_YEAR_RE.search(src.name)

        if span_match:
            file_start = pd.Timestamp(span_match.group(1))
            file_end = pd.Timestamp(span_match.group(2))
            keep = (file_start <= end_date) and (file_end >= start_date)
        elif year_match:
            year = int(year_match.group(1))
            file_start = pd.Timestamp(year=year, month=1, day=1)
            file_end = pd.Timestamp(year=year, month=12, day=31)
            keep = (file_start <= end_date) and (file_end >= start_date)

        if not keep:
            continue

        rel = src.relative_to(input_root)
        dst = output_root / rel
        copy_file(src, dst, force=force)
        written += 1
        satellites_found.add(src.parent.name)
        print(f"[ok] copied {src} -> {dst}")

    if not satellites_found:
        raise FileNotFoundError(
            "No GOES XRS files overlap the requested smoke-test window."
        )
    return written, satellites_found


def validate_outputs(
    output_root: Path,
    goes_mag_sats: set[str],
    goes_xrs_sats: set[str],
) -> None:
    omni_files = sorted((output_root / "omni").glob("omni_min_*.lst"))
    if not omni_files:
        raise FileNotFoundError("OMNI subset output missing.")

    for sat in ["A", "B", "C"]:
        if not list((output_root / "swarm" / sat).rglob("swarm_*.parquet")):
            raise FileNotFoundError(f"Swarm subset missing for satellite {sat}.")

    for sat in sorted(goes_mag_sats):
        if not list((output_root / "goes_mag" / sat).rglob("*.nc")):
            raise FileNotFoundError(f"GOES MAG subset missing for satellite {sat}.")

    for sat in sorted(goes_xrs_sats):
        if not list((output_root / "goes_xrs" / sat).rglob("*.nc")):
            raise FileNotFoundError(f"GOES XRS subset missing for satellite {sat}.")


def main() -> None:
    args = parse_args()
    start_date = normalize_date(args.start_date)
    end_date = normalize_date(args.end_date)
    if start_date > end_date:
        raise ValueError("--start-date must be <= --end-date")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    ensure_source_tree(input_root)
    output_root.mkdir(parents=True, exist_ok=True)

    omni_written = subset_omni(input_root, output_root, start_date, end_date, args.force)
    swarm_written, swarm_sats = subset_swarm(input_root, output_root, start_date, end_date, args.force)
    goes_mag_written, goes_mag_sats = subset_goes_mag(input_root, output_root, start_date, end_date, args.force)
    goes_xrs_written, goes_xrs_sats = subset_goes_xrs(input_root, output_root, start_date, end_date, args.force)
    validate_outputs(output_root, goes_mag_sats, goes_xrs_sats)

    print(f"Date range: {start_date.date()} -> {end_date.date()}")
    print(f"OMNI files written: {omni_written}")
    print(f"Swarm files copied: {swarm_written} satellites={sorted(swarm_sats)}")
    print(f"GOES MAG files copied: {goes_mag_written} satellites={sorted(goes_mag_sats)}")
    print(f"GOES XRS files copied: {goes_xrs_written} satellites={sorted(goes_xrs_sats)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()

