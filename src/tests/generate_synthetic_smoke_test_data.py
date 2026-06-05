#!/usr/bin/env python3
"""Generate deterministic synthetic raw SuperMAG smoke-test data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.clean.interim.supermag import clean_one_file


EXPECTED_COLUMNS = [
    "Timestamp",
    "tval",
    "ext",
    "iaga",
    "glon",
    "glat",
    "mlt",
    "mcolat",
    "decl",
    "sza",
    "N",
    "E",
    "Z",
    "source_station",
    "source_url",
]

SOURCE_URL = "https://supermag.jhuapl.edu/services/data-api/"
FILENAME_RE = re.compile(r"^supermag_[A-Z0-9]{3}_\d{8}\.parquet$")


@dataclass(frozen=True)
class StationMeta:
    glon: float
    glat: float
    mcolat: float
    decl: float
    mlt_offset: float


STATION_METADATA: dict[str, StationMeta] = {
    "ABK": StationMeta(glon=18.8, glat=68.4, mcolat=21.6, decl=8.1, mlt_offset=0.0),
    "MEA": StationMeta(glon=27.2, glat=69.8, mcolat=20.2, decl=6.7, mlt_offset=1.7),
    "SOD": StationMeta(glon=26.6, glat=67.4, mcolat=22.9, decl=7.3, mlt_offset=2.9),
    "YKC": StationMeta(glon=-114.5, glat=62.5, mcolat=27.1, decl=-12.2, mlt_offset=10.8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic raw SuperMAG smoke-test parquet files."
    )
    parser.add_argument("--start-date", type=str, required=True, help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, required=True, help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Target output directory (e.g. data/test/raw/supermag).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def parse_date(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def date_range(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start_date, end_date, freq="D"))


def component_dict_string(geo_value: np.ndarray, nez_value: np.ndarray) -> list[str]:
    # Keep this as stringified dicts to explicitly exercise parser logic.
    out: list[str] = []
    for geo, nez in zip(geo_value, nez_value):
        out.append(str({"geo": float(np.round(geo, 5)), "nez": float(np.round(nez, 5))}))
    return out


def build_daily_dataframe(
    station: str,
    day: pd.Timestamp,
    metadata: StationMeta,
    seed: int,
) -> pd.DataFrame:
    ts = pd.date_range(day, periods=1440, freq="1min", tz="UTC")
    minute_idx = np.arange(1440, dtype=float)

    daily_seed = seed + int(day.strftime("%Y%m%d")) + (sum(ord(ch) for ch in station) * 10_000)
    rng = np.random.default_rng(daily_seed)

    phase = 2.0 * np.pi * minute_idx / 1440.0
    slow_wave = np.sin(phase)
    fast_wave = np.sin(6.0 * phase)
    burst = np.exp(-0.5 * ((minute_idx - 720.0) / 50.0) ** 2)

    base_nez = 18.0 * slow_wave + 4.0 * fast_wave + 14.0 * burst
    n_nez = base_nez + rng.normal(0.0, 0.8, size=1440)
    e_nez = 0.6 * base_nez + rng.normal(0.0, 0.8, size=1440)
    z_nez = -0.4 * base_nez + rng.normal(0.0, 0.8, size=1440)

    n_geo = n_nez + 1.8 + rng.normal(0.0, 0.3, size=1440)
    e_geo = e_nez - 1.5 + rng.normal(0.0, 0.3, size=1440)
    z_geo = z_nez + 0.9 + rng.normal(0.0, 0.3, size=1440)

    frame = pd.DataFrame(
        {
            "Timestamp": ts,
            "tval": (ts.view("int64") // 1_000_000_000).astype("int64"),
            "ext": np.full(1440, 60.0, dtype="float64"),
            "iaga": np.full(1440, station, dtype=object),
            "glon": np.full(1440, metadata.glon, dtype="float64"),
            "glat": np.full(1440, metadata.glat, dtype="float64"),
            "mlt": ((minute_idx / 60.0) + metadata.mlt_offset) % 24.0,
            "mcolat": np.full(1440, metadata.mcolat, dtype="float64"),
            "decl": np.full(1440, metadata.decl, dtype="float64"),
            "sza": 85.0 - 35.0 * np.cos(phase),
            "N": component_dict_string(n_geo, n_nez),
            "E": component_dict_string(e_geo, e_nez),
            "Z": component_dict_string(z_geo, z_nez),
            "source_station": np.full(1440, station, dtype=object),
            "source_url": np.full(1440, SOURCE_URL, dtype=object),
        }
    )
    return frame[EXPECTED_COLUMNS]


def validate_generated_file(path: Path, station: str) -> None:
    if not FILENAME_RE.match(path.name):
        raise ValueError(f"Invalid filename format: {path.name}")

    df = pd.read_parquet(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path} has unexpected schema: {list(df.columns)}")
    if len(df) != 1440:
        raise ValueError(f"{path} expected 1440 rows, found {len(df)}")

    unique_station = set(df["source_station"].astype(str).unique().tolist())
    if unique_station != {station}:
        raise ValueError(f"{path} has invalid source_station values: {sorted(unique_station)}")

    diffs = np.diff(df["tval"].to_numpy(dtype=np.int64))
    if not np.all(diffs == 60):
        raise ValueError(f"{path} has non-1-minute cadence in tval.")

    # Validate cleaner compatibility explicitly.
    cleaned = clean_one_file(path)
    if cleaned.empty:
        raise ValueError(f"{path} could not be cleaned by src.clean.interim.supermag.clean_one_file.")


def main() -> None:
    args = parse_args()

    # Default stations hardcoded since the command line arg was removed
    stations = ["ABK", "MEA", "SOD", "YKC"]

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if start_date > end_date:
        raise ValueError("--start-date must be <= --end-date")

    output_root = args.output_root.resolve()
    days = date_range(start_date, end_date)

    files_written = 0
    for day in days:
        yyyymmdd = day.strftime("%Y%m%d")
        year = day.strftime("%Y")
        month = day.strftime("%m")

        for station in stations:
            out_dir = output_root / station / year / month
            out_path = out_dir / f"supermag_{station}_{yyyymmdd}.parquet"

            if out_path.exists() and not args.force:
                raise FileExistsError(f"Output exists and --force not set: {out_path}")

            out_dir.mkdir(parents=True, exist_ok=True)
            df = build_daily_dataframe(
                station=station,
                day=day,
                metadata=STATION_METADATA[station],
                seed=args.seed,
            )
            df.to_parquet(out_path, index=False)
            validate_generated_file(out_path, station)
            files_written += 1
            print(f"[ok] wrote {out_path}")

    print(f"Stations processed: {len(stations)}")
    print(f"Days generated: {len(days)}")
    print(f"Files written: {files_written}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
