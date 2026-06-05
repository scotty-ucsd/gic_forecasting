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

GOES_SOURCES = {
    "goes15": {
        "generation": "legacy",
        "xray": {
            "base_url": "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes15/xrsf-l2-avg1m_science/",
            "mode": "yearly",
            "filename": "sci_xrsf-l2-avg1m_g15_y{year}_v2-2-1.nc",
        },
        "mag": {
            "base_url": "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/mag/goes15/magn-l2-hires/",
            "mode": "daily_hierarchical",
            "filename": "dn_magn-l2-hires_g15_d{yyyymmdd}_v0_0_2.nc",
        },
    },
    "goes16": {
        "generation": "goes-r",
        "xray": {
            "base_url": "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/",
            "mode": "mission_span",
            "filename_hint": "sci_xrsf-l2-avg1m_g16_*.nc",
        },
        "mag": {
            "base_url": "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/magn-l2-avg1m_science/",
            "mode": "daily_hierarchical",
            "filename": "dn_magn-l2-avg1m_g16_d{yyyymmdd}_*.nc",
        },
    },
    "goes17": {
        "generation": "goes-r",
        "xray": {
            "base_url": "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes17/l2/data/xrsf-l2-avg1m_science/",
            "mode": "mission_span",
            "filename_hint": "sci_xrsf-l2-avg1m_g17_*.nc",
        },
        "mag": {
            "base_url": "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes17/l2/data/magn-l2-avg1m_science/",
            "mode": "daily_hierarchical",
            "filename": "dn_magn-l2-avg1m_g17_d{yyyymmdd}_*.nc",
        },
    },
    "goes18": {
        "generation": "goes-r",
        "xray": {
            "base_url": "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-avg1m_science/",
            "mode": "mission_span",
            "filename_hint": "sci_xrsf-l2-avg1m_g18_*.nc",
        },
        "mag": {
            "base_url": "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/magn-l2-avg1m_science/",
            "mode": "daily_hierarchical",
            "filename": "dn_magn-l2-avg1m_g18_d{yyyymmdd}_*.nc",
        },
    },
}
