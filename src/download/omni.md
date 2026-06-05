# Instructions Download OMNI

* note: download 1 year at a time to prevent reaching data limit
* note: `src/clean/interim/omni.py` expects datat to be a `.lst` file in:

```
data/raw/omni/
├── omni_min_2015.lst
├── omni_min_2016.lst
├── omni_min_2017.lst
├── omni_min_2018.lst
├── omni_min_2019.lst
├── omni_min_2020.lst
├── omni_min_2021.lst
├── omni_min_2022.lst
├── omni_min_2023.lst
└── omni_min_2024.lst
```

1. go to https://omniweb.gsfc.nasa.gov/form/omni_min_def.html
    * [here](https://omniweb.gsfc.nasa.gov/form/omni_min_def.html)

2. Select the variables below:
```json
{
  "Select activity": "Create file",
  "Select resolution": "1-min averaged",
  "Start": "20150101",
  "Stop": "20241231",
  "Select variables": [
    "IMF Spacecraft ID",
    "Plasma Spacecraft ID",
    "# Fine Scale Points in IMF Avgs",
    "# Fine Scale Points in Plasma Avgs",
    "Percent interpolated",
    "Timeshift, sec.",
    "Time btwn observations,sec"
  ],
  "Magnetic field": [
    "IMF Magnitude Avg(Scalar), nT",
    "Bx, GSE/GSM, nT",
    "By, GSM, nT",
    "Bz, GSM, nT"
  ],
  "Plasma": [
    "Flow Speed, km/sec",
    "Vx Velocity, GSE, km/s",
    "Vy Velocity, GSE, km/s",
    "Vz Velocity, GSE, km/s",
    "Proton Density, n/cc",
    "Proton Temperature, K"
  ],
  "Derived Parameters": [
    "Flow Pressure, nPa"
  ],
}
```

3. click `Submit`

4. right click the ASCII `link`

5. clink `save link as`

6. `omni_min{YYYY}.lst` in `data/raw/omni`

## Fill Values

```python
omni_fill_values = {
    "IMF Spacecraft ID": 99,
    "Plasma Spacecraft ID": 99,
    "# of points in IMF averages": 999,
    "# of points in Plasma averages": 999,
    "Percent of Interpolation": 999,
    "Timeshift": 999999,
    "Time btwn observations,sec": 999999,
    "Field magnitude average, nT": 9999.99,
    "BX, nT (GSE, GSM)": 9999.99,
    "BY, nT (GSM)": 9999.99,
    "BZ, nT (GSM)": 9999.99,
    "Speed, km/s": 99999.9,
    "Vx Velocity,km/s": 99999.9,
    "Vy Velocity, km/s": 99999.9,
    "Vz Velocity, km/s": 99999.9,
    "Proton Density, n/cc": 999.99,
    "Proton Temperature, K": 9999999.0,
    "Flow pressure, nPa": 99.99
}
```

## Column Map

```python
omni_column_map = {
    "YYYY": 0,
    "DOY": 1,
    "HR": 2,
    "MN": 3,
    "ID for IMF spacecraft": 4,
    "ID for SW Plasma spacecraft": 5,
    "# of points in IMF averages": 6,
    "# of points in Plasma averages": 7,
    "Percent of Interpolation": 8,
    "Timeshift": 9,
    "Time btwn observations,sec": 10,
    "Field magnitude average, nT": 11,
    "BX, nT (GSE, GSM)": 12,
    "BY, nT (GSM)": 13,
    "BZ, nT (GSM)": 14,
    "Speed, km/s": 15,
    "Vx Velocity,km/s": 16,
    "Vy Velocity, km/s": 17,
    "Vz Velocity, km/s": 18,
    "Proton Density, n/cc": 19,
    "Proton Temperature, K": 20,
    "Flow pressure, nPa": 21
}
```
