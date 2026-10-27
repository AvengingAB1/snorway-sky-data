# snorway-sky-data

Public **data feed** for the Sunsetglow app, **South Norway** region (Husby/Nesna to Aalborg,
the North Sea to Falun: southern Norway, west Sweden, north Denmark). GitHub Actions extract the
latest forecasts to compact JSON and publish them to GitHub Pages. Same code and schemas as
`norway-sky-data` (North Norway); only the boxes and place lists differ.

| File | Source | Cadence |
|---|---|---|
| `meps_latest.json` | MET Norway **MEPS 2.5 km** (HARMONIE-AROME) low/mid/high cloud + visibility (`vis`), OPeNDAP from thredds.met.no, resampled to 0.03° × 0.075°, 5 % steps, +66 h, plus ~30 h of past hours carried forward | every MEPS run (3-hourly), hourly probe |
| `elements_latest.json` | MEPS 10 m wind + hourly precipitation, rendered as map overlays | with each MEPS run |
| `stations*.json` | MET Norway **Locationforecast 2.0**: 47 photo spots + towns (`stations.json`), peaks at summit altitude (`stations_peaks.json`), 12 km land grid (`stations_grid.json`) | with each MEPS run |
| `ecmwf_latest.json`, `ecmwf_elements_latest.json` | ECMWF IFS 9 km from Open-Meteo's open S3 bucket (no API calls), +15 days | 2×/day |
| `meteo_latest.json` | **Open-Meteo** best_match cloud layers + visibility, 14 days, 0.5° × 1.0° (440 points) | 1×/day (08:30 UTC run) |

Boxes: cloud grid 57.05–66.2 N, 0.11–15.63 E (`DEFAULT_BOX` in `extract/extract_meps.py`,
`BOX` in `extract/extract_meps_elements.py`); Open-Meteo/ECMWF grid 56.5–67.0 N, −1.5–17.5 E
(`BOX` in `extract/extract_meteo.py`, `snorway` in `extract/extract_ecmwf.py`).
Weather points: `extract/station_points.json` from `tools/build_nn_station_points.py --region snorway`.

Data © MET Norway (NLOD 2.0 / CC BY 4.0), ECMWF (CC BY 4.0) and Open-Meteo (CC BY 4.0).

Pages: Settings → Pages → Source: **GitHub Actions**. Feed: https://avengingab1.github.io/snorway-sky-data/
