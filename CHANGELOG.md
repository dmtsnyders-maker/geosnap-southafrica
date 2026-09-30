# Changelog

## 0.5.1
- Yearly series: read only the raster window around the requested wards; progress messages during downloads and summing.

## 0.5.0
- Renamed to **geosnap-southafrica** (`import geosnap_southafrica`, `get_south_africa()`).
- Whole-country coverage: all 4,468 wards (Municipal Demarcation Board 2020) joined to the Stats SA ward product.
- Yearly series 2015-2030: modelled, anchored to the 2022 census, using WorldPop annual rasters (`population_basis` column).
- Suburbs per ward: OpenStreetMap points nationwide, City of Cape Town polygons, or any user layer; ward -> suburb interpolation.
- Filters: `province=`, `municipality=`, `ward=`, `boundary=`. New columns `ward_name`, `population_basis`.
- Output columns are identical in every year (no `_2022` suffixes).
- `python -m geosnap_southafrica.selftest` live check against reference totals.
- Fixed: Python 3.11 syntax error, ArcGIS download paging, Stats SA workbook parser (single 2022 vintage, no 2011 data).
- Legacy standalone pipeline moved to `contrib/legacy_pipeline/` (unmaintained).
