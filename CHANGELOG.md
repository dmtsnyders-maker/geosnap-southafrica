# Changelog

## 0.9.6

- README: replaced the outdated "Not included yet" table with the current status of every topic. Licence: MIT only.

## 0.9.5

- README: OSM amenity coverage caveat. Verified end to end on real servers: ward data, yearly series, Census 2011 sub-places, land cover, SANBI vegetation, wealth index, prebuilt WorldPop and OSM tables.

## 0.9.4

- ArcGIS requests (SANBI etc.) identify as geosnap-southafrica and retry once with a browser user-agent on HTTP 403/406.

## 0.9.3

- selftest: `--skip-amenities` skips only the Overpass-dependent amenities check.

## 0.9.2

- `build-cache --amenities`: tries a mirror if Geofabrik fails (it can redirect-loop scripted downloads) and explains the manual `--pbf` route.

## 0.9.1

- Prebuilt OSM amenity counts: `geosnap-southafrica build-cache --amenities` counts amenities per ward for the whole country from
  one Geofabrik extract (needs `pip install osmium`), no Overpass. `extras=["amenities"]` uses the published table
  (`osm_amenity_ward_counts.parquet` on the `data-v1` release) when present, else falls back to Overpass tiles.

## 0.9.0

- Environmental layers. Generic tools `add_polygon_layer`, `add_raster_stats`, `add_point_counts`, `load_layer` work with
  any file, URL or GeoDataFrame (SANBI, geology, land use, temperature, tree cover, species records ...).
- New `extras=` names: SANBI (BGIS) `vegetation`, `biome`, `threat_status`, `protected_areas`, `cba`; global open rasters read
  remotely by window (nothing large is downloaded): `landcover` (ESA WorldCover 2021, incl. tree cover / built-up shares),
  `elevation` (Copernicus GLO-30), `climate_temp`, `climate_heat`, `climate_rain` (CHELSA v2.1 1981-2010).
- `fetch_arcgis_geojson` can clip to a bounding box.

## 0.8.0

- Prebuilt WorldPop ward sums: yearly series skip the ~100 MB-per-year raster downloads when the small
  `worldpop_ward_sums.parquet` is on the GitHub Release `data-v1` (downloaded once, falls back to computing locally).
  `geosnap-southafrica build-cache` creates the file. Disable with `GEOSNAP_SA_NO_PREBUILT=1`.

## 0.7.0

- Census 2011 support from the DataFirst "Community Profiles 2011" Stata files (`census2011=` folder): small-area counts
  summed onto the 2020 wards (`sal_geometry=`), or onto official sub-places (suburbs) with no geometry
  (`subplaces_2011`). Indicators: race, age structure and median age, median household income, low/high-income
  shares, dwelling type, tenure, water, household size, household-head unemployment, foreign-born, mobility,
  cell phone / car / internet, plus education, toilet, electricity and refuse when those files are added.
- `years=[2011, 2022]` gives real two-census panels for geosnap's cluster/transition/sequence tools.

## 0.6.1

- Amenities: smaller Overpass tiles (0.25°), shorter server timeout, four retries with back-off, third mirror. Fixes 504 timeouts on dense areas such as Cape Town.

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
