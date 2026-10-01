# geosnap-southafrica

A **South Africa connector for [geosnap](https://github.com/spatialucr/geosnap)**: ward-level neighbourhood data for the whole country, in geosnap's long-form GeoDataFrame format (`geoid` + `year`).

```python
from geosnap import DataStore
from geosnap_southafrica import get_south_africa

sa = get_south_africa(DataStore(), years=[2022])                                   # all 4,468 wards
ct = get_south_africa(DataStore(), years=[2022], municipality="City of Cape Town") # one city
w  = get_south_africa(DataStore(), years=[2022], municipality="CPT", ward=[1, 2, 3])
```

## What you get

| | |
|---|---|
| Coverage | All nine provinces, 4,468 wards (2020 ward boundaries, Municipal Demarcation Board) |
| Census data | **2022** Stats SA ward product: population, 5-year age bands, sex, population group |
| IDs | `geoid` = 8-digit ward code; `ward_name` e.g. `City of Cape Town - Ward 12`; `ward_label` e.g. `CPT_12` |
| Derived | race shares (`pct_*`), youth / working-age / older shares, dependency and ageing ratios |
| `population_basis` | `stats_sa_2022` (official) or `worldpop_anchored` (modelled, see below) |

## Use it with geosnap's analysis tools

The output is a long-form GeoDataFrame keyed by `geoid` and `year`, which is what geosnap's analysis functions take:

```python
from geosnap import DataStore
from geosnap.analyze import cluster, transition, sequence
from geosnap_southafrica import get_south_africa

gdf = get_south_africa(DataStore(), years=[2015, 2022, 2025], municipality="City of Cape Town")
cols = ["pct_black_african", "pct_coloured", "pct_indian_asian", "pct_white",
        "youth_share", "working_age_share", "older_share"]

gdf = cluster(gdf, columns=cols, method="kmeans", n_clusters=4, random_state=1)   # neighbourhood types
trans = transition(gdf, cluster_col="kmeans")                                      # how types change over time
seq = sequence(gdf, seq_clusters=3, cluster_col="kmeans", dist_type="hamming")     # neighbourhood trajectories
```

`cluster`, `transition` and `sequence` are exercised in the test setup; other `geosnap.analyze` functions take the same `gdf` / `temporal_index="year"` / `unit_index="geoid"` arguments but have not been tried here. Non-2022 years are modelled (see below), so change over time in those runs comes from population counts, not from shifting shares.

## How this differs from geosnap's built-in constructors

| | `geosnap.io.get_census` etc. (US) | `get_south_africa` |
|---|---|---|
| Import | `from geosnap.io import ...` | `from geosnap_southafrica import get_south_africa` |
| Geography filters | `states=`, `counties=`, `boundary=` | `province=`, `municipality=`, `ward=`, `boundary=` |
| Column selection | `columns=` | none - pick columns from the result |
| Time coverage | decennial and ACS vintages, harmonised | one census vintage (2022) + modelled 2015-2030 |
| Variables | income, poverty, education, housing, jobs, ... | population, age, sex, population group (see next section) |
| Status | part of geosnap | separate package; an upstream proposal is sketched in `upstream/` |

## What is included, and what is not

geosnap's US datasets carry far more than population, age, sex and race. This is where this connector stands:

| Topic | Status | How |
|---|---|---|
| Population, age bands, sex, population group | **Official, 2022**, all 4,468 wards | built in |
| Income (median household income, low/high-income shares), dwelling type, tenure, water, household size, household-head unemployment, foreign-born, mobility, cell phone / car / internet | **Official, Census 2011**, re-based to 2020 wards | `census2011=` (DataFirst files, see below) |
| Education, toilet, electricity, refuse | **Supported, needs those DataFirst files** | add the matching `.dta` tables to the same folder |
| Real change over time | **Census 2011 to 2022** two-census panel; other years are *modelled* | `years=[2011, 2022]`; `years=2015..2030` modelled |
| Wealth / deprivation proxy | Proxy | `extras=["rwi"]` (Meta Relative Wealth Index) |
| Schools, clinics, transit, shops | Proxy (OSM, partial coverage) | `extras=["amenities"]` |
| Vegetation, biomes, protected areas, land cover, elevation, climate | Environmental layers | `extras=[...]`, see below |
| Language, disability | Not yet | Census 2011 Community Profiles tables; importer not written |
| Jobs / workplace data (LEHD-like) | Not available | no South African equivalent bundled |
| Night-time lights, heat maps, geology | Not yet | use `add_raster_stats` / `add_polygon_layer` with a file you supply |

Census 2011 ward-level exports from Stats SA's SuperWEB2 use 2011-era ward boundaries, so their ward codes do not match the 2020 ward IDs; they must be re-based (small-area counts are placed in 2020 wards by representative point, see `geosnap_southafrica.census2011`), not joined on code.

## Extra ward variables, fetched automatically

```python
ct = get_south_africa(DataStore(), years=[2022], municipality="CPT", extras=["amenities", "rwi"])
```

| Extra | Columns | Source |
|---|---|---|
| `"amenities"` | `n_schools`, `n_tertiary`, `n_health`, `n_pharmacies`, `n_bus_stops`, `n_rail_stations`, `n_taxi_ranks`, `n_shops`, `n_markets` | OpenStreetMap (ODbL); counts per ward, coverage varies |
| `"rwi"` | `rwi_mean`, `rwi_cells` | Meta Relative Wealth Index, 2.4 km grid, from the Humanitarian Data Exchange. A wealth *proxy*, not income. Its licence is listed as non-commercial: check it before commercial use |

Extras are static (repeated on every year's rows). Amenities are fetched per 0.25° tile and cached, so a city takes seconds; the whole country needs hundreds of requests to the public Overpass servers, so prefer a province or municipality. `extras_errors="warn"` skips a source that cannot be downloaded instead of raising.
Have your own ward table on the 2020 ward codes (for example a Stats SA export)? `add_ward_table(gdf, table, table_on="ward_code")` joins it and refuses if the codes do not match.

## Yearly data

Stats SA publishes the ward product for one vintage (2022) - there is **no official yearly ward series**. For any year 2015-2030 the connector returns a *modelled* series: each ward's 2022 counts scaled by how that ward's WorldPop population changed (annual 100 m rasters, ~100 MB each, downloaded on first use):

```python
sa = get_south_africa(DataStore(), years=range(2015, 2026), municipality="CPT")
sa.groupby("year")["population"].sum()
```

Race and age *shares* are held at 2022 values; only counts change. Treat non-2022 years as estimates, not census counts.

## Filters
`province=`, `municipality=` (name, partial name, or MDB code such as `"CPT"`), `ward=` (id, label, name or number), `boundary=` (any polygon GeoDataFrame).

## Suburbs

```python
ct = get_south_africa(DataStore(), years=[2022], municipality="CPT", suburbs=True)
ct[["ward_name", "main_suburb", "suburb_count", "suburbs"]]
```

- `suburbs=True` / `"osm"`: OpenStreetMap suburb/neighbourhood points, nationwide (coverage varies; (c) OpenStreetMap contributors, ODbL).
- `suburbs="cape_town"`: City of Cape Town official suburb polygons (Cape Town only).
- `suburbs="my_suburbs.shp"` (or URL / GeoDataFrame): any layer; `suburb_name_field=` if the name column is not detected.

With polygon layers, `ward_suburb_overlap()` gives area shares and `interpolate_to_suburbs()` area-weights ward counts onto suburbs (assumes even density in a ward).

## Census 2011 (income, dwellings, services, sub-places)

Stats SA's 2022 ward product only has age, sex and population group. Richer variables come from Census 2011, via the
DataFirst **Census Community Profiles 2011** (free DataFirst login, Stata files, one per topic, at small-area level):

```python
from geosnap_southafrica import get_south_africa, subplaces_2011

# official sub-places (suburbs) with census data - needs only the .dta files
suburbs = subplaces_2011(r"C:\data\census2011")

# 2011 + 2022 on the 2020 wards (needs the small-area polygons to place each small area in a ward)
wards = get_south_africa(years=[2011, 2022], municipality="CPT",
                         census2011=r"C:\data\census2011", sal_geometry=r"C:\data\sal_2011.shp")
```

Each small area goes to the 2020 ward containing its representative point and counts are summed, so totals are preserved
(Cape Town 3.74 M). 2011 rows have `population_basis == "census_2011_rebased"`. Median household income is interpolated
from Stats SA's income bands (annual, nominal 2011 rand). Data are not bundled: download them yourself (licence terms
apply) and keep them out of git.

## Prebuilt tables (faster yearly data)

Only small derived tables are published, never raw source files. `worldpop_ward_sums.parquet` (WorldPop Global2 R2025A,
CC BY 4.0, (c) WorldPop) holds each ward's summed population raster per year, so `years=[2015, ..., 2030]` needs no
raster downloads. It is fetched once from the `data-v1` GitHub Release and cached; if it is not there the connector
computes it locally. Maintainers: `geosnap-southafrica build-cache` (add `--amenities` for the national OpenStreetMap
amenity counts, built from a Geofabrik extract; OSM data (c) OpenStreetMap contributors, ODbL), then attach the files to a release tagged `data-v1`.

**OpenStreetMap coverage is uneven.** Treat amenity counts as *minimums of what is mapped*, not census counts: cities are mapped well
(Cape Town has about 1,000 schools in OSM, close to the real number) but rural wards are under-mapped (nationally OSM holds roughly 11,000 of South
Africa's ~25,000 schools). Compare wards within a city, not a mapped city against a rural area. `n_rail_stations` counts every OSM
`railway=station`/`halt` and can include non-passenger stations.

## Environmental layers (vegetation, land cover, climate, ...)

```python
sa = get_south_africa(municipality="CPT", extras=["vegetation", "threat_status", "protected_areas",
                                                   "landcover", "elevation", "climate_temp", "climate_rain"])
```

| extra | columns | source |
|---|---|---|
| `vegetation`, `biome` | `veg_dominant`, `veg_dominant_share`, `veg_n_classes`, `biome_dominant` ... | SANBI National Vegetation Map 2018 |
| `threat_status` | `threat_dominant`, ... | SANBI NBA 2018 terrestrial ecosystem threat status |
| `protected_areas` | `protected_cover_share` | SANBI BGIS protected areas |
| `cba` | `cba_dominant`, ... | SANBI biodiversity priority areas |
| `landcover` | `lc_share_tree`, `lc_share_built`, `lc_share_water`, ... | ESA WorldCover 2021 (CC BY 4.0) |
| `elevation` | `elevation_mean` | Copernicus GLO-30 |
| `climate_temp`, `climate_heat`, `climate_rain` | `climate_temp_mean`, ... | CHELSA v2.1, 1981-2010 (CC BY 4.0) |

Anything else (geology, temperature or heat-stress maps, tree canopy, species records) works through the generic tools:

```python
from geosnap_southafrica import add_polygon_layer, add_raster_stats, add_point_counts
sa = add_polygon_layer(sa, "geology.gpkg", prefix="geology", class_field="LITHOLOGY", shares=True)
sa = add_raster_stats(sa, "lst_summer.tif", prefix="heat")                    # continuous raster
sa = add_point_counts(sa, "threatened_plants.gpkg", prefix="plants", kind_field="status")
```

SANBI services are read live from BGIS (`bgismaps.sanbi.org`); class fields for the services not yet confirmed are
auto-detected (pass your own layer and `class_field=` if that fails). Rasters are read by window over HTTP, so a city
needs only a few MB. SANBI data are (c) SANBI: check its terms before redistributing. These are landscape-scale
layers: fine for ward summaries, not for site decisions. Vegetation is the *potential* (historical) map, not current cover;
use `landcover` for what is there now.

## Install

```
pip install -e .
```

Python >= 3.10. First use downloads the Stats SA workbook and the ward boundaries, and caches them under the geosnap data directory (`.../south_africa/`).

## Check it works

```
python -m geosnap_southafrica.selftest           # national totals vs Census 2022, provinces, filters, suburbs
python -m geosnap_southafrica.selftest --annual  # also the yearly series
pytest                                           # unit tests (no network)
```

## Status

Unit tests cover the loaders, filters, suburbs and yearly series against synthetic data; the live self-test checks real downloads against published totals. `get_south_africa_population_surface()` (WorldPop 100 m dasymetric surfaces) is experimental. See `docs/methodology.md`. An `upstream/` folder describes how this could be proposed to geosnap itself.

## Data sources and licences
Stats SA (ward statistical product) - Statistics South Africa; Municipal Demarcation Board (ward boundaries); WorldPop (CC BY 4.0); OpenStreetMap (ODbL). Code: MIT (see LICENSE).
