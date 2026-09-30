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
Stats SA (ward statistical product) - Statistics South Africa; Municipal Demarcation Board (ward boundaries); WorldPop (CC BY 4.0); OpenStreetMap (ODbL). Code: BSD-3-Clause.
