# Upstream geosnap contribution plan

## Target API

Current geosnap 0.17.0 exposes dataset constructors from `geosnap.io`, and those constructors return long-form GeoDataFrames with `geoid` and `year`. The South Africa contribution should therefore follow the native current API:

```python
from geosnap import DataStore
from geosnap import io as gio

store = DataStore()
wards = gio.get_south_africa(store, years=[2022], province="Western Cape")
```

The result should be directly usable by the existing geosnap analysis and harmonisation tools.

## Why this is the right extension point

The existing constructors in `geosnap.io.constructors` all take a `DataStore` and return long-form geodataframes. The South African connector follows that same pattern instead of introducing a parallel `Community` abstraction.

## What belongs in core geosnap

The core contribution should contain:

1. Stats SA 2020 electoral-ward geometry and 2022 ward estimates ingestion (plus modelled yearly series).
2. Deterministic schema selection and validation.
3. Stable `geoid`/`year` long-form output.
4. Clear provenance and limitations.
5. Tests for schema selection, year harmonisation, uniqueness and output shape.

## What should remain optional

WorldPop 100 m refinement is valuable, but it is raster processing rather than a basic geosnap tabular constructor. Keep it as an optional companion function/extra so a standard `pip install geosnap` does not acquire another heavy raster dependency solely for South Africa.

Likewise, OSM/Overpass transport accessibility is application-level analysis and should not be required for the basic connector.

## Community terminology

Older geosnap releases exposed a `Community` abstraction. The current 0.17.0 package root does not export `Community`; it exposes `DataStore`, `geosnap.io`, `geosnap.harmonize`, and `geosnap.analyze`. The upstream PR should therefore not add a second Community class merely for South Africa.

The companion package includes `community_from_south_africa()` as a compatibility bridge for older releases that do expose `Community.from_geodataframes()`.
