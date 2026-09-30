# Upstream file map

The companion package is intentionally separable from geosnap core. For an upstream PR, use this mapping:

```text
geosnap_southafrica/connector.py       -> geosnap/io/south_africa.py (core connector)
geosnap_southafrica/ward_data.py       -> geosnap/io/south_africa_data.py or internal helpers
geosnap_southafrica/spatial.py         -> geosnap/io/south_africa_data.py or internal helpers
geosnap_southafrica/downloads.py       -> geosnap/io/south_africa_data.py or internal helpers
```

Then add to `geosnap/io/constructors.py`:

```python
from .south_africa import get_south_africa
```

and add `"get_south_africa"` to `__all__`.

The constructor should be callable as:

```python
from geosnap import DataStore
from geosnap import io as gio

wards = gio.get_south_africa(DataStore(), years=[2022], province="Western Cape")
```

The connector must return the same long-form `GeoDataFrame` convention as the existing geosnap constructors.

Do not move `worldpop.py`, `transport.py`, or the full analytical pipeline into geosnap core as mandatory dependencies. Those are companion capabilities and can be proposed separately.
