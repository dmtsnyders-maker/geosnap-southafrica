"""South Africa connector for geosnap."""

from .census2011 import subplaces_2011, wards_2011, read_sal_counts
from .environment import add_point_counts, add_polygon_layer, add_raster_stats, layer_overlap_table, load_layer
from .connector import get_south_africa, get_south_africa_population_surface, store_south_africa
from .legacy import community_from_south_africa
from .extras import add_ward_table
from .suburbs import interpolate_to_suburbs, load_suburbs, ward_suburb_overlap

__all__ = ["add_polygon_layer", "add_raster_stats", "add_point_counts", "layer_overlap_table", "load_layer", "subplaces_2011", "wards_2011", "read_sal_counts", 
    "get_south_africa", "get_south_africa_population_surface", "store_south_africa", "community_from_south_africa",
    "load_suburbs", "ward_suburb_overlap", "interpolate_to_suburbs", "add_ward_table",
]

try:
    from importlib.metadata import version
    __version__ = version("geosnap-southafrica")
except Exception:
    __version__ = "0.9.6"
