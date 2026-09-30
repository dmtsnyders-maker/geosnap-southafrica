"""South Africa connector for geosnap."""

from .connector import get_south_africa, get_south_africa_population_surface, store_south_africa
from .legacy import community_from_south_africa
from .suburbs import interpolate_to_suburbs, load_suburbs, ward_suburb_overlap

__all__ = [
    "get_south_africa", "get_south_africa_population_surface", "store_south_africa", "community_from_south_africa",
    "load_suburbs", "ward_suburb_overlap", "interpolate_to_suburbs",
]

try:
    from importlib.metadata import version
    __version__ = version("geosnap-southafrica")
except Exception:
    __version__ = "0.5.1"
