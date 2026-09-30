"""Text to add to geosnap.io.constructors in an upstream PR."""

# 1. Add "get_south_africa" to __all__.
#
# __all__ = [
#     ...,
#     "get_south_africa",
# ]
#
# 2. Import it near the other constructors.
#
# from .south_africa import get_south_africa
#
# 3. Add geosnap/io/south_africa.py containing the South Africa connector implementation.
#
# 4. Add tests and documentation. Do not add rasterio, OSM/Overpass, or other
#    heavy optional dependencies to the geosnap core dependency set merely to
#    support the connector. WorldPop raster refinement belongs in an optional
#    companion module / extra.
