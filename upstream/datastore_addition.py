"""Optional upstream DataStore addition."""

# A minimal upstream DataStore method can be added if maintainers want the
# connector to be discoverable as ds.south_africa(...):
#
# def south_africa(self, years=(2022,), province=None, boundary=None, execute=True):
#     from .io.south_africa import _get_south_africa_from_datastore
#     gdf = _get_south_africa_from_datastore(self, years=years, province=province, boundary=boundary)
#     return gdf if execute else gdf
#
# This is optional. The cleaner first contribution is geosnap.io.get_south_africa(...),
# mirroring the existing geosnap.io.get_census(...) pattern.
