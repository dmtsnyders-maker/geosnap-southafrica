import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geosnap_southafrica.environment import (add_point_counts, add_polygon_layer, add_raster_stats, copernicus_dem_urls,
                                             layer_overlap_table, worldcover_urls)


def _wards():
    return gpd.GeoDataFrame({"geoid": ["W1", "W2"]}, geometry=[box(18, -34, 18.5, -33.5), box(18.5, -34, 19, -33.5)], crs=4326)


def _veg():
    return gpd.GeoDataFrame({"Name_18": ["Fynbos", "Renosterveld", "Fynbos"]},
                            geometry=[box(18, -34, 18.4, -33.5), box(18.4, -34, 18.9, -33.5), box(18.9, -34, 19, -33.5)], crs=4326)


def test_dominant_class_and_shares():
    out = add_polygon_layer(_wards(), _veg(), prefix="veg", hints=["Name_18"], shares=True)
    r = out.set_index("geoid")
    assert r.loc["W1", "veg_dominant"] == "Fynbos" and abs(r.loc["W1", "veg_dominant_share"] - 0.8) < 0.02
    assert r.loc["W2", "veg_dominant"] == "Renosterveld" and r.loc["W2", "veg_n_classes"] == 2
    assert abs(r.loc["W2", "veg_share_fynbos"] - 0.2) < 0.02
    t = layer_overlap_table(_wards(), _veg(), "Name_18")
    assert set(t.columns) == {"geoid", "class", "share_of_ward", "area_km2"}


def test_coverage_layer():
    pa = gpd.GeoDataFrame(geometry=[box(18, -34, 18.25, -33.5)], crs=4326)
    out = add_polygon_layer(_wards(), pa, prefix="protected").set_index("geoid")
    assert abs(out.loc["W1", "protected_cover_share"] - 0.5) < 0.02 and out.loc["W2", "protected_cover_share"] == 0


def _raster(path, values, scale=None):
    tr = from_origin(18, -33.5, 0.5, 0.5)   # 2 cols x 2 rows over both wards' extent (x 18-19, y -34..-33)
    with rasterio.open(path, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32", crs=4326, transform=tr, nodata=-9999) as d:
        d.write(np.array(values, dtype="float32"), 1)
        if scale:
            d.scales, d.offsets = (scale[0],), (scale[1],)


def test_raster_mean_scale_offset_and_categorical(tmp_path):
    _raster(tmp_path / "t.tif", [[2990, 2990], [2990, 2990]], scale=(0.1, -273.15))
    out = add_raster_stats(_wards(), tmp_path / "t.tif", prefix="temp").set_index("geoid")
    assert abs(out.loc["W1", "temp_mean"] - 25.85) < 0.01
    _raster(tmp_path / "c.tif", [[10, 50], [10, 50]])
    lc = add_raster_stats(_wards(), tmp_path / "c.tif", prefix="lc", categorical=True, class_names={10: "tree", 50: "built"}).set_index("geoid")
    assert lc.loc["W1", "lc_dominant"] == 10 and lc.loc["W2", "lc_share_built"] == 1.0


def test_point_counts():
    pts = gpd.GeoDataFrame({"kind": ["a", "b", "a"]}, geometry=[box(18.1, -33.9, 18.1, -33.9).centroid, box(18.2, -33.8, 18.2, -33.8).centroid,
                                                                  box(18.7, -33.8, 18.7, -33.8).centroid], crs=4326)
    out = add_point_counts(_wards(), pts, prefix="plants", kind_field="kind").set_index("geoid")
    assert out.loc["W1", "plants_count"] == 2 and out.loc["W2", "plants_count_a"] == 1


def test_tile_urls():
    urls = worldcover_urls((18.3, -34.1, 18.9, -33.5))
    assert any("S36E018" in u for u in urls) and all(u.endswith("_Map.tif") for u in urls)
    assert any("S34_00_E018_00" in u for u in copernicus_dem_urls((18.3, -33.9, 18.9, -33.5)))


def test_http_get_retries_with_browser_agent_on_403(monkeypatch):
    import geosnap_southafrica.downloads as d

    seen = []

    class R:
        def __init__(self, code): self.status_code = code
        def close(self): pass

    def fake_get(url, params=None, headers=None, timeout=None, stream=False):
        seen.append(headers["User-Agent"])
        return R(403 if len(seen) == 1 else 200)

    monkeypatch.setattr(d.requests, "get", fake_get)
    r = d.http_get("https://example.org/x")
    assert r.status_code == 200 and seen[0] == d.USER_AGENT and seen[1] == d.BROWSER_UA


def test_http_get_retries_with_browser_agent_on_403(monkeypatch):
    import geosnap_southafrica.downloads as d

    seen = []

    class R:
        def __init__(self, code):
            self.status_code = code

        def close(self):
            pass

    def fake_get(url, params=None, headers=None, timeout=None, stream=False):
        seen.append(headers["User-Agent"])
        return R(403 if len(seen) == 1 else 200)

    monkeypatch.setattr(d.requests, "get", fake_get)
    r = d.http_get("https://example.org/x")
    assert r.status_code == 200 and seen[0] == d.USER_AGENT and seen[1] == d.BROWSER_UA
