import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import geosnap_southafrica.connector as connector
import geosnap_southafrica.extras as ex
from geosnap_southafrica import add_ward_table, get_south_africa


def _wards():
    return gpd.GeoDataFrame({"geoid": ["W1", "W2"], "year": 2022, "population": [10.0, 20.0]},
                            geometry=[box(18.0, -34.0, 18.1, -33.9), box(18.1, -34.0, 18.2, -33.9)], crs=4326)


def _osm_payload():
    els = [
        {"type": "node", "id": 1, "lat": -33.95, "lon": 18.05, "tags": {"amenity": "school"}},
        {"type": "node", "id": 2, "lat": -33.96, "lon": 18.06, "tags": {"amenity": "school"}},
        {"type": "way", "id": 3, "center": {"lat": -33.95, "lon": 18.15}, "tags": {"amenity": "clinic"}},
        {"type": "node", "id": 4, "lat": -33.95, "lon": 18.16, "tags": {"highway": "bus_stop"}},
        {"type": "node", "id": 5, "lat": -33.95, "lon": 18.17, "tags": {"shop": "supermarket"}},
        {"type": "node", "id": 6, "lat": -33.95, "lon": 18.17, "tags": {"amenity": "bench"}},   # ignored
        {"type": "node", "id": 7, "lat": -10.0, "lon": 30.0, "tags": {"amenity": "school"}},    # outside every ward
    ]
    return {"elements": els}


def test_classify_tags():
    assert ex.classify_tags({"amenity": "university"}) == "n_tertiary"
    assert ex.classify_tags({"railway": "station"}) == "n_rail_stations"
    assert ex.classify_tags({"amenity": "bench"}) is None


def test_amenity_counts_per_ward_and_cache(monkeypatch, tmp_path):
    calls = []

    class R:
        def raise_for_status(self): pass
        def json(self): return _osm_payload()

    monkeypatch.setattr(ex.requests, "post", lambda url, **k: calls.append(url) or R())
    t = ex.osm_amenity_counts(_wards(), tmp_path).set_index("geoid")
    assert t.loc["W1", "n_schools"] == 2 and t.loc["W2", "n_schools"] == 0
    assert t.loc["W2", "n_health"] == 1 and t.loc["W2", "n_bus_stops"] == 1 and t.loc["W2", "n_shops"] == 1
    n = len(calls)
    ex.osm_amenity_counts(_wards(), tmp_path)
    assert len(calls) == n                                   # served from the tile cache


def test_tiles_cover_bounds():
    tiles = ex._tiles((18.0, -34.0, 18.7, -33.9))
    assert min(t[0] for t in tiles) <= 18.0 and max(t[2] for t in tiles) >= 18.7
    assert min(t[1] for t in tiles) <= -34.0 and max(t[3] for t in tiles) >= -33.9


def test_find_rwi_url():
    pkg = {"result": {"resources": [
        {"name": "South Sudan_relative_wealth_index.csv", "url": "https://x/south-sudan.csv"},
        {"name": "South Africa_relative_wealth_index.csv", "url": "https://x/south-africa.csv"}]}}
    assert ex.find_rwi_url(pkg).endswith("south-africa.csv")
    with pytest.raises(RuntimeError):
        ex.find_rwi_url({"result": {"resources": []}})


def _rwi_csv(path):
    pd.DataFrame({"latitude": [-33.95, -33.96, -33.95, -33.50], "longitude": [18.05, 18.06, 18.15, 18.15],
                  "rwi": [1.0, 3.0, -1.0, 0.0], "error": 0.1}).to_csv(path, index=False)


def test_rwi_by_ward_mean_and_nearest_fallback(tmp_path):
    _rwi_csv(tmp_path / "rwi_south_africa.csv")                 # cached file -> no network
    w = pd.concat([_wards(), gpd.GeoDataFrame({"geoid": ["W3"], "year": 2022, "population": [1.0]},
                                              geometry=[box(18.2, -34.0, 18.25, -33.9)], crs=4326)])
    t = ex.rwi_by_ward(gpd.GeoDataFrame(w, crs=4326), tmp_path).set_index("geoid")
    assert t.loc["W1", "rwi_mean"] == pytest.approx(2.0) and t.loc["W1", "rwi_cells"] == 2
    assert t.loc["W2", "rwi_mean"] == pytest.approx(-1.0)
    assert t.loc["W3", "rwi_cells"] == 0 and t.loc["W3", "rwi_mean"] == pytest.approx(-1.0)   # nearest cell (~5 km away)


def test_get_south_africa_extras_and_warn(monkeypatch, tmp_path):
    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: _wards())
    monkeypatch.setattr(ex, "osm_amenity_counts", lambda w, c, **k: pd.DataFrame({"geoid": ["W1", "W2"], "n_schools": [2, 0]}))
    out = get_south_africa(years=[2022], data_dir=tmp_path, extras=["amenities"])
    assert out.set_index("geoid").loc["W1", "n_schools"] == 2 and len(out) == 2

    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(ex, "rwi_by_ward", boom)
    out = get_south_africa(years=[2022], data_dir=tmp_path, extras=["rwi"], extras_errors="warn")
    assert "rwi_mean" not in out.columns
    with pytest.raises(RuntimeError):
        get_south_africa(years=[2022], data_dir=tmp_path, extras=["rwi"])
    with pytest.raises(ValueError, match="Unknown extras"):
        get_south_africa(years=[2022], data_dir=tmp_path, extras=["nope"])


def test_add_ward_table_join_and_mismatch():
    g = _wards()
    t = pd.DataFrame({"ward": ["W1", "W2"], "median_income": [100, 200]})
    out = add_ward_table(g, t, table_on="ward")
    assert out.set_index("geoid").loc["W2", "median_income"] == 200
    with pytest.raises(ValueError, match="matched"):
        add_ward_table(g, pd.DataFrame({"geoid": ["X1", "X2"], "v": [1, 2]}))
