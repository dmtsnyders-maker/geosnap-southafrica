import geopandas as gpd
import pytest
from shapely.geometry import box

from geosnap_southafrica import get_south_africa, interpolate_to_suburbs, load_suburbs, ward_suburb_overlap
from geosnap_southafrica.suburbs import add_suburb_columns


def _wards():
    # two 10x10 wards side by side, in degrees near Cape Town
    x0, y0, s = 18.4, -34.0, 0.1
    return gpd.GeoDataFrame(
        {"geoid": ["W1", "W2"], "year": [2022, 2022], "population": [1000.0, 500.0], "pct_white": [0.2, 0.6]},
        geometry=[box(x0, y0, x0 + s, y0 + s), box(x0 + s, y0, x0 + 2 * s, y0 + s)], crs=4326)


def _suburbs():
    x0, y0, s = 18.4, -34.0, 0.1
    # A: left 60% of W1; B: right 40% of W1 plus the left 50% of W2; C: right 50% of W2
    return gpd.GeoDataFrame(
        {"OFC_SBRB_NAME": ["Alpha", "Beta", "Gamma"], "OBJECTID": ["1", "2", "3"], "note": ["x", "x", "x"]},
        geometry=[box(x0, y0, x0 + 0.06 * 1, y0 + s), box(x0 + 0.06, y0, x0 + 0.15, y0 + s), box(x0 + 0.15, y0, x0 + 0.2, y0 + s)],
        crs=4326)


def test_name_field_autodetected():
    sub = load_suburbs(_suburbs())
    assert sorted(sub["suburb_name"]) == ["Alpha", "Beta", "Gamma"]


def test_overlap_shares_sum_to_one_per_ward():
    sub = load_suburbs(_suburbs())
    ov = ward_suburb_overlap(_wards(), sub)
    assert ov.groupby("geoid")["share_of_ward"].sum().round(6).tolist() == [1.0, 1.0]
    w1 = ov[ov.geoid == "W1"].set_index("suburb_name")["share_of_ward"]
    assert w1["Alpha"] == pytest.approx(0.6, abs=0.02) and w1["Beta"] == pytest.approx(0.4, abs=0.02)


def test_suburb_columns_main_and_list():
    out = add_suburb_columns(_wards(), load_suburbs(_suburbs()))
    r1, r2 = out.set_index("geoid").loc["W1"], out.set_index("geoid").loc["W2"]
    assert r1["main_suburb"] == "Alpha" and r1["suburb_count"] == 2 and "Beta" in r1["suburbs"]
    assert r2["suburb_count"] == 2 and r2["main_suburb"] in {"Beta", "Gamma"}


def test_interpolation_preserves_population_total():
    sub = load_suburbs(_suburbs())
    res = interpolate_to_suburbs(_wards(), sub, extensive=["population"], intensive=["pct_white"])
    assert res["population"].sum() == pytest.approx(1500.0, rel=1e-3)
    assert res["pct_white"].between(0, 1).all()


def test_get_south_africa_with_suburbs(monkeypatch, tmp_path):
    import geosnap_southafrica.connector as connector
    w = _wards()
    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: w.copy())
    out = get_south_africa(years=[2022], data_dir=tmp_path, suburbs=_suburbs())
    assert {"main_suburb", "suburbs", "suburb_count"} <= set(out.columns)
    assert out["suburb_count"].min() >= 1


def test_bad_name_field_raises():
    with pytest.raises(ValueError):
        load_suburbs(_suburbs(), name_field="nope")


def test_osm_parse_and_point_assignment(monkeypatch, tmp_path):
    from geosnap_southafrica import osm_suburbs
    payload = {"elements": [
        {"type": "node", "id": 1, "lat": -33.95, "lon": 18.45, "tags": {"name": "Sunnyside", "place": "suburb"}},
        {"type": "way", "id": 2, "center": {"lat": -33.95, "lon": 18.55}, "tags": {"name": "Bellville", "place": "suburb"}},
        {"type": "node", "id": 3, "lat": -33.95, "lon": 18.46, "tags": {"name": "Rosebank", "place": "neighbourhood"}},
        {"type": "node", "id": 4, "lat": -33.95, "lon": 18.47, "tags": {"place": "suburb"}},   # unnamed -> dropped
    ]}
    pts = osm_suburbs.parse_elements(payload)
    assert sorted(pts["suburb_name"]) == ["Bellville", "Rosebank", "Sunnyside"]

    class R:
        def raise_for_status(self): pass
        def json(self): return payload
    calls = []
    monkeypatch.setattr(osm_suburbs.requests, "post", lambda url, **k: calls.append(url) or R())
    got = load_suburbs("osm", cache_dir=tmp_path, bbox=(18.3, -34.3, 18.7, -33.9))
    assert len(got) == 3 and len(calls) == 1
    load_suburbs("osm", cache_dir=tmp_path, bbox=(18.3, -34.3, 18.7, -33.9))
    assert len(calls) == 1                                          # second call served from cache

    out = add_suburb_columns(_wards(), got)                         # W1 = 18.4-18.5, W2 = 18.5-18.6
    r = out.set_index("geoid")
    assert r.loc["W1", "suburb_count"] == 2
    assert sorted(r.loc["W1", "suburbs"].split("; ")) == ["Rosebank", "Sunnyside"]
    assert r.loc["W2", "main_suburb"] == "Bellville"
    with pytest.raises(ValueError, match="polygons"):
        interpolate_to_suburbs(_wards(), got, extensive=["population"])


def test_overpass_query_scopes():
    from geosnap_southafrica.osm_suburbs import build_query
    assert 'ISO3166-1"="ZA"' in build_query(None)
    assert "(-34.30000,18.30000,-33.90000,18.70000)" in build_query((18.3, -34.3, 18.7, -33.9))
