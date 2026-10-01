import geopandas as gpd
import pytest
from shapely.geometry import box

osmium = pytest.importorskip("osmium")

from geosnap_southafrica.extras import osm_amenity_counts  # noqa: E402
from geosnap_southafrica.osm_pbf import build_amenity_table, read_pbf_points  # noqa: E402


def _pbf(path):
    with osmium.SimpleWriter(str(path)) as w:
        w.add_node(osmium.osm.mutable.Node(id=1, location=(18.1, -33.9), tags={"amenity": "school"}))
        w.add_node(osmium.osm.mutable.Node(id=2, location=(18.2, -33.8), tags={"amenity": "pharmacy"}))
        w.add_node(osmium.osm.mutable.Node(id=3, location=(18.7, -33.8), tags={"highway": "bus_stop"}))
        w.add_node(osmium.osm.mutable.Node(id=4, location=(18.1, -33.7), tags={"amenity": "bench"}))   # ignored


def test_pbf_counts_and_prebuilt_use(tmp_path, monkeypatch):
    pbf = tmp_path / "x.osm.pbf"
    _pbf(pbf)
    pts = read_pbf_points(pbf)
    assert sorted(pts["kind"]) == ["n_bus_stops", "n_pharmacies", "n_schools"]
    wards = gpd.GeoDataFrame({"geoid": ["W1", "W2"]}, geometry=[box(18, -34, 18.5, -33.5), box(18.5, -34, 19, -33.5)], crs=4326)
    out = build_amenity_table(wards, tmp_path / "prebuilt" / "osm_amenity_ward_counts.parquet", pbf_path=pbf)
    import pandas as pd
    t = pd.read_parquet(out).set_index("geoid")
    assert t.loc["W1", "n_schools"] == 1 and t.loc["W2", "n_bus_stops"] == 1
    monkeypatch.delenv("GEOSNAP_SA_NO_PREBUILT", raising=False)
    res = osm_amenity_counts(wards, tmp_path)   # served from the prebuilt table, no network
    assert res.set_index("geoid").loc["W1", "n_pharmacies"] == 1
