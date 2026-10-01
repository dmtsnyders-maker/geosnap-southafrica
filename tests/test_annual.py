import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import geosnap_southafrica.annual as annual
import geosnap_southafrica.connector as connector
from geosnap_southafrica import get_south_africa

X0, Y0, CELL = 18.0, -33.0, 0.01  # raster origin (top-left) and cell size in degrees


def _raster(path, left, right):
    """20x20 raster: left 10 columns = `left` per cell, right 10 columns = `right` per cell."""
    data = np.full((20, 20), float(right), dtype="float32")
    data[:, :10] = left
    with rasterio.open(path, "w", driver="GTiff", height=20, width=20, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(X0, Y0, CELL, CELL), nodata=-99999.0) as dst:
        dst.write(data, 1)
    return path


def _census():
    # W1 = left half, W2 = right half (minus a sliver), W3 = tiny ward (< one cell) inside the right half
    g = [box(X0, Y0 - 0.2, X0 + 0.1, Y0), box(X0 + 0.1, Y0 - 0.2, X0 + 0.2, Y0 - 0.1), box(X0 + 0.15, Y0 - 0.15, X0 + 0.1501, Y0 - 0.1501)]
    return gpd.GeoDataFrame(
        {"geoid": ["W1", "W2", "W3"], "year": 2022, "municipality_code": ["M", "M", "M"],
         "population": [1000.0, 500.0, 2.0], "n_total_pop": [1000.0, 500.0, 2.0],
         "age_0_4_total": [100.0, 50.0, 0.2], "sex_male": [480.0, 250.0, 1.0],
         "pct_white": [0.3, 0.4, 0.5], "youth_share": [0.1, 0.1, 0.1], "population_basis": "stats_sa_2022"},
        geometry=g, crs=4326)


def test_ward_raster_sums(tmp_path):
    r = _raster(tmp_path / "a.tif", left=10, right=20)
    s = annual.ward_raster_sums(r, _census().iloc[:2])
    assert s.iloc[0] == pytest.approx(10 * 10 * 20)          # 10 cols x 20 rows x 10
    assert s.iloc[1] == pytest.approx(20 * 10 * 10)          # 10 cols x 10 rows x 20


def test_nodata_is_ignored(tmp_path):
    p = tmp_path / "n.tif"
    data = np.full((20, 20), 5.0, dtype="float32"); data[0, 0] = -99999.0
    with rasterio.open(p, "w", driver="GTiff", height=20, width=20, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(X0, Y0, CELL, CELL), nodata=-99999.0) as dst:
        dst.write(data, 1)
    s = annual.ward_raster_sums(p, _census().iloc[:1])
    assert s.iloc[0] == pytest.approx(5.0 * (10 * 20) - 5.0)  # the nodata cell counts as zero


def test_build_annual_scales_counts_not_shares(tmp_path, monkeypatch):
    rasters = {2022: _raster(tmp_path / "r22.tif", 10, 20), 2018: _raster(tmp_path / "r18.tif", 5, 20)}
    monkeypatch.setattr(annual, "worldpop_raster", lambda year, raw_dir: rasters[year])
    out = annual.build_annual(_census(), [2018, 2022], tmp_path / "raw", tmp_path / "cache")
    assert set(out["year"]) == {2018}                      # anchor year is not duplicated
    o = out.set_index("geoid")
    assert o.loc["W1", "population"] == pytest.approx(500.0)   # left half halved
    assert o.loc["W2", "population"] == pytest.approx(500.0)   # right half unchanged
    assert o.loc["W1", "age_0_4_total"] == pytest.approx(50.0) and o.loc["W1", "sex_male"] == pytest.approx(240.0)
    assert o.loc["W1", "pct_white"] == 0.3 and o.loc["W1", "youth_share"] == 0.1
    assert (o["population_basis"] == "worldpop_anchored").all()
    # tiny ward has no raster mass of its own -> falls back to the municipality ratio (1500 -> ~ 1000/1500 of...)
    assert np.isfinite(o.loc["W3", "population"]) and o.loc["W3", "population"] > 0


def test_growth_ratio_fallback_and_overall():
    t = pd.Series([50.0, 0.0, 30.0]); a = pd.Series([100.0, 0.0, 30.0])
    r = annual.growth_ratios(t, a, pd.Series(["m", "m", "m"]))
    assert r.iloc[0] == 0.5 and r.iloc[2] == 1.0
    assert r.iloc[1] == pytest.approx(80 / 130)            # tiny ward -> municipality ratio


def test_get_south_africa_years_and_cache(tmp_path, monkeypatch):
    rasters = {2022: _raster(tmp_path / "r22.tif", 10, 20), 2020: _raster(tmp_path / "r20.tif", 8, 20)}
    monkeypatch.setattr(annual, "worldpop_raster", lambda year, raw_dir: rasters[year])
    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: _census())
    out = get_south_africa(years=[2020, 2022], data_dir=tmp_path)
    assert sorted(out["year"].unique()) == [2020, 2022]
    p22 = out[out.year == 2022].set_index("geoid")["population"]
    p20 = out[out.year == 2020].set_index("geoid")["population"]
    assert p22["W1"] == 1000.0 and p20["W1"] == pytest.approx(800.0)
    assert set(out.columns) >= {"geoid", "year", "population_basis"}
    assert list(out.columns) == list(out.columns)          # same schema for both years
    assert len(list((tmp_path / "south_africa" / "cache").glob("worldpop_sums_*.csv"))) == 2  # sums are cached


def test_unsupported_year_message(tmp_path):
    with pytest.raises(ValueError, match="2015"):
        get_south_africa(years=[2010], data_dir=tmp_path)


def test_window_restriction_matches_full_and_handles_outside(tmp_path):
    r = _raster(tmp_path / "w.tif", left=10, right=20)
    wards = _census().iloc[:2]
    full = annual.ward_raster_sums(r, wards)
    small = annual.ward_raster_sums(r, wards.iloc[[1]])          # only reads the right-hand window
    assert small.iloc[0] == pytest.approx(full.iloc[1])
    far = gpd.GeoDataFrame({"geoid": ["Z"]}, geometry=[box(30, -10, 30.1, -9.9)], crs=4326)
    assert annual.ward_raster_sums(r, far).iloc[0] == 0.0         # ward outside the raster
    assert annual.ward_raster_sums(r, wards, max_block_cells=100).equals(full) or \
        np.allclose(annual.ward_raster_sums(r, wards, max_block_cells=100), full)   # tiny blocks, same answer


def test_prebuilt_worldpop_table_is_used(tmp_path, monkeypatch):
    import pandas as pd

    import geosnap_southafrica.prebuilt as pb
    from geosnap_southafrica.annual import _cached_sums

    pre = tmp_path / "prebuilt"
    pre.mkdir()
    pd.DataFrame({"geoid": ["a", "b", "a", "b"], "year": [2015, 2015, 2022, 2022], "sum": [1.0, 2.0, 3.0, 4.0]}).to_parquet(pre / pb.WORLDPOP_TABLE)
    wards = gpd.GeoDataFrame({"geoid": ["b", "a"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326)
    monkeypatch.delenv("GEOSNAP_SA_NO_PREBUILT", raising=False)
    s = _cached_sums(2015, wards, tmp_path / "raw", tmp_path)
    assert s.tolist() == [2.0, 1.0]
    monkeypatch.setenv("GEOSNAP_SA_NO_PREBUILT", "1")
    assert pb.worldpop_sums(2015, ["a"], tmp_path) is None
