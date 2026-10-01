import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from geosnap_southafrica.community import _derive_demographic_indicators
from geosnap_southafrica.connector import _hash_request, get_south_africa


def test_hash_request_is_stable():
    assert _hash_request([2022, 2011], "Western Cape") == _hash_request([2011, 2022], "Western Cape")
    assert _hash_request([2011], None) != _hash_request([2022], None)


def test_demographic_indicators_from_age_bands():
    frame = pd.DataFrame(
        {
            "ward_code": ["A"],
            "population_2011": [100.0],
            "pct_black_african_2011": [60],
            "pct_coloured_2011": [30],
            "pct_indian_asian_2011": [5],
            "pct_white_2011": [5],
            "pct_other_2011": [0],
            "age_0_4_total_2011": [10],
            "age_5_9_total_2011": [10],
            "age_10_14_total_2011": [10],
            "age_15_19_total_2011": [10],
            "age_20_24_total_2011": [10],
            "age_25_29_total_2011": [10],
            "age_30_34_total_2011": [10],
            "age_35_39_total_2011": [10],
            "age_65_69_total_2011": [10],
            "age_70_74_total_2011": [10],
        }
    )
    out = _derive_demographic_indicators(frame, 2011)
    assert abs(out.loc[0, "pct_black_african"] - 0.60) < 1e-9
    assert abs(out.loc[0, "youth_share"] - 0.40) < 1e-9
    assert out.loc[0, "older_share"] == 0.20
    assert out.loc[0, "dependency_ratio"] > 0


def test_get_south_africa_returns_long_form_geodataframe(monkeypatch, tmp_path):
    sample = gpd.GeoDataFrame(
        {
            "geoid": ["1_1", "1_1"],
            "year": [2011, 2022],
            "ward_code": ["1_1", "1_1"],
            "population": [100, 120],
        },
        geometry=[box(0, 0, 1, 1), box(0, 0, 1, 1)],
        crs=4326,
    )
    import geosnap_southafrica.connector as connector

    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: sample.copy())
    out = get_south_africa(years=[2022], boundary=gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs=4326), data_dir=tmp_path)
    assert isinstance(out, gpd.GeoDataFrame)
    assert set(out["year"]) == {2022}
    assert {"geoid", "year", "geometry"}.issubset(out.columns)


def test_get_south_africa_filters_outside_boundary(monkeypatch, tmp_path):
    sample = gpd.GeoDataFrame(
        {
            "geoid": ["inside", "outside"],
            "year": [2022, 2022],
            "ward_code": ["inside", "outside"],
            "population": [10, 20],
        },
        geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)],
        crs=4326,
    )
    import geosnap_southafrica.connector as connector
    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: sample.copy())
    boundary = gpd.GeoDataFrame(geometry=[box(-0.1, -0.1, 1.1, 1.1)], crs=4326)
    out = get_south_africa(years=[2022], boundary=boundary, data_dir=tmp_path)
    assert out["geoid"].tolist() == ["inside"]


def test_store_south_africa_accepts_single_year(monkeypatch, tmp_path):
    import geosnap_southafrica.connector as connector
    sample = gpd.GeoDataFrame({"geoid": ["a"], "year": [2022], "population": [1]}, geometry=[box(0, 0, 1, 1)], crs=4326)
    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: sample.copy())
    assert connector.store_south_africa(years=2022, data_dir=tmp_path).exists()


def test_population_surface_wires_paths(monkeypatch, tmp_path):
    import geosnap_southafrica.connector as connector
    sample = gpd.GeoDataFrame({"geoid": ["a"], "year": [2022], "population": [1]}, geometry=[box(0, 0, 1, 1)], crs=4326)
    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: sample.copy())
    seen = {}
    monkeypatch.setattr(connector, "build_worldpop_surfaces", lambda years, frames, raw, out, **k: seen.update(raw=raw, out=out) or {})
    connector.get_south_africa_population_surface(years=[2022], data_dir=tmp_path)
    assert seen["raw"].exists() and seen["out"].exists()


def test_get_south_africa_rejects_2011(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        get_south_africa(years=[2011, 2022], data_dir=tmp_path)


def test_loader_reads_statssa_layout(tmp_path):
    import io, zipfile
    import numpy as np
    from geosnap_southafrica.ward_data import load_official_ward_estimates

    codes = [19100001, 19100002]
    bands = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"]
    age = pd.DataFrame(np.full((2, 18), 100.0), columns=bands); age.insert(0, "Ward_Code", codes)
    race = pd.DataFrame({"Ward_Code": codes, "Black African": [50, 0], "Coloured": [50, 0], "Indian/Asian": [0, 0], "White": [0, 100], "Other": [0, 0], "Total": [100, 100]})
    sex = pd.DataFrame({"WARD_CODE": codes, "Male": [900.0, 900.0], "Female": [900.0, 900.0], "Total": [1800.0, 1800.0]})

    def wb(name, df):
        b = io.BytesIO()
        with pd.ExcelWriter(b, engine="openpyxl") as w:
            df.head(1).rename(columns={df.columns[0]: "Prov/District/Metro/Local municipality"}).to_excel(w, sheet_name="Munic", index=False)
            df.to_excel(w, sheet_name="Wards", index=False)
        return name, b.getvalue()

    z = tmp_path / "w.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for n, d in (("PP_Age Group_x.xlsx", age), ("PP_Population Group_x.xlsx", race), ("PP_Sex_x.xlsx", sex)):
            zf.writestr(*wb(n, d))
    out = load_official_ward_estimates(z)
    assert out["ward_code"].tolist() == ["19100001", "19100002"]
    assert out.loc[0, "pct_black_african_2022"] == 0.5 and out.loc[1, "pct_white_2022"] == 1.0
    assert out.loc[0, "population_2022"] == 1800.0
    assert "age_85_plus_total_2022" in out.columns


def test_municipality_filter_by_name_and_code(monkeypatch, tmp_path):
    import pytest
    import geosnap_southafrica.connector as connector
    sample = gpd.GeoDataFrame(
        {"geoid": ["1", "2"], "year": [2022, 2022], "municipality_name": ["City of Cape Town Metropolitan Municipality", "Drakenstein Local Municipality"],
         "municipality_code": ["CPT", "WC023"], "population": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)], crs=4326)
    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: sample.copy())
    assert get_south_africa(years=[2022], municipality="city of cape town", data_dir=tmp_path)["geoid"].tolist() == ["1"]
    assert get_south_africa(years=[2022], municipality="wc023", data_dir=tmp_path)["geoid"].tolist() == ["2"]
    with pytest.raises(ValueError):
        get_south_africa(years=[2022], municipality="Nowhere", data_dir=tmp_path)


def test_fetch_arcgis_geojson_pages_and_string_oid_field(monkeypatch, tmp_path):
    import geosnap_southafrica.downloads as dl

    total, page = 5, 2
    feats = [{"type": "Feature", "properties": {"WardID": str(i)},
              "geometry": {"type": "Point", "coordinates": [i, 0]}} for i in range(total)]
    seen = []

    class R:
        status_code = 200

        def __init__(self, data): self.data = data
        def raise_for_status(self): pass
        def json(self): return self.data

    def fake_get(url, params=None, **k):
        if url.endswith("/query"):
            seen.append(params["resultOffset"])
            o = params["resultOffset"]
            return R({"features": feats[o:o + params["resultRecordCount"]]})
        return R({"objectIdField": "FID", "maxRecordCount": page, "fields": []})

    monkeypatch.setattr(dl.requests, "get", fake_get)
    out = dl.fetch_arcgis_geojson("https://x/FeatureServer/0", tmp_path / "w.geojson")
    assert len(gpd.read_file(out)) == total
    assert seen == [0, 2, 4]


def test_municipality_partial_match_and_ambiguity(monkeypatch, tmp_path):
    import pytest
    import geosnap_southafrica.connector as connector
    sample = gpd.GeoDataFrame(
        {"geoid": ["1", "2", "3"], "year": [2022] * 3, "population": [1, 2, 3],
         "municipality_name": ["City of Cape Town Metropolitan Municipality", "Cape Agulhas Local Municipality", "Drakenstein Local Municipality"],
         "municipality_code": ["CPT", "WC033", "WC023"]},
        geometry=[box(i, 0, i + 1, 1) for i in range(3)], crs=4326)
    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: sample.copy())
    assert get_south_africa(years=[2022], municipality="City of Cape Town", data_dir=tmp_path)["geoid"].tolist() == ["1"]
    with pytest.raises(ValueError, match="ambiguous"):
        get_south_africa(years=[2022], municipality="Cape", data_dir=tmp_path)


def test_ward_names_and_ward_filter(monkeypatch, tmp_path):
    import geosnap_southafrica.connector as connector
    sample = gpd.GeoDataFrame(
        {"geoid": ["19100001", "19100002"], "year": [2022, 2022], "population": [1, 2], "ward_number": ["1", "2"],
         "ward_label": ["CPT_1", "CPT_2"], "municipality_name": ["City of Cape Town Metropolitan Municipality"] * 2,
         "municipality_code": ["CPT", "CPT"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326)
    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: sample.copy())
    allw = get_south_africa(years=[2022], data_dir=tmp_path)
    assert allw["ward_name"].tolist() == ["City of Cape Town - Ward 1", "City of Cape Town - Ward 2"]
    assert get_south_africa(years=[2022], ward="CPT_2", data_dir=tmp_path)["geoid"].tolist() == ["19100002"]
    assert get_south_africa(years=[2022], ward=[19100001], data_dir=tmp_path)["geoid"].tolist() == ["19100001"]
    assert get_south_africa(years=[2022], municipality="CPT", ward=2, data_dir=tmp_path)["geoid"].tolist() == ["19100002"]
