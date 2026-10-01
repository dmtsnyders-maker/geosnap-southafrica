import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import geosnap_southafrica.connector as connector
from geosnap_southafrica.census2011 import (derive_indicators, read_sal_counts, sal_to_wards, subplaces_2011,
                                            wards_2011, count_columns)

GEO = {"sp_code": [1, 1, 2, 2], "sp_name": ["Alpha", "Alpha", "Beta", "Beta"], "mp_code": [1] * 4, "mp_name": ["M"] * 4,
       "mn_mdb_c": ["X"] * 4, "mn_code": [1] * 4, "mn_name": ["Town"] * 4, "dc_mdb_c": ["D"] * 4, "dc_code": [1] * 4,
       "dc_name": ["Dist"] * 4, "pr_code": [1] * 4, "pr_name": ["Prov"] * 4}
SALS = [101, 102, 103, 104]


def _write(path, prefix, table, labels):
    df = pd.DataFrame({"sal_code": np.array(SALS, dtype=float), **GEO})
    for i, (lab, vals) in enumerate(zip(labels, table), 1):
        df[f"{prefix}_{i}"] = vals
    names = {f"{prefix}_{i}": f"{lab} - Topic" for i, lab in enumerate(labels, 1)}
    df.to_stata(path, write_index=False, variable_labels={**names, "sal_code": "Small Area Code"})


@pytest.fixture()
def folder(tmp_path):
    _write(tmp_path / "a.dta", "pgrp", [[10, 0, 5, 20], [0, 10, 5, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
           ["Black African", "Coloured", "Indian or Asian", "White", "Other"])
    ages = [str(a) + " - Age in completed years" for a in range(0, 121)]
    df = pd.DataFrame({"sal_code": np.array(SALS, dtype=float), **GEO})
    for a in range(0, 121):
        df[f"age_{a + 1}"] = [1 if a in (3, 30) else 0] * 4
    df.to_stata(tmp_path / "b.dta", write_index=False,
                variable_labels={**{f"age_{a + 1}": f"{a} - Age in completed years" for a in range(121)}, "sal_code": "x"})
    inc = [[0] * 4 for _ in range(12)]
    inc[0] = [1, 1, 1, 1]
    inc[8] = [3, 3, 3, 3]
    _write(tmp_path / "c.dta", "inchh", inc + [[0] * 4],
           ["No income", "R 1 - R 4800", "R 4801 - R 9600", "R 9601 - R 19 600", "R 19 601 - R 38 200", "R 38 201 - R 76 400",
            "R 76 401 - R 153 800", "R 153 801 - R 307 600", "R 307 601 - R 614 400", "R 614 001 - R 1 228 800",
            "R 1 228 801 - R 2 457 600", "R 2 457 601 or more", "Unspecified"])
    _write(tmp_path / "d.dta", "hhsize", [[2] * 4, [1] * 4] + [[0] * 4] * 8, [str(i) for i in range(1, 10)] + ["10+"])
    _write(tmp_path / "e.dta", "dwltyp", [[3] * 4, [0] * 4, [0] * 4, [0] * 4, [0] * 4, [0] * 4, [0] * 4, [1, 0, 1, 0], [0] * 4,
                                          [0] * 4, [0] * 4, [0] * 4, [0] * 4, [0] * 4],
           ["Brick/concrete structure", "Traditional dwelling", "Flat", "Cluster", "Townhouse", "Semi-detached", "Backyard house",
            "Informal dwelling (shack; in backyard)", "Informal dwelling (shack; not in backyard)", "Room/flatlet", "Caravan/tent",
            "Other", "Unspecified", "Not applicable"])
    _write(tmp_path / "skip.dta", "zzz", [[1] * 4, [1] * 4], ["a", "b"])
    return tmp_path


def _wards():
    # SAL 101,102 in ward W1 (x 0-1); 103,104 in W2 (x 1-2)
    return gpd.GeoDataFrame({"geoid": ["W1", "W2"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326)


def _sal_geom():
    boxes = [box(0, 0, .5, 1), box(.5, 0, 1, 1), box(1, 0, 1.5, 1), box(1.5, 0, 2, 1)]
    return gpd.GeoDataFrame({"SAL_CODE": SALS}, geometry=boxes, crs=4326)


def test_read_counts_and_indicators(folder):
    c = read_sal_counts(folder)
    assert len(c) == 4 and c["population"].tolist() == [10, 10, 10, 20]
    assert c["age_0_4_total"].tolist() == [1] * 4 and c["age_30_34_total"].tolist() == [1] * 4
    assert c["dw_formal"].tolist() == [3] * 4 and c["dw_informal"].tolist() == [1, 0, 1, 0]
    ind = derive_indicators(c[count_columns(c)].sum().to_frame().T)
    assert abs(ind.loc[0, "pct_hh_no_income"] - 0.25) < 1e-9 and abs(ind.loc[0, "pct_hh_low_income"] - 0.25) < 1e-9
    assert abs(ind.loc[0, "avg_household_size"] - 4 / 3) < 1e-9  # per small area: 2 one-person + 1 two-person household
    # 3 households per SAL... pct informal 2 of 16 dwellings
    assert abs(ind.loc[0, "pct_informal_dwelling"] - 2 / 14) < 1e-9


def test_median_income_inside_top_band(folder):
    c = read_sal_counts(folder)
    ind = derive_indicators(c[count_columns(c)])
    # 1 no-income + 3 in R307,601-614,400: median falls in that band
    assert (ind["median_hh_income"] > 307600).all()


def test_subplaces_need_no_geometry(folder):
    sp = subplaces_2011(folder)
    assert set(sp["sp_name"]) == {"Alpha", "Beta"}
    assert sp.set_index("sp_name").loc["Alpha", "population"] == 20
    gsp = subplaces_2011(folder, _sal_geom())
    assert isinstance(gsp, gpd.GeoDataFrame) and gsp.geometry.notna().all()


def test_sal_to_wards(folder):
    c = read_sal_counts(folder)
    tab = sal_to_wards(c, _sal_geom(), _wards())
    assert tab.loc["W1", "population"] == 20 and tab.loc["W2", "population"] == 30
    w = wards_2011(folder, _sal_geom(), _wards())
    assert list(w.index) == ["W1", "W2"] and abs(w.loc["W2", "pct_black_african"] - 25 / 30) < 1e-9


def test_get_south_africa_adds_2011_rows(folder, monkeypatch, tmp_path):
    base = _wards().assign(year=2022, ward_code=["W1", "W2"], population=[100.0, 200.0], pct_white=[0.1, 0.2],
                           ward_name=["one", "two"])
    monkeypatch.setattr(connector, "_load_base_longform", lambda *a, **k: base.copy())
    out = connector.get_south_africa(years=[2011, 2022], data_dir=tmp_path / "d", census2011=folder, sal_geometry=_sal_geom())
    assert sorted(out["year"].unique()) == [2011, 2022]
    r = out[(out.geoid == "W2") & (out.year == 2011)].iloc[0]
    assert r["population"] == 30 and r["population_basis"] == "census_2011_rebased"
    assert np.isnan(r["pct_white"]) is np.False_ or np.isnan(r["pct_white"])  # 2022-only columns are blank in 2011 rows unless Census 2011 has them
    assert out[(out.geoid == "W2") & (out.year == 2022)]["population"].iloc[0] == 200
    with pytest.raises(ValueError):
        connector.get_south_africa(years=[2011], data_dir=tmp_path / "d", census2011=folder)
