from __future__ import annotations

import logging
import re

import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
RACE_BASE = ["black_african", "coloured", "indian_asian", "white", "other"]


def _derive_demographic_indicators(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = df.copy()
    suffix = f"_{year}"
    race_cols = []
    for race in RACE_BASE:
        src = f"pct_{race}{suffix}"
        if src in out:
            dst = f"pct_{race}"
            out[dst] = pd.to_numeric(out[src], errors="coerce")
            race_cols.append(dst)
    if race_cols:
        med = out[race_cols].stack().median()
        if pd.notna(med) and med > 1.5:
            out[race_cols] = out[race_cols] / 100.0
        out[race_cols] = out[race_cols].clip(0, 1)
        row_total = out[race_cols].sum(axis=1, skipna=True)
        valid = row_total > 0
        out.loc[valid, race_cols] = out.loc[valid, race_cols].div(row_total[valid], axis=0)

    pcol = f"population_{year}"
    if pcol in out:
        out["population"] = pd.to_numeric(out[pcol], errors="coerce")

    age_cols = [c for c in out.columns if c.startswith("age_") and c.endswith(suffix)]
    bands: dict[str, pd.Series] = {}
    for col in age_cols:
        raw = str(col).lower()
        m = re.search(r"age_(\d{1,2})_(\d{1,2})_(male|female|total)_\d{4}$", raw)
        if m:
            band = f"{m.group(1)}_{m.group(2)}"
        else:
            m = re.search(r"age_(\d{1,2})_plus_(male|female|total)_\d{4}$", raw)
            if not m:
                continue
            band = f"{m.group(1)}_plus"
        value = pd.to_numeric(out[col], errors="coerce").fillna(0)
        bands.setdefault(band, pd.Series(0.0, index=out.index))
        bands[band] = bands[band] + value

    if "population" not in out and bands:
        out["population"] = sum(bands.values())
    pop = pd.to_numeric(out.get("population", pd.Series(np.nan, index=out.index)), errors="coerce")

    def summed(names: list[str]) -> pd.Series:
        available = [bands[n] for n in names if n in bands]
        if not available:
            return pd.Series(np.nan, index=out.index)
        result = available[0].copy()
        for item in available[1:]:
            result = result + item
        return result

    children = summed(["0_4", "5_9", "10_14", "15_19"])
    working = summed(["20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64"])
    older = summed(["65_69", "70_74", "75_79", "80_84", "85_plus"])
    out["working_age_share"] = np.where(pop > 0, working / pop, np.nan)
    out["youth_share"] = np.where(pop > 0, children / pop, np.nan)
    out["older_share"] = np.where(pop > 0, older / pop, np.nan)
    out["dependency_ratio"] = np.where(working > 0, (children + older) / working * 100, np.nan)
    out["youth_dependency_ratio"] = np.where(working > 0, children / working * 100, np.nan)
    out["old_age_dependency_ratio"] = np.where(working > 0, older / working * 100, np.nan)
    out["aging_index"] = np.where(children > 0, older / children * 100, np.nan)
    out["potential_support_ratio"] = np.where(older > 0, working / older, np.nan)
    return out


def prepare_year_frames(gdfs: dict[int, gpd.GeoDataFrame], years: list[int] | None = None) -> dict[int, gpd.GeoDataFrame]:
    selected = years or sorted(gdfs)
    result: dict[int, gpd.GeoDataFrame] = {}
    for year in selected:
        gdf = _derive_demographic_indicators(gdfs[year], int(year)).copy()
        gdf["geoid"] = gdf["ward_code"].astype(str)
        gdf["year"] = int(year)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        result[int(year)] = gdf
    return result


def build_community(gdfs: dict[int, gpd.GeoDataFrame], years: list[int] | None = None):
    """Build a current geosnap-compatible long-form dataset.

    geosnap 0.17.0 uses long-form GeoDataFrames as the constructor/analysis
    contract rather than exporting a ``Community`` class from the package root.
    For older releases that expose ``Community.from_geodataframes``, the
    connector can still construct that legacy object.
    """
    prepared = prepare_year_frames(gdfs, years)
    frames = [prepared[y].copy() for y in sorted(prepared)]
    try:
        from geosnap import Community  # type: ignore
    except ImportError:
        return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True, sort=False), geometry="geometry", crs=frames[0].crs), prepared
    community = Community.from_geodataframes(frames if len(frames) > 1 else frames[0])
    return community, prepared
