"""Suburb layer support: which suburbs sit inside each ward, and ward -> suburb interpolation.

Wards (Municipal Demarcation Board, 2020) are administrative units; suburbs are
a separate geography that neither nests inside nor aligns with wards. The
standard treatment is a spatial overlay:

* :func:`ward_suburb_overlap` - long table of (ward, suburb, overlap shares).
* :func:`add_suburb_columns`  - ``main_suburb``, ``suburbs`` and ``suburb_count`` on the ward table.
* :func:`interpolate_to_suburbs` - area-weighted transfer of ward counts to suburbs
  (assumes population is spread evenly across each ward; treat as an estimate).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .downloads import fetch_arcgis_geojson

log = logging.getLogger(__name__)

# Equal-area CRS so overlap shares are areas, not degrees.
AREA_CRS = "EPSG:6933"
_NAME_HINTS = ("suburb", "sbrb", "subplace", "sub place", "sp name", "place name", "name")


def _detect_name_field(gdf: gpd.GeoDataFrame, override: str | None = None) -> str:
    if override:
        if override not in gdf.columns:
            raise ValueError(f"name_field {override!r} not in suburb layer. Columns: {list(gdf.columns)}")
        return override
    best: tuple[float, str] | None = None
    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        s = gdf[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        norm = str(col).lower().replace("_", " ")
        hint = sum(3 for h in _NAME_HINTS if h in norm)
        nunique = s.nunique(dropna=True)
        if not hint or nunique < 2:
            continue
        score = hint + min(nunique / max(len(s), 1), 1.0) * 2
        if best is None or score > best[0]:
            best = (score, str(col))
    if best is None:
        raise ValueError(f"Could not detect the suburb name column. Pass name_field=. Columns: {list(gdf.columns)}")
    return best[1]


def load_suburbs(source, *, cache_dir: str | Path | None = None, name_field: str | None = None,
                 bbox: tuple[float, float, float, float] | None = None) -> gpd.GeoDataFrame:
    """Load a suburb layer from a GeoDataFrame, a file path, or an ArcGIS FeatureServer/MapServer URL.

    ``source="osm"`` fetches named suburb/neighbourhood points from OpenStreetMap (whole country, or
    only ``bbox`` = (minx, miny, maxx, maxy) in lon/lat). Returns a GeoDataFrame in EPSG:4326 with
    ``suburb_name`` and geometry (polygons, or points for OSM).
    """
    if isinstance(source, gpd.GeoDataFrame):
        raw = source.copy()
    elif str(source).lower() == "osm":
        from .osm_suburbs import fetch_osm_suburbs
        raw = fetch_osm_suburbs(cache_dir or Path.cwd(), bbox)
        name_field = "suburb_name"
    else:
        src = str(source)
        if src.lower().startswith("http") and ("FeatureServer" in src or "MapServer" in src):
            cache_dir = Path(cache_dir) if cache_dir else Path.cwd()
            cache_dir.mkdir(parents=True, exist_ok=True)
            target = cache_dir / f"suburbs_{hashlib.sha256(src.encode()).hexdigest()[:12]}.geojson"
            if not target.exists():
                fetch_arcgis_geojson(src, target)
            raw = gpd.read_file(target)
        else:
            raw = gpd.read_file(src)
    if raw.empty:
        raise ValueError("Suburb layer is empty.")
    raw = raw.to_crs(4326) if raw.crs else raw.set_crs(4326)
    col = _detect_name_field(raw, name_field)
    out = gpd.GeoDataFrame({"suburb_name": raw[col].astype(str).str.strip()}, geometry=raw.geometry, crs=raw.crs)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    invalid = ~out.geometry.is_valid
    if invalid.any():
        out.loc[invalid, "geometry"] = out.loc[invalid, "geometry"].make_valid()
    # Merge multi-part suburbs that share a name so shares are per suburb.
    out = out.dissolve(by="suburb_name", as_index=False)
    log.info("Loaded %d suburbs (name column %r)", len(out), col)
    return out.reset_index(drop=True)


def _is_point_layer(gdf: gpd.GeoDataFrame) -> bool:
    return bool(len(gdf)) and gdf.geom_type.isin(["Point", "MultiPoint"]).all()


def _points_in_wards(wards: gpd.GeoDataFrame, suburbs: gpd.GeoDataFrame, id_col: str) -> pd.DataFrame:
    """(ward, suburb) pairs for suburb POINTS inside wards, with distance to the ward's representative point."""
    w = wards[[id_col, wards.geometry.name]].drop_duplicates(id_col).to_crs(AREA_CRS).reset_index(drop=True)
    s = suburbs[["suburb_name", suburbs.geometry.name]].to_crs(AREA_CRS)
    joined = gpd.sjoin(s, w, predicate="intersects", how="inner")
    rp = w.set_index(id_col).geometry.representative_point()
    joined["distance_m"] = [g.distance(rp[i]) for g, i in zip(joined.geometry, joined[id_col])]
    out = pd.DataFrame({id_col: joined[id_col].values, "suburb_name": joined["suburb_name"].values,
                        "distance_m": joined["distance_m"].values})
    return (out.drop_duplicates([id_col, "suburb_name"]).sort_values([id_col, "distance_m"]).reset_index(drop=True))


def ward_suburb_overlap(wards: gpd.GeoDataFrame, suburbs: gpd.GeoDataFrame, *, id_col: str = "geoid") -> pd.DataFrame:
    """Long table: one row per (ward, suburb) that intersect, with area shares.

    ``share_of_ward``   - fraction of the ward's area lying in the suburb.
    ``share_of_suburb`` - fraction of the suburb's area lying in the ward.
    """
    if _is_point_layer(suburbs):
        return _points_in_wards(wards, suburbs, id_col)  # no area shares for point data
    w = wards[[id_col, wards.geometry.name]].drop_duplicates(id_col).to_crs(AREA_CRS)
    s = suburbs[["suburb_name", suburbs.geometry.name]].to_crs(AREA_CRS)
    w["_w_area"] = w.geometry.area
    s["_s_area"] = s.geometry.area
    inter = gpd.overlay(w, s, how="intersection", keep_geom_type=True)
    inter["_area"] = inter.geometry.area
    inter = inter[inter["_area"] > 0]
    out = pd.DataFrame({
        id_col: inter[id_col].values,
        "suburb_name": inter["suburb_name"].values,
        "share_of_ward": (inter["_area"] / inter["_w_area"]).values,
        "share_of_suburb": (inter["_area"] / inter["_s_area"]).values,
    })
    return out.sort_values([id_col, "share_of_ward"], ascending=[True, False]).reset_index(drop=True)


def add_suburb_columns(wards: gpd.GeoDataFrame, suburbs: gpd.GeoDataFrame, *, min_share: float = 0.05,
                       id_col: str = "geoid") -> gpd.GeoDataFrame:
    """Add ``main_suburb``, ``main_suburb_share``, ``suburbs`` (semicolon list) and ``suburb_count``.

    A suburb is listed when it covers at least ``min_share`` of the ward, or when at
    least half of the suburb lies inside the ward (so small suburbs are not lost).
    """
    overlap = ward_suburb_overlap(wards, suburbs, id_col=id_col)
    if "share_of_ward" in overlap.columns:
        keep = overlap[(overlap["share_of_ward"] >= min_share) | (overlap["share_of_suburb"] >= 0.5)]
    else:  # point layer: keep all, nearest to the ward's centre first
        keep = overlap.assign(share_of_ward=float("nan"))
    grouped = keep.groupby(id_col, sort=False)
    listing = grouped["suburb_name"].agg("; ".join).rename("suburbs")
    count = grouped["suburb_name"].size().rename("suburb_count")
    main = keep.drop_duplicates(id_col).set_index(id_col)[["suburb_name", "share_of_ward"]]
    main.columns = ["main_suburb", "main_suburb_share"]
    out = wards.copy()
    for col in ("main_suburb", "main_suburb_share", "suburbs", "suburb_count"):
        if col in out.columns:
            out = out.drop(columns=col)
    out = out.join(main, on=id_col).join(listing, on=id_col).join(count, on=id_col)
    out["suburbs"] = out["suburbs"].fillna("")
    out["suburb_count"] = out["suburb_count"].fillna(0).astype(int)
    return out


def interpolate_to_suburbs(wards: gpd.GeoDataFrame, suburbs: gpd.GeoDataFrame, *,
                           extensive: list[str] | None = None, intensive: list[str] | None = None) -> gpd.GeoDataFrame:
    """Area-weighted transfer of ward variables to suburbs (tobler, as used by geosnap).

    ``extensive`` variables are counts (e.g. ``population``); totals are preserved.
    ``intensive`` variables are shares/rates (e.g. ``pct_white``); they are averaged.
    Assumes even density inside each ward, so treat results as estimates.
    """
    if _is_point_layer(suburbs):
        raise ValueError("Interpolation needs suburb polygons; this layer holds points (e.g. OpenStreetMap).")
    from tobler.area_weighted import area_interpolate

    extensive = [c for c in (extensive or ["population"]) if c in wards.columns]
    intensive = [c for c in (intensive or []) if c in wards.columns]
    if not extensive and not intensive:
        raise ValueError("No requested variables found on the ward table.")
    src = wards[extensive + intensive + [wards.geometry.name]].to_crs(AREA_CRS)
    tgt = suburbs[["suburb_name", suburbs.geometry.name]].to_crs(AREA_CRS)
    res = area_interpolate(src, tgt, extensive_variables=extensive or None, intensive_variables=intensive or None,
                           allocate_total=True)
    res["suburb_name"] = tgt["suburb_name"].values
    return res.to_crs(4326)
