"""Environmental and other spatial layers summarised per ward / suburb / any polygon.

Generic tools (work with any file, URL or GeoDataFrame):

* :func:`add_polygon_layer` - dominant class and class shares of a polygon layer (vegetation type, geology,
  threat status, land use ...) or simply the share covered (protected areas).
* :func:`add_raster_stats`  - zonal mean or dominant class of a raster (land cover, elevation, climate ...).
* :func:`add_point_counts`  - counts of point records (e.g. threatened-species records) per polygon.

Plus a small catalogue of SANBI (BGIS) services that can be requested by name with
``get_south_africa(..., extras=["vegetation", "threat_status", "protected_areas", "cba"])``.
SANBI data remain (c) SANBI and are subject to its terms of use; check them before redistributing.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .downloads import fetch_arcgis_geojson, http_get

log = logging.getLogger(__name__)
AREA_CRS = "EPSG:6933"

_BGIS = "https://bgismaps.sanbi.org/server/rest/services/BGIS_Projects/"
# name -> service, class-field hints, output prefix. ``verified`` = layer id and fields confirmed by inspection;
# the others are discovered at run time (first polygon layer; class field by hint) and may need class_field=.
CATALOGUE = {
    "vegetation": {"url": _BGIS + "VEGMAP2018_Final/MapServer/0", "class_hints": ["Name_18", "MAPCODE18"],
                   "prefix": "veg", "verified": True,
                   "note": "SANBI National Vegetation Map 2018 (potential/historical vegetation types)"},
    "biome": {"url": _BGIS + "VEGMAP2018_Final/MapServer/0", "class_hints": ["BIOME_18"], "prefix": "biome",
              "verified": True, "note": "Biome from the National Vegetation Map 2018"},
    "threat_status": {"url": _BGIS + "NBA2018_Terrestrial_ThreatStatus_Protection_Level/MapServer",
                      "class_hints": ["threat", "status"], "prefix": "threat", "verified": False,
                      "note": "NBA 2018 terrestrial ecosystem threat status (CR/EN/VU/LC)"},
    "protected_areas": {"url": _BGIS + "Protected_areas/MapServer", "class_hints": None, "prefix": "protected",
                        "verified": False, "note": "Protected areas: share of the ward covered"},
    "cba": {"url": _BGIS + "2023_Biodiversity_Priority_Areas_Map/MapServer", "class_hints": ["cba", "categor", "class"],
            "prefix": "cba", "verified": False, "note": "Biodiversity priority areas (critical biodiversity areas)"},
}
LAYER_EXTRAS = tuple(CATALOGUE)  # polygon services; raster extras are in RASTER_EXTRAS


# --------------------------------------------------------------------------- loading
def _resolve_layer_url(url: str) -> str:
    """Turn a MapServer/FeatureServer service URL into a polygon layer URL (first polygon layer)."""
    u = url.rstrip("/")
    if re.search(r"/(MapServer|FeatureServer)/\d+$", u, flags=re.I):
        return u
    r = http_get(u, params={"f": "json"}, timeout=60)
    r.raise_for_status()
    layers = r.json().get("layers") or []
    if not layers:
        raise RuntimeError(f"No layers listed at {u}")
    for lyr in layers:
        lr = http_get(f"{u}/{lyr['id']}", params={"f": "json"}, timeout=60)
        if lr.ok and "Polygon" in str(lr.json().get("geometryType", "")):
            return f"{u}/{lyr['id']}"
    return f"{u}/{layers[0]['id']}"


def load_layer(source, *, cache_dir: str | Path | None = None, bbox=None) -> gpd.GeoDataFrame:
    """Load a spatial layer from a GeoDataFrame, file, or ArcGIS REST URL (clipped to ``bbox`` when given)."""
    if isinstance(source, gpd.GeoDataFrame):
        g = source.copy()
    else:
        src = str(source)
        if src.lower().startswith("http") and re.search(r"(MapServer|FeatureServer)", src, flags=re.I):
            layer = _resolve_layer_url(src)
            cache = Path(cache_dir or Path.cwd())
            box_key = "" if bbox is None else ",".join(f"{v:.2f}" for v in bbox)
            target = cache / f"layer_{hashlib.sha256((layer + box_key).encode()).hexdigest()[:12]}.geojson"
            if not target.exists():
                fetch_arcgis_geojson(layer, target, bbox=bbox)
            g = gpd.read_file(target)
        else:
            g = gpd.read_file(src, bbox=tuple(bbox) if bbox is not None else None)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g = g.to_crs(4326) if g.crs else g.set_crs(4326)
    bad = ~g.geometry.is_valid
    if bad.any():
        g.loc[bad, g.geometry.name] = g.loc[bad].geometry.make_valid()
    return g


def _detect_class_field(g: gpd.GeoDataFrame, hints, override: str | None) -> str:
    if override:
        if override not in g.columns:
            raise ValueError(f"class_field {override!r} not in layer. Columns: {list(g.columns)}")
        return override
    text_cols = [c for c in g.columns if c != g.geometry.name and (pd.api.types.is_object_dtype(g[c]) or pd.api.types.is_string_dtype(g[c]))]
    for h in hints or []:                                   # exact column names first
        for c in text_cols:
            if c.lower() == h.lower():
                return c
    for h in hints or []:                                   # then substring matches
        for c in text_cols:
            if h.lower() in c.lower():
                return c
    raise ValueError(f"Could not detect the class column. Pass class_field=. Columns: {text_cols}")


def _bbox_of(gdf: gpd.GeoDataFrame, pad: float = 0.02):
    minx, miny, maxx, maxy = gdf.to_crs(4326).total_bounds
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)


def _slug(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40] or "unknown"


# --------------------------------------------------------------------------- polygon layers
def layer_overlap_table(wards: gpd.GeoDataFrame, layer: gpd.GeoDataFrame, class_field: str | None, *,
                        id_col: str = "geoid") -> pd.DataFrame:
    """Long table: ``id_col``, ``class`` (NaN when no class_field), ``share_of_ward``, ``area_km2``."""
    w = wards[[id_col, wards.geometry.name]].drop_duplicates(id_col).to_crs(AREA_CRS)
    keep = [class_field] if class_field else []
    lyr = layer[keep + [layer.geometry.name]].to_crs(AREA_CRS)
    w["_w_area"] = w.geometry.area
    if class_field is None:
        lyr = lyr.assign(_all=1).dissolve(by="_all")[[lyr.geometry.name]].reset_index(drop=True)
        lyr["_class"] = "covered"
        class_field = "_class"
    inter = gpd.overlay(w, lyr.rename(columns={class_field: "_class"}), how="intersection", keep_geom_type=True)
    inter["_area"] = inter.geometry.area
    inter = inter[inter["_area"] > 0]
    grouped = inter.groupby([id_col, "_class"], dropna=False).agg(_area=("_area", "sum"), _w=("_w_area", "first")).reset_index()
    out = pd.DataFrame({id_col: grouped[id_col].values, "class": grouped["_class"].values,
                        "share_of_ward": (grouped["_area"] / grouped["_w"]).clip(upper=1.0).values,
                        "area_km2": (grouped["_area"] / 1e6).values})
    return out.sort_values([id_col, "share_of_ward"], ascending=[True, False]).reset_index(drop=True)


def add_polygon_layer(gdf: gpd.GeoDataFrame, layer, *, prefix: str, class_field: str | None = None,
                      hints=None, shares: bool = False, max_share_columns: int = 30, id_col: str = "geoid",
                      cache_dir: str | Path | None = None) -> gpd.GeoDataFrame:
    """Add ``{prefix}_dominant``, ``{prefix}_dominant_share``, ``{prefix}_n_classes`` for a class layer.

    With ``class_field=None`` and no ``hints`` the layer is treated as a coverage layer (protected areas,
    wetlands ...) and ``{prefix}_cover_share`` (share of the ward covered) is added instead.
    ``shares=True`` also adds one ``{prefix}_share_<class>`` column per class (at most ``max_share_columns``).
    """
    g = load_layer(layer, cache_dir=cache_dir, bbox=_bbox_of(gdf))
    cls = None if (class_field is None and not hints) else _detect_class_field(g, hints, class_field)
    table = layer_overlap_table(gdf, g, cls, id_col=id_col)
    ids = gdf[id_col].astype(str)
    out = gdf.copy()
    if cls is None:
        cover = table.groupby(id_col)["share_of_ward"].sum().clip(upper=1.0)
        out[f"{prefix}_cover_share"] = ids.map(cover.rename(index=str)).fillna(0.0).values
        return out
    table[id_col] = table[id_col].astype(str)
    table["class"] = table["class"].astype(str)
    top = table.drop_duplicates(id_col).set_index(id_col)
    out[f"{prefix}_dominant"] = ids.map(top["class"]).values
    out[f"{prefix}_dominant_share"] = ids.map(top["share_of_ward"]).values
    out[f"{prefix}_n_classes"] = ids.map(table.groupby(id_col)["class"].nunique()).fillna(0).astype(int).values
    if shares:
        totals = table.groupby("class")["share_of_ward"].sum().sort_values(ascending=False)
        for name in totals.index[:max_share_columns]:
            col = table[table["class"] == name].set_index(id_col)["share_of_ward"]
            out[f"{prefix}_share_{_slug(name)}"] = ids.map(col).fillna(0.0).values
    return out


# --------------------------------------------------------------------------- rasters
def _open_path(path) -> str:
    p = str(path)
    return "/vsicurl/" + p if p.lower().startswith("http") else p


def _zonal_accumulate(path, wards: gpd.GeoDataFrame, categorical: bool, max_block_cells: int = 20_000_000):
    """(sums, counts, per-class counts) of one raster over ``wards`` (rows in order). Additive across tiles."""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from rasterio.windows import bounds as window_bounds
    from shapely.geometry import box

    n = len(wards)
    sums, cnts = np.zeros(n), np.zeros(n)
    classes: dict[int, np.ndarray] = {}
    with rasterio.open(_open_path(path)) as src:
        w = wards.to_crs(src.crs)
        rminx, rminy, rmaxx, rmaxy = src.bounds
        minx, miny, maxx, maxy = w.total_bounds
        if maxx < rminx or minx > rmaxx or maxy < rminy or miny > rmaxy:
            return sums, cnts, classes
        minx, miny, maxx, maxy = max(minx, rminx), max(miny, rminy), min(maxx, rmaxx), min(maxy, rmaxy)
        r0, c0 = src.index(minx, maxy)
        r1, c1 = src.index(maxx, miny)
        r0, c0 = max(min(r0, r1) - 1, 0), max(min(c0, c1) - 1, 0)
        r1, c1 = min(max(r0, r1) + 2, src.height), min(max(c0, c1) + 2, src.width)
        scale = (src.scales[0] if src.scales and src.scales[0] is not None else 1.0)
        offset = (src.offsets[0] if src.offsets and src.offsets[0] is not None else 0.0)
        geoms = w.geometry.values
        width = max(c1 - c0, 1)
        block_rows = max(64, min(1024, max_block_cells // width))
        sindex = w.sindex
        for row in range(r0, r1, block_rows):
            window = Window(c0, row, width, min(block_rows, r1 - row))
            idx = sindex.query(box(*window_bounds(window, src.transform)), predicate="intersects")
            if len(idx) == 0:
                continue
            data = src.read(1, window=window).astype("float64")
            valid = np.isfinite(data)
            if src.nodata is not None:
                valid &= data != src.nodata
            labels = rasterize([(geoms[k], int(k) + 1) for k in idx], out_shape=data.shape,
                               transform=src.window_transform(window), fill=0, dtype="int32")
            m = valid & (labels > 0)
            if not m.any():
                continue
            lab, val = labels[m], data[m]
            if categorical:
                for v in np.unique(val):
                    classes.setdefault(int(v), np.zeros(n))
                    classes[int(v)] += np.bincount(lab[val == v], minlength=n + 1)[1:]
            else:
                val = val * scale + offset
                sums += np.bincount(lab, weights=val, minlength=n + 1)[1:]
            cnts += np.bincount(lab, minlength=n + 1)[1:]
    return sums, cnts, classes


def add_raster_stats(gdf: gpd.GeoDataFrame, raster, *, prefix: str, categorical: bool = False,
                     class_names: dict[int, str] | None = None, id_col: str = "geoid") -> gpd.GeoDataFrame:
    """Zonal statistics of one or several rasters (paths or http URLs, read remotely by window) per polygon.

    Continuous rasters (elevation, temperature, rainfall, canopy %): ``{prefix}_mean`` (scale/offset applied).
    Categorical rasters (land cover): ``{prefix}_dominant`` and ``{prefix}_dominant_share``; with
    ``class_names={code: name}`` also ``{prefix}_share_<name>`` for every named class.
    A list of rasters is treated as tiles of one mosaic (cell values are added across tiles).
    """
    paths = list(raster) if isinstance(raster, (list, tuple)) else [raster]
    w = gdf[[id_col, gdf.geometry.name]].drop_duplicates(id_col).reset_index(drop=True)
    n = len(w)
    sums, cnts, classes = np.zeros(n), np.zeros(n), {}
    for p in paths:
        s_, c_, k_ = _zonal_accumulate(p, w, categorical)
        sums += s_
        cnts += c_
        for code, arr in k_.items():
            classes[code] = classes.get(code, np.zeros(n)) + arr
    safe = np.where(cnts > 0, cnts, 1)
    table = pd.DataFrame({id_col: w[id_col].astype(str).values})
    if categorical:
        codes = sorted(classes)
        if codes:
            mat = np.column_stack([classes[c] for c in codes])
            table[f"{prefix}_dominant"] = np.where(cnts > 0, np.array(codes)[mat.argmax(axis=1)], np.nan)
            table[f"{prefix}_dominant_share"] = np.where(cnts > 0, mat.max(axis=1) / safe, np.nan)
        else:
            table[f"{prefix}_dominant"] = np.nan
            table[f"{prefix}_dominant_share"] = np.nan
        for code, name in (class_names or {}).items():
            table[f"{prefix}_share_{_slug(name)}"] = np.where(cnts > 0, classes.get(code, np.zeros(n)) / safe, np.nan)
    else:
        table[f"{prefix}_mean"] = np.where(cnts > 0, sums / safe, np.nan)
    out = gdf.assign(**{id_col: gdf[id_col].astype(str)})
    out = out.drop(columns=[c for c in table.columns if c != id_col and c in out.columns]).merge(table, on=id_col, how="left")
    return gpd.GeoDataFrame(out, geometry=gdf.geometry.name, crs=gdf.crs)


# --- open global rasters, read remotely by window (nothing large is downloaded) ---------------------------------
WORLDCOVER_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow_ice",
                      80: "water", 90: "wetland", 95: "mangrove", 100: "moss_lichen"}


def _tiles_1deg_floor(bbox, step):
    minx, miny, maxx, maxy = bbox
    xs = np.arange(np.floor(minx / step) * step, maxx, step)
    ys = np.arange(np.floor(miny / step) * step, maxy, step)
    return [(int(x), int(y)) for x in xs for y in ys]


def worldcover_urls(bbox) -> list[str]:
    """ESA WorldCover 2021 v200 (10 m, CC BY 4.0) 3-degree tiles covering ``bbox``."""
    base = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{ns}{lat:02d}{ew}{lon:03d}_Map.tif"
    urls = []
    for x, y in _tiles_1deg_floor(bbox, 3):
        lat, lon = (y // 3) * 3, (x // 3) * 3
        urls.append(base.format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="E" if lon >= 0 else "W", lon=abs(lon)))
    return urls


def copernicus_dem_urls(bbox) -> list[str]:
    """Copernicus GLO-30 DEM (30 m) 1-degree COG tiles covering ``bbox``."""
    urls = []
    for x, y in _tiles_1deg_floor(bbox, 1):
        ns, ew = ("N" if y >= 0 else "S"), ("E" if x >= 0 else "W")
        name = f"Copernicus_DSM_COG_10_{ns}{abs(y):02d}_00_{ew}{abs(x):03d}_00_DEM"
        urls.append(f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif")
    return urls


CHELSA = "https://os.zhdk.ethz.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio/CHELSA_{var}_1981-2010_V.2.1.tif"

RASTER_CATALOGUE = {
    "landcover": {"urls": worldcover_urls, "categorical": True, "classes": WORLDCOVER_CLASSES, "prefix": "lc",
                  "note": "ESA WorldCover 2021: share of each land-cover class, incl. tree cover and built-up"},
    "elevation": {"urls": copernicus_dem_urls, "prefix": "elevation", "note": "Copernicus GLO-30 mean elevation (m)"},
    "climate_temp": {"urls": lambda b: [CHELSA.format(var="bio1")], "prefix": "climate_temp",
                     "note": "CHELSA v2.1 1981-2010 mean annual temperature (deg C)"},
    "climate_heat": {"urls": lambda b: [CHELSA.format(var="bio5")], "prefix": "climate_heat",
                     "note": "CHELSA v2.1 1981-2010 max temperature of the warmest month (deg C)"},
    "climate_rain": {"urls": lambda b: [CHELSA.format(var="bio12")], "prefix": "climate_rain",
                     "note": "CHELSA v2.1 1981-2010 mean annual precipitation (mm)"},
}
RASTER_EXTRAS = tuple(RASTER_CATALOGUE)


def apply_raster_extra(gdf: gpd.GeoDataFrame, name: str, cache_dir: str | Path) -> pd.DataFrame:
    """One raster catalogue extra as a table keyed by ``geoid`` (cached, since remote reads are slow)."""
    spec = RASTER_CATALOGUE[name]
    base = gdf.drop_duplicates("geoid")[["geoid", gdf.geometry.name]].reset_index(drop=True)
    key = hashlib.sha256((name + ",".join(sorted(base["geoid"].astype(str)))).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"raster_{name}_{key}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"geoid": str})
    res = add_raster_stats(base, spec["urls"](_bbox_of(base, 0.0)), prefix=spec["prefix"],
                           categorical=spec.get("categorical", False), class_names=spec.get("classes"))
    table = res.drop(columns=[base.geometry.name])
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)
    return table


# --------------------------------------------------------------------------- points
def add_point_counts(gdf: gpd.GeoDataFrame, points, *, prefix: str, kind_field: str | None = None, id_col: str = "geoid",
                     cache_dir: str | Path | None = None) -> gpd.GeoDataFrame:
    """Count point records per polygon: ``{prefix}_count`` (and ``{prefix}_count_<kind>`` per kind_field value)."""
    pts = load_layer(points, cache_dir=cache_dir, bbox=_bbox_of(gdf))
    pts = pts[pts.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    w = gdf[[id_col, gdf.geometry.name]].drop_duplicates(id_col).to_crs(4326)
    j = gpd.sjoin(pts, w, predicate="within", how="inner")
    out = gdf.copy()
    ids = out[id_col].astype(str)
    j[id_col] = j[id_col].astype(str)
    out[f"{prefix}_count"] = ids.map(j.groupby(id_col).size()).fillna(0).astype(int).values
    if kind_field:
        for kind, sub in j.groupby(j[kind_field].astype(str)):
            out[f"{prefix}_count_{_slug(kind)}"] = ids.map(sub.groupby(id_col).size()).fillna(0).astype(int).values
    return out


# --------------------------------------------------------------------------- catalogue / extras
def apply_layer_extra(gdf: gpd.GeoDataFrame, name: str, cache_dir: str | Path) -> pd.DataFrame:
    """Compute one catalogue extra for ``gdf`` and return a table of new columns keyed by ``geoid``."""
    spec = CATALOGUE[name]
    base = gdf.drop_duplicates("geoid")[["geoid", gdf.geometry.name]]
    hints = spec["class_hints"]
    res = add_polygon_layer(base, spec["url"], prefix=spec["prefix"], hints=hints, cache_dir=cache_dir)
    return res.drop(columns=[base.geometry.name])
