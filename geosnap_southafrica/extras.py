"""Optional extra ward-level variables that are fetched automatically and joined by ``geoid``.

    get_south_africa(..., extras=["amenities", "rwi"])

* ``amenities`` - OpenStreetMap counts per ward (schools, tertiary, health, pharmacies, bus stops, rail
  stations, taxi ranks, supermarkets/shops, markets). (c) OpenStreetMap contributors, ODbL. Coverage varies.
* ``rwi``       - Meta Relative Wealth Index (2.4 km grid) averaged per ward: ``rwi_mean``, ``rwi_cells``.
  A wealth *proxy* (higher = wealthier relative to the rest of the country), not income. Check the
  dataset's licence (Humanitarian Data Exchange lists it as non-commercial) before commercial use.

Extras are static (not yearly): the same values are repeated on every year's rows.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from .downloads import USER_AGENT, download_file
from .osm_suburbs import OVERPASS_ENDPOINTS

log = logging.getLogger(__name__)

_BASE_EXTRAS = ("amenities", "rwi")
try:  # SANBI environmental layers live in environment.py
    from .environment import LAYER_EXTRAS, RASTER_EXTRAS
except ImportError:  # pragma: no cover
    LAYER_EXTRAS, RASTER_EXTRAS = (), ()
AVAILABLE = _BASE_EXTRAS + tuple(LAYER_EXTRAS) + tuple(RASTER_EXTRAS)

# --------------------------------------------------------------------------- OSM amenities

AMENITY_COLUMNS = ["n_schools", "n_tertiary", "n_health", "n_pharmacies", "n_bus_stops", "n_rail_stations",
                   "n_taxi_ranks", "n_shops", "n_markets"]
TILE_DEG = 0.25


def classify_tags(tags: dict) -> str | None:
    """Map OSM tags to one of :data:`AMENITY_COLUMNS` (or None)."""
    a, h, r, s = tags.get("amenity"), tags.get("highway"), tags.get("railway"), tags.get("shop")
    if a == "school":
        return "n_schools"
    if a in ("college", "university"):
        return "n_tertiary"
    if a in ("hospital", "clinic", "doctors"):
        return "n_health"
    if a == "pharmacy":
        return "n_pharmacies"
    if h == "bus_stop" or a == "bus_station":
        return "n_bus_stops"
    if r in ("station", "halt"):
        return "n_rail_stations"
    if a == "taxi":
        return "n_taxi_ranks"
    if s in ("supermarket", "convenience", "general"):
        return "n_shops"
    if a == "marketplace":
        return "n_markets"
    return None


def amenity_query(bbox: tuple[float, float, float, float], timeout: int = 90) -> str:
    minx, miny, maxx, maxy = bbox
    b = f"({miny:.5f},{minx:.5f},{maxy:.5f},{maxx:.5f})"
    return (f"[out:json][timeout:{timeout}];("
            f'nw["amenity"~"^(school|college|university|hospital|clinic|doctors|pharmacy|bus_station|taxi|marketplace)$"]{b};'
            f'nw["highway"="bus_stop"]{b};'
            f'nw["railway"~"^(station|halt)$"]{b};'
            f'nw["shop"~"^(supermarket|convenience|general)$"]{b};'
            ");out center tags;")


def parse_amenities(payload: dict) -> gpd.GeoDataFrame:
    rows = []
    for el in payload.get("elements", []):
        kind = classify_tags(el.get("tags") or {})
        if kind is None:
            continue
        if el.get("type") == "node":
            lon, lat = el.get("lon"), el.get("lat")
        else:
            c = el.get("center") or {}
            lon, lat = c.get("lon"), c.get("lat")
        if lon is None or lat is None:
            continue
        rows.append({"osm_id": f'{el.get("type")}/{el.get("id")}', "kind": kind, "geometry": Point(lon, lat)})
    return gpd.GeoDataFrame(rows, columns=["osm_id", "kind", "geometry"], geometry="geometry", crs=4326)


def _tiles(bounds, step: float = TILE_DEG):
    minx, miny, maxx, maxy = bounds
    xs = np.arange(np.floor(minx / step) * step, maxx, step)
    ys = np.arange(np.floor(miny / step) * step, maxy, step)
    return [(float(round(x, 4)), float(round(y, 4)), float(round(x + step, 4)), float(round(y + step, 4))) for x in xs for y in ys]


def _fetch_tile(tile, cache_dir: Path) -> dict:
    key = hashlib.sha256(json.dumps(tile).encode()).hexdigest()[:10]
    cache = cache_dir / f"osm_amenities_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    errors = []
    for attempt in range(4):
        for url in OVERPASS_ENDPOINTS:
            try:
                r = requests.post(url, data={"data": amenity_query(tile)}, headers={"User-Agent": USER_AGENT}, timeout=120)
                r.raise_for_status()
                payload = r.json()
                cache.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            except Exception as exc:  # noqa: BLE001 - try the next endpoint
                errors.append(f"{url}: {exc}")
        wait = 5 * (attempt + 1)
        log.info("Overpass busy for tile %s, retrying in %ds ...", tile, wait)
        time.sleep(wait)
    raise RuntimeError("Overpass failed for tile %s:\n%s" % (tile, "\n".join(errors[-2:])))


def osm_amenity_counts(wards: gpd.GeoDataFrame, cache_dir: str | Path, *, id_col: str = "geoid") -> pd.DataFrame:
    """Count OSM amenities per ward (one row per ward, columns :data:`AMENITY_COLUMNS`)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    w = wards[[id_col, wards.geometry.name]].drop_duplicates(id_col).to_crs(4326)
    from .prebuilt import fetch_prebuilt   # a published national table beats hundreds of Overpass requests
    pre = fetch_prebuilt("osm_amenity_ward_counts.parquet", cache_dir)
    if pre is not None:
        table = pd.read_parquet(pre)
        ids = w[id_col].astype(str)
        if ids.isin(set(table[id_col].astype(str))).all():
            log.info("Using prebuilt OSM amenity counts")
            return table.assign(**{id_col: table[id_col].astype(str)}).set_index(id_col).reindex(ids).rename_axis(id_col).reset_index()
    tiles = _tiles(w.total_bounds)
    log.info("Fetching OSM amenities for %d wards in %d tile(s) ...", len(w), len(tiles))
    if len(tiles) > 60:
        log.warning("Amenities for this area need %d Overpass requests (slow, and the public servers may throttle). "
                    "Consider a province or municipality; results are cached per tile so a re-run resumes.", len(tiles))
    parts = []
    for i, tile in enumerate(tiles, 1):
        parts.append(parse_amenities(_fetch_tile(tile, cache_dir)))
        if len(tiles) > 1 and i % 10 == 0:
            log.info("  amenities: %d/%d tiles", i, len(tiles))
    pts = pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame(columns=["osm_id", "kind", "geometry"])
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=4326).drop_duplicates("osm_id")
    counts = pd.DataFrame(0, index=w[id_col].values, columns=AMENITY_COLUMNS, dtype="int64")
    if len(pts):
        j = gpd.sjoin(pts, w, predicate="within", how="inner")
        tab = j.groupby([id_col, "kind"]).size().unstack(fill_value=0)
        counts.loc[tab.index, tab.columns] = tab
    counts.index.name = id_col
    return counts.reset_index()


# --------------------------------------------------------------------------- Relative Wealth Index

HDX_PACKAGE_API = "https://data.humdata.org/api/3/action/package_show?id=relative-wealth-index"
MAX_NEAREST_M = 5000.0


def find_rwi_url(package: dict) -> str:
    """Pick the South Africa CSV out of the HDX package metadata."""
    best = None
    for res in package.get("result", {}).get("resources", []):
        text = f'{res.get("name", "")} {res.get("url", "")}'.lower()
        if "csv" not in text:
            continue
        if "south" in text and "africa" in text and "sudan" not in text:
            return res["url"]
        if "zaf" in text and best is None:
            best = res["url"]
    if best:
        return best
    raise RuntimeError("No South Africa file found in the HDX Relative Wealth Index package")


def load_rwi(cache_dir: str | Path) -> gpd.GeoDataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv = cache_dir / "rwi_south_africa.csv"
    if not csv.exists():
        r = requests.get(HDX_PACKAGE_API, headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
        download_file(find_rwi_url(r.json()), csv, timeout=600)
    df = pd.read_csv(csv)
    cols = {c.lower(): c for c in df.columns}
    for need in ("latitude", "longitude", "rwi"):
        if need not in cols:
            raise RuntimeError(f"RWI file lacks a '{need}' column: {list(df.columns)}")
    lat, lon, rwi = cols["latitude"], cols["longitude"], cols["rwi"]
    df = df[[lat, lon, rwi]].dropna()
    return gpd.GeoDataFrame({"rwi": df[rwi].astype(float).values},
                            geometry=gpd.points_from_xy(df[lon].astype(float), df[lat].astype(float)), crs=4326)


def rwi_by_ward(wards: gpd.GeoDataFrame, cache_dir: str | Path, *, id_col: str = "geoid") -> pd.DataFrame:
    """Mean RWI of grid cells inside each ward; wards with no cell take the nearest cell within 5 km."""
    pts = load_rwi(cache_dir)
    w = wards[[id_col, wards.geometry.name]].drop_duplicates(id_col).to_crs(4326).reset_index(drop=True)
    j = gpd.sjoin(pts, w, predicate="within", how="inner")
    agg = j.groupby(id_col)["rwi"].agg(rwi_mean="mean", rwi_cells="size")
    out = w[[id_col]].merge(agg.reset_index(), on=id_col, how="left")
    out["rwi_cells"] = out["rwi_cells"].fillna(0).astype(int)
    missing = out["rwi_mean"].isna()
    if missing.any():
        wm = w[missing.values].to_crs(6933)
        near = gpd.sjoin_nearest(wm, pts.to_crs(6933), how="left", max_distance=MAX_NEAREST_M, distance_col="_d")
        near = near.drop_duplicates(id_col).set_index(id_col)["rwi"]
        out.loc[missing, "rwi_mean"] = out.loc[missing, id_col].map(near).values
    return out


# --------------------------------------------------------------------------- entry point

def apply_extras(long: gpd.GeoDataFrame, extras, cache_dir: str | Path, *, errors: str = "raise") -> gpd.GeoDataFrame:
    """Join the requested extras onto ``long`` by ``geoid`` (same values on every year's rows)."""
    if isinstance(extras, str):
        extras = [extras]
    extras = list(dict.fromkeys(extras))
    unknown = [e for e in extras if e not in AVAILABLE]
    if unknown:
        raise ValueError(f"Unknown extras {unknown}. Available: {list(AVAILABLE)}")
    out = long
    for name in extras:
        try:
            if name == "amenities":
                table = osm_amenity_counts(long, cache_dir)
            elif name == "rwi":
                table = rwi_by_ward(long, cache_dir)
            elif name in RASTER_EXTRAS:
                from .environment import apply_raster_extra
                table = apply_raster_extra(long, name, cache_dir)
            else:
                from .environment import apply_layer_extra
                table = apply_layer_extra(long, name, cache_dir)
        except Exception as exc:  # noqa: BLE001
            if errors == "raise":
                raise
            log.warning("Extra %r skipped: %s", name, exc)
            continue
        new = [c for c in table.columns if c != "geoid"]
        out = out.drop(columns=[c for c in new if c in out.columns]).merge(table, on="geoid", how="left")
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=long.crs)
    return out


def add_ward_table(gdf: gpd.GeoDataFrame, table: pd.DataFrame, *, on: str = "geoid", table_on: str | None = None,
                   min_match: float = 0.9) -> gpd.GeoDataFrame:
    """Join your own ward-level table (e.g. a Stats SA export already on 2020 wards) onto the connector output.

    ``table_on`` is the ward-code column in ``table`` (defaults to ``on``). Codes are compared as strings; a
    trailing ``.0`` is ignored. Raises if fewer than ``min_match`` of the wards find a row, which usually
    means the table uses different (older) ward boundaries and needs area-weighting instead.
    """
    key = table_on or on
    t = table.copy()
    t[key] = t[key].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    ids = gdf[on].astype(str).drop_duplicates()
    match = ids.isin(set(t[key])).mean() if len(ids) else 1.0
    if match < min_match:
        raise ValueError(f"Only {match:.0%} of wards matched the table's {key!r} codes. "
                         "Codes must be the 8-digit 2020 ward ids; older ward boundaries need area-weighted re-basing.")
    t = t.drop_duplicates(key).rename(columns={key: on})
    new = [c for c in t.columns if c != on and c in gdf.columns]
    out = gdf.assign(**{on: gdf[on].astype(str)}).drop(columns=new).merge(t, on=on, how="left")
    return gpd.GeoDataFrame(out, geometry=gdf.geometry.name, crs=gdf.crs)
