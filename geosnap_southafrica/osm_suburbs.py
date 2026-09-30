"""National suburb names from OpenStreetMap (place=suburb / neighbourhood / quarter) via Overpass.

OSM data are (c) OpenStreetMap contributors, ODbL. Coverage is uneven: big cities are well mapped,
small towns and rural areas less so. Suburbs are returned as POINTS (node position, or the centre of a
mapped way/relation), so wards are assigned by point-in-polygon, not by area overlap.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import Point

log = logging.getLogger(__name__)

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
PLACE_TYPES = "suburb|neighbourhood|quarter"
USER_AGENT = "geosnap-southafrica/0.4 (+https://github.com/)"


def build_query(bbox: tuple[float, float, float, float] | None, timeout: int = 300) -> str:
    """Overpass QL for suburb-like places, inside South Africa or a (minx, miny, maxx, maxy) box."""
    if bbox is None:
        head = 'area["ISO3166-1"="ZA"][admin_level=2]->.a;'
        scope = "(area.a)"
    else:
        minx, miny, maxx, maxy = bbox
        head = ""
        scope = f"({miny:.5f},{minx:.5f},{maxy:.5f},{maxx:.5f})"
    return (f"[out:json][timeout:{timeout}];{head}"
            f'(node["place"~"^({PLACE_TYPES})$"]{scope};'
            f'way["place"~"^({PLACE_TYPES})$"]{scope};'
            f'relation["place"~"^({PLACE_TYPES})$"]{scope};);out center tags;')


def parse_elements(payload: dict) -> gpd.GeoDataFrame:
    rows = []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        name = (tags.get("name") or "").strip()
        if not name:
            continue
        if el.get("type") == "node":
            lon, lat = el.get("lon"), el.get("lat")
        else:
            c = el.get("center") or {}
            lon, lat = c.get("lon"), c.get("lat")
        if lon is None or lat is None:
            continue
        rows.append({"suburb_name": name, "place_type": tags.get("place"), "osm_id": f'{el.get("type")}/{el.get("id")}',
                     "geometry": Point(lon, lat)})
    return gpd.GeoDataFrame(rows, columns=["suburb_name", "place_type", "osm_id", "geometry"], geometry="geometry", crs=4326)


def fetch_osm_suburbs(cache_dir: str | Path, bbox: tuple[float, float, float, float] | None = None) -> gpd.GeoDataFrame:
    """Download (and cache) suburb points; ``bbox`` = (minx, miny, maxx, maxy) in lon/lat limits the query."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rounded = None if bbox is None else tuple(round(float(v), 2) for v in bbox)
    key = hashlib.sha256(json.dumps(rounded).encode()).hexdigest()[:10]
    cache = cache_dir / f"osm_suburbs_{key}.geojson"
    if cache.exists():
        return gpd.read_file(cache)
    query = build_query(rounded)
    errors = []
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=420)
            r.raise_for_status()
            gdf = parse_elements(r.json())
            if gdf.empty:
                raise RuntimeError("Overpass returned no named suburb/neighbourhood places for this area")
            gdf.to_file(cache, driver="GeoJSON")
            log.info("Fetched %d OSM suburb points", len(gdf))
            return gdf
        except Exception as exc:  # noqa: BLE001 - try the next endpoint
            errors.append(f"{url}: {exc}")
            log.warning("Overpass request failed: %s", exc)
    raise RuntimeError("Could not fetch suburbs from OpenStreetMap:\n" + "\n".join(errors))
