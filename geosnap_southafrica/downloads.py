from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import requests

log = logging.getLogger(__name__)
USER_AGENT = "geosnap-southafrica/0.4 (+https://www.statssa.gov.za/)"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 for a local file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, destination: str | Path, *, timeout: int = 120, chunk_size: int = 1024 * 1024) -> Path:
    """Download a URL to disk with streaming and atomic replacement."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": USER_AGENT}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    tmp.replace(destination)
    return destination


def download_first_working(urls: Iterable[str], destination: str | Path) -> Path:
    """Try URLs in order and keep the first successful download."""
    errors: list[str] = []
    for url in urls:
        try:
            return download_file(url, destination)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            log.warning("Download failed: %s", exc)
    raise RuntimeError("All download candidates failed:\n" + "\n".join(errors))


def arcgis_search_feature_services(search_terms: list[str], limit: int = 100) -> list[dict]:
    """Search the public ArcGIS catalogue for FeatureServer items."""
    q = " ".join(f'"{t}"' for t in search_terms)
    url = "https://www.arcgis.com/sharing/rest/search"
    params = {"q": f"{q} AND type:\"Feature Service\"", "f": "json", "num": min(limit, 100), "start": 1}
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r.json().get("results", [])


def discover_ward_service(url_override: str | None, search_terms: list[str]) -> str:
    """Find a public 2020 South African electoral-ward service with strict ranking."""
    if url_override:
        return url_override.rstrip("/")
    results = arcgis_search_feature_services(search_terms)
    ranked: list[tuple[int, str]] = []
    for item in results:
        title = str(item.get("title", ""))
        url = str(item.get("url", ""))
        text = f"{title} {url}".lower()
        score = sum(points for token, points in [("ward",8),("electoral",7),("2020",10),("south africa",4),("municipal",2)] if token in text)
        if any(old in text for old in ("2011", "2016")): score -= 10
        if url and ("FeatureServer" in url or "MapServer" in url): ranked.append((score, url))
    ranked.sort(reverse=True)
    if not ranked or ranked[0][0] < 12:
        raise RuntimeError("Could not confidently discover a public 2020 ward FeatureServer. Set ward_service_url in config.yaml.")
    if len(ranked) > 1 and ranked[1][0] >= ranked[0][0] - 1:
        raise RuntimeError(f"Ambiguous public ward services: {ranked[:5]}. Set ward_service_url explicitly.")
    return ranked[0][1].rstrip("/")


def _service_layer_url(service_url: str) -> str:
    s = service_url.rstrip("/")
    if re.search(r"/(FeatureServer|MapServer)/\d+$", s, flags=re.I): return s
    if re.search(r"/(FeatureServer|MapServer)$", s, flags=re.I): return s + "/0"
    return s + "/0"


def _oid_field_name(md: dict) -> str | None:
    """ArcGIS reports the OID field as a string ("FID") on some servers and as a dict on others."""
    val = md.get("objectIdField")
    if isinstance(val, dict):
        val = val.get("name")
    if not val:
        val = md.get("objectIdFieldName")
    if not val:
        for f in md.get("fields", []):
            if f.get("type") == "esriFieldTypeOID":
                return f.get("name")
    return val or None


def fetch_arcgis_geojson(service_url: str, out_geojson: str | Path) -> Path:
    """Download all features from a public ArcGIS REST layer as GeoJSON (paged by resultOffset)."""
    import geopandas as gpd
    layer = _service_layer_url(service_url)
    headers = {"User-Agent": USER_AGENT}
    meta = requests.get(layer, params={"f": "json"}, headers=headers, timeout=60)
    meta.raise_for_status()
    md = meta.json()
    if "error" in md:
        raise RuntimeError(md["error"])
    oid_field = _oid_field_name(md)
    page = min(int(md.get("maxRecordCount", 1000) or 1000), 2000)
    query = layer + "/query"
    features: list = []
    offset = 0
    while True:
        params = {"where": "1=1", "outFields": "*", "returnGeometry": "true", "outSR": "4326", "f": "geojson",
                  "resultOffset": offset, "resultRecordCount": page}
        if oid_field:
            params["orderByFields"] = oid_field
        r = requests.get(query, params=params, headers=headers, timeout=180)
        r.raise_for_status()
        payload = r.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        batch = payload.get("features", [])
        features.extend(batch)
        log.info("Fetched %d ward features so far", len(features))
        if len(batch) < page and not payload.get("exceededTransferLimit"):
            break
        if not batch:
            break
        offset += len(batch)
    combined = {"type": "FeatureCollection", "features": features}
    out_geojson = Path(out_geojson)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps(combined), encoding="utf-8")
    if gpd.read_file(out_geojson).empty:
        raise RuntimeError("ArcGIS ward layer returned zero features")
    return out_geojson
