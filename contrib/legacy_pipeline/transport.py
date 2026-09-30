from __future__ import annotations

import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.features import rasterize
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box

from geosnap_southafrica.worldpop import _candidate_indices, _prepare_shapes, _window_iter

log = logging.getLogger(__name__)
DEFAULT_OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]


def _tiles(bounds: tuple[float, float, float, float], step: float = 5.0):
    minx, miny, maxx, maxy = bounds
    x = minx
    while x < maxx:
        y = miny
        x2 = min(x + step, maxx)
        while y < maxy:
            y2 = min(y + step, maxy)
            yield (x, y, x2, y2)
            y = y2
        x = x2


def _query_overpass(bbox: tuple[float, float, float, float], endpoints: list[str], timeout: int = 180) -> list[dict]:
    minx, miny, maxx, maxy = bbox
    query = f"""
    [out:json][timeout:{timeout}];
    (
      nwr[public_transport]({miny},{minx},{maxy},{maxx});
      nwr[highway=bus_stop]({miny},{minx},{maxy},{maxx});
      nwr[railway=station]({miny},{minx},{maxy},{maxx});
      nwr[railway=halt]({miny},{minx},{maxy},{maxx});
      nwr[amenity=bus_station]({miny},{minx},{maxy},{maxx});
    );
    out center tags;
    """
    last: Exception | None = None
    for endpoint in endpoints:
        try:
            r = requests.post(endpoint, data={"data": query}, timeout=timeout + 30,
                               headers={"User-Agent": "geosnap-southafrica-transport/1.0"})
            r.raise_for_status()
            return r.json().get("elements", [])
        except Exception as exc:
            last = exc
            log.warning("Overpass endpoint failed: %s", exc)
            time.sleep(1)
    raise RuntimeError(f"All Overpass endpoints failed: {last}")


def download_public_transport_stops(wards: gpd.GeoDataFrame, output_path: str | Path, *, endpoints: list[str] | None = None,
                                    tile_degrees: float = 5.0, sleep_seconds: float = 0.5) -> Path:
    """Download mapped public transport nodes/stations from OpenStreetMap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wards_wgs = wards.to_crs("EPSG:4326")
    records: dict[str, dict] = {}
    for bbox in _tiles(tuple(wards_wgs.total_bounds), tile_degrees):
        for item in _query_overpass(bbox, endpoints or DEFAULT_OVERPASS):
            tags = item.get("tags", {})
            lat, lon = item.get("lat"), item.get("lon")
            if lat is None or lon is None:
                center = item.get("center", {})
                lat, lon = center.get("lat"), center.get("lon")
            if lat is None or lon is None:
                continue
            records[f"{item.get('type')}:{item.get('id')}"] = {
                "osm_id": str(item.get("id")), "osm_type": item.get("type"), "lat": lat, "lon": lon,
                "name": tags.get("name"), "public_transport": tags.get("public_transport"),
                "railway": tags.get("railway"), "highway": tags.get("highway"), "amenity": tags.get("amenity"),
            }
        time.sleep(sleep_seconds)
    if not records:
        raise RuntimeError("Overpass returned no public-transport features for the analysis area.")
    df = pd.DataFrame(records.values())
    stops = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    stops.to_file(output_path, driver="GPKG")
    return output_path


def _buffer_raster_mask(src, buffers, buffer_sindex, window):
    bbox = window_bounds(window, src.transform)
    clip_box = box(*bbox)
    try:
        hits = buffer_sindex.query(clip_box, predicate="intersects")
        local = [buffers[int(i)] for i in hits]
    except Exception:
        local = [g for g in buffers if g.intersects(clip_box)]
    if not local:
        return np.zeros((int(window.height), int(window.width)), dtype="uint8")
    return rasterize([(g, 1) for g in local], out_shape=(int(window.height), int(window.width)),
                     transform=src.window_transform(window), fill=0, dtype="uint8", all_touched=True)


def compute_transport_accessibility(calibrated_population_raster: str | Path, wards: gpd.GeoDataFrame, stops_path: str | Path,
                                     output_csv: str | Path, radii_m: list[int] | None = None) -> Path:
    """Estimate population within Euclidean service radii of mapped transport features."""
    radii_m = radii_m or [400, 800]
    with rasterio.open(calibrated_population_raster) as src:
        wards_r = _prepare_shapes(wards, src.crs)
        stops = gpd.read_file(stops_path)
        if stops.empty:
            raise RuntimeError("Transport stop layer is empty.")
        metric_crs = stops.estimate_utm_crs()
        if metric_crs is None:
            metric_crs = "EPSG:32734"
        metric_stops = stops.to_crs(metric_crs)
        results = wards_r[["ward_code"]].copy()
        for radius in radii_m:
            buffers = list(metric_stops.geometry.buffer(radius).to_crs(src.crs))
            buffer_sindex = gpd.GeoSeries(buffers, crs=src.crs).sindex
            accessible = np.zeros(len(wards_r), dtype=float)
            for window in _window_iter(src, block_rows=512):
                bbox = window_bounds(window, src.transform)
                idx = _candidate_indices(wards_r, bbox)
                if len(idx) == 0:
                    continue
                ward_ids = rasterize([(wards_r.iloc[i].geometry, int(i)) for i in idx],
                                    out_shape=(int(window.height), int(window.width)),
                                    transform=src.window_transform(window), fill=-1, dtype="int32")
                access_mask = _buffer_raster_mask(src, buffers, buffer_sindex, window)
                pop = src.read(1, window=window).astype(float)
                mask = (ward_ids >= 0) & (access_mask == 1)
                if mask.any():
                    np.add.at(accessible, ward_ids[mask], pop[mask])
            results[f"population_within_{radius}m"] = accessible

        total = np.zeros(len(wards_r), dtype=float)
        for window in _window_iter(src, block_rows=512):
            bbox = window_bounds(window, src.transform)
            idx = _candidate_indices(wards_r, bbox)
            if len(idx) == 0:
                continue
            ids = rasterize([(wards_r.iloc[i].geometry, int(i)) for i in idx],
                            out_shape=(int(window.height), int(window.width)),
                            transform=src.window_transform(window), fill=-1, dtype="int32")
            pop = src.read(1, window=window).astype(float)
            mask = ids >= 0
            if mask.any():
                np.add.at(total, ids[mask], pop[mask])
        results["population_total_100m"] = total
        for radius in radii_m:
            results[f"share_population_within_{radius}m"] = np.divide(
                results[f"population_within_{radius}m"], total,
                out=np.zeros(len(total)), where=total > 0,
            )
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_csv, index=False)
    return output_csv
