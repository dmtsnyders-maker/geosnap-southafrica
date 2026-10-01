"""National OpenStreetMap amenity counts per ward from a Geofabrik extract (no Overpass needed).

One download of ``south-africa-latest.osm.pbf`` (about 500 MB) is read locally, every matching feature is
turned into a point (nodes as-is, ways at their centroid) and counted per ward. The small result table is what
gets published as a prebuilt asset. Needs the optional ``osmium`` package (``pip install osmium``).

OpenStreetMap data are (c) OpenStreetMap contributors, ODbL: credit them wherever the counts are shown.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .downloads import download_file
from .extras import AMENITY_COLUMNS, classify_tags

log = logging.getLogger(__name__)
GEOFABRIK_URL = "https://download.geofabrik.de/africa/south-africa-latest.osm.pbf"
MIRROR_URLS = (GEOFABRIK_URL, "https://download.openstreetmap.fr/extracts/africa/south_africa.osm.pbf")
AMENITY_TABLE = "osm_amenity_ward_counts.parquet"
_TAG_PAIRS = ([("amenity", v) for v in ("school", "college", "university", "hospital", "clinic", "doctors", "pharmacy",
                                        "bus_station", "taxi", "marketplace")]
              + [("highway", "bus_stop")] + [("railway", v) for v in ("station", "halt")]
              + [("shop", v) for v in ("supermarket", "convenience", "general")])


def read_pbf_points(pbf_path: str | Path) -> gpd.GeoDataFrame:
    """Amenity points (columns ``osm_id``, ``kind``, geometry) from a PBF file."""
    try:
        import osmium
        from osmium.filter import TagFilter
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Reading OSM extracts needs osmium: pip install osmium") from exc
    rows = []
    fp = osmium.FileProcessor(str(pbf_path), osmium.osm.NODE | osmium.osm.WAY).with_locations().with_filter(TagFilter(*_TAG_PAIRS))
    for obj in fp:
        kind = classify_tags({t.k: t.v for t in obj.tags})
        if kind is None:
            continue
        if obj.is_node():
            if not obj.location.valid():
                continue
            lon, lat = obj.location.lon, obj.location.lat
        else:
            locs = [(n.lon, n.lat) for n in obj.nodes if n.location.valid()]
            if not locs:
                continue
            lon = sum(x for x, _ in locs) / len(locs)
            lat = sum(y for _, y in locs) / len(locs)
        rows.append({"osm_id": f"{'node' if obj.is_node() else 'way'}/{obj.id}", "kind": kind, "geometry": Point(lon, lat)})
    log.info("Read %d amenity points from %s", len(rows), Path(pbf_path).name)
    return gpd.GeoDataFrame(rows, columns=["osm_id", "kind", "geometry"], geometry="geometry", crs=4326)


def count_points_by_ward(points: gpd.GeoDataFrame, wards: gpd.GeoDataFrame, *, id_col: str = "geoid") -> pd.DataFrame:
    w = wards[[id_col, wards.geometry.name]].drop_duplicates(id_col).to_crs(4326)
    counts = pd.DataFrame(0, index=w[id_col].astype(str).values, columns=AMENITY_COLUMNS, dtype="int64")
    if len(points):
        j = gpd.sjoin(points, w, predicate="within", how="inner")
        j[id_col] = j[id_col].astype(str)
        tab = j.groupby([id_col, "kind"]).size().unstack(fill_value=0)
        counts.loc[tab.index, tab.columns] = tab
    counts.index.name = id_col
    return counts.reset_index()


def build_amenity_table(wards: gpd.GeoDataFrame, out_path: str | Path, *, pbf_path: str | Path | None = None,
                        raw_dir: str | Path | None = None) -> Path:
    """Count amenities per ward from a PBF (downloaded from Geofabrik when not given) and save Parquet."""
    if pbf_path is None:
        dest = Path(raw_dir or ".") / "south-africa-latest.osm.pbf"
        errors = []
        for url in MIRROR_URLS:
            try:
                pbf_path = download_file(url, dest, timeout=1800)
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                errors.append(f"{url}: {exc}")
        if pbf_path is None:
            raise RuntimeError(
                "Could not download the OpenStreetMap extract automatically:\n  " + "\n  ".join(errors)
                + "\nDownload south-africa-latest.osm.pbf in your browser from "
                  "https://download.geofabrik.de/africa/south-africa.html and re-run with --pbf <that file>.")
    table = count_points_by_ward(read_pbf_points(pbf_path), wards)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False, compression="zstd")
    return out_path
