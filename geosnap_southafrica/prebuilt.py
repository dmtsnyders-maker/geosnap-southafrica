"""Prebuilt, small derived tables (published as GitHub Release assets) that save users slow first-run work.

Only DERIVED numbers are shared here, never raw source files. The WorldPop ward sums are derived from
WorldPop Global2 R2025A (CC BY 4.0, (c) WorldPop, University of Southampton): for each 2020 ward and year the
summed 100 m population raster. Set ``GEOSNAP_SA_NO_PREBUILT=1`` to always compute locally.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from .annual import ANCHOR_YEAR, WORLDPOP_YEARS, ward_raster_sums, worldpop_raster
from .downloads import download_file

log = logging.getLogger(__name__)

RELEASE_BASE = "https://github.com/dmtsnyders-maker/geosnap-southafrica/releases/download/data-v1/"
WORLDPOP_TABLE = "worldpop_ward_sums.parquet"
_failed: set[str] = set()


def disabled() -> bool:
    return os.environ.get("GEOSNAP_SA_NO_PREBUILT", "").strip().lower() in {"1", "true", "yes"}


def fetch_prebuilt(name: str, cache_dir: str | Path, *, timeout: int = 60) -> Path | None:
    """Local path of a prebuilt asset, downloading it from the GitHub Release once; None if unavailable."""
    if disabled():
        return None
    dest = Path(cache_dir) / "prebuilt" / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if name in _failed:
        return None
    try:
        log.info("Downloading prebuilt %s ...", name)
        return download_file(RELEASE_BASE + name, dest, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - fall back to computing locally
        _failed.add(name)
        log.info("Prebuilt %s not available (%s); computing locally.", name, exc)
        return None


def worldpop_sums(year: int, geoids, cache_dir: str | Path) -> pd.Series | None:
    """Prebuilt WorldPop ward sums for ``year`` aligned to ``geoids``, or None if not available/complete."""
    path = fetch_prebuilt(WORLDPOP_TABLE, cache_dir)
    if path is None:
        return None
    table = pd.read_parquet(path)
    sub = table[table["year"] == int(year)].set_index("geoid")["sum"]
    ids = pd.Index([str(g) for g in geoids])
    if not ids.isin(sub.index).all():
        return None
    return sub.reindex(ids)


def build_worldpop_table(wards, raw_dir: str | Path, out_path: str | Path, years=WORLDPOP_YEARS) -> Path:
    """Compute ward sums for every year (needs ~100 MB per year of downloads) and save one Parquet file."""
    rows = []
    for year in sorted(set(years) | {ANCHOR_YEAR}):
        log.info("WorldPop %d ...", year)
        s = ward_raster_sums(worldpop_raster(year, raw_dir), wards)
        rows.append(pd.DataFrame({"geoid": wards["geoid"].astype(str).values, "year": int(year), "sum": s.values}))
    out = pd.concat(rows, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")
    return out_path
