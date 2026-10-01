"""Annual ward population series anchored to the 2022 census ward estimates.

Stats SA publishes the ward product for a single vintage (2022). For other years
this module rescales each ward's 2022 census counts by how that ward's WorldPop
(Global2 R2025A, 100 m, constrained) population changed between the anchor year
and the requested year:

    pop_year(ward) = pop_2022(ward) * WorldPop_year(ward) / WorldPop_2022(ward)

Everything count-like (population, age bands, sex) is scaled by the same factor;
shares (race, age structure) are carried over from 2022. These rows are MODELLED
estimates - they are flagged with ``population_basis == "worldpop_anchored"`` and
years after the latest WorldPop observations are projections.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .downloads import download_file

log = logging.getLogger(__name__)

ANCHOR_YEAR = 2022
WORLDPOP_YEARS = tuple(range(2015, 2031))
WORLDPOP_URL = ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{year}/ZAF/v1/100m/"
                "constrained/zaf_pop_{year}_CN_100m_R2025A_v1.tif")
# A ward with fewer people than this in the anchor raster is too small to trust its own ratio.
MIN_ANCHOR_POP = 5.0


def worldpop_raster(year: int, raw_dir: str | Path) -> Path:
    """Return the local path of the WorldPop raster for ``year``, downloading it if needed."""
    if year not in WORLDPOP_YEARS:
        raise ValueError(f"WorldPop Global2 covers {WORLDPOP_YEARS[0]}-{WORLDPOP_YEARS[-1]}; got {year}")
    dest = Path(raw_dir) / f"worldpop_zaf_{year}_100m_R2025A.tif"
    if not dest.exists():
        log.info("Downloading WorldPop %d (about 100 MB, first use only) ...", year)
        download_file(WORLDPOP_URL.format(year=year), dest, timeout=900)
    return dest


def ward_raster_sums(raster_path: str | Path, wards: gpd.GeoDataFrame, *, max_block_cells: int = 20_000_000) -> pd.Series:
    """Sum raster values inside each ward (cell-centre rule).

    Only the window of the raster covering the wards' bounding box is read, in row blocks, so a
    single city touches a tiny part of the national raster.
    """
    try:
        import rasterio
        from rasterio.features import rasterize
        from rasterio.windows import Window
        from rasterio.windows import bounds as window_bounds
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Annual series need rasterio: pip install rasterio") from exc
    from shapely.geometry import box

    with rasterio.open(raster_path) as src:
        w = wards.reset_index(drop=True).to_crs(src.crs)
        geoms = w.geometry.values
        n = len(w)
        sums = np.zeros(n, dtype="float64")
        minx, miny, maxx, maxy = w.total_bounds
        r0, c0 = src.index(minx, maxy)          # top-left cell of the wards' extent
        r1, c1 = src.index(maxx, miny)          # bottom-right cell
        r0, c0 = max(r0 - 1, 0), max(c0 - 1, 0)
        r1, c1 = min(r1 + 2, src.height), min(c1 + 2, src.width)
        if r1 > r0 and c1 > c0:
            width = c1 - c0
            block_rows = max(64, min(1024, max_block_cells // max(width, 1)))
            sindex = w.sindex
            for row in range(r0, r1, block_rows):
                window = Window(c0, row, width, min(block_rows, r1 - row))
                idx = sindex.query(box(*window_bounds(window, src.transform)), predicate="intersects")
                if len(idx) == 0:
                    continue
                data = src.read(1, window=window).astype("float64")
                data[~np.isfinite(data) | (data < 0)] = 0.0  # WorldPop nodata is a large negative number
                if not data.any():
                    continue
                labels = rasterize([(geoms[i], int(i) + 1) for i in idx], out_shape=data.shape,
                                   transform=src.window_transform(window), fill=0, dtype="int32")
                sums += np.bincount(labels.ravel(), weights=data.ravel(), minlength=n + 1)[1:]
    return pd.Series(sums, index=wards["geoid"].values if "geoid" in wards.columns else None)


def _cached_sums(year: int, wards: gpd.GeoDataFrame, raw_dir: Path, cache_dir: Path) -> pd.Series:
    key = hashlib.sha256(",".join(sorted(wards["geoid"].astype(str))).encode()).hexdigest()[:12]
    cache = cache_dir / f"worldpop_sums_{year}_{key}.csv"
    if cache.exists():
        s = pd.read_csv(cache, dtype={"geoid": str}).set_index("geoid")["sum"]
        return s.reindex(wards["geoid"].astype(str))
    from .prebuilt import worldpop_sums   # local import: prebuilt imports this module

    pre = worldpop_sums(year, wards["geoid"], cache_dir)
    if pre is not None:
        log.info("Using prebuilt WorldPop %d ward sums", year)
        return pre
    raster = worldpop_raster(year, raw_dir)
    log.info("Summing WorldPop %d over %d wards ...", year, len(wards))
    s = ward_raster_sums(raster, wards)
    s.index = wards["geoid"].astype(str).values
    cache_dir.mkdir(parents=True, exist_ok=True)
    s.rename("sum").rename_axis("geoid").reset_index().to_csv(cache, index=False)
    return s


def growth_ratios(target: pd.Series, anchor: pd.Series, group: pd.Series | None = None) -> pd.Series:
    """Per-ward ratio target/anchor, falling back to the municipality (then overall) ratio for tiny wards."""
    ratio = target / anchor.where(anchor > 0)
    if group is not None:
        tot = pd.DataFrame({"t": target.values, "a": anchor.values, "g": group.values}, index=target.index)
        gsum = tot.groupby("g")[["t", "a"]].transform("sum")
        gratio = gsum["t"] / gsum["a"].where(gsum["a"] > 0)
        ratio = ratio.where(anchor >= MIN_ANCHOR_POP, gratio)
    overall = target.sum() / anchor.sum() if anchor.sum() > 0 else 1.0
    return ratio.fillna(overall)


def scalable_columns(df: pd.DataFrame) -> list[str]:
    """Count-like columns that scale with population (shares and ratios do not)."""
    cols = [c for c in ("population", "n_total_pop") if c in df.columns]
    cols += [c for c in df.columns if c.startswith(("age_", "sex_")) and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def build_annual(census: gpd.GeoDataFrame, years: list[int], raw_dir: str | Path, cache_dir: str | Path) -> gpd.GeoDataFrame:
    """Return long-form rows for ``years`` (excluding the anchor year), one block per year."""
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    base = census[census["year"] == ANCHOR_YEAR].reset_index(drop=True)
    if base.empty:
        raise ValueError("Census anchor rows (2022) are required to build the annual series.")
    anchor = _cached_sums(ANCHOR_YEAR, base, raw_dir, cache_dir)
    group = base["municipality_code"].astype(str) if "municipality_code" in base.columns else None
    scale_cols = scalable_columns(base)
    blocks = []
    for year in sorted(set(years) - {ANCHOR_YEAR}):
        target = _cached_sums(year, base, raw_dir, cache_dir)
        ratio = growth_ratios(target.reset_index(drop=True), anchor.reset_index(drop=True), group)
        block = base.copy()
        for c in scale_cols:
            block[c] = block[c] * ratio.values
        block["year"] = int(year)
        block["population_basis"] = "worldpop_anchored"
        blocks.append(block)
        log.info("Year %d: national growth vs %d = %.3f", year, ANCHOR_YEAR, float(target.sum() / anchor.sum()))
    return gpd.GeoDataFrame(pd.concat(blocks, ignore_index=True), geometry="geometry", crs=base.crs) if blocks else base.iloc[0:0]
