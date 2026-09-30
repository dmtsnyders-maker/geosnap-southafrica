from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box

from .downloads import download_file, download_first_working

log = logging.getLogger(__name__)


def download_worldpop(year: int, raw_dir: str | Path, *, url_2022: str, urls_2011: Iterable[str]) -> Path:
    """Download the configured WorldPop raster for a year."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if year == 2022:
        return download_file(url_2022, raw_dir / "worldpop_2022_100m.tif", timeout=600)
    if year == 2011:
        return download_first_working(urls_2011, raw_dir / "worldpop_2011_100m.tif")
    raise ValueError(f"Unsupported WorldPop year: {year}")


def _prepare_shapes(wards: gpd.GeoDataFrame, raster_crs) -> gpd.GeoDataFrame:
    out = wards.copy()
    if out.crs is None:
        out = out.set_crs("EPSG:4326")
    return out.to_crs(raster_crs)


def _window_iter(src: rasterio.io.DatasetReader, block_rows: int = 512, block_cols: int | None = None, extent: Window | None = None):
    extent = extent or Window(0, 0, src.width, src.height)
    block_cols = block_cols or src.width
    row_start = int(extent.row_off)
    row_end = int(extent.row_off + extent.height)
    col_start = int(extent.col_off)
    col_end = int(extent.col_off + extent.width)
    for row_off in range(row_start, row_end, block_rows):
        height = min(block_rows, row_end - row_off)
        for col_off in range(col_start, col_end, block_cols):
            width = min(block_cols, col_end - col_off)
            yield Window(col_off, row_off, width, height)


def _candidate_indices(wards: gpd.GeoDataFrame, bbox: tuple[float, float, float, float]) -> np.ndarray:
    minx, miny, maxx, maxy = bbox
    try:
        hits = wards.sindex.query(box(minx, miny, maxx, maxy), predicate="intersects")
        return np.asarray(hits, dtype=np.int64)
    except Exception:
        return np.arange(len(wards), dtype=np.int64)


def calibrate_worldpop_to_wards(
    worldpop_path: str | Path,
    wards: gpd.GeoDataFrame,
    population_column: str,
    output_path: str | Path,
    checks_path: str | Path,
    *,
    zero_weight_fallback: str = "uniform",
    block_rows: int = 512,
    crop_to_wards: bool = True,
) -> tuple[Path, Path]:
    """Calibrate a WorldPop raster to exact ward control totals using windowed processing.

    Parameters
    ----------
    zero_weight_fallback : {"uniform", "fail"}
        Behaviour when a ward has no positive WorldPop weight. ``uniform`` assigns
        equal weight to valid raster cells intersecting the ward.
    """
    worldpop_path = Path(worldpop_path)
    output_path = Path(output_path)
    checks_path = Path(checks_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checks_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(worldpop_path) as src:
        wards_r = _prepare_shapes(wards, src.crs)
        targets = pd.to_numeric(wards_r[population_column], errors="coerce").fillna(0).to_numpy(float)
        if (targets < 0).any():
            raise ValueError("Population controls cannot be negative.")

        profile = src.profile.copy()
        if crop_to_wards:
            total_bounds = tuple(wards_r.total_bounds)
            try:
                win = src.window(*total_bounds)
                win = win.round_offsets().round_lengths()
                win = Window(max(0, win.col_off), max(0, win.row_off),
                             min(src.width - max(0, win.col_off), win.width),
                             min(src.height - max(0, win.row_off), win.height))
            except Exception:
                win = Window(0, 0, src.width, src.height)
        else:
            win = Window(0, 0, src.width, src.height)

        weights = np.zeros(len(wards_r), dtype="float64")
        cell_counts = np.zeros(len(wards_r), dtype="int64")

        # Pass 1: calculate positive WorldPop weights and fallback cell counts.
        for window in _window_iter(src, block_rows=block_rows, extent=win):
            bbox = window_bounds(window, src.transform)
            idx = _candidate_indices(wards_r, bbox)
            if len(idx) == 0:
                continue
            shapes = [(wards_r.iloc[i].geometry, int(i)) for i in idx]
            ids = rasterize(shapes, out_shape=(int(window.height), int(window.width)),
                            transform=src.window_transform(window), fill=-1, dtype="int32")
            data = src.read(1, window=window, masked=True).astype("float64")
            arr = np.where(np.ma.getmaskarray(data) | ~np.isfinite(np.asarray(data)) | (np.asarray(data) <= 0), 0.0, np.asarray(data))
            flat_ids = ids.ravel()
            flat_arr = arr.ravel()
            positive = (flat_ids >= 0) & (flat_arr > 0)
            if positive.any():
                np.add.at(weights, flat_ids[positive], flat_arr[positive])
            valid_mask = (~np.ma.getmaskarray(data)).ravel() & np.isfinite(np.asarray(data)).ravel() & (flat_ids >= 0)
            if valid_mask.any():
                np.add.at(cell_counts, flat_ids[valid_mask], 1)

        zero = (weights <= 0) & (targets > 0)
        if zero.any() and zero_weight_fallback == "fail":
            bad = wards_r.iloc[np.flatnonzero(zero)]["ward_code"].tolist()
            raise RuntimeError(f"WorldPop has zero positive weight in {len(bad)} populated wards: {bad[:10]}")
        if zero.any() and zero_weight_fallback == "uniform":
            log.warning("Applying uniform-cell fallback to %d populated wards with zero WorldPop weight", int(zero.sum()))

        effective_weights = weights.copy()
        effective_weights[zero] = cell_counts[zero].astype(float)
        factors = np.divide(targets, effective_weights, out=np.zeros_like(targets), where=effective_weights > 0)

        profile.update(dtype="float32", count=1, compress="deflate", predictor=2, nodata=0.0)
        profile.update(width=int(win.width), height=int(win.height), transform=src.window_transform(win))

        with rasterio.open(output_path, "w", **profile) as dst:
            for window in _window_iter(src, block_rows=block_rows, extent=win):
                out_window = Window(window.col_off - win.col_off, window.row_off - win.row_off, window.width, window.height)
                bbox = window_bounds(window, src.transform)
                idx = _candidate_indices(wards_r, bbox)
                if len(idx) == 0:
                    dst.write(np.zeros((int(window.height), int(window.width)), dtype="float32"), 1, window=out_window)
                    continue
                shapes = [(wards_r.iloc[i].geometry, int(i)) for i in idx]
                ids = rasterize(shapes, out_shape=(int(window.height), int(window.width)),
                                transform=src.window_transform(window), fill=-1, dtype="int32")
                data = src.read(1, window=window, masked=True).astype("float64")
                data_mask = np.ma.getmaskarray(data)
                arr = np.where(data_mask | ~np.isfinite(np.asarray(data)) | (np.asarray(data) <= 0), 0.0, np.asarray(data))
                positive = arr > 0
                valid_cells = (~data_mask) & np.isfinite(np.asarray(data))
                safe_ids = np.clip(ids, 0, len(factors) - 1)
                values = arr * factors[safe_ids]
                # Uniform fallback is only used in zero-weight wards, so every valid raster cell in that ward gets equal weight.
                if zero.any():
                    fb_mask = np.isin(ids, np.flatnonzero(zero)) & (ids >= 0) & valid_cells
                    values[fb_mask] = np.take(
                        np.divide(targets, cell_counts, out=np.zeros_like(targets), where=cell_counts > 0),
                        safe_ids[fb_mask],
                    )
                values = np.where((ids >= 0) & valid_cells & (positive | zero[np.clip(ids, 0, len(zero)-1)]), values, 0.0)
                dst.write(values.astype("float32"), 1, window=out_window)

    # Pass 3: validate by re-reading the output in windows and summing by ward.
    actual = np.zeros(len(wards_r), dtype=float)
    with rasterio.open(output_path) as out_src:
        # Output has same grid as cropped source. Rebuild IDs window by window using original ward geometry.
        for window in _window_iter(out_src, block_rows=block_rows):
            bbox = window_bounds(window, out_src.transform)
            idx = _candidate_indices(wards_r, bbox)
            if len(idx) == 0:
                continue
            ids = rasterize([(wards_r.iloc[i].geometry, int(i)) for i in idx],
                            out_shape=(int(window.height), int(window.width)),
                            transform=out_src.window_transform(window), fill=-1, dtype="int32")
            vals = out_src.read(1, window=window).astype(float)
            flat_ids = ids.ravel()
            flat_vals = vals.ravel()
            maskv = flat_ids >= 0
            if maskv.any():
                np.add.at(actual, flat_ids[maskv], flat_vals[maskv])

    checks = wards_r[["ward_code"]].copy()
    checks[population_column] = targets
    checks["worldpop_weight_total"] = weights
    checks["fallback_cell_count"] = cell_counts
    checks["allocation_method"] = np.where(zero, "uniform", "worldpop")
    checks["calibration_factor"] = factors
    checks["calibrated_total"] = actual
    checks["absolute_difference"] = actual - targets
    checks["relative_difference_pct"] = np.where(targets > 0, 100 * (actual - targets) / targets, 0)
    checks.to_csv(checks_path, index=False)
    max_err = float(np.nanmax(np.abs(checks["relative_difference_pct"]))) if len(checks) else 0.0
    log.info("WorldPop calibration finished: max ward relative error %.8f%%", max_err)
    return output_path, checks_path


def disaggregate_ward_share_to_raster(calibrated_population_raster: str | Path, wards: gpd.GeoDataFrame,
                                      ward_share_column: str, output_path: str | Path) -> Path:
    """Disaggregate a ward share to cells using the calibrated population surface."""
    calibrated_population_raster = Path(calibrated_population_raster)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(calibrated_population_raster) as src:
        wards_r = _prepare_shapes(wards, src.crs)
        shares = pd.to_numeric(wards_r[ward_share_column], errors="coerce").fillna(0).to_numpy(float)
        profile = src.profile.copy()
        with rasterio.open(output_path, "w", **profile) as dst:
            for window in _window_iter(src):
                bbox = window_bounds(window, src.transform)
                idx = _candidate_indices(wards_r, bbox)
                ids = rasterize([(wards_r.iloc[i].geometry, int(i)) for i in idx],
                                out_shape=(int(window.height), int(window.width)),
                                transform=src.window_transform(window), fill=-1, dtype="int32") if len(idx) else np.full((int(window.height), int(window.width)), -1, dtype=np.int32)
                out = np.where(ids >= 0, shares[np.clip(ids, 0, len(shares)-1)], 0.0).astype("float32")
                dst.write(out, 1, window=window)
    return output_path


def build_worldpop_surfaces(
    years: list[int],
    wards_by_year: dict[int, gpd.GeoDataFrame],
    raw_dir: str | Path,
    outputs_dir: str | Path,
    *,
    mode: str,
    reference_year: int,
    url_2022: str,
    urls_2011: Iterable[str],
    population_columns: dict[int, str],
    zero_weight_fallback: str = "uniform",
    crop_to_wards: bool = True,
) -> dict[int, dict[str, Path | str]]:
    """Build year-specific or longitudinally comparable calibrated WorldPop surfaces."""
    raw_dir, outputs_dir = Path(raw_dir), Path(outputs_dir)
    if mode not in {"comparable", "year_specific"}:
        raise ValueError("worldpop mode must be 'comparable' or 'year_specific'")
    if mode == "comparable":
        if reference_year not in {2011, 2022}:
            raise ValueError("reference_year must be 2011 or 2022")
        ref = download_worldpop(reference_year, raw_dir, url_2022=url_2022, urls_2011=urls_2011)
        source_by_year = {y: ref for y in years}
    else:
        source_by_year = {y: download_worldpop(y, raw_dir, url_2022=url_2022, urls_2011=urls_2011) for y in years}

    outputs: dict[int, dict[str, Path | str]] = {}
    for year in years:
        suffix = f"worldpop_{mode}_{year}_100m.tif"
        out = outputs_dir / suffix
        checks = outputs_dir / f"worldpop_{mode}_ward_checks_{year}.csv"
        calibrate_worldpop_to_wards(
            source_by_year[year], wards_by_year[year], population_columns[year], out, checks,
            zero_weight_fallback=zero_weight_fallback, crop_to_wards=crop_to_wards,
        )
        outputs[year] = {"source": source_by_year[year], "output": out, "checks": checks, "mode": mode}
    return outputs
