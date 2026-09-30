from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

from .downloads import discover_ward_service, fetch_arcgis_geojson

log = logging.getLogger(__name__)


def _norm_col(c: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(c).lower())).strip()


def find_column(columns: Iterable[object], *terms: str) -> str | None:
    best: tuple[int, str] | None = None
    terms_n = [_norm_col(t) for t in terms]
    for c in columns:
        n = _norm_col(c)
        score = sum(20 if t == n else 8 if t in n else 0 for t in terms_n)
        if score and (best is None or score > best[0]):
            best = (score, str(c))
    return best[1] if best else None


def _clean_code(value: object) -> str:
    return re.sub(r"\.0$", "", str(value).strip())


def _ward_number(value: object, municipality: object | None = None) -> str:
    raw = _clean_code(value)
    muni = _clean_code(municipality) if municipality is not None else ""
    for sep in ("_", "|", ":"):
        prefix = muni + sep
        if muni and raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def load_wards(*, cache_dir: str | Path, service_url: str | None = None, search_terms: list[str] | None = None,
               province_filter: str | None = None) -> gpd.GeoDataFrame:
    """Load, standardise and validate the public 2020 electoral ward layer."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = cache_dir / "mdb_2020_electoral_wards.geojson"
    if not geojson_path.exists():
        service = discover_ward_service(service_url, search_terms or ["South Africa", "Electoral Ward", "2020"])
        log.info("Using ArcGIS ward service: %s", service)
        fetch_arcgis_geojson(service, geojson_path)
        (cache_dir / "ward_service_url.txt").write_text(service, encoding="utf-8")
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise RuntimeError("Ward geometry layer is empty.")
    gdf = gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")
    ward_col = find_column(gdf.columns, "ward code", "ward_code", "ward id", "ward_id", "ward number", "ward no", "ward")
    muni_col = find_column(gdf.columns, "municipality code", "municipality_code", "mun code", "cat b")
    prov_col = find_column(gdf.columns, "province", "province name", "prov name")
    if ward_col is None:
        raise RuntimeError(f"Could not identify ward identifier in MDB layer. Columns: {gdf.columns.tolist()}")

    out = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs)
    # Keep every ward-like identifier column so estimates can be matched on whichever one they use.
    for col in gdf.columns:
        n = _norm_col(col)
        if "ward" in n and col != gdf.geometry.name and gdf[col].dtype != "geometry":
            vals = gdf[col].map(_clean_code)
            if vals.str.fullmatch(r"\d{7,9}").mean() > 0.9:
                out["ward_id_raw"] = vals
                break
    if muni_col:
        out["municipality_code"] = gdf[muni_col].map(_clean_code)
        out["ward_number"] = [_ward_number(w, m) for w, m in zip(gdf[ward_col], gdf[muni_col])]
        out["ward_code"] = [f"{m}_{w}" for m, w in zip(out["municipality_code"], out["ward_number"])]
    else:
        out["ward_number"] = gdf[ward_col].map(_clean_code)
        out["ward_code"] = out["ward_number"].astype(str)
        if out["ward_code"].duplicated().any():
            out["ward_code"] = out["ward_code"] + "_" + out.groupby("ward_code").cumcount().astype(str)
    if prov_col:
        out["province_name"] = gdf[prov_col].astype(str).str.strip()
    # MDB layer truncates the field to "Municipali"; accept it and the full name.
    name_col = next((c for c in gdf.columns if _norm_col(c) in {"municipali", "municipality", "municipality name"}), None)
    if name_col:
        out["municipality_name"] = gdf[name_col].astype(str).str.strip()
    for label in ("municipality", "municipality name", "local municipality", "district", "district name"):
        col = find_column(gdf.columns, label)
        if col and col not in out.columns:
            out[_norm_col(label).replace(" ", "_")] = gdf[col].astype(str)

    if province_filter and "province_name" in out.columns:
        out = out[out["province_name"].str.casefold().str.strip() == province_filter.casefold().strip()].copy()
    out = out.drop_duplicates(subset="ward_code", keep="first").reset_index(drop=True)
    if out.geometry.isna().any() or (~out.geometry.is_valid).any():
        log.warning("Repairing invalid/empty ward geometries with make_valid().")
        out["geometry"] = out.geometry.make_valid()
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if len(out) < 4000 and province_filter is None:
        log.warning("Ward layer has only %d features; expected roughly 4,400+ 2020 wards.", len(out))
    return out


def join_ward_estimates(wards: gpd.GeoDataFrame, estimates: pd.DataFrame, province_filter: str | None = None) -> dict[int, gpd.GeoDataFrame]:
    """Join Stats SA ward estimates to the common 2020 ward geography deterministically."""
    if "ward_code" not in estimates.columns:
        raise ValueError("estimate table must contain ward_code")
    wards = wards.copy()
    estimates = estimates.copy()
    estimates["ward_code"] = estimates["ward_code"].map(_clean_code)
    id_cols = [c for c in ("ward_id_raw", "ward_code", "ward_number") if c in wards.columns]
    est_codes = set(estimates["ward_code"])
    best_col = max(id_cols, key=lambda c: wards[c].map(_clean_code).isin(est_codes).sum()) if id_cols else None
    if "municipality_code" not in estimates.columns and best_col is not None:
        wards["_join_key"] = wards[best_col].map(_clean_code)
        estimates["_join_key"] = estimates["ward_code"]
        log.info("Joining Stats SA wards on geometry column %r", best_col)
    elif "municipality_code" in wards.columns and "municipality_code" in estimates.columns:
        wards["_join_key"] = [f"{m}|{_ward_number(w, m)}" for m, w in zip(wards["municipality_code"], wards["ward_code"])]
        estimates["municipality_code"] = estimates["municipality_code"].map(_clean_code)
        estimates["_join_key"] = [f"{m}|{_ward_number(w, m)}" for m, w in zip(estimates["municipality_code"], estimates["ward_code"])]
    else:
        wards["_join_key"] = wards["ward_code"].map(_clean_code)
        estimates["_join_key"] = estimates["ward_code"].map(_clean_code)

    if "year" not in estimates.columns:
        raise ValueError("Stats SA estimates must contain a year field")
    outputs: dict[int, gpd.GeoDataFrame] = {}
    for year in sorted(int(y) for y in pd.to_numeric(estimates["year"], errors="coerce").dropna().unique()):
        e = estimates[pd.to_numeric(estimates["year"], errors="coerce") == year].copy()
        merged = wards.merge(e, on="_join_key", how="left", suffixes=("", "_estimate"))
        indicator_cols = [c for c in e.columns if c.startswith(("population_", "pct_", "age_", "sex_"))]
        if not indicator_cols:
            raise RuntimeError(f"No demographic variables were found in Stats SA {year} estimates.")
        matched = merged[indicator_cols].notna().any(axis=1)
        unmatched = int((~matched).sum())
        if unmatched:
            log.warning("[%d] %d of %d ward geometries did not match Stats SA estimates.", year, unmatched, len(merged))
        coverage = float(matched.mean()) if len(merged) else 0.0
        if coverage < 0.99:
            raise RuntimeError(f"Only {coverage:.2%} of ward geometries matched Stats SA {year} estimates; refusing to continue.")
        merged = merged[matched].copy()
        if "ward_code_estimate" in merged.columns:
            # Estimates keyed on the 8-digit Stats SA/MDB ward id: make it the public ward_code.
            merged["ward_label"] = merged["ward_code"]
            merged["ward_code"] = merged["ward_code_estimate"]
            merged = merged.drop(columns=["ward_code_estimate"])
        merged = merged.drop_duplicates(subset="ward_code", keep="first").drop(columns=["_join_key"], errors="ignore")
        merged["year"] = year
        outputs[year] = merged.reset_index(drop=True)
        log.info("[%d] joined %d wards", year, len(outputs[year]))
    if not outputs:
        raise RuntimeError("No Stats SA year slices could be joined to ward geometry.")
    return outputs
