"""South African data connector for geosnap.

The public ``get_south_africa`` function follows the current geosnap 0.17 constructor
convention: it returns a long-form GeoDataFrame with ``geoid`` and ``year``
columns. The connector uses the Stats SA 2020-ward statistical product as its
observed/control data and can optionally pass the result into the existing
WorldPop refinement layer.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

from .config import load_config
from .downloads import download_file
from .spatial import join_ward_estimates, load_wards
from .ward_data import load_official_ward_estimates
from .worldpop import build_worldpop_surfaces
from .annual import ANCHOR_YEAR, WORLDPOP_YEARS, build_annual
from .community import prepare_year_frames
from .suburbs import add_suburb_columns, load_suburbs

log = logging.getLogger(__name__)

CENSUS_YEARS = (2022,)  # years with an official Stats SA ward product
SUPPORTED_YEARS = tuple(sorted(set(CENSUS_YEARS) | set(WORLDPOP_YEARS)))  # other years are modelled (see annual.py)
SCHEMA_VERSION = 2  # bump when cached output columns change


def _normalise_years(years) -> list[int]:
    if years == "all":
        return list(CENSUS_YEARS)
    if isinstance(years, (int, str)):
        return [int(years)]
    return sorted({int(y) for y in years})


def _data_root(datastore, data_dir: str | Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    if datastore is not None and hasattr(datastore, "data_dir"):
        return Path(datastore.data_dir).expanduser().resolve()
    try:
        from platformdirs import user_data_dir
        return Path(user_data_dir("geosnap", "geosnap")).resolve()
    except Exception:
        return Path("~/.geosnap").expanduser().resolve()


def _hash_request(years: Iterable[int], province: str | None) -> str:
    payload = json.dumps({"years": sorted(int(y) for y in years), "province": province, "schema": SCHEMA_VERSION}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _filter_municipality(gdf: gpd.GeoDataFrame, municipality: str | None) -> gpd.GeoDataFrame:
    """Filter by MDB code or name: exact match first, then a unique partial name match."""
    if not municipality:
        return gdf
    want = municipality.strip().casefold()
    names = gdf["municipality_name"].astype(str).str.strip() if "municipality_name" in gdf.columns else pd.Series("", index=gdf.index)
    codes = gdf["municipality_code"].astype(str).str.strip() if "municipality_code" in gdf.columns else pd.Series("", index=gdf.index)
    mask = (names.str.casefold() == want) | (codes.str.casefold() == want)
    if not mask.any():
        partial = names.str.casefold().str.contains(want, regex=False)
        matched = sorted(set(names[partial]))
        if len(matched) > 1:
            raise ValueError(f"Municipality {municipality!r} is ambiguous; matches: {matched}")
        mask = partial
    if not mask.any():
        raise ValueError(f"No wards found for municipality {municipality!r}. Examples: {sorted(set(names))[:20]}")
    return gdf.loc[mask].copy()


_MUNI_SUFFIXES = (" Metropolitan Municipality", " Local Municipality", " District Municipality")


def _add_ward_names(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a readable ``ward_name`` such as "City of Cape Town - Ward 12" (derived, not cached)."""
    if gdf.empty or "ward_name" in gdf.columns:
        return gdf
    out = gdf.copy()
    if "municipality_name" in out.columns:
        base = out["municipality_name"].astype(str).str.strip()
        for suffix in _MUNI_SUFFIXES:
            base = base.str.removesuffix(suffix)
    elif "municipality_code" in out.columns:
        base = out["municipality_code"].astype(str)
    else:
        base = pd.Series("", index=out.index)
    num = out["ward_number"].astype(str) if "ward_number" in out.columns else out["geoid"].astype(str)
    out["ward_name"] = (base + " - Ward " + num).str.lstrip(" -")
    return out


def _filter_ward(gdf: gpd.GeoDataFrame, ward) -> gpd.GeoDataFrame:
    """Keep specific wards by 8-digit id, label ("CPT_1"), name, or ward number (best with municipality=)."""
    if ward is None:
        return gdf
    wanted = {str(w).strip().casefold() for w in ([ward] if isinstance(ward, (str, int)) else ward)}
    keys = [c for c in ("geoid", "ward_label", "ward_name", "ward_number") if c in gdf.columns]
    mask = pd.Series(False, index=gdf.index)
    for c in keys:
        mask |= gdf[c].astype(str).str.strip().str.casefold().isin(wanted)
    if not mask.any():
        raise ValueError(f"No wards matched {sorted(wanted)}. Use an 8-digit id, label like 'CPT_1', a ward_name, or a ward number.")
    return gdf.loc[mask].copy()


def _filter_boundary(gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    if boundary is None:
        return gdf
    b = boundary.copy()
    if b.crs is None:
        raise ValueError("boundary must have a CRS")
    b = b.to_crs(4326)
    mask = gdf.to_crs(4326).representative_point().intersects(b.union_all())
    return gdf.loc[mask].copy()


def _load_base_longform(
    data_root: Path,
    *,
    years: list[int],
    province: str | None,
) -> gpd.GeoDataFrame:
    """Build the official Stats SA ward dataset and cache it as GeoParquet."""
    sa_root = data_root / "south_africa"
    raw_dir = sa_root / "raw"
    cache_dir = sa_root / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"south_africa_wards_{_hash_request(years, province)}.parquet"
    if cache_file.exists():
        log.info("Using cached South Africa geosnap dataset: %s", cache_file)
        return gpd.read_parquet(cache_file)

    package_config = Path(__file__).with_name("default_config.yaml")
    cfg = load_config(package_config)
    archive = raw_dir / cfg.get("stats_sa_ward_zip_filename", "Ward-Product_Locked-spreadsheets.zip")
    if not archive.exists():
        download_file(cfg["stats_sa_ward_zip_url"], archive, timeout=600)

    schema_cache = cache_dir / "stats_sa_schema.json"
    estimates = load_official_ward_estimates(archive, schema_cache=schema_cache)
    estimates = estimates[estimates["year"].isin(years)].copy()

    wards = load_wards(
        cache_dir=cache_dir,
        service_url=cfg.get("ward_service_url") or None,
        search_terms=cfg.get("ward_search_terms"),
        province_filter=province,
    )
    joined = join_ward_estimates(wards, estimates, province)
    frames = prepare_year_frames(joined, years)

    out_frames: list[gpd.GeoDataFrame] = []
    for year in sorted(frames):
        frame = frames[year].copy()
        frame = frame.to_crs(4326)
        # Geosnap's current constructor contract uses a stable geographic ID plus year.
        frame["geoid"] = frame["ward_code"].astype(str)
        frame["year"] = int(year)
        # Same column names in every year: drop the "_<year>" duplicates/suffixes of the source columns.
        suffix = f"_{year}"
        frame = frame.drop(columns=[c for c in frame.columns if c.endswith(suffix) and (c.startswith("pct_") or c.startswith("population_"))])
        frame = frame.rename(columns={c: c[: -len(suffix)] for c in frame.columns if c.endswith(suffix) and c.startswith(("age_", "sex_"))})
        frame["population_basis"] = f"stats_sa_{year}"

        # Explicit geosnap-style aliases while retaining the transparent Stats SA names.
        if "population" in frame.columns:
            frame["n_total_pop"] = pd.to_numeric(frame["population"], errors="coerce")
        aliases = {
            "pct_black_african": "p_black_african",
            "pct_coloured": "p_coloured",
            "pct_indian_asian": "p_indian_asian",
            "pct_white": "p_white",
            "pct_other": "p_other",
        }
        for src, dst in aliases.items():
            if src in frame.columns:
                frame[dst] = frame[src]
        out_frames.append(frame)

    # Keep the same schema across time, as geosnap's U.S. constructors do.
    common = set.intersection(*(set(f.columns) for f in out_frames))
    preferred = [
        "geoid", "year", "ward_code", "ward_number", "municipality_code", "province_name",
        "population", "n_total_pop",
        "pct_black_african", "pct_coloured", "pct_indian_asian", "pct_white", "pct_other",
        "p_black_african", "p_coloured", "p_indian_asian", "p_white", "p_other",
        "working_age_share", "youth_share", "older_share", "dependency_ratio",
        "youth_dependency_ratio", "old_age_dependency_ratio", "aging_index",
        "potential_support_ratio", "geometry",
    ]
    ordered = [c for c in preferred if c in common] + [c for c in sorted(common) if c not in preferred and c != "geometry"]
    long = gpd.GeoDataFrame(pd.concat([f[ordered] for f in out_frames], ignore_index=True), geometry="geometry", crs=4326)
    long = long.drop_duplicates(subset=["geoid", "year"]).reset_index(drop=True)
    long.to_parquet(cache_file, index=False)
    return long


def get_south_africa(
    datastore=None,
    *,
    years: str | int | Iterable[int] = "all",
    province: str | None = None,
    boundary: gpd.GeoDataFrame | None = None,
    data_dir: str | Path | None = None,
    municipality: str | None = None,
    ward: str | int | Iterable | None = None,
    suburbs=None,
    suburb_name_field: str | None = None,
    suburb_min_share: float = 0.05,
) -> gpd.GeoDataFrame:
    """Extract South African ward-level neighbourhood data in geosnap format.

    Parameters
    ----------
    datastore : geosnap.DataStore, optional
        Existing geosnap data store. Its ``data_dir`` is used for connector cache files.
    years : {"all", int, iterable of int}, optional
        2022 (official Stats SA ward estimates) and/or any year 2015-2030 (modelled, anchored to 2022).
        ``"all"`` means the census years only (2022).
    province : str, optional
        Province name used to restrict the returned wards.
    boundary : geopandas.GeoDataFrame, optional
        Study-area boundary. Ward representative points inside the boundary are retained.
    data_dir : str or pathlib.Path, optional
        Override the cache/data directory.
    municipality : str, optional
        Municipality name (e.g. ``"City of Cape Town"``; a unique partial name is enough) or
        MDB code (e.g. ``"CPT"``), matched case-insensitively. Applied after the province filter.

    ward : str, int or iterable, optional
        Select individual wards by 8-digit ward id (``19100001``), label (``"CPT_1"``),
        ``ward_name`` (``"City of Cape Town - Ward 1"``) or ward number. A bare number
        matches that ward in every municipality, so combine it with ``municipality=``.

    suburbs : bool, str, pathlib.Path or geopandas.GeoDataFrame, optional
        Add ``main_suburb``, ``main_suburb_share``, ``suburbs`` (semicolon-separated list) and
        ``suburb_count`` by overlaying a suburb layer on the wards. ``True`` / ``"osm"`` uses
        OpenStreetMap suburb points (nationwide; wards get the suburbs whose point falls inside,
        nearest the ward centre first). ``"cape_town"`` uses the City of Cape Town official suburb
        polygons. Or pass a file path, an ArcGIS layer URL or a GeoDataFrame of any suburb layer.
    suburb_name_field : str, optional
        Column holding suburb names (auto-detected when omitted).
    suburb_min_share : float, optional
        Minimum share of a ward a suburb must cover to be listed (default 5%).

    Returns
    -------
    geopandas.GeoDataFrame
        Long-form geodataframe with ``geoid`` and ``year`` fields, suitable for
        geosnap's current analysis/harmonisation functions.

    Notes
    -----
    The Stats SA ward product is a single (2022) vintage on the 2020 ward geography; ``geoid`` is the
    8-digit ward code. ``years=2022`` is official census-based data (``population_basis == "stats_sa_2022"``).
    Any other year in 2015-2030 is a MODELLED series: each ward's 2022 counts are scaled by the change in
    its WorldPop population (``population_basis == "worldpop_anchored"``), with race and age shares held at
    their 2022 values. See :mod:`geosnap_southafrica.annual`.
    """
    requested = _normalise_years(years)
    unsupported = sorted(set(requested) - set(SUPPORTED_YEARS))
    if unsupported:
        raise ValueError(f"Unsupported years: {unsupported}. Census year: {CENSUS_YEARS}; modelled annual years: "
                         f"{WORLDPOP_YEARS[0]}-{WORLDPOP_YEARS[-1]}.")

    root = _data_root(datastore, data_dir)
    # The census ward table (2022) is always the anchor; other years are derived from it.
    long = _load_base_longform(root, years=list(CENSUS_YEARS), province=province)
    long = _add_ward_names(long)
    long = _filter_municipality(long, municipality)
    long = _filter_ward(long, ward)
    long = _filter_boundary(long, boundary)
    long = long.reset_index(drop=True)

    extra = [y for y in requested if y not in CENSUS_YEARS]
    if extra:
        sa_root = root / "south_africa"
        annual = build_annual(long, extra, sa_root / "raw", sa_root / "cache")
        long = gpd.GeoDataFrame(pd.concat([long, annual], ignore_index=True), geometry="geometry", crs=long.crs)
    long = long[long["year"].isin(requested)].sort_values(["year", "geoid"]).reset_index(drop=True)

    if suburbs is not None and suburbs is not False:
        source = suburbs
        if suburbs is True:
            source = "osm"
        elif isinstance(suburbs, str) and suburbs.lower() == "cape_town":
            source = load_config(Path(__file__).with_name("default_config.yaml")).get("suburb_service_url")
        # Limit OSM queries to the (padded) extent of the wards actually requested.
        minx, miny, maxx, maxy = long.total_bounds
        bbox = (minx - 0.02, miny - 0.02, maxx + 0.02, maxy + 0.02)
        sub = load_suburbs(source, cache_dir=root / "south_africa" / "cache", name_field=suburb_name_field, bbox=bbox)
        parts = [add_suburb_columns(long[long["year"] == y], sub, min_share=suburb_min_share) for y in sorted(long["year"].unique())]
        long = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=long.crs)
    return long


def store_south_africa(
    datastore=None,
    *,
    years: str | int | Iterable[int] = "all",
    province: str | None = None,
    data_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Materialize the South African connector output to a GeoParquet file."""
    root = _data_root(datastore, data_dir)
    path = root / "south_africa" / f"south_africa_wards_{_hash_request(_normalise_years(years), province)}.parquet"
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = get_south_africa(datastore, years=years, province=province, data_dir=root)
    gdf.to_parquet(path, index=False)
    return path


def get_south_africa_population_surface(
    datastore=None,
    *,
    years: Iterable[int] = (2022,),
    province: str | None = None,
    mode: str = "comparable",
    data_dir: str | Path | None = None,
) -> dict[int, dict]:
    """Build the optional WorldPop-calibrated 100 m population surfaces.

    This is intentionally a companion to ``get_south_africa`` rather than part of the
    core geosnap tabular connector because geosnap's native input is a long-form
    GeoDataFrame; raster allocation is an auxiliary refinement step.
    """
    package_config = Path(__file__).with_name("default_config.yaml")
    cfg = load_config(package_config)
    root = _data_root(datastore, data_dir)
    raw_dir = root / "south_africa" / "raw"
    outputs_dir = root / "south_africa" / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    long = get_south_africa(datastore, years=years, province=province, data_dir=root)
    frames = {int(y): long[long["year"] == int(y)].copy() for y in sorted(set(int(x) for x in years))}
    pop_cols = {y: "population" for y in frames}
    return build_worldpop_surfaces(
        sorted(frames), frames, raw_dir, outputs_dir,
        mode=mode, reference_year=int(cfg.get("worldpop", {}).get("reference_year", 2022)),
        url_2022=cfg["worldpop_url_2022"], urls_2011=cfg.get("worldpop_urls_2011", []),
        population_columns=pop_cols,
        zero_weight_fallback=cfg.get("worldpop", {}).get("zero_weight_fallback", "uniform"),
        crop_to_wards=bool(cfg.get("worldpop", {}).get("crop_to_wards", True)),
    )
