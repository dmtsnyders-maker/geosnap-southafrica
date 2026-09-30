from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from geosnap_southafrica.analysis import run_analytics
from geosnap_southafrica.community import build_community
from geosnap_southafrica.config import load_config, project_paths
from geosnap_southafrica.downloads import download_file, sha256_file
from geosnap_southafrica.socioeconomic import disaggregate_socioeconomics, join_socioeconomics_to_wards, load_municipality_socioeconomics
from geosnap_southafrica.spatial import join_ward_estimates, load_wards
from geosnap_southafrica.transport import compute_transport_accessibility, download_public_transport_stops
from geosnap_southafrica.validation import run_full_qa
from geosnap_southafrica.ward_data import load_official_ward_estimates
from geosnap_southafrica.worldpop import build_worldpop_surfaces

log = logging.getLogger(__name__)


def _find_population_column(gdf, year: int) -> str:
    for c in (f"population_{year}", "population", f"pop_{year}", f"total_population_{year}"):
        if c in gdf.columns:
            return c
    raise RuntimeError(f"Could not find total population for {year}. Available fields: {gdf.columns.tolist()}")


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_all(config_path: str | Path = "config.yaml", province: str | None = None) -> dict[str, Any]:
    """Run the complete ten-stage South African neighbourhood analytics workflow."""
    cfg = load_config(config_path)
    if province is not None:
        cfg["province_filter"] = province
    paths = project_paths(cfg)

    # 1–2. Official Stats SA ingestion + deterministic schema validation.
    stats_zip = paths["raw"] / cfg.get("stats_sa_ward_zip_filename", "Ward-Product_Locked-spreadsheets.zip")
    if not stats_zip.exists():
        download_file(cfg["stats_sa_ward_zip_url"], stats_zip, timeout=600)
    schema_cache = paths["cache"] / "stats_sa_schema.json"
    estimates = load_official_ward_estimates(stats_zip, schema_cache=schema_cache)
    estimates.to_csv(paths["processed"] / "ward_estimates_long.csv", index=False)

    # Common 2020 electoral ward geometry.
    wards = load_wards(cache_dir=paths["cache"], service_url=cfg.get("ward_service_url") or None,
                       search_terms=cfg.get("ward_search_terms"), province_filter=cfg.get("province_filter"))
    gdfs = join_ward_estimates(wards, estimates, cfg.get("province_filter"))
    requested = [int(y) for y in cfg.get("years", [2022])]
    years = [y for y in requested if y in gdfs]
    if not years:
        raise RuntimeError(f"None of requested years {requested} were found after joining. Available: {sorted(gdfs)}")

    # 3. Formal QA on official ward slices.
    community, frames = build_community(gdfs, years)
    prepared = {y: frames[i] for i, y in enumerate(sorted(years))}
    qa_pre = run_full_qa(prepared, paths["outputs"])
    if qa_pre["status"] == "FAIL" and cfg.get("qa", {}).get("fail_pipeline_on_pre_analysis", True):
        raise RuntimeError(f"Pre-analysis QA failed; see {qa_pre['json']}")

    # 4–8. Longitudinal geosnap, multiscalar segregation, clustering/stability, and change.
    analysis = run_analytics(prepared, paths["outputs"], cfg.get("regionalization", {}), community=community)
    prepared = analysis.pop("frames", prepared)

    # 3–4 WorldPop refinement and comparable/year-specific modes.
    wp_cfg = cfg.get("worldpop", {})
    pop_cols = {y: _find_population_column(gdfs[y], y) for y in years}
    wp_surfaces = build_worldpop_surfaces(
        years, gdfs, paths["raw"], paths["outputs"],
        mode=wp_cfg.get("mode", "comparable"), reference_year=int(wp_cfg.get("reference_year", 2022)),
        url_2022=cfg["worldpop_url_2022"], urls_2011=cfg.get("worldpop_urls_2011", []),
        population_columns=pop_cols,
        zero_weight_fallback=wp_cfg.get("zero_weight_fallback", "uniform"),
        crop_to_wards=bool(wp_cfg.get("crop_to_wards", True)),
    )

    # Re-run QA including exact WorldPop calibration checks.
    qa_post = run_full_qa(prepared, paths["outputs"], [Path(v["checks"]) for v in wp_surfaces.values()])
    if qa_post["status"] == "FAIL" and cfg.get("qa", {}).get("fail_pipeline_on_post_worldpop", True):
        raise RuntimeError(f"Post-WorldPop QA failed; see {qa_post['json']}")

    # 9. Socioeconomic disaggregation when a deterministic municipal table is present.
    socio_result: dict[str, Any] = {"enabled": False, "reason": "no municipality socioeconomic table"}
    socio_path = Path(cfg.get("municipality_socioeconomic_csv", "data/raw/municipality_socioeconomic.csv"))
    if socio_path.exists():
        try:
            socio = load_municipality_socioeconomics(socio_path)
            for year in years:
                joined = join_socioeconomics_to_wards(gdfs[year], socio)
                socio_outputs = disaggregate_socioeconomics(wp_surfaces[year]["output"], joined, "municipality_code", paths["outputs"] / f"socioeconomic_{year}")
                socio_result = {"enabled": True, "year": year, "outputs": socio_outputs}
        except Exception as exc:
            socio_result = {"enabled": False, "error": str(exc)}
            log.warning("Socioeconomic stage failed but the demographic pipeline can continue: %s", exc)
    else:
        example = paths["raw"] / "municipality_socioeconomic.csv.example"
        if not example.exists():
            example.write_text("municipality_code,unemployment_rate,tertiary_education_share\nCPT,0.0,0.0\n", encoding="utf-8")

    # 10. Transport proximity/accessibility from public OSM data.
    transport_result: dict[str, Any] = {"enabled": False, "reason": "disabled"}
    tr_cfg = cfg.get("transport", {})
    if bool(tr_cfg.get("enabled", True)):
        try:
            stop_path = paths["outputs"] / "osm_public_transport_stops.gpkg"
            if not stop_path.exists():
                download_public_transport_stops(gdfs[max(years)], stop_path,
                                                endpoints=tr_cfg.get("overpass_endpoints"),
                                                tile_degrees=float(tr_cfg.get("tile_degrees", 5.0)),
                                                sleep_seconds=float(tr_cfg.get("sleep_seconds", 0.5)))
            access_csvs = {}
            for year in years:
                access_csvs[year] = compute_transport_accessibility(
                    wp_surfaces[year]["output"], gdfs[year], stop_path,
                    paths["outputs"] / f"transport_accessibility_{year}.csv",
                    radii_m=[int(x) for x in tr_cfg.get("radii_m", [400, 800])],
                )
            transport_result = {"enabled": True, "stops": stop_path, "accessibility": access_csvs,
                                "method": "Euclidean raster catchment around OSM public-transport features"}
        except Exception as exc:
            transport_result = {"enabled": False, "error": str(exc)}
            log.warning("Transport accessibility stage failed; demographic analyses remain available: %s", exc)

    checksums = {}
    for label, file_path in {
        "stats_sa_ward_archive": stats_zip,
        "ward_geometry": paths["cache"] / "mdb_2020_electoral_wards.geojson",
    }.items():
        if Path(file_path).exists():
            checksums[label] = sha256_file(file_path)
    for year, meta in wp_surfaces.items():
        checksums[f"worldpop_source_{year}"] = sha256_file(meta["source"])

    manifest = {
        "pipeline_version": "0.2.0",
        "config": cfg,
        "years": years,
        "ward_count_by_year": {str(y): int(len(gdfs[y])) for y in years},
        "source_checksums_sha256": checksums,
        "stats_sa_schema_cache": schema_cache,
        "analysis": analysis,
        "worldpop": wp_surfaces,
        "socioeconomics": socio_result,
        "transport": transport_result,
        "qa_pre": {"status": qa_pre["status"], "json": qa_pre["json"]},
        "qa_post": {"status": qa_post["status"], "json": qa_post["json"]},
        "methodological_notes": [
            "Stats SA Ward-level Small Area Population Estimates are the official ward control totals and are based on 2020 electoral wards.",
            "WorldPop is a within-ward spatial weighting surface, not observed 100m census data.",
            "Comparable WorldPop mode holds the spatial weighting surface constant and calibrates it separately to each year's official ward total.",
            "Year-specific WorldPop mode uses the year-specific WorldPop surface and should not be interpreted as a pure longitudinal spatial-composition comparison.",
            "Zero-weight populated wards use uniform-cell fallback unless configured to fail.",
            "Transport accessibility is Euclidean proximity to mapped OSM public-transport features and is not a travel-time or service-frequency measure.",
        ],
    }
    _write_manifest(paths["outputs"] / "pipeline_manifest.json", manifest)
    return {"community": community, "manifest": manifest, "frames": prepared}
