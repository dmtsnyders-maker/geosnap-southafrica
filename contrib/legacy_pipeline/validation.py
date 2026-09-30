from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class QACheck:
    name: str
    status: str
    value: Any = None
    expected: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check(name: str, ok: bool, value: Any = None, expected: Any = None, message: str = "") -> QACheck:
    return QACheck(name=name, status="PASS" if ok else "FAIL", value=value, expected=expected, message=message)


def validate_geometry(gdf: gpd.GeoDataFrame) -> list[QACheck]:
    checks: list[QACheck] = []
    checks.append(_check("geometry_nonempty", not gdf.empty, len(gdf), "> 0"))
    checks.append(_check("geometry_valid", bool(gdf.geometry.is_valid.all()), int((~gdf.geometry.is_valid).sum()), 0,
                         "Invalid geometries must be repaired before spatial analysis."))
    checks.append(_check("unique_ward_codes", not gdf["ward_code"].astype(str).duplicated().any(),
                         int(gdf["ward_code"].duplicated().sum()), 0))
    return checks


def validate_demographics(frame: gpd.GeoDataFrame, year: int, *, tolerance: float = 0.02) -> list[QACheck]:
    checks: list[QACheck] = []
    pop = pd.to_numeric(frame.get("population"), errors="coerce")
    checks.append(_check(f"{year}_population_nonnegative", bool((pop.fillna(0) >= 0).all()),
                         int((pop < 0).sum()), 0))

    races = [c for c in ["pct_black_african", "pct_coloured", "pct_indian_asian", "pct_white", "pct_other"] if c in frame]
    if races:
        shares = frame[races].apply(pd.to_numeric, errors="coerce")
        row_sum = shares.sum(axis=1, skipna=True)
        valid = row_sum.notna() & (pop > 0)
        err = np.abs(row_sum[valid] - 1.0) if valid.any() else pd.Series(dtype=float)
        max_err = float(err.max()) if len(err) else 0.0
        checks.append(_check(f"{year}_race_shares_sum", max_err <= tolerance, max_err, f"<= {tolerance}"))

    age_cols = [c for c in frame.columns if c.startswith("age_") and str(year) in c]
    if age_cols and pop.notna().any():
        age_sum = frame[age_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
        ratio = np.divide(age_sum, pop, out=np.full(len(frame), np.nan), where=pop.to_numpy(dtype=float) > 0)
        valid = np.isfinite(ratio)
        median_ratio = float(np.nanmedian(ratio[valid])) if valid.any() else np.nan
        checks.append(_check(f"{year}_age_to_population", bool(valid.any() and abs(median_ratio - 1) <= 0.05),
                             median_ratio, "~1.0", "Age-band totals should approximately reconcile to population."))
    return checks


def validate_common_geography(frames: dict[int, gpd.GeoDataFrame]) -> list[QACheck]:
    checks: list[QACheck] = []
    years = sorted(frames)
    if not years:
        return checks
    base = set(frames[years[0]]["geoid"].astype(str))
    for year in years[1:]:
        current = set(frames[year]["geoid"].astype(str))
        checks.append(_check(f"common_geography_{years[0]}_{year}", base == current,
                             {"only_first": len(base-current), "only_second": len(current-base)}, "same geoid set"))
    return checks


def validate_worldpop_checks(check_path: str | Path, max_relative_error_pct: float = 0.01) -> list[QACheck]:
    path = Path(check_path)
    if not path.exists():
        return [_check("worldpop_check_file", False, str(path), "exists")]
    df = pd.read_csv(path)
    if df.empty:
        return [_check("worldpop_check_rows", False, 0, ">0")]
    err = pd.to_numeric(df["relative_difference_pct"], errors="coerce").abs()
    max_err = float(err.max()) if err.notna().any() else np.inf
    zero_weight = int((pd.to_numeric(df.get("worldpop_weight_total", 0), errors="coerce") <= 0).sum())
    return [
        _check("worldpop_calibration_error", max_err <= max_relative_error_pct, max_err, f"<= {max_relative_error_pct}%"),
        _check("worldpop_zero_weight_wards", zero_weight == 0, zero_weight, 0,
               "Zero-weight wards should use the configured fallback allocation method."),
    ]


def write_qa_report(checks: list[QACheck], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [c.to_dict() for c in checks]
    csv_path = output_dir / "qa_report.csv"
    json_path = output_dir / "qa_report.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    payload = {
        "status": "PASS" if all(c.status == "PASS" for c in checks) else "FAIL",
        "checks": rows,
        "failures": sum(c.status == "FAIL" for c in checks),
        "passes": sum(c.status == "PASS" for c in checks),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return csv_path, json_path


def run_full_qa(frames: dict[int, gpd.GeoDataFrame], output_dir: str | Path, worldpop_check_paths: list[Path] | None = None) -> dict[str, Any]:
    checks: list[QACheck] = []
    if frames:
        checks.extend(validate_geometry(frames[sorted(frames)[0]]))
        for year, frame in sorted(frames.items()):
            checks.extend(validate_demographics(frame, year))
        checks.extend(validate_common_geography(frames))
    for path in worldpop_check_paths or []:
        checks.extend(validate_worldpop_checks(path))
    csv_path, json_path = write_qa_report(checks, output_dir)
    log.info("QA complete: %d pass, %d fail", sum(c.status == "PASS" for c in checks), sum(c.status == "FAIL" for c in checks))
    return {"checks": checks, "csv": csv_path, "json": json_path, "status": "PASS" if all(c.status == "PASS" for c in checks) else "FAIL"}
