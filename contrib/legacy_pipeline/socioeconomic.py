from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import bounds as window_bounds

from geosnap_southafrica.worldpop import _candidate_indices, _prepare_shapes, _window_iter


def load_municipality_socioeconomics(path: str | Path) -> pd.DataFrame:
    """Read a municipality socioeconomic table with a deterministic identifier."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if not any(c in df.columns for c in ("municipality_code", "municipality_name")):
        raise ValueError("Socioeconomic table must contain municipality_code or municipality_name.")
    return df


def join_socioeconomics_to_wards(wards: gpd.GeoDataFrame, socio: pd.DataFrame) -> gpd.GeoDataFrame:
    """Join municipality attributes to wards by municipality code."""
    if "municipality_code" not in wards.columns or "municipality_code" not in socio.columns:
        raise ValueError("Deterministic socioeconomic joins require municipality_code in both datasets.")
    out = wards.copy()
    s = socio.copy()
    out["municipality_code"] = out["municipality_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    s["municipality_code"] = s["municipality_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return out.merge(s, on="municipality_code", how="left", suffixes=("", "_socio"))


def _is_rate_column(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in ("_share", "_pct", "_percent", "rate", "ratio"))


def disaggregate_socioeconomics(calibrated_population_raster: str | Path, wards: gpd.GeoDataFrame,
                                municipality_column: str, output_dir: str | Path) -> dict[str, Path]:
    """Create 100m socioeconomic surfaces from municipality-level inputs.

    Rate/share fields are spatially constant within their source municipality.
    Additive count fields are distributed in proportion to calibrated population.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = {"geometry", municipality_column, "ward_code", "population", "year"}
    numeric_cols = [c for c in wards.columns if c not in protected and pd.api.types.is_numeric_dtype(wards[c])]
    outputs: dict[str, Path] = {}
    if not numeric_cols:
        return outputs

    with rasterio.open(calibrated_population_raster) as src:
        wards_r = _prepare_shapes(wards, src.crs)
        municipality = wards_r[municipality_column].astype(str).to_numpy()
        pop = pd.to_numeric(wards_r.get("population", pd.Series(1.0, index=wards_r.index)), errors="coerce").fillna(0).to_numpy(float)
        profile = src.profile.copy()
        for col in numeric_cols:
            out_path = output_dir / f"socio_{col}_100m.tif"
            values_by_muni = wards_r.groupby(municipality_column)[col].mean(numeric_only=True).to_dict()
            count_controls = wards_r.groupby(municipality_column)[col].sum(min_count=1).to_dict()
            pop_controls = pd.Series(pop, index=wards_r.index).groupby(wards_r[municipality_column]).sum().to_dict()
            with rasterio.open(out_path, "w", **profile) as dst:
                for window in _window_iter(src, block_rows=512):
                    bbox = window_bounds(window, src.transform)
                    idx = _candidate_indices(wards_r, bbox)
                    if len(idx) == 0:
                        dst.write(np.zeros((int(window.height), int(window.width)), dtype="float32"), 1, window=window)
                        continue
                    ids = rasterize([(wards_r.iloc[i].geometry, int(i)) for i in idx],
                                    out_shape=(int(window.height), int(window.width)),
                                    transform=src.window_transform(window), fill=-1, dtype="int32")
                    if _is_rate_column(col):
                        ward_values = pd.to_numeric(wards_r[col], errors="coerce").fillna(0).to_numpy(float)
                        arr = np.where(ids >= 0, ward_values[np.clip(ids, 0, len(ward_values)-1)], 0)
                    else:
                        cell_pop = src.read(1, window=window).astype(float)
                        arr = np.zeros_like(cell_pop, dtype=float)
                        for m in np.unique(municipality[idx]):
                            local_ward_idx = idx[municipality[idx] == m]
                            msum = cell_pop[np.isin(ids, local_ward_idx)].sum()
                            control = pd.to_numeric(pd.Series(count_controls.get(m, np.nan)), errors="coerce").iloc[0]
                            if np.isfinite(control) and msum > 0:
                                mask = np.isin(ids, local_ward_idx) & (cell_pop > 0)
                                arr[mask] = cell_pop[mask] / msum * float(control)
                    dst.write(arr.astype("float32"), 1, window=window)
            outputs[col] = out_path
    return outputs
