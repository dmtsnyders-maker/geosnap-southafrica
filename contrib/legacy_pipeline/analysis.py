from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)
RACE_GROUPS = ["pct_black_african", "pct_coloured", "pct_indian_asian", "pct_white"]


def _as_counts(frame: gpd.GeoDataFrame, group: str) -> tuple[gpd.GeoDataFrame, str, str]:
    work = frame[["geometry", group, "population"]].copy()
    group_count = f"{group}_count"
    work[group_count] = pd.to_numeric(work[group], errors="coerce").clip(0, 1) * pd.to_numeric(work["population"], errors="coerce").fillna(0)
    return work, group_count, "population"


def run_multiscalar_segregation(frame: gpd.GeoDataFrame, year: int, distances: Iterable[int], output_dir: Path) -> list[dict]:
    """Run Euclidean multiscalar segregation profiles using PySAL-segregation."""
    rows: list[dict] = []
    try:
        from segregation.dynamics import compute_multiscalar_profile
        from segregation.multigroup import MultiInfoTheory
        from segregation.singlegroup import Dissim
    except Exception as exc:
        log.warning("Multiscalar segregation unavailable: %s", exc)
        return rows

    work = frame.to_crs(frame.estimate_utm_crs() if frame.crs and frame.crs.is_geographic else frame.crs)
    distances = list(distances)
    for group in RACE_GROUPS:
        if group not in work.columns or "population" not in work.columns:
            continue
        data, group_count, total = _as_counts(work, group)
        try:
            profile = compute_multiscalar_profile(data, segregation_index=Dissim, group_pop_var=group_count,
                                                  total_pop_var=total, distances=distances)
            for dist, value in profile.items():
                rows.append({"year": year, "profile": "single_group_dissimilarity", "group": group, "distance_m": float(dist), "value": float(value)})
        except Exception as exc:
            log.warning("Dissimilarity profile failed for %s/%s: %s", year, group, exc)

    group_counts = []
    valid_groups = []
    for group in RACE_GROUPS:
        if group in work.columns:
            vals = pd.to_numeric(work[group], errors="coerce").clip(0, 1) * pd.to_numeric(work["population"], errors="coerce").fillna(0)
            name = group.replace("pct_", "")
            data = work.copy()
            data[name] = vals
            group_counts.append(name)
            valid_groups.append(data[name])
    if len(group_counts) >= 2:
        data = work[["geometry", "population"]].copy()
        for name, vals in zip(group_counts, valid_groups):
            data[name] = vals
        try:
            profile = compute_multiscalar_profile(data, segregation_index=MultiInfoTheory, groups=group_counts, distances=distances)
            for dist, value in profile.items():
                rows.append({"year": year, "profile": "multigroup_information_theory", "group": "all_race_groups", "distance_m": float(dist), "value": float(value)})
        except Exception as exc:
            log.warning("Multigroup information-theory profile failed for %s: %s", year, exc)

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(output_dir / "multiscalar_segregation.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        for key, sub in out.groupby(["profile", "group"]):
            ax.plot(sub["distance_m"], sub["value"], marker="o", label=f"{key[0]} | {key[1]}")
        ax.set_xlabel("Neighbourhood distance (m)")
        ax.set_ylabel("Segregation statistic")
        ax.set_title(f"Multiscalar segregation profile ({year})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"multiscalar_segregation_{year}.png", dpi=180)
        plt.close(fig)
    return rows


def run_global_segregation(frames: dict[int, gpd.GeoDataFrame], output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        from segregation.multigroup import MultiInfoTheory
        from segregation.singlegroup import Dissim, Gini
    except Exception as exc:
        log.warning("Segregation package unavailable: %s", exc)
        return rows
    for year, frame in sorted(frames.items()):
        counts = frame.copy()
        for group in RACE_GROUPS:
            if group in counts.columns:
                counts[group + "_count"] = pd.to_numeric(counts[group], errors="coerce").clip(0, 1) * pd.to_numeric(counts["population"], errors="coerce").fillna(0)
                try:
                    for cls, metric in ((Dissim, "dissimilarity"), (Gini, "gini")):
                        stat = cls(counts, group_pop_var=group + "_count", total_pop_var="population").statistic
                        rows.append({"year": year, "group": group, "metric": metric, "value": float(stat)})
                except Exception as exc:
                    log.warning("Global segregation failed %s/%s: %s", year, group, exc)
        names = []
        for group in RACE_GROUPS:
            col = group + "_count"
            if col in counts.columns:
                names.append(col)
        if len(names) >= 2:
            try:
                stat = MultiInfoTheory(counts, groups=names).statistic
                rows.append({"year": year, "group": "all_race_groups", "metric": "multi_information_theory", "value": float(stat)})
            except Exception as exc:
                log.warning("Global multigroup segregation failed %s: %s", year, exc)
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "segregation.csv", index=False)
    return rows


def _standardized_features(frame: gpd.GeoDataFrame, features: list[str]) -> np.ndarray:
    x = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    return StandardScaler().fit_transform(x)


def _queen_connectivity(frame: gpd.GeoDataFrame):
    """Build a sparse Queen-style contiguity matrix, using libpysal when installed."""
    try:
        from libpysal.weights import Queen
        return Queen.from_dataframe(frame, use_index=False, silence_warnings=True).sparse
    except Exception:
        from scipy.sparse import coo_matrix
        pairs = frame.sindex.query(frame.geometry, predicate="touches")
        rows, cols = pairs[0], pairs[1]
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
        data = np.ones(len(rows), dtype=float)
        return coo_matrix((data, (rows, cols)), shape=(len(frame), len(frame))).tocsr()


def _spatial_ward_clustering(frame: gpd.GeoDataFrame, features: list[str], n_clusters: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    x = _standardized_features(frame, features)
    connectivity = _queen_connectivity(frame)
    kwargs = dict(n_clusters=n_clusters, linkage="ward", connectivity=connectivity)
    try:
        model = AgglomerativeClustering(metric="euclidean", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="euclidean", **kwargs)
    return model.fit_predict(x)


def cluster_stability(frame: gpd.GeoDataFrame, features: list[str], ks: Iterable[int], output_dir: Path, perturbation_pct: float = 0.01, repetitions: int = 5) -> pd.DataFrame:
    """Assess sensitivity of spatial Ward clusters to cluster count and small feature perturbations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    base_x = _standardized_features(frame, features)
    rng = np.random.default_rng(42)
    for k in ks:
        if k < 2 or k >= len(frame):
            continue
        base_labels = _spatial_ward_clustering(frame, features, k)
        sil = silhouette_score(base_x, base_labels) if len(np.unique(base_labels)) > 1 else np.nan
        aris = []
        for rep in range(repetitions):
            perturbed = frame.copy()
            scale = perturbed[features].std(ddof=0).replace(0, 1)
            noise = rng.normal(0, perturbation_pct, size=(len(perturbed), len(features))) * scale.to_numpy()
            perturbed.loc[:, features] = perturbed[features].astype(float).to_numpy() + noise
            labels = _spatial_ward_clustering(perturbed, features, k)
            aris.append(adjusted_rand_score(base_labels, labels))
        rows.append({"k": k, "silhouette": float(sil), "ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)), "repetitions": repetitions})
        # Save the baseline labels for the selected k later.
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(output_dir / "cluster_stability.csv", index=False)
    return out


def run_regionalization(frame: gpd.GeoDataFrame, features: list[str], cfg: dict, output_dir: Path) -> tuple[gpd.GeoDataFrame | None, dict]:
    latest = int(frame["year"].iloc[0])
    k = int(cfg.get("n_regions", 12))
    result = None
    method = "spatial_ward"
    try:
        result = frame.to_crs(frame.estimate_utm_crs()).copy() if frame.crs and frame.crs.is_geographic else frame.copy()
        result["region"] = _spatial_ward_clustering(result, features, k)
        result.to_file(output_dir / f"regionalized_{latest}.gpkg", driver="GPKG")
    except Exception as exc:
        log.warning("Spatial Ward regionalization failed: %s", exc)
        return None, {"created": False, "method": None, "error": str(exc)}

    fig, ax = plt.subplots(figsize=(10, 10))
    result.plot(column="region", categorical=True, legend=True, ax=ax, linewidth=0.15, edgecolor="white")
    ax.set_title(f"South Africa geodemographic regions ({latest})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"regionalized_{latest}.png", dpi=180)
    plt.close(fig)

    ks = cfg.get("sensitivity_k", list(range(max(4, k-4), k+5, 2)))
    stability = cluster_stability(result, features, ks, output_dir,
                                  perturbation_pct=float(cfg.get("perturbation_pct", 0.01)),
                                  repetitions=int(cfg.get("repetitions", 5)))
    return result, {"created": True, "method": method, "n_regions": k, "stability_rows": len(stability)}


def build_neighbourhood_change(frames: dict[int, gpd.GeoDataFrame], output_dir: Path, cluster_col: str = "region") -> dict:
    """Create descriptive 2011→2022 change indicators and cluster transitions."""
    years = sorted(frames)
    if len(years) < 2:
        return {"created": False, "reason": "need_at_least_two_years"}
    y0, y1 = years[0], years[-1]
    a, b = frames[y0].copy(), frames[y1].copy()
    key = "geoid"
    cols = ["population", "pct_black_african", "pct_coloured", "pct_indian_asian", "pct_white", "dependency_ratio", "youth_share", "older_share", "working_age_share", "aging_index"]
    cols = [c for c in cols if c in a.columns and c in b.columns]
    left = a[[key] + cols].copy().set_index(key)
    right = b[[key] + cols].copy().set_index(key)
    change = right[cols].subtract(left[cols], fill_value=np.nan).reset_index()
    change = gpd.GeoDataFrame(change, geometry=b.set_index(key).geometry, crs=b.crs)
    for c in cols:
        arr = pd.to_numeric(change[c], errors="coerce")
        sd = float(arr.std(ddof=0))
        change[f"z_change_{c}"] = (arr - float(arr.mean())) / sd if sd > 0 else 0.0
    zcols = [c for c in change.columns if c.startswith("z_change_")]
    change["absolute_standardized_change"] = change[zcols].abs().mean(axis=1) if zcols else 0.0
    change["change_class"] = pd.cut(change["absolute_standardized_change"], bins=[-np.inf, .5, 1.0, 2.0, np.inf], labels=["stable", "modest_change", "substantial_change", "high_change"])
    change.to_file(output_dir / "neighbourhood_change.gpkg", driver="GPKG")
    change.drop(columns="geometry").to_csv(output_dir / "neighbourhood_change.csv", index=False)

    # Transition of assigned cluster types when they are available.
    transitions = None
    if cluster_col in a.columns and cluster_col in b.columns:
        transitions = b[[key, cluster_col]].rename(columns={cluster_col: f"{cluster_col}_{y1}"}).merge(
            a[[key, cluster_col]].rename(columns={cluster_col: f"{cluster_col}_{y0}"}), on=key, how="inner")
        transitions.to_csv(output_dir / "neighbourhood_transitions.csv", index=False)
    return {"created": True, "years": [y0, y1], "variables": cols, "transitions_created": transitions is not None}


def run_temporal_geosnap(community, frames: dict[int, gpd.GeoDataFrame], features: list[str], output_dir: Path) -> dict:
    """Run geosnap clustering/transition where supported; retain our deterministic transition outputs as fallback."""
    result = {"geosnap_transition_created": False}
    try:
        from geosnap.analyze import cluster, transition
        long_gdf = gpd.GeoDataFrame(pd.concat([frames[y] for y in sorted(frames)], ignore_index=True), crs=next(iter(frames.values())).crs)
        typed, model = cluster(long_gdf, columns=features, method="ward", n_clusters=min(8, max(3, len(features) + 2)), return_model=True)
        # Ensure geoid and year survive.
        typed.to_file(output_dir / "neighborhood_types.gpkg", driver="GPKG")
        if "ward" in typed.columns:
            sm = transition(typed, cluster_col="ward")
            (output_dir / "transition_summary.txt").write_text(str(sm.summary()), encoding="utf-8")
            result["geosnap_transition_created"] = True
    except Exception as exc:
        log.warning("geosnap temporal methods unavailable: %s", exc)
        result["geosnap_transition_error"] = str(exc)
    return result


def profile_region_clusters(clustered: gpd.GeoDataFrame, features: list[str], output_dir: Path) -> Path:
    """Write interpretable cluster profiles without imposing normative labels."""
    label = "region" if "region" in clustered.columns else "ward"
    if label not in clustered.columns:
        raise ValueError("No cluster label column available")
    profile = clustered.groupby(label)[features].mean(numeric_only=True).reset_index()
    for c in features:
        profile[f"rank_{c}"] = profile[c].rank(pct=True)
    path = output_dir / "region_profiles.csv"
    profile.to_csv(path, index=False)
    return path


def _common_temporal_types(frames: dict[int, gpd.GeoDataFrame], features: list[str], output_dir: Path, n_types: int = 8) -> tuple[dict[int, gpd.GeoDataFrame], dict]:
    """Assign common neighbourhood types to every year using a pooled geosnap model.

    A pooled fit is essential for longitudinal transitions because independently fitted
    clusters have arbitrary labels and cannot be compared safely across years.
    """
    years = sorted(frames)
    long = gpd.GeoDataFrame(pd.concat([frames[y] for y in years], ignore_index=True), crs=frames[years[0]].crs)
    result: dict = {"created": False, "method": None}
    try:
        from geosnap.analyze import cluster, transition
        typed, model = cluster(long, columns=features, method="ward", n_clusters=min(n_types, max(3, len(long) // 20)), return_model=True)
        label_col = "ward" if "ward" in typed.columns else next((c for c in typed.columns if c not in long.columns and c != "geometry"), None)
        if label_col is None:
            raise RuntimeError("geosnap cluster returned no cluster label")
        typed = typed.copy()
        typed["temporal_type"] = typed[label_col]
        typed.to_file(output_dir / "neighborhood_types.gpkg", driver="GPKG")
        try:
            sm = transition(typed, cluster_col=label_col)
            (output_dir / "transition_summary.txt").write_text(str(sm.summary()), encoding="utf-8")
        except Exception as exc:
            log.warning("geosnap transition summary unavailable: %s", exc)
        out: dict[int, gpd.GeoDataFrame] = {}
        for year in years:
            subset = typed[typed["year"].astype(int) == year][["geoid", "temporal_type"]].copy()
            out[year] = frames[year].merge(subset, on="geoid", how="left")
        result.update({"created": True, "method": "geosnap_pooled_ward", "label_column": label_col})
        return out, result
    except Exception as exc:
        log.warning("Pooled geosnap temporal clustering failed; using KMeans fallback: %s", exc)
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            x = long[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
            scaler = StandardScaler().fit(x)
            model = KMeans(n_clusters=min(n_types, max(3, len(long) // 20)), n_init=20, random_state=42)
            model.fit(scaler.transform(x))
            long["temporal_type"] = model.labels_
            out = {}
            for year in years:
                subset = long[long["year"].astype(int) == year][["geoid", "temporal_type"]]
                out[year] = frames[year].merge(subset, on="geoid", how="left")
            gpd.GeoDataFrame(long, geometry="geometry", crs=frames[years[0]].crs).to_file(output_dir / "neighborhood_types.gpkg", driver="GPKG")
            result.update({"created": True, "method": "pooled_kmeans_fallback", "error": str(exc)})
            return out, result
        except Exception as exc2:
            return frames, {"created": False, "error": f"{exc}; fallback={exc2}"}


def run_analytics(frames: dict[int, gpd.GeoDataFrame], outputs_dir: str | Path, regionalization_cfg: dict, community=None) -> dict:
    """Run the full analytical layer: QA-ready indicators, segregation, multiscalar profiles, regionalisation, stability, change, and transitions."""
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    seg = run_global_segregation(frames, outputs_dir)
    distances = regionalization_cfg.get("segregation_distances_m", [500, 1000, 2000, 5000, 10000])
    multiscalar_rows = []
    for year, frame in sorted(frames.items()):
        multiscalar_rows.extend(run_multiscalar_segregation(frame, year, distances, outputs_dir))

    latest = max(frames)
    latest_frame = frames[latest].copy()
    features = [c for c in regionalization_cfg.get("features", []) if c in latest_frame.columns]
    regional_meta: dict = {}
    if len(features) >= 2:
        clustered, regional_meta = run_regionalization(latest_frame, features, regionalization_cfg, outputs_dir)
        if clustered is not None:
            profile_region_clusters(clustered, features, outputs_dir)
            region_col = "region" if "region" in clustered.columns else next((c for c in clustered.columns if c not in latest_frame.columns and c != "geometry"), None)
            if region_col:
                frames[latest] = latest_frame.merge(clustered[["geoid", region_col]].rename(columns={region_col: "region"}), on="geoid", how="left")
                latest_frame = frames[latest]

    temporal_frames, temporal_meta = _common_temporal_types(frames, features, outputs_dir, n_types=int(regionalization_cfg.get("temporal_types", 8))) if len(features) >= 2 and len(frames) >= 2 else (frames, {"created": False})
    change_meta = build_neighbourhood_change(temporal_frames, outputs_dir, cluster_col="temporal_type")

    for year, frame in sorted(temporal_frames.items()):
        cols = [c for c in ["geoid", "ward_code", "year", "population", *RACE_GROUPS, "dependency_ratio", "youth_dependency_ratio", "old_age_dependency_ratio", "aging_index", "potential_support_ratio", "region", "temporal_type"] if c in frame.columns]
        frame[cols].to_csv(outputs_dir / f"ward_indicators_{year}.csv", index=False)
        frame.to_file(outputs_dir / f"ward_community_{year}.gpkg", driver="GPKG")

    if community is not None and len(temporal_frames) >= 2 and len(features) >= 2:
        temporal_meta["community_available"] = True
        try:
            # Verify the actual geosnap Community can be queried by year after our pooled labels are built.
            for year in sorted(temporal_frames):
                community.from_year(year)
            temporal_meta["community_year_queries"] = True
        except Exception as exc:
            temporal_meta["community_year_queries"] = False
            temporal_meta["community_query_error"] = str(exc)

    return {
        "latest_year": latest,
        "features": features,
        "segregation_rows": len(seg),
        "multiscalar_rows": len(multiscalar_rows),
        "regionalization": regional_meta,
        "temporal_geosnap": temporal_meta,
        "neighbourhood_change": change_meta,
        "frames": temporal_frames,
    }
