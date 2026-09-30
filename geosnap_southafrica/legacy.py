"""Compatibility bridge for historical geosnap Community APIs."""
from __future__ import annotations

from typing import Any

from .connector import get_south_africa


def community_from_south_africa(*args: Any, **kwargs: Any):
    """Return a historical ``geosnap.Community`` when the installed release exposes it.

    Current geosnap 0.17.0 follows a long-form GeoDataFrame constructor contract,
    so on modern geosnap this function raises a helpful error and directs callers
    to :func:`geosnap_southafrica.get_south_africa`.
    """
    try:
        from geosnap import Community  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The installed geosnap release does not export Community. "
            "geosnap 0.17.0 uses long-form GeoDataFrames as the native data contract. "
            "Use `from geosnap_southafrica import get_south_africa` or install a legacy geosnap release "
            "that provides Community.from_geodataframes."
        ) from exc
    gdf = get_south_africa(*args, **kwargs)
    years = sorted(gdf["year"].unique())
    frames = [gdf[gdf["year"] == year].copy() for year in years]
    return Community.from_geodataframes(frames if len(frames) > 1 else frames[0])
