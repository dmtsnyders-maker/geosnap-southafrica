"""Candidate upstream geosnap.io constructor for South Africa.

This file documents the small public-facing addition that should live in
``geosnap/io/constructors.py`` (or be imported there from a dedicated module)
when the connector is proposed upstream.
"""
from __future__ import annotations


def get_south_africa(
    datastore,
    years="all",
    province=None,
    boundary=None,
):
    """Extract South African ward-level neighbourhood data as a long-form GeoDataFrame.

    Parameters
    ----------
    datastore : geosnap.DataStore
        geosnap data store used as the cache root for the South African connector.
    years : {"all", int, list-like}, optional
        The Stats SA ward product is a single 2022 vintage; other years are modelled (2015-2030).
    province : str, optional
        Optional South African province filter.
    boundary : geopandas.GeoDataFrame, optional
        Optional study-area boundary. Wards whose representative points fall
        inside the boundary are retained.

    Returns
    -------
    geopandas.GeoDataFrame
        Long-form geodataframe containing ``geoid`` and ``year`` columns,
        matching the current geosnap constructor convention.
    """
    # During an upstream merge, the implementation from the companion
    # geosnap-southafrica project is moved into geosnap/io/south_africa.py and imported here.
    from .south_africa import get_south_africa as _get_south_africa

    return _get_south_africa(
        datastore,
        years=years,
        province=province,
        boundary=boundary,
    )
