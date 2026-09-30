"""Optional runtime registration helper for local projects.

This is intentionally not used by the library automatically. Upstream geosnap
should expose the connector natively through ``geosnap.io`` rather than relying
on monkey-patching.
"""
from __future__ import annotations


def register_with_geosnap() -> None:
    """Register ``get_south_africa`` on ``geosnap.io`` in the current Python process."""
    import geosnap.io as gio

    from .connector import get_south_africa

    if not hasattr(gio, "get_south_africa"):
        gio.get_south_africa = get_south_africa  # type: ignore[attr-defined]
