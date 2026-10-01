from __future__ import annotations

import argparse
import logging
import sys

from .connector import get_south_africa, store_south_africa


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="South Africa connector for geosnap")
    sub = p.add_subparsers(dest="command", required=False)
    get = sub.add_parser("get", help="build/load the long-form South African ward dataset")
    get.add_argument("--years", nargs="+", type=int, default=[2022])
    get.add_argument("--province", default=None)
    get.add_argument("--data-dir", default=None)
    get.add_argument("--verbose", action="store_true")
    store = sub.add_parser("store", help="materialize the connector output to GeoParquet")
    store.add_argument("--years", nargs="+", type=int, default=[2022])
    store.add_argument("--province", default=None)
    store.add_argument("--data-dir", default=None)
    store.add_argument("--overwrite", action="store_true")
    store.add_argument("--verbose", action="store_true")
    bc = sub.add_parser("build-cache", help="build the small prebuilt WorldPop ward-sum table (to attach to a GitHub Release)")
    bc.add_argument("--data-dir", default=None)
    bc.add_argument("--out", default="worldpop_ward_sums.parquet")
    bc.add_argument("--years", nargs="+", type=int, default=None)
    bc.add_argument("--amenities", action="store_true", help="also build osm_amenity_ward_counts.parquet from a Geofabrik extract (needs: pip install osmium)")
    bc.add_argument("--pbf", default=None, help="use this local .osm.pbf instead of downloading one")
    bc.add_argument("--skip-worldpop", action="store_true")
    bc.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    command = args.command or "get"
    try:
        if command == "get":
            gdf = get_south_africa(years=args.years, province=args.province, data_dir=args.data_dir)
            print(gdf.head().to_string(index=False))
            print(f"\nRows: {len(gdf):,}; years: {sorted(gdf.year.unique().tolist())}")
        elif command == "build-cache":
            import os
            os.environ["GEOSNAP_SA_NO_PREBUILT"] = "1"
            from .annual import WORLDPOP_YEARS
            from .prebuilt import build_worldpop_table
            wards = get_south_africa(years=[2022], data_dir=args.data_dir)
            from .connector import _data_root
            raw = _data_root(None, args.data_dir) / "south_africa" / "raw"
            if not args.skip_worldpop:
                path = build_worldpop_table(wards, raw, args.out, args.years or WORLDPOP_YEARS)
                print(f"{path} ({path.stat().st_size / 1e6:.1f} MB) - attach this to a GitHub Release named data-v1")
            if args.amenities:
                from .osm_pbf import AMENITY_TABLE, build_amenity_table
                path = build_amenity_table(wards, AMENITY_TABLE, pbf_path=args.pbf, raw_dir=raw)
                print(f"{path} ({path.stat().st_size / 1e6:.1f} MB) - attach this to the same GitHub Release (OSM data: ODbL, credit OpenStreetMap contributors)")
        else:
            path = store_south_africa(years=args.years, province=args.province, data_dir=args.data_dir, overwrite=args.overwrite)
            print(path)
    except Exception as exc:  # noqa: BLE001
        logging.exception("South African geosnap connector failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
