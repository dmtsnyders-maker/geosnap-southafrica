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
        else:
            path = store_south_africa(years=args.years, province=args.province, data_dir=args.data_dir, overwrite=args.overwrite)
            print(path)
    except Exception as exc:  # noqa: BLE001
        logging.exception("South African geosnap connector failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
