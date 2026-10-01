"""Live self-test: run the connector against the real data and check it.

    python -m geosnap_southafrica.selftest                 # national + Western Cape + Cape Town + suburbs
    python -m geosnap_southafrica.selftest --annual        # also the yearly series (downloads ~100 MB per year)
    python -m geosnap_southafrica.selftest --suburbs suburbs.shp --suburb-field SUBURB
    python -m geosnap_southafrica.selftest --skip-suburbs --no-store

Reference totals: Census 2022 South Africa 62,027,503; Stats SA ward product municipal sheets:
Western Cape 7,433,020, City of Cape Town 4,772,846.
"""
from __future__ import annotations

import argparse
import sys
import traceback

REF_ZA = 62_027_503
REF_WC = 7_433_020
REF_CPT = 4_772_846


def _run(argv=None) -> int:
    ap = argparse.ArgumentParser(description="geosnap-southafrica live self-test")
    ap.add_argument("--suburbs", default=None, help="suburb polygon file/URL (default: OpenStreetMap points)")
    ap.add_argument("--suburb-field", default=None, help="suburb name column, if auto-detect picks wrong")
    ap.add_argument("--skip-suburbs", action="store_true")
    ap.add_argument("--skip-amenities", action="store_true", help="skip only the (slow, Overpass-dependent) amenities check")
    ap.add_argument("--skip-extras", action="store_true", help="skip the OpenStreetMap amenities and wealth-index checks")
    ap.add_argument("--annual", action="store_true", help="also test the modelled yearly series (2015, 2022, 2025)")
    ap.add_argument("--census2011", default=None, help="folder with the DataFirst Census 2011 .dta files")
    ap.add_argument("--sal", default=None, help="Census 2011 small-area polygons (file) to place them in 2020 wards")
    ap.add_argument("--no-store", action="store_true", help="do not build a geosnap DataStore (skips DuckDB)")
    args = ap.parse_args(argv)

    import logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("  ... %(message)s"))
    pkg_log = logging.getLogger("geosnap_southafrica")
    pkg_log.addHandler(handler)
    pkg_log.setLevel(logging.INFO)

    results: list[tuple[str, str, str]] = []

    def check(name, fn, skip_on_error=False):
        try:
            msg = fn()
            results.append(("PASS", name, msg or ""))
            return True
        except AssertionError as e:
            results.append(("FAIL", name, str(e)))
        except Exception as e:  # noqa: BLE001
            results.append(("SKIP" if skip_on_error else "FAIL", name, f"{type(e).__name__}: {str(e)[:300]}"))
            if not skip_on_error:
                traceback.print_exc()
        return False

    from geosnap_southafrica import get_south_africa, interpolate_to_suburbs

    store = None
    if not args.no_store:
        def _store():
            nonlocal store
            from geosnap import DataStore
            store = DataStore()
            return "geosnap DataStore created"
        check("geosnap DataStore", _store)

    state: dict = {}

    def national():
        sa = get_south_africa(store, years=[2022])
        state["sa"] = sa
        assert 4300 <= len(sa) <= 4600, f"{len(sa)} wards (expected ~4,468)"
        assert sa["geoid"].is_unique and sa["geoid"].str.fullmatch(r"\d{8}").all(), "geoid not unique 8-digit"
        assert sa["province_name"].nunique() == 9, f"{sa['province_name'].nunique()} provinces"
        assert set(sa["year"]) == {2022} and (sa["population_basis"] == "stats_sa_2022").all()
        assert sa.crs is not None and sa.crs.to_epsg() == 4326 and sa.geometry.is_valid.all()
        tot = sa["population"].sum()
        assert abs(tot / REF_ZA - 1) < 0.02, f"population {tot:,.0f} vs Census 2022 {REF_ZA:,}"
        return f"{len(sa):,} wards, 9 provinces, population {tot:,.0f}"

    check("whole of South Africa", national)
    sa = state.get("sa")

    if sa is not None:
        def provinces():
            by = sa.groupby("province_name")["population"].sum().sort_values(ascending=False)
            assert by.min() > 0
            wc = by.get("Western Cape", 0)
            assert abs(wc / REF_WC - 1) < 0.01, f"Western Cape {wc:,.0f} vs Stats SA {REF_WC:,}"
            return "; ".join(f"{k} {v / 1e6:.1f}M" for k, v in by.head(3).items()) + " ..."
        check("province totals", provinces)

        def shares():
            race = sa[["pct_black_african", "pct_coloured", "pct_indian_asian", "pct_white", "pct_other"]]
            assert (((race.sum(axis=1)) - 1).abs() < 1e-6).all(), "race shares do not sum to 1"
            assert sa["youth_share"].between(0, 1).all() and sa["working_age_share"].between(0, 1).all()
            return "race shares sum to 1; age shares in [0,1]"
        check("demographic shares", shares)

        def age_total():
            bands = [c for c in sa.columns if c.startswith("age_") and c.endswith("_total")]
            ratio = (sa[bands].sum(axis=1) / sa["population"]).median()
            assert abs(ratio - 1) < 0.02, f"age bands sum to {ratio:.3f} x population (median)"
            return f"age bands / population = {ratio:.4f}"
        check("age bands add up to population", age_total)

        def cpt():
            c = get_south_africa(store, years=[2022], municipality="City of Cape Town")
            state["cpt"] = c
            tot = c["population"].sum()
            assert abs(tot / REF_CPT - 1) < 0.01, f"population {tot:,.0f} vs Stats SA {REF_CPT:,}"
            assert (c["ward_name"].str.startswith("City of Cape Town - Ward")).all()
            return f"{len(c)} wards, population {tot:,.0f}"
        check("City of Cape Town filter", cpt)

        def ward_filter():
            w = get_south_africa(store, years=[2022], municipality="CPT", ward=[1, 2, 3])
            assert len(w) == 3, f"expected 3 wards, got {len(w)}"
            return ", ".join(w["ward_name"])
        check("ward filter", ward_filter)

        def other_city():
            j = get_south_africa(store, years=[2022], municipality="City of Johannesburg")
            assert len(j) > 100 and j["province_name"].eq("Gauteng").all(), "Johannesburg not found in Gauteng"
            return f"{len(j)} wards in Gauteng, population {j['population'].sum():,.0f}"
        check("another city (Johannesburg)", other_city)

    cpt_df = state.get("cpt")

    if cpt_df is not None and args.annual:
        def annual():
            a = get_south_africa(store, years=[2015, 2022, 2025], municipality="CPT")
            tot = a.groupby("year")["population"].sum()
            assert abs(tot[2022] / REF_CPT - 1) < 0.01, "2022 no longer matches the census"
            assert tot[2015] < tot[2022] < tot[2025], f"not increasing: {tot.round(0).to_dict()}"
            assert 0.75 < tot[2015] / tot[2022] < 1.0, f"2015/2022 = {tot[2015] / tot[2022]:.3f}"
            basis = a.groupby("year")["population_basis"].first().to_dict()
            assert basis[2022] == "stats_sa_2022" and basis[2015] == "worldpop_anchored"
            return "Cape Town " + ", ".join(f"{y}: {v:,.0f}" for y, v in tot.items())
        check("yearly series (modelled, anchored to 2022)", annual)

    if cpt_df is not None and not args.skip_extras:
        def amenities():
            a = get_south_africa(store, years=[2022], municipality="CPT", extras=["amenities"])
            assert len(a) == len(cpt_df), "extras changed the number of wards"
            schools, health = int(a["n_schools"].sum()), int(a["n_health"].sum())
            assert schools > 50, f"only {schools} schools found in Cape Town (OSM fetch incomplete?)"
            assert a["n_schools"].notna().all()
            return f"Cape Town: {schools} schools, {health} health facilities, {int(a['n_bus_stops'].sum())} bus stops"
        if not args.skip_amenities:
            check("extra: OpenStreetMap amenities", amenities, skip_on_error=True)

        def rwi():
            r = get_south_africa(store, years=[2022], municipality="CPT", extras=["rwi"])
            share = r["rwi_mean"].notna().mean()
            assert share > 0.9, f"only {share:.0%} of wards have a wealth index"
            return f"{share:.0%} of wards; ward mean RWI ranges {r['rwi_mean'].min():.2f} to {r['rwi_mean'].max():.2f}"
        check("extra: Relative Wealth Index", rwi, skip_on_error=True)

    if cpt_df is not None and not args.skip_extras:
        def landcover():
            r = get_south_africa(store, years=[2022], municipality="CPT", extras=["landcover"])
            tree, built = r["lc_share_tree"].mean(), r["lc_share_built"].mean()
            assert r["lc_share_built"].notna().mean() > 0.9 and 0 < built < 1, f"built share {built}"
            return f"Cape Town wards: mean tree share {tree:.0%}, built-up {built:.0%}"
        check("extra: land cover (ESA WorldCover)", landcover, skip_on_error=True)

        def vegetation():
            r = get_south_africa(store, years=[2022], municipality="CPT", extras=["vegetation"])
            assert r["veg_dominant"].notna().mean() > 0.8, "few wards matched a vegetation type"
            return f"{r['veg_dominant'].nunique()} dominant vegetation types, e.g. {r['veg_dominant'].mode().iloc[0]}"
        check("extra: SANBI vegetation map", vegetation, skip_on_error=True)

    if cpt_df is not None and not args.skip_suburbs:
        def suburbs():
            src = args.suburbs if args.suburbs else True
            s = get_south_africa(store, years=[2022], municipality="CPT", suburbs=src, suburb_name_field=args.suburb_field)
            state["sub"] = s
            covered = (s["suburb_count"] > 0).mean()
            assert covered > 0.8, f"only {covered:.0%} of wards matched any suburb"
            ex = s.iloc[0]
            return f"{covered:.0%} of wards have suburbs; e.g. {ex['ward_name']}: {ex['suburbs'][:80]}"
        if not check("suburbs per ward", suburbs):
            results.append(("HINT", "suburbs", "try --suburbs <polygon file or URL> [--suburb-field <column>]"))

        def interp():
            from geosnap_southafrica.suburbs import load_suburbs
            src = args.suburbs if args.suburbs else "cape_town"
            if src == "cape_town":
                from pathlib import Path

                import geosnap_southafrica as pkg
                from geosnap_southafrica.config import load_config
                src = load_config(Path(pkg.__file__).with_name("default_config.yaml"))["suburb_service_url"]
            sub = load_suburbs(src, name_field=args.suburb_field)
            res = interpolate_to_suburbs(cpt_df, sub, extensive=["population"])
            tot = res["population"].sum()
            assert abs(tot / cpt_df["population"].sum() - 1) < 0.02, f"suburb total {tot:,.0f} != ward total"
            return f"{len(res)} suburbs, population {tot:,.0f} (within 2% of ward total)"
        check("ward -> suburb population interpolation", interp, skip_on_error=not args.suburbs)

    if args.census2011:
        def subplaces():
            from geosnap_southafrica.census2011 import read_sal_counts, subplaces_2011
            c = read_sal_counts(args.census2011)
            assert abs(c["population"].sum() / 51_770_560 - 1) < 0.01, f"population {c['population'].sum():,.0f} vs Census 2011 51,770,560"
            sp = subplaces_2011(args.census2011)
            return f"{len(c):,} small areas, {len(sp):,} sub-places, population {c['population'].sum():,.0f}"
        check("Census 2011 sub-places (no geometry needed)", subplaces)

        if args.sal and cpt_df is not None:
            def wards11():
                w = get_south_africa(store, years=[2011, 2022], municipality="CPT", census2011=args.census2011, sal_geometry=args.sal)
                tot = w.groupby("year")["population"].sum()
                assert abs(tot[2011] / 3_740_026 - 1) < 0.02, f"Cape Town 2011 population {tot[2011]:,.0f} vs 3,740,026"
                cols = ["median_hh_income", "pct_formal_dwelling", "pct_owner_occupied", "head_unemployment_rate"]
                med = w[w.year == 2011][cols].median().round(3).to_dict()
                return f"Cape Town 2011 {tot[2011]:,.0f} people in {int((w.year == 2011).sum())} wards; median ward {med}"
            check("Census 2011 re-based onto 2020 wards", wards11)

    width = max(len(n) for _, n, _ in results)
    print()
    for status, name, msg in results:
        print(f"[{status:4}] {name:<{width}}  {msg}")
    failed = [r for r in results if r[0] == "FAIL"]
    print(f"\n{sum(r[0] == 'PASS' for r in results)} passed, {len(failed)} failed, {sum(r[0] == 'SKIP' for r in results)} skipped")
    return 1 if failed else 0


def main() -> None:
    sys.exit(_run())


if __name__ == "__main__":
    main()
