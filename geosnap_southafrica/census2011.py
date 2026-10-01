"""Census 2011 small-area data (DataFirst "Community Profiles 2011", Stata files) -> ward / sub-place indicators.

The DataFirst download is one Stata file per topic, one row per Census 2011 *small area* (SAL, ~84,900 in
South Africa). Every row also carries its sub-place (suburb), main place, municipality, district and province
names, so the data can be summarised per official suburb with no geometry at all. To get ward rows on the
2020 ward geography, each small area is assigned to the ward containing its representative point (small
areas are tiny, so this is far more accurate than area-weighting) and the counts are summed.

Only counts are added up; shares, medians and ratios are derived AFTER aggregation.

Recognised topic files (matched by their variable prefix, so file names do not matter): population group
(``pgrp``), age in single years (``age``), annual household income (``inchh``), dwelling type (``dwltyp``),
tenure (``tensta``), water source (``wtrsrs``), household size (``hhsize``), employment of household head
(``emsthhh``), place of birth (``brthreg``), year moved (``yrmved``), internet / cell phone / motor car
(``inter``, ``cell``, ``mtrcar``), enumeration-area and geography type (``eatyp``, ``geotyp``); and, when you
download them, education (``edu``-style), toilet, energy and refuse variables (see ``TOPICS``).
Unrecognised files are skipped with a log message.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GEO_COLS = ["sal_code", "sp_code", "sp_name", "mp_code", "mp_name", "mn_mdb_c", "mn_code", "mn_name",
            "dc_mdb_c", "dc_code", "dc_name", "pr_code", "pr_name"]
_GEO_PREFIXES = {"sal", "sp", "mp", "mn", "dc", "pr"}
AGE_BANDS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49),
             (50, 54), (55, 59), (60, 64), (65, 69), (70, 74), (75, 79), (80, 84)]
RACES = ["black_african", "coloured", "indian_asian", "white", "other"]
LOW_INCOME_BANDS = 4      # bands 1-4 = no income up to R19,600 a year (about R1,600 a month)
HIGH_INCOME_FROM = 8      # band 8 starts at R153,801 a year (about R12,800 a month)


def _clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def _prefix(columns) -> str | None:
    counts: dict[str, int] = {}
    for c in columns:
        m = re.fullmatch(r"([a-z]+)_(\d+)", str(c))
        if m and m.group(1) not in _GEO_PREFIXES:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return max(counts, key=counts.get) if counts else None


def _read_topic(path: Path) -> tuple[str, pd.DataFrame, dict[str, str]] | None:
    reader = pd.read_stata(path, iterator=True, convert_categoricals=False)
    df = reader.read()
    labels = {k: _clean(v) for k, v in reader.variable_labels().items()}
    prefix = _prefix(df.columns)
    if prefix is None or "sal_code" not in df.columns:
        return None
    df = df[df["sal_code"].notna()].copy()
    df["sal_code"] = df["sal_code"].astype("int64")
    return prefix, df, labels


def _cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(rf"{prefix}_\d+", str(c))]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df[list(cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _by_label(labels: dict[str, str], cols, *needles: str) -> list[str]:
    """Columns whose label starts with (or contains) one of ``needles`` (case-insensitive)."""
    out = []
    for c in cols:
        text = labels.get(c, "").lower()
        if any(n in text for n in needles):
            out.append(c)
    return out


# --------------------------------------------------------------------------- per-topic handlers
# Each returns a DataFrame of ADDITIVE columns indexed like ``df``.

def _h_pgrp(df, labels):
    cols = _cols(df, "pgrp")
    out = pd.DataFrame(index=df.index)
    for race, col in zip(RACES, cols):
        out[f"n_{race}"] = _num(df, [col])[col]
    out["population"] = out[[f"n_{r}" for r in RACES]].sum(axis=1)
    return out


def _h_age(df, labels):
    cols = _cols(df, "age")
    ages = {}
    for c in cols:
        m = re.match(r"(\d+)\b", labels.get(c, ""))
        ages[c] = int(m.group(1)) if m else int(c.rsplit("_", 1)[1]) - 1
    vals = _num(df, cols)
    out = pd.DataFrame(index=df.index)
    for lo, hi in AGE_BANDS:
        out[f"age_{lo}_{hi}_total"] = vals[[c for c in cols if lo <= ages[c] <= hi]].sum(axis=1)
    out["age_85_plus_total"] = vals[[c for c in cols if ages[c] >= 85]].sum(axis=1)
    return out


def _income_bounds(label: str) -> tuple[float, float] | None:
    text = label.split(" - ")[0].replace(" ", "").replace("R", "")
    if text.lower().startswith("noincome"):
        return 0.0, 0.0
    m = re.match(r"(\d+)-(\d+)$", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"(\d+)ormore", text.lower())
    if m:
        return float(m.group(1)), float("inf")
    return None


def _h_inchh(df, labels):
    cols = _cols(df, "inchh")
    vals = _num(df, cols)
    out = pd.DataFrame(index=df.index)
    for i, c in enumerate(cols, 1):
        out[f"hh_income_band_{i:02d}"] = vals[c]
    out["households_income"] = vals.sum(axis=1)
    return out


def _h_dwltyp(df, labels):
    cols = _cols(df, "dwltyp")
    v = _num(df, cols)
    n = lambda *ix: v[[cols[i - 1] for i in ix if i <= len(cols)]].sum(axis=1)  # noqa: E731
    return pd.DataFrame({"dw_formal": n(1, 3, 4, 5, 6, 7, 10), "dw_traditional": n(2), "dw_informal": n(8, 9),
                         "dw_other": n(11, 12)}, index=df.index)


def _h_tensta(df, labels):
    cols = _cols(df, "tensta")
    v = _num(df, cols)
    lab = lambda *k: v[_by_label(labels, cols, *k)].sum(axis=1)  # noqa: E731
    return pd.DataFrame({"ten_owned": lab("owned"), "ten_rented": lab("rented"),
                         "ten_rent_free": lab("rent-free", "rent free"), "ten_other": lab("other")}, index=df.index)


def _h_wtrsrs(df, labels):
    cols = _cols(df, "wtrsrs")
    v = _num(df, cols)
    lab = lambda *k: v[_by_label(labels, cols, *k)].sum(axis=1)  # noqa: E731
    known = v[[c for c in cols if "not applicable" not in labels.get(c, "").lower()]].sum(axis=1)
    return pd.DataFrame({"water_scheme": lab("regional/local water scheme"), "water_known": known}, index=df.index)


def _h_hhsize(df, labels):
    cols = _cols(df, "hhsize")
    v = _num(df, cols)
    sizes = np.arange(1, len(cols) + 1, dtype=float)
    if len(sizes):
        sizes[-1] = max(sizes[-1], 11.0)      # "10+" treated as 11
    return pd.DataFrame({"households": v.sum(axis=1), "hh_persons_est": (v * sizes).sum(axis=1)}, index=df.index)


def _h_emsthhh(df, labels):
    cols = _cols(df, "emsthhh")
    v = _num(df, cols)
    lab = lambda *k: v[_by_label(labels, cols, *k)].sum(axis=1)  # noqa: E731
    return pd.DataFrame({"head_employed": lab("employed - "), "head_unemployed": lab("unemployed - "),
                         "head_discouraged": lab("discouraged"), "head_other_nea": lab("other not economically")},
                        index=df.index)


def _h_brthreg(df, labels):
    cols = _cols(df, "brthreg")
    v = _num(df, cols)
    local = v[_by_label(labels, cols, "born in south africa")].sum(axis=1)
    unspec = v[_by_label(labels, cols, "unspecified")].sum(axis=1)
    return pd.DataFrame({"born_sa": local, "born_abroad": v.sum(axis=1) - local - unspec}, index=df.index)


def _h_yrmved(df, labels):
    cols = _cols(df, "yrmved")
    v = _num(df, cols)
    years = {c: (int(m.group(1)) if (m := re.match(r"(\d{4})\b", labels.get(c, ""))) else None) for c in cols}
    dated = [c for c in cols if years[c] is not None]
    recent = [c for c in dated if years[c] >= 2006]
    return pd.DataFrame({"moved_recent": v[recent].sum(axis=1), "moved_known": v[dated].sum(axis=1)}, index=df.index)


def _yes_no(prefix, name):
    def handler(df, labels):
        cols = _cols(df, prefix)
        v = _num(df, cols)
        yes = v[_by_label(labels, cols, "yes -")].sum(axis=1)
        return pd.DataFrame({f"{name}_yes": yes, f"{name}_known": v.sum(axis=1)}, index=df.index)
    return handler


def _h_inter(df, labels):
    cols = _cols(df, "inter")
    v = _num(df, cols)
    none = v[_by_label(labels, cols, "no access")].sum(axis=1)
    return pd.DataFrame({"internet_none": none, "internet_known": v.sum(axis=1)}, index=df.index)


def _share_handler(prefix, mapping):
    """Generic handler: sum columns whose label contains a needle into the named count columns."""
    def handler(df, labels):
        cols = _cols(df, prefix)
        v = _num(df, cols)
        out = pd.DataFrame(index=df.index)
        for name, needles in mapping.items():
            out[name] = v[_by_label(labels, cols, *needles)].sum(axis=1)
        out[f"{prefix}_known"] = v.sum(axis=1)
        return out
    return handler


def _h_comm(df, labels):
    """Difficulty communicating (a disability proxy): share with some, a lot of difficulty or unable."""
    cols = _cols(df, "comm")
    v = _num(df, cols)
    some = v[_by_label(labels, cols, "some difficulty", "a lot of difficulty", "cannot do")].sum(axis=1)
    none = v[_by_label(labels, cols, "no difficulty")].sum(axis=1)
    return pd.DataFrame({"comm_difficulty": some, "comm_known": some + none}, index=df.index)


def _h_relhh(df, labels):
    """Household composition from relationship to the household head."""
    cols = _cols(df, "relhh")
    v = _num(df, cols)
    lab = lambda *k: v[_by_label(labels, cols, *k)].sum(axis=1)  # noqa: E731
    known = v[[c for c in cols if not any(w in labels.get(c, "").lower() for w in ("unspecified", "not applicable"))]].sum(axis=1)
    return pd.DataFrame({"rel_head": lab("head/acting head"), "rel_partner": lab("husband/wife/partner"),
                         "rel_child": lab("son/daughter -", "adopted", "stepchild"),
                         "rel_grandchild": lab("grand/great-grandchild"), "rel_nonrelated": lab("non-related"),
                         "rel_known": known}, index=df.index)


TOPICS = {
    "pgrp": _h_pgrp, "age": _h_age, "inchh": _h_inchh, "dwltyp": _h_dwltyp, "tensta": _h_tensta,
    "wtrsrs": _h_wtrsrs, "hhsize": _h_hhsize, "emsthhh": _h_emsthhh, "brthreg": _h_brthreg,
    "yrmved": _h_yrmved, "cell": _yes_no("cell", "cellphone"), "mtrcar": _yes_no("mtrcar", "car"),
    "inter": _h_inter, "comm": _h_comm, "relhh": _h_relhh,
    "eatyp": _share_handler("eatyp", {"ea_formal": ("formal residential",), "ea_informal": ("informal residential",),
                                      "ea_traditional": ("traditional residential",), "ea_farm": ("farms",)}),
    "geotyp": _share_handler("geotyp", {"geo_urban": ("urban",), "geo_tribal": ("tribal",), "geo_farm": ("farm",)}),
}
# Extra topics (education, toilet, energy, refuse) are recognised by their labels once you download them.
_LABEL_TOPICS = {
    "education": (("highest education", "education level", "educational"), "edu"),
    "toilet": (("toilet",), "toilet"),
    "lighting": (("energy", "lighting"), "energy"),
    "refuse": (("refuse",), "refuse"),
}


def _h_education(df, labels, prefix):
    cols = _cols(df, prefix)
    v = _num(df, cols)
    lab = lambda *k: v[_by_label(labels, cols, *k)].sum(axis=1)  # noqa: E731
    known = v[[c for c in cols if not any(w in labels.get(c, "").lower() for w in ("unspecified", "not applicable"))]].sum(axis=1)
    return pd.DataFrame({
        "edu_none": lab("no schooling"),
        "edu_some_primary": lab("some primary"),
        "edu_some_secondary": lab("some secondary"),
        "edu_matric": lab("grade 12", "matric"),
        "edu_higher": lab("higher", "bachelor", "diploma", "certificate", "degree", "post"),
        "edu_known": known,
    }, index=df.index)


def _h_toilet(df, labels, prefix):
    cols = _cols(df, prefix)
    v = _num(df, cols)
    lab = lambda *k: v[_by_label(labels, cols, *k)].sum(axis=1)  # noqa: E731
    known = v[[c for c in cols if "not applicable" not in labels.get(c, "").lower()]].sum(axis=1)
    return pd.DataFrame({"toilet_flush": lab("flush toilet"), "toilet_none": lab("none"), "toilet_known": known},
                        index=df.index)


def _h_energy(df, labels, prefix):
    cols = _cols(df, prefix)
    v = _num(df, cols)
    lab = lambda *k: v[_by_label(labels, cols, *k)].sum(axis=1)  # noqa: E731
    known = v[[c for c in cols if not any(w in labels.get(c, "").lower() for w in ("unspecified", "not applicable"))]].sum(axis=1)
    return pd.DataFrame({"energy_electricity": lab("electricity"), "energy_known": known}, index=df.index)


def _h_refuse(df, labels, prefix):
    cols = _cols(df, prefix)
    v = _num(df, cols)
    lab = lambda *k: v[_by_label(labels, cols, *k)].sum(axis=1)  # noqa: E731
    known = v[[c for c in cols if not any(w in labels.get(c, "").lower() for w in ("unspecified", "not applicable"))]].sum(axis=1)
    return pd.DataFrame({"refuse_municipal": lab("removed by local authority", "local authority"), "refuse_known": known},
                        index=df.index)


_LABEL_HANDLERS = {"education": _h_education, "toilet": _h_toilet, "lighting": _h_energy, "refuse": _h_refuse}


def _label_topic(prefix, labels, cols) -> str | None:
    text = " ".join(labels.get(c, "").lower() for c in cols[:3])
    title = text.split(" - ")[-1] if " - " in text else text
    for topic, (needles, _) in _LABEL_TOPICS.items():
        if any(n in text for n in needles) and topic != "lighting":
            return topic
    if "energy" in text or "lighting" in text:
        return "lighting"
    return None


# --------------------------------------------------------------------------- loading
def read_sal_counts(folder: str | Path) -> pd.DataFrame:
    """Read every recognised ``.dta`` in ``folder`` into one table of ADDITIVE counts per small area.

    Returns a DataFrame with the geography columns (``sal_code``, ``sp_name``, ``mn_name``, ``pr_name`` ...)
    followed by count columns. Files may be named anything; topics are recognised from their variables.
    """
    folder = Path(folder)
    files = sorted(folder.rglob("*.dta"))
    if not files:
        raise FileNotFoundError(f"No .dta files found in {folder}. Download the DataFirst Census 2011 "
                                "Community Profiles (Stata) files into that folder.")
    geo: pd.DataFrame | None = None
    parts: list[pd.DataFrame] = []
    seen: set[str] = set()
    for path in files:
        topic = _read_topic(path)
        if topic is None:
            log.info("Skipping %s (no small-area topic recognised)", path.name)
            continue
        prefix, df, labels = topic
        cols = _cols(df, prefix)
        handler = TOPICS.get(prefix)
        if handler is not None:
            counts = handler(df, labels)
        else:
            kind = _label_topic(prefix, labels, cols)
            if kind is None:
                log.info("Skipping %s (variables %s_* not recognised)", path.name, prefix)
                continue
            counts = _LABEL_HANDLERS[kind](df, labels, prefix)
            prefix = kind
        if prefix in seen:
            log.info("Skipping %s (topic %s already loaded)", path.name, prefix)
            continue
        seen.add(prefix)
        counts.insert(0, "sal_code", df["sal_code"].values)
        counts = counts.drop_duplicates("sal_code").set_index("sal_code")
        parts.append(counts)
        if geo is None:
            geo = df[[c for c in GEO_COLS if c in df.columns]].drop_duplicates("sal_code").set_index("sal_code")
            for c in ("sp_code", "mp_code", "mn_code", "dc_code", "pr_code"):
                if c in geo:
                    geo[c] = pd.to_numeric(geo[c], errors="coerce").astype("Int64")
    if geo is None:
        raise ValueError(f"None of the .dta files in {folder} looked like Census 2011 small-area tables.")
    out = pd.concat([geo] + [p.reindex(geo.index) for p in parts], axis=1)
    out = out.loc[:, ~out.columns.duplicated()].reset_index()
    log.info("Census 2011: %d small areas, topics: %s", len(out), ", ".join(sorted(seen)))
    return out


def count_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in GEO_COLS and pd.api.types.is_numeric_dtype(table[c])]


# --------------------------------------------------------------------------- indicators
def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.where(b > 0)).astype(float)


def _median_from_bands(counts: pd.DataFrame, bounds: list[tuple[float, float]]) -> pd.Series:
    """Median from grouped counts, linear inside the band that holds the middle observation."""
    vals = counts.to_numpy(dtype=float)
    total = vals.sum(axis=1)
    half = total / 2.0
    cum = np.cumsum(vals, axis=1)
    out = np.full(len(vals), np.nan)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    for i in range(len(vals)):
        if total[i] <= 0:
            continue
        j = int(np.searchsorted(cum[i], half[i]))
        j = min(j, len(bounds) - 1)
        below = cum[i][j - 1] if j > 0 else 0.0
        width_hi = hi[j] if np.isfinite(hi[j]) else lo[j] * 2
        frac = (half[i] - below) / vals[i][j] if vals[i][j] > 0 else 0.5
        out[i] = lo[j] + frac * (width_hi - lo[j])
    return pd.Series(out, index=counts.index)


# Annual household income band bounds in rand (Census 2011 bands 1..12; 13 is "unspecified" and excluded).
INCOME_BOUNDS = [(0, 0), (1, 4800), (4801, 9600), (9601, 19600), (19601, 38200), (38201, 76400), (76401, 153800),
                 (153801, 307600), (307601, 614400), (614401, 1228800), (1228801, 2457600), (2457601, float("inf"))]


def derive_indicators(c: pd.DataFrame) -> pd.DataFrame:
    """Turn aggregated counts into geosnap-style indicators (shares, medians, ratios). Index is preserved."""
    o = pd.DataFrame(index=c.index)
    has = lambda *cols: all(x in c.columns for x in cols)  # noqa: E731
    if "population" in c:
        o["population"] = c["population"]
    for r in RACES:
        if f"n_{r}" in c and "population" in c:
            o[f"pct_{r}"] = _ratio(c[f"n_{r}"], c[[f"n_{x}" for x in RACES]].sum(axis=1))
    band_cols = [x for x in c.columns if re.fullmatch(r"age_\d+_(\d+|85_plus)_total|age_85_plus_total", x)]
    for x in band_cols:
        o[x] = c[x]
    if band_cols:
        pop = c[band_cols].sum(axis=1)
        s = lambda lo, hi: c[[f"age_{a}_{b}_total" for a, b in AGE_BANDS if lo <= a <= hi]].sum(axis=1)  # noqa: E731
        kids, work = s(0, 15), s(20, 60)
        older = s(65, 80) + c["age_85_plus_total"]
        o["youth_share"] = _ratio(kids, pop)
        o["working_age_share"] = _ratio(work, pop)
        o["older_share"] = _ratio(older, pop)
        o["dependency_ratio"] = _ratio(kids + older, work) * 100
        o["youth_dependency_ratio"] = _ratio(kids, work) * 100
        o["old_age_dependency_ratio"] = _ratio(older, work) * 100
        o["aging_index"] = _ratio(older, kids) * 100
        o["potential_support_ratio"] = _ratio(work, older)
        mids = np.array([(a + b + 1) / 2 for a, b in AGE_BANDS] + [90.0])
        vals = c[[f"age_{a}_{b}_total" for a, b in AGE_BANDS] + ["age_85_plus_total"]].to_numpy(float)
        o["median_age"] = _median_from_bands(pd.DataFrame(vals, index=c.index),
                                             [(a, b + 1) for a, b in AGE_BANDS] + [(85, 95)])
        del mids
    if "households" in c:
        o["households"] = c["households"]
        o["avg_household_size"] = _ratio(c["hh_persons_est"], c["households"])
    inc = [f"hh_income_band_{i:02d}" for i in range(1, 13)]
    if all(x in c for x in inc):
        tot = c[inc].sum(axis=1)
        o["median_hh_income"] = _median_from_bands(c[inc], INCOME_BOUNDS)
        o["pct_hh_low_income"] = _ratio(c[inc[:LOW_INCOME_BANDS]].sum(axis=1), tot)
        o["pct_hh_no_income"] = _ratio(c[inc[0]], tot)
        o["pct_hh_high_income"] = _ratio(c[inc[HIGH_INCOME_FROM - 1:]].sum(axis=1), tot)
    if has("dw_formal", "dw_traditional", "dw_informal", "dw_other"):
        tot = c[["dw_formal", "dw_traditional", "dw_informal", "dw_other"]].sum(axis=1)
        o["pct_formal_dwelling"] = _ratio(c["dw_formal"], tot)
        o["pct_informal_dwelling"] = _ratio(c["dw_informal"], tot)
        o["pct_traditional_dwelling"] = _ratio(c["dw_traditional"], tot)
    if has("ten_owned", "ten_rented", "ten_rent_free", "ten_other"):
        tot = c[["ten_owned", "ten_rented", "ten_rent_free", "ten_other"]].sum(axis=1)
        o["pct_owner_occupied"] = _ratio(c["ten_owned"], tot)
        o["pct_renter"] = _ratio(c["ten_rented"], tot)
    if has("water_scheme", "water_known"):
        o["pct_water_scheme"] = _ratio(c["water_scheme"], c["water_known"])
    if has("head_employed", "head_unemployed", "head_discouraged", "head_other_nea"):
        lf = c["head_employed"] + c["head_unemployed"]
        o["head_unemployment_rate"] = _ratio(c["head_unemployed"], lf)
        o["head_unemployment_rate_expanded"] = _ratio(c["head_unemployed"] + c["head_discouraged"], lf + c["head_discouraged"])
    if has("born_sa", "born_abroad"):
        o["pct_foreign_born"] = _ratio(c["born_abroad"], c["born_sa"] + c["born_abroad"])
    if has("moved_recent", "moved_known"):
        o["pct_moved_recently"] = _ratio(c["moved_recent"], c["moved_known"])
    if has("cellphone_yes", "cellphone_known"):
        o["pct_cellphone"] = _ratio(c["cellphone_yes"], c["cellphone_known"])
    if has("car_yes", "car_known"):
        o["pct_car"] = _ratio(c["car_yes"], c["car_known"])
    if has("internet_none", "internet_known"):
        o["pct_internet_access"] = 1 - _ratio(c["internet_none"], c["internet_known"])
    if has("comm_difficulty", "comm_known"):
        o["pct_comm_difficulty"] = _ratio(c["comm_difficulty"], c["comm_known"])
    if has("rel_head", "rel_partner", "rel_child", "rel_known"):
        o["pct_rel_child_of_head"] = _ratio(c["rel_child"], c["rel_known"])
        o["pct_rel_grandchild"] = _ratio(c["rel_grandchild"], c["rel_known"])
        o["pct_rel_nonrelated"] = _ratio(c["rel_nonrelated"], c["rel_known"])
        o["pct_rel_extended"] = _ratio(c["rel_known"] - c["rel_head"] - c["rel_partner"] - c["rel_child"], c["rel_known"])
    if has("ea_formal", "ea_informal", "eatyp_known"):
        o["pct_ea_informal"] = _ratio(c["ea_informal"], c["eatyp_known"])
    if has("geo_urban", "geotyp_known"):
        o["pct_geo_urban"] = _ratio(c["geo_urban"], c["geotyp_known"])
        o["pct_geo_tribal"] = _ratio(c["geo_tribal"], c["geotyp_known"])
    if has("edu_none", "edu_known"):
        o["pct_edu_no_schooling"] = _ratio(c["edu_none"], c["edu_known"])
        o["pct_edu_matric"] = _ratio(c["edu_matric"], c["edu_known"])
        o["pct_edu_higher"] = _ratio(c["edu_higher"], c["edu_known"])
    if has("toilet_flush", "toilet_known"):
        o["pct_flush_toilet"] = _ratio(c["toilet_flush"], c["toilet_known"])
    if has("energy_electricity", "energy_known"):
        o["pct_electricity"] = _ratio(c["energy_electricity"], c["energy_known"])
    if has("refuse_municipal", "refuse_known"):
        o["pct_refuse_removal"] = _ratio(c["refuse_municipal"], c["refuse_known"])
    return o


# --------------------------------------------------------------------------- aggregation
def _find_code_column(geom: gpd.GeoDataFrame, codes: pd.Series) -> str:
    target = set(codes.astype("int64"))
    best, best_n = None, 0
    for col in geom.columns:
        if col == geom.geometry.name:
            continue
        s = pd.to_numeric(geom[col], errors="coerce").dropna().astype("int64")
        n = s.isin(target).sum()
        if n > best_n:
            best, best_n = col, n
    if best is None or best_n < 0.5 * min(len(geom), len(target)):
        raise ValueError("Could not match the small-area geometry to Census 2011 small-area codes "
                         f"(columns: {[c for c in geom.columns if c != geom.geometry.name]}). Pass sal_code_field=.")
    return best


def load_sal_geometry(source, codes: pd.Series, *, sal_code_field: str | None = None) -> gpd.GeoDataFrame:
    """Small-area polygons with an integer ``sal_code`` column (source: file path or GeoDataFrame)."""
    g = source.copy() if isinstance(source, gpd.GeoDataFrame) else gpd.read_file(source)
    col = sal_code_field or _find_code_column(g, codes)
    g = g.rename(columns={col: "sal_code"})
    g["sal_code"] = pd.to_numeric(g["sal_code"], errors="coerce")
    g = g[g["sal_code"].notna() & g.geometry.notna() & ~g.geometry.is_empty][["sal_code", g.geometry.name]].copy()
    g["sal_code"] = g["sal_code"].astype("int64")
    g = g.set_geometry(g.geometry.name)
    return g.to_crs(4326) if g.crs else g.set_crs(4326)


def sal_to_wards(counts: pd.DataFrame, sal_geometry, wards: gpd.GeoDataFrame, *, id_col: str = "geoid",
                 sal_code_field: str | None = None, max_snap_m: float = 2000.0) -> pd.DataFrame:
    """Sum small-area counts into ``wards`` (rows: ward ``id_col`` -> count columns).

    Each small area goes to the ward that contains its representative point; the few whose point falls
    in no ward (coast, gaps) go to the nearest ward within ``max_snap_m`` metres.
    """
    sal = load_sal_geometry(sal_geometry, counts["sal_code"], sal_code_field=sal_code_field)
    minx, miny, maxx, maxy = wards.to_crs(4326).total_bounds
    sal = sal.cx[minx - 0.05:maxx + 0.05, miny - 0.05:maxy + 0.05]
    counts = counts[counts["sal_code"].isin(sal["sal_code"])]
    pts = sal.copy()
    pts["geometry"] = sal.geometry.representative_point()
    w = wards[[id_col, wards.geometry.name]].to_crs(4326)
    joined = gpd.sjoin(pts, w, predicate="within", how="left").drop_duplicates("sal_code")
    miss = joined[id_col].isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[pts["sal_code"].isin(joined.loc[miss, "sal_code"])].to_crs(6933),
                                 w.to_crs(6933), how="left", max_distance=max_snap_m).drop_duplicates("sal_code")
        joined = joined.set_index("sal_code")
        joined.loc[near["sal_code"].values, id_col] = near[id_col].values
        joined = joined.reset_index()
    lookup = joined.set_index("sal_code")[id_col]
    cols = count_columns(counts)
    tab = counts.set_index("sal_code")[cols]
    tab = tab.assign(**{id_col: lookup.reindex(tab.index).values}).dropna(subset=[id_col])
    matched = float(tab["population"].sum() / counts["population"].sum()) if "population" in counts else np.nan
    log.info("Census 2011: %d/%d small areas placed in wards (%.1f%% of population)",
             len(tab), len(counts), matched * 100 if matched == matched else float("nan"))
    return tab.groupby(id_col).sum()


def wards_2011(folder: str | Path, sal_geometry, wards: gpd.GeoDataFrame, *, id_col: str = "geoid",
               sal_code_field: str | None = None) -> pd.DataFrame:
    """Census 2011 indicators per 2020 ward (index = ward id). Wards with no small area get NaN."""
    counts = read_sal_counts(folder)
    by_ward = sal_to_wards(counts, sal_geometry, wards, id_col=id_col, sal_code_field=sal_code_field)
    ind = derive_indicators(by_ward)
    return ind.reindex(wards[id_col].astype(str).values).rename_axis(id_col)


PLACE_LEVELS = {
    "sub_place": ("sp_code", "sp_name", "mp_name", "mn_name", "pr_name"),
    "main_place": ("mp_code", "mp_name", "mn_name", "pr_name"),
    "municipality": ("mn_code", "mn_name", "dc_name", "pr_name"),
    "district": ("dc_code", "dc_name", "pr_name"),
    "province": ("pr_code", "pr_name"),
}


def places_2011(folder: str | Path, level: str = "sub_place", sal_geometry=None, *, sal_code_field: str | None = None):
    """Census 2011 indicators per official place: ``level`` is one of ``PLACE_LEVELS`` (sub_place = suburb,
    main_place = town/township/village, municipality, district, province). Needs no geometry; with
    ``sal_geometry`` the small areas are dissolved into a polygon per place and a GeoDataFrame is returned.
    """
    if level not in PLACE_LEVELS:
        raise ValueError(f"level must be one of {list(PLACE_LEVELS)}")
    counts = read_sal_counts(folder)
    keys = [k for k in PLACE_LEVELS[level] if k in counts.columns]
    grouped = counts.groupby(keys, dropna=False)[count_columns(counts)].sum()
    ind = derive_indicators(grouped).reset_index()
    if sal_geometry is None:
        return ind
    code = keys[0]
    g = load_sal_geometry(sal_geometry, counts["sal_code"], sal_code_field=sal_code_field)
    g = g.merge(counts[["sal_code", code]], on="sal_code")
    poly = g.dissolve(by=code, as_index=False)[[code, g.geometry.name]]
    return gpd.GeoDataFrame(ind.merge(poly, on=code, how="left"), geometry=g.geometry.name, crs=g.crs)


def subplaces_2011(folder: str | Path, sal_geometry=None, *, sal_code_field: str | None = None):
    """Census 2011 indicators per official sub-place (suburb). See :func:`places_2011`."""
    return places_2011(folder, "sub_place", sal_geometry, sal_code_field=sal_code_field)
