from __future__ import annotations

import io
import logging
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WARD_CODE_ALIASES = ["wardcode", "ward_code", "ward no", "ward number", "ward", "wardnr", "wardno", "ward_num", "wardid", "ward_id", "wardnum"]
RACES = ("black_african", "coloured", "indian_asian", "white", "other")


class SchemaError(RuntimeError):
    pass


def normalize_text(value: object) -> str:
    s = "" if value is None else str(value)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" ".join(str(x) for x in col if str(x) != "nan") for col in out.columns]
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[ ,%]", "", regex=True).replace({"..": np.nan, "-": np.nan, "": np.nan}), errors="coerce")


DEFAULT_PRODUCT_YEAR = 2022


def _member(zf: zipfile.ZipFile, *keywords: str) -> str:
    names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls")) and "__macosx" not in n.lower()]
    for n in names:
        base = normalize_text(Path(n).name)
        if all(k in base for k in keywords):
            return n
    raise SchemaError(f"No workbook matching {keywords} in archive. Members: {names}")


def _ward_sheet(raw: bytes, engine: str) -> pd.DataFrame:
    """Return the sheet keyed by a ward code column (not the municipal summary sheets)."""
    xls = pd.ExcelFile(io.BytesIO(raw), engine=engine)
    for sheet in xls.sheet_names:
        df = _flatten_columns(xls.parse(sheet, header=0))
        if df.shape[1] > 1 and normalize_text(df.columns[0]) in {normalize_text(a) for a in WARD_CODE_ALIASES}:
            return df
    raise SchemaError(f"No ward-level sheet found. Sheets: {xls.sheet_names}")


def _read_ward_table(zf: zipfile.ZipFile, *keywords: str) -> pd.DataFrame:
    name = _member(zf, *keywords)
    engine = "openpyxl" if name.lower().endswith(".xlsx") else "xlrd"
    df = _ward_sheet(zf.read(name), engine)
    df = df.rename(columns={df.columns[0]: "ward_code"})
    df["ward_code"] = df["ward_code"].map(lambda v: re.sub(r"\.0$", "", str(v).strip()))
    df = df[df["ward_code"].str.fullmatch(r"\d+")].copy()
    return df


def _age_band(col: object) -> str | None:
    raw = str(col).strip().lower()
    m = re.fullmatch(r"(\d{1,2})\s*[-\u2013]\s*(\d{1,2})", raw)
    if m:
        return f"{int(m.group(1))}_{int(m.group(2))}"
    m = re.fullmatch(r"(\d{1,3})\s*(?:\+|plus|and over|and older|or older|and above)", raw)
    if m:
        return f"{int(m.group(1))}_plus"
    return None


def load_official_ward_estimates(zip_path: str | Path, *, schema_cache: str | Path | None = None,
                                 year: int = DEFAULT_PRODUCT_YEAR) -> pd.DataFrame:
    """Load the Stats SA ward product (age, population group, sex workbooks) for a single vintage.

    The archive holds three workbooks, each with a ward-level sheet keyed by an
    8-digit ward code. No year is stated inside the sheets, so the vintage is
    supplied by the caller (default 2022). ``schema_cache`` is accepted for
    backwards compatibility and ignored.
    """
    with zipfile.ZipFile(Path(zip_path)) as zf:
        age = _read_ward_table(zf, "age", "group")
        race = _read_ward_table(zf, "population", "group")
        sex = _read_ward_table(zf, "sex")

    out = pd.DataFrame({"ward_code": age["ward_code"]})
    age_cols = {}
    for c in age.columns[1:]:
        band = _age_band(c)
        if band:
            out[f"age_{band}_total_{year}"] = _numeric(age[c]).values
            age_cols[band] = c
    if len(age_cols) < 10:
        raise SchemaError(f"Age workbook: only {len(age_cols)} age-band columns recognised: {list(age.columns)}")

    race_map = {"black_african": "black african", "coloured": "coloured", "indian_asian": "indian",
                "white": "white", "other": "other"}
    race = race.set_index("ward_code")
    counts = {}
    for key, token in race_map.items():
        col = next((c for c in race.columns if token in normalize_text(c)), None)
        if col is None:
            raise SchemaError(f"Population group workbook has no '{token}' column: {list(race.columns)}")
        counts[key] = _numeric(race[col])
    counts = pd.DataFrame(counts)
    shares = counts.div(counts.sum(axis=1).where(counts.sum(axis=1) > 0), axis=0)
    for key in race_map:
        out[f"pct_{key}_{year}"] = out["ward_code"].map(shares[key])
    total_col = next((c for c in race.columns if normalize_text(c) == "total"), None)
    pop_from_race = out["ward_code"].map(_numeric(race[total_col])) if total_col else None

    sex = sex.set_index("ward_code")
    for label in ("male", "female"):
        col = next((c for c in sex.columns if normalize_text(c) == label), None)
        if col is not None:
            out[f"sex_{label}_{year}"] = out["ward_code"].map(_numeric(sex[col]))
    if f"sex_male_{year}" in out and f"sex_female_{year}" in out:
        out[f"population_{year}"] = out[f"sex_male_{year}"] + out[f"sex_female_{year}"]
    elif pop_from_race is not None:
        out[f"population_{year}"] = pop_from_race
    else:
        out[f"population_{year}"] = out[[c for c in out.columns if c.startswith("age_")]].sum(axis=1)

    out["year"] = int(year)
    out = out.drop_duplicates(subset=["ward_code"], keep="first").reset_index(drop=True)
    if out[f"population_{year}"].dropna().empty:
        raise SchemaError("No population totals could be read from the Stats SA workbooks.")
    return out
