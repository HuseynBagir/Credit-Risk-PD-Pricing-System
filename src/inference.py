from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import build_features, emp_length_to_num


def _parse_lc_month(val: Any) -> pd.Timestamp:
    """Parse LendingClub month strings like 'Dec-2017'. Also tolerates ISO dates."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return pd.NaT
    s = str(val).strip()
    if not s:
        return pd.NaT

    dt = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce")
    return dt


def _emp_num_to_str(n: Any) -> str | None:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return None
    try:
        n_int = int(round(float(n)))
    except Exception:
        return None

    if n_int <= 0:
        return "< 1 year"
    if n_int >= 10:
        return "10+ years"
    if n_int == 1:
        return "1 year"
    return f"{n_int} years"


def load_expected_cols(schema_path: Path) -> list[str]:
    """Load the exact feature list (num + cat) used at train time."""
    schema = joblib.load(schema_path)
    num_cols = list(schema.get("num_cols", []))
    cat_cols = list(schema.get("cat_cols", []))
    return num_cols + cat_cols


def prepare_inference_frame(payload: dict, expected_cols: list[str]) -> pd.DataFrame:
    """Make inference inputs match the training-time feature schema."""
    df = pd.DataFrame([payload]).copy()

    # Ensure required raw columns exist (build_features expects these)
    for col in ("fico_range_low", "fico_range_high"):
        if col not in df.columns:
            df[col] = np.nan

    # 1) If caller provides engineered-only fields, backfill raw fields
    # FICO
    if pd.isna(df.loc[0, "fico_range_low"]) or pd.isna(df.loc[0, "fico_range_high"]):
        if "fico_score" in df.columns and not pd.isna(df.loc[0, "fico_score"]):
            df.loc[0, "fico_range_low"] = float(df.loc[0, "fico_score"])
            df.loc[0, "fico_range_high"] = float(df.loc[0, "fico_score"])

    # Term
    if ("term" not in df.columns or pd.isna(df.loc[0, "term"])) and (
        "term_months" in df.columns and not pd.isna(df.loc[0, "term_months"])
    ):
        df.loc[0, "term"] = f"{int(float(df.loc[0, 'term_months']))} months"

    # Employment
    if ("emp_length" not in df.columns or pd.isna(df.loc[0, "emp_length"])) and (
        "emp_length_num" in df.columns and not pd.isna(df.loc[0, "emp_length_num"])
    ):
        df.loc[0, "emp_length"] = _emp_num_to_str(df.loc[0, "emp_length_num"])

    # 2) Parse dates if present (needed for credit_hist_months)
    if "issue_d" in df.columns:
        df["issue_d"] = df["issue_d"].map(_parse_lc_month)
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = df["earliest_cr_line"].map(_parse_lc_month)

    # 3) Feature engineering (same function as training)
    df = build_features(df)

    # Safety: if user passed a weird emp_length string, try parse again
    if "emp_length" in df.columns and "emp_length_num" in df.columns:
        if pd.isna(df.loc[0, "emp_length_num"]) and not pd.isna(df.loc[0, "emp_length"]):
            df.loc[0, "emp_length_num"] = emp_length_to_num(df.loc[0, "emp_length"])

    # 4) Drop raw dates (training drops these before preprocessing)
    for col in ("issue_d", "earliest_cr_line"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 5) Enforce schema: add missing cols as NaN and order exactly like training
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[expected_cols]

    return df
