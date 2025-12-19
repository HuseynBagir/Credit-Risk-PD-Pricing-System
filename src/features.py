import numpy as np
import pandas as pd

def emp_length_to_num(x: str):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x == "10+ years":
        return 10
    if x == "< 1 year":
        return 0
    # e.g. "3 years", "1 year"
    digits = "".join([ch for ch in x if ch.isdigit()])
    return float(digits) if digits else np.nan

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['fico_score'] = (df["fico_range_high"] + df['fico_range_low']) / 2

    if "emp_length" in df.columns:
        df["emp_length_num"] = df["emp_length"].map(emp_length_to_num)

    # Credit history length at origination (months)
    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        delta = (df["issue_d"] - df["earliest_cr_line"]).dt.days
        df["credit_hist_months"] = (delta / 30.4375).clip(lower=0)

    # term like " 36 months" -> 36
    if "term" in df.columns:
        df["term_months"] = (
            df["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        )

    return df
