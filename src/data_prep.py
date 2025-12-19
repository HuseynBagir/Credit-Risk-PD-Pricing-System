import pandas as pd

DEFAULT_STATUSES = {"Charged Off", "Default"}
GOOD_STATUSES = {"Fully Paid"}
DROP_STATUSES = {
    "Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"
}

def make_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df["loan_status"].isin(DROP_STATUSES)]
    df = df[df["loan_status"].isin(DEFAULT_STATUSES | GOOD_STATUSES)]
    df["target_default"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    return df

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # LendingClub date formats often like "Dec-2017"
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    return df

def select_application_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Curated set of mostly application-time features.
    Notes:
    - Exclude LC assigned pricing/underwriting shortcuts like grade/sub_grade/int_rate
      (otherwise your model can become a proxy of their internal model).
    """
    keep = [
        # identifiers/time
        "issue_d",

        # numeric
        "loan_amnt", "funded_amnt",  # you can later choose EAD from funded_amnt
        "annual_inc", "dti",
        "fico_range_high", "fico_range_low",
        "delinq_2yrs", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_bal", "revol_util", "total_acc",
        "mort_acc",

        # categorical
        "term", "emp_length", "home_ownership", "verification_status",
        "purpose", "addr_state", "application_type", "sub_grade",

        # dates for derived features
        "earliest_cr_line",

        # label source
        "loan_status",
        "target_default",
    ]
    existing = [c for c in keep if c in df.columns]
    return df[existing].copy()
