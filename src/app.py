from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

from .pricing import decide_and_price
from .logger import log_decision
from .inference import prepare_inference_frame, load_expected_cols

import joblib

from .config import Paths

# Resolve model paths consistently
paths = Paths()
clf = joblib.load(paths.models_dir / "pd_model_calibrated.joblib")
expected_cols = load_expected_cols(paths.models_dir / "schema.joblib")

app = FastAPI(title="LendingClub PD + Pricing (XGBoost)")
# -----------------------------
# API schema: accept BOTH raw & engineered
# -----------------------------
class Applicant(BaseModel):
    loan_amnt: float
    funded_amnt: float | None = None

    annual_inc: float
    dti: float

    fico_score: float | None = None
    fico_range_low: float | None = None
    fico_range_high: float | None = None

    delinq_2yrs: float | None = None
    inq_last_6mths: float | None = None
    open_acc: float | None = None
    pub_rec: float | None = None
    revol_bal: float | None = None
    revol_util: float | None = None
    total_acc: float | None = None
    mort_acc: float | None = None

    term: str | None = None
    term_months: float | None = None

    emp_length: str | None = None
    emp_length_num: float | None = None

    issue_d: str | None = None
    earliest_cr_line: str | None = None
    credit_hist_months: float | None = None

    home_ownership: str
    verification_status: str
    purpose: str
    addr_state: str
    application_type: str
    sub_grade: str


@app.post("/predict")
def predict(a: Applicant):
    payload = a.model_dump()
    X = prepare_inference_frame(payload, expected_cols)

    pd_hat = float(clf.predict_proba(X)[:, 1][0])

    decision, apr = decide_and_price(np.array([pd_hat]))
    decision = "approve" if int(decision[0]) == 1 else "decline"
    apr_val = None if np.isnan(apr[0]) else float(apr[0])

    log_decision(
        pd=pd_hat,
        decision=decision,
        apr=apr_val,
        channel="FastAPI",
        model_version="pd_xgb_calibrated_v1"
    )

    return {"pd": pd_hat, "decision": decision, "apr": apr_val}
