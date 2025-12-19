import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.inference import prepare_inference_frame, load_expected_cols
from src.logger import log_decision

# ----------------------------
# Paths / imports
# ----------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(ROOT))  # so `import src.*` works

from src.pricing import decide_and_price, expected_profit, PricingConfig

st.set_page_config(page_title="LendingClub PD + Pricing Demo", layout="wide")

CAL_MODEL_PATH = ROOT / "models" / "pd_model_calibrated.joblib"
BASE_MODEL_PATH = ROOT / "models" / "pd_model_pipeline.joblib"

SCHEMA_PATH = ROOT / "models" / "schema.joblib"

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_calibrated_model():
    if not CAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing: {CAL_MODEL_PATH}")
    return joblib.load(CAL_MODEL_PATH)


@st.cache_resource
def load_model_expected_cols():
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing: {SCHEMA_PATH}")
    return load_expected_cols(SCHEMA_PATH)


@st.cache_resource
def load_base_pipeline_for_shap():
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing base pipeline for SHAP: {BASE_MODEL_PATH}\n"
            f"Run training to create it."
        )
    return joblib.load(BASE_MODEL_PATH)

def score_one(model, expected_cols, row: dict) -> dict:

    df = prepare_inference_frame(row, expected_cols)

    pd_hat = float(model.predict_proba(df)[:, 1][0])
    decision, apr = decide_and_price(np.array([pd_hat]))
    decision_str = "approve" if int(decision[0]) == 1 else "decline"
    apr_val = None if np.isnan(apr[0]) else float(apr[0])

    ead = float(df["funded_amnt"].fillna(df["loan_amnt"]).iloc[0])
    cfg = PricingConfig()
    prof = float(expected_profit(np.array([pd_hat]), np.array([ead]), np.array([apr[0]]), cfg)[0])


    # after scoring
    log_decision(
        pd=pd_hat,
        decision=decision_str,
        apr=apr_val,
        channel="streamlit",
        model_version="pd_xgb_calibrated_v1"
    )


    return {
        "pd": pd_hat,
        "decision": decision_str,
        "apr": apr_val,
        "expected_profit_proxy": prof,
        "df_raw": pd.DataFrame([row])
    }

def shap_local_explain(df_raw: pd.DataFrame, expected_cols):
    """
    Explain base model output (pre-calibration).
    Calibration changes probability mapping but not feature contributions.
    """
    try:
        import shap
    except Exception:
        st.error("SHAP is not installed. Run: pip install shap")
        return

    base = load_base_pipeline_for_shap()
    pre = base.named_steps["preprocess"]
    xgb = base.named_steps["model"]

    df = prepare_inference_frame(df_raw.iloc[0].to_dict(), expected_cols)

    Xtr = pre.transform(df)

    # Feature names (best effort)
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(Xtr.shape[1])])

    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(Xtr)[0]  # single row

    # Top-K contributions by absolute magnitude
    k = 18
    idx = np.argsort(np.abs(shap_vals))[::-1][:k]
    top_names = np.array(feature_names)[idx]
    top_vals = shap_vals[idx]

    st.caption(
        "Note: SHAP explains the **base model signal** (pre-calibration). "
        "Calibration improves PD accuracy but doesn’t change feature attributions."
    )

    fig = plt.figure()
    plt.barh(range(len(top_vals)), top_vals)
    plt.yticks(range(len(top_vals)), [names[5:] for names in top_names])
    plt.xlabel("SHAP value (impact on log-odds)")
    plt.title("Top local feature contributions")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)


# ----------------------------
# UI
# ----------------------------
st.title("LendingClub Credit Risk Demo — PD + Pricing (XGBoost)")

cal_model = load_calibrated_model()
expected_cols = load_model_expected_cols()

st.markdown(
    """
This demo predicts **Probability of Default (PD)** and returns a simple **approve/decline + APR tier**.
- PD is from the **calibrated** model (`pd_model_calibrated.joblib`)
- Explainability uses **SHAP** from the base model (`pd_model_pipeline.joblib`)
"""
)

# session state for explain button
if "do_explain" not in st.session_state:
    st.session_state.do_explain = False

# Layout: form on left, results on right
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("Single applicant scoring")

    st.caption("Fill the fields below. Tooltips explain what each feature means.")

    with st.form("applicant_form", clear_on_submit=False):
        st.markdown("### Loan details")
        c1, c2 = st.columns(2)
        with c1:
            loan_amnt = st.number_input(
                "Loan amount (loan_amnt)",
                min_value=0.0, value=10000.0, step=500.0,
                help="Requested principal amount."
            )
            term = st.selectbox(
                "Term (term)",
                [" 36 months", " 60 months"],
                help="Loan duration in months."
            )
        with c2:
            funded_amnt = st.number_input(
                "Funded amount (funded_amnt)",
                min_value=0.0, value=10000.0, step=500.0,
                help="Actual funded amount. If unknown, set equal to loan amount."
            )
            purpose = st.selectbox(
                "Purpose (purpose)",
                ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
                 "small_business", "car", "medical", "moving", "vacation", "other"],
                help="Borrower-stated purpose of the loan."
            )

        st.markdown("### Borrower finances")
        c3, c4, c5 = st.columns(3)
        with c3:
            annual_inc = st.number_input(
                "Annual income (annual_inc)",
                min_value=0.0, value=60000.0, step=1000.0,
                help="Stated annual income."
            )
            dti = st.number_input(
                "Debt-to-income ratio (dti)",
                min_value=0.0, value=15.0, step=0.5,
                help="Total monthly debt payments / monthly income (%). Higher = riskier."
            )
        with c4:
            fico_score = st.number_input(
                "FICO score (fico_score)",
                min_value=300.0, value=680.0, step=1.0,
                help="FICO score range. Higher = safer."
            )
        with c5:
            revol_util = st.number_input(
                "Revolving utilization % (revol_util)",
                min_value=0.0, value=45.0, step=1.0,
                help="Percent of credit line used. Higher = riskier."
            )
            revol_bal = st.number_input(
                "Revolving balance (revol_bal)",
                min_value=0.0, value=8000.0, step=100.0,
                help="Total revolving credit balance."
            )

        st.markdown("### Credit history & other risk signals")
        c6, c7, c8 = st.columns(3)
        with c6:
            delinq_2yrs = st.number_input(
                "Delinquencies in last 2y (delinq_2yrs)",
                min_value=0.0, value=0.0, step=1.0,
                help="Number of delinquencies in past 2 years."
            )
            inq_last_6mths = st.number_input(
                "Inquiries last 6m (inq_last_6mths)",
                min_value=0.0, value=0.0, step=1.0,
                help="Recent credit inquiries (more can indicate risk)."
            )
        with c7:
            open_acc = st.number_input(
                "Open accounts (open_acc)",
                min_value=0.0, value=6.0, step=1.0,
                help="Number of open credit lines."
            )
            total_acc = st.number_input(
                "Total accounts (total_acc)",
                min_value=0.0, value=20.0, step=1.0,
                help="Total credit lines ever."
            )
        with c8:
            pub_rec = st.number_input(
                "Public records (pub_rec)",
                min_value=0.0, value=0.0, step=1.0,
                help="Number of derogatory public records."
            )
            mort_acc = st.number_input(
                "Mortgage accounts (mort_acc)",
                min_value=0.0, value=1.0, step=1.0,
                help="Number of mortgage accounts."
            )

        st.markdown("### Identity / verification")
        c9, c10, c11 = st.columns(3)
        with c9:
            emp_length = st.selectbox(
                "Employment length (emp_length)",
                ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                 "6 years", "7 years", "8 years", "9 years", "10+ years"],
                help="Longer employment can indicate stability."
            )
            home_ownership = st.selectbox(
                "Home ownership (home_ownership)",
                ["RENT", "MORTGAGE", "OWN", "OTHER", "NONE", "ANY"],
                help="Borrower's home ownership status."
            )
        with c10:
            verification_status = st.selectbox(
                "Verification status (verification_status)",
                ["Verified", "Source Verified", "Not Verified"],
                help="Whether income/identity was verified."
            )
            application_type = st.selectbox(
                "Application type (application_type)",
                ["Individual", "Joint App"],
                help="Individual or joint application."
            )
        with c11:
            addr_state = st.text_input(
                "State (addr_state)",
                value="CA",
                help="US state code in the dataset (e.g., CA, NY)."
            )
            sub_grade = st.text_input(
                "Sub-grade (sub_grade)",
                value="B3",
                help="Lender risk band proxy like A1..G5 (used to improve model realism)."
            )

        submitted = st.form_submit_button("Score applicant")

    if submitted:
        st.session_state.do_explain = False  # reset explain state on new score

        row = {
            "loan_amnt": loan_amnt,
            "funded_amnt": funded_amnt,
            "annual_inc": annual_inc,
            "dti": dti,
            "fico_score": fico_score,
            "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_bal": revol_bal,
            "revol_util": revol_util,
            "total_acc": total_acc,
            "mort_acc": mort_acc,
            "term": term,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "verification_status": verification_status,
            "purpose": purpose,
            "addr_state": addr_state,
            "application_type": application_type,
            "sub_grade": sub_grade,
        }

        st.session_state.last_row = row  # keep for explain
        st.session_state.last_scored = score_one(cal_model, expected_cols, row)

with right:
    st.subheader("Results")

    if "last_scored" not in st.session_state:
        st.info("Fill the form and click **Score applicant**.")
    else:
        out = st.session_state.last_scored
        pd_hat = out["pd"]

        r1, r2 = st.columns(2)
        r1.metric("Predicted PD", f"{pd_hat:.2%}")
        r2.metric("Decision", out["decision"])

        r3, r4 = st.columns(2)
        r3.metric("APR tier", "-" if out["apr"] is None else f"{out['apr']:.2%}")
        r4.metric("Expected profit (proxy)", f"{out['expected_profit_proxy']:.2f}")

        st.markdown("---")
        st.markdown("### Explain this applicant (SHAP)")

        # FIX: use session_state flag instead of a nested button that disappears on rerun
        if st.button("Explain this applicant"):
            st.session_state.do_explain = True

        if st.session_state.do_explain:
            try:
                df_raw = pd.DataFrame([st.session_state.last_row])
                shap_local_explain(df_raw, expected_cols)
            except Exception as e:
                st.error(f"Explain failed: {e}")
