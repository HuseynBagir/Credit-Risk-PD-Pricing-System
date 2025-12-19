import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from .config import Paths

import re

def sanitize_feature_names(names: list[str]) -> list[str]:
    """
    XGBoost disallows feature names containing: [, ], <
    Also keep names unique after sanitization.
    """
    cleaned = []
    seen = {}
    for n in names:
        # replace bad chars with safe tokens
        c = n
        c = c.replace("[", "_lb_").replace("]", "_rb_").replace("<", "_lt_")
        # also remove other annoying chars that can appear
        c = re.sub(r"[^0-9a-zA-Z_:\-\.]", "_", c)

        # ensure uniqueness
        if c in seen:
            seen[c] += 1
            c = f"{c}__{seen[c]}"
        else:
            seen[c] = 0

        cleaned.append(c)
    return cleaned


def get_transformed_feature_names(preprocessor) -> list[str]:
    """
    Works for ColumnTransformer with:
      - ("num", Pipeline(imputer), num_cols)
      - ("cat", Pipeline(imputer, onehot), cat_cols)
    """
    out_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue

        # Pipeline -> take last step
        if hasattr(transformer, "named_steps"):
            # numeric: last step is imputer (no new names)
            if "onehot" in transformer.named_steps:
                ohe = transformer.named_steps["onehot"]
                ohe_names = ohe.get_feature_names_out(cols)
                out_names.extend(ohe_names.tolist())
            else:
                # numeric block: keep original column names
                out_names.extend(list(cols))
        else:
            # In case you ever use direct transformers
            if hasattr(transformer, "get_feature_names_out"):
                out_names.extend(transformer.get_feature_names_out(cols).tolist())
            else:
                out_names.extend(list(cols))

    return out_names

def main():
    paths = Paths()
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    # Use the *pipeline* for explainability (not calibrated wrapper),
    # because calibrated model wraps the estimator.
    clf = joblib.load(paths.models_dir / "pd_model_pipeline.joblib")
    test = joblib.load(paths.models_dir / "test_pack.joblib")

    X = test["X_test"].sample(5000, random_state=42)

    pre = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    # Transform and name features
    X_trans = pre.transform(X)
    feature_names = sanitize_feature_names(get_transformed_feature_names(pre))

    # SHAP prefers dense for plotting (5k x ~100s features is OK)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_trans_df)

    plt.figure()
    shap.summary_plot(shap_values, X_trans_df, show=False)
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "shap_summary_named.png", dpi=150)
    print("Saved:", paths.figures_dir / "shap_summary_named.png")

if __name__ == "__main__":
    main()
