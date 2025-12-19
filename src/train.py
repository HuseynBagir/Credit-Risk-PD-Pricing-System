from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier

from .config import Paths, SplitConfig, TrainConfig
from .data_prep import make_binary_target, parse_dates, select_application_time_columns
from .features import build_features

def time_split(df: pd.DataFrame, split: SplitConfig):
    train_end = pd.to_datetime(split.train_end)
    valid_end = pd.to_datetime(split.valid_end)

    train = df[df["issue_d"] <= train_end].copy()
    valid = df[(df["issue_d"] > train_end) & (df["issue_d"] <= valid_end)].copy()
    test  = df[df["issue_d"] > valid_end].copy()
    return train, valid, test

def main():
    paths = Paths()
    split = SplitConfig()
    cfg = TrainConfig()

    paths.models_dir.mkdir(parents=True, exist_ok=True)

    # Read (big file) - using low_memory=False helps mixed dtypes
    df = pd.read_csv(paths.data_csv, low_memory=False)

    df = make_binary_target(df)
    df = parse_dates(df)
    df = select_application_time_columns(df)
    df = build_features(df)

    # Drop rows with missing issue_d after parsing
    df = df.dropna(subset=["issue_d"]).reset_index(drop=True)

    train_df, valid_df, test_df = time_split(df, split)

    # Separate target
    y_train = train_df["target_default"].astype(int).values
    y_valid = valid_df["target_default"].astype(int).values
    y_test  = test_df["target_default"].astype(int).values

    # Features: drop label columns
    drop_cols = {"loan_status", "target_default"}
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_valid = valid_df.drop(columns=[c for c in drop_cols if c in valid_df.columns])
    X_test  = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # Don't model raw dates directly
    for X in (X_train, X_valid, X_test):
        for col in ["issue_d", "earliest_cr_line"]:
            if col in X.columns:
                X.drop(columns=[col], inplace=True)

    # Column types
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # Class imbalance handling
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / max(pos, 1))
    
    # --- monotonic constraints ---
    # numeric columns are emitted first by ColumnTransformer
    mono_map = {
        "dti": 1,                 # higher dti -> higher risk
        "fico_score": -1,     # higher fico -> lower risk
        "annual_inc": -1,         # higher income -> lower risk (weakly monotonic)
        "revol_util": 1       
    }

    mono_num = [mono_map.get(c, 0) for c in num_cols]

    # Fit preprocessor first to know output dimensionality
    pre.fit(X_train)

    Xtr = pre.transform(X_train)

    n_total = Xtr.shape[1]
    n_num = len(num_cols)
    n_onehot = n_total - n_num

    # Monotonic constraints: numeric block + zeros for one-hot block
    mono_constraints = tuple(mono_num + [0] * n_onehot)

    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        random_state=cfg.random_state,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=mono_constraints,
    )


    clf = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])


    clf.fit(X_train, y_train)

    # Quick sanity check
    valid_pd = clf.predict_proba(X_valid)[:, 1]
    test_pd = clf.predict_proba(X_test)[:, 1]
    print(f"Valid AUC: {roc_auc_score(y_valid, valid_pd):.4f}")
    print(f"Test  AUC: {roc_auc_score(y_test,  test_pd):.4f}")

    joblib.dump(clf, paths.models_dir / "pd_model_pipeline.joblib")
    joblib.dump({"cat_cols": cat_cols, "num_cols": num_cols}, paths.models_dir / "schema.joblib")

    # Save split samples for evaluation script
    test_pack = {"X_test": X_test, "y_test": y_test}
    valid_pack = {"X_valid": X_valid, "y_valid": y_valid}
    joblib.dump(valid_pack, paths.models_dir / "valid_pack.joblib")
    joblib.dump(test_pack, paths.models_dir / "test_pack.joblib")

    print("Saved: models/pd_model_pipeline.joblib")

if __name__ == "__main__":
    main()
