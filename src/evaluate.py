import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

import os
MODEL_PATH = os.getenv("MODEL_PATH", "models/pd_model_pipeline.joblib")

clf = joblib.load(MODEL_PATH)

from .config import Paths

def main():
    paths = Paths()
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    clf = joblib.load(paths.models_dir / "pd_model_pipeline.joblib")
    valid = joblib.load(paths.models_dir / "valid_pack.joblib")
    test  = joblib.load(paths.models_dir / "test_pack.joblib")

    Xv, yv = valid["X_valid"], valid["y_valid"]
    Xt, yt = test["X_test"], test["y_test"]

    pd_v = clf.predict_proba(Xv)[:, 1]
    pd_t = clf.predict_proba(Xt)[:, 1]

    print("VALID:")
    print("  AUC   :", roc_auc_score(yv, pd_v))
    print("  PR-AUC:", average_precision_score(yv, pd_v))
    print("  Brier :", brier_score_loss(yv, pd_v))

    print("TEST:")
    print("  AUC   :", roc_auc_score(yt, pd_t))
    print("  PR-AUC:", average_precision_score(yt, pd_t))
    print("  Brier :", brier_score_loss(yt, pd_t))

    # Calibration curve (test)
    frac_pos, mean_pred = calibration_curve(yt, pd_t, n_bins=10, strategy="quantile")

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Observed default rate")
    plt.title("Calibration Curve (Test)")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "calibration_test.png", dpi=150)

    # PD bucket default rates
    bins = np.array([0, 0.02, 0.05, 0.10, 0.20, 1.0])
    bucket = np.digitize(pd_t, bins)  # 1..len(bins)
    bucket_stats = []
    for b in range(1, len(bins)):
        m = bucket == b
        if m.sum() == 0:
            continue
        bucket_stats.append((bins[b-1], bins[b], m.sum(), float(yt[m].mean())))
    print("\nPD Buckets (Test): [low, high) | n | default_rate")
    for lo, hi, n, dr in bucket_stats:
        print(f"  [{lo:.2%}, {hi:.2%}) | {n:6d} | {dr:.2%}")

    print(f"\nSaved figure: {paths.figures_dir / 'calibration_test.png'}")

if __name__ == "__main__":
    main()
