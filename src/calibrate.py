import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

from .config import Paths

def main():
    paths = Paths()
    clf = joblib.load(paths.models_dir / "pd_model_pipeline.joblib")
    valid = joblib.load(paths.models_dir / "valid_pack.joblib")
    test  = joblib.load(paths.models_dir / "test_pack.joblib")

    Xv, yv = valid["X_valid"], valid["y_valid"]
    Xt, yt = test["X_test"], test["y_test"]

    # Calibrate on validation (prefit model)
    cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(Xv, yv)

    pd_v = cal.predict_proba(Xv)[:, 1]
    pd_t = cal.predict_proba(Xt)[:, 1]

    print("CALIBRATED VALID:")
    print("  AUC  :", roc_auc_score(yv, pd_v))
    print("  Brier:", brier_score_loss(yv, pd_v))

    print("CALIBRATED TEST:")
    print("  AUC  :", roc_auc_score(yt, pd_t))
    print("  Brier:", brier_score_loss(yt, pd_t))

    joblib.dump(cal, paths.models_dir / "pd_model_calibrated.joblib")
    print("Saved: models/pd_model_calibrated.joblib")

if __name__ == "__main__":
    main()
