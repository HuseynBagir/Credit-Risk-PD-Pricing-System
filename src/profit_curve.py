import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .pricing import expected_profit, decide_and_price, PricingConfig
from .config import Paths

def main():
    paths = Paths()
    cfg = PricingConfig()

    clf = joblib.load(paths.models_dir / "pd_model_calibrated.joblib")
    test = joblib.load(paths.models_dir / "test_pack.joblib")

    X, y = test["X_test"], test["y_test"]
    pd_hat = clf.predict_proba(X)[:, 1]
    ead = X["funded_amnt"].fillna(X["loan_amnt"]).values

    thresholds = np.linspace(0.01, 0.25, 50)
    profits = []
    approval_rates = []

    for t in thresholds:
        approved = pd_hat <= t
        decision, apr = decide_and_price(pd_hat)
        profit = expected_profit(pd_hat[approved], ead[approved], apr[approved], cfg)
        profits.append(profit.sum())
        approval_rates.append(approved.mean())

    plt.figure()
    plt.plot(approval_rates, profits)
    plt.xlabel("Approval rate")
    plt.ylabel("Total expected profit")
    plt.title("Profit vs Approval Rate")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "profit_curve.png", dpi=150)

    print("Saved profit curve")

if __name__ == "__main__":
    main()
