import numpy as np
import pandas as pd
import joblib

from .config import Paths


def _make_bin_edges(expected: np.ndarray, bins: int = 10, strategy: str = "quantile") -> np.ndarray:
    """
    Create robust bin edges for PSI.
    - strategy="quantile": percentile edges from expected distribution (common in PSI),
      with safeguards for repeated edges.
    - strategy="equal": equal-width edges between min/max of expected.
    """
    expected = np.asarray(expected, dtype=float)
    expected = expected[~np.isnan(expected)]

    if expected.size < 2:
        return np.array([-np.inf, np.inf], dtype=float)

    if strategy not in {"quantile", "equal"}:
        raise ValueError("strategy must be 'quantile' or 'equal'")

    if strategy == "equal":
        lo, hi = float(np.nanmin(expected)), float(np.nanmax(expected))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.array([-np.inf, np.inf], dtype=float)
        edges = np.linspace(lo, hi, bins + 1, dtype=float)
    else:
        # Quantile edges (percentiles)
        pct = np.linspace(0, 100, bins + 1)
        edges = np.percentile(expected, pct).astype(float)

    # Remove repeated edges (can happen with discrete or capped variables)
    edges = np.unique(edges)

    # If too few unique edges, fall back to equal-width bins
    if edges.size < 3:
        lo, hi = float(np.nanmin(expected)), float(np.nanmax(expected))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.array([-np.inf, np.inf], dtype=float)
        edges = np.linspace(lo, hi, bins + 1, dtype=float)
        edges = np.unique(edges)

    # Add open-ended bounds so we don't lose values outside expected min/max
    # (and so max value is always included)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi(expected, actual, bins: int = 10, strategy: str = "quantile") -> float:
    """
    Population Stability Index (PSI) between expected (baseline) and actual (new) samples.
    Robust to:
      - values equal to the last edge (we use histogram bins with last bin inclusive by construction)
      - repeated quantile edges (we unique edges + fall back)
      - NaNs and small sample sizes
    """
    eps = 1e-6

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if expected.size == 0 or actual.size == 0:
        return float("nan")

    edges = _make_bin_edges(expected, bins=bins, strategy=strategy)

    # np.histogram uses half-open bins [a, b) except the last which is [a, b]
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)

    e = e_counts / max(expected.size, 1)
    a = a_counts / max(actual.size, 1)

    # PSI sum over bins
    return float(np.sum((e - a) * np.log((e + eps) / (a + eps))))


def main():
    paths = Paths()
    valid_pack = joblib.load(paths.models_dir / "valid_pack.joblib")
    test_pack = joblib.load(paths.models_dir / "test_pack.joblib")

    X_expected = valid_pack["X_valid"]
    X_actual = test_pack["X_test"]

    # Default: compute PSI for all shared numeric columns (wider coverage than 4 features)
    numeric_cols = (
        X_expected.select_dtypes(include=[np.number]).columns
        .intersection(X_actual.select_dtypes(include=[np.number]).columns)
        .tolist()
    )

    if not numeric_cols:
        print("No shared numeric columns found for PSI.")
        return

    print(f"Computing PSI for {len(numeric_cols)} numeric features...")
    for f in numeric_cols:
        exp_vals = X_expected[f].to_numpy()
        act_vals = X_actual[f].to_numpy()
        val = psi(exp_vals, act_vals, bins=10, strategy="quantile")
        if np.isnan(val):
            continue
        print(f"PSI {f}: {val:.4f}")


if __name__ == "__main__":
    main()
