"""Polynomial/interaction expansion of squared lags — test for curvature.

Tests whether interactions between squared lags capture non-linear
relationships that plain Ridge misses. Uses degree-2 polynomial
expansion of the 12 squared lag features:
- 12 original features
- 78 pairwise interactions (12 choose 2)
- 12 squared terms (already squared, so these are 4th powers)
Total: 102 features

Same verified walk-forward + held-out pipeline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

HORIZON = 12
STRIDE = 60
N_LAGS = 12


def bootstrap_ci(y_true, y_pred, baseline_rmse, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    imps = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        m_rmse = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        imps[b] = (1 - m_rmse / baseline_rmse) * 100
    return float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))


def main():
    print("=" * 70)
    print("POLYNOMIAL EXPANSION OF SQUARED LAGS")
    print("=" * 70)

    # Load data
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    nr = df["norm_return"].values.astype(np.float64)
    nr = np.nan_to_num(nr, nan=0.0)

    # Build squared-lag features + volatility target
    n = len(nr)
    rows_X, rows_Y = [], []
    for i in range(0, n - HORIZON, STRIDE):
        if i < N_LAGS:
            continue
        lags = [nr[i - j] ** 2 for j in range(1, N_LAGS + 1)]
        rows_X.append(lags)
        rows_Y.append(np.sqrt(np.mean(nr[i + 1: i + 1 + HORIZON] ** 2)))
    X = np.array(rows_X)
    Y = np.array(rows_Y)

    # Polynomial expansion
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out([f"sq{j}" for j in range(1, N_LAGS + 1)])

    print(f"Original features: {X.shape[1]}")
    print(f"After poly expansion: {X_poly.shape[1]}")
    print(f"  ({X.shape[1]} original + {X_poly.shape[1] - X.shape[1]} interaction/poly terms)")

    # Walk-forward
    n_folds = 4
    fold_size = len(Y) // 5
    alphas = np.logspace(-2, 6, 30)

    print(f"\n--- Ridge + poly expansion ({X_poly.shape[1]} features) ---")
    fold_results = []
    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, len(Y))

        X_train, Y_train = X_poly[:train_end], Y[:train_end]
        X_test, Y_test = X_poly[test_start:test_end], Y[test_start:test_end]

        t0 = time.time()
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
        model.fit(X_train_s, Y_train)
        preds = model.predict(X_test_s)
        elapsed = time.time() - t0

        baseline_rmse = np.std(Y_train)
        model_rmse = np.sqrt(mean_squared_error(Y_test, preds))
        improvement = (1 - model_rmse / baseline_rmse) * 100
        r2 = 1 - (model_rmse / baseline_rmse) ** 2
        ci = bootstrap_ci(Y_test, preds, baseline_rmse)

        fold_results.append({"improvement": improvement, "ci": ci, "alpha": model.alpha_})
        sig = "+" if improvement > 0 else ""
        print(f"  Fold {fold+1}: {sig}{improvement:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  alpha={model.alpha_:.1f} ({elapsed:.1f}s)")

    # Held-out
    held_start = 4 * fold_size
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_poly[:held_start])
    X_held_s = scaler.transform(X_poly[held_start:])
    Y_tr, Y_held = Y[:held_start], Y[held_start:]

    model = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
    model.fit(X_tr_s, Y_tr)
    preds = model.predict(X_held_s)

    base = np.std(Y_tr)
    rmse = np.sqrt(mean_squared_error(Y_held, preds))
    imp = (1 - rmse / base) * 100
    ci = bootstrap_ci(Y_held, preds, base, n_boot=10000)

    sig = "+" if imp > 0 else ""
    print(f"  Held-out: {sig}{imp:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  alpha={model.alpha_:.1f}")

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Config':<30} {'Held-out':>9} {'95% CI':>18}")
    print(f"  {'-'*58}")
    print(f"  {'Ridge + 12 sq lags':<30} {'+6.78':>9} {'[+4.36, +9.08]':>18}")
    sig = "+" if imp > 0 else ""
    print(f"  {'Ridge + poly expansion':<30} {sig}{imp:>8.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")

    # Check if poly helps
    if imp > 7.0 and ci[0] > 5.0:
        print("\n  Poly expansion IMPROVES over plain squared lags.")
        print("  There is curvature/interaction structure Ridge was missing.")
    elif imp > 6.78 and ci[0] > 4.0:
        print("\n  Marginal improvement, but CIs overlap with plain squared lags.")
        print("  Not conclusive — might just be noise.")
    else:
        print("\n  Poly expansion does NOT improve. Plain Ridge already captures")
        print("  the available signal. This is likely the ceiling for linear models.")


if __name__ == "__main__":
    main()
