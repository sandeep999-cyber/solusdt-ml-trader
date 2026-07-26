"""Ridge + lagged returns baseline - walk-forward.

Tests whether adding lagged returns as features lets Ridge recover
GARCH's autocorrelation edge. Isolates "does the information help"
from "does the architecture help."

If Ridge+lags closes the gap to GARCH: the information was always
available, the 10-feature set just didn't expose it.
If Ridge+lags still fails: the GRU's failure wasn't about missing
lagged-return features, pointing back to optimization/regularization.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

HORIZON = 12
STRIDE = 60


def build_lagged_features(returns, n_lags=12, horizon=HORIZON, stride=STRIDE):
    """Build feature matrix with lagged returns + original 10 features.

    Returns X (n_windows, n_features), Y (n_windows,), timestamps.
    """
    # Load original features
    import pyarrow.parquet as pq
    df = pq.read_table(_root / "data/processed/v1/SOLUSDT/1m").to_pandas()
    df = df.sort_values("timestamp")

    # Select only numeric float64 columns as features
    exclude = {"timestamp", "close_time", "norm_return", "year", "month"}
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64")]
    orig_feats = df[feat_cols].values.astype(np.float64)
    nr = df["norm_return"].values.astype(np.float64)
    ts = df["timestamp"].values

    # Replace NaN with 0
    orig_feats = np.nan_to_num(orig_feats, nan=0.0)

    n = len(nr)
    indices = list(range(0, n - horizon, stride))

    # Build lagged return features
    rows_X = []
    rows_Y = []
    rows_ts = []
    for i in indices:
        if i < n_lags:
            continue
        # Lagged returns: r[i-1], r[i-2], ..., r[i-n_lags]
        lags = [nr[i - j] for j in range(1, n_lags + 1)]
        # Original features
        orig = orig_feats[i].tolist()
        rows_X.append(lags + orig)
        rows_Y.append(np.sqrt(np.mean(nr[i + 1: i + 1 + horizon] ** 2)))
        rows_ts.append(ts[i + horizon])

    return np.array(rows_X), np.array(rows_Y), np.array(rows_ts)


def walk_forward_ridge(X, Y, timestamps, n_folds=5):
    """Walk-forward Ridge with lagged returns."""
    from sklearn.preprocessing import StandardScaler

    n = len(Y)
    fold_size = n // n_folds

    all_preds = []
    all_targets = []
    fold_results = []

    alphas = np.logspace(-2, 4, 20)

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_start >= n:
            break

        X_train, Y_train = X[:train_end], Y[:train_end]
        X_test, Y_test = X[test_start:test_end], Y[test_start:test_end]

        t0 = time.time()
        # Standardize features (fit on train, transform both)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
        model.fit(X_train_s, Y_train)
        preds = model.predict(X_test_s)
        elapsed = time.time() - t0

        baseline_rmse = np.std(Y_train)
        model_rmse = np.sqrt(np.mean((Y_test - preds) ** 2))
        improvement = (1 - model_rmse / baseline_rmse) * 100
        r2 = 1 - (model_rmse / baseline_rmse) ** 2

        # Bootstrap CI
        rng = np.random.RandomState(42)
        n_boot = 5000
        imps = np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.choice(len(Y_test), len(Y_test), replace=True)
            m_rmse = np.sqrt(np.mean((Y_test[idx] - preds[idx]) ** 2))
            imps[b] = (1 - m_rmse / baseline_rmse) * 100
        ci_lo, ci_hi = float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))

        all_preds.extend(preds.tolist())
        all_targets.extend(Y_test.tolist())
        fold_results.append({
            "fold": fold + 1,
            "n_train": train_end,
            "n_test": len(Y_test),
            "baseline_rmse": baseline_rmse,
            "model_rmse": model_rmse,
            "improvement": improvement,
            "r2": r2,
            "ci": (ci_lo, ci_hi),
            "alpha": model.alpha_,
            "time": elapsed,
            "n_features": X.shape[1],
            "n_lags": 12,
        })

        sig = "+" if improvement > 0 else ""
        print(f"  Fold {fold+1}: train={train_end} test={len(Y_test)} ({elapsed:.1f}s)")
        print(f"    alpha={model.alpha_:.4f}")
        print(f"    RMSE={model_rmse:.6f} (base={baseline_rmse:.6f})")
        print(f"    Improvement: {sig}{improvement:.2f}%  CI [{ci_lo:+.2f}, {ci_hi:+.2f}]  R2={r2:.6f}")

    # Stacked
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    stacked_rmse_base = np.std(all_targets)
    stacked_rmse_model = np.sqrt(np.mean((all_targets - all_preds) ** 2))
    stacked_imp = (1 - stacked_rmse_model / stacked_rmse_base) * 100
    stacked_r2 = 1 - (stacked_rmse_model / stacked_rmse_base) ** 2

    return fold_results, stacked_imp, stacked_r2


def main():
    print("=" * 70)
    print("RIDGE + LAGGED RETURNS — WALK-FORWARD")
    print("=" * 70)

    print("\nBuilding features with 12 lagged returns...")
    X, Y, ts = build_lagged_features(
        np.zeros(1),  # placeholder, loaded inside
        n_lags=12, horizon=HORIZON, stride=STRIDE,
    )
    print(f"  X shape: {X.shape} (12 lags + 10 orig = {X.shape[1]} features)")
    print(f"  Y shape: {Y.shape}")

    # Also run without lagged returns for comparison
    print("\n--- With lagged returns (22 features) ---")
    fold_results, stacked_imp, stacked_r2 = walk_forward_ridge(X, Y, ts)

    print("\n" + "=" * 70)
    print("FOLD-BY-FOLD SUMMARY")
    print("=" * 70)
    print(f"  {'Fold':>4} {'N_train':>7} {'N_test':>6} {'RMSE':>10} {'Base RMSE':>10} {'Improve%':>9} {'95% CI':>18} {'R2':>8} {'alpha':>8}")
    print(f"  {'-'*85}")
    for r in fold_results:
        sig = "+" if r["improvement"] > 0 else ""
        print(f"  {r['fold']:4d} {r['n_train']:7d} {r['n_test']:6d} "
              f"{r['model_rmse']:10.6f} {r['baseline_rmse']:10.6f} "
              f"{sig}{r['improvement']:8.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] "
              f"{r['r2']:8.6f} {r['alpha']:8.1f}")
    sig = "+" if stacked_imp > 0 else ""
    total_n = sum(r["n_test"] for r in fold_results)
    print(f"  {'All':>4} {'':>7} {total_n:6d} "
          f"{'':10} {'':10} "
          f"{sig}{stacked_imp:8.2f} {'':18} "
          f"{stacked_r2:8.6f}")

    # Comparison with GARCH and original 10-feature Ridge
    print("\n" + "=" * 70)
    print("COMPARISON (walk-forward)")
    print("=" * 70)
    print(f"  {'Model':<25} {'Fold 1':>8} {'Fold 2':>8} {'Fold 3':>8} {'Fold 4':>8} {'Stacked':>8}")
    print(f"  {'-'*65}")
    print(f"  {'GARCH(1,1)':<25} {'+4.79':>8} {'+1.27':>8} {'+1.19':>8} {'+0.68':>8} {'(see D024)':>8}")
    print(f"  {'Ridge (10 feat)':<25} {'-0.36':>8} {'-19.69':>8} {'+1.86':>8} {'+5.35':>8} {'-3.76':>8}")
    sigs = ["+" if r["improvement"] > 0 else "" for r in fold_results]
    print(f"  {'Ridge+12 lags (22 feat)':<25} "
          f"{sigs[0]}{fold_results[0]['improvement']:>7.2f} "
          f"{sigs[1]}{fold_results[1]['improvement']:>7.2f} "
          f"{sigs[2]}{fold_results[2]['improvement']:>7.2f} "
          f"{sigs[3]}{fold_results[3]['improvement']:>7.2f} "
          f"{sig}{stacked_imp:>7.2f}")

    print()
    if stacked_imp > 0:
        print("  Ridge+lags recovers GARCH edge. The information was available,")
        print("  the 10-feature set just didn't expose it.")
    else:
        print("  Ridge+lags does NOT recover GARCH edge. The GRU's failure")
        print("  wasn't about missing lagged-return features. Points at")
        print("  optimization/regularization as the bottleneck.")


if __name__ == "__main__":
    main()
