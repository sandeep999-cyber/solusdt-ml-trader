"""Ridge + squared-lagged-returns — the real informational vs structural test.

GARCH autoregresses on squared returns (shock magnitude), not raw returns.
This script tests whether a linear model with the RIGHT transform of the
underlying data can recover GARCH's edge.

Also checks multicollinearity diagnostics from the raw-lags run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

HORIZON = 12
STRIDE = 60


def load_data():
    import pyarrow.parquet as pq
    df = pq.read_table(_root / "data/processed/v1/SOLUSDT/1m").to_pandas()
    df = df.sort_values("timestamp")
    exclude = {"timestamp", "close_time", "norm_return", "year", "month"}
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64")]
    orig_feats = df[feat_cols].values.astype(np.float64)
    nr = df["norm_return"].values.astype(np.float64)
    ts = df["timestamp"].values
    orig_feats = np.nan_to_num(orig_feats, nan=0.0)
    return nr, ts, orig_feats, feat_cols


def build_features(nr, n_lags=12, variant="raw", orig_feats=None):
    """Build feature matrix with lagged returns (raw or squared).

    variant="raw": lagged returns r[t-1], ..., r[t-k]
    variant="squared": lagged squared returns r[t-1]^2, ..., r[t-k]^2
    variant="abs": lagged absolute returns |r[t-1]|, ..., |r[t-k]|
    """
    n = len(nr)
    indices = list(range(0, n - HORIZON, STRIDE))

    rows_X = []
    rows_Y = []
    for i in indices:
        if i < n_lags:
            continue
        if variant == "raw":
            lags = [nr[i - j] for j in range(1, n_lags + 1)]
        elif variant == "squared":
            lags = [nr[i - j] ** 2 for j in range(1, n_lags + 1)]
        elif variant == "abs":
            lags = [abs(nr[i - j]) for j in range(1, n_lags + 1)]
        else:
            raise ValueError(variant)

        orig = orig_feats[i].tolist() if orig_feats is not None else []
        rows_X.append(lags + orig)
        rows_Y.append(np.sqrt(np.mean(nr[i + 1: i + 1 + HORIZON] ** 2)))

    return np.array(rows_X), np.array(rows_Y)


def walk_forward_ridge(X, Y, n_folds=5, label=""):
    """Walk-forward Ridge, return fold results."""
    from sklearn.metrics import mean_squared_error

    n = len(Y)
    fold_size = n // n_folds
    alphas = np.logspace(-2, 4, 20)

    fold_results = []
    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_start >= n:
            break

        X_train, Y_train = X[:train_end], Y[:train_end]
        X_test, Y_test = X[test_start:test_end], Y[test_start:test_end]

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

        # Bootstrap CI
        rng = np.random.RandomState(42)
        n_boot = 5000
        imps = np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.choice(len(Y_test), len(Y_test), replace=True)
            m_rmse = np.sqrt(mean_squared_error(Y_test[idx], preds[idx]))
            imps[b] = (1 - m_rmse / baseline_rmse) * 100
        ci_lo, ci_hi = float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))

        # Condition number of X_train_s (multicollinearity check)
        try:
            cond = np.linalg.cond(X_train_s)
        except Exception:
            cond = float("inf")

        fold_results.append({
            "fold": fold + 1,
            "n_train": train_end, "n_test": len(Y_test),
            "baseline_rmse": baseline_rmse, "model_rmse": model_rmse,
            "improvement": improvement, "r2": r2,
            "ci": (ci_lo, ci_hi), "alpha": model.alpha_,
            "cond": cond, "time": elapsed,
        })

        sig = "+" if improvement > 0 else ""
        print(f"  Fold {fold+1}: train={train_end} test={len(Y_test)} ({elapsed:.1f}s) alpha={model.alpha_:.1f} cond={cond:.0f}")
        print(f"    RMSE={model_rmse:.6f} (base={baseline_rmse:.6f})")
        print(f"    Improvement: {sig}{improvement:.2f}%  CI [{ci_lo:+.2f}, {ci_hi:+.2f}]  R2={r2:.6f}")

    return fold_results


def main():
    print("=" * 70)
    print("RIDGE + SQUARED-LAGGED-RETURNS — STRUCTURAL VS INFORMATIONAL TEST")
    print("=" * 70)

    nr, ts, orig_feats, feat_cols = load_data()
    print(f"Loaded: {len(nr)} returns, {len(feat_cols)} original features")

    variants = [
        ("raw", "Raw lagged returns (signed)"),
        ("squared", "Squared lagged returns (shock magnitude)"),
        ("abs", "Absolute lagged returns (|shock|)"),
    ]

    all_results = {}
    for variant, desc in variants:
        print(f"\n{'='*60}")
        print(f"VARIANT: {desc}")
        print(f"{'='*60}")
        X, Y = build_features(nr, n_lags=12, variant=variant, orig_feats=orig_feats)
        n_lags = 12
        n_orig = orig_feats.shape[1]
        print(f"  X shape: {X.shape} ({n_lags} lags + {n_orig} orig = {X.shape[1]} features)")
        fold_results = walk_forward_ridge(X, Y, label=variant)
        all_results[variant] = fold_results

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON: All variants + GARCH")
    print("=" * 70)
    print(f"  {'Variant':<30} {'Fold 1':>8} {'Fold 2':>8} {'Fold 3':>8} {'Fold 4':>8}")
    print(f"  {'-'*62}")

    # GARCH
    print(f"  {'GARCH(1,1)':<30} {'+4.79':>8} {'+1.27':>8} {'+1.19':>8} {'+0.68':>8}")

    # Ridge variants
    for variant, desc in variants:
        fr = all_results[variant]
        sigs = ["+" if r["improvement"] > 0 else "" for r in fr[:4]]
        vals = [f"{sigs[i]}{fr[i]['improvement']:>7.2f}" for i in range(min(4, len(fr)))]
        while len(vals) < 4:
            vals.append(f"{'':>8}")
        print(f"  {'Ridge+' + variant:<30} {'  '.join(vals)}")

    # Ridge 10 feat (from previous run)
    print(f"  {'Ridge (10 feat, no lags)':<30} {'-0.36':>8} {'-19.69':>8} {'+1.86':>8} {'+5.35':>8}")

    # Condition numbers
    print("\n" + "=" * 70)
    print("MULTICOLLINEARITY CHECK (condition number of standardized X)")
    print("=" * 70)
    for variant, desc in variants:
        fr = all_results[variant]
        conds = [r["cond"] for r in fr[:4]]
        print(f"  {variant:<12}: {conds[0]:>10.0f}  {conds[1]:>10.0f}  {conds[2]:>10.0f}  {conds[3]:>10.0f}")


if __name__ == "__main__":
    main()
