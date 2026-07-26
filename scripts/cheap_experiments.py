"""Two cheap experiments:

1. Shock flag: binary feature = 1 if |return| > 3*rolling_std (60-bar window)
   Tests: does separating "something just broke" from "ordinary choppiness" help?

2. Squared lags on return-prediction task (target = norm_return at t+horizon)
   Tests: does the return-prediction task revive with squared lags?

Both use the same verified walk-forward + held-out pipeline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
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


def walk_forward_ridge(X, Y, n_folds=4, alphas=None):
    """Walk-forward Ridge, return fold results + held-out."""
    if alphas is None:
        alphas = np.logspace(-2, 4, 20)

    n = len(Y)
    fold_size = n // 5  # use 5-fold structure even if we only eval 4

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
        ci = bootstrap_ci(Y_test, preds, baseline_rmse)

        fold_results.append({
            "fold": fold + 1, "n_train": train_end, "n_test": len(Y_test),
            "improvement": improvement, "r2": r2, "ci": ci,
            "alpha": model.alpha_, "time": elapsed,
        })

        sig = "+" if improvement > 0 else ""
        print(f"    Fold {fold+1}: {sig}{improvement:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  alpha={model.alpha_:.1f} ({elapsed:.1f}s)")

    # Held-out
    held_start = 4 * fold_size
    X_tr = X[:held_start]
    Y_tr = Y[:held_start]
    X_held = X[held_start:]
    Y_held = Y[held_start:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_held_s = scaler.transform(X_held)

    model = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
    model.fit(X_tr_s, Y_tr)
    preds = model.predict(X_held_s)

    base = np.std(Y_tr)
    rmse = np.sqrt(mean_squared_error(Y_held, preds))
    imp = (1 - rmse / base) * 100
    ci = bootstrap_ci(Y_held, preds, base, n_boot=10000)
    r2 = 1 - (rmse / base) ** 2

    sig = "+" if imp > 0 else ""
    print(f"    Held-out: {sig}{imp:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  alpha={model.alpha_:.1f}")

    return fold_results, imp, ci, r2


def main():
    # Load data
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    nr = df["norm_return"].values.astype(np.float64)
    nr = np.nan_to_num(nr, nan=0.0)

    # ================================================================
    # EXPERIMENT 1: Shock flag
    # ================================================================
    print("=" * 70)
    print("EXPERIMENT 1: SHOCK FLAG")
    print("=" * 70)
    print("Shock = |return| > 3 * rolling_std(60 bars)")
    print()

    # Compute shock flag: 1 if current return is extreme
    abs_nr = np.abs(nr)
    rolling_std = pd.Series(nr).rolling(60, min_periods=1).std().values
    shock_threshold = 3.0 * rolling_std
    shock_flag = (abs_nr > shock_threshold).astype(np.float64)

    print(f"  Shock rate: {shock_flag.mean():.4f} ({shock_flag.sum():.0f} shocks out of {len(shock_flag)} bars)")

    # Build features: squared lags + shock flag
    n = len(nr)
    rows_X, rows_Y = [], []
    for i in range(0, n - HORIZON, STRIDE):
        if i < N_LAGS:
            continue
        lags = [nr[i - j] ** 2 for j in range(1, N_LAGS + 1)]
        rows_X.append(lags + [shock_flag[i]])
        rows_Y.append(np.sqrt(np.mean(nr[i + 1: i + 1 + HORIZON] ** 2)))
    X_shock = np.array(rows_X)
    Y_vol = np.array(rows_Y)

    # Also build: squared lags only (baseline)
    rows_X_sq = []
    for i in range(0, n - HORIZON, STRIDE):
        if i < N_LAGS:
            continue
        rows_X_sq.append([nr[i - j] ** 2 for j in range(1, N_LAGS + 1)])
    X_sq = np.array(rows_X_sq)

    print(f"\n  Baseline: Ridge + 12 squared lags")
    fold_sq, held_sq, ci_sq, r2_sq = walk_forward_ridge(X_sq, Y_vol)

    print(f"\n  Test: Ridge + 12 squared lags + shock flag (13 features)")
    fold_shock, held_shock, ci_shock, r2_shock = walk_forward_ridge(X_shock, Y_vol)

    delta = held_shock - held_sq
    print(f"\n  Delta (shock - sq_only): {delta:+.2f}%")
    if delta > 0.5 and ci_shock[0] > ci_sq[1]:
        print("  Shock flag adds signal beyond squared lags.")
    elif delta > 0:
        print("  Marginal positive, but CIs overlap. Not conclusive.")
    else:
        print("  Shock flag adds nothing or hurts.")

    # ================================================================
    # EXPERIMENT 2: Squared lags on return-prediction
    # ================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: SQUARED LAGS ON RETURN PREDICTION")
    print("=" * 70)
    print("Target: norm_return[t+horizon] (direction + magnitude)")
    print("Baseline: predict 0 (unconditional mean)")
    print()

    # Build features: squared lags
    # Target: norm_return at t+horizon (not volatility)
    rows_X_ret, rows_Y_ret = [], []
    for i in range(0, n - HORIZON, STRIDE):
        if i < N_LAGS:
            continue
        lags = [nr[i - j] ** 2 for j in range(1, N_LAGS + 1)]
        rows_X_ret.append(lags)
        rows_Y_ret.append(nr[i + HORIZON])  # raw return at t+horizon
    X_ret = np.array(rows_X_ret)
    Y_ret = np.array(rows_Y_ret)

    print(f"  Target stats: mean={Y_ret.mean():.6f} std={Y_ret.std():.6f}")
    print(f"  Baseline RMSE (predict 0): {Y_ret.std():.6f}")

    fold_ret, held_ret, ci_ret, r2_ret = walk_forward_ridge(X_ret, Y_ret)

    # Compare with earlier return-prediction baseline
    print(f"\n  Earlier OLS (10 feat, held-out): +1.9% (D016)")
    print(f"  Earlier Ridge (10 feat, walk-forward): -0.36% to +5.35% (D022)")
    sig = "+" if held_ret > 0 else ""
    print(f"  Ridge + squared lags (12 feat, held-out): {sig}{held_ret:.2f}% CI [{ci_ret[0]:+.2f}, {ci_ret[1]:+.2f}]")


if __name__ == "__main__":
    main()
