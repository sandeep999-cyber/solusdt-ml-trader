"""Volatility prediction — corrected metrics + walk-forward validation.

Single consistent baseline: predict training-mean volatility for all val samples.
R² = 1 - MSE_model / MSE_baseline (where MSE_baseline = mean((y_val - train_mean)^2)).
Improvement = (1 - RMSE_model / RMSE_baseline) * 100.

Also implements walk-forward expanding window validation (5 folds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES
from model.config.splits import get_split_mask
from model.inference.engine import _ffill_np


def get_windows_volatility(df, horizon=12, window=60, stride=None):
    """Extract (X_flat, y_vol) pairs with realized volatility target."""
    if stride is None:
        stride = horizon
    feat = _ffill_np(df[PHASE_A_FEATURES].values.astype(np.float32))
    feat = np.nan_to_num(feat, nan=0.0)
    nr = df["norm_return"].values.astype(np.float64)
    wv = np.lib.stride_tricks.sliding_window_view(feat, window_shape=window, axis=0)
    bi = np.arange(window - 1, len(feat))[::stride]
    X, Y = [], []
    for idx in bi:
        if idx + 1 + horizon > len(nr):
            continue
        start = idx - (window - 1)
        X.append(wv[start].T.copy().flatten())
        future = nr[idx + 1 : idx + 1 + horizon]
        Y.append(np.sqrt(np.mean(future ** 2)))
    return np.array(X), np.array(Y)


def compute_metrics(y_true, y_pred, train_mean):
    """Compute RMSE, improvement, R² — all against same baseline (training mean)."""
    baseline_mse = np.mean((y_true - train_mean) ** 2)
    model_mse = np.mean((y_true - y_pred) ** 2)
    baseline_rmse = np.sqrt(baseline_mse)
    model_rmse = np.sqrt(model_mse)
    improvement = (1 - model_rmse / baseline_rmse) * 100
    r2 = 1 - model_mse / baseline_mse  # consistent R²
    return {
        "baseline_rmse": baseline_rmse,
        "model_rmse": model_rmse,
        "improvement": improvement,
        "r2": r2,
        "baseline_mse": baseline_mse,
        "model_mse": model_mse,
    }


def bootstrap_ci(y_true, y_pred, train_mean, n_boot=10000, seed=42):
    """Bootstrap 95% CI for improvement %."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    imps = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        b_rmse = np.sqrt(np.mean((y_true[idx] - train_mean) ** 2))
        m_rmse = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        imps[i] = (1 - m_rmse / b_rmse) * 100
    return float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))


def fit_ridge(X_train, y_train, X_val):
    """Fit Ridge with alpha CV, return predictions."""
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
    ridge.fit(X_train, y_train)
    return ridge.predict(X_val), ridge.alpha_


def standardize(X_train, X_val):
    """Standardize features (zero mean, unit variance on train)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).clip(min=1e-6)
    return (X_train - mean) / std, (X_val - mean) / std


def main():
    print("=" * 70)
    print("VOLATILITY PREDICTION — CORRECTED METRICS")
    print("=" * 70)
    print()
    print("Baseline: predict training-mean volatility for all val samples.")
    print("R² = 1 - MSE_model / MSE_baseline (training mean baseline).")
    print("Improvement = (1 - RMSE_model / RMSE_baseline) * 100.")
    print()

    # Load data
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    train_df = df[get_split_mask(df, "train")].reset_index(drop=True)
    val_df = df[get_split_mask(df, "val")].reset_index(drop=True)

    # ====================================================================
    # PART 1: Single split (corrected metrics)
    # ====================================================================
    print("=" * 70)
    print("PART 1: SINGLE TRAIN/VAL SPLIT (corrected)")
    print("=" * 70)

    horizons = [1, 3, 5, 12]
    results = []

    for h in horizons:
        X_tr, y_tr = get_windows_volatility(train_df, horizon=h, stride=h)
        X_v, y_v = get_windows_volatility(val_df, horizon=h, stride=h)

        # Cap for speed
        max_n = 50000
        if len(X_tr) > max_n:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_tr), max_n, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]

        train_mean = y_tr.mean()
        X_tr_z, X_v_z = standardize(X_tr, X_v)

        # Ridge
        ridge_pred, alpha = fit_ridge(X_tr_z, y_tr, X_v_z)
        ridge_m = compute_metrics(y_v, ridge_pred, train_mean)
        ridge_ci = bootstrap_ci(y_v, ridge_pred, train_mean)

        print(f"\n  H={h}:")
        print(f"    Train: {len(y_tr)}, Val: {len(y_v)}, Train mean: {train_mean:.6f}")
        print(f"    Ridge (alpha={alpha:.2f}):")
        print(f"      RMSE: {ridge_m['model_rmse']:.6f} (baseline: {ridge_m['baseline_rmse']:.6f})")
        print(f"      Improvement: {ridge_m['improvement']:+.2f}%  CI [{ridge_ci[0]:+.2f}, {ridge_ci[1]:+.2f}]")
        print(f"      R²: {ridge_m['r2']:.6f}")
        print(f"      MSE: {ridge_m['model_mse']:.8f} (baseline: {ridge_m['baseline_mse']:.8f})")

        results.append({"horizon": h, "ridge": ridge_m, "ci": ridge_ci, "alpha": alpha})

    # Summary
    print("\n" + "-" * 70)
    print("SINGLE-SPLIT SUMMARY")
    print("-" * 70)
    print(f"  {'H':>3} {'N_val':>6} {'Alpha':>6} {'Base RMSE':>10} {'Model RMSE':>11} "
          f"{'Improve%':>9} {'95% CI':>18} {'R²':>10}")
    print(f"  {'-'*78}")
    for r in results:
        rm = r["ridge"]
        n_val = len(get_windows_volatility(val_df, r["horizon"], stride=r["horizon"])[1])
        print(f"  {r['horizon']:3d} {n_val:6d} "
              f"{r['alpha']:6.2f} {rm['baseline_rmse']:10.6f} {rm['model_rmse']:11.6f} "
              f"{rm['improvement']:+9.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] {rm['r2']:10.6f}")

    # ====================================================================
    # PART 2: Walk-forward expanding window (5 folds)
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 2: WALK-FORWARD EXPANDING WINDOW (5 folds)")
    print("=" * 70)

    # Get all data with stride=12 (non-overlapping windows)
    X_all, y_all = get_windows_volatility(df, horizon=12, stride=12)
    n_total = len(y_all)
    print(f"  Total windows: {n_total}")

    # 5-fold expanding window: train on first k/5, test on (k+1)/5
    n_folds = 5
    fold_size = n_total // n_folds
    all_preds = []
    all_targets = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_total)

        if test_start >= n_total:
            break

        X_train_fold = X_all[:train_end]
        y_train_fold = y_all[:train_end]
        X_test_fold = X_all[test_start:test_end]
        y_test_fold = y_all[test_start:test_end]

        train_mean_fold = y_train_fold.mean()
        X_tr_z, X_te_z = standardize(X_train_fold, X_test_fold)

        # Ridge
        ridge_pred_fold, _ = fit_ridge(X_tr_z, y_train_fold, X_te_z)
        m = compute_metrics(y_test_fold, ridge_pred_fold, train_mean_fold)
        ci = bootstrap_ci(y_test_fold, ridge_pred_fold, train_mean_fold, n_boot=5000)

        all_preds.extend(ridge_pred_fold.tolist())
        all_targets.extend(y_test_fold.tolist())

        print(f"\n  Fold {fold+1}: train={train_end}, test={test_start}-{test_end}")
        print(f"    Train mean: {train_mean_fold:.6f}")
        print(f"    Ridge: RMSE={m['model_rmse']:.6f}, Improvement={m['improvement']:+.2f}%, "
              f"R²={m['r2']:.6f}, CI=[{ci[0]:+.2f},{ci[1]:+.2f}]")

    # Stacked metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    overall_mean = np.mean(all_targets)  # grand mean for overall R²
    # Use the training mean from the largest fold (fold 5 = full train)
    X_tr_full, y_tr_full = get_windows_volatility(train_df, horizon=12, stride=12)
    full_train_mean = y_tr_full.mean()
    stacked_m = compute_metrics(all_targets, all_preds, full_train_mean)
    stacked_ci = bootstrap_ci(all_targets, all_preds, full_train_mean, n_boot=10000)

    print(f"\n  STACKED (all folds combined):")
    print(f"    N: {len(all_targets)}")
    print(f"    RMSE: {stacked_m['model_rmse']:.6f} (baseline: {stacked_m['baseline_rmse']:.6f})")
    print(f"    Improvement: {stacked_m['improvement']:+.2f}%  CI [{stacked_ci[0]:+.2f}, {stacked_ci[1]:+.2f}]")
    print(f"    R²: {stacked_m['r2']:.6f}")
    print(f"    MSE: {stacked_m['model_mse']:.8f} (baseline: {stacked_m['baseline_mse']:.8f})")

    # ====================================================================
    # PART 3: Reconcile — show why old numbers were wrong
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 3: METRIC RECONCILIATION")
    print("=" * 70)
    print()
    print("  Old GRU script used: R² = 1 - val_rmse² / np.var(targets)")
    print("  This is NOT consistent with improvement = (1 - rmse/baseline_rmse) * 100.")
    print()
    print("  Old Ridge script used: sklearn r2_score() = 1 - SS_res / SS_tot")
    print("  where SS_tot = sum((y - y_val_mean)^2). Different baseline again.")
    print()
    print("  CORRECT (this script):")
    print("    R² = 1 - MSE_model / MSE_baseline")
    print("    where MSE_baseline = mean((y_val - train_mean)^2)")
    print("    This is consistent with improvement = (1 - rmse/baseline_rmse) * 100.")
    print()
    print("  Relationship: R² = 1 - (1 - improvement/100)²")
    print("  For +11.4%: R² = 1 - (0.886)² = 0.215")
    print("  For +19.6%: R² = 1 - (0.804)² = 0.354")
    print()
    print("  Previous reported R² values were WRONG due to inconsistent baselines.")
    print("  The corrected R² values are higher (more impressive) than previously reported.")


if __name__ == "__main__":
    main()
