"""Volatility prediction — Ridge baseline with alpha CV.

Target: sqrt(mean(squared returns over next H steps)) — realized volatility.
Baseline: unconditional mean of training volatility.
Reports: RMSE, R²(OOS), bootstrap CI of improvement.
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
        win = wv[start].T.copy()
        X.append(win.flatten())
        # Realized volatility: sqrt(mean(squared returns over next H steps))
        future_returns = nr[idx + 1 : idx + 1 + horizon]
        y_vol = np.sqrt(np.mean(future_returns ** 2))
        Y.append(y_vol)
    return np.array(X), np.array(Y)


def bootstrap_rmse_ci(y_true, y_pred, n_boot=10000, seed=42):
    """Bootstrap 95% CI for RMSE."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    rmses = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        rmses[i] = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
    return float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))


def main():
    print("=" * 70)
    print("VOLATILITY PREDICTION — RIDGE BASELINE")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    train_df = df[get_split_mask(df, "train")].reset_index(drop=True)
    val_df = df[get_split_mask(df, "val")].reset_index(drop=True)

    # Test multiple horizons
    horizons = [1, 3, 5, 12]
    results = []

    for h in horizons:
        print(f"\n--- H={h} ---")
        X_tr, y_tr = get_windows_volatility(train_df, horizon=h, stride=h)
        X_v, y_v = get_windows_volatility(val_df, horizon=h, stride=h)

        # Cap for speed on H=1
        max_n = 50000
        if len(X_tr) > max_n:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_tr), max_n, replace=False)
            X_tr_sub, y_tr_sub = X_tr[idx], y_tr[idx]
        else:
            X_tr_sub, y_tr_sub = X_tr, y_tr

        # Standardize features
        mean = X_tr_sub.mean(axis=0)
        std = X_tr_sub.std(axis=0).clip(min=1e-6)
        X_tr_z = (X_tr_sub - mean) / std
        X_v_z = (X_v - mean) / std

        # Baseline: unconditional mean of training volatility
        baseline_pred = np.full_like(y_v, y_tr_sub.mean())
        baseline_rmse = np.sqrt(np.mean((y_v - baseline_pred) ** 2))

        # Ridge with alpha CV
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        ridge = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
        ridge.fit(X_tr_z, y_tr_sub)
        pred = ridge.predict(X_v_z)

        model_rmse = np.sqrt(np.mean((y_v - pred) ** 2))
        improvement = (1 - model_rmse / baseline_rmse) * 100
        r2 = r2_score(y_v, pred)

        # Bootstrap CI for improvement
        n = len(y_v)
        rng = np.random.RandomState(42)
        imps = np.zeros(10000)
        for i in range(10000):
            idx = rng.choice(n, n, replace=True)
            b_rmse = np.sqrt(np.mean((y_v[idx] - baseline_pred[idx]) ** 2))
            m_rmse = np.sqrt(np.mean((y_v[idx] - pred[idx]) ** 2))
            imps[i] = (1 - m_rmse / b_rmse) * 100
        ci_lo, ci_hi = float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))

        # Bootstrap CI for RMSE
        rmse_ci = bootstrap_rmse_ci(y_v, pred)

        print(f"  Train: {len(y_tr_sub)}, Val: {len(y_v)}")
        print(f"  Best alpha: {ridge.alpha_:.4f}")
        print(f"  Baseline RMSE: {baseline_rmse:.6f}")
        print(f"  Model RMSE:    {model_rmse:.6f}")
        print(f"  Improvement:   {improvement:+.2f}%")
        print(f"  95% CI:        [{ci_lo:+.2f}%, {ci_hi:+.2f}%]")
        print(f"  R²(OOS):       {r2:.6f}")

        results.append({
            "horizon": h,
            "n_train": len(y_tr_sub),
            "n_val": len(y_v),
            "alpha": ridge.alpha_,
            "baseline_rmse": baseline_rmse,
            "model_rmse": model_rmse,
            "improvement": improvement,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "r2": r2,
        })

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'H':>3} {'N_val':>6} {'Alpha':>7} {'Base RMSE':>10} {'Model RMSE':>11} "
          f"{'Improve%':>9} {'95% CI':>18} {'R²':>10}")
    print(f"  {'-'*78}")
    for r in results:
        sig = "*" if r["ci_lo"] > 0 else ""
        print(f"  {r['horizon']:3d} {r['n_val']:6d} {r['alpha']:7.2f} "
              f"{r['baseline_rmse']:10.6f} {r['model_rmse']:11.6f} "
              f"{r['improvement']:+9.2f} [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] "
              f"{r['r2']:10.6f} {sig}")

    print()
    best = max(results, key=lambda x: x["improvement"])
    if best["improvement"] > 0.5 and best["ci_lo"] > 0:
        print(f"  DECISION: Pivot to volatility. Best H={best['horizon']} "
              f"(+{best['improvement']:.2f}%, CI [{best['ci_lo']:+.2f},{best['ci_hi']:+.2f}%])")
    elif best["improvement"] > 0.5:
        print(f"  DECISION: Marginal signal at H={best['horizon']} "
              f"(+{best['improvement']:.2f}%, CI [{best['ci_lo']:+.2f},{best['ci_hi']:+.2f}%]). "
              "Not statistically significant.")
    else:
        print("  DECISION: No signal for volatility. Feature set is completely empty.")


if __name__ == "__main__":
    main()
