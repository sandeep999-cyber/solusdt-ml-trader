"""Two quick diagnostics: training AUC per horizon + volatility prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES
from model.config.splits import get_split_mask
from model.inference.engine import _ffill_np


def get_windows_with_horizon(df, horizon, window=60, stride=None):
    if stride is None:
        stride = horizon
    feat = _ffill_np(df[PHASE_A_FEATURES].values.astype(np.float32))
    feat = np.nan_to_num(feat, nan=0.0)
    nr = df["norm_return"].values.astype(np.float64)
    wv = np.lib.stride_tricks.sliding_window_view(feat, window_shape=window, axis=0)
    bi = np.arange(window - 1, len(feat))[::stride]
    X, Y, last_ret = [], [], []
    for idx in bi:
        if idx + 1 + horizon > len(nr):
            continue
        start = idx - (window - 1)
        win = wv[start].T.copy()
        X.append(win.flatten())
        Y.append(nr[idx + 1 : idx + 1 + horizon].mean())
        last_ret.append(nr[idx])
    return np.array(X), np.array(Y), np.array(last_ret)


def main():
    print("=" * 70)
    print("DIAGNOSTIC 1: TRAINING AUC PER HORIZON")
    print("=" * 70)

    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    train_df = df[get_split_mask(df, "train")].reset_index(drop=True)
    val_df = df[get_split_mask(df, "val")].reset_index(drop=True)

    horizons = [1, 3, 5, 12]
    results = []

    for h in horizons:
        X_tr, Y_tr, _ = get_windows_with_horizon(train_df, h, stride=h)
        X_v, Y_v, _ = get_windows_with_horizon(val_df, h, stride=h)
        y_tr = (Y_tr > 0).astype(int)
        y_v = (Y_v > 0).astype(int)

        # Cap training for speed
        max_train = 50000
        if len(X_tr) > max_train:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_tr), max_train, replace=False)
            X_tr_sub = X_tr[idx]
            y_tr_sub = y_tr[idx]
        else:
            X_tr_sub = X_tr
            y_tr_sub = y_tr

        mean = X_tr_sub.mean(axis=0)
        std = X_tr_sub.std(axis=0).clip(min=1e-6)
        X_tr_z = (X_tr_sub - mean) / std
        X_v_z = (X_v - mean) / std

        lr = LogisticRegression(max_iter=500, C=1.0, solver="liblinear")
        lr.fit(X_tr_z, y_tr_sub)

        train_auc = roc_auc_score(y_tr_sub, lr.predict_proba(X_tr_z)[:, 1])
        val_auc = roc_auc_score(y_v, lr.predict_proba(X_v_z)[:, 1])
        gap = train_auc - val_auc
        results.append((h, train_auc, val_auc, gap))

        verdict = "NOISE" if train_auc < 0.55 else "OVERFITTING" if gap > 0.05 else "MARGINAL"
        print(f"  H={h:2d}: train_AUC={train_auc:.4f}  val_AUC={val_auc:.4f}  "
              f"gap={gap:+.4f}  [{verdict}]")

    print()
    best_train = max(r[1] for r in results)
    if best_train < 0.55:
        print("  VERDICT: All train AUC < 0.55. Features are pure noise for direction.")
    else:
        print(f"  VERDICT: Best train AUC = {best_train:.4f}. Signal exists in-sample.")

    print()
    print("=" * 70)
    print("DIAGNOSTIC 2: VOLATILITY PREDICTION (Ridge on abs return)")
    print("=" * 70)

    # Use H=12 windows for consistency
    X_tr_vol, Y_tr_vol, _ = get_windows_with_horizon(train_df, 12, stride=12)
    X_v_vol, Y_v_vol, _ = get_windows_with_horizon(val_df, 12, stride=12)

    # Target: absolute return (realized volatility proxy)
    y_tr_vol = np.abs(Y_tr_vol)
    y_v_vol = np.abs(Y_v_vol)

    # Standardize features
    mean = X_tr_vol.mean(axis=0)
    std = X_tr_vol.std(axis=0).clip(min=1e-6)
    X_tr_z = (X_tr_vol - mean) / std
    X_v_z = (X_v_vol - mean) / std

    # Standardize targets for Ridge stability
    y_mean = y_tr_vol.mean()
    y_std = y_tr_vol.std()
    y_tr_z = (y_tr_vol - y_mean) / y_std
    y_v_z = (y_v_vol - y_mean) / y_std

    # Baseline: predict mean
    baseline_pred_z = np.zeros_like(y_v_z)
    baseline_rmse = np.sqrt(np.mean((y_v_z - baseline_pred_z) ** 2))

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_z, y_tr_z)
    pred_vol_z = ridge.predict(X_v_z)
    model_rmse = np.sqrt(np.mean((y_v_z - pred_vol_z) ** 2))

    improvement = (1 - model_rmse / baseline_rmse) * 100

    print(f"  Target: absolute 12-step norm_return")
    print(f"  Train: {len(y_tr_vol)} windows, Val: {len(y_v_vol)} windows")
    print(f"  Baseline RMSE: {baseline_rmse:.6f}")
    print(f"  Model RMSE:    {model_rmse:.6f}")
    print(f"  Improvement:   {improvement:+.2f}%")
    print()

    if improvement > 0.5:
        print("  VERDICT: Features have signal for volatility. Pivot to risk targets.")
    else:
        print("  VERDICT: Features have no signal for volatility either.")
        print("  The feature set is completely empty. Change data source or resolution.")

    # Also test H=1,3,5 for completeness
    print()
    print("  --- Shorter horizons ---")
    for h in [1, 3, 5]:
        X_tr_h, Y_tr_h, _ = get_windows_with_horizon(train_df, h, stride=h)
        X_v_h, Y_v_h, _ = get_windows_with_horizon(val_df, h, stride=h)
        y_tr_h = np.abs(Y_tr_h)
        y_v_h = np.abs(Y_v_h)

        # Cap for speed/memory
        max_n = 50000
        if len(X_tr_h) > max_n:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_tr_h), max_n, replace=False)
            X_tr_h = X_tr_h[idx]
            y_tr_h = y_tr_h[idx]

        mean_h = X_tr_h.mean(axis=0)
        std_h = X_tr_h.std(axis=0).clip(min=1e-6)
        X_tr_z_h = (X_tr_h - mean_h) / std_h
        X_v_z_h = (X_v_h - mean_h) / std_h
        y_tr_z_h = (y_tr_h - y_tr_h.mean()) / y_tr_h.std()
        y_v_z_h = (y_v_h - y_tr_h.mean()) / y_tr_h.std()

        base_rmse_h = np.sqrt(np.mean((y_v_z_h - 0) ** 2))
        ridge_h = Ridge(alpha=1.0)
        ridge_h.fit(X_tr_z_h, y_tr_z_h)
        model_rmse_h = np.sqrt(np.mean((y_v_z_h - ridge_h.predict(X_v_z_h)) ** 2))
        imp_h = (1 - model_rmse_h / base_rmse_h) * 100
        print(f"  H={h}: baseline_rmse={base_rmse_h:.4f}  model_rmse={model_rmse_h:.4f}  "
              f"improvement={imp_h:+.2f}%")


if __name__ == "__main__":
    main()
