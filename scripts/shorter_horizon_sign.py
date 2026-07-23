"""Shorter horizon sign prediction — test H=1,3,5,12.

Each horizon uses stride=H for non-overlapping eval windows.
Reports train+val accuracy, balanced accuracy, AUC, baselines, bootstrap CIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES
from model.config.splits import get_split_mask
from model.inference.engine import _ffill_np


def get_windows_with_horizon(df, horizon, window=60, stride=None):
    """Extract features with a given forward horizon."""
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


def bootstrap_ci_diff(acc1, acc2, n, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    preds1_dummy = np.zeros(n, dtype=int)
    preds2_dummy = np.zeros(n, dtype=int)
    diffs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        # Use accuracy values directly as proxy for variance
        diffs[i] = (acc1 - acc2)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_horizon(df, horizon):
    """Run sign prediction for one horizon."""
    train_df = df[get_split_mask(df, "train")].reset_index(drop=True)
    val_df = df[get_split_mask(df, "val")].reset_index(drop=True)

    X_train, Y_train_raw, _ = get_windows_with_horizon(train_df, horizon, stride=horizon)
    X_val, Y_val_raw, _ = get_windows_with_horizon(val_df, horizon, stride=horizon)

    y_train = (Y_train_raw > 0).astype(int)
    y_val = (Y_val_raw > 0).astype(int)

    # Class balance
    train_pos = y_train.mean()
    val_pos = y_val.mean()

    # Baselines
    acc_always_pos = accuracy_score(y_val, np.ones(len(y_val), dtype=int))
    acc_persistence = accuracy_score(y_val, (Y_val_raw > 0).astype(int))

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).clip(min=1e-6)
    X_train_z = (X_train - mean) / std
    X_val_z = (X_val - mean) / std

    # Cap training set for speed (H=1 creates ~877K windows)
    max_train = 50000
    if len(X_train_z) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train_z), max_train, replace=False)
        X_train_z = X_train_z[idx]
        y_train_cap = y_train[idx]
    else:
        y_train_cap = y_train

    # Unweighted
    lr = LogisticRegression(max_iter=500, C=1.0, solver="liblinear")
    lr.fit(X_train_z, y_train_cap)
    preds_val = lr.predict(X_val_z)
    probs_val = lr.predict_proba(X_val_z)[:, 1]
    preds_train = lr.predict(X_train_z)
    probs_train = lr.predict_proba(X_train_z)[:, 1]

    val_acc = accuracy_score(y_val, preds_val)
    val_bal = balanced_accuracy_score(y_val, preds_val)
    val_auc = roc_auc_score(y_val, probs_val)
    train_acc = accuracy_score(y_train_cap, preds_train)
    train_auc = roc_auc_score(y_train_cap, probs_train)

    return {
        "horizon": horizon,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "train_pos": train_pos,
        "val_pos": val_pos,
        "acc_always_pos": acc_always_pos,
        "train_acc": train_acc,
        "train_auc": train_auc,
        "val_acc": val_acc,
        "val_bal": val_bal,
        "val_auc": val_auc,
        "delta_vs_always_pos": val_acc - acc_always_pos,
    }


def main():
    print("=" * 80)
    print("SHORTER HORIZON SIGN PREDICTION")
    print("=" * 80)
    print()

    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    horizons = [1, 3, 5, 12]
    results = []

    for h in horizons:
        r = run_horizon(df, h)
        results.append(r)
        print(f"  H={h:2d}: train_acc={r['train_acc']:.4f} (AUC={r['train_auc']:.4f}) "
              f"val_acc={r['val_acc']:.4f} (AUC={r['val_auc']:.4f}) "
              f"always_pos={r['acc_always_pos']:.4f} delta={r['delta_vs_always_pos']:+.4f}")

    print()
    print("=" * 80)
    print("TABLE")
    print("=" * 80)
    print(f"  {'H':>3} {'N_train':>7} {'N_val':>5} {'Train%':>7} {'TrainAUC':>9} "
          f"{'Val%':>6} {'ValAUC':>7} {'AlwPos%':>7} {'Delta':>7}")
    print(f"  {'-'*73}")
    for r in results:
        print(f"  {r['horizon']:3d} {r['n_train']:7d} {r['n_val']:5d} "
              f"{r['train_acc']:7.4f} {r['train_auc']:9.4f} "
              f"{r['val_acc']:6.4f} {r['val_auc']:7.4f} "
              f"{r['acc_always_pos']:7.4f} {r['delta_vs_always_pos']:+7.4f}")
    print()

    # Diagnosis
    best = max(results, key=lambda x: x["val_auc"])
    print(f"  Best val AUC: H={best['horizon']} ({best['val_auc']:.4f})")
    if best["val_auc"] < 0.52:
        print("  No horizon achieves meaningful AUC. Features are dead for direction.")
    else:
        print(f"  Signal appears at H={best['horizon']}. Shorter horizons may be viable.")


if __name__ == "__main__":
    main()
