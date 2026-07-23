"""Sign prediction experiment: can the 10 features predict return direction?

Tests whether the features have directional information at the 12-step horizon,
even though they can't predict magnitude (D016).

Baselines: majority class, lag-1 persistence.
Model: logistic regression (sklearn).
Metrics: accuracy, AUC, precision/recall/F1, bootstrap CIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES
from model.config.splits import get_split_mask
from model.inference.engine import _ffill_np


def get_windows(df, stride=60):
    """Extract (X_flat, y_raw) pairs for sign prediction."""
    feat = _ffill_np(df[PHASE_A_FEATURES].values.astype(np.float32))
    feat = np.nan_to_num(feat, nan=0.0)
    nr = df["norm_return"].values.astype(np.float64)
    H, W = 12, 60
    wv = np.lib.stride_tricks.sliding_window_view(feat, window_shape=W, axis=0)
    bi = np.arange(W - 1, len(feat))[::stride]
    X, Y, last_return = [], [], []
    for idx in bi:
        if idx + 1 + H > len(nr):
            continue
        start = idx - (W - 1)
        win = wv[start].T.copy()
        X.append(win.flatten())  # (600,)
        # Target: sign of mean return over next 12 steps
        Y.append(nr[idx + 1 : idx + 1 + H].mean())
        # Lag-1: sign of the most recent return (last bar in window)
        last_return.append(nr[idx])
    return np.array(X), np.array(Y), np.array(last_return)


def bootstrap_ci(acc1, acc2, y_true, n_boot=10000, seed=42):
    """Bootstrap CI for accuracy difference (acc1 - acc2)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        diffs[i] = np.mean(acc1[idx] == y_true[idx]) - np.mean(acc2[idx] == y_true[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def permutation_test(acc_model, acc_baseline, y_true, n_perm=10000, seed=42):
    """Permutation test: is model accuracy significantly better than baseline?"""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    observed_diff = acc_model - acc_baseline
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        # Simulate null: model predictions randomly shuffled relative to truth
        acc_perm = np.mean(perm == np.arange(n))  # not useful, use different approach
        # Actually: compare model preds vs baseline preds under permutation
        break  # simplify: use bootstrap CI instead
    # Fall back to bootstrap CI for the difference
    return None


def main():
    print("=" * 70)
    print("SIGN PREDICTION EXPERIMENT")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    train_df = df[get_split_mask(df, "train")].reset_index(drop=True)
    val_df = df[get_split_mask(df, "val")].reset_index(drop=True)

    X_train, Y_train_raw, last_ret_train = get_windows(train_df, stride=60)
    X_val, Y_val_raw, last_ret_val = get_windows(val_df, stride=60)

    # Binary targets: sign of mean 12-step return
    y_train = (Y_train_raw > 0).astype(int)
    y_val = (Y_val_raw > 0).astype(int)

    print(f"Train: {len(X_train)} windows, Val: {len(X_val)} windows")
    print()

    # Class balance
    print("=== CLASS BALANCE ===")
    train_pos_rate = y_train.mean()
    val_pos_rate = y_val.mean()
    print(f"  Train: {y_train.sum()}/{len(y_train)} positive ({train_pos_rate:.1%})")
    print(f"  Val:   {y_val.sum()}/{len(y_val)} positive ({val_pos_rate:.1%})")
    print()

    # === BASELINES ===
    print("=== BASELINES ===")

    # 1. Majority class
    majority_class = 1 if train_pos_rate > 0.5 else 0
    majority_preds = np.full(len(y_val), majority_class)
    acc_majority = accuracy_score(y_val, majority_preds)
    print(f"  Majority class (always {'+' if majority_class else '-'}): accuracy={acc_majority:.4f}")

    # 2. Lag-1 persistence (sign of last return)
    persistence_preds = (last_ret_val > 0).astype(int)
    acc_persistence = accuracy_score(y_val, persistence_preds)
    print(f"  Lag-1 persistence: accuracy={acc_persistence:.4f}")
    print()

    # === LOGISTIC REGRESSION ===
    print("=== LOGISTIC REGRESSION ===")
    print(f"  Features: {X_train.shape[1]} (60 windows x 10 indicators)")

    # Standardize features (important for logistic regression)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).clip(min=1e-6)
    X_train_z = (X_train - mean) / std
    X_val_z = (X_val - mean) / std

    # Fit logistic regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    lr.fit(X_train_z, y_train)

    # Predictions
    preds = lr.predict(X_val_z)
    probs = lr.predict_proba(X_val_z)[:, 1]

    # Metrics
    acc_model = accuracy_score(y_val, preds)
    bal_acc = balanced_accuracy_score(y_val, preds)
    auc = roc_auc_score(y_val, probs)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average=None, labels=[0, 1])
    cm = confusion_matrix(y_val, preds)

    print(f"  Accuracy:         {acc_model:.4f}")
    print(f"  Balanced accuracy:{bal_acc:.4f}")
    print(f"  AUC-ROC:          {auc:.4f}")
    print(f"  Precision (neg):  {prec[0]:.4f}")
    print(f"  Recall (neg):     {rec[0]:.4f}")
    print(f"  F1 (neg):         {f1[0]:.4f}")
    print(f"  Precision (pos):  {prec[1]:.4f}")
    print(f"  Recall (pos):     {rec[1]:.4f}")
    print(f"  F1 (pos):         {f1[1]:.4f}")
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    print()

    # === BOOTSTRAP CIs ===
    print("=== BOOTSTRAP CIs (10,000 resamples) ===")

    ci_vs_majority = bootstrap_ci(preds, majority_preds, y_val)
    ci_vs_persistence = bootstrap_ci(preds, persistence_preds, y_val)

    print(f"  vs majority class:  diff={acc_model-acc_majority:+.4f}, 95% CI [{ci_vs_majority[0]:+.4f}, {ci_vs_majority[1]:+.4f}]")
    print(f"  vs persistence:     diff={acc_model-acc_persistence:+.4f}, 95% CI [{ci_vs_persistence[0]:+.4f}, {ci_vs_persistence[1]:+.4f}]")
    print()

    # === INTERPRETATION ===
    print("=== INTERPRETATION ===")
    beats_majority = ci_vs_majority[0] > 0
    beats_persistence = ci_vs_persistence[0] > 0

    if beats_majority and beats_persistence:
        print("  SIGNIFICANT: Model beats both baselines (CI excludes 0).")
        print("  Features contain directional information at 12-step horizon.")
        print("  Next: build direction-based trading signal.")
    elif beats_majority:
        print("  PARTIAL: Model beats majority class but not persistence.")
        print("  Features add nothing beyond autoregressive structure.")
    else:
        print("  NULL: Model fails to beat majority class baseline.")
        print("  Features are uninformative for direction at this horizon.")
        print("  Consider: shorter horizon, new features, or different task.")
    print()

    # === FEATURE IMPORTANCE (top 10) ===
    print("=== TOP 10 FEATURES BY ABS COEFFICIENT ===")
    coefs = lr.coef_[0]
    feature_names = []
    for w in range(60):
        for f_name in PHASE_A_FEATURES:
            feature_names.append(f"w{w:02d}_{f_name}")
    top_idx = np.argsort(np.abs(coefs))[::-1][:10]
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1:2d}. {feature_names[idx]:30s}  coef={coefs[idx]:+.4f}")
    print()


if __name__ == "__main__":
    main()
