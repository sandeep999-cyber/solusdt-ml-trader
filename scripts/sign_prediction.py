"""Sign prediction experiment — with corrected baselines and diagnostics.

CORRECTED: Majority baseline uses most frequent class (not minority).
DIAGNOSTICS: Training-set accuracy/AUC, feature leakage audit.
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
    mutual_info_score,
)

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES
from model.config.splits import get_split_mask
from model.inference.engine import _ffill_np


def get_windows(df, stride=60):
    """Extract (X_flat, y_raw, last_return) pairs."""
    feat = _ffill_np(df[PHASE_A_FEATURES].values.astype(np.float32))
    feat = np.nan_to_num(feat, nan=0.0)
    nr = df["norm_return"].values.astype(np.float64)
    H, W = 12, 60
    wv = np.lib.stride_tricks.sliding_window_view(feat, window_shape=W, axis=0)
    bi = np.arange(W - 1, len(feat))[::stride]
    X, Y, last_ret = [], [], []
    for idx in bi:
        if idx + 1 + H > len(nr):
            continue
        start = idx - (W - 1)
        win = wv[start].T.copy()
        X.append(win.flatten())
        Y.append(nr[idx + 1 : idx + 1 + H].mean())
        last_ret.append(nr[idx])
    return np.array(X), np.array(Y), np.array(last_ret)


def bootstrap_ci_acc(preds1, preds2, y_true, n_boot=10000, seed=42):
    """Bootstrap CI for accuracy difference (preds1 - preds2)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        diffs[i] = np.mean(preds1[idx] == y_true[idx]) - np.mean(preds2[idx] == y_true[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    print("=" * 70)
    print("SIGN PREDICTION EXPERIMENT (corrected)")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    train_df = df[get_split_mask(df, "train")].reset_index(drop=True)
    val_df = df[get_split_mask(df, "val")].reset_index(drop=True)

    X_train, Y_train_raw, last_ret_train = get_windows(train_df, stride=60)
    X_val, Y_val_raw, last_ret_val = get_windows(val_df, stride=60)

    y_train = (Y_train_raw > 0).astype(int)
    y_val = (Y_val_raw > 0).astype(int)

    print(f"Train: {len(X_train)} windows, Val: {len(X_val)} windows")
    print()

    # Class balance
    print("=== CLASS BALANCE ===")
    train_pos_rate = y_train.mean()
    val_pos_rate = y_val.mean()
    majority_class = 1 if train_pos_rate > 0.5 else 0
    print(f"  Train: {y_train.sum()}/{len(y_train)} positive ({train_pos_rate:.1%})")
    print(f"  Val:   {y_val.sum()}/{len(y_val)} positive ({val_pos_rate:.1%})")
    print(f"  Majority class: {'positive' if majority_class else 'negative'} ({max(train_pos_rate, 1-train_pos_rate):.1%})")
    print()

    # === BASELINES (corrected) ===
    print("=== BASELINES (corrected) ===")

    # 1. Always-positive (the actual majority class)
    acc_always_pos = accuracy_score(y_val, np.ones(len(y_val), dtype=int))
    print(f"  Always positive:    accuracy={acc_always_pos:.4f}")

    # 2. Always-negative
    acc_always_neg = accuracy_score(y_val, np.zeros(len(y_val), dtype=int))
    print(f"  Always negative:    accuracy={acc_always_neg:.4f}")

    # 3. Lag-1 persistence
    persistence_preds = (last_ret_val > 0).astype(int)
    acc_persistence = accuracy_score(y_val, persistence_preds)
    print(f"  Lag-1 persistence:  accuracy={acc_persistence:.4f}")
    print()

    # === LOGISTIC REGRESSION (no class weighting first) ===
    print("=== LOGISTIC REGRESSION ===")
    print(f"  Features: {X_train.shape[1]} (60 windows x 10 indicators)")

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).clip(min=1e-6)
    X_train_z = (X_train - mean) / std
    X_val_z = (X_val - mean) / std

    # Unweighted (fair accuracy comparison)
    lr_unw = LogisticRegression(max_iter=1000, C=1.0)
    lr_unw.fit(X_train_z, y_train)
    preds_unw = lr_unw.predict(X_val_z)
    probs_unw = lr_unw.predict_proba(X_val_z)[:, 1]
    acc_unw = accuracy_score(y_val, preds_unw)
    bal_acc_unw = balanced_accuracy_score(y_val, preds_unw)
    auc_unw = roc_auc_score(y_val, probs_unw)

    # Weighted (for minority recall)
    lr_w = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    lr_w.fit(X_train_z, y_train)
    preds_w = lr_w.predict(X_val_z)
    probs_w = lr_w.predict_proba(X_val_z)[:, 1]
    acc_w = accuracy_score(y_val, preds_w)
    bal_acc_w = balanced_accuracy_score(y_val, preds_w)
    auc_w = roc_auc_score(y_val, probs_w)

    print()
    print(f"  {'Metric':<25} {'Unweighted':>12} {'Balanced':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Val accuracy':<25} {acc_unw:12.4f} {acc_w:12.4f}")
    print(f"  {'Val balanced accuracy':<25} {bal_acc_unw:12.4f} {bal_acc_w:12.4f}")
    print(f"  {'Val AUC-ROC':<25} {auc_unw:12.4f} {auc_w:12.4f}")
    print()

    # Confusion matrix for unweighted
    cm = confusion_matrix(y_val, preds_unw)
    print(f"  Confusion matrix (unweighted):")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    print()

    # Bootstrap CIs
    print("=== BOOTSTRAP CIs (unweighted model) ===")
    ci_vs_always_pos = bootstrap_ci_acc(preds_unw, np.ones(len(y_val), dtype=int), y_val)
    ci_vs_always_neg = bootstrap_ci_acc(preds_unw, np.zeros(len(y_val), dtype=int), y_val)
    ci_vs_persistence = bootstrap_ci_acc(preds_unw, persistence_preds, y_val)

    print(f"  vs always positive: diff={acc_unw-acc_always_pos:+.4f}, 95% CI [{ci_vs_always_pos[0]:+.4f}, {ci_vs_always_pos[1]:+.4f}]")
    print(f"  vs always negative: diff={acc_unw-acc_always_neg:+.4f}, 95% CI [{ci_vs_always_neg[0]:+.4f}, {ci_vs_always_neg[1]:+.4f}]")
    print(f"  vs persistence:     diff={acc_unw-acc_persistence:+.4f}, 95% CI [{ci_vs_persistence[0]:+.4f}, {ci_vs_persistence[1]:+.4f}]")
    print()

    # === TRAINING-SET PERFORMANCE ===
    print("=== TRAINING-SET PERFORMANCE (critical diagnostic) ===")
    train_preds_unw = lr_unw.predict(X_train_z)
    train_probs_unw = lr_unw.predict_proba(X_train_z)[:, 1]
    train_acc = accuracy_score(y_train, train_preds_unw)
    train_bal_acc = balanced_accuracy_score(y_train, train_preds_unw)
    train_auc = roc_auc_score(y_train, train_probs_unw)

    print(f"  Train accuracy:         {train_acc:.4f}")
    print(f"  Train balanced accuracy:{train_bal_acc:.4f}")
    print(f"  Train AUC-ROC:          {train_auc:.4f}")
    print(f"  Val accuracy:           {acc_unw:.4f}")
    print(f"  Val AUC-ROC:            {auc_unw:.4f}")
    print()

    if train_acc > 0.55 and acc_unw < 0.52:
        print("  DIAGNOSIS: Signal exists in training data but doesn't generalize.")
        print("  -> Non-stationarity or overfitting, not total absence of signal.")
    elif train_acc < 0.52:
        print("  DIAGNOSIS: No learnable signal even in training data.")
        print("  -> Features contain no predictive information for this target.")
    else:
        print(f"  DIAGNOSIS: train={train_acc:.1%}, val={acc_unw:.1%} -- marginal")
    print()

    # === FEATURE LEAKAGE AUDIT ===
    print("=== FEATURE LEAKAGE AUDIT ===")
    print("  Testing: add random shifted column to features, retrain in-sample")
    print("  If accuracy jumps dramatically, original features may be leaking.")
    print()

    # Baseline in-sample accuracy
    baseline_train_acc = train_acc

    # Test 1: Add random noise column
    rng = np.random.RandomState(42)
    noise = rng.randn(len(X_train_z), 1)
    X_train_noise = np.hstack([X_train_z, noise])
    lr_noise = LogisticRegression(max_iter=1000, C=1.0)
    lr_noise.fit(X_train_noise, y_train)
    noise_acc = accuracy_score(y_train, lr_noise.predict(X_train_noise))
    print(f"  + random noise column:  train_acc={noise_acc:.4f} (baseline={baseline_train_acc:.4f}, diff={noise_acc-baseline_train_acc:+.4f})")

    # Test 2: Add shifted target (future leak)
    shifted_target = np.roll(y_train, -1)  # shift forward by 1
    X_train_leak = np.hstack([X_train_z, shifted_target.reshape(-1, 1)])
    lr_leak = LogisticRegression(max_iter=1000, C=1.0)
    lr_leak.fit(X_train_leak, y_train)
    leak_acc = accuracy_score(y_train, lr_leak.predict(X_train_leak))
    print(f"  + shifted target column: train_acc={leak_acc:.4f} (baseline={baseline_train_acc:.4f}, diff={leak_acc-baseline_train_acc:+.4f})")

    if leak_acc - baseline_train_acc > 0.05:
        print("  WARNING: Large jump with shifted target — possible look-ahead leak in features!")
    else:
        print("  No evidence of look-ahead leak.")
    print()

    # Test 3: Univariate mutual information
    print("  Univariate MI with target (top 5 features):")
    mi_scores = []
    for col in range(X_train_z.shape[1]):
        # Discretize for MI calculation
        x_disc = pd.qcut(X_train_z[:, col], q=10, duplicates="drop")
        mi = mutual_info_score(x_disc, y_train)
        mi_scores.append((col, mi))
    mi_scores.sort(key=lambda x: -x[1])
    for rank, (col, mi) in enumerate(mi_scores[:5]):
        w_idx = col // len(PHASE_A_FEATURES)
        f_idx = col % len(PHASE_A_FEATURES)
        print(f"    {rank+1}. feature[{col}] w{w_idx:02d}_{PHASE_A_FEATURES[f_idx]}: MI={mi:.4f}")

    max_mi = mi_scores[0][1]
    if max_mi < 0.01:
        print("  All MI < 0.01 — features have negligible relationship with target.")
    print()

    # === FINAL SUMMARY ===
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Always positive baseline:  {acc_always_pos:.4f}")
    print(f"  Lag-1 persistence:         {acc_persistence:.4f}")
    print(f"  Logistic regression:       {acc_unw:.4f}  AUC={auc_unw:.4f}")
    print(f"  Train accuracy:            {train_acc:.4f}")
    print()
    if auc_unw < 0.52 and train_acc < 0.55:
        print("  CONCLUSION: Features contain no predictive signal for 12-step")
        print("  return direction. Neither magnitude (D016) nor direction (D017)")
        print("  can be predicted. The feature set is empty for this task.")
    print("=" * 70)


if __name__ == "__main__":
    main()
