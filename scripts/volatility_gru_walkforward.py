"""GRU volatility walk-forward — 5-fold expanding window.

Trains GRU h32 from scratch on each fold's expanding training set.
Reports fold-by-fold RMSE, improvement, R² + stacked metrics.
Same folds as volatility_corrected.py for direct comparison with Ridge.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES
from model.config.splits import get_split_mask
from model.body.gru_encoder import FEATURE_CENTER, FEATURE_SCALE, _INPUT_CLAMP
from model.inference.engine import _ffill_np


class VolatilityDataset(Dataset):
    def __init__(self, df, features, horizon=12, window=60, stride=1):
        feat = _ffill_np(df[features].values.astype(np.float32))
        feat = np.nan_to_num(feat, nan=0.0)
        nr = df["norm_return"].values.astype(np.float64)
        wv = np.lib.stride_tricks.sliding_window_view(feat, window_shape=window, axis=0)
        bi = np.arange(window - 1, len(feat))[::stride]
        X_list, Y_list = [], []
        for idx in bi:
            if idx + 1 + horizon > len(nr):
                continue
            start = idx - (window - 1)
            X_list.append(wv[start].T.copy())
            future = nr[idx + 1 : idx + 1 + horizon]
            Y_list.append(np.sqrt(np.mean(future ** 2)))
        self.X = np.array(X_list, dtype=np.float32)
        self.Y = np.array(Y_list, dtype=np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.Y[idx])


class GRUVolModel(nn.Module):
    def __init__(self, n_features=10, hidden_size=32, dropout=0.2):
        super().__init__()
        center = torch.tensor(FEATURE_CENTER, dtype=torch.float32) if n_features == len(FEATURE_CENTER) else torch.zeros(n_features)
        scale = torch.tensor(FEATURE_SCALE, dtype=torch.float32) if n_features == len(FEATURE_CENTER) else torch.ones(n_features)
        self.register_buffer("feature_center", center)
        self.register_buffer("feature_scale", scale)
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = (x - self.feature_center) / self.feature_scale
        x = torch.clamp(x, min=-_INPUT_CLAMP, max=_INPUT_CLAMP)
        _, h_n = self.gru(x)
        state = self.dropout(h_n[-1])
        return self.head(state).squeeze(-1)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        pred = model(X.to(device))
        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def compute_metrics(y_true, y_pred, train_mean):
    baseline_mse = np.mean((y_true - train_mean) ** 2)
    model_mse = np.mean((y_true - y_pred) ** 2)
    baseline_rmse = np.sqrt(baseline_mse)
    model_rmse = np.sqrt(model_mse)
    improvement = (1 - model_rmse / baseline_rmse) * 100
    r2 = 1 - model_mse / baseline_mse
    return {
        "baseline_rmse": baseline_rmse,
        "model_rmse": model_rmse,
        "improvement": improvement,
        "r2": r2,
    }


def bootstrap_ci(y_true, y_pred, train_mean, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    imps = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        b_rmse = np.sqrt(np.mean((y_true[idx] - train_mean) ** 2))
        m_rmse = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        imps[i] = (1 - m_rmse / b_rmse) * 100
    return float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))


def train_gru_fold(X_train, y_train, X_test, y_test, device, epochs=15, lr=1e-3, batch_size=256):
    """Train GRU from scratch on one fold, return predictions + metrics."""
    # Cap training set for speed
    max_train = 100000
    if len(X_train) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), max_train, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    train_mean = y_train.mean()

    train_ds = VolatilityDataset.__new__(VolatilityDataset)
    train_ds.X = X_train
    train_ds.Y = y_train
    test_ds = VolatilityDataset.__new__(VolatilityDataset)
    test_ds.X = X_test
    test_ds.Y = y_test

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GRUVolModel(n_features=X_train.shape[2], hidden_size=32, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_rmse = float("inf")
    best_preds = None
    best_epoch = -1

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, device)
        preds, targets = evaluate(model, test_loader, device)
        rmse = np.sqrt(np.mean((targets - preds) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_preds = preds.copy()
            best_epoch = epoch

    m = compute_metrics(y_test, best_preds, train_mean)
    ci = bootstrap_ci(y_test, best_preds, train_mean)
    return best_preds, m, ci, best_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Load full dataset
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")

    # Get all windows (same as volatility_corrected.py)
    full_ds = VolatilityDataset(df, PHASE_A_FEATURES, horizon=12, window=60, stride=12)
    X_all = full_ds.X
    y_all = full_ds.Y
    n_total = len(y_all)
    print(f"Total windows: {n_total}")

    # 5-fold expanding window (same splits as Ridge)
    n_folds = 5
    fold_size = n_total // n_folds

    all_preds = []
    all_targets = []
    fold_results = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_total)

        if test_start >= n_total:
            break

        X_train = X_all[:train_end]
        y_train = y_all[:train_end]
        X_test = X_all[test_start:test_end]
        y_test = y_all[test_start:test_end]

        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}: train={train_end}, test={test_start}-{test_end} ({len(y_test)} windows)")
        print(f"{'='*60}")

        t0 = time.time()
        preds, m, ci, best_ep = train_gru_fold(X_train, y_train, X_test, y_test, device)
        elapsed = time.time() - t0

        all_preds.extend(preds.tolist())
        all_targets.extend(y_test.tolist())
        fold_results.append({"fold": fold+1, "n_train": train_end, "n_test": len(y_test),
                             "metrics": m, "ci": ci, "best_epoch": best_ep, "time": elapsed})

        sig = "+" if m["improvement"] > 0 else ""
        print(f"  Best epoch: {best_ep}")
        print(f"  RMSE: {m['model_rmse']:.6f} (baseline: {m['baseline_rmse']:.6f})")
        print(f"  Improvement: {sig}{m['improvement']:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        print(f"  R²: {m['r2']:.6f}")
        print(f"  Time: {elapsed:.1f}s")

    # Stacked
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    # Use full training set mean as baseline
    train_mask = get_split_mask(df, "train")
    train_df = df[train_mask].reset_index(drop=True)
    train_ds_full = VolatilityDataset(train_df, PHASE_A_FEATURES, horizon=12, window=60, stride=12)
    full_train_mean = train_ds_full.Y.mean()

    stacked_m = compute_metrics(all_targets, all_preds, full_train_mean)
    stacked_ci = bootstrap_ci(all_targets, all_preds, full_train_mean, n_boot=10000)

    print(f"\n{'='*60}")
    print(f"STACKED (all folds combined)")
    print(f"{'='*60}")
    print(f"  N: {len(all_targets)}")
    print(f"  RMSE: {stacked_m['model_rmse']:.6f} (baseline: {stacked_m['baseline_rmse']:.6f})")
    sig = "+" if stacked_m["improvement"] > 0 else ""
    print(f"  Improvement: {sig}{stacked_m['improvement']:.2f}%  CI [{stacked_ci[0]:+.2f}, {stacked_ci[1]:+.2f}]")
    print(f"  R²: {stacked_m['r2']:.6f}")

    # Summary table
    print(f"\n{'='*60}")
    print("FOLD-BY-FOLD SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Fold':>4} {'N_train':>7} {'N_test':>6} {'RMSE':>10} {'Base RMSE':>10} {'Improve%':>9} {'95% CI':>18} {'R²':>8} {'BestEp':>6}")
    print(f"  {'-'*84}")
    for r in fold_results:
        m = r["metrics"]
        ci = r["ci"]
        sig = "+" if m["improvement"] > 0 else ""
        print(f"  {r['fold']:4d} {r['n_train']:7d} {r['n_test']:6d} "
              f"{m['model_rmse']:10.6f} {m['baseline_rmse']:10.6f} "
              f"{sig}{m['improvement']:8.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] "
              f"{m['r2']:8.6f} {r['best_epoch']:6d}")
    sig = "+" if stacked_m["improvement"] > 0 else ""
    print(f"  {'All':>4} {n_total:7d} {len(all_targets):6d} "
          f"{stacked_m['model_rmse']:10.6f} {stacked_m['baseline_rmse']:10.6f} "
          f"{sig}{stacked_m['improvement']:8.2f} [{stacked_ci[0]:+.2f},{stacked_ci[1]:+.2f}] "
          f"{stacked_m['r2']:8.6f}")

    # Ridge comparison (precomputed from volatility_corrected.py)
    print(f"\n{'='*60}")
    print("COMPARISON: GRU vs Ridge (walk-forward)")
    print(f"{'='*60}")
    ridge_stacked_imp = -3.76
    ridge_stacked_r2 = -0.077
    print(f"  {'Model':<12} {'Stacked Improve%':>17} {'Stacked R²':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Ridge':<12} {ridge_stacked_imp:+17.2f} {ridge_stacked_r2:12.6f}")
    sig = "+" if stacked_m["improvement"] > 0 else ""
    print(f"  {'GRU h32':<12} {sig}{stacked_m['improvement']:+16.2f} {stacked_m['r2']:12.6f}")


if __name__ == "__main__":
    main()
