"""Walk-forward error analysis — check if fold failures are concentrated or broad-based.

For each fold, analyzes:
- Per-window squared errors distribution
- Top-10 worst windows: what timestamps, what actual vol, what predicted
- % of total MSE contributed by top-10% worst windows
- Median error vs mean error (ratio indicates tail sensitivity)
- Time distribution of worst errors (clustered or spread?)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
        ts = df["timestamp"].values
        wv = np.lib.stride_tricks.sliding_window_view(feat, window_shape=window, axis=0)
        bi = np.arange(window - 1, len(feat))[::stride]
        X_list, Y_list, ts_list = [], [], []
        for idx in bi:
            if idx + 1 + horizon > len(nr):
                continue
            if idx + window >= len(ts):
                continue
            start = idx - (window - 1)
            X_list.append(wv[start].T.copy())
            future = nr[idx + 1 : idx + 1 + horizon]
            Y_list.append(np.sqrt(np.mean(future ** 2)))
            ts_list.append(ts[idx + window])  # timestamp of the prediction point
        self.X = np.array(X_list, dtype=np.float32)
        self.Y = np.array(Y_list, dtype=np.float32)
        self.timestamps = np.array(ts_list)

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
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def analyze_fold(name, y_true, y_pred, timestamps):
    """Detailed error analysis for one fold."""
    errors = y_true - y_pred
    sq_errors = errors ** 2
    abs_pct_errors = np.abs(errors / np.clip(y_true, 1e-8, None))

    # Basic stats
    rmse = np.sqrt(np.mean(sq_errors))
    median_ae = np.median(np.abs(errors))
    mean_ae = np.mean(np.abs(errors))
    tail_ratio = mean_ae / max(median_ae, 1e-8)

    # Concentration: what % of total MSE comes from top-K% windows?
    n = len(sq_errors)
    sorted_sq = np.sort(sq_errors)[::-1]
    cumsum = np.cumsum(sorted_sq)
    total_mse = np.sum(sq_errors)
    pct_top10 = cumsum[min(int(n * 0.1), n-1)] / total_mse * 100
    pct_top5 = cumsum[min(int(n * 0.05), n-1)] / total_mse * 100
    pct_top1 = cumsum[min(int(n * 0.01), n-1)] / total_mse * 100

    # Top-10 worst windows
    worst_idx = np.argsort(sq_errors)[-10:][::-1]

    # Time distribution of worst-10% windows
    worst_10pct_threshold = np.sort(sq_errors)[int(n * 0.9)]
    worst_10pct_mask = sq_errors >= worst_10pct_threshold

    # Check if worst errors are clustered in time
    worst_timestamps = timestamps[worst_10pct_mask]
    if len(worst_timestamps) > 1:
        # Convert to datetime for clustering analysis
        worst_dt = pd.to_datetime(worst_timestamps)
        time_diffs = np.diff(worst_dt.sort_values()).astype('timedelta64[h]').astype(float)
        mean_gap_hours = np.mean(time_diffs) if len(time_diffs) > 0 else 0
        max_gap_hours = np.max(time_diffs) if len(time_diffs) > 0 else 0
    else:
        mean_gap_hours = 0
        max_gap_hours = 0

    print(f"\n  === {name} ===")
    print(f"  N: {n}, RMSE: {rmse:.6f}")
    print(f"  Median |error|: {median_ae:.6f}, Mean |error|: {mean_ae:.6f}")
    print(f"  Tail ratio (mean/median): {tail_ratio:.2f}x {'(TAIL-DRIVEN)' if tail_ratio > 2 else '(broad-based)' if tail_ratio < 1.5 else ''}")
    print(f"  MSE concentration:")
    print(f"    Top 1% windows contribute: {pct_top1:.1f}% of total MSE")
    print(f"    Top 5% windows contribute: {pct_top5:.1f}% of total MSE")
    print(f"    Top 10% windows contribute: {pct_top10:.1f}% of total MSE")
    print(f"  Worst-10% time gap: mean={mean_gap_hours:.1f}h, max={max_gap_hours:.1f}h")
    print(f"  {'CLUSTERED' if mean_gap_hours < 48 else 'SPREAD'} in time")

    print(f"\n  Top-10 worst windows:")
    print(f"  {'Rank':>4} {'Timestamp':>22} {'Actual':>10} {'Predicted':>10} {'Error':>10} {'Sq Error':>10} {'% of Total':>10}")
    for rank, idx in enumerate(worst_idx):
        ts_str = str(timestamps[idx])[:19]
        actual = y_true[idx]
        pred = y_pred[idx]
        err = errors[idx]
        sq_err = sq_errors[idx]
        pct = sq_err / total_mse * 100
        print(f"  {rank+1:4d} {ts_str:>22} {actual:10.6f} {pred:10.6f} {err:+10.6f} {sq_err:10.8f} {pct:9.1f}%")

    return {
        "rmse": rmse,
        "median_ae": median_ae,
        "mean_ae": mean_ae,
        "tail_ratio": tail_ratio,
        "pct_top1": pct_top1,
        "pct_top5": pct_top5,
        "pct_top10": pct_top10,
        "mean_gap_hours": mean_gap_hours,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    full_ds = VolatilityDataset(df, PHASE_A_FEATURES, horizon=12, window=60, stride=12)
    X_all, y_all, ts_all = full_ds.X, full_ds.Y, full_ds.timestamps
    n_total = len(y_all)
    print(f"Total windows: {n_total}")

    n_folds = 5
    fold_size = n_total // n_folds
    fold_analyses = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_total)
        if test_start >= n_total:
            break

        X_train, y_train = X_all[:train_end], y_all[:train_end]
        X_test, y_test, ts_test = X_all[test_start:test_end], y_all[test_start:test_end], ts_all[test_start:test_end]

        # Cap training
        max_train = 100000
        if len(X_train) > max_train:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), max_train, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]

        # Train
        train_ds = VolatilityDataset.__new__(VolatilityDataset)
        train_ds.X, train_ds.Y = X_train, y_train
        test_ds = VolatilityDataset.__new__(VolatilityDataset)
        test_ds.X, test_ds.Y = X_test, y_test

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

        model = GRUVolModel(n_features=X_train.shape[2], hidden_size=32, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_rmse = float("inf")
        best_preds = None
        for ep in range(15):
            train_epoch(model, train_loader, optimizer, device)
            preds, targets = predict(model, test_loader, device)
            rmse = np.sqrt(np.mean((targets - preds) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_preds = preds.copy()

        analysis = analyze_fold(f"Fold {fold+1} (train={train_end}, test={test_start}-{test_end})",
                                y_test, best_preds, ts_test)
        fold_analyses.append(analysis)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Error Concentration Analysis")
    print("=" * 70)
    print(f"  {'Fold':>6} {'RMSE':>8} {'Tail Ratio':>11} {'Top1% MSE':>10} {'Top5% MSE':>10} {'Top10% MSE':>11} {'Clustering':>12}")
    print(f"  {'-'*72}")
    for i, a in enumerate(fold_analyses):
        verdict = "CONCENTRATED" if a["tail_ratio"] > 2 else "BROAD" if a["tail_ratio"] < 1.5 else "MIXED"
        print(f"  {i+1:6d} {a['rmse']:8.4f} {a['tail_ratio']:10.2f}x {a['pct_top1']:9.1f}% {a['pct_top5']:9.1f}% {a['pct_top10']:10.1f}% {verdict:>12}")

    print()
    avg_tail = np.mean([a["tail_ratio"] for a in fold_analyses])
    avg_top10 = np.mean([a["pct_top10"] for a in fold_analyses])
    print(f"  Average tail ratio: {avg_tail:.2f}x")
    print(f"  Average top-10% MSE concentration: {avg_top10:.1f}%")
    if avg_tail > 2:
        print("  VERDICT: Errors are TAIL-DRIVEN. A few extreme events dominate the score.")
        print("  → Robust loss (Huber, MAE) or outlier handling would help.")
    elif avg_tail < 1.5:
        print("  VERDICT: Errors are BROAD-BASED. Model is wrong most of the time.")
        print("  → Signal is genuinely unstable across regimes.")
    else:
        print("  VERDICT: MIXED. Some tail sensitivity, some broad-based error.")


if __name__ == "__main__":
    main()
