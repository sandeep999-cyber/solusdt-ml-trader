"""GRU volatility prediction — standalone training + evaluation.

Target: sqrt(mean(squared returns over next H steps)) — realized volatility.
Model: GRU h32 with MSE loss (single scalar output).
Baseline: unconditional mean of training volatility.
Reports: RMSE, R²(OOS), bootstrap CI of improvement vs Ridge.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from torch.utils.data import Dataset, DataLoader

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES, RunConfig
from model.config.splits import get_split_mask
from model.body.gru_encoder import FEATURE_CENTER, FEATURE_SCALE, _INPUT_CLAMP
from model.inference.engine import _ffill_np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VolatilityDataset(Dataset):
    """Windows of features -> scalar volatility target."""

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


# ---------------------------------------------------------------------------
# Model — GRU h32 with MSE loss for scalar target
# ---------------------------------------------------------------------------

class GRUVolModel(nn.Module):
    """GRU encoder -> single scalar output for volatility prediction."""

    def __init__(self, n_features=10, hidden_size=32, dropout=0.2):
        super().__init__()
        if n_features == len(FEATURE_CENTER):
            center = torch.tensor(FEATURE_CENTER, dtype=torch.float32)
            scale = torch.tensor(FEATURE_SCALE, dtype=torch.float32)
        else:
            center = torch.zeros(n_features, dtype=torch.float32)
            scale = torch.ones(n_features, dtype=torch.float32)
        self.register_buffer("feature_center", center)
        self.register_buffer("feature_scale", scale)

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = (x - self.feature_center) / self.feature_scale
        x = torch.clamp(x, min=-_INPUT_CLAMP, max=_INPUT_CLAMP)
        _, h_n = self.gru(x)
        state = h_n[-1]
        state = self.dropout(state)
        return self.head(state).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loaders(df, features, horizon, window, train_stride, val_stride, batch_size, device):
    train_mask = get_split_mask(df, "train")
    val_mask = get_split_mask(df, "val")
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    train_ds = VolatilityDataset(train_df, features, horizon, window, train_stride)
    val_ds = VolatilityDataset(val_df, features, horizon, window, val_stride)

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=use_cuda)
    return train_loader, val_loader, train_ds.Y


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
        X = X.to(device)
        pred = model(X)
        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return preds, targets


def bootstrap_improvement(targets, pred_model, pred_baseline, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(targets)
    imps = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        b_rmse = np.sqrt(np.mean((targets[idx] - pred_baseline[idx]) ** 2))
        m_rmse = np.sqrt(np.mean((targets[idx] - pred_model[idx]) ** 2))
        imps[i] = (1 - m_rmse / b_rmse) * 100
    return float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    # Config
    horizon = 12
    window = 60
    train_stride = 1
    val_stride = 60  # non-overlapping
    batch_size = 256
    lr = 1e-3
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")

    train_loader, val_loader, train_vol = make_loaders(
        df, PHASE_A_FEATURES, horizon, window, train_stride, val_stride, batch_size, device,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Cap training set for speed (H=1 stride=1 creates ~877K windows)
    max_train = 50000
    if len(train_loader.dataset) > max_train:
        rng = np.random.RandomState(42)
        subset_idx = rng.choice(len(train_loader.dataset), max_train, replace=False)
        from torch.utils.data import Subset
        train_loader = DataLoader(
            Subset(train_loader.dataset, subset_idx),
            batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=device.type == "cuda",
        )
        print(f"  Capped training to {max_train} samples")

    # Baseline: unconditional mean
    baseline_pred = np.full(len(val_loader.dataset), train_vol.mean())

    # --- Ridge baseline (cap at 50K for SVD speed) ---
    print("\n--- Ridge baseline ---")
    X_tr_flat = train_loader.dataset.dataset.X.reshape(len(train_loader.dataset.dataset), -1)
    y_tr = train_loader.dataset.dataset.Y
    X_v_flat = val_loader.dataset.X.reshape(len(val_loader.dataset), -1)
    y_v = val_loader.dataset.Y

    max_ridge = 50000
    if len(X_tr_flat) > max_ridge:
        rng = np.random.RandomState(42)
        ridx = rng.choice(len(X_tr_flat), max_ridge, replace=False)
        X_tr_ridge, y_tr_ridge = X_tr_flat[ridx], y_tr[ridx]
    else:
        X_tr_ridge, y_tr_ridge = X_tr_flat, y_tr

    # Standardize
    mean_f = X_tr_ridge.mean(axis=0)
    std_f = X_tr_ridge.std(axis=0).clip(min=1e-6)
    X_tr_z = (X_tr_ridge - mean_f) / std_f
    X_v_z = (X_v_flat - mean_f) / std_f

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_tr_z, y_tr_ridge)
    ridge_pred = ridge.predict(X_v_z)

    ridge_rmse = np.sqrt(np.mean((y_v - ridge_pred) ** 2))
    base_rmse = np.sqrt(np.mean((y_v - baseline_pred) ** 2))
    ridge_imp = (1 - ridge_rmse / base_rmse) * 100
    ridge_r2 = 1 - ridge_rmse**2 / np.var(y_v)
    ridge_ci = bootstrap_improvement(y_v, ridge_pred, baseline_pred)
    print(f"  alpha={ridge.alpha_:.2f}  RMSE={ridge_rmse:.6f}  improvement={ridge_imp:+.2f}%  "
          f"CI=[{ridge_ci[0]:+.2f},{ridge_ci[1]:+.2f}]  R²={ridge_r2:.6f}")

    # --- GRU training ---
    print(f"\n--- GRU training (epochs={epochs}, lr={lr}) ---")
    model = GRUVolModel(n_features=len(PHASE_A_FEATURES), hidden_size=32, dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        preds, targets = evaluate(model, val_loader, device)
        val_rmse = np.sqrt(np.mean((targets - preds) ** 2))
        val_r2 = 1 - val_rmse**2 / np.var(targets)
        elapsed = time.time() - t0

        is_best = val_rmse < best_val_loss
        if is_best:
            best_val_loss = val_rmse
            best_epoch = epoch
            best_preds = preds.copy()

        marker = " *" if is_best else ""
        print(f"  Epoch {epoch+1:3d}/{epochs}  train_loss={train_loss:.6f}  "
              f"val_rmse={val_rmse:.6f}  val_r2={val_r2:.6f}  {elapsed:.1f}s{marker}")

    # --- Final comparison ---
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    gru_rmse = np.sqrt(np.mean((targets - best_preds) ** 2))
    gru_imp = (1 - gru_rmse / base_rmse) * 100
    gru_r2 = 1 - gru_rmse**2 / np.var(targets)
    gru_ci = bootstrap_improvement(targets, best_preds, baseline_pred)

    # Ridge vs GRU
    rg_imps = np.zeros(10000)
    rng = np.random.RandomState(42)
    n = len(targets)
    for i in range(10000):
        idx = rng.choice(n, n, replace=True)
        r_rmse = np.sqrt(np.mean((targets[idx] - ridge_pred[idx]) ** 2))
        g_rmse = np.sqrt(np.mean((targets[idx] - best_preds[idx]) ** 2))
        rg_imps[i] = (1 - g_rmse / r_rmse) * 100
    rg_ci = (float(np.percentile(rg_imps, 2.5)), float(np.percentile(rg_imps, 97.5)))

    print(f"  {'Model':<20} {'RMSE':>10} {'Improve%':>10} {'95% CI':>20} {'R²':>10}")
    print(f"  {'-'*72}")
    print(f"  {'Baseline':<20} {base_rmse:10.6f} {'---':>10} {'---':>20} {'0.000':>10}")
    print(f"  {'Ridge':<20} {ridge_rmse:10.6f} {ridge_imp:+10.2f} [{ridge_ci[0]:+.2f},{ridge_ci[1]:+.2f}] {ridge_r2:10.6f}")
    print(f"  {'GRU h32':<20} {gru_rmse:10.6f} {gru_imp:+10.2f} [{gru_ci[0]:+.2f},{gru_ci[1]:+.2f}] {gru_r2:10.6f}")
    print(f"  {'GRU vs Ridge':<20} {'':>10} {float(np.mean(rg_imps)):+10.2f} [{rg_ci[0]:+.2f},{rg_ci[1]:+.2f}]")
    print()

    if rg_ci[0] > 0:
        print("  GRU beats Ridge (nonlinear edge exists).")
    elif rg_ci[1] < 0:
        print("  Ridge beats GRU (GRU overfits).")
    else:
        print("  GRU and Ridge are equivalent (no nonlinear edge). Linear is simpler.")


if __name__ == "__main__":
    main()
