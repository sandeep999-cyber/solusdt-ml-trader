"""GRU with squared-lagged-returns — walk-forward + held-out.

Tests whether the GRU can extract the shock-magnitude signal that Ridge
successfully exploits. Uses the exact same verified windowing as Ridge:
- 12 squared lagged returns as input
- stride=60, 2-bar gap, same fold structure
- Held-out: Aug-Dec 2024 (untouched during debugging)

Success criteria (pre-registered):
  A: GRU recovers ~Ridge (+4-9%) -> earlier failure was about features
  B: GRU fails badly (near/below baseline) -> GRU has its own pathology
  C: GRU intermediate (positive but < Ridge) -> check CIs before trusting
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

HORIZON = 12
STRIDE = 60
N_LAGS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleGRU(nn.Module):
    """Single-layer GRU for squared-lag volatility prediction."""

    def __init__(self, input_size=1, hidden_size=32, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout if hidden_size > 1 else 0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def build_squared_lag_data(nr, orig_feats=None):
    """Build squared-lag features and volatility targets.

    Same windowing as Ridge: stride=60, 2-bar gap.
    Returns X (n, 12, 1) for GRU input, Y (n,), indices.
    """
    n = len(nr)
    rows_X, rows_Y, rows_idx = [], [], []
    for i in range(0, n - HORIZON, STRIDE):
        if i < N_LAGS:
            continue
        lags = [nr[i - j] ** 2 for j in range(1, N_LAGS + 1)]
        rows_X.append(lags)
        rows_Y.append(np.sqrt(np.mean(nr[i + 1: i + 1 + HORIZON] ** 2)))
        rows_idx.append(i)

    X = np.array(rows_X, dtype=np.float32)
    Y = np.array(rows_Y, dtype=np.float32)
    # Reshape for GRU: (n, 12, 1) — 12 timesteps, 1 feature
    X = X[:, :, np.newaxis]
    return X, Y, np.array(rows_idx)


def train_gru_fold(X_train, Y_train, X_val, Y_val, hidden=32, lr=1e-3, epochs=30, patience=10):
    """Train GRU on one fold, return best model and metrics."""
    model = SimpleGRU(input_size=1, hidden_size=hidden, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    Y_tr = torch.tensor(Y_train, dtype=torch.float32).to(DEVICE)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    Y_v = torch.tensor(Y_val, dtype=torch.float32).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = criterion(pred, Y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, Y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_v).cpu().numpy()
    return preds, best_val_loss


def bootstrap_ci(y_true, y_pred, baseline_rmse, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    imps = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        m_rmse = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        imps[b] = (1 - m_rmse / baseline_rmse) * 100
    return float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))


def main():
    print("=" * 70)
    print("GRU + SQUARED LAGS - WALK-FORWARD + HELD-OUT")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Features: 12 squared lagged returns (input_size=1, seq_len=12)")
    print(f"Windowing: stride=60, 2-bar gap (same verified pipeline as Ridge)")
    print()

    # Load data
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    nr = df["norm_return"].values.astype(np.float64)
    nr = np.nan_to_num(nr, nan=0.0)
    ts = df["timestamp"].values

    # Build features
    X, Y, indices = build_squared_lag_data(nr)
    print(f"Total windows: {len(Y)}")

    # Walk-forward 5-fold
    n = len(Y)
    fold_size = n // 5
    fold_results = []

    for fold in range(4):  # skip fold 5 (only 2 test windows)
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        X_train, Y_train = X[:train_end], Y[:train_end]
        X_test, Y_test = X[test_start:test_end], Y[test_start:test_end]

        t0 = time.time()
        preds, val_loss = train_gru_fold(X_train, Y_train, X_test, Y_test,
                                          hidden=32, lr=1e-3, epochs=30)
        elapsed = time.time() - t0

        baseline_rmse = np.std(Y_train)
        model_rmse = np.sqrt(np.mean((Y_test - preds) ** 2))
        improvement = (1 - model_rmse / baseline_rmse) * 100
        r2 = 1 - (model_rmse / baseline_rmse) ** 2
        ci = bootstrap_ci(Y_test, preds, baseline_rmse)

        fold_results.append({
            "fold": fold + 1, "n_train": train_end, "n_test": len(Y_test),
            "baseline_rmse": baseline_rmse, "model_rmse": model_rmse,
            "improvement": improvement, "r2": r2, "ci": ci, "time": elapsed,
        })

        sig = "+" if improvement > 0 else ""
        print(f"  Fold {fold+1}: train={train_end} test={len(Y_test)} ({elapsed:.1f}s)")
        print(f"    RMSE={model_rmse:.6f} (base={baseline_rmse:.6f})")
        print(f"    Improvement: {sig}{improvement:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  R2={r2:.6f}")

    # Held-out: train on folds 1-4, evaluate on last ~20%
    held_out_start = 4 * fold_size
    X_train_full, Y_train_full = X[:held_out_start], Y[:held_out_start]
    X_held, Y_held = X[held_out_start:], Y[held_out_start:]

    print(f"\n  HELD-OUT: train={len(X_train_full)} test={len(X_held)}")
    print(f"  Period: {ts[indices[held_out_start]]} to {ts[indices[-1]]}")

    t0 = time.time()
    preds_held, _ = train_gru_fold(X_train_full, Y_train_full, X_held, Y_held,
                                    hidden=32, lr=1e-3, epochs=30)
    elapsed_held = time.time() - t0

    baseline_rmse_held = np.std(Y_train_full)
    model_rmse_held = np.sqrt(np.mean((Y_held - preds_held) ** 2))
    improvement_held = (1 - model_rmse_held / baseline_rmse_held) * 100
    r2_held = 1 - (model_rmse_held / baseline_rmse_held) ** 2
    ci_held = bootstrap_ci(Y_held, preds_held, baseline_rmse_held, n_boot=10000)

    sig = "+" if improvement_held > 0 else ""
    print(f"    RMSE={model_rmse_held:.6f} (base={baseline_rmse_held:.6f}) ({elapsed_held:.1f}s)")
    print(f"    Improvement: {sig}{improvement_held:.2f}%  CI [{ci_held[0]:+.2f}, {ci_held[1]:+.2f}]  R2={r2_held:.6f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON (walk-forward + held-out)")
    print("=" * 70)
    print(f"  {'Model':<25} {'Fold 1':>8} {'Fold 2':>8} {'Fold 3':>8} {'Fold 4':>8} {'Held-out':>9}")
    print(f"  {'-'*68}")

    # GARCH
    print(f"  {'GARCH(1,1)':<25} {'+4.79':>8} {'+1.27':>8} {'+1.19':>8} {'+0.68':>8} {'(see D024)':>9}")

    # Ridge+squared (from earlier run)
    print(f"  {'Ridge+squared lags':<25} {'+8.71':>8} {'+7.15':>8} {'+4.12':>8} {'+6.75':>8} {'+6.78':>9}")

    # GRU+squared (this run)
    sigs = ["+" if r["improvement"] > 0 else "" for r in fold_results]
    vals = [f"{sigs[i]}{fold_results[i]['improvement']:>7.2f}" for i in range(4)]
    sig_held = "+" if improvement_held > 0 else ""
    print(f"  {'GRU+squared lags':<25} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {sig_held}{improvement_held:>8.2f}")

    # Interpretation
    print()
    avg_fold_imp = np.mean([r["improvement"] for r in fold_results])
    if improvement_held > 3 and ci_held[0] > 0:
        print("  OUTCOME A: GRU recovers Ridge's edge.")
        print("  Earlier catastrophic failure was about features, not GRU optimization.")
    elif improvement_held < 0 or ci_held[1] < 0:
        print("  OUTCOME B: GRU fails despite good features.")
        print("  GRU has its own training pathology, independent of feature quality.")
    else:
        print(f"  OUTCOME C: GRU intermediate ({sig_held}{improvement_held:.2f}%).")
        if ci_held[0] > 0:
            print("  Statistically significant but below Ridge. GRU partially learns the signal.")
        else:
            print("  CI includes 0. Effect not statistically significant on held-out.")


if __name__ == "__main__":
    main()
