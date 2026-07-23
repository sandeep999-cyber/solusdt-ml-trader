"""Linear regression baseline: same features, same horizon, closed-form OLS.

Tests whether the harm is GRU-specific or inherent to the features/target.
If linear model is also harmful → features/target are the problem.
If linear model is harmless → GRU capacity is causing the harm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES, RunConfig
from model.config.splits import get_split_mask
from model.inference.engine import _ffill_np


def main():
    print("Loading data...")
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")

    val_mask = get_split_mask(df, "val")
    val_df = df[val_mask].reset_index(drop=True)

    feat = _ffill_np(val_df[PHASE_A_FEATURES].values.astype(np.float32))
    feat = np.nan_to_num(feat, nan=0.0)
    norm_return = val_df["norm_return"].values.astype(np.float64)

    baseline_var = norm_return[~np.isnan(norm_return)].var()
    print(f"Baseline var: {baseline_var:.6f}")
    print(f"Val rows: {len(val_df)}")
    print(f"Features: {len(PHASE_A_FEATURES)}")
    print()

    H = 12
    W = 60
    stride = 60

    windows_view = np.lib.stride_tricks.sliding_window_view(feat, window_shape=W, axis=0)
    bar_indices = np.arange(W - 1, len(feat))[::stride]

    # Build (X, Y) pairs: X = flattened window (W*F features), Y = horizon targets
    X_list = []
    Y_list = []
    for idx in bar_indices:
        if idx + 1 + H > len(norm_return):
            continue
        start = idx - (W - 1)
        win = windows_view[start].T  # (W, F)
        win = np.ascontiguousarray(win)
        X_list.append(win.flatten())  # (W*F,)
        Y_list.append(norm_return[idx + 1 : idx + 1 + H])

    X = np.array(X_list)  # (n_windows, W*F)
    Y = np.array(Y_list)  # (n_windows, H)
    n = len(X)
    d = X.shape[1]
    print(f"Samples: {n}, features: {d} ({W} windows x {len(PHASE_A_FEATURES)} indicators)")
    print()

    # OLS: solve for B in Y = X @ B
    # Using numpy lstsq (no sklearn needed)
    B, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    Y_pred = X @ B

    # Per-horizon MSE
    print("=== Linear Regression OLS ===")
    print(f"{'Horizon':<10} {'MSE':>10} {'vs baseline':>12}")
    print("-" * 35)
    horizon_mses = []
    for h in range(H):
        mse = float(((Y[:, h] - Y_pred[:, h]) ** 2).mean())
        horizon_mses.append(mse)
        delta = (mse - baseline_var) / baseline_var * 100
        print(f"  t+{h+1:<7} {mse:.6f}  {delta:>+8.1f}%")
    print("-" * 35)
    avg_mse = float(((Y - Y_pred) ** 2).mean())
    delta = (avg_mse - baseline_var) / baseline_var * 100
    print(f"  {'Average':<8} {avg_mse:.6f}  {delta:>+8.1f}%")
    print()
    print(f"  Baseline var: {baseline_var:.6f}")
    print()

    # Bootstrap CI for average MSE
    print("Bootstrap CI (10,000 resamples)...")
    N_BOOT = 10000
    rng = np.random.RandomState(42)
    window_mses = ((Y - Y_pred) ** 2).mean(axis=1)  # (n,) per-window MSE
    boots = np.array([window_mses[rng.choice(n, n, replace=True)].mean() for _ in range(N_BOOT)])
    ci_lo, ci_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    print(f"  Linear MSE: {avg_mse:.6f}  95% CI [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"  Baseline:   {baseline_var:.6f}")
    print(f"  CI excludes baseline? {'YES (better)' if ci_hi < baseline_var else 'YES (worse)' if ci_lo > baseline_var else 'NO'}")
    print()

    # Compare to GRU models (load their MSE values from comparison output)
    print("=== Comparison to GRU models ===")
    print(f"{'Model':<25} {'MSE':>10} {'vs baseline':>12}")
    print("-" * 50)
    print(f"{'Baseline (uncond. var)':<25} {baseline_var:.6f}  {'---':>12}")
    print(f"{'Linear OLS':<25} {avg_mse:.6f}  {delta:>+8.1f}%")
    gru_mses = {"stride=1 (GRU)": 1.217897, "stride=15 (GRU)": 1.220167, "stride=60 (GRU)": 1.219981}
    for name, mse in gru_mses.items():
        d = (mse - baseline_var) / baseline_var * 100
        print(f"{name:<25} {mse:.6f}  {d:>+8.1f}%")
    print("=" * 50)
    print()

    if avg_mse < baseline_var - 0.005:
        print("CONCLUSION: Linear model beats baseline -> GRU capacity is causing the harm.")
        print("  The features have signal, but the GRU memorizes noise.")
        print("  Next: reduce GRU capacity or switch to linear/lighter model.")
    elif avg_mse > baseline_var + 0.005:
        print("CONCLUSION: Linear model also harmful -> features/target are the problem.")
        print("  No architecture change will fix this. Need different features or target.")
        print("  Next: CVD+price-only to test if a subset of features is less harmful.")
    else:
        print("CONCLUSION: Linear model approx baseline -> features have no useful signal.")
        print("  Neither linear nor nonlinear can do better than unconditional mean.")
        print("  Need fundamentally different features or task formulation.")


if __name__ == "__main__":
    main()
