"""Linear regression baseline — HELD-OUT evaluation only.

Tests whether the 10-feature set has predictive power for 12-step-ahead
norm_return, using OLS on the full 60-step flattened window.

CORRECTED: Fits on train, evaluates on val (held-out).
Previous version (linear_baseline.py) had an in-sample bug — it fit on val
and evaluated on the same val set, producing misleadingly good results.
See D016 in decisions.md for the retraction.

Result: Held-out OLS val_mse=1.241, +1.9% vs baseline (1.217).
The features have no genuine predictive power for this target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES
from model.config.splits import get_split_mask
from model.inference.engine import _ffill_np


def main():
    print("Loading data...")
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")

    train_df = df[get_split_mask(df, "train")].reset_index(drop=True)
    val_df = df[get_split_mask(df, "val")].reset_index(drop=True)

    def get_windows(df):
        feat = _ffill_np(df[PHASE_A_FEATURES].values.astype(np.float32))
        feat = np.nan_to_num(feat, nan=0.0)
        nr = df["norm_return"].values.astype(np.float64)
        H, W, stride = 12, 60, 60
        wv = np.lib.stride_tricks.sliding_window_view(feat, window_shape=W, axis=0)
        bi = np.arange(W - 1, len(feat))[::stride]
        X, Y = [], []
        for idx in bi:
            if idx + 1 + H > len(nr):
                continue
            X.append(wv[idx - (W - 1)].T.flatten())
            Y.append(nr[idx + 1 : idx + 1 + H])
        return np.array(X), np.array(Y)

    train_X, train_Y = get_windows(train_df)
    val_X, val_Y = get_windows(val_df)
    val_var = val_Y.var()
    print(f"Train: {len(train_X)}, Val: {len(val_X)}, Val var: {val_var:.6f}")
    print()

    # OLS: fit on train, evaluate on val (HELD-OUT)
    print("=== OLS (held-out: fit on train, eval on val) ===")
    B, _, _, _ = np.linalg.lstsq(train_X, train_Y, rcond=None)
    val_pred = val_X @ B
    val_mse = ((val_Y - val_pred) ** 2).mean()
    delta = (val_mse - val_var) / val_var * 100
    print(f"  val_mse:    {val_mse:.6f}")
    print(f"  baseline:   {val_var:.6f}")
    print(f"  vs baseline: {delta:+.1f}%")
    print()
    print("  CONCLUSION: OLS does not beat baseline on held-out data.")
    print("  The 10-feature set has no genuine predictive power for this target.")
    print()
    print("  WARNING: The previous linear_baseline.py had an in-sample bug.")
    print("  It fit OLS on the val set and evaluated on the same val set,")
    print("  producing val_mse=0.894 (-12% vs baseline). That result was WRONG.")
    print("  See D016 in decisions.md for the retraction.")


if __name__ == "__main__":
    main()
