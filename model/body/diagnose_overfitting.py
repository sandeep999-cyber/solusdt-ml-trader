"""Compare val NLL on full-overlap vs strided (non-overlapping) windows.

If stride=1 and stride=60 give similar val NLL → the train/val gap is real.
If stride=60 gives significantly lower NLL → the "overfitting" is mostly an
overlap artifact, and the real fix is different (stride-aware validation,
or subsampling training windows, not capacity reduction).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES
from model.config.splits import get_split_mask
from model.inference.engine import _ffill_np, _resolve_best_checkpoint
from model.inference.engine import _build_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("diagnose")
logger.setLevel(logging.INFO)


def compute_val_nll(model, feat_data, targets, stride, horizon_weight_decay):
    """Compute val NLL consistently with train.py's _validate / compute_loss.

    NLL per step = 0.5 * (log_var + (target - mean)^2 / exp(log_var)).
    Weighted mean over horizon (uniform per step), averaged across the batch.
    """
    H = model.horizon
    W = model.window_length
    n = len(feat_data)
    device = next(model.parameters()).device

    if horizon_weight_decay > 0:
        steps = np.arange(H, dtype=np.float32)
        w = (horizon_weight_decay ** steps)
        w = w / w.sum()
    else:
        w = np.ones(H, dtype=np.float32) / H

    windows_view = np.lib.stride_tricks.sliding_window_view(
        feat_data, window_shape=W, axis=0
    )
    # windows_view[k] contains raw features for positions [k, k+W-1].
    # bar index for window k is k + W - 1.

    bar_indices = np.arange(W - 1, n)[::stride]
    indices = bar_indices - (W - 1)  # start positions in windows_view
    n_windows = len(bar_indices)

    total_nll = 0.0
    total_mse = 0.0
    total_valid_steps = 0
    count = 0

    BATCH = 256
    with torch.no_grad():
        for batch_start in range(0, n_windows, BATCH):
            batch_end = min(batch_start + BATCH, n_windows)
            batch_pos = indices[batch_start:batch_end]
            batch_bar = bar_indices[batch_start:batch_end]

            batch_win = windows_view[batch_pos]
            batch_win = batch_win.transpose(0, 2, 1)  # (B, F, W) → (B, W, F)
            batch_win = np.ascontiguousarray(batch_win)
            batch_t = torch.from_numpy(batch_win).to(device)

            mean, log_var = model(batch_t)

            for b, idx in enumerate(batch_bar):
                if idx + 1 + H > len(targets):
                    continue
                tgt = targets[idx + 1: idx + 1 + H]
                valid = ~np.isnan(tgt)
                if not valid.any():
                    continue

                m = mean[b].cpu().numpy()
                lv = log_var[b].cpu().numpy()

                lv = np.clip(lv, -10.0, 10.0)
                var = np.exp(lv)

                se = (tgt - m) ** 2
                step_nll = 0.5 * (lv + se / var)
                step_nll = np.nan_to_num(step_nll, nan=0.0)

                # Uniform-weighted average over valid steps
                step_nll_weighted = (step_nll * w)
                valid_mask = valid.astype(np.float32)
                w_sum = (w * valid_mask).sum()
                if w_sum < 1e-8:
                    continue

                total_nll += step_nll_weighted.sum() / w_sum
                total_mse += (se * valid_mask).sum()
                total_valid_steps += valid_mask.sum()
                count += 1

    if count == 0:
        return {"nll": float("nan"), "mse": float("nan"), "windows": 0}

    # nll as flat mean per element (matching compute_loss's 'nll' reporting)
    nll_val = total_nll / count
    mse_val = total_mse / max(total_valid_steps, 1)
    return {
        "nll": round(float(nll_val), 6),
        "mse": round(float(mse_val), 6),
        "windows": count,
        "valid_steps": int(total_valid_steps),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load data
    data_path = _root / "data" / "processed" / "v1" / "SOLUSDT" / "1m"
    logger.info("Loading feature data from %s", data_path)
    df = pd.read_parquet(data_path).sort_values("timestamp")
    val_mask = get_split_mask(df, "val")
    val_df = df[val_mask].reset_index(drop=True)
    logger.info("Val split: %d rows", len(val_df))

    # Load checkpointed model
    checkpoint_path = _resolve_best_checkpoint()
    n_features = len(PHASE_A_FEATURES)
    logger.info("Loading model from %s", checkpoint_path)
    model = _build_model(checkpoint_path, device)
    model.eval()

    # Prep data (normalization is inside the model forward pass)
    feat_data = val_df[PHASE_A_FEATURES].values.astype(np.float32)
    targets = val_df["norm_return"].values.astype(np.float64)

    # NaN handling per CONTRACT.md §2.3 (identical to engine.py)
    feat_data = _ffill_np(feat_data)
    feat_data = np.nan_to_num(feat_data, nan=0.0)

    horizon_weight_decay = 0.0  # uniform weighting, matches checkpoint config

    logger.info("\n── Overlap Diagnostic ──")
    logger.info("%-25s %s", "", "val NLL     MSE    windows")
    for stride, label in [(1, "stride=1 (full overlap)"),
                          (60, "stride=60 (60-min spaced)")]:
        result = compute_val_nll(model, feat_data, targets, stride, horizon_weight_decay)
        logger.info("  %-25s  %.4f  %.4f  %d",
                    label, result["nll"], result["mse"], result["windows"])

    logger.info("\n── Baseline for reference ──")
    norm_var = targets[~np.isnan(targets)].var()
    baseline_nll = 0.5 * (np.nanmean(targets**2 / norm_var + np.log(norm_var)))
    logger.info("  Persistence baseline (val var=%.6f): %.6f", norm_var, baseline_nll)

    # Also compute striding over the FULL range (not just val) to compare
    logger.info("\n── Full-dataset context ──")
    full_feat = _ffill_np(df[PHASE_A_FEATURES].values.astype(np.float32))
    full_feat = np.nan_to_num(full_feat, nan=0.0)
    full_tgt = df["norm_return"].values.astype(np.float64)
    for stride, label in [(1, "full stride=1"),
                          (60, "full stride=60")]:
        r = compute_val_nll(model, full_feat, full_tgt, stride, horizon_weight_decay)
        logger.info("  %-25s  %.4f  %.4f  %d windows",
                    label, r["nll"], r["mse"], r["windows"])


if __name__ == "__main__":
    main()
