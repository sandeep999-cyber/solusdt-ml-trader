"""Compute the persistence baseline under the Gaussian NLL metric.

Model: predicted mean = 0 (no change), predicted variance = realized_vol
       (clamped to EPS to prevent degenerate NLL from near-zero vol).

Baseline NLL represents "NLL per prediction" — averaged over batch
and horizon dimensions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from model.config.splits import get_split_mask

logger = logging.getLogger(__name__)

EPS = 1e-4  # minimum realized_vol floor (well below the 1st percentile)


def compute_baseline(
    data_path: str = "data/processed/v1/SOLUSDT/1m",
    window_length: int = 60,
    horizon: int = 12,
    split: str = "test",
) -> dict:
    df = pq.ParquetDataset(data_path).read().to_pandas()
    df = df.sort_values("timestamp").reset_index(drop=True)

    mask = get_split_mask(df, split)
    split_df = df[mask].reset_index(drop=True)
    logger.info("Split '%s': %d rows", split, len(split_df))

    n_windows = len(split_df) - window_length - horizon + 1
    if n_windows <= 0:
        raise ValueError(f"Not enough rows ({len(split_df)}) for window={window_length}+horizon={horizon}")

    # Compute the unconditional variance of norm_return over this split
    nr = split_df["norm_return"].dropna()
    pred_var = float(nr.var())
    logger.info("norm_return variance over split: %.6f", pred_var)

    nlls = []
    for i in range(n_windows):
        # Predicted distribution: mean=0, variance = norm_return's unconditional variance
        # norm_return = log_return / realized_vol, so its scale is ~N(0,1) regardless
        # of the current realized_vol.  The unconditional variance is the right
        # constant baseline.
        log_var = np.log(pred_var)
        var = pred_var

        # Targets: norm_return at t+1 .. t+horizon
        targets = split_df["norm_return"].values[i + window_length : i + window_length + horizon]

        # Gaussian NLL per step, mean=0
        #   NLL_i = 0.5 * (log_var + target_i^2 / var)
        per_step = 0.5 * (log_var + (targets ** 2) / var)  # (horizon,)
        nll = per_step.mean()  # average over horizon
        nlls.append(nll)

    mean_nll = float(np.mean(nlls))
    logger.info("Baseline NLL = %.6f over %d windows (%.6f per prediction)", mean_nll, len(nlls), mean_nll)

    return {
        "metric_version": "gaussian_nll_v1",
        "task": "regression_with_uncertainty",
        "prediction": f"mean=0, var=norm_return_var ({pred_var:.4f})",
        "split": split,
        "nll": round(mean_nll, 6),
        "n_samples": len(nlls),
        "window_length": window_length,
        "horizon": horizon,
        "date_computed": "2026-07-20",
        "method": (
            "persistence: mean=0, variance=unconditional norm_return variance "
            "(computed over the split).  norm_return = log_return / realized_vol "
            "has variance~1, so realized_vol is not the right variance scale. "
            "NLL is mean over batch AND horizon — each prediction contributes equally. "
            "No clamping needed because the unconditional variance is always well-behaved."
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = compute_baseline()

    out_path = Path(__file__).resolve().parent / "persistence_2024.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
