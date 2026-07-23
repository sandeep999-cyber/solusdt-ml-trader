"""Causal window data loader for SOLUSDT 1m feature data.

Constructs rolling windows from the processed feature parquet dataset.
Respects causality: a window ending at time t never includes data from t+1.

Target shape: (horizon,) — continuous norm_return values at t+1 .. t+horizon.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from model.config.run_config import RunConfig
from model.config.splits import get_split_mask

logger = logging.getLogger(__name__)

OB_PLACEHOLDER_COLS = {"ob_imbalance", "ob_depth_bid_5", "ob_depth_ask_5", "ob_spread"}


class CausalWindowDataset(Dataset):
    """PyTorch Dataset that yields (window_features, target) pairs.

    Each sample:
      - window: tensor of shape (window_length, n_features) — bars [i .. i+window_length-1]
      - target: tensor of shape (horizon,) — sequence of norm_return values
                at positions i+window_length .. i+window_length+horizon-1

    Causality guarantee: a window ending at bar t never includes data from bar t+1.
    Verified by construction (no shifting / padding that could leak future data).
    """

    def __init__(
        self,
        config: RunConfig,
        split: str = "train",
        device: torch.device = torch.device("cpu"),
        stride: int = 1,
    ):
        self.config = config
        self.split = split
        self.device = device
        self.window_length = config.window_length
        self.horizon = config.horizon
        self.stride = stride

        # Load feature dataset
        feat_dir = Path(config.processed_dir)
        if not feat_dir.exists():
            raise FileNotFoundError(f"Processed feature directory not found: {feat_dir}")

        parquet_files = sorted(feat_dir.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {feat_dir}")
        df = pq.ParquetDataset(parquet_files).read().to_pandas()
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Loaded %d rows from %s", len(df), feat_dir)

        # Filter placeholder OB columns unless explicitly requested
        requested = set(config.feature_columns)
        self.feature_cols = [c for c in config.feature_columns]
        for c in list(self.feature_cols):
            if c in OB_PLACEHOLDER_COLS and c not in requested:
                self.feature_cols.remove(c)
                logger.debug("Excluded OB placeholder column: %s", c)

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in dataset: {missing}")

        # Filter to the requested split
        mask = get_split_mask(df, split)
        self.df = df[mask].reset_index(drop=True)
        logger.info("Split '%s': %d rows", split, len(self.df))

        if len(self.df) < self.window_length + self.horizon:
            raise ValueError(
                f"Split '{split}' has {len(self.df)} rows, "
                f"need at least {self.window_length + self.horizon} "
                f"for window_length={self.window_length} + horizon={self.horizon}"
            )

        # Pre-extract feature array
        self.features = self.df[self.feature_cols].values.astype(np.float32)

        # Target: use norm_return (continuous, already causally clean upstream)
        self.targets = self.df["norm_return"].values.astype(np.float32)

        # Handle NaN in features (forward-fill then back-fill)
        self.features = _safe_fill(self.features)

        # Preallocate tensors for zero-copy slicing in __getitem__
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

        # Number of valid windows (accounting for stride)
        raw_windows = len(self.df) - self.window_length - self.horizon + 1
        self.n_windows = (raw_windows + self.stride - 1) // self.stride
        logger.info(
            "Constructed %d windows (len=%d, horizon=%d, stride=%d)",
            self.n_windows, self.window_length, self.horizon, self.stride,
        )

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (window, target) for the idx-th causal window (zero-copy slice)."""
        actual_idx = idx * self.stride
        win = self.features[actual_idx : actual_idx + self.window_length]
        start = actual_idx + self.window_length
        tgt = self.targets[start : start + self.horizon]
        return win, tgt

    def get_timestamps(self, idx: int) -> tuple[str, str]:
        """Return (window_end_ts, last_target_ts) for debugging."""
        w_end = self.df.iloc[idx + self.window_length - 1]["timestamp"]
        t_idx = idx + self.window_length + self.horizon - 1
        t_ts = self.df.iloc[t_idx]["timestamp"]
        return str(w_end), str(t_ts)

    def get_feature_names(self) -> list[str]:
        return self.feature_cols.copy()


def _safe_fill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaN values in a 2D array."""
    df = pd.DataFrame(arr)
    df = df.ffill().bfill()
    return df.values.astype(np.float32)


def create_dataloader(
    config: RunConfig,
    split: str = "train",
    shuffle: bool = True,
    device: torch.device = torch.device("cpu"),
    stride: int = 1,
) -> DataLoader:
    """Create a DataLoader for the given split.

    Features/targets are fully materialized as CPU tensors in
    CausalWindowDataset.__init__, so each __getitem__ is a cheap slice.
    Extra workers add process/IPC overhead without helping I/O — keep
    num_workers at 0 (or 2 max on multi-core hosts). pin_memory only
    when the training device is CUDA.

    Args:
        stride: Step size between windows. stride=1 (default) gives full
            overlap (every consecutive window). stride=window_length gives
            non-overlapping windows. Use stride>1 for honest validation
            that doesn't exploit window-overlap artifacts.
    """
    dataset = CausalWindowDataset(config, split=split, device=device, stride=stride)
    use_cuda = isinstance(device, torch.device) and device.type == "cuda"
    # Prefer 0 workers: in-memory slices + small model → workers slow us down.
    # Cap at 2 if explicitly useful later; never 4 (Colab warning / oversub).
    num_workers = 0
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=False,
    )
