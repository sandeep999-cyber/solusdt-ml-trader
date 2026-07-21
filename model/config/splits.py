"""Chronological train/validation/test splits for SOLUSDT 1m Phase A.

Splits are defined as [start, end) half-open intervals on the timestamp column.
All dates are UTC.

Rationale:
  - Train:     2023-01-01 – 2024-09-01  (20 months, ~876K bars)
  - Validation: 2024-09-01 – 2024-11-01  (2 months, ~88K bars)
  - Test:       2024-11-01 – 2025-01-01  (2 months, ~88K bars)

The validation and test sets each begin at a month boundary so that
anchored VWAP resets naturally. The split is strictly chronological:
no future data leaks into earlier splits (verified by leakage audit).

History note: training was extended backwards from 2024-01-01 to
2023-01-01 (2023 backfill, 2026-07-20) so the model sees more than one
market regime. Validation and test windows were deliberately left
untouched so the holdout stays meaningful across that change.
"""

import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    name: str
    start: str  # ISO-8601, inclusive
    end: str    # ISO-8601, exclusive


SPLITS = [
    Split("train", "2023-01-01", "2024-09-01"),
    Split("val",   "2024-09-01", "2024-11-01"),
    Split("test",  "2024-11-01", "2025-01-01"),
]

SPLIT_MAP = {s.name: s for s in SPLITS}


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'split' column to the dataframe based on the timestamp."""
    ts = df["timestamp"]
    df = df.copy()
    df["split"] = "none"
    for s in SPLITS:
        start = pd.Timestamp(s.start, tz="UTC")
        end = pd.Timestamp(s.end, tz="UTC")
        mask = (ts >= start) & (ts < end)
        df.loc[mask, "split"] = s.name
    return df


def get_split_mask(df: pd.DataFrame, name: str) -> pd.Series:
    """Return boolean mask for rows belonging to the named split."""
    s = SPLIT_MAP[name]
    ts = df["timestamp"]
    start = pd.Timestamp(s.start, tz="UTC")
    end = pd.Timestamp(s.end, tz="UTC")
    return (ts >= start) & (ts < end)


def split_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return row counts per split."""
    return df.groupby("split").size().to_dict()
