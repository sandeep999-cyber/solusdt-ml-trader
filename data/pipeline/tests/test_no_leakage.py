"""Regression test: Verify no feature column at bar i uses data from bar > i.

Constructs a deterministic price series with known properties, runs the full
feature pipeline, then checks that each output row depends only on current
and past rows — never on future data.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tempfile
from pathlib import Path

from data.pipeline.features import compute_features


def _make_monotonic_dataset(n_bars: int = 1000) -> pd.DataFrame:
    """Create a strictly monotonic price series where every bar has a unique
    contribution that should NOT appear in earlier bars' features."""
    np.random.seed(42)
    base_price = 100.0
    closes = base_price + np.cumsum(np.random.randn(n_bars) * 0.1)
    closes = np.maximum(closes, 1.0)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_bars, freq="min", tz="UTC"),
        "open": closes * 0.9995,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": np.random.uniform(100, 1000, n_bars),
        "close_time": pd.date_range("2024-01-01", periods=n_bars, freq="min", tz="UTC")
                       + pd.Timedelta(seconds=59),
        "quote_volume": np.random.uniform(10000, 100000, n_bars),
        "trade_count": np.random.randint(10, 500, n_bars),
        "taker_buy_volume": np.random.uniform(40, 600, n_bars),
        "taker_buy_quote_volume": np.random.uniform(4000, 60000, n_bars),
        "ignore": 0,
    })
    return df


def _write_raw_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, dest / "data.parquet", compression="zstd")


FEATURE_COLS = [
    "cvd", "cvd_quote", "vwap_20", "vwap_50", "anchored_vwap",
    "realized_vol", "log_return", "norm_return", "return_pct",
    "vol_profile_low_bucket",
]


def test_no_feature_uses_future_data():
    """For every feature column, verify that row i depends only on rows <= i."""
    n = 500
    df = _make_monotonic_dataset(n)

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        src = base / "SOLUSDT" / "1m"
        out = base / "processed"
        _write_raw_parquet(df, src)

        # Run feature pipeline
        compute_features(
            parquet_dir=base,
            output_dir=out,
            symbol="SOLUSDT",
            interval="1m",
        )

        # Read back features
        result = pq.ParquetDataset(out).read().to_pandas()
        result = result.sort_values("timestamp").reset_index(drop=True)

    assert len(result) == n, f"Expected {n} rows, got {len(result)}"

    # Verify split-point consistency: features for rows <= k computed from the full
    # dataset must match features for the same rows computed from the truncated dataset.
    split = n // 2
    df_prefix = df.iloc[:split].copy()

    # Re-run pipeline on first half only
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        src_prefix = base / "SOLUSDT" / "1m"
        out_prefix = base / "processed"
        _write_raw_parquet(df_prefix, src_prefix)

        compute_features(
            parquet_dir=base,
            output_dir=out_prefix,
            symbol="SOLUSDT",
            interval="1m",
        )
        prefix_result = pq.ParquetDataset(out_prefix).read().to_pandas()
        prefix_result = prefix_result.sort_values("timestamp").reset_index(drop=True)

    # Compare the full-dataset features vs prefix-only features up to split
    full_prefix_result = result.iloc[:split].reset_index(drop=True)

    for col in FEATURE_COLS:
        for i in range(len(prefix_result)):
            v_full = full_prefix_result.iloc[i][col]
            v_prefix = prefix_result.iloc[i][col]
            if pd.isna(v_full) and pd.isna(v_prefix):
                continue
            if pd.isna(v_full) or pd.isna(v_prefix):
                if i < 50:
                    continue
                raise AssertionError(
                    f"{col}[{i}]: NaN mismatch: full={v_full}, prefix={v_prefix}"
                )
            if abs(v_full) > 1e-8:
                rel_err = abs(v_full - v_prefix) / abs(v_full)
                assert rel_err < 1e-5, (
                    f"{col}[{i}]: relative error {rel_err:.2e}"
                )
            else:
                abs_err = abs(v_full - v_prefix)
                assert abs_err < 1e-8, (
                    f"{col}[{i}]: absolute error {abs_err:.2e}"
                )


def test_all_ob_columns_are_nan():
    """Order-book columns are placeholders — must be all NaN."""
    df = _make_monotonic_dataset(100)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        src = base / "SOLUSDT" / "1m"
        out = base / "processed"
        _write_raw_parquet(df, src)
        compute_features(parquet_dir=base, output_dir=out)

        result = pq.ParquetDataset(out).read().to_pandas()

    for col in ["ob_imbalance", "ob_depth_bid_5", "ob_depth_ask_5", "ob_spread"]:
        assert result[col].isna().all(), f"{col} should be all NaN"


def test_no_empty_months():
    """Every month partition should have >0 rows."""
    n = 50000  # span multiple months
    df = _make_monotonic_dataset(n)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        src = base / "SOLUSDT" / "1m"
        out = base / "processed"
        _write_raw_parquet(df, src)
        compute_features(parquet_dir=base, output_dir=out)

        for parquet_file in out.rglob("features.parquet"):
            tbl = pq.read_table(parquet_file)
            assert tbl.num_rows > 0, f"Empty partition: {parquet_file}"
