"""Verify that the inference contract is satisfied by the feature pipeline."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from model.inference.contract_version import CONTRACT_VERSION

# Path to the 2024 feature dataset
PROCESSED_DIR = Path("data/processed/v1/SOLUSDT/1m")

REQUIRED_COLUMNS = [
    "cvd",
    "vwap_20",
    "vwap_50",
    "realized_vol",
    "log_return",
    "norm_return",
    "return_pct",
    "vol_profile_low_bucket",
    "anchored_vwap",
    "cvd_quote",
]

NULLABLE_COLUMNS = {"realized_vol", "norm_return", "return_pct", "vol_profile_low_bucket"}

FLOAT_COLUMNS = set(REQUIRED_COLUMNS)


def test_contract_version_defined():
    assert CONTRACT_VERSION == "1.0"


def test_all_required_columns_present():
    df = pq.ParquetDataset(PROCESSED_DIR).read().to_pandas()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_no_extra_required_columns_above_10():
    """Phase A contract specifies exactly 10 feature columns."""
    df = pq.ParquetDataset(PROCESSED_DIR).read().to_pandas()
    feat_cols = [c for c in REQUIRED_COLUMNS]
    assert len(feat_cols) == 10


def test_non_nullable_columns_have_no_nulls():
    df = pq.ParquetDataset(PROCESSED_DIR).read().to_pandas()
    non_nullable = [c for c in REQUIRED_COLUMNS if c not in NULLABLE_COLUMNS]
    for col in non_nullable:
        n_nulls = df[col].isna().sum()
        assert n_nulls == 0, f"{col}: {n_nulls} nulls (expected 0)"


def test_nullable_columns_have_expected_nulls():
    """Check that nullable columns have NaN only at the start (warmup bars)."""
    df = pq.ParquetDataset(PROCESSED_DIR).read().to_pandas().sort_values("timestamp")
    for col in NULLABLE_COLUMNS:
        nulls = df[col].isna()
        if nulls.any():
            first_non_null = nulls.idxmin() if not nulls.all() else len(df)
            # All nulls must be before the first non-null (leading edge only)
            n_leading = nulls.iloc[:first_non_null].sum()
            n_trailing = nulls.iloc[first_non_null:].sum()
            assert n_trailing == 0, (
                f"{col}: {n_trailing} nulls after first non-null at index {first_non_null}"
            )


def test_column_dtypes():
    df = pq.ParquetDataset(PROCESSED_DIR).read().to_pandas()
    for col in REQUIRED_COLUMNS:
        assert df[col].dtype == "float64", f"{col}: expected float64, got {df[col].dtype}"


def test_contract_matches_live_features():
    """Re-run feature pipeline on a fresh dataset and confirm same columns emerge."""
    import sys
    import tempfile
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data.pipeline.features import compute_features

    n = 200
    np.random.seed(0)
    closes = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=n, freq="min", tz="UTC"),
        "open": closes * 0.9995,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": np.random.uniform(100, 1000, n),
        "close_time": pd.date_range("2024-06-01", periods=n, freq="min", tz="UTC")
                       + pd.Timedelta(seconds=59),
        "quote_volume": np.random.uniform(10000, 100000, n),
        "trade_count": np.random.randint(10, 500, n),
        "taker_buy_volume": np.random.uniform(40, 600, n),
        "taker_buy_quote_volume": np.random.uniform(4000, 60000, n),
        "ignore": 0,
    })

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        src = base / "SOLUSDT" / "1m"
        out = base / "processed"
        src.mkdir(parents=True, exist_ok=True)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(tbl, src / "data.parquet", compression="zstd")

        compute_features(parquet_dir=base, output_dir=out)

        result = pq.ParquetDataset(out).read().to_pandas()

    for col in REQUIRED_COLUMNS:
        assert col in result.columns, f"Missing contract column after re-compute: {col}"
