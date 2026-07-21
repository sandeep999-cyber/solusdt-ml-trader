"""Tests for data/pipeline/parse.py — idempotent re-parse under build.py's --force."""

import shutil
import tempfile
from pathlib import Path

import pytest
import pyarrow.parquet as pq

from data.pipeline.parse import archive_to_parquet

SAMPLE_ARCHIVE = Path("data/raw/binance/klines/1m/SOLUSDT-1m-2024-01.zip")


def test_force_reparse_does_not_duplicate():
    """Simulate build.py's --force: delete target dir before re-parsing."""
    if not SAMPLE_ARCHIVE.exists():
        pytest.skip("Sample archive not found; skipping")
    with tempfile.TemporaryDirectory() as tmp:
        parquet_dir = Path(tmp)
        # Parse once
        out = archive_to_parquet(SAMPLE_ARCHIVE, "SOLUSDT", "1m", parquet_dir=parquet_dir)
        assert out is not None
        n1 = pq.ParquetDataset(out).read().num_rows

        # Delete target to simulate --force
        shutil.rmtree(out)
        out2 = archive_to_parquet(SAMPLE_ARCHIVE, "SOLUSDT", "1m", parquet_dir=parquet_dir)
        assert out2 is not None
        combined = pq.ParquetDataset(out2).read().to_pandas()
        assert len(combined) == n1, f"Expected {n1} rows after force-re-parse, got {len(combined)}"
        assert combined["timestamp"].duplicated().sum() == 0
