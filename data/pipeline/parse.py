import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.pipeline.config import (
    DTYPES,
    KLINE_COLUMNS,
    KLINE_ARCHIVE_DIR,
    PARQUET_DIR,
)

logger = logging.getLogger(__name__)


def _parse_timestamp(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def parse_archive(archive_path: Path, symbol: str, interval: str) -> pd.DataFrame | None:
    """Parse a single Binance kline zip archive into a DataFrame."""
    try:
        df = pd.read_csv(
            archive_path,
            header=None,
            names=KLINE_COLUMNS,
            dtype={k: v for k, v in DTYPES.items() if k in KLINE_COLUMNS},
        )
    except Exception as e:
        logger.error("Failed to parse %s: %s", archive_path.name, e)
        return None

    df["timestamp"] = df["timestamp"].apply(_parse_timestamp)
    df["close_time"] = df["close_time"].apply(_parse_timestamp)
    df.drop(columns=["ignore"], inplace=True)

    df["symbol"] = symbol
    df["interval"] = interval

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def _run_sanity_checks(df: pd.DataFrame, label: str) -> None:
    """Run and log sanity checks on parsed data."""
    dups = df["timestamp"].duplicated().sum()
    if dups:
        logger.warning("%s: %d duplicate timestamps found", label, dups)

    if not df["timestamp"].is_monotonic_increasing:
        logger.warning("%s: timestamp index is not monotonically increasing", label)

    violations = (df["taker_buy_volume"] > df["volume"]).sum()
    if violations:
        logger.warning(
            "%s: %d rows where taker_buy_volume > volume", label, violations
        )

    if len(df) >= 2:
        gap_minutes = (
            df["timestamp"].diff().dropna().dt.total_seconds() / 60
        )
        expected = int(df["interval"].iloc[0].rstrip("m")) if df["interval"].iloc[0].endswith("m") else 1
        gaps = (gap_minutes > expected * 1.5).sum()
        if gaps:
            logger.warning("%s: %d gaps > %.0f minutes detected", label, gaps, expected * 1.5)

    logger.info(
        "%s: %d rows, %s -> %s",
        label, len(df),
        df["timestamp"].min(), df["timestamp"].max(),
    )


def archive_to_parquet(
    archive_path: Path,
    symbol: str,
    interval: str,
    parquet_dir: Path | None = None,
) -> Path | None:
    """Parse one archive and append to the per-month parquet dataset. Returns parquet path."""
    df = parse_archive(archive_path, symbol, interval)
    if df is None:
        return None

    label = f"{symbol}/{interval}/{archive_path.stem}"
    _run_sanity_checks(df, label)

    out_dir = (parquet_dir or PARQUET_DIR) / symbol / interval
    out_dir.mkdir(parents=True, exist_ok=True)

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    for (year, month), group in df.groupby(["year", "month"]):
        part_path = out_dir / f"year={year}" / f"month={month:02d}" / f"data.parquet"
        part_path.parent.mkdir(parents=True, exist_ok=True)

        cols = [c for c in group.columns if c not in ("symbol", "interval", "year", "month")]
        table = pa.Table.from_pandas(group[cols], preserve_index=False)

        if part_path.exists():
            existing = pq.read_table(part_path)
            combined = pa.concat_tables([existing, table])
            combined = combined.combine_chunks()
            pq.write_table(combined, part_path, compression="zstd")
        else:
            pq.write_table(table, part_path, compression="zstd")

        logger.debug("Wrote %d rows to %s", len(group), part_path)

    return out_dir


def parse_all_archives(
    symbol: str,
    interval: str,
    archive_dir: Path | None = None,
) -> Path:
    """Parse all archives for a symbol/interval into partitioned parquet."""
    src_dir = (archive_dir or KLINE_ARCHIVE_DIR) / interval
    archives = sorted(src_dir.glob(f"{symbol}-{interval}-*.zip"))
    if not archives:
        raise FileNotFoundError(f"No archives found in {src_dir} for {symbol}/{interval}")

    logger.info("Parsing %d archives for %s/%s", len(archives), symbol, interval)
    for ap in archives:
        archive_to_parquet(ap, symbol, interval)
    logger.info("Parse complete for %s/%s", symbol, interval)

    return PARQUET_DIR / symbol / interval
