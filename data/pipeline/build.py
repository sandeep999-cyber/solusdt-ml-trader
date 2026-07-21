#!/usr/bin/env python3
"""CLI: python -m data.pipeline.build --symbol SOLUSDT --interval 1m --start 2023-01-01 --end 2026-07-01"""

import argparse
import logging
import shutil
import sys
from datetime import date

from data.pipeline.config import (
    DEFAULT_INTERVAL,
    DEFAULT_SYMBOL,
    FEATURE_DIR,
    KLINE_ARCHIVE_DIR,
    PARQUET_DIR,
)
from data.pipeline.download import download_range
from data.pipeline.features import compute_features
from data.pipeline.parse import parse_all_archives

logger = logging.getLogger(__name__)


def _is_step_done(step_marker: str, symbol: str, interval: str) -> bool:
    """Check if a pipeline step has already been completed for this symbol/interval."""
    if step_marker == "download":
        archive_dir = KLINE_ARCHIVE_DIR / interval
        files = list(archive_dir.glob(f"{symbol}-{interval}-*.zip"))
        return len(files) > 0
    elif step_marker == "parse":
        parquet_dir = PARQUET_DIR / symbol / interval
        return parquet_dir.exists() and any(parquet_dir.rglob("*.parquet"))
    elif step_marker == "features":
        feat_dir = FEATURE_DIR / symbol / interval
        return feat_dir.exists() and any(feat_dir.rglob("features.parquet"))
    return False


def _run_pipeline(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    force: bool = False,
    skip_download: bool = False,
    skip_parse: bool = False,
    skip_features: bool = False,
) -> None:
    # Step 1: Download
    if skip_download:
        logger.info("Skipping download step")
    elif not force and _is_step_done("download", symbol, interval):
        logger.info("Download already complete for %s/%s", symbol, interval)
    else:
        logger.info("Step 1/3: Downloading archives %s -> %s", start, end)
        paths = download_range(symbol, interval, start, end, force=force)
        logger.info("Downloaded %d archives", len(paths))
        if not paths:
            logger.warning("No archives were downloaded — check symbol/interval/date range")

    # Step 2: Parse
    if skip_parse:
        logger.info("Skipping parse step")
    elif not force and _is_step_done("parse", symbol, interval):
        logger.info("Parse already complete for %s/%s", symbol, interval)
    else:
        if force:
            parquet_to_delete = PARQUET_DIR / symbol / interval
            if parquet_to_delete.exists():
                shutil.rmtree(parquet_to_delete)
                logger.info("Deleted existing parquet: %s", parquet_to_delete)
        logger.info("Step 2/3: Parsing archives -> Parquet")
        parse_all_archives(symbol, interval)
        logger.info("Parse complete")

    # Step 3: Features
    if skip_features:
        logger.info("Skipping feature computation step")
    elif not force and _is_step_done("features", symbol, interval):
        logger.info("Features already computed for %s/%s", symbol, interval)
    else:
        if force:
            feat_to_delete = FEATURE_DIR / symbol / interval
            if feat_to_delete.exists():
                shutil.rmtree(feat_to_delete)
                logger.info("Deleted existing features: %s", feat_to_delete)
        logger.info("Step 3/3: Computing features")
        compute_features(symbol=symbol, interval=interval)
        logger.info("Features complete")

    logger.info("Pipeline finished for %s/%s [%s -> %s]", symbol, interval, start, end)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Binance data pipeline: download -> parse -> features"
    )
    parser.add_argument(
        "--symbol", default=DEFAULT_SYMBOL, help=f"Trading pair (default: {DEFAULT_SYMBOL})"
    )
    parser.add_argument(
        "--interval", default=DEFAULT_INTERVAL, help=f"Candle interval (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--start", type=_parse_date, default=date(2023, 1, 1),
        help="Start date (YYYY-MM-DD, default: 2023-01-01)",
    )
    parser.add_argument(
        "--end", type=_parse_date, default=date(2026, 7, 1),
        help="End date (YYYY-MM-DD, default: 2026-07-01)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and re-process even if already done",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip download step",
    )
    parser.add_argument(
        "--skip-parse", action="store_true", help="Skip parse step",
    )
    parser.add_argument(
        "--skip-features", action="store_true", help="Skip feature computation",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    _run_pipeline(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        force=args.force,
        skip_download=args.skip_download,
        skip_parse=args.skip_parse,
        skip_features=args.skip_features,
    )


if __name__ == "__main__":
    main()
