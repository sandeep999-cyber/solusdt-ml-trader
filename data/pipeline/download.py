import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator, List, Tuple

import requests

from data.pipeline.config import (
    BASE_URL,
    CHECKSUM_SUFFIX,
    DATA_TYPE,
    DAILY_URL_T,
    KLINE_ARCHIVE_DIR,
    MONTHLY_URL_T,
)

logger = logging.getLogger(__name__)


def _month_range(start: date, end: date) -> Iterator[Tuple[int, int]]:
    """Yield (year, month) tuples covering [start, end]."""
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        yield cursor.year, cursor.month
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


def _daily_range(start: date, end: date) -> Iterator[Tuple[int, int, int]]:
    """Yield (year, month, day) tuples covering [start, end]."""
    cursor = start
    while cursor <= end:
        yield cursor.year, cursor.month, cursor.day
        cursor += timedelta(days=1)


def _checksum_url(archive_url: str) -> str:
    return archive_url + CHECKSUM_SUFFIX


def _expected_checksum(checksum_url: str) -> str:
    resp = requests.get(checksum_url, timeout=30)
    resp.raise_for_status()
    line = resp.text.strip().split("\n")[0]
    return line.split()[0]


def _local_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    logger.info("Downloading %s -> %s", url, dest)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)


def ensure_archive(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    day: int | None = None,
    force: bool = False,
) -> Path | None:
    """Download and verify a single archive. Returns path on success, None on failure."""
    archive_dir = KLINE_ARCHIVE_DIR / interval
    if day is not None:
        filename = f"{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
        url = DAILY_URL_T.format(
            base=BASE_URL, type=DATA_TYPE,
            symbol=symbol, interval=interval,
            year=year, month=month, day=day,
        )
    else:
        filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
        url = MONTHLY_URL_T.format(
            base=BASE_URL, type=DATA_TYPE,
            symbol=symbol, interval=interval,
            year=year, month=month,
        )

    dest = archive_dir / filename
    if dest.exists() and not force:
        try:
            cs_url = _checksum_url(url)
            expected = _expected_checksum(cs_url)
            actual = _local_checksum(dest)
            if actual == expected:
                logger.debug("Archive valid, skipping: %s", filename)
                return dest
            logger.warning("Checksum mismatch for %s, re-downloading", filename)
        except Exception:
            logger.warning("Could not verify %s, re-downloading", filename)

    try:
        _download_file(url, dest)
        cs_url = _checksum_url(url)
        expected = _expected_checksum(cs_url)
        actual = _local_checksum(dest)
        if actual != expected:
            logger.error("Checksum failed after download for %s", filename)
            dest.unlink(missing_ok=True)
            return None
        return dest
    except requests.HTTPError as e:
        if day is not None:
            logger.debug("Daily archive not found: %s — %s", filename, e)
        else:
            logger.warning("Monthly archive missing: %s — %s", filename, e)
        return None
    except Exception as e:
        logger.error("Failed to download %s: %s", filename, e)
        return None


def download_range(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    force: bool = False,
    max_workers: int = 4,
) -> List[Path]:
    """Download all archives for the date range. Returns list of valid archive paths."""
    paths: List[Path] = []

    # Try monthly archives first
    monthly_args = []
    today = date.today()
    for year, month in _month_range(start, end):
        if date(year, month, 1) > today.replace(day=1):
            break
        monthly_args.append((symbol, interval, year, month, None, force))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_key = {
            pool.submit(ensure_archive, *a[:4], day=a[4], force=a[5]): a
            for a in monthly_args
        }
        for fut in as_completed(fut_to_key):
            result = fut.result()
            if result is not None:
                paths.append(result)

    # Fall back to daily for most recent partial month where monthly succeeded partially
    downloaded_months = set()
    remaining_daily = []
    for p in paths:
        parts = p.stem.split("-")
        downloaded_months.add((int(parts[-2]), int(parts[-1])))

    for year, month in _month_range(start, end):
        if (year, month) not in downloaded_months:
            month_start = date(year, month, 1)
            month_end = date(year, month, 1)
            if month == 12:
                month_end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = date(year, month + 1, 1) - timedelta(days=1)
            daily_range_start = max(start, month_start)
            daily_range_end = min(end, month_end, today)
            for y, m, d in _daily_range(daily_range_start, daily_range_end):
                remaining_daily.append((symbol, interval, y, m, d, force))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_key = {
            pool.submit(ensure_archive, *a[:4], day=a[4], force=a[5]): a
            for a in remaining_daily
        }
        for fut in as_completed(fut_to_key):
            result = fut.result()
            if result is not None:
                paths.append(result)

    return sorted(paths)
