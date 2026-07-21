from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
BINANCE_DIR = RAW_DIR / "binance"
KLINE_ARCHIVE_DIR = BINANCE_DIR / "klines"
PARQUET_DIR = RAW_DIR / "parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VERSION = "v1"
FEATURE_DIR = PROCESSED_DIR / VERSION

BASE_URL = "https://data.binance.vision"
DATA_TYPE = "spot"

KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

DTYPES = {
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "quote_volume": "float64",
    "trade_count": "int64",
    "taker_buy_volume": "float64",
    "taker_buy_quote_volume": "float64",
}

MONTHLY_URL_T = (
    "{base}/data/{type}/monthly/klines/{symbol}/{interval}/"
    "{symbol}-{interval}-{year}-{month:02d}.zip"
)
DAILY_URL_T = (
    "{base}/data/{type}/daily/klines/{symbol}/{interval}/"
    "{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
)
CHECKSUM_SUFFIX = ".CHECKSUM"

DEFAULT_INTERVAL = "1m"
DEFAULT_SYMBOL = "SOLUSDT"
