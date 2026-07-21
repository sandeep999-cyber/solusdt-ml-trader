#!/usr/bin/env python3
"""Generate sample feature parquet files for UI development.
Writes to data/processed/sample/v1/SOLUSDT/1m/ in partitioned format.
Never touches the live feature path (data/processed/v1/).
"""

import logging
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
LIVE_FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "v1"
SAMPLE_DIR = PROJECT_ROOT / "data" / "processed" / "sample"
VERSION = "v1"
FEATURE_DIR = SAMPLE_DIR / VERSION

RNG = np.random.default_rng(42)


def _price_series(n: int, base_price: float = 100.0, drift: float = 0.00002, vol: float = 0.002) -> np.ndarray:
    prices = np.empty(n, dtype=np.float64)
    prices[0] = base_price
    for i in range(1, n):
        ret = drift + vol * RNG.normal()
        prices[i] = prices[i - 1] * (1 + ret)
    return prices


def _ohlc_from_prices(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(prices)
    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    for i in range(n):
        opens[i] = prices[i - 1] if i > 0 else prices[0] * 0.999
        spread = RNG.uniform(0.001, 0.005) * prices[i]
        highs[i] = max(prices[i], opens[i]) + RNG.uniform(0, spread)
        lows[i] = min(prices[i], opens[i]) - RNG.uniform(0, spread)
    closes = prices
    return opens, highs, lows


def generate_features(
    symbol: str = "SOLUSDT",
    interval: str = "1m",
    n_bars: int = 10080,
    base_price: float = 100.0,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    logger.info("Generating %d bars of %s for %s starting %s", n_bars, interval, symbol, start_date)
    prices = _price_series(n_bars, base_price)
    opens, highs, lows = _ohlc_from_prices(prices)
    closes = prices
    typical = (highs + lows + closes) / 3.0

    # Volume: correlated with volatility
    log_ret = np.zeros(n_bars)
    log_ret[1:] = np.log(closes[1:] / closes[:-1])
    rolling_vol = np.full(n_bars, 0.002)
    for i in range(20, n_bars):
        rolling_vol[i] = np.std(log_ret[i - 19 : i + 1])
    amplitude = highs - lows
    base_vol = 10000.0
    volumes = base_vol * (1 + amplitude / closes * 100) * (1 + 0.3 * RNG.uniform(size=n_bars))

    taker_buy_frac = 0.45 + 0.1 * np.sin(np.arange(n_bars) * 0.01) + 0.03 * RNG.uniform(size=n_bars)
    taker_buy_frac = np.clip(taker_buy_frac, 0.3, 0.7)
    taker_buy_vol = volumes * taker_buy_frac

    quote_volumes = volumes * typical
    taker_buy_quote = taker_buy_vol * typical

    timestamps = pd.date_range(
        start=datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc),
        periods=n_bars,
        freq="1min",
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "quote_volume": quote_volumes,
        "trade_count": np.random.poisson(lam=100, size=n_bars).astype(np.int64),
        "taker_buy_volume": taker_buy_vol,
        "taker_buy_quote_volume": taker_buy_quote,
    })

    # --- CVD ---
    cvd = np.empty(n_bars)
    running = 0.0
    for i in range(n_bars):
        taker_sell = volumes[i] - taker_buy_vol[i]
        running += taker_buy_vol[i] - taker_sell
        cvd[i] = running
    df["cvd"] = cvd

    cvd_q = np.empty(n_bars)
    running_q = 0.0
    for i in range(n_bars):
        taker_sell_q = quote_volumes[i] - taker_buy_quote[i]
        running_q += taker_buy_quote[i] - taker_sell_q
        cvd_q[i] = running_q
    df["cvd_quote"] = cvd_q

    # --- VWAP ---
    for w in [20, 50]:
        vwap = np.empty(n_bars)
        cum_pv = 0.0
        cum_v = 0.0
        for i in range(n_bars):
            cum_pv += typical[i] * volumes[i]
            cum_v += volumes[i]
            if i >= w:
                j = i - w
                cum_pv -= typical[j] * volumes[j]
                cum_v -= volumes[j]
            vwap[i] = cum_pv / cum_v if cum_v > 0 else typical[i]
        df[f"vwap_{w}"] = vwap

    # Anchored VWAP
    df["anchor"] = df["timestamp"].dt.tz_localize(None).dt.to_period("M")
    df["anchored_vwap"] = np.nan
    for anchor, group in df.groupby("anchor"):
        idx = group.index
        cum_pv = (group["close"] * group["volume"]).cumsum()
        cum_v = group["volume"].cumsum()
        df.loc[idx, "anchored_vwap"] = (cum_pv / cum_v).values

    # Returns
    df["log_return"] = log_ret
    df["return_pct"] = df["close"].pct_change()
    eps = 1e-10
    df["norm_return"] = df["log_return"] / (rolling_vol + eps)

    # Realized vol
    df["realized_vol"] = rolling_vol

    # Volume profile (simplified)
    vol_profile = np.full(n_bars, np.nan)
    for i in range(50, n_bars):
        chunk_p = typical[i - 50 : i]
        chunk_v = volumes[i - 50 : i]
        low = chunk_p.min()
        high = chunk_p.max()
        span = high - low
        if span < 1e-10:
            vol_profile[i] = 0.0
            continue
        bucket_size = span / 10
        buckets = np.zeros(10)
        for j in range(50):
            idx = min(int((chunk_p[j] - low) / bucket_size), 9)
            buckets[idx] += chunk_v[j]
        total = buckets.sum()
        vol_profile[i] = buckets[0] / total if total > 0 else 0.0
    df["vol_profile_low_bucket"] = vol_profile

    # OB features (placeholders)
    df["ob_imbalance"] = np.nan
    df["ob_depth_bid_5"] = np.nan
    df["ob_depth_ask_5"] = np.nan
    df["ob_spread"] = np.nan

    df.drop(columns=["anchor"], inplace=True)
    return df


def write_partitioned(df: pd.DataFrame, symbol: str, interval: str) -> None:
    out = FEATURE_DIR / symbol / interval
    logger.info("Writing to %s", out)

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    for (year, month), group in df.groupby(["year", "month"]):
        part_path = out / f"year={year}" / f"month={month:02d}" / "features.parquet"
        part_path.parent.mkdir(parents=True, exist_ok=True)

        cols = [c for c in group.columns if c not in ("year", "month")]
        tbl = pa.Table.from_pandas(group[cols], preserve_index=False)
        pq.write_table(tbl, part_path, compression="zstd")
        logger.debug("Wrote %d rows to %s", len(group), part_path)

    (out / ".version").write_text(VERSION)
    logger.info("Done — %d bars written to %s", len(df), out)


def main():
    if FEATURE_DIR == LIVE_FEATURE_DIR:
        raise RuntimeError(
            "Refusing to write sample data into the live feature path "
            f"({LIVE_FEATURE_DIR}). Set FEATURE_DIR to a sample subdirectory."
        )
    df = generate_features(n_bars=10080)  # 7 days of 1m data
    write_partitioned(df, "SOLUSDT", "1m")


if __name__ == "__main__":
    main()
