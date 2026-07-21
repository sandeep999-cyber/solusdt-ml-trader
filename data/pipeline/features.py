import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit

from data.pipeline.config import FEATURE_DIR, PARQUET_DIR, VERSION

logger = logging.getLogger(__name__)


@njit
def _cumulative_cvd_jit(
    taker_buy: np.ndarray, total_volume: np.ndarray
) -> np.ndarray:
    """Compute cumulative volume delta (CVD) from bar-level data."""
    n = len(taker_buy)
    cvd = np.empty(n, dtype=np.float64)
    running = 0.0
    for i in range(n):
        taker_sell = total_volume[i] - taker_buy[i]
        running += taker_buy[i] - taker_sell
        cvd[i] = running
    return cvd


@njit
def _rolling_vwap_jit(
    price: np.ndarray, volume: np.ndarray, window: int
) -> np.ndarray:
    """Rolling VWAP using typical price = (H+L+C)/3 via input price."""
    n = len(price)
    vwap = np.empty(n, dtype=np.float64)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        cum_pv += price[i] * volume[i]
        cum_v += volume[i]
        if i >= window:
            j = i - window
            cum_pv -= price[j] * volume[j]
            cum_v -= volume[j]
        vwap[i] = cum_pv / cum_v if cum_v > 0 else price[i]
    return vwap


@njit
def _realized_vol_jit(
    log_returns: np.ndarray, window: int
) -> np.ndarray:
    """Rolling standard deviation of log returns."""
    n = len(log_returns)
    vol = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        s = 0.0
        mean = 0.0
        for j in range(i - window + 1, i + 1):
            mean += log_returns[j]
        mean /= window
        for j in range(i - window + 1, i + 1):
            d = log_returns[j] - mean
            s += d * d
        vol[i] = np.sqrt(s / (window - 1))
    return vol


@njit
def _volume_profile_jit(
    price: np.ndarray, volume: np.ndarray, window: int, n_buckets: int
) -> np.ndarray:
    """Volume profile: fraction of window volume in the lowest price bucket."""
    n = len(price)
    frac_low = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        chunk_p = price[i - window : i]
        chunk_v = volume[i - window : i]
        low = chunk_p.min()
        high = chunk_p.max()
        span = high - low
        if span < 1e-10:
            frac_low[i] = 0.0
            continue
        bucket_size = span / n_buckets
        bucket_volumes = np.zeros(n_buckets, dtype=np.float64)
        for j in range(window):
            idx = min(int((chunk_p[j] - low) / bucket_size), n_buckets - 1)
            bucket_volumes[idx] += chunk_v[j]
        total = bucket_volumes.sum()
        frac_low[i] = bucket_volumes[0] / total if total > 0 else 0.0
    return frac_low


def compute_features(
    parquet_dir: Path | None = None,
    output_dir: Path | None = None,
    symbol: str = "SOLUSDT",
    interval: str = "1m",
    vwap_windows: tuple[int, ...] = (20, 50),
    vol_window: int = 20,
    profile_window: int = 50,
    profile_buckets: int = 10,
) -> Path:
    """Read parquet dataset, compute features, write processed output."""
    src = (parquet_dir or PARQUET_DIR) / symbol / interval
    if not src.exists():
        raise FileNotFoundError(f"Parquet directory not found: {src}")

    dataset = pq.ParquetDataset(src)
    table = dataset.read()
    df = table.to_pandas()
    logger.info("Loaded %d rows from %s", len(df), src)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- CVD ---
    price = (df["high"].values + df["low"].values + df["close"].values) / 3.0
    log_ret = np.full(len(df), 0.0, dtype=np.float64)
    log_ret[1:] = np.log(df["close"].values[1:] / df["close"].values[:-1])
    log_ret[0] = 0.0

    df["cvd"] = _cumulative_cvd_jit(
        df["taker_buy_volume"].values, df["volume"].values
    )
    df["cvd_quote"] = _cumulative_cvd_jit(
        df["taker_buy_quote_volume"].values, df["quote_volume"].values
    )

    # --- VWAP ---
    for w in vwap_windows:
        df[f"vwap_{w}"] = _rolling_vwap_jit(price, df["volume"].values, w)

    # Anchored VWAP — anchored to the start of each month
    df["anchor"] = df["timestamp"].dt.tz_localize(None).dt.to_period("M")
    df["anchored_vwap"] = np.nan
    for anchor, group in df.groupby("anchor"):
        idx = group.index
        cum_pv = (group["close"] * group["volume"]).cumsum()
        cum_v = group["volume"].cumsum()
        df.loc[idx, "anchored_vwap"] = (cum_pv / cum_v).values

    # --- Realized volatility ---
    df["realized_vol"] = _realized_vol_jit(log_ret, vol_window)

    # --- Returns relative to volatility ---
    df["log_return"] = log_ret
    eps = 1e-10
    df["norm_return"] = df["log_return"] / (df["realized_vol"] + eps)

    # --- Raw return % for convenience ---
    df["return_pct"] = df["close"].pct_change()

    # --- Volume Profile ---
    df["vol_profile_low_bucket"] = _volume_profile_jit(
        price, df["volume"].values, profile_window, profile_buckets
    )

    # --- Placeholder for order-book features (filled later from sol-recorder) ---
    df["ob_imbalance"] = np.nan
    df["ob_depth_bid_5"] = np.nan
    df["ob_depth_ask_5"] = np.nan
    df["ob_spread"] = np.nan

    # --- Drop intermediate columns ---
    drop_cols = ["anchor", "ignore"] if "ignore" in df.columns else ["anchor"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # --- Write output ---
    out = (output_dir or FEATURE_DIR) / symbol / interval
    out.mkdir(parents=True, exist_ok=True)

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    for (year, month), group in df.groupby(["year", "month"]):
        part_path = out / f"year={year}" / f"month={month:02d}" / f"features.parquet"
        part_path.parent.mkdir(parents=True, exist_ok=True)

        cols = [c for c in group.columns if c not in ("year", "month")]
        tbl = pa.Table.from_pandas(group[cols], preserve_index=False)
        pq.write_table(tbl, part_path, compression="zstd")

        logger.debug("Wrote %d feature rows to %s", len(group), part_path)

    # Write version marker
    (out / ".version").write_text(VERSION)

    logger.info("Feature pipeline complete for %s/%s -> %s", symbol, interval, out)
    return out
