from fastapi import APIRouter, HTTPException, Query

from ui.backend.state import get_state, reload_state
from model.inference.engine import get_latest_checkpoint

router = APIRouter()


@router.get("/series")
async def get_series(
    symbol: str = Query("SOLUSDT"),
    start: str | None = None,
    end: str | None = None,
    limit: int | None = Query(500, description="Max bars to return (default: 500)"),
):
    state = get_state()
    df = state["df"]
    engine = state["engine"]

    if df.empty:
        raise HTTPException(404, "No data loaded")

    # Filter by range
    if start or end:
        ts = df["timestamp"]
        mask = True
        if start:
            mask = mask & (ts >= start)
        if end:
            mask = mask & (ts <= end)
        subset = df[mask]
    else:
        subset = df

    # Apply limit from the end (most recent bars)
    if limit is not None and limit < len(subset):
        subset = subset.iloc[-limit:]

    # Return OHLCV + core features
    cols = [
        "timestamp", "open", "high", "low", "close",
        "volume", "quote_volume", "trade_count",
        "cvd", "cvd_quote", "vwap_20", "vwap_50",
        "anchored_vwap", "realized_vol", "log_return",
        "norm_return", "return_pct", "vol_profile_low_bucket",
    ]
    available = [c for c in cols if c in subset.columns]
    bars = subset[available].to_dict(orient="records")

    # JSON-safe: replace NaN/Inf with None
    import math
    def _safe(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    for bar in bars:
        for k, v in bar.items():
            bar[k] = _safe(v)

    return {
        "symbol": symbol,
        "interval": "1m",
        "bar_count": len(bars),
        "bars": bars,
    }


def _clean_ts(ts: str) -> str:
    """Normalize a timestamp query parameter — take just the datetime portion."""
    # Strip timezone offset: "2024-01-01T01:00:00+00:00" or "2024-01-01T01:00:00 00:00"
    return ts[:19].replace("T", " ")


@router.get("/inference")
async def get_inference(
    symbol: str = Query("SOLUSDT"),
    timestamp: str | None = None,
):
    state = get_state()
    engine = state["engine"]

    if timestamp:
        timestamp = _clean_ts(timestamp)
        result = engine.get_by_timestamp(timestamp)
    else:
        result = engine.get_latest()

    if result is None:
        raise HTTPException(404, f"No inference at timestamp={timestamp}")

    return result


@router.get("/inference/range")
async def get_inference_range(
    symbol: str = Query("SOLUSDT"),
    start: str | None = None,
    end: str | None = None,
    limit: int | None = Query(500, description="Max results to return (default: 500)"),
):
    state = get_state()
    engine = state["engine"]
    df = state["df"]

    if df.empty:
        raise HTTPException(404, "No data loaded")

    results = engine.get_all()
    if not results:
        raise HTTPException(404, "No inference results")

    # Filter by timestamp range
    filtered = results
    if start:
        filtered = [r for r in filtered if r["timestamp"] >= start]
    if end:
        filtered = [r for r in filtered if r["timestamp"] <= end]

    # Apply limit from the end (most recent)
    if limit is not None and limit < len(filtered):
        filtered = filtered[-limit:]

    all_decisions = [
        {"timestamp": r["timestamp"], "decision": r["decision"]}
        for r in filtered
    ]

    return {
        "symbol": symbol,
        "interval": "1m",
        "results": filtered,
        "all_decisions": all_decisions,
    }


@router.post("/reload")
async def reload():
    """Force the backend to reload feature data and re-run inference.
    Useful after a new training run produces a checkpoint.
    """
    state = reload_state()
    engine = state["engine"]
    return {
        "bar_count": len(state["df"]),
        "inference_steps": len(engine.get_all()),
        "message": "State reloaded",
    }


@router.get("/checkpoint")
async def checkpoint_status():
    """Return metadata about the latest checkpoint (if any)."""
    ckpt = get_latest_checkpoint()
    if ckpt is None:
        raise HTTPException(404, "No checkpoint found")
    return ckpt
