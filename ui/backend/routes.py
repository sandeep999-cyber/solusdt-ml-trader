import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from model.inference.engine import LATEST_POINTER_PATH, RUNS_DIR, get_latest_checkpoint
from ui.backend.state import get_state, reload_state

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


@router.get("/checkpoints/list")
async def list_checkpoints():
    """List all available training runs with their metrics."""
    if not RUNS_DIR.exists():
        return {"checkpoints": []}

    entries = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name.startswith("__"):
            continue

        ckpt_path = run_dir / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            continue

        metrics = None
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    lines = f.readlines()
                if lines:
                    metrics = json.loads(lines[-1])
            except (json.JSONDecodeError, OSError):
                pass

        config_info = {}
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                if cfg:
                    config_info = {
                        "model_class": cfg.get("model_class", ""),
                        "notes": cfg.get("notes", ""),
                        "num_epochs": cfg.get("num_epochs"),
                    }
            except Exception:
                pass

        entries.append({
            "run_name": run_dir.name,
            "is_smoketest": "smoketest" in run_dir.name,
            "has_best_pt": True,
            "metrics": metrics,
            "config": config_info,
        })

    return {"checkpoints": entries}


@router.post("/checkpoints/select")
async def select_checkpoint(run_name: str = Form(...)):
    """Switch the active checkpoint to a different training run."""
    run_dir = RUNS_DIR / run_name
    ckpt_path = run_dir / "checkpoints" / "best.pt"

    if not run_dir.is_dir():
        raise HTTPException(404, f"Run '{run_name}' not found")
    if not ckpt_path.exists():
        raise HTTPException(404, f"No best.pt in run '{run_name}'")

    pointer = {
        "run_name": run_name,
        "checkpoint_path": str(ckpt_path.resolve()),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    LATEST_POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_POINTER_PATH.write_text(json.dumps(pointer, indent=2))

    # Reload engine immediately so next request is fresh
    reload_state()

    return {"selected": run_name, "checkpoint": pointer}


@router.post("/checkpoints/upload")
async def upload_checkpoint(file: UploadFile = File(...)):
    """Upload a best.pt file and register it as a new run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"uploaded_{timestamp}"
    run_dir = RUNS_DIR / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dest = ckpt_dir / "best.pt"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pointer = {
        "run_name": run_name,
        "checkpoint_path": str(dest.resolve()),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    LATEST_POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_POINTER_PATH.write_text(json.dumps(pointer, indent=2))

    reload_state()

    return {"selected": run_name, "checkpoint": pointer}


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
