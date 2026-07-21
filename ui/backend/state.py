"""Lazy-loaded application state (data + engine) with checkpoint hot-reload.

On each call to get_state(), the module checks whether the latest checkpoint
pointer file (model/checkpoints/latest.json) has been modified.  If so, it
re-initializes the inference engine so the UI always serves the most recent
model output without a server restart.

The /reload endpoint in routes.py provides an explicit trigger as well.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ui.backend.data_service import load_feature_table
from model.inference.engine import ModelInferenceEngine, LATEST_POINTER_PATH

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {"df": pd.DataFrame(), "engine": None}
_last_mtime: float = 0.0
_last_check_time: float = 0.0
_CHECK_INTERVAL: float = 5.0  # seconds between filesystem checks


def _check_pointer_mtime() -> bool:
    """Return True if the latest checkpoint pointer has been updated since last check.
    Debounced to at most once per _CHECK_INTERVAL seconds to avoid stat() spam."""
    global _last_mtime, _last_check_time
    now = time.monotonic()
    if now - _last_check_time < _CHECK_INTERVAL:
        return False
    _last_check_time = now
    if not LATEST_POINTER_PATH.exists():
        return False
    try:
        current = os.path.getmtime(str(LATEST_POINTER_PATH))
        if current > _last_mtime:
            _last_mtime = current
            return True
    except OSError:
        pass
    return False


def _init_state(force: bool = False):
    if _state["engine"] is not None and not force:
        return
    logger.info("Initializing state: loading feature data and precomputing inference…")
    df = load_feature_table()
    engine = ModelInferenceEngine(df)
    _state["df"] = df
    _state["engine"] = engine
    logger.info("State initialized — %d bars, %d inference steps", len(df), len(engine.get_all()))


def get_state() -> dict[str, Any]:
    if _check_pointer_mtime():
        logger.info("Checkpoint pointer modified — reloading state")
        _init_state(force=True)
    else:
        _init_state(force=False)
    return _state


def reload_state() -> dict[str, Any]:
    """Force a full reload of the feature data and inference engine."""
    _init_state(force=True)
    return _state
