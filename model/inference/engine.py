"""Model-backed inference engine using trained GRUEncoder checkpoint.

Replaces the heuristic SimulatedInferenceEngine with real model predictions.
Loads the latest checkpoint from model/checkpoints/latest.json, runs batched
inference over all bars, and produces the same output contract for the UI.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from model.checkpoints.load import load_checkpoint
from model.config.run_config import PHASE_A_FEATURES

logger = logging.getLogger(__name__)

WINDOW_SIZE = 60
PREDICTION_HORIZON = 12

_ROOT = Path(__file__).resolve().parents[2]
LATEST_POINTER_PATH = _ROOT / "model" / "checkpoints" / "latest.json"
RUNS_DIR = _ROOT / "model" / "runs"

_BATCH_SIZE = 4096



def get_latest_checkpoint() -> Optional[dict[str, Any]]:
    """Read the latest checkpoint pointer for UI consumption."""
    if not LATEST_POINTER_PATH.exists():
        return None
    try:
        with open(LATEST_POINTER_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read latest pointer: %s", e)
        return None


def _resolve_best_checkpoint() -> Path:
    """Resolve the path to best.pt from the latest checkpoint pointer."""
    ckpt_info = get_latest_checkpoint()
    if ckpt_info is None:
        raise FileNotFoundError(
            "No checkpoint pointer found at model/checkpoints/latest.json. "
            "Run training first."
        )

    run_name = ckpt_info.get("run_name")
    if not run_name:
        raise RuntimeError("Checkpoint pointer missing run_name")

    best_path = RUNS_DIR / run_name / "checkpoints" / "best.pt"
    if best_path.exists():
        return best_path

    fallback = ckpt_info.get("checkpoint_path")
    if fallback:
        pointer_path = Path(fallback)
        if pointer_path.exists():
            logger.warning("best.pt not found at %s, falling back to %s", best_path, pointer_path)
            return pointer_path

    raise FileNotFoundError(f"No checkpoint found at {best_path}")


def _build_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Instantiate GRUEncoder and load checkpoint state dict.

    Handles the _orig_mod. prefix added by torch.compile wrapping so that
    checkpoints saved with compile enabled load cleanly into an eager model.
    Metadata validation (contract version, feature columns, window/horizon) is
    delegated to load_checkpoint(strict=True).
    """
    from model.body.gru_encoder import GRUEncoder

    n_features = len(PHASE_A_FEATURES)
    model = GRUEncoder(
        n_features=n_features,
        window_length=WINDOW_SIZE,
        horizon=PREDICTION_HORIZON,
    ).to(device)

    # Validate metadata via the proper loader (strict mode with feature order check)
    load_checkpoint(
        checkpoint_path,
        model=None,
        expected_features=list(PHASE_A_FEATURES),
        expected_window=WINDOW_SIZE,
        expected_horizon=PREDICTION_HORIZON,
        strict=True,
    )

    # Load state dict manually to handle _orig_mod. prefix from torch.compile
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = raw_ckpt["model_state_dict"]
    # Strip _orig_mod. prefix from torch.compile-wrapped checkpoints
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    # Strip encoder. prefix from DistributedDataParallel / wrapper checkpoints
    if any(k.startswith("encoder.") for k in state_dict):
        state_dict = {k.removeprefix("encoder."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    logger.info("Loaded model state dict from %s", checkpoint_path)
    model.eval()
    return model


def _ffill_np(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values along axis=0 (pure numpy, no copy)."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    idx = np.where(~mask, np.arange(len(arr), dtype=np.intp)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return np.take_along_axis(arr, idx, axis=0)


class ModelInferenceEngine:
    """Batched model inference over all bars using the trained GRUEncoder.

    Produces the same output schema as the old SimulatedInferenceEngine so the
    UI and API routes require no changes.

    Outputs per bar:
      - timestamp / window_start / window_end
      - predicted_future_state : list of {timestamp, price, lower, upper}
      - uncertainty            : float [0, 1]
      - surprise               : float [0, 1]
      - decision               : "long" | "short" | "flat"
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
        horizon: int = PREDICTION_HORIZON,
    ):
        df = df.reset_index(drop=True)
        self.df = df
        self.window_size = window_size
        self.horizon = horizon
        self._results: dict[int, dict[str, Any]] = {}
        self._ts_index: dict[str, int] = {}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = _resolve_best_checkpoint()
        logger.info("Loading model from %s on %s", checkpoint_path, device)
        self.model = _build_model(checkpoint_path, device)
        self._device = device

        self._run_batched_inference()

    # ------------------------------------------------------------------
    # Public API (same as SimulatedInferenceEngine)
    # ------------------------------------------------------------------

    def get(self, idx: int) -> dict[str, Any] | None:
        return self._results.get(idx)

    @staticmethod
    def _norm_ts(s: str) -> str:
        """Normalize a timestamp for comparison: drop tz, fractional seconds, 'T'."""
        s = s.replace("+00:00", "").replace("Z", "").replace("T", " ").strip()
        return s.split(".")[0]

    def get_by_timestamp(self, ts: str) -> dict[str, Any] | None:
        q = self._norm_ts(ts)
        hit = self._ts_index.get(q)
        if hit is not None:
            return self.get(hit)
        # Fallback: prefix match for partial queries (e.g. "2024-12-31 17")
        for k, idx in self._ts_index.items():
            if k.startswith(q):
                return self.get(idx)
        return None

    def get_range(self, start_idx: int, end_idx: int) -> list[dict[str, Any]]:
        return [self.get(i) for i in range(start_idx, end_idx + 1) if i in self._results]

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._results.values())

    def get_latest(self) -> dict[str, Any] | None:
        if not self._results:
            return None
        max_idx = max(self._results.keys())
        return self._results[max_idx]

    def get_all_decisions(self) -> list[dict[str, Any]]:
        return [
            {"timestamp": v["timestamp"], "decision": v["decision"]}
            for v in self._results.values()
        ]

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    def _run_batched_inference(self) -> None:
        t0 = __import__("time").monotonic()

        feat_data = self.df[PHASE_A_FEATURES].values.astype(np.float32)
        close = self.df["close"].values.astype(np.float64)
        realized_vol = self.df["realized_vol"].values.astype(np.float64)
        # Precompute tz-aware ISO strings ("2024-12-31T15:40:00+00:00") — these
        # must string-match the /series endpoint's pandas/jsonable_encoder output
        # exactly, because the frontend compares timestamps as strings.
        ts_iso = self.df["timestamp"].map(lambda t: t.isoformat()).to_numpy()
        ts_pd = list(self.df["timestamp"])
        norm_return = self.df["norm_return"].values.astype(np.float64)

        n = len(self.df)
        logger.info("Running inference over %d rows (%d windows)...", n, n - self.window_size + 1)

        # NaN handling per CONTRACT.md §2.3: forward-fill then back-fill with 0
        # for both feature inputs and auxiliary arrays
        feat_data = _ffill_np(feat_data)
        feat_data = np.nan_to_num(feat_data, nan=0.0)
        realized_vol = _ffill_np(realized_vol.reshape(-1, 1)).ravel()
        realized_vol = np.nan_to_num(realized_vol, nan=0.002)

        # Sliding window view over the feature array (no copy)
        # feat_data shape: (n, F) -> windows shape: (n - W + 1, W, F)
        windows_view = np.lib.stride_tricks.sliding_window_view(
            feat_data, window_shape=self.window_size, axis=0
        )

        bar_indices = list(range(self.window_size - 1, n))
        n_windows = len(bar_indices)
        total_batches = (n_windows + _BATCH_SIZE - 1) // _BATCH_SIZE
        last_reported_pct = -1

        # Store predictions for surprise computation (lookback 1 bar)
        prev_mean_0: float | None = None
        prev_log_var_0: float | None = None

        for batch_start in range(0, n_windows, _BATCH_SIZE):
            batch_end = min(batch_start + _BATCH_SIZE, n_windows)

            # Log progress every 10%
            pct = (batch_start * 100) // n_windows
            if pct >= last_reported_pct + 10:
                last_reported_pct = pct - (pct % 10)
                elapsed = __import__("time").monotonic() - t0
                logger.info(
                    "  inference %d%% (%d/%d batches) — %.1fs elapsed",
                    pct, batch_start // _BATCH_SIZE + 1, total_batches, elapsed,
                )
            batch_slice = slice(batch_start, batch_end)
            batch_idx = bar_indices[batch_slice]

            # windows_view is indexed by position, not bar index
            pos_start = batch_start
            pos_end = batch_end
            batch_windows = windows_view[pos_start:pos_end]  # (B, F, W) — window dim is last
            batch_windows = batch_windows.transpose(0, 2, 1)  # (B, W, F)
            batch_windows = np.ascontiguousarray(batch_windows)

            batch_tensor = torch.from_numpy(batch_windows).to(self._device)

            with torch.no_grad():
                mean, log_var = self.model(batch_tensor)

            mean_np = mean.cpu().numpy()  # (B, H)
            log_var_np = log_var.cpu().numpy()

            for b, idx in enumerate(batch_idx):
                means_b = mean_np[b]
                log_vars_b = log_var_np[b]

                # Surprise: compare actual norm_return with model's prediction
                # from the previous bar's first horizon step
                if prev_mean_0 is not None and prev_log_var_0 is not None:
                    surprise_val = self._compute_surprise(
                        actual=norm_return[idx],
                        pred_mean=prev_mean_0,
                        pred_var=np.exp(prev_log_var_0),
                    )
                else:
                    surprise_val = 0.0

                # Update lookback for next bar's surprise
                prev_mean_0 = float(means_b[0])
                prev_log_var_0 = float(log_vars_b[0])

                uncertainty = self._compute_uncertainty(log_vars_b)
                decision = self._compute_decision(means_b, realized_vol[idx])

                ws = idx - self.window_size + 1
                ts_min = ts_iso[ws]
                ts_max = ts_iso[idx]

                future_state = self._compute_future_state(
                    means=means_b,
                    log_vars=log_vars_b,
                    current_price=float(close[idx]),
                    current_vol=float(realized_vol[idx]),
                    current_ts=ts_pd[idx],
                )

                self._results[idx] = {
                    "timestamp": ts_max,
                    "window_start": ts_min,
                    "window_end": ts_max,
                    "predicted_future_state": future_state,
                    "uncertainty": round(uncertainty, 4),
                    "surprise": round(surprise_val, 4),
                    "decision": decision,
                }

        self._ts_index = {
            self._norm_ts(v["timestamp"]): idx for idx, v in self._results.items()
        }

        elapsed = time.monotonic() - t0
        logger.info(
            "Inference complete: %d bars in %.1fs (%.1f bars/s)",
            len(self._results), elapsed, len(self._results) / elapsed if elapsed > 0 else 0,
        )

    # ------------------------------------------------------------------
    # Derived field computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_uncertainty(log_vars: np.ndarray) -> float:
        """Uncertainty from mean predicted variance, normalized to [0, 1].
        
        The model predicts log_var for norm_return at each horizon step.
        var = exp(log_var). Mean variance across all horizon steps is the
        aleatoric uncertainty. Clamp and scale to [0, 1].
        """
        var = np.exp(np.clip(log_vars, -5.0, 5.0))
        mean_var = float(np.mean(var))
        # Scale: var=0 → uncertainty 0, var=4 → uncertainty ~0.8, var=10+ → ~1.0
        return min(mean_var / 5.0, 1.0)

    @staticmethod
    def _compute_surprise(actual: float, pred_mean: float, pred_var: float) -> float:
        """Normalized prediction error (z-score magnitude) clamped to [0, 1]."""
        std = max(np.sqrt(pred_var), 1e-8)
        z = abs(actual - pred_mean) / std
        return min(z / 3.0, 1.0)  # z=3 → surprise=1.0

    @staticmethod
    def _compute_decision(means: np.ndarray, current_vol: float) -> str:
        """Decision from cumulative predicted log_return over the horizon.
        
        Sum of predicted norm_return × realized_vol = cumulative log_return.
        If significantly positive → long, negative → short, else flat.
        """
        cum_log_return = float(np.sum(means) * current_vol)
        threshold = 0.001  # ~0.1% cumulative move
        if cum_log_return > threshold:
            return "long"
        elif cum_log_return < -threshold:
            return "short"
        else:
            return "flat"

    @staticmethod
    def _compute_future_state(
        means: np.ndarray,
        log_vars: np.ndarray,
        current_price: float,
        current_vol: float,
        current_ts: Any,
    ) -> list[dict[str, Any]]:
        """Convert norm_return predictions to price path with 95% CI.
        
        For each horizon step k:
          - pred_log_return_k = mean_k * current_vol
          - cumulative price = current_price * exp(sum pred_log_return)
          - cumulative log_return variance = sum(exp(log_var_k) * current_vol^2)
          - 95% CI: price * exp(±1.96 * sqrt(cumulative_variance))
        """
        if not isinstance(current_ts, pd.Timestamp):
            current_ts = pd.Timestamp(current_ts)

        predictions = []
        cum_log_return = 0.0
        cum_var = 0.0

        for step in range(len(means)):
            m = float(means[step])
            lv = float(log_vars[step])

            step_log_return = m * current_vol
            step_var = np.exp(min(lv, 10.0)) * (current_vol ** 2)

            cum_log_return += step_log_return
            cum_var += step_var
            cum_std = np.sqrt(cum_var)

            pred_price = current_price * np.exp(cum_log_return)
            ci_factor = np.exp(1.96 * cum_std)
            lower = pred_price / ci_factor
            upper = pred_price * ci_factor

            pred_ts = current_ts + pd.Timedelta(minutes=step + 1)

            predictions.append({
                "timestamp": pred_ts.isoformat(),
                "price": round(float(pred_price), 2),
                "lower": round(float(lower), 2),
                "upper": round(float(upper), 2),
            })

        return predictions
