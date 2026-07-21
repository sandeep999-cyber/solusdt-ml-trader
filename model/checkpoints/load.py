"""Checkpoint loader with metadata guard.

Every checkpoint stores its inference contract version, feature columns,
window length, and horizon. The loader validates all of these against the
current model configuration before allowing a load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from model.inference.contract_version import CONTRACT_VERSION

logger = logging.getLogger(__name__)


class ContractMismatchError(RuntimeError):
    """Raised when checkpoint metadata does not match the current configuration."""


def load_checkpoint(
    path: str | Path,
    model: Optional[nn.Module] = None,
    expected_features: Optional[list[str]] = None,
    expected_window: Optional[int] = None,
    expected_horizon: Optional[int] = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint with full metadata validation.

    Args:
        path: Path to the checkpoint .pt file.
        model: Optional model instance to load state dict into.
        expected_features: Expected feature column list. Mismatch raises error.
        expected_window: Expected window_length. Mismatch raises error.
        expected_horizon: Expected horizon. Mismatch raises error.
        strict: If True, all mismatches raise ContractMismatchError.
               If False, mismatches are logged as warnings only.

    Returns:
        The full checkpoint dict (metadata + state dict).

    Raises:
        ContractMismatchError: On any metadata mismatch (if strict=True).
        FileNotFoundError: If the checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    errors = []

    # Contract version
    ckpt_contract = ckpt.get("contract_version", "unknown")
    if ckpt_contract != CONTRACT_VERSION:
        msg = (
            f"Checkpoint contract v{ckpt_contract} does not match "
            f"code contract v{CONTRACT_VERSION}"
        )
        errors.append(msg)

    # Feature columns (ordered-list comparison — order matters for GRU encoder)
    ckpt_features = ckpt.get("feature_columns", [])
    if expected_features is not None and ckpt_features:
        if ckpt_features != expected_features:
            msg = (
                f"Checkpoint expects {len(ckpt_features)} features "
                f"({ckpt_features}), model configured for "
                f"{len(expected_features)} ({expected_features})"
            )
            errors.append(msg)

    # Window length
    ckpt_window = ckpt.get("window_length")
    if expected_window is not None and ckpt_window is not None:
        if ckpt_window != expected_window:
            msg = (
                f"Checkpoint window_length={ckpt_window}, "
                f"model expected={expected_window}"
            )
            errors.append(msg)

    # Horizon
    ckpt_horizon = ckpt.get("horizon")
    if expected_horizon is not None and ckpt_horizon is not None:
        if ckpt_horizon != expected_horizon:
            msg = (
                f"Checkpoint horizon={ckpt_horizon}, "
                f"model expected={expected_horizon}"
            )
            errors.append(msg)

    if errors:
        combined = "; ".join(errors)
        if strict:
            raise ContractMismatchError(
                f"Cannot load checkpoint '{path.name}': {combined}. "
                f"Either update the model config to match or retrain with the current config."
            )
        else:
            for e in errors:
                logger.warning("Checkpoint mismatch (non-strict): %s", e)

    # Load state dict if model provided
    if model is not None and "model_state_dict" in ckpt:
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded model state dict from %s", path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load state dict into model: {e}. "
                f"Check that the model architecture matches the checkpoint."
            ) from e

    logger.info("Checkpoint loaded: %s (contract v%s)", path.name, ckpt_contract)
    return ckpt


def load_best_checkpoint(
    run_dir: str | Path,
    model: nn.Module,
    expected_features: Optional[list[str]] = None,
    expected_window: Optional[int] = None,
    expected_horizon: Optional[int] = None,
) -> dict[str, Any]:
    """Load the best checkpoint from a run directory."""
    run_dir = Path(run_dir)
    best_path = run_dir / "checkpoints" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No best checkpoint found in {run_dir}")
    return load_checkpoint(
        best_path,
        model=model,
        expected_features=expected_features,
        expected_window=expected_window,
        expected_horizon=expected_horizon,
    )


def load_latest_pointer() -> Optional[dict[str, Any]]:
    """Read the latest checkpoint pointer file."""
    path = Path("model/checkpoints/latest.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
