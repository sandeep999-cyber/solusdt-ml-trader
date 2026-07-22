"""Generic training harness for SOLUSDT 1m models.

Usage:
    python -m model.train --config configs/example.yaml
    python -m model.train --config configs/example.yaml --smoke-test-first
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.checkpoints.load import load_checkpoint
from model.config.run_config import RunConfig
from model.data.loader import create_dataloader
from model.inference.contract_version import CONTRACT_VERSION

logger = logging.getLogger(__name__)

BASELINE_PATH = Path("model/baselines/persistence_2024.json")
LATEST_POINTER_PATH = Path("model/checkpoints/latest.json")


def _load_baseline() -> Optional[dict[str, Any]]:
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            return json.load(f)
    logger.warning("Baseline file not found at %s — delta logging is no-op", BASELINE_PATH)
    return None


def _update_latest_pointer(run_name: str, checkpoint_path: Path) -> None:
    data = {
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    LATEST_POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LATEST_POINTER_PATH, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Updated latest pointer -> %s", checkpoint_path)


def _restore_best_val_loss(metrics_path: Path) -> float:
    """Read past metrics to find the best val loss, for accurate resume comparison."""
    best = float("inf")
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    v = entry.get("loss", float("inf"))
                    if v < best:
                        best = v
                except json.JSONDecodeError:
                    continue
    return best


def _resolve_model_class(model_class_path: str) -> type[nn.Module]:
    """Resolve a dotted path like 'model.body.simple_mlp.SimpleMLP' to a class."""
    parts = model_class_path.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _save_checkpoint(
    run_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_metrics: dict[str, float],
    is_best: bool,
    config: RunConfig,
    data_meta: dict[str, Any],
    smoke_test: bool = False,
) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "config": config.to_json(),
        "contract_version": CONTRACT_VERSION,
        "feature_columns": config.feature_columns,
        "window_length": config.window_length,
        "horizon": config.horizon,
        "data": data_meta,
    }

    # Periodic checkpoint
    path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    torch.save(ckpt, path)
    logger.info("Saved checkpoint: %s", path)

    # Best checkpoint (overwrite)
    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save(ckpt, best_path)
        logger.info("Saved best checkpoint: %s (val_loss=%.6f)", best_path, val_metrics.get("loss", 0))

    # Update the "latest" pointer.  Smoke tests don't touch the production
    # pointer.  When a best checkpoint is saved the pointer goes to best.pt so
    # the inference engine loads the best model, not the last periodic one.
    if not smoke_test:
        pointer_path = (ckpt_dir / "best.pt") if is_best else path
        _update_latest_pointer(config.run_name, pointer_path)
    return path


def _load_latest_checkpoint(
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: RunConfig,
) -> int:
    """Load the most recent checkpoint from run_dir. Returns starting epoch."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return 0

    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpt_files:
        return 0

    latest = ckpt_files[-1]
    logger.info("Resuming from checkpoint: %s", latest)
    ckpt = load_checkpoint(
        latest,
        model=model,
        expected_features=config.feature_columns,
        expected_window=config.window_length,
        expected_horizon=config.horizon,
        strict=True,
    )
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"] + 1  # resume from next epoch


def _log_metrics(metrics_path: Path, data: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a") as f:
        f.write(json.dumps(data) + "\n")


def _read_data_version(config: RunConfig) -> str:
    """Read the processed-data version marker written by the feature pipeline."""
    marker = Path(config.processed_dir) / ".version"
    if marker.exists():
        return marker.read_text().strip()
    return "unknown"


def _validate_config(config: RunConfig) -> None:
    """Fail fast on config errors that would otherwise surface deep in a run."""
    errors = []
    if config.window_length <= 0:
        errors.append(f"window_length must be positive, got {config.window_length}")
    if config.horizon <= 0:
        errors.append(f"horizon must be positive, got {config.horizon}")
    if config.horizon_weighting not in ("uniform", "decay"):
        errors.append(
            f"horizon_weighting must be 'uniform' or 'decay', "
            f"got '{config.horizon_weighting}'"
        )

    feat_dir = Path(config.processed_dir)
    if not feat_dir.exists():
        errors.append(f"processed_dir not found: {feat_dir}")
    else:
        parquet_files = sorted(feat_dir.glob("**/*.parquet"))
        if not parquet_files:
            errors.append(f"no .parquet files found in {feat_dir}")
        else:
            schema_names = set(pq.ParquetDataset(parquet_files).schema.names)
            required = set(config.feature_columns) | {"timestamp", "norm_return"}
            missing = sorted(required - schema_names)
            if missing:
                errors.append(
                    f"columns not found in {feat_dir}: {missing}. "
                    "Check feature_columns against the processed dataset schema."
                )

    if errors:
        raise ValueError("Invalid run config:\n  - " + "\n  - ".join(errors))


def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    config: RunConfig,
    device: torch.device,
    baseline: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_nll = 0.0
    total_mse = 0.0
    total_var = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0

    with torch.no_grad():
        for windows, targets in val_loader:
            non_blocking = device.type == "cuda"
            windows = windows.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            outputs = model(windows)
            loss, sub_metrics = model.compute_loss(outputs, targets, config)

            total_loss += loss.item()
            total_nll += sub_metrics.get("nll", loss.item())
            total_mse += sub_metrics.get("mse", 0.0)
            total_var += sub_metrics.get("var_mean", 0.0)
            n_batches += 1

            # Collect predictions for accuracy (only for models that expose a
            # discrete predict(); Phase A regression outputs are (mean, log_var)
            # tuples — accuracy is not meaningful there)
            if hasattr(model, "predict"):
                preds = model.predict(outputs)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

    avg_loss = total_loss / max(n_batches, 1)
    avg_nll = total_nll / max(n_batches, 1)
    avg_mse = total_mse / max(n_batches, 1)
    avg_var = total_var / max(n_batches, 1)
    metrics = {
        "loss": round(avg_loss, 6),
        "nll": round(avg_nll, 6),
        "mse": round(avg_mse, 6),
        "var_mean": round(avg_var, 6),
    }

    # Accuracy
    if all_preds:
        preds = torch.cat(all_preds).view(-1)
        tgts = torch.cat(all_targets).view(-1)
        valid = tgts != 0
        if valid.any():
            acc = (preds[valid] == tgts[valid]).float().mean().item()
            metrics["accuracy"] = round(acc, 4)

    # Baseline delta (from unweighted NLL so it stays comparable across runs
    # with different horizon_weighting settings)
    if baseline is not None:
        base_loss = baseline.get("nll")
        if base_loss is not None:
            metrics["baseline_loss"] = base_loss
            metrics["baseline_delta"] = round(avg_nll - base_loss, 6)

    # Phase B calibration (no-op for Phase A — wire up when Phase B lands)
    if config.phase == "B":
        metrics["calibration_note"] = "Phase B calibration not yet implemented"

    return metrics


def train(config_path: str, smoke_test: bool = False) -> None:
    config = RunConfig.from_yaml(config_path)
    _validate_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load baseline
    baseline = _load_baseline()

    # Append smoke suffix so it doesn't collide with the real run
    if smoke_test:
        config.run_name = config.run_name + "_smoketest"

    # Create run directory and snapshot the config
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(run_dir / "config.yaml")
    logger.info("Run directory: %s", run_dir)

    # Data loaders
    train_loader = create_dataloader(config, split="train", shuffle=True, device=device)
    val_loader = create_dataloader(config, split="val", shuffle=False, device=device)

    # Data provenance for run/checkpoint metadata
    train_df = train_loader.dataset.df
    data_meta = {
        "data_version": _read_data_version(config),
        "processed_dir": config.processed_dir,
        "train_rows": len(train_df),
        "train_start": str(train_df["timestamp"].min()),
        "train_end": str(train_df["timestamp"].max()),
    }
    logger.info(
        "Data: version=%s, train rows=%d (%s -> %s)",
        data_meta["data_version"], data_meta["train_rows"],
        data_meta["train_start"], data_meta["train_end"],
    )

    if smoke_test:
        # Truncate to a few batches
        train_loader = _truncate_loader(train_loader, 3)
        val_loader = _truncate_loader(val_loader, 2)
        config.num_epochs = min(config.num_epochs, 2)
        logger.info("Smoke-test mode: truncated to 2 epochs, 3 train / 2 val batches")

    # Model
    model_class_path = getattr(config, "model_class", "model.body.simple_mlp.SimpleMLP")
    try:
        model_class = _resolve_model_class(model_class_path)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not resolve model class '{model_class_path}'. "
            f"Create it or set model_class in the run config. Error: {e}"
        ) from e

    n_features = len(config.feature_columns)
    model = model_class(
        n_features=n_features,
        window_length=config.window_length,
        horizon=config.horizon,
        phase=config.phase,
    ).to(device)
    # Enable torch.compile (fall back to eager if Inductor isn't available, e.g.
    # Windows without MSVC). suppress_errors makes the first forward fall through
    # gracefully instead of raising on compilation failure.
    torch._dynamo.config.suppress_errors = True
    try:
        model = torch.compile(model)
        logger.info("torch.compile enabled")
    except Exception:
        logger.warning("torch.compile not available, using eager mode")

    logger.info("Model: %s (%d params)", model_class.__name__, sum(p.numel() for p in model.parameters()))
    logger.info("Features: %d (%s)", n_features, config.feature_columns)

    # Optimizer
    lr = config.learning_rate
    if config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    # Resume support
    start_epoch = _load_latest_checkpoint(run_dir, model, optimizer, config)
    if start_epoch > 0:
        logger.info("Resuming from epoch %d", start_epoch)
    else:
        logger.info("Starting fresh training run")

    best_val_loss = _restore_best_val_loss(config.metrics_path())
    metrics_path = config.metrics_path()

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_train_batches = 0
        t0 = time.time()

        for batch_idx, (windows, targets) in enumerate(train_loader):
            non_blocking = device.type == "cuda"
            windows = windows.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            outputs = model(windows)
            loss, sub_metrics = model.compute_loss(outputs, targets, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss ({loss.item()}) at epoch {epoch}, "
                    f"batch {batch_idx}. Halting instead of training on garbage; "
                    f"checkpoints saved so far remain in {run_dir / 'checkpoints'}."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_loss / max(n_train_batches, 1)
        elapsed = time.time() - t0

        # Validation
        val_metrics = _validate(model, val_loader, config, device, baseline=baseline)

        # Log
        log_data = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "elapsed_sec": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log_data.update(val_metrics)
        _log_metrics(metrics_path, log_data)

        logger.info(
            "Epoch %3d/%d | train_loss=%.6f | val_loss=%.6f | val_acc=%.4f | %.1fs",
            epoch + 1, config.num_epochs, avg_train_loss,
            val_metrics.get("loss", 0), val_metrics.get("accuracy", 0),
            elapsed,
        )

        # Checkpoint
        is_best = val_metrics.get("loss", float("inf")) < best_val_loss
        if is_best:
            best_val_loss = val_metrics.get("loss", float("inf"))
        if (epoch + 1) % 5 == 0 or is_best or epoch == config.num_epochs - 1:
            _save_checkpoint(
                run_dir, epoch, model, optimizer, val_metrics, is_best, config,
                data_meta, smoke_test=smoke_test,
            )

    logger.info("Training complete. Run: %s", config.run_name)


def _truncate_loader(loader: DataLoader, n_batches: int) -> DataLoader:
    """Wrap a DataLoader to yield only the first n_batches."""
    from torch.utils.data import BatchSampler, SequentialSampler, Subset

    dataset = loader.dataset
    # Take first n_batches * batch_size samples
    n_samples = min(n_batches * loader.batch_size, len(dataset))
    subset = Subset(dataset, list(range(n_samples)))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        drop_last=False,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a SOLUSDT 1m model")
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    parser.add_argument(
        "--smoke-test-first", action="store_true",
        help="Run a fast smoke test before the full training run",
    )
    parser.add_argument(
        "--smoke-test-only", action="store_true",
        help="Run only the smoke test and exit (no full training)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.smoke_test_only:
        logger.info("Running smoke test only...")
        train(args.config, smoke_test=True)
        logger.info("Smoke test complete.")
        return

    if args.smoke_test_first:
        logger.info("Running smoke test before full training...")
        train(args.config, smoke_test=True)
        logger.info("Smoke test passed. Starting full training run.")

    train(args.config, smoke_test=False)


if __name__ == "__main__":
    main()
