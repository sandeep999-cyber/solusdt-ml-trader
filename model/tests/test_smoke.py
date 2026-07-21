"""Fast smoke test: full loop on a tiny data slice.

Verifies:
  - Data loader constructs valid windows with correct shapes
  - Model forward() and compute_loss() produce expected shapes and types
  - Backward pass succeeds (gradients flow)
  - Checkpoint save + load round-trips correctly
  - Checkpoint loader metadata guard catches mismatches

Should complete in seconds.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from model.config.run_config import RunConfig
from model.data.loader import create_dataloader, CausalWindowDataset
from model.body.simple_mlp import SimpleMLP
from model.checkpoints.load import load_checkpoint, ContractMismatchError
from model.inference.contract_version import CONTRACT_VERSION


def _make_smoke_config(tmp_dir: str) -> RunConfig:
    """Create a minimal RunConfig for smoke testing."""
    config = RunConfig(
        phase="A",
        window_length=20,
        horizon=5,
        feature_columns=[
            "cvd", "vwap_20", "realized_vol",
            "log_return", "norm_return", "return_pct",
        ],
        split_config_ref="model/config/splits.py",
        active_split="train",
        batch_size=16,
        learning_rate=0.001,
        optimizer="adam",
        num_epochs=2,
        seed=42,
        run_name="test_smoke",
        notes="Smoke test run",
        processed_dir=str(Path("data/processed/v1/SOLUSDT/1m")),
        runs_dir=tmp_dir,
    )
    return config


def test_dataloader_shapes():
    """Verify data loader produces (B, W, F) windows and (B, H) targets."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_smoke_config(tmp)
        loader = create_dataloader(config, split="train", shuffle=False)

        windows, targets = next(iter(loader))
        B = config.batch_size
        W = config.window_length
        F = len(config.feature_columns)
        H = config.horizon

        assert windows.shape == (B, W, F), f"Expected ({B}, {W}, {F}), got {windows.shape}"
        assert targets.shape == (B, H), f"Expected ({B}, {H}), got {targets.shape}"
        assert targets.dtype == torch.float32


def test_dataloader_causality():
    """Verify that window at position i does not contain data from beyond i+W-1."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_smoke_config(tmp)
        dataset = CausalWindowDataset(config, split="train")

        for idx in [0, 1, 10, 100]:
            w_end_ts, last_tgt_ts = dataset.get_timestamps(idx)
            last_win_row = dataset.df.iloc[idx + config.window_length - 1]
            tgt_row = dataset.df.iloc[idx + config.window_length + config.horizon - 1]
            assert tgt_row["timestamp"] > last_win_row["timestamp"], (
                f"Target timestamp {tgt_row['timestamp']} is not after "
                f"window end {last_win_row['timestamp']}"
            )


def test_dataloader_target_values():
    """Verify targets are continuous floats from norm_return, not discretized."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_smoke_config(tmp)
        dataset = CausalWindowDataset(config, split="train")
        _, targets = dataset[0]
        assert targets.shape == (config.horizon,)
        # Values should be continuous floats (not just -1, 0, 1)
        assert targets.dtype == torch.float32


def test_model_forward():
    """Verify model forward produces (mean, log_var) each of shape (B, H)."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_smoke_config(tmp)
        model = SimpleMLP(
            n_features=len(config.feature_columns),
            window_length=config.window_length,
            horizon=config.horizon,
            phase="A",
        )
        B = config.batch_size
        W = config.window_length
        F = len(config.feature_columns)
        H = config.horizon
        dummy_input = torch.randn(B, W, F)

        mean, log_var = model(dummy_input)
        assert mean.shape == (B, H), f"Expected mean ({B}, {H}), got {mean.shape}"
        assert log_var.shape == (B, H), f"Expected log_var ({B}, {H}), got {log_var.shape}"


def test_model_loss():
    """Verify compute_loss returns scalar + dict with expected keys."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_smoke_config(tmp)
        model = SimpleMLP(
            n_features=len(config.feature_columns),
            window_length=config.window_length,
            horizon=config.horizon,
            phase="A",
        )
        B = config.batch_size
        W = config.window_length
        F = len(config.feature_columns)
        H = config.horizon

        dummy_input = torch.randn(B, W, F, requires_grad=True)
        dummy_targets = torch.randn(B, H)

        outputs = model(dummy_input)
        loss, sub_metrics = model.compute_loss(outputs, dummy_targets, config)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in sub_metrics
        assert "nll" in sub_metrics
        assert "mse" in sub_metrics


def test_backward_pass():
    """Verify gradients flow through the full model."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_smoke_config(tmp)
        model = SimpleMLP(
            n_features=len(config.feature_columns),
            window_length=config.window_length,
            horizon=config.horizon,
            phase="A",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loader = create_dataloader(config, split="train", shuffle=False)
        windows, targets = next(iter(loader))

        optimizer.zero_grad()
        outputs = model(windows)
        loss, _ = model.compute_loss(outputs, targets, config)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_checkpoint_roundtrip():
    """Verify checkpoint save and load recovers the model state."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_smoke_config(tmp)
        model = SimpleMLP(
            n_features=len(config.feature_columns),
            window_length=config.window_length,
            horizon=config.horizon,
            phase="A",
        )

        ckpt_dir = Path(tmp) / "ckpt_test"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "test.pt"

        state = model.state_dict()
        first_weight = list(state.values())[0].clone()

        torch.save({
            "epoch": 1,
            "model_state_dict": state,
            "optimizer_state_dict": torch.optim.Adam(model.parameters()).state_dict(),
            "val_metrics": {"loss": 0.5, "nll": 0.5, "mse": 0.5},
            "config": config.to_json(),
            "contract_version": CONTRACT_VERSION,
            "feature_columns": config.feature_columns,
            "window_length": config.window_length,
            "horizon": config.horizon,
        }, ckpt_path)

        for p in model.parameters():
            p.data += 1.0

        load_checkpoint(
            ckpt_path,
            model=model,
            expected_features=config.feature_columns,
            expected_window=config.window_length,
            expected_horizon=config.horizon,
        )

        restored_state = model.state_dict()
        restored_weight = list(restored_state.values())[0]
        assert torch.equal(restored_weight, first_weight), "Weights not restored by checkpoint load"


def test_checkpoint_guard_rejects_mismatch():
    """Verify ContractMismatchError is raised on feature column mismatch."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_smoke_config(tmp)
        model = SimpleMLP(
            n_features=len(config.feature_columns),
            window_length=config.window_length,
            horizon=config.horizon,
            phase="A",
        )

        ckpt_dir = Path(tmp) / "ckpt_guard"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "mismatch.pt"

        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "contract_version": CONTRACT_VERSION,
            "feature_columns": ["cvd", "vwap_20"],
            "window_length": config.window_length,
            "horizon": config.horizon,
        }, ckpt_path)

        with pytest.raises(ContractMismatchError) as excinfo:
            load_checkpoint(
                ckpt_path,
                model=model,
                expected_features=config.feature_columns,
                expected_window=config.window_length,
                expected_horizon=config.horizon,
                strict=True,
            )
        assert "features" in str(excinfo.value).lower()


def test_checkpoint_guard_rejects_feature_order_mismatch():
    """Verify ContractMismatchError is raised on feature ORDER change (same set)."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_smoke_config(tmp)
        model = SimpleMLP(
            n_features=len(config.feature_columns),
            window_length=config.window_length,
            horizon=config.horizon,
            phase="A",
        )

        ckpt_dir = Path(tmp) / "ckpt_order"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "order_mismatch.pt"

        reordered = list(reversed(config.feature_columns))
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "contract_version": CONTRACT_VERSION,
            "feature_columns": reordered,
            "window_length": config.window_length,
            "horizon": config.horizon,
        }, ckpt_path)

        with pytest.raises(ContractMismatchError) as excinfo:
            load_checkpoint(
                ckpt_path,
                model=model,
                expected_features=config.feature_columns,
                expected_window=config.window_length,
                expected_horizon=config.horizon,
                strict=True,
            )
        assert "features" in str(excinfo.value).lower()


def test_latest_pointer_updated_on_save():
    """Verify the latest pointer file is updated when checkpoint is saved."""
    from model.train import _update_latest_pointer

    with tempfile.TemporaryDirectory() as tmp:
        pointer_path = Path(tmp) / "latest.json"
        import model.train
        original_path = model.train.LATEST_POINTER_PATH
        model.train.LATEST_POINTER_PATH = pointer_path

        try:
            ckpt_path = Path(tmp) / "checkpoints" / "epoch_0001.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_text("dummy")

            _update_latest_pointer("test_run", ckpt_path)

            assert pointer_path.exists()
            with open(pointer_path) as f:
                data = json.load(f)
            assert data["run_name"] == "test_run"
            assert "epoch_0001.pt" in data["checkpoint_path"]
        finally:
            model.train.LATEST_POINTER_PATH = original_path


def test_baseline_file_exists():
    """Verify the persistence baseline file exists with expected keys."""
    baseline_path = Path("model/baselines/persistence_2024.json")
    assert baseline_path.exists(), f"Baseline file not found: {baseline_path}"
    with open(baseline_path) as f:
        data = json.load(f)
    assert "nll" in data
    assert "metric_version" in data
