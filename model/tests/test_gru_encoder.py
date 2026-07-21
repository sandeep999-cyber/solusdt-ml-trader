"""Unit tests for the GRU Phase A body (model/body/gru_encoder.py).

Verifies the model-side contract from model/INTERFACE.md:
  - forward() produces (mean, log_var) each of shape (B, H)
  - compute_loss() returns a scalar + required sub-metric keys
  - gradients flow through the whole model
  - the zero-initialized head starts exactly at the persistence baseline
  - raw-scale inputs (e.g. cvd_quote ~1e8) cannot produce non-finite outputs
  - non-canonical feature counts fall back to identity scaling

All tests use dummy tensors — no dataset dependency, runs in seconds.
"""

from __future__ import annotations

import torch

from model.body.gru_encoder import GRUEncoder
from model.config.run_config import RunConfig


def _make_model(n_features: int = 10, horizon: int = 12) -> GRUEncoder:
    return GRUEncoder(
        n_features=n_features,
        window_length=60,
        horizon=horizon,
        phase="A",
    )


def _make_config(horizon: int = 12) -> RunConfig:
    return RunConfig(horizon=horizon, run_name="test_gru")


def test_forward_shapes():
    model = _make_model()
    B, W, F, H = 8, 60, 10, 12
    mean, log_var = model(torch.randn(B, W, F))
    assert mean.shape == (B, H), f"Expected mean ({B}, {H}), got {mean.shape}"
    assert log_var.shape == (B, H), f"Expected log_var ({B}, {H}), got {log_var.shape}"


def test_zero_init_starts_at_persistence_baseline():
    """With a zeroed head, mean=0 and log_var=0 everywhere, so NLL per step is
    0.5 * target^2 — i.e. the persistence baseline (var=1 for norm_return)."""
    model = _make_model()
    B, W, F, H = 4, 60, 10, 12
    config = _make_config(horizon=H)

    mean, log_var = model(torch.randn(B, W, F))
    assert torch.all(mean == 0), "mean should be exactly 0 at init"
    assert torch.all(log_var == 0), "log_var should be exactly 0 at init"

    targets = torch.randn(B, H)
    loss, sub = model.compute_loss((mean, log_var), targets, config)
    expected = (0.5 * targets.pow(2)).mean().item()
    assert abs(loss.item() - expected) < 1e-5, (
        f"init loss {loss.item():.6f} != 0.5*E[t^2] {expected:.6f}"
    )
    assert abs(sub["nll"] - expected) < 1e-5


def test_compute_loss_returns_scalar_and_keys():
    model = _make_model()
    B, W, F, H = 4, 60, 10, 12
    config = _make_config(horizon=H)
    outputs = model(torch.randn(B, W, F))
    targets = torch.randn(B, H)
    loss, sub = model.compute_loss(outputs, targets, config)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    for key in ("loss", "nll", "mse", "var_mean"):
        assert key in sub, f"missing sub-metric '{key}'"


def test_backward_pass_all_params_get_gradients():
    model = _make_model()
    # The head is zero-initialized by design, which blocks gradient flow into
    # the GRU on the very first backward pass. Perturb the head so this test
    # exercises gradient plumbing through the whole model.
    torch.nn.init.normal_(model.head.weight, std=0.01)
    B, W, F, H = 4, 60, 10, 12
    config = _make_config(horizon=H)
    outputs = model(torch.randn(B, W, F))
    targets = torch.randn(B, H)
    loss, _ = model.compute_loss(outputs, targets, config)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_raw_scale_inputs_stay_finite():
    """Regression test for the SimpleMLP instability: inputs at raw dataset
    scales (cvd_quote ~1e8) must not produce NaN/Inf outputs or loss."""
    model = _make_model()
    B, W, F, H = 4, 60, 10, 12
    config = _make_config(horizon=H)

    x = torch.randn(B, W, F)
    x[:, :, 1] = 1e8      # cvd_quote-scale values
    x[:, :, 2] = 200.0    # price-scale values
    mean, log_var = model(x)
    assert torch.isfinite(mean).all(), "non-finite mean on raw-scale input"
    assert torch.isfinite(log_var).all(), "non-finite log_var on raw-scale input"

    loss, _ = model.compute_loss((mean, log_var), torch.randn(B, H), config)
    assert torch.isfinite(loss), "non-finite loss on raw-scale input"


def test_nan_targets_are_masked():
    model = _make_model()
    B, W, F, H = 4, 60, 10, 12
    config = _make_config(horizon=H)
    outputs = model(torch.randn(B, W, F))
    targets = torch.randn(B, H)
    targets[:, :3] = float("nan")  # warmup-style NaNs
    loss, sub = model.compute_loss(outputs, targets, config)
    assert torch.isfinite(loss), "loss should be finite with NaN targets"
    assert sub["nll"] == sub["nll"], "nll should not be NaN"


def test_noncanonical_feature_count_falls_back_to_identity():
    model = _make_model(n_features=6, horizon=5)
    assert torch.all(model.feature_center == 0)
    assert torch.all(model.feature_scale == 1)
    mean, log_var = model(torch.randn(2, 60, 6))
    assert mean.shape == (2, 5) and log_var.shape == (2, 5)
