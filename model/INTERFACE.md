# Model Interface Contract

Every trainable model must implement this interface. The training harness
(`model/train.py`) calls only `forward()` and `compute_loss()` — it never
touches model internals directly.

This makes `model/body/` and `model/heads/` freely rewritable: as long as a
model class satisfies this contract, it works with the harness unchanged.

---

## Class Requirements

```python
import torch.nn as nn

class MyModel(nn.Module):
    """Minimal required interface. Add any helper methods you like."""

    def forward(self, window_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            window_batch: tensor of shape (batch_size, window_length, n_features)

        Returns:
            Tuple of (mean, log_var), each of shape (batch_size, horizon).
              - mean: predicted expected value of norm_return at each horizon step
              - log_var: log of predicted variance at each horizon step
                (unconstrained; internally exponentiated for positivity)
        """

    def compute_loss(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        config: "RunConfig",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Gaussian negative log-likelihood loss per horizon step.

        Args:
            outputs: (mean, log_var) from forward(), each (B, H)
            targets: tensor of shape (batch_size, horizon) — continuous
                     norm_return values at the prediction targets
            config: the active RunConfig (provides horizon weighting, phase, etc.)

        Returns:
            Tuple of (scalar loss tensor, dict of sub-metrics for logging).
            The sub-metrics dict must at minimum include:
              - "loss": float (same as the scalar loss tensor)
              - "nll": float (unweighted mean NLL per step)
            Additional entries (e.g. "mse", "var_mean") are optional but
            recommended for monitoring.
        """
```

---

## Shape Conventions

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `W` | window_length (number of input bars) |
| `F` | number of feature columns |
| `H` | horizon (prediction steps ahead) |

- Input: `(B, W, F)` — float32
- Output (return trajectory): `(mean, log_var)` — each `(B, H)` — float32
- Output (volatility): single tensor `(B,)` — float32
- Target (return trajectory): `(B, H)` — float32, continuous values drawn from `norm_return`
- Target (volatility): `(B,)` — float32, `sqrt(mean(norm_return^2 over H steps))`

---

## Phase Differences

### Phase A (regression with uncertainty)
- `forward()` returns `(mean, log_var)` — predicted trajectory and uncertainty per step.
- `compute_loss()` uses Gaussian negative log-likelihood per step, averaged over the horizon with configurable step weighting.
- Horizon weighting: near-term steps can be weighted more heavily via `config.horizon_weighting`. Default is `"uniform"` (all steps equal). When set to `"decay"`, weights decay geometrically with `config.horizon_decay_rate` per step (e.g. rate=0.9 means step 1 weight=1.0, step 2 weight=0.9, step 3 weight=0.81, ...). The rationale is that far-horizon prediction is inherently noisier and should not dominate the loss.
- Discrete decisions (`"long"` / `"short"` / `"flat"`) are not produced in Phase A. They belong to Phase B, which consumes Phase A's uncertainty-aware trajectory as input.
- Sub-metrics: `loss`, `nll` (unweighted mean NLL), `mse`, `var_mean` (mean predicted variance across batch/horizon).

### Phase A — Volatility Variant (D020)
- When `config.target_type == "volatility"`, the target is a single scalar per window: `sqrt(mean(norm_return^2 over next H steps))`.
- `forward()` returns a single tensor of shape `(B,)` — predicted volatility (no log_var).
- `compute_loss()` uses plain MSE loss: `nn.functional.mse_loss(pred, target)`.
- The `CausalWindowDataset` handles the target computation automatically when `target_type="volatility"`.
- Output shape: `(B,)` not `(B, H)`. Target shape: `(B,)` not `(B, H)`.

### Phase B (uncertainty-aware + decision)
- `forward()` returns a tuple `(state_prediction, uncertainty_logits, decision_logits)`.
- `compute_loss()` combines prediction loss + uncertainty calibration loss + decision loss.
- Sub-metrics: `loss`, `pred_loss`, `uncert_loss`, `decision_loss`, `calibration_error`.

---

## Horizon Weighting

The `compute_loss` method must apply per-step weighting when averaging the per-step Gaussian NLL over the horizon:

- **`"uniform"`** (default): `weight_i = 1.0 / H` for all steps. The averaged loss is simply the arithmetic mean across steps.
- **`"decay"`**: `weight_i = decay_rate^i / sum(decay_rate^j for j=1..H)`. This down-weights far-future steps, reflecting the fact that near-term predictions are more actionable and less noisy.

```python
def _compute_horizon_weights(H: int, weighting: str, decay_rate: float = 0.9) -> torch.Tensor:
    if weighting == "uniform":
        return torch.ones(H) / H
    elif weighting == "decay":
        steps = torch.arange(H, dtype=torch.float)
        weights = decay_rate ** steps
        return weights / weights.sum()
```

The unweighted mean NLL per step must always be included in sub-metrics as `"nll"` alongside the weighted total `"loss"`, so both are trackable.

---

## Framework

Models are `torch.nn.Module` subclasses. The harness uses PyTorch's standard
optimizers and loss functions. No custom framework is needed.

---

## Verification

The smoke test (`model/tests/test_smoke.py`) instantiates a model, runs a few
forward/backward steps, and confirms the interface contract is satisfied. If
your model passes the smoke test, it will work with the training harness.
