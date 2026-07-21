"""Minimal MLP model for SOLUSDT 1m regression with uncertainty (Phase A).

This is a reference implementation of the TradingModel interface defined
in model/INTERFACE.md. Replace with any architecture that implements the
same forward() and compute_loss() methods.

Output: (mean, log_var) each of shape (B, H) — Gaussian NLL loss per step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_horizon_weights(
    H: int, weighting: str = "uniform", decay_rate: float = 0.9
) -> torch.Tensor:
    """Compute per-step weights for horizon averaging."""
    if weighting == "uniform":
        return torch.ones(H) / H
    elif weighting == "decay":
        steps = torch.arange(H, dtype=torch.float)
        weights = decay_rate ** steps
        return weights / weights.sum()
    else:
        raise ValueError(f"Unknown horizon_weighting: {weighting}")


class SimpleMLP(nn.Module):
    """Simple MLP that flattens the window and predicts (mean, log_var) per step."""

    def __init__(
        self,
        n_features: int = 10,
        window_length: int = 60,
        horizon: int = 12,
        phase: str = "A",
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_length = window_length
        self.horizon = horizon
        self.phase = phase

        input_dim = n_features * window_length
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2 * horizon),  # mean + log_var per step
        )

    def forward(self, window_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            window_batch: (B, W, F) tensor

        Returns:
            (mean, log_var) each of shape (B, H)
        """
        B = window_batch.shape[0]
        flat = window_batch.view(B, -1)
        raw = self.net(flat)  # (B, 2*H)
        mean = raw[:, : self.horizon]  # (B, H)
        log_var = raw[:, self.horizon :]  # (B, H)
        return mean, log_var

    def compute_loss(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        config: "RunConfig",  # noqa: F821
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Gaussian negative log-likelihood loss per horizon step.

        loss = 0.5 * sum_i [ w_i * ( log_var_i + (target_i - mean_i)^2 / exp(log_var_i) ) ]
        where w_i are per-step weights from config.horizon_weighting.

        NaN targets are masked out (norm_return can be NaN for warmup bars).

        Returns (scalar_loss, metrics_dict).
        """
        mean, log_var = outputs
        H = self.horizon

        # Per-step weighting
        weighting = getattr(config, "horizon_weighting", "uniform")
        decay_rate = getattr(config, "horizon_decay_rate", 0.9)
        weights = _compute_horizon_weights(H, weighting, decay_rate).to(mean.device)

        # Mask NaN targets
        nan_mask = torch.isnan(targets)

        # Clamp log_var to avoid numerical instability from division by near-zero variance
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        var = torch.exp(log_var)

        # Gaussian NLL per element: 0.5 * (log_var + (t - m)^2 / exp(log_var))
        squared_error = (targets - mean) ** 2
        per_step_nll = 0.5 * (log_var + squared_error / var)  # (B, H)

        # Zero out NaN contributions and compute per-batch-element valid masks
        per_step_nll = per_step_nll.masked_fill(nan_mask, 0.0)
        valid_count = (~nan_mask).sum(dim=1, keepdim=True).float().clamp(min=1)

        # Weighted mean over horizon per batch element, normalized by valid count
        w = weights.unsqueeze(0).expand_as(per_step_nll)
        weighted = per_step_nll * w
        loss_per_element = weighted.sum(dim=1) / valid_count.squeeze(1)
        loss = loss_per_element.mean()  # mean over batch

        # Unweighted mean NLL per valid step (across entire batch)
        total_valid = (~nan_mask).sum().float().clamp(min=1)
        nll = (per_step_nll.sum() / total_valid).item()

        # MSE for interpretability (only valid targets)
        mse = (squared_error.masked_fill(nan_mask, 0.0).sum() / total_valid).item()

        # Mean predicted variance
        var_mean = var.mean().item()

        return loss, {
            "loss": round(loss.item(), 6),
            "nll": round(nll, 6),
            "mse": round(mse, 6),
            "var_mean": round(var_mean, 6),
        }
