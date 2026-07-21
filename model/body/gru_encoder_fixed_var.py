"""Diagnostic: GRUEncoder with variance frozen at 1.0 (log_var=0).

If MSE drops significantly below the unconditional variance (~1.018) under
this regime, the flat-mean problem in full-NLL training is a variance-shortcut
pathology, not a genuine predictive ceiling.

Design: wraps GRUEncoder, forces log_var to zero in forward(), computes
plain MSE loss (equivalent to 2 * NLL when var=1). All other architecture
details (GRU hidden size, dropout, head init) are identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.body.gru_encoder import GRUEncoder, _compute_horizon_weights


class GRUEncoderFixedVar(nn.Module):
    """GRUEncoder with frozen variance (log_var=0) — plain MSE diagnostic."""

    def __init__(
        self,
        n_features: int = 10,
        window_length: int = 60,
        horizon: int = 12,
        phase: str = "A",
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = GRUEncoder(
            n_features=n_features,
            window_length=window_length,
            horizon=horizon,
            phase=phase,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.horizon = horizon

    def forward(self, window_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, _ = self.encoder(window_batch)
        log_var = torch.zeros_like(mean)
        return mean, log_var

    def compute_loss(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        config: "RunConfig",  # noqa: F821
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mean, _ = outputs
        H = self.horizon

        weighting = getattr(config, "horizon_weighting", "uniform")
        decay_rate = getattr(config, "horizon_decay_rate", 0.9)
        weights = _compute_horizon_weights(H, weighting, decay_rate).to(mean.device)

        valid = ~torch.isnan(targets)
        squared_error = (targets - mean) ** 2

        # Weighted MSE (equivalent to 2 * NLL when var=1)
        per_step_mse = torch.nan_to_num(squared_error, nan=0.0)

        w = weights.unsqueeze(0).expand_as(per_step_mse) * valid.float()
        w_sum = w.sum(dim=1).clamp(min=1e-8)
        loss_per_element = (per_step_mse * w).sum(dim=1) / w_sum
        loss = loss_per_element.mean()

        total_valid = valid.sum().float().clamp(min=1)
        mse = (squared_error.masked_fill(~valid, 0.0).sum() / total_valid).item()

        return loss, {
            "loss": round(loss.item(), 6),
            "nll": round(0.5 * loss.item(), 6),
            "mse": round(mse, 6),
            "var_mean": 1.0,
        }
