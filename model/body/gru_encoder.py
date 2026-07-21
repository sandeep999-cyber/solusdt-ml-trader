"""GRU sequence encoder for SOLUSDT 1m regression with uncertainty (Phase A).

The first real "body", replacing the SimpleMLP placeholder. A single-layer GRU
reads the input window in time order; its final hidden state is the bottleneck
summary of "what's going on right now". A single linear head maps that state
to (mean, log_var) per horizon step — one head produces both the predicted
trajectory and its own uncertainty, per the Phase A design in
01-project-overview.md.

Two deliberate design choices:

1. Fixed input scaling. Features arrive in raw units (CONTRACT.md section 2.2:
   no normalization is baked into the pipeline), with scales from ~1e-3
   (realized_vol) to ~1e8 (cvd_quote). Feeding those straight into a recurrent
   net makes optimization unstable — see the SimpleMLP run's train_loss
   oscillating between 1e4 and 0.69. The constants below are the mean/std of
   the train split (2023-01-01 -> 2024-09-01), computed once on 2026-07-20 via:

       df = pq.ParquetDataset("data/processed/v1/SOLUSDT/1m").read().to_pandas()
       tr = df[get_split_mask(df, "train")]
       tr[PHASE_A_FEATURES].mean(), tr[PHASE_A_FEATURES].std()

   They are frozen (not fit per-window, not re-fit per-run), so train and
   inference see identical transforms and no information from the current
   window's future is used. Scaled inputs are clamped to [-8, 8] so outliers
   (and slow drift of cumulative features like cvd) cannot saturate the GRU.
   If n_features differs from the canonical 10-feature Phase A set, scaling
   falls back to identity (with a logged warning) — recompute constants if you
   change the feature set.

2. Zero-initialized head. The output head starts at all zeros, so at init the
   model predicts mean=0, log_var=0 — exactly the persistence baseline
   (var = 1 for norm_return). Epoch-0 validation NLL should therefore sit at
   the baseline (~0.51), and any learning moves strictly below it.

Loss note: Gaussian NLL per horizon step, averaged with the configured horizon
weighting, per model/INTERFACE.md. Weights sum to 1 over valid (non-NaN)
steps, so the returned "loss" is a true weighted mean NLL — directly
comparable to the persistence baseline (0.509187). (SimpleMLP divided by H a
second time, making its optimized loss ~12x smaller than true NLL; that quirk
is not replicated here.)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Canonical Phase A feature order (model/config/run_config.py PHASE_A_FEATURES):
#   cvd, cvd_quote, vwap_20, vwap_50, anchored_vwap,
#   realized_vol, log_return, norm_return, return_pct, vol_profile_low_bucket
FEATURE_CENTER = [
    -5312299.0,      # cvd
    1.4735266e+08,   # cvd_quote
    74.609734,       # vwap_20
    74.606118,       # vwap_50
    72.056018,       # anchored_vwap
    0.0011595219,    # realized_vol
    2.9733645e-06,   # log_return
    0.000203198,     # norm_return
    3.9777394e-06,   # return_pct
    0.10638564,      # vol_profile_low_bucket
]
FEATURE_SCALE = [
    2928185.2,       # cvd
    3.9530223e+08,   # cvd_quote
    60.33279,        # vwap_20
    60.330386,       # vwap_50
    58.154358,       # anchored_vwap
    0.00081950439,   # realized_vol
    0.0014166867,    # log_return
    1.0072028,       # norm_return
    0.0014187154,    # return_pct
    0.078262108,     # vol_profile_low_bucket
]

_INPUT_CLAMP = 8.0


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


class GRUEncoder(nn.Module):
    """GRU sequence encoder -> single (mean, log_var) head per horizon step."""

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
        self.n_features = n_features
        self.window_length = window_length
        self.horizon = horizon
        self.phase = phase
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if n_features == len(FEATURE_CENTER):
            center = torch.tensor(FEATURE_CENTER, dtype=torch.float32)
            scale = torch.tensor(FEATURE_SCALE, dtype=torch.float32)
        else:
            logger.warning(
                "n_features=%d != %d (canonical Phase A set) — input scaling "
                "falls back to identity. Recompute FEATURE_CENTER/FEATURE_SCALE "
                "if you change the feature set.",
                n_features, len(FEATURE_CENTER),
            )
            center = torch.zeros(n_features, dtype=torch.float32)
            scale = torch.ones(n_features, dtype=torch.float32)
        self.register_buffer("feature_center", center)
        self.register_buffer("feature_scale", scale)

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 2 * horizon)
        # Start exactly at the persistence baseline: mean=0, var=exp(0)=1.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, window_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            window_batch: (B, W, F) tensor

        Returns:
            (mean, log_var) each of shape (B, H)
        """
        x = (window_batch - self.feature_center) / self.feature_scale
        x = torch.clamp(x, min=-_INPUT_CLAMP, max=_INPUT_CLAMP)

        _, h_n = self.gru(x)          # h_n: (num_layers, B, hidden)
        state = h_n[-1]               # (B, hidden) — final layer's last state
        state = self.dropout(state)

        raw = self.head(state)        # (B, 2*H)
        mean = raw[:, : self.horizon]
        log_var = raw[:, self.horizon :]
        return mean, log_var

    def compute_loss(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        config: "RunConfig",  # noqa: F821
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Gaussian negative log-likelihood loss per horizon step.

        loss = weighted mean over valid steps of
               0.5 * ( log_var_i + (target_i - mean_i)^2 / exp(log_var_i) )
        with per-step weights from config.horizon_weighting (renormalized over
        valid steps). NaN targets are masked out (norm_return can be NaN for
        warmup bars).

        Returns (scalar_loss, metrics_dict).
        """
        mean, log_var = outputs
        H = self.horizon

        weighting = getattr(config, "horizon_weighting", "uniform")
        decay_rate = getattr(config, "horizon_decay_rate", 0.9)
        weights = _compute_horizon_weights(H, weighting, decay_rate).to(mean.device)

        valid = ~torch.isnan(targets)

        # Clamp log_var to avoid numerical instability from division by
        # near-zero variance
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        var = torch.exp(log_var)

        squared_error = (targets - mean) ** 2
        per_step_nll = 0.5 * (log_var + squared_error / var)  # (B, H)
        per_step_nll = torch.nan_to_num(per_step_nll, nan=0.0)

        # Weighted mean over valid steps per batch element (weights renormalized
        # over valid entries so the result stays a true weighted average)
        w = weights.unsqueeze(0).expand_as(per_step_nll)
        w_valid = w * valid
        w_sum = w_valid.sum(dim=1).clamp(min=1e-8)
        loss_per_element = (per_step_nll * w_valid).sum(dim=1) / w_sum
        loss = loss_per_element.mean()

        # Unweighted mean NLL per valid step (across entire batch)
        total_valid = valid.sum().float().clamp(min=1)
        nll = (per_step_nll.sum() / total_valid).item()

        # MSE for interpretability (only valid targets)
        mse = (
            squared_error.masked_fill(~valid, 0.0).sum() / total_valid
        ).item()

        # Mean predicted variance
        var_mean = var.mean().item()

        return loss, {
            "loss": round(loss.item(), 6),
            "nll": round(nll, 6),
            "mse": round(mse, 6),
            "var_mean": round(var_mean, 6),
        }
