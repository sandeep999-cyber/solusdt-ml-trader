# Technical Architecture & Modeling Philosophy

This document outlines the model architecture, self-supervised trajectory objective, uncertainty formulation, inference contract, and cloud training workflow for the SOLUSDT Teacher Model.

---

## 1. Core Philosophy

Traditional algorithmic trading models suffer from two primary traps:
1. **Hand-Crafted Human Framing**: Forcing models to predict RSI crossovers, ZigZag swing labels, or fixed directional classes (e.g. `+1` / `-1` classification) introduces human bias and ignores continuous market dynamics.
2. **Uncalibrated Confidence**: Using naive argmax probabilities as trade sizing leads to catastrophic drawdown during market distribution shifts.

### Solution Architecture
- **Phase A (Self-Supervised Future Trajectory)**: The model compresses continuous input features into a bottlenecked state vector using a sequence encoder (`GRUEncoder`), predicting both future return means $\mu_t$ and log variances $\log \sigma^2_t$ across a future horizon $H$.
- **Heteroscedastic Gaussian NLL Loss**:
  $$\mathcal{L}_{\text{NLL}} = \frac{1}{2H} \sum_{h=1}^H \left( \exp(-\log \sigma_h^2) (y_h - \mu_h)^2 + \log \sigma_h^2 \right)$$
- **Phase B (Cost-Aware Decision Head)**: A dedicated decision layer trained on top of Phase A representations with transaction fee penalties and explicit abstention incentives.

---

## 2. Sequence Encoder (`GRUEncoder`)

- **Input Tensor**: Window of length $W$ bars (e.g., $W=60$) containing 10 continuous microstructural features:
  - `cvd`: Cumulative Volume Delta from taker buy volume.
  - `cvd_quote`: CVD in quote currency.
  - `vwap_20`: 20-period Volume-Weighted Average Price.
  - `vwap_50`: 50-period Volume-Weighted Average Price.
  - `anchored_vwap`: Anchored VWAP.
  - `realized_vol`: Rolling realized volatility.
  - `log_return`: Log return.
  - `norm_return`: Log return normalized by rolling volatility.
  - `return_pct`: Percentage return.
  - `vol_profile_low_bucket`: Volume profile low bucket.
- **Normalization Layer**: Per-feature input scaling computed from training split statistics (mean/std, clamped to $\pm 8.0$).
- **Recurrent Body**: Single-layer GRU (hidden dimension 32, ~5K parameters) with dropout ($p=0.2$).
- **Prediction Head**: Zero-initialized linear head producing $2 \times H$ outputs $(\mu_h, \log \sigma^2_h)$.

---

## 3. Inference Contract

The model inference contract is frozen (`model/inference/CONTRACT.md`) to guarantee decoupling between PyTorch modeling code and web client UI consumers.

### Derivation of Uncertainty Bands
Given predicted log variance $\log \sigma^2_h$ and starting price $P_0$:
1. Standard deviation of return: $\sigma_h = \sqrt{\exp(\log \sigma^2_h)}$
2. Predicted return bounds: $r_{\text{upper}} = \mu_h + 2\sigma_h$, $r_{\text{lower}} = \mu_h - 2\sigma_h$
3. Dollar price bounds:
   $$P_{\text{upper}} = P_0 \cdot \exp(r_{\text{upper}})$$
   $$P_{\text{lower}} = P_0 \cdot \exp(r_{\text{lower}})$$

---

## 4. Colab & Cloud Iteration Loop

The model training and checkpoint synchronization loop spans three environments seamlessly:
1. **Local Windows**: Code development in PyTorch / React.
2. **Google Colab (T4 GPU)**: Fast GPU execution, loss logging, mid-run Drive mirroring.
3. **Google Drive Sync**: `scripts/pull_checkpoint.py` automatically pulls checkpoints to `model/runs/` and updates `experiments.md`.
