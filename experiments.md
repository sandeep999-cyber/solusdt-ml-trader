# Experiment Log — SOLUSDT 1m Intraday Volatility Prediction

Auto-populated by `scripts/pull_checkpoint.py`. Each entry is a completed training run synced from Drive. Manual entries added for local experiments.

---

## phaseA_20260722_101708 (NLL run — return trajectory)
- **pulled:** 2026-07-22 12:03:26 UTC
- **model_class:** model.body.gru_encoder.GRUEncoder
- **phase:** A, **window/horizon:** 60/12, **epochs:** 30, **lr:** 0.001
- **hidden_size:** 32, **dropout:** 0.2, **params:** 5,016
- **target:** norm_return (return trajectory, NLL loss)
- **Best val:** NLL=0.493065, MSE=1.015851, var_mean=1.073
- **Baseline delta:** -0.014705 (beats 0.507770 baseline on NLL)
- **Verdict:** MSE ≈ unconditional variance. Variance-shortcut pathology (D011).

## phaseA_20260722_103726 (Fixed-var diagnostic)
- **model_class:** model.body.gru_encoder_fixed_var.GRUEncoderFixedVar
- **target:** norm_return (MSE loss, log_var=0)
- **Best val:** MSE=1.015688
- **Verdict:** Confirms ceiling is real. MSE cannot go below ~1.0157 with these features.

## stride_s1_control (training stride=1, evaluated at stride=60)
- **training_stride:** 1, **val_stride:** 60
- **Best val MSE:** 1.2189
- **Verdict:** Active harm on non-overlapping windows (+20% vs baseline).

## stride_s15_intermediate (training stride=15, evaluated at stride=60)
- **training_stride:** 15, **val_stride:** 60
- **Best val MSE:** 1.2212
- **Verdict:** Training stride has no effect (D015).

## stride_s60_nonoverlap (training stride=60, evaluated at stride=60)
- **training_stride:** 60, **val_stride:** 60
- **Best val MSE:** 1.2211
- **Verdict:** Identical to stride=1 and stride=15. Overlap-exploitation hypothesis refuted (D015).

## Linear baseline (OLS, held-out)
- **script:** scripts/linear_baseline.py (corrected)
- **val_mse:** 1.241 (+1.9% vs baseline)
- **Verdict:** In-sample artifact retracted (D016). Held-out OLS also fails.

## GD-linear vs OLS vs GRU
- **script:** scripts/gd_vs_ols_clean.py
- **OLS val_mse:** 1.241, **GD-linear val_mse:** 1.239, **GRU val_mse:** 1.225
- **Baseline:** 1.217
- **Verdict:** None beat baseline on held-out data (D016).

## Sign prediction (direction, H=12)
- **script:** scripts/sign_prediction.py (corrected)
- **Model val_acc:** 49.0%, **val_AUC:** 0.507
- **Always-positive baseline:** 53.5% (corrected in D018)
- **Verdict:** Model WORSE than trivial. CI [-8.6%, -0.3%]. Direction is dead.

## Shorter horizon sign prediction (H=1,3,5,12)
- **script:** scripts/shorter_horizon_sign.py
- **Best val AUC:** H=1 (0.509) — still noise
- **Verdict:** No horizon achieves AUC > 0.52. Features uninformative for direction (D019).

## Volatility Ridge baseline (H=12)
- **script:** scripts/volatility_ridge.py
- **Target:** sqrt(mean(squared returns))
- **Baseline RMSE:** 0.2517, **Model RMSE:** 0.2232
- **Improvement:** +11.4%, CI [10.3, 12.4], **R²:** 0.068
- **Verdict:** Real signal for volatility. Pivot confirmed (D020).

## Volatility GRU h32 (H=12, 30 epochs)
- **script:** scripts/volatility_gru_train.py
- **Target:** sqrt(mean(squared returns)), MSE loss
- **Baseline RMSE:** 0.2517, **GRU RMSE:** 0.2025
- **Improvement:** +19.6%, CI [17.4, 21.7], **R²:** 0.233
- **GRU vs Ridge:** +9.2%, CI [7.5, 11.0]
- **Verdict:** Nonlinear edge confirmed. GRU explains 23% of volatility variance. Best epoch: 26.

---

## Summary Table

| Experiment | Task | Model | Val Metric | Baseline | Improvement | Verdict |
|---|---|---|---|---|---|---|
| NLL run | Return (NLL) | GRU h32 | NLL=0.493 | 0.508 | -2.9% (NLL) | Variance shortcut |
| Fixed-var | Return (MSE) | GRU fixvar | MSE=1.016 | 1.017 | -0.01% | Ceiling real |
| Stride=1 | Return (MSE) | GRU h32 | MSE=1.219 | 1.017 | +20% | Harmful |
| OLS held-out | Return (MSE) | Linear | MSE=1.241 | 1.217 | +1.9% | Fails |
| GD-linear | Return (MSE) | Linear GD | MSE=1.239 | 1.217 | +1.8% | Fails |
| Sign pred | Direction | LR | AUC=0.507 | 0.535 | -4.5% | Dead |
| H=1,3,5,12 | Direction | LR | AUC<0.51 | 0.50 | ~0% | Dead |
| **Vol Ridge** | **Volatility** | **Ridge** | **RMSE=0.223** | **0.252** | **+11.4%** | **Signal** |
| **Vol GRU** | **Volatility** | **GRU h32** | **RMSE=0.203** | **0.252** | **+19.6%** | **Strong** |
