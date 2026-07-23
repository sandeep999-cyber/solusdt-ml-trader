# Decisions

Architecture and design choices with reasoning. One entry per real reversal or significant choice. When you're cold-starting and wonder "why is horizon 12 and not 1?", look here.

---

## D001: Phase A target — regression, not classification
- **Date:** 2026-07-20
- **Context:** Early design used {-1, 0, 1} direction labels (cross-entropy loss).
- **Reversal:** Changed to continuous `norm_return` regression (Gaussian NLL).
- **Why:** Classification reintroduces a hardcoded human framing — "up/down/flat" is a human concept, not a market mechanic. Regression lets the model learn the full distribution of future returns, including variance (uncertainty). The {-1, 0, 1} decision belongs in Phase B, trained with a cost-aware reward, not bolted onto Phase A.
- **Code:** `gru_encoder.py` outputs `(mean, log_var)` per horizon step.

## D002: Hidden size 64 → 32
- **Date:** 2026-07-22
- **Context:** `phase_a_gru.yaml` originally specified hidden_size=64 (~16K params). Overfitting diagnostics showed h=64 overfit the train split.
- **Reversal:** Changed to hidden_size=32 (5,016 params). `phase_a_gru_h32.yaml` created; `phase_a_gru.yaml` now uses the same defaults.
- **Why:** The 10 Phase A features don't carry enough signal to justify 16K params. h=32 is the smallest GRU that still has a meaningful bottleneck — smaller would compress too much. The fixed-variance diagnostic confirmed even h=32 learns zero signal, so capacity isn't the bottleneck; features are.
- **Code:** `gru_encoder.py` defaults: `hidden_size=32, dropout=0.2`.

## D003: 10 Phase A features (not reduced)
- **Date:** 2026-07-22
- **Context:** Considered reducing to just CVD + price features.
- **Decision:** Keep all 10: `cvd, cvd_quote, vwap_20, vwap_50, anchored_vwap, realized_vol, log_return, norm_return, return_pct, vol_profile_low_bucket`.
- **Why:** The feature set is already minimal — removing any would lose information the model might need. The problem isn't too many features; it's that the GRU can't extract signal from them. Feature reduction should happen after the architecture proves it can learn something, not before.
- **Code:** `run_config.py` `PHASE_A_FEATURES` list.

## D004: Horizon = 12, Window = 60
- **Date:** 2026-07-20
- **Context:** Need to predict the next N steps of norm_return.
- **Decision:** Window=60 bars (1 hour of 1m data), horizon=12 bars (12 minutes ahead).
- **Why:** 60 bars gives the GRU enough context to see recent price action without being so long that old data dilutes the signal. 12-step horizon is short enough that predictions are still correlated with the current state — longer horizons would be pure noise for a 1m model. Both were reasonable starting points; tuning comes after proving the architecture works.
- **Code:** `configs/phase_a_gru_h32.yaml`: `window_length: 60, horizon: 12`.

## D005: Zero-initialized head
- **Date:** 2026-07-20
- **Context:** Model needs to start at the persistence baseline (mean=0, log_var=0 for norm_return).
- **Decision:** Linear head initialized to all zeros.
- **Why:** If the head starts at random, epoch-0 validation NLL is far from baseline, making it hard to tell if learning is happening. Zero-init means the model starts exactly at baseline — any improvement is real learning, any degradation is pathology. This is a standard technique for residual-like heads.
- **Code:** `gru_encoder.py:143-144`: `nn.init.zeros_(self.head.weight)` and `nn.init.zeros_(self.head.bias)`.

## D006: Frozen input scaling (train-split mean/std)
- **Date:** 2026-07-20
- **Context:** Raw features span 1e-3 to 1e8 — feeding them straight broke the SimpleMLP (train_loss oscillated 1e4→0.69).
- **Decision:** Per-feature scaling using train-split mean/std, clamped to ±8. Constants frozen at init, not re-fit per window.
- **Why:** Per-window normalization would leak future information (the window's own mean/std includes future bars). Frozen constants from the train split ensure train and inference see identical transforms. Clamping at ±8 prevents cumulative features (like CVD) from saturating the GRU after drift.
- **Code:** `gru_encoder.py:56-81`: `FEATURE_CENTER`, `FEATURE_SCALE`, `_INPUT_CLAMP`.

## D007: Colab-based training loop (not local)
- **Date:** 2026-07-22
- **Context:** No local GPU available.
- **Decision:** Train on Colab T4 via `colab_train.ipynb`. Code synced via GitHub tarball, data via Google Drive.
- **Why:** Colab T4 gives 16GB VRAM for free. The tarball approach avoids git clone failures in Colab. Drive for Desktop syncs checkpoints back to local Windows for analysis. No manual file transfers at any step.
- **Code:** `colab_train.ipynb` (badge-open, tarball download, subprocess smoke tests).

## D008: NLL loss with optional horizon weighting
- **Date:** 2026-07-20
- **Context:** Need to average NLL over the 12-step horizon.
- **Decision:** Gaussian NLL per step, averaged with configurable horizon weighting (uniform or decay). Weights sum to 1 over valid steps.
- **Why:** Uniform weighting treats all steps equally. Decay weighting lets near-term steps count more, which is useful if the model's predictive power degrades with horizon. The SimpleMLP had a bug where it divided by H twice — GRUEncoder's loss is a true weighted-mean NLL, directly comparable to the persistence baseline.
- **Code:** `gru_encoder.py:167-206`: `compute_loss`, `_compute_horizon_weights`.

## D009: experiments.md as append-only journal
- **Date:** 2026-07-22
- **Context:** Training run results were scattered across chat messages and local files.
- **Decision:** Auto-append one section per run to `experiments.md` at repo root, triggered by `pull_checkpoint.py`.
- **Why:** A single file tells the whole story. Append-only means no merge conflicts. Auto-populated means no manual bookkeeping. Human-editable means annotations can be added after the fact.
- **Code:** `scripts/pull_checkpoint.py` `_append_experiments()`.

## D010: Open-questions tracker
- **Date:** 2026-07-23
- **Context:** Multiple items (strided-validation result, fixed-variance MSE numbers, Phase B design, order-book join, holdout sufficiency) were raised in conversation but dropped without resolution because each new message pulled attention to something newer.
- **Decision:** Create `open-questions.md` with one line per item: what's open, when it was raised, what would close it. Check at the start of every session.
- **Why:** The cheapest fix for the most expensive recurring problem — items don't get forgotten because they don't matter, they get forgotten because context windows are finite. A file checked at session start catches this directly.
- **Code:** `open-questions.md` (repo root).

## D011: Mean prediction floor — fixed-variance diagnostic rules out training-schedule fixes
- **Date:** 2026-07-23
- **Context:** NLL run (`phaseA_20260722_101708`) achieved 2.9% baseline delta (best val NLL 0.493065). The question: was this from learning the mean, or from inflating variance as a shortcut?
- **Diagnostic:** Ran `GRUEncoderFixedVar` (`phaseA_20260722_103726`) — same architecture, but log_var locked to 0 (plain MSE). Forces the model to improve the mean or fail.
- **Result:** Fixed-var best MSE: 1.015688. NLL run best MSE: 1.015851. Difference: 0.000163 (0.016%). The NLL run's entire baseline delta came from inflating var_mean to ~1.07, not from improving the mean at all.
- **Conclusion:** The mean prediction is at its floor (~1.0157 MSE) with GRU h32 + 10 Phase A features at 1-minute granularity. No training-schedule cleverness (warmup, LR annealing, etc.) will move it. This is a feature/timescale problem — either the features don't carry enough signal, or 1-minute bars are too noisy for the model to extract it. Next experiments: feature reduction (CVD+price only) or coarser timescale (5-min bars).
- **Code:** `model/runs/phaseA_20260722_101708/metrics.jsonl`, `model/runs/phaseA_20260722_103726/metrics.jsonl`.

## D012: Overfitting is real, not a window-overlap artifact
- **Date:** 2026-07-23
- **Context:** The h=64 model showed high train/val gap. Was this genuine overfitting, or an artifact of stride=1 validation (where windows overlap by 59/60 bars)?
- **Diagnostic:** Ran `diagnose_overfitting.py` on the fixed-variance checkpoint, comparing val NLL at stride=1 vs stride=60.
- **Result:**
  - stride=1: NLL=0.5078, MSE=1.0157 (at baseline)
  - stride=60: NLL=0.6089, MSE=1.2178 (20% worse than unconditional variance)
  - Baseline: NLL=0.507834, var=1.015785
- **Conclusion:** The model's mean prediction is **harmful** on non-overlapping windows — it's doing worse than just predicting the mean. This confirms hidden=32 was solving the right problem (capacity reduction was necessary), but the real bottleneck is the feature set. The model is exploiting window-overlap patterns that don't generalize. Next step: feature reduction (CVD+price only) to test whether fewer, more informative features improve generalization.
- **Code:** `model/body/diagnose_overfitting.py`.

## D013: Validation stride — fix the methodology, not just the result
- **Date:** 2026-07-23
- **Context:** Stride=60 diagnostic revealed the model's mean prediction is 20% worse than baseline on non-overlapping windows (D012). But the comparison was done outside the training loop — the actual `train.py` validation used stride=1, meaning every prior run's val metrics (including D011's "ceiling is real" comparison) were evaluated under the same optimistic methodology.
- **Diagnostic:** Re-ran fixed-variance vs NLL comparison at stride=60. Both models tie: MSE 1.2175 vs 1.2178 (difference: 0.0003, noise). D011's conclusion survives honest evaluation — the ceiling is real. But both models are equally bad at stride=60.
- **Fix:** Added `stride` parameter to `CausalWindowDataset` and `create_dataloader` (`model/data/loader.py`). Validation now uses `stride=window_length` (60), producing non-overlapping windows. Training still uses stride=1. Smoke test confirmed: val_loss=0.595 at stride=60 vs ~0.508 at stride=1.
- **Why:** Stride=1 validation inflates metrics because overlapping windows let the model exploit temporal proximity patterns that don't generalize. Any future experiment evaluated under stride=1 would produce misleadingly optimistic results. This is a scaffolding-level fix — it affects every future run, not just one experiment.
- **Code:** `model/data/loader.py` (CausalWindowDataset, create_dataloader), `model/train.py:365` (val_loader stride).

## D014: Models actively harmful on fresh data — not just "no signal"
- **Date:** 2026-07-23
- **Context:** Stride=60 diagnostic showed both models at MSE 1.2175 vs baseline 1.0158 (20% worse). But n=1,463 non-overlapping windows is thin — could be regime-specific noise.
- **Diagnostic:** Bootstrap CI (10,000 resamples) on per-window MSE at stride=60.
- **Result:**
  - NLL run MSE: 1.2175, 95% CI [1.1905, 1.2442]
  - Fixed-var MSE: 1.2178, 95% CI [1.1916, 1.2443]
  - Diff: -0.0003, 95% CI [-0.038, 0.038] (includes 0)
  - Both CIs **comfortably exclude** baseline var (1.0158) — lower bound 1.19 is 17% above baseline
- **Conclusion:** The finding is statistically significant. Both models are **actively harmful** on non-overlapping windows — not just "no signal," but memorized overlap patterns that hurt predictions on fresh data. This is more serious than a ceiling: the model learned something from training that makes it worse than doing nothing on genuinely unseen windows. The high per-window std (0.52) indicates regime-specific variation, but even the lower CI bound excludes baseline. This likely reflects training on stride=1 windows teaching the model to exploit temporal redundancy that doesn't generalize. Worth investigating whether training itself needs coarser stride, not just validation.
- **Code:** `model/runs/phaseA_20260722_101708/checkpoints/best.pt`, `model/runs/phaseA_20260722_103726/checkpoints/best.pt`.

## D015: Training stride has no effect — harm is fundamental
- **Date:** 2026-07-23
- **Context:** D014 confirmed both models are actively harmful at stride=60. Hypothesis: stride=1 training teaches overlap-exploitation. Test: three configs (stride=1,15,60), all evaluated at stride=60 with bootstrap CIs.
- **Result:**
  - stride=1: MSE=1.2179, 95% CI [1.1916, 1.2446]
  - stride=15: MSE=1.2202, 95% CI [1.1939, 1.2469]
  - stride=60: MSE=1.2200, 95% CI [1.1936, 1.2467]
  - All three CIs overlap. Pairwise diffs < 0.003. All +20% vs baseline.
- **Conclusion:** Training stride has no effect on the harm. The hypothesis that stride=1 training teaches overlap-exploitation is **refuted**. The model is actively harmful regardless of training window construction. This points to a deeper problem: either the 10-feature set is fundamentally uninformative for this task, or the GRU encoder architecture is wrong for this data.
- **Code:** `scripts/compare_stride_experiment.py`, `configs/stride_s1_control.yaml`, `configs/stride_s15_intermediate.yaml`, `configs/stride_s60_nonoverlap.yaml`.

## D016: OLS comparison was in-sample — retracted
- **Date:** 2026-07-23
- **Context:** `linear_baseline.py` reported OLS val_mse=0.894, "-12% vs baseline," suggesting features have signal but GRU destroys it. This was cited as evidence that "architecture is the problem."
- **Error:** The OLS was fit on the val set and evaluated on the same val set (in-sample). When fit on train and evaluated on val (held-out): val_mse=1.241, **+1.9% vs baseline**. The "12% improvement" was an in-sample artifact.
- **Corrected result:**
  - OLS held-out: val_mse=1.241, +1.9% vs baseline
  - Linear GD held-out: val_mse=1.239, +1.8% vs baseline
  - GRU held-out: val_mse=1.225, +0.7% vs baseline
  - Baseline: 1.217
  - **None of them beat baseline on held-out data.**
- **Conclusion:** The 10-feature set has no genuine predictive power for 12-step-ahead norm_return. The earlier "12% improvement" was wrong. The GRU isn't "too expressive" — there's nothing to learn. The features don't predict this target.
- **Retraction:** D017 (not committed) proposed "architecture is the problem" based on the in-sample OLS result. That conclusion is retracted. The correct conclusion is that the features lack signal for this task.
- **Code:** `scripts/linear_baseline.py` (has in-sample bug), `scripts/gd_vs_ols_clean.py` (corrected held-out comparison).

## D017: Sign prediction — NULL result, features uninformative for direction
- **Date:** 2026-07-23
- **Context:** D016 showed 10-feature set has no power for magnitude prediction. Test: can it predict direction (sign) instead? Logistic regression on same features, same splits, stride=60.
- **Result (initial, baseline corrected in D018):**
  - Class balance: 50.0% positive (train), 53.5% (val) — nearly balanced
  - ~~Majority class baseline: accuracy=0.465 (always predict negative)~~ **WRONG — corrected in D018**
  - Lag-1 persistence baseline: accuracy=0.502
  - Logistic regression: accuracy=0.485, AUC=0.507
  - vs majority: ~~+2.0%~~ **-4.5% (corrected)**
  - Top features: all `realized_vol` at different window positions with alternating signs (noise fitting)
- **Conclusion:** The 10-feature set has no directional information at the 12-step horizon. Neither magnitude (D016) nor direction (D017) can be predicted. The features are genuinely uninformative for this task. **See D018 for corrected baselines and D019 for shorter horizons.**
- **Code:** `scripts/sign_prediction.py`.

## D018: Majority baseline corrected — model worse than trivial
- **Date:** 2026-07-23
- **Context:** D017 used minority class (46.5%) as baseline. Correct baseline is always-positive (53.5%).
- **Corrected result:**
  - Always-positive: accuracy=0.535 (true majority)
  - Lag-1 persistence: accuracy=0.502
  - Logistic regression: accuracy=0.490, AUC=0.507
  - vs always-positive: **-4.5%, 95% CI [-8.6%, -0.3%]** — excludes 0. Model is SIGNIFICANTLY worse.
- **Diagnostic:** Training accuracy=0.554, AUC=0.572. Signal exists in training data but doesn't generalize. Non-stationarity, not total absence of signal.
- **Leakage audit:** No look-ahead leak (shifted target test passed). All MI < 0.01.
- **Conclusion:** Model learns spurious pattern that bets against prevailing drift. Val accuracy worse than trivial always-positive.
- **Code:** `scripts/sign_prediction.py` (corrected version).

## D019: Shorter horizons — still nothing
- **Date:** 2026-07-23
- **Context:** D018 showed no signal at H=12. Test: does signal appear at shorter horizons?
- **Result (stride=H, non-overlapping):**

| H | N_train | Train AUC | Val AUC | Always-pos | Delta |
|---|---------|-----------|---------|------------|-------|
| 1 | 876,819 | 0.582 | 0.509 | 0.480 | +2.8% |
| 3 | 292,273 | 0.544 | 0.505 | 0.503 | -0.6% |
| 5 | 175,363 | 0.546 | 0.508 | 0.506 | -0.2% |
| 12 | 73,068 | 0.544 | 0.502 | 0.503 | -0.2% |

- **Best AUC: H=1 (0.509)** — still noise. Train AUC 0.582 → val 0.509 is pure overfitting.
- **H=1 baseline shift:** At H=1, always-positive=48% (more 1-step negative returns), not 53.5%. The +2.8% delta is baseline artifact, not signal.
- **Conclusion:** No horizon achieves meaningful AUC (<0.52). The 10-feature set is definitively uninformative for directional prediction at any horizon.
- **Code:** `scripts/shorter_horizon_sign.py`.

## D020: Volatility pivot — Ridge confirms signal, nonlinear edge exists
- **Date:** 2026-07-23
- **Context:** D017-D019 showed features are uninformative for direction. D018 diagnostic showed marginal signal in training data (train AUC 0.582). User suggested testing volatility prediction as alternative target.
- **Target:** `sqrt(mean(squared returns over next H steps))` — realized volatility.
- **Ridge results (stride=H, non-overlapping val, CORRECTED):**

| H | Improvement | 95% CI | R² |
|---|-------------|--------|-----|
| 1 | +0.33% | [0.24, 0.42] | 0.007 |
| 3 | +1.70% | [1.53, 1.87] | 0.034 |
| 5 | +2.67% | [2.38, 2.96] | 0.053 |
| 12 | **+4.85%** | **[4.25, 5.43]** | **0.095** |

- **GRU vs Ridge (H=12, stride=60 val, CORRECTED):**

| Model | RMSE | Improvement | R² |
|-------|------|-------------|-----|
| Baseline | 0.2517 | --- | 0.000 |
| Ridge | 0.2243 | +4.85% | 0.095 |
| GRU h32 | **0.2025** | **+19.6%** | **~0.354** |
| GRU vs Ridge | | **+9.2%** | CI [7.5, 11.0] |

- **Conclusion:** Single-split shows real but modest signal for volatility (Ridge R²=0.095). GRU shows strong nonlinear edge (+19.6%). **BUT: walk-forward (D022) shows Ridge stacked R²=-0.077.** Single-split results are not robust. GRU walk-forward still needed.
- **Pivot decision:** Kill all direction tasks. Volatility forecasting is the new target. The features capture information about upcoming return magnitude (volatility clustering), not direction.
- **Code:** `scripts/volatility_ridge.py`, `scripts/volatility_gru_train.py`.

## D022: Metric correction + walk-forward — single-split results are not robust
- **Date:** 2026-07-23
- **Context:** D020 R² values used inconsistent baselines. Corrected formula: R² = 1 - (1 - improvement/100)² where improvement is % RMSE reduction. Corrected single-split R² is actually higher (Ridge 0.095, GRU ~0.354). But walk-forward (5-fold expanding window) tells a different story.
- **Pattern of single-split reversals:** This is the THIRD time a single-split result reversed under honest evaluation:
  1. NLL run: "beat baseline" on stride=1 → 20% harmful at stride=60
  2. Fixed-var MSE: "tied" at stride=1 → diverged at stride=60
  3. **Ridge volatility: +4.85% single-split → -3.76% walk-forward**
- **Walk-forward Ridge (H=12):**
  - Fold 1: -0.36%, Fold 2: -19.69%, Fold 3: +1.86%, Fold 4: +5.35%, Fold 5: -5.04%
  - Stacked: RMSE=0.242, improvement=-3.76%, R²=-0.077
  - Signal does NOT survive temporal distribution shift
- **Implication for GRU:** The GRU's un-walk-forward-tested +19.6% / R²≈0.354 should be treated as the LEAST trustworthy number. Given the pattern of 3 reversals, expect GRU walk-forward to also evaporate or reverse.
- **R² formula confirmed:** `improvement` = % RMSE reduction = (1 - rmse_model/rmse_baseline) * 100. Therefore R² = 1 - (rmse_model/rmse_baseline)² = 1 - (1 - improvement/100)². The squared relationship is correct.
- **Decision:** Run GRU walk-forward next (not GARCH) — it's cheaper and determines whether there's any nonlinear edge worth comparing against an econometric baseline.

## D021: Loader volatility target support
- **Date:** 2026-07-23
- **Context:** D020 showed volatility is the correct target. Need pipeline support for volatility target in the training harness.
- **Changes:**
  - Added `target_type: str = "return"` field to `RunConfig` (supports "return" or "volatility")
  - Modified `CausalWindowDataset.__getitem__()` to compute `sqrt(mean(tgt^2))` when `target_type="volatility"`, returning shape (1,) instead of (horizon,)
- **Impact:** Existing runs unaffected (default is "return"). New volatility runs use `target_type: "volatility"` in config.
- **Code:** `model/config/run_config.py`, `model/data/loader.py`.
