# Open Questions

Check this file at the **start** of every session. If anything here is stale, surface it before starting new work.

Format: one line per item. What's open, when it was raised, what would close it.

---

## Active

- **Volatility GRU hyperparameter sweep** — GRU h32 achieves +19.6% RMSE improvement, R²=0.233. Next: tune hidden_size (16, 32, 64), dropout (0.1, 0.2, 0.3), learning rate, training stride. Also test if training stride=60 (non-overlapping) improves generalization. Close: run 5-10 configs, report best RMSE + R² on val. (Raised: 2026-07-23)

- **Order-book feature join** — `ob_imbalance`, `ob_depth_bid_5`, `ob_depth_ask_5`, `ob_spread` are NaN placeholders in `features.py`. The [[sol-recorder]] data exists but hasn't been joined. Close: implement the join in `data/pipeline/features.py`, verify no NaN rows remain in the OB columns, and re-run the feature pipeline. (Raised: 2026-07-20, still open)

- **Holdout sensitivity analysis** — The original concern was "is 2024 alone enough holdout?" A partial mitigation was applied: 2023 was added to the training set (now 20 months), while val/test stayed fixed at Sep–Nov and Nov–Jan 2024–2025. But the sensitivity analysis itself was never run: does val/test behavior actually change with train-set length? Without that test, we don't know if the extension helped or was just a checkbox. Close: train on 2023 only, evaluate on 2024-09+, and compare val metrics to the full-train run. If they're similar, the holdout is sufficient regardless of train length. (Raised: 2026-07-21)

- **Phase B reward design** — The {-1, 0, 1} decision head needs a cost-aware, abstention-biased reward (transaction costs subtracted, churn penalized, flat unpunished). No design exists yet. Close: write a `decisions.md` entry specifying the reward function, test it on synthetic data, and confirm it produces meaningful abstention. (Raised: 2026-07-20, still open)

## Resolved

- **Fixed-variance diagnostic MSE numbers** — NLL run best MSE: 1.015851. Fixed-var run best MSE: 1.015688. Difference: 0.000163 (0.016%). The NLL run's entire 2.9% baseline delta came from inflating var_mean to ~1.07, not from improving the mean. **Ceiling is real** — GRU h32 + 10 features cannot push MSE below ~1.0157. Feature/timescale problem, not a training shortcut. (Closed: 2026-07-23)

- **Strided-validation diagnostic** — Ran `diagnose_overfitting.py` on the fixed-variance checkpoint. Results:
  - stride=1 (full overlap): NLL=0.5078, MSE=1.0157 (at baseline)
  - stride=60 (non-overlapping): NLL=0.6089, MSE=1.2178 (**20% worse than unconditional variance**)
  - Baseline: NLL=0.507834, var=1.015785
  - **Conclusion: the "overfitting" is real, not a window-overlap artifact.** The model's mean prediction is harmful on non-overlapping windows — it's doing worse than just predicting the mean. This confirms hidden=32 was solving the right problem (capacity), but the real bottleneck is the feature set, not the architecture. (Closed: 2026-07-23)

- **Validation methodology fix** — Re-ran fixed-var vs NLL comparison at stride=60: both models tie (MSE 1.2175 vs 1.2178, difference 0.0003). D011's "ceiling is real" survives honest evaluation. Fixed `model/data/loader.py` to accept `stride` parameter; validation now uses stride=60 (non-overlapping). Every future run's val metrics will now be honest. (Closed: 2026-07-23)

- **Statistical significance of harmful-on-fresh-data finding** — Bootstrap CI (10,000 resamples, n=1,463 per model) on stride=60 per-window MSE. Both models' CIs [1.19, 1.24] comfortably exclude baseline var (1.0158) by 17%+. The finding is statistically significant: both models are actively harmful on non-overlapping windows, not just noise. High per-window std (0.52) indicates regime-specific variation but doesn't explain away the gap. This is more serious than a ceiling — the model learned something from stride=1 training that makes it worse on fresh data. (Closed: 2026-07-23, raised training-stride question in Active)

- **Training stride question** — Ran three configs (stride=1,15,60), all evaluated at stride=60. All three models' CIs overlap (MSE ~1.218-1.220, all +20% vs baseline). Training stride has no effect on the harm. The hypothesis that stride=1 training teaches overlap-exploitation is **refuted**. (Closed: 2026-07-23)

- **OLS comparison was in-sample** — `linear_baseline.py` reported OLS val_mse=0.894 (-12% vs baseline). ERROR: OLS was fit on val and evaluated on same val (in-sample). Held-out: val_mse=1.241 (+1.9% vs baseline). The "12% improvement" was an artifact. All models (OLS, linear GD, GRU) fail to beat baseline on held-out data. The 10-feature set has no genuine predictive power for 12-step norm_return. (Closed: 2026-07-23)

- **Sign prediction** — Logistic regression on 10 features, 12-step sign prediction. Accuracy: 48.5%, AUC: 0.507. Fails to beat baselines (majority class: 46.5%, persistence: 50.2%). Bootstrap CIs include 0 for both comparisons. Top features are all `realized_vol` with alternating signs (noise fitting). Features have no directional information at 12-step horizon. (Closed: 2026-07-23)

- **Shorter horizons** — Tested H=1,3,5,12 with stride=H (non-overlapping). Best val AUC: H=1 (0.509) — still noise. Train AUC 0.582 → val 0.509 is pure overfitting. H=1 always-positive baseline is 48% (not 53.5%) — the +2.8% delta is baseline artifact, not signal. No horizon achieves AUC > 0.52. Features are definitively uninformative for directional prediction at any horizon. (Closed: 2026-07-23)

- **OLS comparison retracted** — `linear_baseline.py` reported OLS val_mse=0.894 (-12%). ERROR: fit on val, evaluated on same val (in-sample). Held-out OLS: val_mse=1.241 (+1.9%). Retracted in D016. (Closed: 2026-07-23)

- **Feature reformulation for direction** — 10-feature set tested for direction at all horizons (H=1,3,5,12). All AUC < 0.52. Resolved: features have no directional signal. Pivot to volatility (D020) succeeded. (Closed: 2026-07-23)
