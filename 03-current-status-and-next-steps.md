# 03 — Current Status and Next Steps

See `01-project-overview.md` for what this project is and why it's structured this way, and `02-agent-musts.md` for hard constraints to follow while working on any of this.

---

## Current Status

**Scaffolding: functionally complete.** All of the following exist and are tested:
- Data pipeline: download (Binance archives, checksum-verified) → parse (Parquet, partitioned) → features (CVD from taker-buy proxy, VWAP, realized vol, norm_return, volume profile) → CLI orchestrator. Verified against real data. Dataset now covers **2023-01-01 → 2024-12-31** (1,052,559 bars): 2023 backfill completed 2026-07-20 (525,519 bars; only two real gaps — 1 min on 2023-02-14, 81 min on 2023-03-24 during the Binance spot halt; zero duplicates; 2024 data untouched at 527,040 bars). Note: CVD is cumulative from dataset start, so absolute cvd levels differ from the 2024-only computation — rolling features are unchanged.
- Causal window data loader with chronological, non-overlapping train/val/holdout splits, plus a lookahead-leakage audit and regression test.
- Generic training harness (`train.py`) — architecture-agnostic, depends only on the two-method model interface (`forward`, `compute_loss`).
- Run tracking (`model/runs/{run_name}/`: config snapshot, metrics.jsonl, checkpoints), checkpoint metadata + mismatch guard, `latest.json` pointer.
- UI backend + frontend scrubber, with checkpoint hot-reload (polls `latest.json` mtime, plus manual `POST /reload`) so the UI can pick up new checkpoints without a restart. (Engine currently simulated — `SimulatedInferenceEngine` uses heuristics, not model weights. Model-backed inference is the next step after training.)
- Run comparison CLI (`python -m model.runs.compare`).
- Frozen inference JSON contract (`model/inference/CONTRACT.md`), including the lower/upper-from-variance derivation formula.
- 29/29 tests passing (11 smoke + 7 GRU body + 7 contract + 3 pipeline + 1 parse).

**Resolved: persistence baseline NLL bug.** Originally 779,978 — implausibly large. Root cause: the baseline used `realized_vol` (or its square) as the predicted variance for `norm_return`, but `norm_return` is already volatility-normalized, so its true unconditional variance is close to 1.0 — using raw `realized_vol` (~0.001, min 0.00015) as variance instead inflated the NLL 100–1000×. Fix: `persistence_model.py` now uses `norm_return.var()` (≈1.0185) as the constant baseline variance. Recomputed and hand-verified value: **0.507770 NLL per prediction** (recomputed 2026-07-21 on the correct `val-eval` split; previous 0.509187 was on a now-meaningless `test` split). `baseline_delta` in `metrics.jsonl` is now trustworthy.

**Resolved: 2024-only history question.** Train split extended backwards to 2023-01-01 (`splits.py`); val/test windows deliberately untouched (still 2024-09→2024-11 and 2024-11→2025-01) so the holdout stays meaningful. Split sizes now: train 876,879 / val 87,840 / test 87,840.

**Resolved: all three hardening items (done 2026-07-20, before first real run), plus two more found while wiring the GRU:**
- NaN/Inf guard in `train.py` — halts with a clear message (checkpoints saved so far are preserved); grad clipping (max_norm=1.0) added alongside.
- Data version + train range/row-count now recorded in every checkpoint under `"data"`.
- Config validation at startup — bad `feature_columns`, non-positive window/horizon, or unknown `horizon_weighting` fail fast with a clear error.
- **Baseline-key fix:** `train.py` looked up `"cross_entropy_loss"` in the baseline file, but `persistence_2024.json` stores `"nll"` — `baseline_delta` was silently never logged. Fixed; smoke run now logs `baseline_loss: 0.507770`.
- **`_validate` tuple crash:** it computed `(outputs > 0)` unconditionally, which raises `TypeError` on Phase A `(mean, log_var)` tuples — any Phase A run would have died at the first validation. Now accuracy is only computed for models exposing `predict()`.

**Resolved: first real Phase A body built.** `model/body/gru_encoder.py` (GRUEncoder): fixed per-feature input scaling (constants = train-split mean/std, clamped ±8 — raw features span 1e-3..1e8 and broke the SimpleMLP run, whose train_loss oscillated 1e4→0.69), single-layer GRU (hidden 64, ~16K params), final hidden state → one linear head → `(mean, log_var)` per horizon step. Head is zero-initialized so the model starts exactly at the persistence baseline. 29/29 tests pass (7 new in `model/tests/test_gru_encoder.py`); smoke run completes with sane metrics (val NLL 0.525 vs baseline 0.509 at epoch 0, as expected pre-training). Note: GRUEncoder's loss is a true weighted-mean NLL; SimpleMLP divided by H a second time (its optimized loss was ~12× true NLL) — documented in the body file, not replicated. Old runs under `model/runs/phaseA_20260720_1043*` are classification-era artifacts (val 0.6931 = ln 2) — do not compare against them.

**Resolved: scaffolding audit (19 findings, all fixed 2026-07-21).** Full phased cleanup:
- Phase 1 (data-safety): `build.py --force` now deletes parquet/feature dirs before re-parse (no more dedup); `generate_sample_data.py` retargeted to `sample/` path with guard against live data.
- Phase 2 (training harness): checkpoint `_save_checkpoint` writes `CONTRACT_VERSION` (not hardcoded "1.0"); pointer semantics fixed (smoke runs skip pointer, best saves point to `best.pt`); resume guard (`_restore_best_val_loss` + strict `load_checkpoint` call); feature column guard switched from set to ordered-list comparison; `_validate` now accumulates `nll`/`mse`/`var_mean` sub-metrics and uses avg_nll for `baseline_delta`; `compare.py` selects min-loss entry.
- Phase 3 (baseline): recomputed 0.507770 on correct `val-eval` split; `features.py` fixed duplicate log line and hardcoded VERSION.
- Phase 4 (tests): `pytest.ini` with testpaths/pythonpath; new `test_parse.py` for force-re-parse idempotence; new `test_checkpoint_guard_rejects_feature_order_mismatch`.
- Phase 5 (UI backend): `/series` default limit 5000 (was no limit); `/inference` no longer returns `all_decisions`; engine NaN-vol guard fixed in `_compute_details`.
- Phase 6 (UI frontend): `api.ts` passes limit param; `App.tsx` uses default limit 5000; `PriceChart.tsx` click handler uses O(n) loop instead of `reduce` (faster on 5K window).
- Phase 7 (hygiene): stale `model/checkpoints/latest.json` deleted; `.gitignore` extended (runs/, sample/, pytest_cache/); `example.yaml` comment corrected; unused `get_available_range` removed from `data_service.py`.
- Phase 8 (verification): 29/29 pass, 9.3 s.

**Deliberately deferred (not oversights):**
- Order-book feature join (`ob_imbalance`, `ob_depth_*`, `ob_spread`) from [[sol-recorder]] — placeholder NaN columns, not required for Phase A.
- Phase B reward/target design for the decision head — coupled to model output, scoped as model-design work, not scaffolding.
- CONTRACT.md doc says v1.1 while `contract_version.py` is "1.0" (doc/code drift; harmless — the guard compares checkpoint vs code constant, both "1.0"). Reconcile when the contract next changes for real (e.g. model-backed inference).
- The UI's inference engine is still `SimulatedInferenceEngine` (heuristic, no weights) — hot-reload watches `latest.json` but reloads the simulator. Step "UI inspection of the trained model" requires a model-backed engine in `model/inference/engine.py` (contained change; not yet built).
- `tests/test_smoke.py` still imports `CONTRACT_VERSION` directly rather than testing the integration path — fine for a smoke test, but the contract-inference integration (`model/inference/test_contract.py`) covers the full pipeline separately.
- `model/checkpoints/latest.json` has been deleted (was a stale smoke-test pointer). It will be re-created by the next training run's best checkpoint save. Until then, the UI shows "No checkpoint found" — expected.
- `train.py`'s optimizer section (line 352) hardcodes `torch.optim.Adam` with only `lr` from config — no `weight_decay`, no AdamW, no scheduler. This constrains architecture iteration: you cannot add weight decay, change optimizer, or apply LR scheduling without editing `train.py` (scaffolding). If the current run confirms capacity/regularization as the lever, exposing `optimizer_kwargs` from config would unblock the next step without breaking existing runs. (Flagged per 02-agent-musts.md rule 15: flag scaffolding gaps rather than working around them silently.)

---

## Workflow / Iteration Loop (how any future session should run)

1. Edit `model/body/` (architecture) and/or `model/heads/` (Phase B decision head, once reached).
2. Write or copy a run config (`model/config/run_config.py` fields — phase, window_length, horizon, feature_columns, split_config_ref, batch_size, lr, optimizer, num_epochs, seed, model_class as dotted path, horizon_weighting, notes).
3. Run smoke test first: `python -m model.train --config path.yaml --smoke-test-first`.
4. Run the real training job: `python -m model.train --config path.yaml`. Long-running, launched from terminal/background.
5. Watch `model/runs/{run_name}/metrics.jsonl`; compare against baseline_delta and across past runs via `python -m model.runs.compare`.
6. Point the UI scrubber at the run (auto-picks up `latest.json`) and manually inspect: does the predicted trajectory look reasonable, does uncertainty behave sensibly, does anything resembling swing-sensitivity or trap-caution show up (expect weak/absent until Phase B exists — early Phase A runs are just about "does it beat the baseline and look non-degenerate").
7. Iterate on `model/body/`/`model/heads/` based on what's observed; scaffolding should not need to change during this loop.

---

## Immediate Next Steps (in order)

1. ~~Hardening items~~ — done (see Resolved above).
2. ~~Confirm backfill~~ — done: 2023+2024 verified (counts, gaps, duplicates).
3. ~~Split strategy~~ — done: train starts 2023-01-01, val/test untouched.
4. ~~Build the real Phase A body~~ — done: `model/body/gru_encoder.py` (GRUEncoder), 29/29 tests, smoke run passed.
5. ~~Scaffolding audit (19 findings)~~ — done, see Resolved above.
6. **Launch the first real Phase A training run** (human-owned, ~2 h on CPU):
   `python -m model.train --config configs/phase_a_gru.yaml --smoke-test-first`
   Then watch `model/runs/{run_name}/metrics.jsonl` — success = val NLL clearly below the 0.507770 baseline (negative `baseline_delta`), no NaN halt.
7. Build the model-backed inference engine in `model/inference/engine.py` (loads best checkpoint via `model/checkpoints/load.py`, emits the frozen JSON contract) and inspect the trained model in the UI scrubber: does the predicted trajectory look reasonable, does uncertainty rise in choppy periods, does anything resembling swing-sensitivity show up (expect weak/absent until Phase B).
8. Only after Phase A looks reasonable: design Phase B's decision head and its cost-aware reward.

**Scaffolding is done, audit complete, baseline recomputed, the real GRU body is in place and smoke-tested — the only thing left before the first long run is pressing enter on it.**
