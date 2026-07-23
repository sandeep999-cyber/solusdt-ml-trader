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

**Resolved: Model-backed Inference Engine built & UI integrated (2026-07-22).**
- `model/inference/engine.py` (`ModelInferenceEngine`) implemented: loads checkpoints via `model/checkpoints/load.py` / `latest.json`, computes windowed GRU batch inference, and produces contract-compliant JSON.
- `ui/backend/state.py` updated to instantiate `ModelInferenceEngine` instead of `SimulatedInferenceEngine`.

**Resolved: Cloud/Colab training harness & automated checkpoint sync (2026-07-22).**
- Created `colab_train.ipynb` for GPU training runs.
- Added `scripts/pull_checkpoint.py`, `scripts/watch_checkpoints.py`, and `scripts/analyze_run.py` to automatically sync checkpoints from Google Drive, write `latest.json`, and log training runs into `experiments.md`.

**Resolved: First Phase A training runs & overfitting diagnostics (2026-07-22).**
- Run `phaseA_20260722_101708`: `GRUEncoder` (hidden size 32, ~5K params), achieved best val NLL **0.493065** (baseline delta **-0.014705** vs 0.507770 baseline).
- Run `phaseA_20260722_103726`: `GRUEncoderFixedVar` (hidden size 32, fixed variance baseline head), achieved best val NLL **0.491322** (baseline delta **-0.016448**).
- Created `model/body/diagnose_overfitting.py` to evaluate train vs val generalization across stride intervals, identifying high-capacity overfitting issues at h=64 and confirming h=32 stability.

**Resolved: Loop hardening — reproducibility, resilience, auto-documentation (2026-07-22).**

Five friction points eliminated from the Colab → Drive → local training loop:

1. **Git commit provenance.** Every run records the exact code that produced it.
   - `train.py` calls `_git_commit()`: reads `GIT_COMMIT` env var (set by Colab Cell 1 via GitHub API), falls back to `git rev-parse HEAD` locally.
   - Written to `run_dir/provenance.json` at run start (separate from `config.yaml` so hyperparams stay pure).
   - Embedded in every checkpoint under `data_meta.git_commit`.
   - Colab Cell 1 fetches the commit SHA from `api.github.com/repos/.../commits/main` and sets `os.environ['GIT_COMMIT']`.

2. **Mid-run Drive mirror.** A Colab disconnect no longer loses the entire run.
   - `train.py` has `_mirror_to_drive()`: copies `best.pt`, `metrics.jsonl`, `config.yaml`, and `summary.md` to Drive whenever `DRIVE_CKPT_DIR` env var is set (non-smoke-runs only).
   - Triggered on every best save AND every periodic epoch checkpoint (every 5 epochs).
   - Colab Cell 1 sets `os.environ['DRIVE_CKPT_DIR'] = DRIVE_CKPT`.
   - Cell 7 still runs a final idempotent sweep — mid-run mirror ensures ≤5 epochs lost on disconnect.
   - `pull_checkpoint.py` Unicode `✓` → `[OK]` fixed (Windows cp1252 crash).

3. **Auto `summary.md` per run.** One markdown file tells the whole story — no more stitching from chat.
   - New module `model/reporting/summary.py` with `build_summary(run_dir) -> str`.
   - Contents: metadata (git commit, model class, hyperparams), data provenance, best checkpoint metrics, full epoch trajectory table, analysis (MSE trend, variance-shortcut flags, baseline delta).
   - Written to `run_dir/summary.md` at end of `train()` (non-smoke only).
   - `scripts/analyze_run.py` refactored to use `build_summary`; gained `--write` flag.
   - `pull_checkpoint.py` syncs `{run}_summary.md` ↔ `summary.md`.

4. **Append-only `experiments.md`.** Durable experiment journal in the repo.
   - New `experiments.md` at repo root with header template.
   - `pull_checkpoint.py` appends one section per run when `summary.md` exists — includes full summary + a concise `> best_loss/epoch | mse | var_mean` result line.
   - Deduplicated: checks if run-section header already exists before appending.

5. **Data copy caching.** Cell 1 skips the 1–2 min Drive copy when data hasn't changed.
   - `_drive_fingerprint()` computes a JSON fingerprint of all `year=*` partitions (mtime + total parquet size per partition).
   - Saved to `.drive_fingerprint` after copy; compared on next session.
   - Copy skipped if fingerprint matches AND `year=` partitions already present locally.
   - Junk files (non-`year=` items, including leftover notebook copies) still cleaned up each session.

**New files created:**
- `model/reporting/__init__.py`, `model/reporting/summary.py` — `build_summary()`
- `experiments.md` — append-only experiment journal (repo root)

**Deliberately deferred (not oversights):**
- Order-book feature join (`ob_imbalance`, `ob_depth_*`, `ob_spread`) from [[sol-recorder]] — placeholder NaN columns, not required for Phase A.
- Phase B reward/target design for the decision head — coupled to model output, scoped as model-design work, not scaffolding.
- CONTRACT.md doc says v1.1 while `contract_version.py` is "1.0" (doc/code drift; harmless — the guard compares checkpoint vs code constant, both "1.0"). Reconcile when the contract next changes for real (e.g. model-backed inference).
- `tests/test_smoke.py` still imports `CONTRACT_VERSION` directly rather than testing the integration path — fine for a smoke test, but the contract-inference integration (`model/inference/test_contract.py`) covers the full pipeline separately.
- `train.py`'s optimizer section (line 352) hardcodes `torch.optim.Adam` with only `lr` from config — no `weight_decay`, no AdamW, no scheduler. This constrains architecture iteration: you cannot add weight decay, change optimizer, or apply LR scheduling without editing `train.py` (scaffolding). If the current run confirms capacity/regularization as the lever, exposing `optimizer_kwargs` from config would unblock the next step without breaking existing runs. (Flagged per 02-agent-musts.md rule 15: flag scaffolding gaps rather than working around them silently.)

---

## Workflow / Iteration Loop (Colab-based)

The full loop spans three environments: local Windows (code edits), Colab T4 (GPU training), Google Drive (data + checkpoints). No manual file transfers at any step.

### Step-by-step

1. **Edit code locally** in `model/body/`, `model/data/`, `model/config/`, or `configs/*.yaml`. Also update `colab_train.ipynb` if the notebook itself needs changes.

2. **Push to GitHub:**
   ```bash
   git add -A
   git commit -m "describe change"
   git push
   ```

3. **Open Colab from the badge** in `colab_train.ipynb` (always fetches latest notebook from GitHub). Runtime → T4 GPU.

4. **Cell 0**: Mount Drive (`drive.mount('/content/drive')`).

5. **Cell 1**: Download latest code tarball from GitHub → record `GIT_COMMIT` via GitHub API → set `DRIVE_CKPT_DIR` env var → copy feature data from Drive to local SSD (skipped if fingerprint cache matches) → clean junk files.

6. **Cell 2**: Verify GPU (T4, ~16 GB VRAM).

7. **Cell 3**: Install deps (pandas, pyarrow, pyyaml — torch is preinstalled).

8. **Cell 4**: Smoke test — 2 epochs, truncated batches. Validates config, data loading, model init, and a few forward/backward passes. Run name gets `_smoketest` suffix; does NOT write to Drive or `experiments.md`, does NOT write `summary.md`.

9. **Cell 5**: Full Phase A training (30 epochs):
   - `provenance.json` written at start (git commit, timestamp)
   - `config.yaml` snapshot in run directory
   - Every 5 epochs, on best val loss, and at final epoch: checkpoint saved locally AND mirrored to Drive (`_mirror_to_drive`)
   - `summary.md` written at end (non-smoke)
   - If Colab disconnects mid-run: latest best.pt + metrics.jsonl already on Drive — worst case 5 epochs lost

10. **Cell 6a–6b**: Diagnostic run with `GRUEncoderFixedVar` (15 epochs, same mirror behavior).

11. **Cell 7**: Final full-sweep copy to Drive (idempotent — mid-run mirror may have already written these files). Copies: `{run}_best.pt`, `{run}_metrics.jsonl`, `{run}_config.yaml`, `{run}_summary.md`, `latest.json`, `best.pt`.

12. **Locally**: Google Drive for Desktop auto-syncs `G:\My Drive\ModelProject\checkpoints\`.

13. **Sync and analyze:**
    ```bash
    python scripts/pull_checkpoint.py        # Drive → model/runs/ + experiments.md
    python scripts/analyze_run.py            # print (and --write) summary.md
    ```
    Or leave running:
    ```bash
    python scripts/watch_checkpoints.py      # polls Drive every 30s, auto-syncs + analyzes
    ```

14. **Read** `experiments.md` or per-run `summary.md` for results. Decide next change.

15. **Repeat from step 1.**

### Data that persists across sessions

| Data | Location | Method |
|---|---|---|
| Feature parquet files | `G:\My Drive\ModelProject\year=*/` | Permanent; never modified by training |
| Checkpoints + metrics | `G:\My Drive\ModelProject\checkpoints/` | Written by Colab Cells 5/6/7; synced by Drive for Desktop |
| Code | GitHub `sandeep999-cyber/solusdt-ml-trader` | git push; Colab downloads tarball |
| Run artifacts (local) | `model/runs/{run_name}/` | Synced by `pull_checkpoint.py` |
| Experiment journal | `experiments.md` (repo root) | Auto-appended by `pull_checkpoint.py` |
| Colab VM files | `/content/ModelProject/` | Destroyed when Colab session ends |

### What each run directory contains

```
model/runs/phaseA_YYYYMMDD_HHMMSS/
  config.yaml          ← frozen hyperparameter snapshot
  provenance.json      ← {run_name, git_commit, started}
  metrics.jsonl        ← one JSON line per epoch (train_loss, val_loss, nll, mse, var_mean, baseline_delta)
  summary.md           ← auto-generated markdown (metadata, trajectory table, variance-shortcut flags)
  checkpoints/
    best.pt            ← best val loss checkpoint (overwritten)
    epoch_0000.pt      ← periodic checkpoints (epoch 0, 5, 10, ...)
```

---

## Immediate Next Steps (in order)

1. ~~Hardening items~~ — done (see Resolved above).
2. ~~Confirm backfill~~ — done: 2023+2024 verified (counts, gaps, duplicates).
3. ~~Split strategy~~ — done: train starts 2023-01-01, val/test untouched.
4. ~~Build the real Phase A body~~ — done: `model/body/gru_encoder.py` (GRUEncoder), 29/29 tests, smoke run passed.
5. ~~Scaffolding audit (19 findings)~~ — done, see Resolved above.
6. ~~Launch the first real Phase A training run~~ — done: `phaseA_20260722_101708` (val NLL 0.493065) & `phaseA_20260722_103726` (val NLL 0.491322) both beat the 0.507770 baseline.
7. ~~Build the model-backed inference engine~~ — done: `ModelInferenceEngine` in `model/inference/engine.py` integrated into `ui/backend/state.py`.
8. ~~Loop hardening~~ — done: git commit provenance, mid-run Drive mirror, auto summary.md per run, experiments.md journal, data copy fingerprint caching. See Resolved section above.
9. **Phase A Architecture Investigation**: The first two runs show MSE ≈ unconditional variance (~1.0185) — the GRU h32 with 24 technical-indicator features has learned zero predictive signal. The variance-shortcut pathology (NLL run improved baseline delta by −2.5% purely through variance inflation to ~1.09) was confirmed by the fixed-variance diagnostic run (MSE identical at ~1.016). Possible directions: add order-book features (currently NaN placeholders), try more complex architectures, or accept this as a feature ceiling and move to Phase B with a calibrated noise model.
10. **UI Inspection & Trajectory Evaluation**: Run the UI scrubber (`launch.bat`) and inspect model-backed predictions, trajectory shapes, and uncertainty bounds on the 2024 val dataset.
11. **Phase A Refinement / Phase B Design**: Evaluate Phase A architecture tweaks (window size, input features, regularization) vs proceeding to Phase B decision head and cost-aware reward.
