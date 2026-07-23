# 01 — Project Overview: Goal, Structure, and Why

Read this first for context. It does not contain task instructions — see `02-agent-musts.md` for hard rules and `03-current-status-and-next-steps.md` for what's done and what's next.

---

## Goal

Build a model that trades SOLUSDT by learning the mechanics of the market itself — not by being handed technical indicators or human-defined patterns. The model should:
- Understand that the market is constantly, if slightly, changing (non-stationary), and adapt rather than assume a fixed distribution.
- Recognize that most of the time, price movement is noise, and default to doing nothing rather than needing a hand-tuned threshold to justify sitting out.
- Discover its own concepts of market state, swings, and traps from raw data — the way a self-supervised model discovers "objects" in video without being told what an object is — rather than being trained against RSI, zigzag-defined swings, or other hardcoded human labels.
- Because backtesting is not trusted as proof that any of this works, the architecture must be robust to distribution shift *by construction*, and validated through calibration and disciplined chronological holdouts rather than a historical P&L number.

---

## How the core ideas turn into design decisions

- **Swings emerge from prediction, not rules.** A model trained to predict future market state naturally attends to persistent moves (predictable) and ignores reverting wiggles (unpredictable, not worth attention). This is the mechanism expected to produce swing-sensitivity without ever defining "swing" as a rule.
- **"Surprise" is the proxy for "this mattered."** How much the model's internal state had to revise, given what it expected, stands in for significance — a big, sticky revision is a real swing; a small, quickly-corrected one is noise.
- **Traps are a special case of the same mechanism.** A trap looks like an early swing (it triggers a revision) but the underlying conditions don't support it. The tell is disagreement between channels — price says "breakout," volume/order-flow says "no real conviction." This is why price alone isn't enough input; volume, order flow, and ideally liquidations are needed for trap-recognition to be learnable at all.
- **Uncertainty should come from the prediction itself, not a bolted-on threshold.** Rather than predicting a single number and thresholding confidence afterward, the model predicts both a mean and a variance — its own stated uncertainty is part of the output, not a separate mechanism.
- **"Don't trade" must be a real, valid output**, not a fallback — achieved by training the decision layer with a cost-aware, abstention-biased reward (transaction costs subtracted, churn penalized, flat unpunished) rather than a simple accuracy target.

---

## Architecture Plan

**v1 target: one model, three heads.** Not the full slow/fast-weight split from early discussion — that's an idea to revisit only if experiments show the model can't keep up with drift, not something to build preemptively.

```
Input window → Body (sequence encoder) → state vector
                                            ├─ Head 1: Future-state prediction (self-supervised, Phase A)
                                            ├─ Head 2: Uncertainty (folded into Head 1 — see below)
                                            └─ Head 3: Decision — long/short/flat (Phase B)
```

**Body**: a sequence encoder (GRU/LSTM to start, not a transformer — simpler, cheaper, easier to debug; complexity should earn its way in through experiments, not be assumed upfront). It compresses the input window into one bottlenecked state vector. The bottleneck is what forces the model to keep predictive signal and drop noise — the same reason an executive summary of a meeting only keeps what mattered.

**v1 implementation:** [gru_encoder.py](file:///d:/ModelProject/model/body/gru_encoder.py) (`GRUEncoder`) — single-layer GRU (hidden=32, 5,016 params), per-feature input scaling (train-split mean/std, clamped ±8), zero-initialized linear head producing `(mean, log_var)` per horizon step. Dropout=0.2 on the GRU final hidden state. Variant [gru_encoder_fixed_var.py](file:///d:/ModelProject/model/body/gru_encoder_fixed_var.py) wraps GRUEncoder with log_var forced to 0 — plain MSE diagnostic to detect variance-shortcut pathology. **Pivot (D020):** Phase A target changed from norm_return trajectory to volatility (`sqrt(mean(squared returns))`). GRU h32 achieves +19.6% RMSE improvement over baseline, R²=0.233, and beats linear Ridge by +9.2%. Direction prediction is dead at all horizons (D017-D019). See `decisions.md` D020-D021 and `experiments.md` for full history.

**Phase A (self-supervised, no trading yet):**
- **Target**: realized volatility — `sqrt(mean(squared norm_returns over next H steps))`. This captures the magnitude of upcoming price movement (volatility clustering), which is predictable from the 10-feature set. Direction (sign of returns) is NOT predictable from these features (D017-D019). The output is a single scalar per window, trained with MSE loss.
- **Output**: `(mean, log_var)` per horizon step for the return-trajectory variant, or a single scalar for the volatility variant. The volatility variant uses plain MSE loss (no NLL). See `model/INTERFACE.md` for both output formats.
- **Loss**: For volatility: MSE on `sqrt(mean(r^2))`. For return trajectory (legacy): Gaussian NLL per step with configurable horizon weighting.
- The UI's `predicted_future_state` lower/upper bands are derived from predicted variance — the exact formula lives in `model/inference/CONTRACT.md`.

**Phase B (decision-making, built after Phase A looks reasonable):**
- The {-1, 0, 1} long/short/flat decision belongs here, not in Phase A. Head 3 consumes Phase A's predicted trajectory + uncertainty and is trained with the cost-aware, abstention-biased reward described above — not cross-entropy against a hindsight-correct direction label.

---

## Project Structure

```
project/
├── colab_train.ipynb            # Colab training notebook (badge-open, no uploads)
├── experiments.md               # append-only experiment journal (auto-populated)
├── configs/
│   ├── example.yaml             # SimpleMLP reference config
│   ├── phase_a_gru.yaml         # GRUEncoder h=32, dropout=0.2 (legacy alias)
│   ├── phase_a_gru_h32.yaml     # GRUEncoder h=32, dropout=0.2 (5K params)
│   └── diag_fixed_var.yaml      # GRUEncoderFixedVar diagnostic (MSE-only, 15 epochs)
├── scripts/
│   ├── pull_checkpoint.py       # Drive → model/runs/ syncer (+ experiments.md append)
│   ├── watch_checkpoints.py     # polls Drive, auto-syncs + analyzes
│   ├── analyze_run.py           # CLI for build_summary (thin wrapper around model/reporting/)
│   ├── volatility_ridge.py      # Ridge baseline for volatility prediction (alpha CV, bootstrap CI)
│   ├── volatility_gru_train.py  # GRU training for volatility (standalone, MSE loss)
│   ├── sign_prediction.py       # Direction prediction diagnostics (corrected baselines)
│   ├── shorter_horizon_sign.py  # Multi-horizon direction test (H=1,3,5,12)
│   └── final_diagnostics.py     # Training AUC + volatility quick test
├── data/
│   ├── raw/binance/klines/      # downloaded Binance archive zips
│   ├── raw/parquet/             # parsed, partitioned by year/month
│   ├── processed/v1/            # feature tables, partitioned by year/month
│   │   └── SOLUSDT/1m/
│   │       ├── year=2023/
│   │       └── year=2024/
│   ├── pipeline/                # download.py, parse.py, features.py, build.py, config.py
│   ├── scripts/                 # synthetic sample data generator (dev/testing only)
│   └── reports/                 # data-quality reports
├── model/
│   ├── body/                    # GRUEncoder (gru_encoder.py), GRUEncoderFixedVar, SimpleMLP, diagnose_overfitting
│   ├── heads/                   # Phase B decision head goes here
│   ├── config/                  # run_config.py, splits.py, features.md
│   ├── data/                    # loader.py — CausalWindowDataset (windowing, splits, targets)
│   ├── checkpoints/             # load.py (mismatch guard), latest.json (pointer, gitignored)
│   ├── inference/               # engine.py (ModelInferenceEngine), CONTRACT.md (frozen JSON schema)
│   ├── reporting/               # summary.py — build_summary() (markdown per run)
│   ├── runs/                    # per-run artifacts (gitignored, synced from Drive)
│   ├── baselines/               # persistence_2024.json (baseline NLL = 0.507770)
│   ├── tests/                   # test_smoke.py, test_gru_encoder.py
│   ├── INTERFACE.md             # model contract (forward/compute_loss)
│   └── train.py                 # generic training loop (git commit, Drive mirror, summary)
└── ui/
    ├── backend/                 # FastAPI replay server (ModelInferenceEngine)
    └── frontend/                # React/TS scrubber
```

**Training loop:** `model/train.py` → `model/data/loader.py` → `model/body/gru_encoder.py`.

**Checkpoint pipeline:** Colab → Drive → Drive for Desktop → `pull_checkpoint.py` → `model/runs/` + `experiments.md`.

**See `03-current-status-and-next-steps.md` § Workflow for the full 15-step iteration loop.**

**Why it's split this way:** `data/` doesn't know the model exists — it just emits a feature table. `ui/` only knows the JSON contract emitted by inference, not the model's internals. `model/` is deliberately the only folder expected to change shape as architecture evolves — `body/`, `heads/`, `reporting/`, and `inference/` are separated precisely so a radical rework stays contained there without touching data or UI. `scripts/` is the operational layer (Drive sync, analysis, watcher) — it talks to `model/reporting/` for logic but has no model internals. `colab_train.ipynb` is the Colab entry point, kept at repo root for single-click badge-open. `experiments.md` is the durable decision journal, auto-populated but human-editable. See `02-agent-musts.md` for the hard rules this implies.

**Existing adjacent infra reused, not rebuilt:**
- [[sol-recorder]] — multi-exchange L2/trade capture; not yet joined into `data/processed/` (order-book columns are NaN placeholders on purpose).
- [[sol-usdt-observatory]] — CVD/VWAP/Volume Profile computation logic, ported into `data/pipeline/features.py` rather than recomputed from scratch.

---

## Terminology

- **"Scaffolding"** (used interchangeably with "harness" in this project): everything except `model/body/` and `model/heads/` — data layer, UI layer, training loop, run tracking, checkpointing, inference contract.
- **Training loop**: specifically `model/train.py`.
- **Data layer**: `data/pipeline/` + `model/data/loader.py`.
- **UI layer**: `ui/backend/` + `ui/frontend/`.
- **Interface/contract**: `model/INTERFACE.md` (model ↔ training loop) + `model/inference/CONTRACT.md` (model ↔ UI).

The intended end state: once scaffolding is done, a normal iteration session is scoped to editing `model/body/`, editing `model/heads/`, and writing a run config. Nothing else should need to change for a typical experiment.
