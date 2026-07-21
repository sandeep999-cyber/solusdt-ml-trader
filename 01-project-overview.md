# 01 — Project Overview: Goal, Structure, and Why

Read this first for context. It does not contain task instructions — see `02-agent-musts.md` for hard rules and `03-status-and-next-steps.md` for what's done and what's next.

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

**v1 implementation:** model/body/gru_encoder.py (GRUEncoder) — single-layer GRU with hidden size 64, per-feature input scaling (train-split mean/std, clamped ±8), zero-initialized linear head producing (mean, log_var) per horizon step. ~16K parameters. See the file for full design rationale.

**Phase A (self-supervised, no trading yet):**
- **Target**: the continuous `norm_return` values for the next `horizon` steps — a real trajectory, not a discretized label. (Early design had this wrong as a {-1, 0, 1} direction classification — corrected because that reintroduces a hardcoded human framing.)
- **Output**: `(mean, log_var)` per horizon step — heteroscedastic regression. This produces calibrated uncertainty as a direct byproduct of Phase A, with no separate ensemble required to get started (an ensemble can be added later as an independent second uncertainty estimate).
- **Loss**: Gaussian negative log-likelihood per step, averaged over the horizon (with an optional decay so near-term steps count more than far-future ones).
- The UI's `predicted_future_state` lower/upper bands are derived directly from this predicted variance — the exact formula lives in `model/inference/CONTRACT.md`.

**Phase B (decision-making, built after Phase A looks reasonable):**
- The {-1, 0, 1} long/short/flat decision belongs here, not in Phase A. Head 3 consumes Phase A's predicted trajectory + uncertainty and is trained with the cost-aware, abstention-biased reward described above — not cross-entropy against a hindsight-correct direction label.

---

## Project Structure

```
project/
├── configs/                     # run config YAMLs (example.yaml, phase_a_gru.yaml)
├── data/
│   ├── raw/binance/klines/   # downloaded Binance archive zips
│   ├── raw/parquet/           # parsed, partitioned by year/month
│   ├── processed/v1/          # feature tables, partitioned by year/month
│   ├── pipeline/               # download.py, parse.py, features.py, build.py, config.py
│   ├── scripts/                 # synthetic sample data generator (dev/testing only)
│   └── reports/                  # data-quality reports
├── model/
│   ├── body/                    # GRUEncoder (gru_encoder.py), SimpleMLP reference (simple_mlp.py)
│   ├── heads/                    # Phase B decision head goes here
│   ├── config/                    # run_config.py, splits.py, features.md
│   ├── data/loader.py              # CausalWindowDataset — windowing, splits, targets
│   ├── checkpoints/                 # load.py (mismatch guard), latest.json (pointer)
│   ├── inference/                    # engine.py, CONTRACT.md (frozen JSON schema)
│   ├── runs/                          # per-run artifacts + compare.py CLI
│   ├── baselines/                      # persistence_2024.json
│   ├── tests/test_smoke.py              # fast end-to-end wiring test
│   ├── tests/test_gru_encoder.py        # GRU body unit tests
│   ├── INTERFACE.md                      # model contract (forward/compute_loss)
│   └── train.py                           # generic training loop
└── ui/
    ├── backend/                            # FastAPI replay server
    └── frontend/                            # React/TS scrubber
```

**Why it's split this way:** `data/` doesn't know the model exists — it just emits a feature table. `ui/` only knows the JSON contract emitted by inference, not the model's internals. `model/` is deliberately the only folder expected to change shape as architecture evolves — `body/`, `heads/`, and `inference/` are separated precisely so a radical rework stays contained there without touching data or UI. See `02-agent-musts.md` for the hard rule this implies.

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
