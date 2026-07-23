# 02 — Absolute Musts for the Agent

These are hard constraints, not suggestions. If a task seems to require violating one of these, stop and flag it rather than proceeding. See `01-project-overview.md` for why these rules exist and `03-current-status-and-next-steps.md` for current task context.

---

## Scope boundaries

1. **Do not modify data layer, UI layer, training loop, run tracking, or checkpointing code to accommodate a model change.** If a new architecture seems to require changing `train.py`, `model/data/loader.py`, `ui/`, or the checkpoint/run-tracking code, the actual fix is almost always to conform the model to the existing interface (`model/INTERFACE.md`), not the reverse. Scaffolding is intentionally frozen. Flag it if a genuine scaffolding change seems required — don't just make it.
2. **All architecture work happens in `model/body/` and `model/heads/` only.** These are the only folders expected to change shape as the project evolves.
3. **`model/inference/CONTRACT.md` and `model/INTERFACE.md` are frozen contracts.** Don't alter field names, types, or shapes in either without it being an explicit, called-out task — both the UI and the training harness depend on these staying stable.

## Data integrity

4. **No lookahead leakage.** Any feature or target computed for a bar at time `t` must only use data from `t` and earlier. This was audited once already (`docs/leakage_audit.md`, `data/pipeline/tests/test_no_leakage.py`) — any new feature or target logic must preserve this and ideally extend the existing test, not just eyeball it.
5. **Splits are chronological, never shuffled.** Train/val/holdout ranges come from `model/config/splits.py` and must never overlap or be randomly sampled across the boundary.
6. **The holdout period is not to be touched during experimentation.** Not used for tuning, not used for early inspection, not used to "just check something." It exists specifically to stay meaningful.

## Modeling constraints (the philosophy, as hard rules)

7. **No hardcoded technical indicators or human-defined pattern labels as training targets.** No RSI, no zigzag-defined swing, no rule-based "this is a trap" label. If a task seems to call for one, it's likely a Phase B decision-layer concern being confused with a Phase A learning target — check `01-project-overview.md`'s Phase A/B split before proceeding.
8. **Phase A predicts a continuous target with uncertainty, not a discrete direction label.** The primary target is now volatility (`sqrt(mean(squared norm_returns))`) — a single scalar per window trained with MSE. The legacy return-trajectory variant outputs `(mean, log_var)` per horizon step with Gaussian NLL. Direction prediction (sign of returns) is NOT learnable from the current feature set (D017-D019). Do not reintroduce a {-1, 0, 1} classification target into Phase A.
9. **The discrete long/short/flat decision belongs to Phase B (`model/heads/`) only**, trained against a cost-aware, abstention-biased reward — never plain accuracy or cross-entropy against a hindsight-correct label.
10. **"Flat" must remain a valid, unpunished action** in any Phase B reward design — never structure a reward that implicitly punishes not trading.

## Process

11. **Run the smoke test before any real training run.** `python -m model.train --config <path> --smoke-test-first` must pass before committing to a full, long-running job.
12. **Never launch or babysit a long training run as an interactive chat action.** Writing/editing training code is the agent's job; executing an hours-long run is a background/terminal process the human owns. Don't simulate "training" results — if asked what a model learned, run the actual code and report real output, or state plainly that the run hasn't happened yet.
13. **Every checkpoint must carry its metadata** (architecture/schema version, feature list, window length, horizon) and go through `model/checkpoints/load.py`'s mismatch guard — never load or save a checkpoint bypassing this.
14. **Don't treat any metric as trustworthy without checking its provenance first.** The baseline NLL was wrong for a while (779,978, later corrected to 0.509187) before anyone caught it — sanity-check new metrics by hand on a small slice before trusting them at scale, the way that fix was verified.

## Honesty

15. **If a task can't be completed within these constraints, say so explicitly rather than working around them silently.** E.g. if fixing a bug seems to genuinely require touching frozen scaffolding, state that plainly and propose the change as its own explicit step — don't quietly do it inside an unrelated task.
