# Inference Contract — Phase A

**Version:** 1.1  
**Status:** Frozen  
**Freeze date:** 2026-07-20  

Changes to this contract require a version bump and must not break existing checkpoints.

---

## 1. Input Schema

The inference engine receives a single row with exactly these 10 feature columns, in any order, with the specified dtypes and semantics:

| # | Column | Type | Nullable? | Derivation |
|---|---|---|---|---|
| 1 | `cvd` | float64 | No | Cumulative sum of `taker_buy - taker_sell` from dataset start |
| 2 | `vwap_20` | float64 | No | 20-bar rolling VWAP using typical price `(H+L+C)/3` |
| 3 | `vwap_50` | float64 | No | 50-bar rolling VWAP |
| 4 | `realized_vol` | float64 | No* | 20-bar rolling std of log returns. *First 19 bars are NaN — engine must handle |
| 5 | `log_return` | float64 | No | `ln(close[i] / close[i-1])`; first bar = 0.0 |
| 6 | `norm_return` | float64 | No* | `log_return / realized_vol`. *NaN where realized_vol is NaN |
| 7 | `return_pct` | float64 | No* | `(close[i] - close[i-1]) / close[i-1]`. *First bar is NaN |
| 8 | `vol_profile_low_bucket` | float64 | No* | Fraction of window volume in lowest price decile. *First 50 bars are NaN |
| 9 | `anchored_vwap` | float64 | No | Cumulative VWAP anchored to month start |
| 10 | `cvd_quote` | float64 | No | CVD in quote-currency terms |

**Not included:** `ob_imbalance`, `ob_depth_bid_5`, `ob_depth_ask_5`, `ob_spread` — all NaN and not required for inference. Will be added in Phase B.

---

## 2. Feature Integrity Rules

Computed by `data/pipeline/features.py` (frozen commit). No inference-time feature recomputation is allowed — features are pre-computed once and served from Parquet.

### 2.1 Lookahead Guarantee
No feature at bar `i` reads data from any bar `> i`. See `docs/leakage_audit.md` for the full audit.

### 2.2 Normalization Guarantee
All features are in their raw computation units. The inference engine must apply its own normalization (none is baked into the features).

### 2.3 NaN Handling
The engine must tolerate NaN in `realized_vol`, `norm_return`, `return_pct`, and `vol_profile_low_bucket` for the first N bars of a session. Recommended approach: forward-fill then back-fill with 0.

---

## 3. Output Contract

The inference engine produces per-bar outputs with these fields:

| Field | Type | Description |
|---|---|---|
| `timestamp` | ISO-8601 | Bar open time (UTC) |
| `window_start` | ISO-8601 | First bar of the input window (UTC) |
| `window_end` | ISO-8601 | Last bar of the input window (UTC) |
| `predicted_future_state` | list[dict] | List of `{timestamp, price, lower, upper}` for each horizon step |
| `uncertainty` | float [0, 1] | Scalar — high when vol is high |
| `surprise` | float [0, 1] | Prediction error magnitude at the current bar |
| `decision` | `"long" \| "short" \| "flat"` | Trading decision (trend-following heuristic, not from trained model) |

### 3.1 Confidence Interval Derivation

The `lower` and `upper` fields inside each `predicted_future_state` entry define a
95 % predictive interval for the asset price at that horizon step.

**Assumption:** Log returns are i.i.d. normal with drift μ and volatility σ.
Under this assumption the log-price at step *k* is:

    ln(S_k) = ln(S_0) + μ · k + σ · √k · Z,    Z ~ N(0, 1)

The 95 % interval for ln(S_k) is:

    ln(S_0) + μ · k ± 1.96 · σ · √k

**Current implementation** uses a first-order Taylor approximation of the
exponential, giving a symmetric additive spread:

    spread = 1.96 · σ · S_0 · √k
    lower  = pred_price - spread
    upper  = pred_price + spread

where `pred_price ≈ S_0 · exp(μ + σ · 0.5 · noise)` and the mean-reverting
noise term is deterministic per (idx, step).

**Lognormal reference** (not currently implemented, shown for documentation):

    lower = S_0 · exp(μ · k - 1.96 · σ · √k)
    upper = S_0 · exp(μ · k + 1.96 · σ · √k)

The linear approximation is accurate when `σ · √k ≪ 1`.  At σ ≈ 0.002 and
k = 12, `σ · √12 ≈ 0.007`, so the approximation is tight.

---

## 4. Versioning

| Version | Date | Changes |
|---|---|---|
| 1.1 | 2026-07-20 | Documented `predicted_future_state` output fields and CI derivation formula |
| 1.0 | 2026-07-20 | Initial freeze — 10 features, Phase A |

### 4.1 Contract Check
Each checkpoint stores its contract version. The loader refuses to load a checkpoint whose contract version does not match the current `CONTRACT.md` version.

### 4.2 Bumping
To change the contract:
1. Update this file with the new version and a changelog entry.
2. Increment the version (semver).
3. Update `model/inference/contract_version.py` with the new version string.
4. Re-generate all checkpoints that depend on the old contract.
