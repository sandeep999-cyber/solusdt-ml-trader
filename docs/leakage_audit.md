# Leakage Audit — Feature Pipeline v1

**Date:** 2026-07-20  
**Audited file:** `data/pipeline/features.py`  
**Method:** Line-by-line code review  

---

## Summary

**Verdict: No lookahead leakage found.**

Every feature at bar `i` uses data only from bars `[0..i]` (historical + current). No column reads data from bar `>i`. The dataset is safe for supervised learning where the target is bar `i+1` or later.

---

## Feature-by-Feature Audit

### 1. CVD (`cvd`, `cvd_quote`) — Lines 128–133

**Implementation:** Cumulative sum of `taker_buy - taker_sell` from bar 0.

```
cvd[i] = Σ_{k=0..i} (taker_buy[k] - (volume[k] - taker_buy[k]))
```

**Lookahead?** ✅ No. Uses only bars `[0..i]`. Correct for cumulative-online features.

---

### 2. Rolling VWAP (`vwap_20`, `vwap_50`) — Lines 136–137

**Implementation:** `_rolling_vwap_jit` maintains a sliding window sum of `price * volume` and `volume`. At bar `i`:
- If `i < window`: uses `cum_pv[0..i] / cum_v[0..i]` — warmup period.
- If `i >= window`: subtracts bar `i-window` from the accumulators.

```
vwap_w[i] = Σ_{k=i-window+1..i} (price[k] * volume[k]) / Σ_{k=i-window+1..i} volume[k]
```

**Lookahead?** ✅ No. Window is `[i-window+1 .. i]`. Includes current bar `i`, which is acceptable for features serving the next-bar prediction.

---

### 3. Anchored VWAP (`anchored_vwap`) — Lines 140–146

**Implementation:** Grouped by month. Within each group, `cumsum()` from the first bar of the month to bar `i`.

```
anchored_vwap[i] = Σ_{k=first_of_month..i} (close[k] * volume[k]) / Σ_{k=first_of_month..i} volume[k]
```

**Lookahead?** ✅ No. Cumulative from month start to bar `i`. Current bar `i` is the last included.

---

### 4. Realized Volatility (`realized_vol`) — Lines 148–149

**Implementation:** `_realized_vol_jit` computes rolling standard deviation of log returns over a 20-bar window.

```
realized_vol[i] = std(log_return[i-window+1 .. i])
```

Includes `log_return[i] = ln(close[i] / close[i-1])` — the current bar's return.

**⚠️ Design note:** This includes the current bar's return in the vol computation. This is **not lookahead** — at bar `i`, `close[i]` is known. When used as a feature for predicting `return[i+1]`, using `close[i]` is legitimate.

**Boundary warmth:** `log_return[0]` is hardcoded to `0.0` (line 126). This slightly compresses the first valid realized_vol estimate (bars 0..19). Effect is negligible on 527K bars.

**Lookahead?** ✅ No.

---

### 5. Norm Return (`norm_return`) — Line 154

**Implementation:**
```
norm_return[i] = log_return[i] / realized_vol[i]
```

**Statistical coupling:** `realized_vol[i]` contains `log_return[i]` in its window, so the same return appears in both numerator and denominator. This can compress norm_return on high-vol bars. This is a feature-engineering concern, not a leakage concern.

**Lookahead?** ✅ No.

---

### 6. Return % (`return_pct`) — Line 157

**Implementation:** `pct_change()` = `(close[i] - close[i-1]) / close[i-1]`.

**Lookahead?** ✅ No.

---

### 7. Volume Profile (`vol_profile_low_bucket`) — Lines 159–162

**Implementation:** `_volume_profile_jit` partitions bar `i`'s lookback window into 10 price buckets by volume. The window is exclusive of current bar:

```
chunk_p = price[i - window : i]      # i.e., bars [i-50 .. i-1]
```

**Lookahead?** ✅ No. Explicitly excludes current bar. Window is `[i-window .. i-1]`.

---

### 8. Placeholder OB Columns (`ob_*`) — Lines 165–168

**Implementation:** All set to `NaN`. These are future slots for WebSocket order-book data.

**Lookahead?** ✅ N/A. No computation.

---

## Pipeline Invocation Context

The `compute_features()` function reads the full Parquet dataset at once (`pq.ParquetDataset(src).read()`), so features are computed across the entire time series — not per-month in isolation. This means CVD and anchored VWAP are continuous across month boundaries, which is correct behavior.

## Conclusion

| Feature | Lookahead? | Notes |
|---|---|---|
| `cvd` | ✅ No | Cumulative-online |
| `cvd_quote` | ✅ No | Cumulative-online |
| `vwap_20` | ✅ No | Rolling window, includes current bar |
| `vwap_50` | ✅ No | Rolling window, includes current bar |
| `anchored_vwap` | ✅ No | Monthly cumulative |
| `realized_vol` | ✅ No | Rolling window, includes current bar |
| `log_return` | ✅ No | Current vs previous close |
| `norm_return` | ✅ No | Statistically coupled, not leaked |
| `return_pct` | ✅ No | Current vs previous close |
| `vol_profile_low_bucket` | ✅ No | Window excludes current bar |
| `ob_*` | ✅ N/A | Placeholder |

**No changes required.**
