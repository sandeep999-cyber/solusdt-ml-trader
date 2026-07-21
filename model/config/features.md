# Phase A Feature Set — SOLUSDT 1m

**Version:** 1.0  
**Total features:** 10  

---

## Inventory

### 1. CVD (`cvd`) — Cumulative Volume Delta
- **Type:** Taker-flow
- **Window:** Full history
- **Unit:** Base currency (SOL)
- **Null policy:** Never null
- **Use case:** Measures net aggressive buying pressure over time

### 2. CVD Quote (`cvd_quote`)
- **Type:** Taker-flow
- **Window:** Full history
- **Unit:** Quote currency (USDT)
- **Null policy:** Never null
- **Use case:** Same as CVD but in dollar terms, useful for cross-asset comparison

### 3. VWAP 20 (`vwap_20`)
- **Type:** Volume-price
- **Window:** 20 bars (~20 min)
- **Null policy:** Computed from bar 0 (warmup = full cumulative before window fills)
- **Use case:** Short-term volume-weighted average price

### 4. VWAP 50 (`vwap_50`)
- **Type:** Volume-price
- **Window:** 50 bars (~50 min)
- **Null policy:** Same as vwap_20
- **Use case:** Medium-term volume-weighted average price

### 5. Anchored VWAP (`anchored_vwap`)
- **Type:** Volume-price
- **Window:** Month-to-date
- **Anchor:** First bar of each calendar month
- **Null policy:** Never null
- **Use case:** Intra-month fair-value reference

### 6. Realized Volatility (`realized_vol`)
- **Type:** Risk
- **Window:** 20 bars
- **Aggregation:** Standard deviation of log returns
- **Null policy:** First 19 bars are NaN
- **Use case:** Recent price variance

### 7. Log Return (`log_return`)
- **Type:** Return
- **Window:** 1 bar
- **Formula:** `ln(close_i / close_{i-1})`; first bar = 0.0
- **Null policy:** Never null
- **Use case:** Primary return metric (additive across time)

### 8. Norm Return (`norm_return`)
- **Type:** Return
- **Window:** 1 bar, scaled by 20-bar vol
- **Formula:** `log_return / realized_vol`
- **Null policy:** NaN where realized_vol is NaN (first 19 bars)
- **Use case:** Volatility-normalized return signal

### 9. Return % (`return_pct`)
- **Type:** Return
- **Window:** 1 bar
- **Formula:** `(close_i - close_{i-1}) / close_{i-1}`
- **Null policy:** First bar is NaN
- **Use case:** Convenience metric for non-log return

### 10. Volume Profile Low Bucket (`vol_profile_low_bucket`)
- **Type:** Volume
- **Window:** 50 bars (excludes current bar)
- **Buckets:** 10 price deciles
- **Output:** Fraction of window volume in lowest price decile
- **Null policy:** First 50 bars are NaN
- **Use case:** Measures concentration of volume near the low of the range — proxy for support-testing

---

## Excluded (Phase A)

| Feature | Reason |
|---|---|
| `ob_imbalance` | Requires WebSocket L2 book data — Phase B |
| `ob_depth_bid_5` | Same |
| `ob_depth_ask_5` | Same |
| `ob_spread` | Same |

---

## Feature Groups

| Group | Features | Count |
|---|---|---|
| Taker-flow | `cvd`, `cvd_quote` | 2 |
| Volume-price | `vwap_20`, `vwap_50`, `anchored_vwap` | 3 |
| Risk | `realized_vol` | 1 |
| Return | `log_return`, `norm_return`, `return_pct` | 3 |
| Volume | `vol_profile_low_bucket` | 1 |
| **Total** | | **10** |
