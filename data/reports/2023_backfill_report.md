# 2023 Backfill Report — SOLUSDT 1m

**Generated:** 2026-07-20  
**Symbol:** SOLUSDT  
**Interval:** 1m  
**Source:** `data.binance.vision` (monthly archives)  
**Pipeline version:** v1  

---

## Summary

| Metric | Value |
|---|---|
| Expected bars (365d × 1440m) | 525,600 |
| Actual bars | **525,519** |
| Missing bars | **81** (explained by 2 exchange-side gaps, see below) |
| Duplicate timestamps | **0** |
| Empty months | **0** (12/12) |

The 2023 backfill extends the dataset backwards so the train split covers more
than one market regime (2023: post-FTX recovery, SOL $9.69–$126.36). Validation
and test splits were deliberately left untouched (both remain inside 2024).

---

## Monthly Coverage

| Month | Bars | Expected | Diff | Note |
|---|---|---|---|---|
| 2023-01 | 44,640 | 44,640 | 0 | |
| 2023-02 | 40,319 | 40,320 | −1 | 1-min gap on 2023-02-14 |
| 2023-03 | 44,560 | 44,640 | −80 | 81-min gap on 2023-03-24 (Binance spot halt) |
| 2023-04 | 43,200 | 43,200 | 0 | |
| 2023-05 | 44,640 | 44,640 | 0 | |
| 2023-06 | 43,200 | 43,200 | 0 | |
| 2023-07 | 44,640 | 44,640 | 0 | |
| 2023-08 | 44,640 | 44,640 | 0 | |
| 2023-09 | 43,200 | 43,200 | 0 | |
| 2023-10 | 44,640 | 44,640 | 0 | |
| 2023-11 | 43,200 | 43,200 | 0 | |
| 2023-12 | 44,640 | 44,640 | 0 | |
| **Total** | **525,519** | **525,600** | **−81** | |

---

## Gap Detail (exchange-side, not pipeline errors)

| Gap start (UTC) | Gap end (UTC) | Missing bars | Cause |
|---|---|---|---|
| 2023-02-14 16:38 → 16:40 | 2.0 min | 1 | Brief Binance data gap |
| 2023-03-24 12:39 → 14:00 | 81.0 min | 80 | Binance spot trading halt (matching-engine incident) |

Both are real market events present in the source data. The minute bars simply
do not exist in Binance's own archive. Windows spanning these jumps are left
as-is (crypto trades 24/7; no session handling in v1).

---

## Combined Dataset State (after this backfill)

| Metric | 2023 | 2024 | Combined |
|---|---|---|---|
| Bars | 525,519 | 527,040 | **1,052,559** |
| Range | 2023-01-01 → 2023-12-31 | 2024-01-01 → 2024-12-31 | 2023-01-01 → 2024-12-31 |
| Duplicates | 0 | 0 | 0 |

**Recompute note:** features are computed over the full dataset in one pass, so
this backfill rewrote all processed partitions. Effects on 2024 data:
- `cvd` / `cvd_quote`: cumulative from **2023-01-01** now — absolute levels
  differ from the 2024-only computation (statistics in
  `2024_backfill_report.md` are stale for these two columns; counts and gap
  checks there remain valid).
- Rolling features (`realized_vol`, `norm_return`, `vwap_*`, `return_pct`,
  `vol_profile_low_bucket`): unchanged except the first ~50 bars of Jan 2024,
  which previously were warmup NaNs and are now populated.
- Warmup nulls now sit at the start of **2023** (realized_vol 19, norm_return
  19, return_pct 1, vol_profile 50).

---

## Data Quality Checks

| Check | Result |
|---|---|
| 2023 row count = 525,519 (expected −81, fully explained) | ✅ |
| No duplicate timestamps | ✅ 0 |
| No missing months | ✅ 12/12 |
| Gaps limited to the two documented exchange events | ✅ |
| 2024 partition row count unchanged (no double-parse) | ✅ 527,040 |
| Contract tests (`model/inference/test_contract.py`) against live features | ✅ pass |
| Leakage regression test (`data/pipeline/tests/test_no_leakage.py`) | ✅ pass |

---

## Pipeline Verification

- **Download:** 12 monthly ZIPs, SHA256 checksums verified (`download_range` over 2023 only)
- **Parse:** 2023 archives only, appended into fresh `year=2023` partitions (2024 partitions deliberately not re-parsed — `archive_to_parquet` appends, re-parsing would duplicate rows)
- **Features:** full recompute over 2023+2024 via `compute_features()` (idempotent, deterministic)
- **Splits:** `model/config/splits.py` train start moved 2024-01-01 → 2023-01-01; val/test unchanged. Split sizes: train 876,879 / val 87,840 / test 87,840

---

## Artifact Paths

```
data/raw/binance/klines/1m/          ← 24 ZIP archives (2023 + 2024)
data/raw/parquet/SOLUSDT/1m/         ← Raw parsed Parquet (24 monthly partitions)
data/processed/v1/SOLUSDT/1m/        ← Partitioned features (year=2023|2024/month=XX/)
data/reports/2023_backfill_report.md ← This report
data/reports/2024_backfill_report.md ← 2024 report (see recompute note above)
```

---

## Conclusion

The 2023 backfill is **complete and verified**: 525,519 of 525,600 expected
bars, with all 81 missing bars accounted for by two documented exchange-side
events. The combined 2023+2024 dataset (1,052,559 bars, zero duplicates) is
ready for Phase A training with the extended train split.
