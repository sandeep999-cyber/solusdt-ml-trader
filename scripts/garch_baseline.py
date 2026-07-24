"""GARCH(1,1) baseline for volatility prediction — walk-forward.

GARCH models volatility clustering using only past squared returns.
No engineered features needed — tests whether vol is inherently
predictable from its own autocorrelation structure.

Walk-forward: 5-fold expanding window, same splits as Ridge/GRU.
Approach: Fit GARCH once per fold on training data, forecast forward.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def compute_metrics(y_true, y_pred, train_mean):
    baseline_mse = np.mean((y_true - train_mean) ** 2)
    model_mse = np.mean((y_true - y_pred) ** 2)
    baseline_rmse = np.sqrt(baseline_mse)
    model_rmse = np.sqrt(model_mse)
    improvement = (1 - model_rmse / baseline_rmse) * 100
    r2 = 1 - model_mse / baseline_mse
    return {
        "baseline_rmse": baseline_rmse,
        "model_rmse": model_rmse,
        "improvement": improvement,
        "r2": r2,
    }


def bootstrap_ci(y_true, y_pred, train_mean, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    imps = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        b_rmse = np.sqrt(np.mean((y_true[idx] - train_mean) ** 2))
        m_rmse = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        imps[i] = (1 - m_rmse / b_rmse) * 100
    return float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))


def garch_fold(train_returns, test_returns, horizon=12):
    """Fit GARCH(1,1) on training data, forecast variance for test period.

    Returns predicted volatility over the horizon window for each test step.
    """
    # GARCH works better with percentage-scale returns
    train_pct = train_returns * 100
    test_pct = test_returns * 100
    n_test = len(test_pct)

    # Fit once on training data
    t0 = time.time()
    model = arch_model(train_pct, vol="Garch", p=1, q=1, mean="Constant", rescale=False)
    result = model.fit(disp="off", show_warning=False)
    fit_time = time.time() - t0

    # GARCH(1,1): sigma2[t] = omega + alpha * eps[t-1]^2 + beta * sigma2[t-1]
    omega = result.params.get("omega", 0)
    alpha = result.params.get("alpha[1]", 0)
    beta = result.params.get("beta[1]", 0)

    # Get last training values
    resid = result.resid
    prev_eps2 = resid[-1] ** 2
    prev_sigma2 = result.conditional_volatility[-1] ** 2

    # Recursive GARCH(1,1) forecast on test period
    forecasts = np.zeros(n_test)
    for t in range(n_test):
        # 1-step ahead forecast
        sigma2_t = omega + alpha * prev_eps2 + beta * prev_sigma2
        forecasts[t] = sigma2_t

        # Update for next step
        eps_t = test_pct[t]
        prev_eps2 = eps_t ** 2
        prev_sigma2 = sigma2_t

    # Convert variance to volatility: sqrt(var) * sqrt(horizon) / 100
    # var is in (percentage)^2, so sqrt(var) is in percentage
    pred_vol = np.sqrt(np.maximum(forecasts, 0)) / 100 * np.sqrt(horizon)

    # Train mean volatility for baseline
    train_vol_mean = np.sqrt(np.mean(train_returns ** 2)) * np.sqrt(horizon)

    return pred_vol, train_vol_mean, fit_time


def garch_walk_forward(returns_1min, horizon=12, n_folds=5):
    """Walk-forward GARCH(1,1)."""
    n_total = len(returns_1min)
    fold_size = n_total // n_folds

    # Clean NaN/inf
    clean_mask = np.isfinite(returns_1min)
    if not clean_mask.all():
        returns_1min = returns_1min.copy()
        returns_1min[~clean_mask] = 0.0

    all_preds = []
    all_targets = []
    fold_results = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_total)

        if test_start >= n_total:
            break

        train_returns = returns_1min[:train_end]
        test_returns = returns_1min[test_start:test_end]

        # Realized volatility: vectorized
        n_test = len(test_returns)
        # cumsum trick for rolling mean of squared returns
        cs2 = np.concatenate([[0], np.cumsum(test_returns ** 2)])
        # mean of squares from i+1 to i+horizon
        roll_mean_sq = (cs2[horizon + 1:] - cs2[1:n_test - horizon + 1]) / horizon
        test_vol = np.sqrt(np.maximum(roll_mean_sq, 0)) * np.sqrt(horizon)

        # GARCH predictions
        pred_vol, train_mean_vol, fit_time = garch_fold(train_returns, test_returns, horizon)

        min_len = min(len(pred_vol), len(test_vol))
        pred_vol = pred_vol[:min_len]
        test_vol = test_vol[:min_len]

        m = compute_metrics(test_vol, pred_vol, train_mean_vol)
        ci = bootstrap_ci(test_vol, pred_vol, train_mean_vol)

        all_preds.extend(pred_vol.tolist())
        all_targets.extend(test_vol.tolist())
        fold_results.append({
            "fold": fold + 1, "metrics": m, "ci": ci, "time": fit_time,
            "n_train": train_end, "n_test": min_len,
        })

        sig = "+" if m["improvement"] > 0 else ""
        print(f"  Fold {fold+1}: train={train_end} test={min_len}")
        print(f"    GARCH(1,1): RMSE={m['model_rmse']:.6f} (base={m['baseline_rmse']:.6f})")
        print(f"    Improvement: {sig}{m['improvement']:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  R²={m['r2']:.6f}")
        print(f"    Fit time: {fit_time:.1f}s")

    # Stacked
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    full_train_mean = np.sqrt(np.mean(returns_1min[:fold_results[-1]["n_train"]] ** 2)) * np.sqrt(horizon)
    stacked_m = compute_metrics(all_targets, all_preds, full_train_mean)
    stacked_ci = bootstrap_ci(all_targets, all_preds, full_train_mean, n_boot=10000)

    return fold_results, stacked_m, stacked_ci


def main():
    print("=" * 70)
    print("GARCH(1,1) BASELINE — WALK-FORWARD")
    print("=" * 70)

    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    returns_1min = df["norm_return"].values.astype(np.float64)
    print(f"Total 1-min returns: {len(returns_1min)}")
    finite = np.sum(np.isfinite(returns_1min))
    nan_count = np.sum(np.isnan(returns_1min))
    inf_count = np.sum(np.isinf(returns_1min))
    print(f"Clean: {finite} finite, {nan_count} NaN, {inf_count} inf")

    print("\nRunning GARCH(1,1) walk-forward (5 folds)...")
    fold_results, stacked_m, stacked_ci = garch_walk_forward(returns_1min)

    # Summary
    print("\n" + "=" * 70)
    print("FOLD-BY-FOLD SUMMARY")
    print("=" * 70)
    print(f"  {'Fold':>4} {'N_train':>7} {'N_test':>6} {'RMSE':>10} {'Base RMSE':>10} {'Improve%':>9} {'95% CI':>18} {'R²':>8}")
    print(f"  {'-'*72}")
    for r in fold_results:
        m = r["metrics"]
        ci = r["ci"]
        sig = "+" if m["improvement"] > 0 else ""
        print(f"  {r['fold']:4d} {r['n_train']:7d} {r['n_test']:6d} "
              f"{m['model_rmse']:10.6f} {m['baseline_rmse']:10.6f} "
              f"{sig}{m['improvement']:8.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] "
              f"{m['r2']:8.6f}")
    sig = "+" if stacked_m["improvement"] > 0 else ""
    total_n = sum(r["n_test"] for r in fold_results)
    print(f"  {'All':>4} {'':>7} {total_n:6d} "
          f"{stacked_m['model_rmse']:10.6f} {stacked_m['baseline_rmse']:10.6f} "
          f"{sig}{stacked_m['improvement']:8.2f} [{stacked_ci[0]:+.2f},{stacked_ci[1]:+.2f}] "
          f"{stacked_m['r2']:8.6f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: GARCH vs Ridge vs GRU (walk-forward stacked)")
    print("=" * 70)
    print(f"  {'Model':<12} {'Stacked Improve%':>17} {'Stacked R²':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Ridge':<12} {'-3.76':>17} {'-0.077':>12}")
    print(f"  {'GRU h32':<12} {'-57.81':>17} {'-1.490':>12}")
    sig = "+" if stacked_m["improvement"] > 0 else ""
    print(f"  {'GARCH(1,1)':<12} {sig}{stacked_m['improvement']:>+16.2f} {stacked_m['r2']:>12.6f}")

    print()
    if stacked_m["improvement"] > 0 and stacked_ci[0] > 0:
        print("  GARCH generalizes. Volatility IS predictable from its own structure.")
        print("  The problem is the 10 features / model class, not volatility itself.")
    elif stacked_m["improvement"] > 0:
        print("  GARCH shows marginal positive signal but CI includes 0.")
        print("  Weak evidence that vol autocorrelation has predictive power.")
    else:
        print("  GARCH fails too. Volatility is close to a random walk at 1-min horizon.")
        print("  The problem is not the features — vol itself is not predictable here.")


if __name__ == "__main__":
    main()
