"""GARCH(1,1) baseline for volatility prediction - walk-forward (corrected).

Key correction: GARCH's 1-step conditional variance is iterated h steps
forward via the recursion sigma2(t+h) = omega + (alpha+beta)*sigma2(t+h-1),
and the implied variances are summed and averaged. This correctly accounts
for mean-reversion dynamics, unlike the previous sqrt(horizon) scaling.

Target: sqrt(mean(r^2)) over next 12 steps - same as Ridge/GRU.
Baseline: std of realized vol targets in training set.
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

HORIZON = 12


def realized_vol_targets(returns, horizon=HORIZON, stride=1):
    """Compute realized vol targets: sqrt(mean(r[t+1:t+1+h]^2))."""
    n = len(returns)
    end = n - horizon
    if end <= 0:
        return np.array([])
    indices = np.arange(0, end, stride)
    # Vectorized: for each index, compute mean of squared returns over horizon
    # Use cumsum trick
    r2 = returns ** 2
    cs2 = np.concatenate([[0], np.cumsum(r2)])
    # mean of squares from i+1 to i+horizon for each i in indices
    means = (cs2[indices + horizon + 1] - cs2[indices + 1]) / horizon
    return np.sqrt(np.maximum(means, 0))


def compute_metrics(y_true, y_pred, baseline_rmse):
    model_mse = np.mean((y_true - y_pred) ** 2)
    baseline_mse = baseline_rmse ** 2
    model_rmse = np.sqrt(model_mse)
    improvement = (1 - model_rmse / baseline_rmse) * 100
    r2 = 1 - model_mse / baseline_mse
    return {
        "baseline_rmse": baseline_rmse,
        "model_rmse": model_rmse,
        "improvement": improvement,
        "r2": r2,
    }


def bootstrap_ci(y_true, y_pred, baseline_rmse, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    imps = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        m_rmse = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        imps[i] = (1 - m_rmse / baseline_rmse) * 100
    return float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))


def garch_fold(train_returns, test_returns, horizon=HORIZON):
    """Fit GARCH(1,1) on training data, multi-step forecast for test period.

    For each test point t:
      1. Compute sigma2(t+1) from GARCH recursion
      2. Iterate h-1 more steps: sigma2(t+1+k) = omega + (alpha+beta)*sigma2(t+k)
      3. Average the h implied variances -> avg_var
      4. Predicted vol = sqrt(avg_var)

    This correctly handles mean-reversion: for persistent processes (alpha+beta
    close to 1), multi-step variance is higher than h * 1-step variance because
    shocks persist.
    """
    train_pct = train_returns * 100
    test_pct = test_returns * 100
    n_test = len(test_pct)

    t0 = time.time()
    model = arch_model(train_pct, vol="Garch", p=1, q=1, mean="Constant", rescale=False)
    result = model.fit(disp="off", show_warning=False)
    fit_time = time.time() - t0

    omega = result.params.get("omega", 0)
    alpha = result.params.get("alpha[1]", 0)
    beta = result.params.get("beta[1]", 0)

    # Last training state
    resid = result.resid
    prev_eps2 = resid[-1] ** 2
    prev_sigma2 = result.conditional_volatility[-1] ** 2

    # Multi-step forecast for each test point
    forecasts = np.zeros(n_test)
    for t in range(n_test):
        # sigma2(t+1) = omega + alpha*eps(t)^2 + beta*sigma2(t)
        s2 = omega + alpha * prev_eps2 + beta * prev_sigma2
        cum_var = s2
        prev_s2 = s2
        for step in range(1, horizon):
            prev_s2 = omega + (alpha + beta) * prev_s2
            cum_var += prev_s2
        forecasts[t] = cum_var / horizon  # average variance over horizon

        # Update with actual observation
        eps_t = test_pct[t]
        prev_eps2 = eps_t ** 2
        prev_sigma2 = omega + alpha * prev_eps2 + beta * prev_sigma2

    # Predicted vol: sqrt(avg_var) / 100 (from percentage scale)
    pred_vol = np.sqrt(np.maximum(forecasts, 0)) / 100

    return pred_vol, fit_time, omega, alpha, beta


def garch_walk_forward(returns_1min, horizon=HORIZON, n_folds=5):
    """Walk-forward GARCH(1,1)."""
    n_total = len(returns_1min)
    fold_size = n_total // n_folds

    # Clean NaN/inf
    clean_mask = np.isfinite(returns_1min)
    if not clean_mask.all():
        returns_1min = returns_1min.copy()
        returns_1min[~clean_mask] = 0.0

    # Pre-compute all realized vol targets (stride=1 for full resolution)
    r2_all = returns_1min ** 2
    cs2_all = np.concatenate([[0], np.cumsum(r2_all)])
    indices_all = np.arange(0, n_total - horizon)
    means_all = (cs2_all[indices_all + horizon + 1] - cs2_all[indices_all + 1]) / horizon
    all_rv = np.sqrt(np.maximum(means_all, 0))
    print(f"  Realized vol targets: {len(all_rv)} total")

    fold_results = []
    all_preds = []
    all_targets = []

    for fold in range(n_folds):
        train_end_rv = (fold + 1) * fold_size  # in 1-min returns space
        test_start_rv = train_end_rv
        test_end_rv = min(test_start_rv + fold_size, n_total)

        if test_start_rv >= n_total:
            break

        train_returns = returns_1min[:train_end_rv]
        test_returns = returns_1min[test_start_rv:test_end_rv]

        # Baseline: std of all training realized vol targets (stride=60 for consistency with Ridge/GRU)
        stride = 60
        train_r2 = train_returns ** 2
        cs2t = np.concatenate([[0], np.cumsum(train_r2)])
        end_t = len(train_returns) - horizon
        if end_t > 0:
            idx_t = np.arange(0, end_t, stride)
            means_t = (cs2t[idx_t + horizon + 1] - cs2t[idx_t + 1]) / horizon
            train_rv = np.sqrt(np.maximum(means_t, 0))
        else:
            train_rv = np.array([1.0])
        baseline_rmse = np.std(train_rv) if len(train_rv) > 0 else 1.0

        # Test realized vol targets (stride=1 for full resolution)
        # all_rv[i] = realized vol of returns[i+1:i+1+horizon]
        # For test returns starting at test_start_rv, the first valid target is all_rv[test_start_rv]
        test_rv_start = test_start_rv
        test_rv_end = min(test_end_rv, len(all_rv))
        test_rv = all_rv[test_rv_start:test_rv_end]

        # GARCH predictions
        t0 = time.time()
        pred_vol, fit_time, omega, alpha, beta = garch_fold(train_returns, test_returns, horizon)

        min_len = min(len(pred_vol), len(test_rv))
        pred_vol = pred_vol[:min_len]
        test_rv = test_rv[:min_len]

        m = compute_metrics(test_rv, pred_vol, baseline_rmse)
        ci = bootstrap_ci(test_rv, pred_vol, baseline_rmse)

        all_preds.extend(pred_vol.tolist())
        all_targets.extend(test_rv.tolist())
        fold_results.append({
            "fold": fold + 1, "metrics": m, "ci": ci, "time": fit_time,
            "n_train": train_end_rv, "n_test": min_len,
            "alpha": alpha, "beta": beta, "persistence": alpha + beta,
        })

        sig = "+" if m["improvement"] > 0 else ""
        print(f"\n  Fold {fold+1}: train={train_end_rv} test={min_len} (fit {fit_time:.1f}s)")
        print(f"    GARCH(1,1): alpha={alpha:.4f} beta={beta:.4f} persistence={alpha+beta:.4f}")
        print(f"    RMSE={m['model_rmse']:.6f} (base={m['baseline_rmse']:.6f})")
        print(f"    Improvement: {sig}{m['improvement']:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  R2={m['r2']:.6f}")

    # Stacked
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    stacked_rmse_base = np.std(all_targets)
    stacked_m = compute_metrics(all_targets, all_preds, stacked_rmse_base)
    stacked_ci = bootstrap_ci(all_targets, all_preds, stacked_rmse_base, n_boot=10000)

    return fold_results, stacked_m, stacked_ci


def main():
    print("=" * 70)
    print("GARCH(1,1) BASELINE - WALK-FORWARD (CORRECTED)")
    print("=" * 70)

    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    returns_1min = df["norm_return"].values.astype(np.float64)
    print(f"Total 1-min returns: {len(returns_1min)}")

    print("\nRunning GARCH(1,1) walk-forward (5 folds)...")
    fold_results, stacked_m, stacked_ci = garch_walk_forward(returns_1min)

    # Summary
    print("\n" + "=" * 70)
    print("FOLD-BY-FOLD SUMMARY")
    print("=" * 70)
    print(f"  {'Fold':>4} {'N_train':>7} {'N_test':>6} {'RMSE':>10} {'Base RMSE':>10} {'Improve%':>9} {'95% CI':>18} {'R2':>8} {'a+b':>5}")
    print(f"  {'-'*80}")
    for r in fold_results:
        m = r["metrics"]
        ci = r["ci"]
        sig = "+" if m["improvement"] > 0 else ""
        print(f"  {r['fold']:4d} {r['n_train']:7d} {r['n_test']:6d} "
              f"{m['model_rmse']:10.6f} {m['baseline_rmse']:10.6f} "
              f"{sig}{m['improvement']:8.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] "
              f"{m['r2']:8.6f} {r['persistence']:5.3f}")
    sig = "+" if stacked_m["improvement"] > 0 else ""
    total_n = sum(r["n_test"] for r in fold_results)
    avg_persist = np.mean([r["persistence"] for r in fold_results])
    print(f"  {'All':>4} {'':>7} {total_n:6d} "
          f"{stacked_m['model_rmse']:10.6f} {stacked_m['baseline_rmse']:10.6f} "
          f"{sig}{stacked_m['improvement']:8.2f} [{stacked_ci[0]:+.2f},{stacked_ci[1]:+.2f}] "
          f"{stacked_m['r2']:8.6f} {avg_persist:5.3f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: GARCH vs Ridge vs GRU (walk-forward stacked)")
    print("=" * 70)
    print(f"  {'Model':<12} {'Stacked Improve%':>17} {'Stacked R2':>12}")
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
        print("  GARCH fails. Volatility is close to a random walk at 1-min horizon.")
        print("  The problem is not the features - vol itself is not predictable here.")


if __name__ == "__main__":
    main()
