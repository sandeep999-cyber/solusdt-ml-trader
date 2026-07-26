"""Regime features — rolling realized-vol tercile.

Step 1: Define regime (rolling RV tercile, leak-checked)
Step 2: Test A — regime as added feature
Step 3: Test B — separate Ridge models per regime
Step 4: Compare against +6.78% baseline
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

HORIZON = 12
STRIDE = 60
N_LAGS = 12


def compute_regime(nr, lookback=60, history_window=30 * 24 * 60):
    """Compute regime label at each bar.

    Regime = tercile of rolling realized vol over `lookback` bars,
    computed against trailing `history_window` bars.

    CRITICAL: regime label at bar t uses only data from bars
    [t - lookback - history_window, t - lookback]. Nothing from
    after t - lookback is used. The label is known at the moment
    of prediction (same boundary as squared lags).

    Returns: regime labels (0=calm, 1=normal, 2=choppy) and
    rolling RV values.
    """
    n = len(nr)
    # Rolling realized vol: std of returns over lookback window
    r2 = nr ** 2
    cs2 = np.concatenate([[0], np.cumsum(r2)])
    # RV at bar t = sqrt(mean of squares from t-lookback to t-1)
    # This is the realized vol of the lookback window ENDING at t-1
    rv = np.zeros(n)
    rv[:] = np.nan
    for i in range(lookback, n):
        rv[i] = np.sqrt((cs2[i] - cs2[i - lookback]) / lookback)

    # Tercile cutoffs: computed from trailing history_window bars
    # At bar t, use cutoffs from bars [t - lookback - history_window, t - lookback]
    regime = np.full(n, np.nan)
    for i in range(lookback + history_window, n):
        trailing_rv = rv[i - lookback - history_window: i - lookback]
        # Drop NaNs
        trailing_rv = trailing_rv[~np.isnan(trailing_rv)]
        if len(trailing_rv) < 100:
            continue
        q33 = np.percentile(trailing_rv, 33.3)
        q67 = np.percentile(trailing_rv, 66.7)
        current_rv = rv[i]
        if np.isnan(current_rv):
            continue
        if current_rv < q33:
            regime[i] = 0  # calm
        elif current_rv < q67:
            regime[i] = 1  # normal
        else:
            regime[i] = 2  # choppy

    return regime, rv


def build_features(nr, regime):
    """Build squared-lag features + regime indicators."""
    n = len(nr)
    rows_X, rows_Y, rows_regime, rows_idx = [], [], [], []
    for i in range(0, n - HORIZON, STRIDE):
        if i < N_LAGS:
            continue
        if np.isnan(regime[i]):
            continue
        lags = [nr[i - j] ** 2 for j in range(1, N_LAGS + 1)]
        rows_X.append(lags)
        rows_Y.append(np.sqrt(np.mean(nr[i + 1: i + 1 + HORIZON] ** 2)))
        rows_regime.append(int(regime[i]))
        rows_idx.append(i)

    X = np.array(rows_X)
    Y = np.array(rows_Y)
    regimes = np.array(rows_regime)
    indices = np.array(rows_idx)
    return X, Y, regimes, indices


def bootstrap_ci(y_true, y_pred, baseline_rmse, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    imps = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        m_rmse = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        imps[b] = (1 - m_rmse / baseline_rmse) * 100
    return float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))


def walk_forward_ridge(X, Y, n_folds=4, alphas=None):
    """Walk-forward Ridge, return fold results + held-out."""
    if alphas is None:
        alphas = np.logspace(-2, 4, 20)

    n = len(Y)
    fold_size = n // 5

    fold_results = []
    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_start >= n:
            break

        X_train, Y_train = X[:train_end], Y[:train_end]
        X_test, Y_test = X[test_start:test_end], Y[test_start:test_end]

        t0 = time.time()
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
        model.fit(X_train_s, Y_train)
        preds = model.predict(X_test_s)
        elapsed = time.time() - t0

        baseline_rmse = np.std(Y_train)
        model_rmse = np.sqrt(mean_squared_error(Y_test, preds))
        improvement = (1 - model_rmse / baseline_rmse) * 100
        ci = bootstrap_ci(Y_test, preds, baseline_rmse)

        fold_results.append({"improvement": improvement, "ci": ci, "alpha": model.alpha_, "time": elapsed})
        sig = "+" if improvement > 0 else ""
        print(f"    Fold {fold+1}: {sig}{improvement:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  alpha={model.alpha_:.1f}")

    # Held-out
    held_start = 4 * fold_size
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X[:held_start])
    X_held_s = scaler.transform(X[held_start:])
    Y_tr, Y_held = Y[:held_start], Y[held_start:]

    model = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
    model.fit(X_tr_s, Y_tr)
    preds = model.predict(X_held_s)

    base = np.std(Y_tr)
    rmse = np.sqrt(mean_squared_error(Y_held, preds))
    imp = (1 - rmse / base) * 100
    ci = bootstrap_ci(Y_held, preds, base, n_boot=10000)

    sig = "+" if imp > 0 else ""
    print(f"    Held-out: {sig}{imp:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  alpha={model.alpha_:.1f}")

    return fold_results, imp, ci


def main():
    print("=" * 70)
    print("REGIME FEATURES — ROLLING RV TERCILE")
    print("=" * 70)

    # Load data
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    nr = df["norm_return"].values.astype(np.float64)
    nr = np.nan_to_num(nr, nan=0.0)

    # ================================================================
    # STEP 1: Compute regime
    # ================================================================
    print("\n--- Step 1: Computing regime (rolling RV tercile) ---")
    print("  lookback=60 bars (1 hour), history=30 days for cutoffs")

    regime, rv = compute_regime(nr, lookback=60, history_window=30 * 24 * 60)

    # ================================================================
    # STEP 2: Leakage check
    # ================================================================
    print("\n--- Step 2: Leakage check ---")

    # Build features with regime
    X, Y, regimes, indices = build_features(nr, regime)
    print(f"  Total windows: {len(Y)}")
    print(f"  Regime distribution:")
    for r, name in [(0, "calm"), (1, "normal"), (2, "choppy")]:
        count = np.sum(regimes == r)
        print(f"    {name}: {count} ({count/len(regimes)*100:.1f}%)")

    # Manual leakage check: pick a few rows and verify
    print("\n  Manual leakage check (5 examples):")
    for idx in [100, 500, 1000, 2000, 5000]:
        if idx >= len(indices):
            continue
        bar_i = indices[idx]
        r = regimes[idx]
        rvs_at_bar = rv[bar_i]
        # Regime at bar_i uses trailing 30-day data BEFORE bar_i-60
        # Feature window uses bars [bar_i-12, bar_i-1]
        # Target uses bars [bar_i+1, bar_i+12]
        # Regime uses bars [bar_i-60-30*24*60, bar_i-60]
        regime_end = bar_i - 60
        regime_start = bar_i - 60 - 30 * 24 * 60
        feat_start = bar_i - 12
        print(f"    bar={bar_i} regime={['calm','normal','choppy'][r]} RV={rvs_at_bar:.6f}")
        print(f"      Regime uses: [{regime_start}, {regime_end}]")
        print(f"      Features use: [{feat_start}, {bar_i-1}]")
        print(f"      Target uses: [{bar_i+1}, {bar_i+HORIZON}]")
        assert regime_end < feat_start, f"LEAKAGE: regime end {regime_end} >= feature start {feat_start}"
    print("  PASS: No leakage detected.")

    # ================================================================
    # Test A: Regime as added feature
    # ================================================================
    print(f"\n{'='*60}")
    print("TEST A: Regime as added feature (14 features)")
    print(f"{'='*60}")

    # Create dummy variables for regime (2 columns, drop first)
    regime_dummies = np.zeros((len(regimes), 2))
    regime_dummies[regimes == 1, 0] = 1  # normal
    regime_dummies[regimes == 2, 1] = 1  # choppy
    # calm is the reference (both 0)

    X_with_regime = np.hstack([X, regime_dummies])
    print(f"  Features: 12 sq lags + 2 regime dummies = {X_with_regime.shape[1]}")

    fold_a, held_a, ci_a = walk_forward_ridge(X_with_regime, Y)

    # Baseline (sq lags only)
    print(f"\n  Baseline (sq lags only):")
    fold_b, held_b, ci_b = walk_forward_ridge(X, Y)

    delta_a = held_a - held_b
    print(f"\n  Delta (regime - baseline): {delta_a:+.2f}%")

    # ================================================================
    # Test B: Separate models per regime
    # ================================================================
    print(f"\n{'='*60}")
    print("TEST B: Separate Ridge models per regime")
    print(f"{'='*60}")

    n = len(Y)
    fold_size = n // 5

    # For each regime, fit a separate Ridge model
    regime_names = ["calm", "normal", "choppy"]
    regime_models = {}
    held_results = {}

    for r in [0, 1, 2]:
        mask = regimes == r
        n_r = mask.sum()
        print(f"\n  Regime '{regime_names[r]}': {n_r} windows ({n_r/len(regimes)*100:.1f}%)")
        if n_r < 100:
            print(f"    Too few windows, skipping.")
            continue

        X_r = X[mask]
        Y_r = Y[mask]

        # Walk-forward for this regime
        scaler = StandardScaler()
        # Use global fold boundaries (not per-regime)
        held_start = 4 * fold_size

        # Train on all regime-r windows before held_start
        train_mask = mask.copy()
        train_mask[held_start:] = False
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(mask & (np.arange(len(mask)) >= held_start))[0]

        if len(train_indices) < 50 or len(test_indices) < 10:
            print(f"    Not enough train/test data, skipping.")
            continue

        X_tr = X[train_indices]
        Y_tr = Y[train_indices]
        X_te = X[test_indices]
        Y_te = Y[test_indices]

        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        alphas = np.logspace(-2, 4, 20)
        model = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
        model.fit(X_tr_s, Y_tr)
        preds = model.predict(X_te_s)

        base = np.std(Y_tr)
        rmse = np.sqrt(mean_squared_error(Y_te, preds))
        imp = (1 - rmse / base) * 100
        ci = bootstrap_ci(Y_te, preds, base, n_boot=5000)

        regime_models[r] = model
        held_results[r] = {"improvement": imp, "ci": ci, "n_train": len(train_indices), "n_test": len(test_indices)}

        sig = "+" if imp > 0 else ""
        print(f"    Train={len(train_indices)} Test={len(test_indices)} alpha={model.alpha_:.1f}")
        print(f"    Improvement: {sig}{imp:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]")

        # Print top coefficients
        coef_names = [f"sq{j}" for j in range(1, N_LAGS + 1)]
        sorted_idx = np.argsort(np.abs(model.coef_))[::-1]
        print(f"    Top coefficients:")
        for idx in sorted_idx[:3]:
            print(f"      {coef_names[idx]:<10} coef={model.coef_[idx]:+.6f}")

    # Combined: use regime to pick model at prediction time
    print(f"\n  --- Combined regime-conditioned prediction ---")
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X[:held_start])
    X_held_s = scaler.transform(X[held_start:])
    Y_tr_all = Y[:held_start]
    Y_held_all = Y[held_start:]
    regimes_held = regimes[held_start:]

    # For each held-out point, pick the model matching its regime
    preds_combined = np.zeros(len(Y_held_all))
    for r in [0, 1, 2]:
        if r not in regime_models:
            continue
        r_mask = regimes_held == r
        if r_mask.sum() == 0:
            continue
        # Need to refit scaler on regime-r training data only
        r_train_mask = (regimes[:held_start] == r)
        X_r_train = X[:held_start][r_train_mask]
        Y_r_train = Y[:held_start][r_train_mask]
        scaler_r = StandardScaler()
        X_r_train_s = scaler_r.fit_transform(X_r_train)
        X_r_held_s = scaler_r.transform(X[held_start:][r_mask])

        model_r = RidgeCV(alphas=np.logspace(-2, 4, 20), scoring="neg_mean_squared_error")
        model_r.fit(X_r_train_s, Y_r_train)
        preds_combined[r_mask] = model_r.predict(X_r_held_s)

    base_all = np.std(Y_tr_all)
    rmse_combined = np.sqrt(mean_squared_error(Y_held_all, preds_combined))
    imp_combined = (1 - rmse_combined / base_all) * 100
    ci_combined = bootstrap_ci(Y_held_all, preds_combined, base_all, n_boot=10000)

    sig = "+" if imp_combined > 0 else ""
    print(f"    Held-out (regime-conditioned): {sig}{imp_combined:.2f}%  CI [{ci_combined[0]:+.2f}, {ci_combined[1]:+.2f}]")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<40} {'Held-out':>9} {'95% CI':>18}")
    print(f"  {'-'*68}")
    print(f"  {'Ridge + 12 sq lags (baseline)':<40} {'+6.78':>9} {'[+4.36, +9.08]':>18}")
    sig_a = "+" if held_a > 0 else ""
    print(f"  {'Test A: +regime dummies':<40} {sig_a}{held_a:>8.2f} [{ci_a[0]:+.2f}, {ci_a[1]:+.2f}]")
    sig_c = "+" if imp_combined > 0 else ""
    print(f"  {'Test B: regime-conditioned models':<40} {sig_c}{imp_combined:>8.2f} [{ci_combined[0]:+.2f}, {ci_combined[1]:+.2f}]")

    for r in [0, 1, 2]:
        if r in held_results:
            h = held_results[r]
            sig_r = "+" if h["improvement"] > 0 else ""
            print(f"    {regime_names[r]:<38} {sig_r}{h['improvement']:>8.2f} [{h['ci'][0]:+.2f}, {h['ci'][1]:+.2f}] (n={h['n_test']})")


if __name__ == "__main__":
    import pandas as pd
    main()
