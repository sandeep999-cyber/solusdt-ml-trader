"""Ridge + squared lags + order-flow features — walk-forward + held-out.

Adds order-flow features derived from existing trade data:
- buy_ratio: taker_buy_volume / volume (order flow imbalance)
- volume_spike: volume / rolling_mean(volume, 60) (unusual activity)
- trade_intensity: trade_count (market participation)
- large_trade_ratio: volume / max(trade_count, 1) (avg trade size)
- buy_quote_ratio: taker_buy_quote_volume / quote_volume

Tests whether these add signal on top of squared lags.
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


def compute_order_flow_features(df):
    """Compute order-flow features from existing trade data."""
    vol = df["volume"].values.astype(np.float64)
    taker_buy = df["taker_buy_volume"].values.astype(np.float64)
    quote_vol = df["quote_volume"].values.astype(np.float64)
    taker_buy_quote = df["taker_buy_quote_volume"].values.astype(np.float64)
    trade_count = df["trade_count"].values.astype(np.float64)

    features = pd.DataFrame(index=df.index)

    # Buy ratio: order flow imbalance (0=all sells, 1=all buys)
    features["buy_ratio"] = np.where(vol > 0, taker_buy / vol, 0.5)

    # Buy quote ratio
    features["buy_quote_ratio"] = np.where(quote_vol > 0, taker_buy_quote / quote_vol, 0.5)

    # Volume spike: ratio to 60-bar rolling mean
    vol_series = pd.Series(vol)
    rolling_mean = vol_series.rolling(60, min_periods=1).mean()
    features["volume_spike"] = np.where(rolling_mean > 0, vol / rolling_mean, 1.0)

    # Trade intensity
    features["trade_intensity"] = trade_count

    # Large trade ratio: volume per trade
    features["large_trade_ratio"] = np.where(trade_count > 0, vol / trade_count, 0.0)

    return features


def build_features(nr, of_features, n_lags=N_LAGS):
    """Build squared-lag features + order-flow features.

    Returns X (n, n_features, 1) for GRU-like input or (n, n_features) for Ridge.
    For Ridge, we flatten: [sq_lag_1..12, of_features_at_t].
    """
    n = len(nr)
    of_vals = of_features.values.astype(np.float64)
    of_names = of_features.columns.tolist()

    rows_X, rows_Y, rows_idx = [], [], []
    for i in range(0, n - HORIZON, STRIDE):
        if i < n_lags:
            continue
        # Squared lags
        lags = [nr[i - j] ** 2 for j in range(1, n_lags + 1)]
        # Order-flow features at time t (current bar)
        of = of_vals[i].tolist()
        rows_X.append(lags + of)
        rows_Y.append(np.sqrt(np.mean(nr[i + 1: i + 1 + HORIZON] ** 2)))
        rows_idx.append(i)

    X = np.array(rows_X, dtype=np.float64)
    Y = np.array(rows_Y, dtype=np.float64)
    feature_names = [f"sq_lag_{j}" for j in range(1, n_lags + 1)] + of_names
    return X, Y, np.array(rows_idx), feature_names


def bootstrap_ci(y_true, y_pred, baseline_rmse, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    imps = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        m_rmse = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        imps[b] = (1 - m_rmse / baseline_rmse) * 100
    return float(np.percentile(imps, 2.5)), float(np.percentile(imps, 97.5))


def walk_forward_ridge(X, Y, n_folds=5, label=""):
    """Walk-forward Ridge."""
    n = len(Y)
    fold_size = n // n_folds
    alphas = np.logspace(-2, 4, 20)

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
        r2 = 1 - (model_rmse / baseline_rmse) ** 2
        ci = bootstrap_ci(Y_test, preds, baseline_rmse)

        fold_results.append({
            "fold": fold + 1, "n_train": train_end, "n_test": len(Y_test),
            "baseline_rmse": baseline_rmse, "model_rmse": model_rmse,
            "improvement": improvement, "r2": r2, "ci": ci,
            "alpha": model.alpha_, "time": elapsed,
            "n_features": X.shape[1],
        })

        sig = "+" if improvement > 0 else ""
        print(f"  Fold {fold+1}: train={train_end} test={len(Y_test)} ({elapsed:.1f}s) alpha={model.alpha_:.1f}")
        print(f"    RMSE={model_rmse:.6f} (base={baseline_rmse:.6f})")
        print(f"    Improvement: {sig}{improvement:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  R2={r2:.6f}")

    return fold_results


def main():
    print("=" * 70)
    print("RIDGE + SQUARED LAGS + ORDER-FLOW FEATURES")
    print("=" * 70)

    # Load data
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    nr = df["norm_return"].values.astype(np.float64)
    nr = np.nan_to_num(nr, nan=0.0)
    ts = df["timestamp"].values

    # Compute order-flow features
    print("\nComputing order-flow features...")
    of_features = compute_order_flow_features(df)
    print(f"  Order-flow features: {of_features.columns.tolist()}")
    for c in of_features.columns:
        vals = of_features[c].values
        print(f"  {c:<25} min={np.nanmin(vals):.4f} max={np.nanmax(vals):.4f} mean={np.nanmean(vals):.4f}")

    # Build feature matrix
    X, Y, indices, feature_names = build_features(nr, of_features)
    print(f"\nTotal windows: {len(Y)}, features: {X.shape[1]} ({len(feature_names)} names)")

    # --- Test 1: Squared lags + order-flow (17 features) ---
    print(f"\n{'='*60}")
    print("TEST 1: Squared lags (12) + Order-flow (5) = 17 features")
    print(f"{'='*60}")
    fold_results_of = walk_forward_ridge(X, Y)

    # --- Test 2: Order-flow only (5 features, no squared lags) ---
    print(f"\n{'='*60}")
    print("TEST 2: Order-flow only (5 features)")
    print(f"{'='*60}")
    X_of_only = X[:, N_LAGS:]  # just the OF features
    fold_results_of_only = walk_forward_ridge(X_of_only, Y)

    # --- Test 3: Squared lags only (12 features, for comparison) ---
    print(f"\n{'='*60}")
    print("TEST 3: Squared lags only (12 features) — baseline")
    print(f"{'='*60}")
    X_sq_only = X[:, :N_LAGS]
    fold_results_sq = walk_forward_ridge(X_sq_only, Y)

    # Held-out for all three
    held_out_start = 4 * (len(Y) // 5)
    print(f"\n{'='*60}")
    print(f"HELD-OUT (Aug-Dec 2024): train={held_out_start} test={len(Y)-held_out_start}")
    print(f"{'='*60}")

    scaler = StandardScaler()
    held_results = {}
    for name, X_full in [("Sq+OF", X), ("OF only", X_of_only), ("Sq only", X_sq_only)]:
        X_tr = scaler.fit_transform(X_full[:held_out_start])
        X_held = scaler.transform(X_full[held_out_start:])
        Y_tr = Y[:held_out_start]
        Y_held = Y[held_out_start:]

        model = RidgeCV(alphas=np.logspace(-2, 4, 20), scoring="neg_mean_squared_error")
        model.fit(X_tr, Y_tr)
        preds = model.predict(X_held)

        base = np.std(Y_tr)
        rmse = np.sqrt(mean_squared_error(Y_held, preds))
        imp = (1 - rmse / base) * 100
        ci = bootstrap_ci(Y_held, preds, base, n_boot=10000)
        r2 = 1 - (rmse / base) ** 2

        held_results[name] = {"improvement": imp, "ci": ci, "r2": r2, "alpha": model.alpha_}
        sig = "+" if imp > 0 else ""
        print(f"  {name:<12}: alpha={model.alpha_:.1f} RMSE={rmse:.6f} base={base:.6f}")
        print(f"    Improvement: {sig}{imp:.2f}%  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  R2={r2:.6f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (walk-forward folds + held-out)")
    print(f"{'='*70}")
    print(f"  {'Config':<20} {'F1':>7} {'F2':>7} {'F3':>7} {'F4':>7} {'Held':>8} {'95% CI':>18}")
    print(f"  {'-'*72}")

    # Previous baselines
    print(f"  {'GARCH(1,1)':<20} {'+4.79':>7} {'+1.27':>7} {'+1.19':>7} {'+0.68':>7} {'(D024)':>8}")
    print(f"  {'Ridge+sq (prev)':<20} {'+8.71':>7} {'+7.15':>7} {'+4.12':>7} {'+6.75':>7} {'+6.78':>8} {'[+4.36,+9.08]':>18}")

    for name, fr in [("Sq+OF", fold_results_of), ("OF only", fold_results_of_only), ("Sq only", fold_results_sq)]:
        sigs = ["+" if r["improvement"] > 0 else "" for r in fr[:4]]
        vals = [f"{sigs[i]}{fr[i]['improvement']:>6.2f}" for i in range(min(4, len(fr)))]
        while len(vals) < 4:
            vals.append(f"{'':>7}")
        h = held_results[name]
        sig_h = "+" if h["improvement"] > 0 else ""
        print(f"  {name:<20} {vals[0]:>7} {vals[1]:>7} {vals[2]:>7} {vals[3]:>7} "
              f"{sig_h}{h['improvement']:>6.2f} [{h['ci'][0]:+.2f},{h['ci'][1]:+.2f}]")

    # Feature importance (from held-out model)
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE (held-out model, squared lags + OF)")
    print(f"{'='*70}")
    X_tr = scaler.fit_transform(X[:held_out_start])
    X_held = scaler.transform(X[held_out_start:])
    model = RidgeCV(alphas=np.logspace(-2, 4, 20), scoring="neg_mean_squared_error")
    model.fit(X_tr, Y[:held_out_start])
    coefs = model.coef_
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    for idx in sorted_idx:
        print(f"  {feature_names[idx]:<25} coef={coefs[idx]:+.6f}")


if __name__ == "__main__":
    main()
