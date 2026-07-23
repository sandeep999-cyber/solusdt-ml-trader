"""Compare stride experiment results: three models, one table, bootstrap CIs.

Usage:
    python scripts/compare_stride_experiment.py \
        model/runs/stride_s1_control/checkpoints/best.pt \
        model/runs/stride_s15_intermediate/checkpoints/best.pt \
        model/runs/stride_s60_nonoverlap/checkpoints/best.pt

Produces a table with per-window MSE at stride=60 and 95% bootstrap CIs
for all three models, plus a trend判断 (gradual, cliff, or no effect).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES
from model.config.splits import get_split_mask
from model.inference.engine import _ffill_np, _build_model


def load_val_data():
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    val_mask = get_split_mask(df, "val")
    val_df = df[val_mask].reset_index(drop=True)
    feat = _ffill_np(val_df[PHASE_A_FEATURES].values.astype(np.float32))
    feat = np.nan_to_num(feat, nan=0.0)
    norm_return = val_df["norm_return"].values.astype(np.float64)
    return feat, norm_return


def per_window_mse(model, feat, norm_return, H=12, W=60, stride=60):
    windows_view = np.lib.stride_tricks.sliding_window_view(feat, window_shape=W, axis=0)
    bar_indices = np.arange(W - 1, len(feat))[::stride]

    model.eval()
    mses = []
    for idx in bar_indices:
        if idx + 1 + H > len(norm_return):
            continue
        start = idx - (W - 1)
        win = windows_view[start].T  # (W, F)
        win = np.ascontiguousarray(win)
        batch_t = torch.from_numpy(win).unsqueeze(0)
        with torch.no_grad():
            mean, _ = model(batch_t)
        m = mean[0].numpy()
        tgt = norm_return[idx + 1 : idx + 1 + H]
        valid = ~np.isnan(tgt)
        if not valid.any():
            continue
        mses.append(float(((tgt[valid] - m[valid]) ** 2).mean()))
    return np.array(mses)


def bootstrap_ci(mses, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    boots = np.array([mses[rng.choice(len(mses), len(mses), replace=True)].mean() for _ in range(n_boot)])
    return mses.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def judge_trend(results, baseline):
    """判断 three models show gradual, cliff, or no effect."""
    mses = [r["mean"] for r in results]
    cis = [(r["ci_lo"], r["ci_hi"]) for r in results]

    # Check if any model is below baseline (signal found)
    any_below = any(m < baseline for m in mses)

    # Check if all three are indistinguishable (overlapping CIs)
    all_overlap = all(
        cis[i][0] <= cis[j][1] and cis[j][0] <= cis[i][1]
        for i in range(len(cis)) for j in range(i + 1, len(cis))
    )

    if all_overlap:
        return "NO_EFFECT: All three models' CIs overlap. Stride alone doesn't resolve the harm."

    # Check pairwise overlaps first (cliff detection before gradual)
    s1_s15_overlap = cis[0][0] <= cis[1][1] and cis[1][0] <= cis[0][1]
    s15_s60_overlap = cis[1][0] <= cis[2][1] and cis[2][0] <= cis[1][1]

    # Cliff: adjacent pairs differ in overlap status
    if s1_s15_overlap and not s15_s60_overlap:
        return "CLIFF_AT_60: stride=1 and stride=15 overlap, but stride=60 is different. Harm drops off at full non-overlap."
    elif not s1_s15_overlap and s15_s60_overlap:
        return "CLIFF_AT_15: stride=15 and stride=60 overlap, but stride=1 is different. Harm exists only at full overlap."

    # Gradual: monotonic trend with middle distinguishable from both extremes
    monotonic_up = all(mses[i] <= mses[i + 1] for i in range(len(mses) - 1))
    monotonic_down = all(mses[i] >= mses[i + 1] for i in range(len(mses) - 1))

    if monotonic_up or monotonic_down:
        mid_from_lo = not (cis[0][0] <= mses[1] <= cis[0][1])
        mid_from_hi = not (cis[2][0] <= mses[1] <= cis[2][1])

        if mid_from_lo and mid_from_hi:
            return "GRADUAL: Middle model (stride=15) is distinguishable from both extremes. Harm scales with overlap."
        elif not s1_s15_overlap and not s15_s60_overlap:
            return "GRADUAL_WELL_SEPARATED: No adjacent CIs overlap, but trend is monotonic. Strong signal for gradual effect."
        else:
            return "PARTIAL_GRADUAL: Monotonic trend but middle model overlaps with one extreme. Weak signal for gradual effect."

    return "MIXED: Non-monotonic, non-cliff pattern. Results are ambiguous — need more stride points or investigation."


def main():
    if len(sys.argv) < 4:
        print("Usage: python compare_stride_experiment.py <s1_ckpt> <s15_ckpt> <s60_ckpt>")
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:4]]
    labels = ["stride=1", "stride=15", "stride=60"]

    print("Loading val data...")
    feat, norm_return = load_val_data()
    baseline = norm_return[~np.isnan(norm_return)].var()
    print(f"Baseline var: {baseline:.6f}")
    print(f"Val windows (stride=60): {len(norm_return) // 60}")
    print()

    device = torch.device("cpu")
    results = []
    for path, label in zip(paths, labels):
        print(f"Loading {label} from {path}...")
        model = _build_model(path, device)
        mses = per_window_mse(model, feat, norm_return)
        mean, ci_lo, ci_hi = bootstrap_ci(mses)
        results.append({"label": label, "path": str(path), "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(mses)})
        print(f"  {label}: MSE={mean:.6f}  95% CI [{ci_lo:.6f}, {ci_hi:.6f}]  n={len(mses)}")

    print()
    print("=" * 72)
    print(f"{'Model':<15} {'MSE':>10} {'95% CI':>28} {'vs baseline':>12}")
    print("-" * 72)
    for r in results:
        delta = (r["mean"] - baseline) / baseline * 100
        ci_str = f"[{r['ci_lo']:.6f}, {r['ci_hi']:.6f}]"
        print(f"{r['label']:<15} {r['mean']:.6f}  {ci_str:>28}  {delta:>+8.1f}%")
    print("-" * 72)
    print(f"{'baseline var':<15} {baseline:.6f}")
    print("=" * 72)

    # Check pairwise CI overlap
    print()
    print("Pairwise CI overlap:")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            r_i, r_j = results[i], results[j]
            overlap = r_i["ci_lo"] <= r_j["ci_hi"] and r_j["ci_lo"] <= r_i["ci_hi"]
            diff = r_j["mean"] - r_i["mean"]
            print(f"  {r_i['label']} vs {r_j['label']}: diff={diff:+.6f}, overlap={'YES' if overlap else 'NO'}")

    print()
    trend = judge_trend(results, baseline)
    print(f"TREND: {trend}")


if __name__ == "__main__":
    main()
