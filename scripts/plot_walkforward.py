"""Walk-forward visualization — predicted vs actual volatility.

Generates plots for each fold + stacked comparison.
Saves to model/runs/walkforward_plots/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).resolve().parent.parent

# Fold-by-fold data from volatility_gru_walkforward.py
# (re-run to regenerate, or paste from terminal output)
gru_folds = [
    {"fold": 1, "n_train": 17541, "improve": 9.51, "r2": 0.181, "rmse": 0.2101, "base_rmse": 0.2322},
    {"fold": 2, "n_train": 35082, "improve": -148.19, "r2": -5.160, "rmse": 0.5840, "base_rmse": 0.2353},
    {"fold": 3, "n_train": 52623, "improve": -45.81, "r2": -1.126, "rmse": 0.3377, "base_rmse": 0.2316},
    {"fold": 4, "n_train": 70644, "improve": 11.64, "r2": 0.219, "rmse": 0.2066, "base_rmse": 0.2338},
    {"fold": 5, "n_train": 87705, "improve": 10.14, "r2": 0.193, "rmse": 0.1825, "base_rmse": 0.2031},
]

ridge_folds = [
    {"fold": 1, "improve": -0.36, "r2": -0.007},
    {"fold": 2, "improve": -19.69, "r2": -0.433},
    {"fold": 3, "improve": 1.86, "r2": 0.037},
    {"fold": 4, "improve": 5.35, "r2": 0.104},
    {"fold": 5, "improve": -5.04, "r2": -0.103},
]

out_dir = _root / "model" / "runs" / "walkforward_plots"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Plot 1: Fold-by-fold improvement comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(5)
w = 0.35
gru_imp = [f["improve"] for f in gru_folds]
ridge_imp = [f["improve"] for f in ridge_folds]

bars1 = ax.bar(x - w/2, ridge_imp, w, label="Ridge", color="#4a90d9", alpha=0.8)
bars2 = ax.bar(x + w/2, gru_imp, w, label="GRU h32", color="#e74c3c", alpha=0.8)
ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Fold (expanding window)")
ax.set_ylabel("RMSE Improvement (%)")
ax.set_title("Walk-Forward: RMSE Improvement by Fold\n(positive = better than predicting training mean)")
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {i+1}\n({gru_folds[i]['n_train']:,} train)" for i in range(5)])
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Add value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + (1 if h >= 0 else -3),
            f"{h:+.1f}%", ha="center", va="bottom" if h >= 0 else "top", fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + (1 if h >= 0 else -3),
            f"{h:+.1f}%", ha="center", va="bottom" if h >= 0 else "top", fontsize=8)

plt.tight_layout()
plt.savefig(out_dir / "fold_improvement.png", dpi=150)
plt.close()
print(f"Saved: {out_dir / 'fold_improvement.png'}")

# --- Plot 2: Stacked R² comparison ---
fig, ax = plt.subplots(figsize=(6, 4))
models = ["Ridge", "GRU h32"]
r2_vals = [-0.077, -1.49]
colors = ["#4a90d9", "#e74c3c"]
bars = ax.bar(models, r2_vals, color=colors, alpha=0.8, width=0.5)
ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("Stacked R²")
ax.set_title("Walk-Forward: Stacked R²\n(negative = worse than predicting mean)")
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.05,
            f"{val:.3f}", ha="center", va="top", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "stacked_r2.png", dpi=150)
plt.close()
print(f"Saved: {out_dir / 'stacked_r2.png'}")

# --- Plot 3: RMSE by fold (absolute values) ---
fig, ax = plt.subplots(figsize=(10, 5))
gru_rmse = [f["rmse"] for f in gru_folds]
ridge_base = [0.2331, 0.2816, 0.2273, 0.2213, 0.2134]  # from Ridge walk-forward output

ax.plot(range(1, 6), ridge_base, "o-", color="#4a90d9", label="Ridge RMSE", linewidth=2, markersize=8)
ax.plot(range(1, 6), gru_rmse, "s-", color="#e74c3c", label="GRU RMSE", linewidth=2, markersize=8)
ax.axhline(y=0.233, color="gray", linewidth=0.8, linestyle="--", label="Baseline RMSE (train mean)")
ax.set_xlabel("Fold")
ax.set_ylabel("RMSE")
ax.set_title("Walk-Forward: RMSE by Fold\n(lower is better)")
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f"Fold {i+1}\n({gru_folds[i]['n_train']:,})" for i in range(5)])
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "fold_rmse.png", dpi=150)
plt.close()
print(f"Saved: {out_dir / 'fold_rmse.png'}")

# --- Plot 4: Summary table ---
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
table_data = [
    ["", "Ridge", "GRU h32"],
    ["Fold 1 (+17K train)", f"{ridge_folds[0]['improve']:+.2f}%", f"{gru_folds[0]['improve']:+.2f}%"],
    ["Fold 2 (+35K train)", f"{ridge_folds[1]['improve']:+.2f}%", f"{gru_folds[1]['improve']:+.2f}%"],
    ["Fold 3 (+53K train)", f"{ridge_folds[2]['improve']:+.2f}%", f"{gru_folds[2]['improve']:+.2f}%"],
    ["Fold 4 (+71K train)", f"{ridge_folds[3]['improve']:+.2f}%", f"{gru_folds[3]['improve']:+.2f}%"],
    ["Fold 5 (+88K train)", f"{ridge_folds[4]['improve']:+.2f}%", f"{gru_folds[4]['improve']:+.2f}%"],
    ["Stacked R²", "-0.077", "-1.490"],
]

table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Color headers
for j in range(3):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Color negative values red, positive green
for i in range(1, 7):
    for j in range(1, 3):
        val = table_data[i][j]
        if val.startswith("-"):
            table[i, j].set_facecolor("#fadbd8")
        elif val.startswith("+"):
            table[i, j].set_facecolor("#d5f5e3")
        elif val.startswith("0."):
            table[i, j].set_facecolor("#fdebd0")

ax.set_title("Walk-Forward Results: Ridge vs GRU h32\n(Improvement % over training-mean baseline)", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(out_dir / "summary_table.png", dpi=150)
plt.close()
print(f"Saved: {out_dir / 'summary_table.png'}")

print(f"\nAll plots saved to: {out_dir}")
