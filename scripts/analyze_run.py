"""Analyze a training run — checkpoint metrics + full training trajectory.

Usage:
    python scripts/analyze_run.py                        # latest run
    python scripts/analyze_run.py --run phaseA_20260721_120158
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "model" / "runs"
POINTER_PATH = PROJECT_ROOT / "model" / "checkpoints" / "latest.json"


def _load_checkpoint(run_dir: Path) -> dict:
    best = run_dir / "checkpoints" / "best.pt"
    if not best.exists():
        return {}
    import torch
    ckpt = torch.load(best, map_location="cpu", weights_only=True)
    return {
        "epoch": ckpt.get("epoch"),
        "val_metrics": ckpt.get("val_metrics", {}),
        "window_length": ckpt.get("window_length"),
        "horizon": ckpt.get("horizon"),
        "feature_columns": ckpt.get("feature_columns"),
    }


def _load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_config(run_dir: Path) -> dict:
    path = run_dir / "config.yaml"
    if path.exists():
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    # Fallback: config embedded in checkpoint
    best = run_dir / "checkpoints" / "best.pt"
    if best.exists():
        import torch
        ckpt = torch.load(best, map_location="cpu", weights_only=True)
        raw = ckpt.get("config")
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
    return {}


def _find_run(run_name: str | None) -> Path:
    if run_name:
        p = RUNS_DIR / run_name
        if not p.exists():
            print(f"Run not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p
    # Latest from pointer
    if POINTER_PATH.exists():
        pointer = json.loads(POINTER_PATH.read_text())
        p = RUNS_DIR / pointer["run_name"]
        if p.exists():
            return p
    # Fallback: newest directory
    dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir() and "smoketest" not in d.name])
    if dirs:
        return dirs[-1]
    print("No runs found", file=sys.stderr)
    sys.exit(1)


def analyze(run_name: str | None = None) -> None:
    run_dir = _find_run(run_name)
    print(f"Run: {run_dir.name}")
    print(f"{'='*60}")

    config = _load_config(run_dir)
    ckpt = _load_checkpoint(run_dir)
    metrics = _load_metrics(run_dir)

    if config:
        print(f"Model:      {config.get('model_class', '?')}")
        print(f"Window/Horizon: {config.get('window_length', '?')}/{config.get('horizon', '?')}")
        print(f"Epochs:     {config.get('num_epochs', '?')}")

    print()

    # — Checkpoint summary —
    vm = ckpt.get("val_metrics", {})
    if vm:
        print("Best checkpoint (val):")
        for k, v in sorted(vm.items()):
            print(f"  {k:20s} = {v}")
        print()

    # — Training trajectory —
    if metrics:
        epochs = [m.get("epoch", 0) for m in metrics]
        train_losses = [m.get("train_loss", float("nan")) for m in metrics]
        val_losses = [m.get("loss", float("nan")) for m in metrics]
        mses = [m.get("mse", float("nan")) for m in metrics]
        vars_ = [m.get("var_mean", float("nan")) for m in metrics]
        baseline_deltas = [m.get("baseline_delta", float("nan")) for m in metrics]

        print("Trajectory:")
        print(f"  {'Epoch':>5} {'TrainLoss':>10} {'ValLoss':>10} {'MSE':>10} {'VarMean':>10} {'DeltaBL':>10}")
        for i in range(len(epochs)):
            print(f"  {epochs[i]:5d} {train_losses[i]:10.6f} {val_losses[i]:10.6f} {mses[i]:10.6f} {vars_[i]:10.6f} {baseline_deltas[i]:10.6f}")

        # Trend analysis
        if len(mses) >= 2:
            first_mse, last_mse = mses[0], mses[-1]
            mse_delta = last_mse - first_mse

            print()
            print("Trend analysis:")
            print(f"  MSE: {first_mse:.6f} -> {last_mse:.6f} ({'↑' if mse_delta > 0 else '↓'}{abs(mse_delta):.6f})")
            if first_mse > 0:
                print(f"  MSE change: {mse_delta/first_mse*100:+.2f}%")

            if vars_:
                first_var, last_var = vars_[0], vars_[-1]
                var_delta = last_var - first_var
                print(f"  Var: {first_var:.6f} -> {last_var:.6f} ({'↑' if var_delta > 0 else '↓'}{abs(var_delta):.6f})")

            if baseline_deltas and any(not math.isnan(b) for b in baseline_deltas):
                first_delta = baseline_deltas[0]
                last_delta = baseline_deltas[-1]
                print(f"  Baseline delta: {first_delta:.6f} -> {last_delta:.6f}")

        # Variance shortcut detection
        if vars_ and mses and len(vars_) >= 2:
            last_mse_val = mses[-1]
            last_var_val = vars_[-1]
            if not math.isnan(last_mse_val) and not math.isnan(last_var_val):
                print()
                print("Variance shortcut check:")
                print(f"  Final MSE / unconditional-var proxy: {last_mse_val:.6f}")
                print(f"  Final predicted variance: {last_var_val:.6f}")
                if last_var_val > 1.01:
                    print(f"  ⚠ Variance inflated to {last_var_val:.4f} — possible shortcut (model inflates uncertainty instead of improving mean)")
                if abs(last_mse_val - last_var_val) < 0.01:
                    print(f"  ⚠ MSE ≈ variance — mean contributes near nothing. High shortcut risk.")
                if last_var_val < 1.005 or abs(last_var_val - 1.0) < 0.01:
                    print(f"  ✓ Variance near 1.0 (fixed or naturally calibrated)")

    print()
    print(f"{'='*60}")
    print("To analyze a different run: python scripts/analyze_run.py --run <run_name>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a training run")
    parser.add_argument("--run", help="Run name (default: latest)")
    args = parser.parse_args()
    analyze(args.run)


if __name__ == "__main__":
    main()
