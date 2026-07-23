"""Build a markdown summary of a training run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional


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
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def _load_provenance(run_dir: Path) -> dict:
    path = run_dir / "provenance.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _load_checkpoint_metrics(run_dir: Path) -> dict:
    best = run_dir / "checkpoints" / "best.pt"
    if not best.exists():
        return {}
    try:
        import torch
        ckpt = torch.load(best, map_location="cpu", weights_only=True)
        return ckpt.get("val_metrics", {})
    except Exception:
        return {}


def build_summary(
    run_dir: Path,
    config: Optional[dict] = None,
    data_meta: Optional[dict] = None,
    git_commit: str = "",
) -> str:
    """Build a markdown summary string for a completed training run."""
    if config is None:
        config = _load_config(run_dir)
    # Handle RunConfig dataclass (from train.py) — convert to dict for .get() access
    if hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict
        config = asdict(config)
    if data_meta is None:
        prov = _load_provenance(run_dir)
        git_commit = prov.get("git_commit", "")
    else:
        git_commit = data_meta.get("git_commit", git_commit)

    metrics = _load_metrics(run_dir)
    ckpt_vm = _load_checkpoint_metrics(run_dir)

    lines: list[str] = []
    _h = lambda t: lines.append(f"## {t}")
    _p = lambda t: lines.append(t)

    _h(f"Run: {run_dir.name}")
    _p("")

    # Metadata
    _p("### Metadata")
    _p(f"- **git commit:** `{git_commit}`")
    _p(f"- **model_class:** {config.get('model_class', '?')}")
    _p(f"- **phase:** {config.get('phase', '?')}")
    _p(f"- **window / horizon:** {config.get('window_length', '?')} / {config.get('horizon', '?')}")
    _p(f"- **num_epochs:** {config.get('num_epochs', '?')}")
    _p(f"- **learning_rate:** {config.get('learning_rate', '?')}")
    _p(f"- **optimizer:** {config.get('optimizer', '?')}")
    _p(f"- **batch_size:** {config.get('batch_size', '?')}")
    _p(f"- **horizon_weighting:** {config.get('horizon_weighting', '?')}")
    _p(f"- **hidden_size:** {config.get('hidden_size', '?')}")
    _p(f"- **notes:** {config.get('notes', '')}")
    _p("")

    # Data
    if data_meta:
        _p("### Data")
        _p(f"- **version:** {data_meta.get('data_version', '?')}")
        _p(f"- **train rows:** {data_meta.get('train_rows', '?')}")
        _p(f"- **train range:** {data_meta.get('train_start', '?')} -> {data_meta.get('train_end', '?')}")
        _p(f"- **processed_dir:** `{data_meta.get('processed_dir', '?')}`")
        _p("")

    # Best checkpoint
    if ckpt_vm:
        _p("### Best checkpoint (validation)")
        for k, v in sorted(ckpt_vm.items()):
            _p(f"- **{k}:** {v}")
        _p("")

    # Trajectory
    if metrics:
        epochs = [m.get("epoch", 0) for m in metrics]
        mses = [m.get("mse", float("nan")) for m in metrics]
        vars_ = [m.get("var_mean", float("nan")) for m in metrics]
        baseline_deltas = [m.get("baseline_delta", float("nan")) for m in metrics]
        val_losses = [m.get("loss", float("nan")) for m in metrics]

        _p("### Training trajectory")
        _p("")
        _p("| Epoch | ValLoss | MSE | VarMean | DeltaBL |")
        _p("|-------|---------|-----|---------|---------|")
        for i in range(len(epochs)):
            _p(
                f"| {epochs[i]:5d} | {val_losses[i]:.6f} | {mses[i]:.6f} "
                f"| {vars_[i]:.6f} | {baseline_deltas[i]:+.6f} |"
            )
        _p("")

        # Analysis
        _p("### Analysis")
        if len(mses) >= 2:
            first_mse, last_mse = mses[0], mses[-1]
            mse_delta = last_mse - first_mse
            direction = "worsened" if mse_delta > 0 else "improved"
            _p(f"- **MSE:** {first_mse:.6f} -> {last_mse:.6f} ({direction} by {abs(mse_delta):.6f})")
            if first_mse > 0:
                _p(f"- **MSE change:** {mse_delta / first_mse * 100:+.2f}%")

        # Baseline delta
        if baseline_deltas and any(not math.isnan(b) for b in baseline_deltas):
            first_delta = baseline_deltas[0]
            last_delta = baseline_deltas[-1]
            _p(f"- **Baseline delta:** {first_delta:.6f} -> {last_delta:.6f}")

        # Variance shortcut
        if vars_ and mses:
            last_mse_val = mses[-1]
            last_var_val = vars_[-1]
            if not math.isnan(last_mse_val) and not math.isnan(last_var_val):
                flags = []
                if last_var_val > 1.01:
                    flags.append(
                        f"Variance inflated to {last_var_val:.4f} — "
                        "model inflates uncertainty instead of improving mean"
                    )
                if abs(last_mse_val - last_var_val) < 0.01:
                    flags.append(
                        "MSE ≈ variance — mean contributes near nothing. "
                        "High shortcut risk."
                    )
                if last_var_val < 1.005:
                    flags.append("Variance near 1.0 (fixed or naturally calibrated)")
                for flag in flags:
                    _p(f"- [WARNING] {flag}")

    _p("")
    _p("---")
    return "\n".join(lines)
