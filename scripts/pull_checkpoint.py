"""Sync Colab-trained checkpoints + metrics from Google Drive to local.

Syncs every run's best.pt, metrics.jsonl, and config.yaml from Drive to
model/runs/<run_name>/, then sets the latest pointer for the UI.

Usage:
    python scripts/pull_checkpoint.py                          # auto-detect Drive
    python scripts/pull_checkpoint.py D:/path/to/checkpoints   # manual Drive path
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "model" / "runs"
POINTER_PATH = PROJECT_ROOT / "model" / "checkpoints" / "latest.json"

DRIVE_CANDIDATES = [
    Path("G:/MyDrive/ModelProject/checkpoints"),
    Path.home() / "Google Drive/MyDrive/ModelProject/checkpoints",
    Path.home() / "Library/CloudStorage/GoogleDrive-*/MyDrive/ModelProject/checkpoints",
]


def _find_drive_folder(custom: str | None = None) -> Path | None:
    if custom:
        p = Path(custom)
        return p if p.exists() else None
    for p in DRIVE_CANDIDATES:
        # Handle glob pattern in candidate paths
        parts = list(p.parts)
        if "*" in str(p):
            from glob import glob
            matches = glob(str(p))
            if matches:
                return Path(matches[0])
        if p.exists():
            return p
    return None


def _sync_run(run_dir: Path, drive_dir: Path) -> None:
    """Sync one run from Drive to local runs directory."""
    run_name = run_dir.name
    local_run = RUNS_DIR / run_name
    local_ckpt = local_run / "checkpoints"
    local_ckpt.mkdir(parents=True, exist_ok=True)

    # best.pt
    src = run_dir / "best.pt"
    if src.exists():
        dst = local_ckpt / "best.pt"
        shutil.copy2(src, dst)
        print(f"  ✓ best.pt")

    # metrics.jsonl
    src_metrics = drive_dir / f"{run_name}_metrics.jsonl"
    if src_metrics.exists():
        dst_metrics = local_run / "metrics.jsonl"
        shutil.copy2(src_metrics, dst_metrics)
        print(f"  ✓ metrics.jsonl")

    # config.yaml
    src_config = drive_dir / f"{run_name}_config.yaml"
    if src_config.exists():
        dst_config = local_run / "config.yaml"
        shutil.copy2(src_config, dst_config)
        print(f"  ✓ config.yaml")


def sync_from_drive(drive_path: str | None = None) -> None:
    drive = _find_drive_folder(drive_path)
    if drive is None:
        print(
            "Drive checkpoints folder not found. Either:\n"
            "  1. Set up Google Drive for Desktop and sync MyDrive/ModelProject/checkpoints/\n"
            "  2. Pass the path manually:\n"
            "       python scripts/pull_checkpoint.py D:/path/to/checkpoints",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Drive folder: {drive}")

    # Discover all run checkpoints (files named <run_name>_best.pt)
    synced = []
    for f in sorted(drive.glob("*_best.pt")):
        run_name = f.stem.replace("_best", "")
        print(f"Syncing {run_name}...")
        _sync_run(Path(run_name), drive)
        synced.append(run_name)

    if not synced:
        # Fallback: single best.pt from the Colab notebook
        pointer_path = drive / "latest.json"
        if pointer_path.exists():
            pointer = json.loads(pointer_path.read_text())
            run_name = pointer.get("run_name", "colab_run")
            print(f"Syncing {run_name} (from latest.json)...")
            src = drive / "best.pt"
            if src.exists():
                local_run = RUNS_DIR / run_name
                local_ckpt = local_run / "checkpoints"
                local_ckpt.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, local_ckpt / "best.pt")
                print(f"  ✓ best.pt")
                synced.append(run_name)

    if not synced:
        print(f"No checkpoints found in {drive}", file=sys.stderr)
        sys.exit(1)

    # Update pointer to latest run
    latest = synced[-1]
    _update_pointer(latest)
    print(f"\nLatest run: {latest}")
    print(f"Total runs synced: {len(synced)}")

    # Print summary
    for name in synced:
        metrics_path = RUNS_DIR / name / "metrics.jsonl"
        if metrics_path.exists():
            with open(metrics_path) as f:
                lines = f.readlines()
            last = json.loads(lines[-1]) if lines else {}
            print(f"  {name}: mse={last.get('mse', '?')}  loss={last.get('loss', '?')}  epochs={last.get('epoch', '?')}")


def _update_pointer(run_name: str) -> None:
    pointer = {
        "run_name": run_name,
        "checkpoint_path": str(RUNS_DIR / run_name / "checkpoints" / "best.pt"),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    POINTER_PATH.write_text(json.dumps(pointer, indent=2))
    print(f"\nPointer -> {run_name}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    sync_from_drive(path)


if __name__ == "__main__":
    main()
