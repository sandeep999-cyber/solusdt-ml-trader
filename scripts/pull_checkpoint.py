"""Sync Colab-trained checkpoints + metrics from Google Drive to local.

Expects Drive layout (written by Colab Cell 7):
    G:/My Drive/ModelProject/checkpoints/
        <run_name>_best.pt
        <run_name>_metrics.jsonl
        <run_name>_config.yaml
        best.pt
        latest.json

Syncs into model/runs/<run_name>/ and updates model/checkpoints/latest.json.

Usage:
    python scripts/pull_checkpoint.py
    python scripts/pull_checkpoint.py "G:/My Drive/ModelProject/checkpoints"
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
    Path("G:/My Drive/ModelProject/checkpoints"),
    Path("G:/MyDrive/ModelProject/checkpoints"),
    Path.home() / "Google Drive/My Drive/ModelProject/checkpoints",
    Path.home() / "Google Drive/MyDrive/ModelProject/checkpoints",
]


def _find_drive_folder(custom: str | None = None) -> Path | None:
    if custom:
        p = Path(custom)
        return p if p.exists() else None
    for p in DRIVE_CANDIDATES:
        if p.exists():
            return p
    return None


def _sync_run(run_name: str, drive_dir: Path) -> bool:
    """Copy one run's artifacts from Drive into model/runs/<run_name>/."""
    local_run = RUNS_DIR / run_name
    local_ckpt = local_run / "checkpoints"
    local_ckpt.mkdir(parents=True, exist_ok=True)
    copied = False

    src_best = drive_dir / f"{run_name}_best.pt"
    if src_best.exists():
        shutil.copy2(src_best, local_ckpt / "best.pt")
        print(f"  ✓ best.pt")
        copied = True

    src_metrics = drive_dir / f"{run_name}_metrics.jsonl"
    if src_metrics.exists():
        shutil.copy2(src_metrics, local_run / "metrics.jsonl")
        print(f"  ✓ metrics.jsonl")
        copied = True

    src_config = drive_dir / f"{run_name}_config.yaml"
    if src_config.exists():
        shutil.copy2(src_config, local_run / "config.yaml")
        print(f"  ✓ config.yaml")
        copied = True

    if not copied:
        print(f"  (no files found for {run_name})")
    return copied


def sync_from_drive(drive_path: str | None = None) -> None:
    drive = _find_drive_folder(drive_path)
    if drive is None:
        print(
            "Drive checkpoints folder not found. Either:\n"
            "  1. Wait for Colab Cell 7 to create G:/My Drive/ModelProject/checkpoints/\n"
            "  2. Pass the path manually:\n"
            '       python scripts/pull_checkpoint.py "G:/My Drive/ModelProject/checkpoints"',
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Drive folder: {drive}")

    # Discover runs from <run_name>_best.pt files
    synced: list[str] = []
    for f in sorted(drive.glob("*_best.pt")):
        # stem = "phaseA_xxx_best" → run_name = "phaseA_xxx"
        run_name = f.name[: -len("_best.pt")]
        print(f"Syncing {run_name}...")
        if _sync_run(run_name, drive):
            synced.append(run_name)

    # Fallback: plain best.pt + latest.json (older Colab runs)
    if not synced:
        pointer_path = drive / "latest.json"
        src = drive / "best.pt"
        if pointer_path.exists() and src.exists():
            pointer = json.loads(pointer_path.read_text())
            run_name = pointer.get("run_name", "colab_run")
            print(f"Syncing {run_name} (from latest.json)...")
            local_run = RUNS_DIR / run_name
            local_ckpt = local_run / "checkpoints"
            local_ckpt.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local_ckpt / "best.pt")
            print("  ✓ best.pt")
            # Also try metrics/config under that name
            _sync_run(run_name, drive)
            synced.append(run_name)

    if not synced:
        print(f"No checkpoints found in {drive}", file=sys.stderr)
        sys.exit(1)

    # Prefer Drive's latest.json if present; else last sorted name
    pointer_path = drive / "latest.json"
    if pointer_path.exists():
        try:
            latest = json.loads(pointer_path.read_text()).get("run_name", synced[-1])
            if latest not in synced:
                latest = synced[-1]
        except json.JSONDecodeError:
            latest = synced[-1]
    else:
        latest = synced[-1]

    _update_pointer(latest)
    print(f"\nLatest run: {latest}")
    print(f"Total runs synced: {len(synced)}")

    for name in synced:
        metrics_path = RUNS_DIR / name / "metrics.jsonl"
        if metrics_path.exists():
            with open(metrics_path) as f:
                lines = [ln for ln in f if ln.strip()]
            last = json.loads(lines[-1]) if lines else {}
            print(
                f"  {name}: mse={last.get('mse', '?')}  "
                f"loss={last.get('loss', '?')}  epochs={last.get('epoch', '?')}"
            )


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
