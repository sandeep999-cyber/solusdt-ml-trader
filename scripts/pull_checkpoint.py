"""Sync latest Colab-trained checkpoint from Google Drive to local UI.

Usage:
    python scripts/pull_checkpoint.py                          # auto-detect Drive path
    python scripts/pull_checkpoint.py D:/path/to/best.pt       # manual file
    python scripts/pull_checkpoint.py --run-name phaseA_xxx    # from model/runs/ (already local)

Auto-detection searches common Drive for Desktop locations.
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

# Common Google Drive for Desktop sync paths
DRIVE_CANDIDATES = [
    Path("G:/MyDrive/ModelProject/checkpoints"),
    Path("G:/.shortcut-targets-by-id/1aK8bWx9ZcQrXyZ/ModelProject/checkpoints"),
    Path.home() / "Google Drive/MyDrive/ModelProject/checkpoints",
    Path.home() / "Library/CloudStorage/GoogleDrive-sandeepchanda119@gmail.com/MyDrive/ModelProject/checkpoints",
]


def _find_drive_folder() -> Path | None:
    for p in DRIVE_CANDIDATES:
        if p.exists():
            return p
    return None


def _copy_checkpoint(src: Path, run_name: str) -> None:
    run_dir = RUNS_DIR / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dest = ckpt_dir / "best.pt"
    shutil.copy2(src, dest)
    print(f"Copied {src} -> {dest}")


def _update_pointer(run_name: str) -> None:
    pointer = {
        "run_name": run_name,
        "checkpoint_path": str(RUNS_DIR / run_name / "checkpoints" / "best.pt"),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    POINTER_PATH.write_text(json.dumps(pointer, indent=2))
    print(f"Updated {POINTER_PATH}")
    print(f"  -> UI will hot-reload within 5s on next request")


def from_manual(path_arg: str) -> None:
    src = Path(path_arg)
    if not src.exists():
        print(f"File not found: {src}", file=sys.stderr)
        sys.exit(1)
    run_name = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _copy_checkpoint(src, run_name)
    _update_pointer(run_name)
    print("Done. Restart the UI server or wait 5s for hot-reload.")


def from_drive() -> None:
    drive = _find_drive_folder()
    if drive is None:
        print(
            "Google Drive sync folder not found at any expected path.\n"
            "Either:\n"
            "  1. Install Google Drive for Desktop and sync ModelProject/checkpoints/\n"
            "  2. Pass the checkpoint path manually:\n"
            "     python scripts/pull_checkpoint.py D:/path/to/best.pt",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read the pointer file from Drive to get the latest run name
    pointer_path = drive / "latest.json"
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text())
        run_name = pointer.get("run_name", "drive_sync")
    else:
        run_name = "drive_sync"

    best_path = drive / f"{run_name}_best.pt"
    if not best_path.exists():
        best_path = drive / "best.pt"
    if not best_path.exists():
        print(f"No checkpoint found in {drive}", file=sys.stderr)
        sys.exit(1)

    _copy_checkpoint(best_path, run_name)
    _update_pointer(run_name)
    print("Done. Restart the UI server or wait 5s for hot-reload.")


def from_local_run(run_name: str) -> None:
    ckpt = RUNS_DIR / run_name / "checkpoints" / "best.pt"
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)
    _update_pointer(run_name)
    print("Done. Restart the UI server or wait 5s for hot-reload.")


def main() -> None:
    if len(sys.argv) == 1:
        from_drive()
    elif sys.argv[1] == "--run-name":
        if len(sys.argv) < 3:
            print("Usage: python scripts/pull_checkpoint.py --run-name <name>", file=sys.stderr)
            sys.exit(1)
        from_local_run(sys.argv[2])
    else:
        from_manual(sys.argv[1])


if __name__ == "__main__":
    main()
