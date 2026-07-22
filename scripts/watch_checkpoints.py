"""Watch Drive for new checkpoints, auto-sync + analyze.

When Colab Cell 7 writes <run>_best.pt to Drive, this detects it,
pulls it locally, and runs analyze_run.py.

Usage:
    python scripts/watch_checkpoints.py
    python scripts/watch_checkpoints.py --interval 60
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "model" / "runs"

DRIVE_CANDIDATES = [
    Path("G:/My Drive/ModelProject/checkpoints"),
    Path("G:/MyDrive/ModelProject/checkpoints"),
]


def _find_drive() -> Path | None:
    for p in DRIVE_CANDIDATES:
        if p.exists():
            return p
    return None


def _wait_for_drive(interval: int) -> Path:
    """Poll until checkpoints folder appears (created by first Colab Cell 7)."""
    print("Waiting for Drive checkpoints folder...")
    print("  expected: G:/My Drive/ModelProject/checkpoints/")
    print("  (created automatically when Colab Cell 7 runs)\n")
    while True:
        drive = _find_drive()
        if drive is not None:
            print(f"Found: {drive}\n")
            return drive
        time.sleep(interval)


def _drive_runs(drive: Path) -> set[str]:
    runs = set()
    for f in drive.glob("*_best.pt"):
        # "phaseA_xxx_best.pt" → "phaseA_xxx"
        runs.add(f.name[: -len("_best.pt")])
    return runs


def _local_runs() -> set[str]:
    if not RUNS_DIR.exists():
        return set()
    return {
        d.name
        for d in RUNS_DIR.iterdir()
        if d.is_dir() and "smoketest" not in d.name and d.name != "__pycache__"
    }


def _sync_and_analyze(new_runs: set[str]) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] New runs: {sorted(new_runs)}")
    print("Syncing from Drive...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "pull_checkpoint.py")],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("pull_checkpoint.py failed — will retry next cycle")
        return

    for run_name in sorted(new_runs):
        print(f"\n--- Analysis: {run_name} ---")
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "analyze_run.py"),
                "--run",
                run_name,
            ],
            cwd=PROJECT_ROOT,
        )

    print(f"\n[{time.strftime('%H:%M:%S')}] Done. Watching for more runs...")


def watch(interval: int = 30) -> None:
    drive = _find_drive()
    if drive is None:
        drive = _wait_for_drive(interval)

    print(f"Watching {drive} every {interval}s")
    print("Press Ctrl+C to stop.\n")

    seen = _local_runs() | _drive_runs(drive)
    # If Drive already has runs we don't have locally, sync them now
    missing = _drive_runs(drive) - _local_runs()
    if missing:
        print(f"Found {len(missing)} run(s) on Drive not yet local — syncing now")
        _sync_and_analyze(missing)
        seen = _local_runs() | _drive_runs(drive)
    else:
        print(f"Already known: {len(seen)} run(s)")

    while True:
        try:
            time.sleep(interval)
            current = _drive_runs(drive)
            new = current - seen
            if new:
                _sync_and_analyze(new)
                seen = current | _local_runs()
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch Drive for new checkpoints")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval seconds")
    args = parser.parse_args()
    watch(args.interval)
