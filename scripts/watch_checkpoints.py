"""Watch Drive for new checkpoints, auto-sync + analyze.

Runs in a loop. When Colab saves a new checkpoint to Drive, this detects
it within ~30s, syncs it locally, runs analysis, and prints results.

Usage:
    python scripts/watch_checkpoints.py
    python scripts/watch_checkpoints.py --interval 60   # check every 60s
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "model" / "runs"
POINTER_PATH = PROJECT_ROOT / "model" / "checkpoints" / "latest.json"

DRIVE_CANDIDATES = [
    Path("G:/My Drive/ModelProject/checkpoints"),
    Path("G:/MyDrive/ModelProject/checkpoints"),
]


def _find_drive() -> Path | None:
    for p in DRIVE_CANDIDATES:
        if p.exists():
            return p
    return None


def _known_runs() -> set[str]:
    return {d.name for d in RUNS_DIR.iterdir() if d.is_dir() and "smoketest" not in d.name}


def _sync_and_analyze(drive: Path, new_runs: set[str]) -> None:
    """Run pull_checkpoint and analyze_run for the given runs."""
    import subprocess

    print(f"[{time.strftime('%H:%M:%S')}] New runs detected: {new_runs}")
    print("Syncing from Drive...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "pull_checkpoint.py")],
        cwd=PROJECT_ROOT, capture_output=False,
    )

    for run_name in sorted(new_runs):
        print(f"\n--- Analysis: {run_name} ---")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "analyze_run.py"), "--run", run_name],
            cwd=PROJECT_ROOT, capture_output=False,
        )

    print(f"\n[{time.strftime('%H:%M:%S')}] Done. Watching for more runs...")


def watch(interval: int = 30) -> None:
    drive = _find_drive()
    if not drive:
        print("Drive checkpoints folder not found.")
        print("Make sure Google Drive for Desktop is syncing G:\\My Drive\\")
        sys.exit(1)

    print(f"Watching {drive} every {interval}s for new checkpoints...")
    print(f"Press Ctrl+C to stop.\n")

    seen = _known_runs()
    print(f"Already synced: {len(seen)} run(s) ({', '.join(seen) if seen else 'none'})")

    while True:
        try:
            time.sleep(interval)

            # Check for new run directories in Drive
            current = set()
            for f in drive.glob("*_best.pt"):
                run_name = f.stem.replace("_best", "")
                current.add(run_name)

            new = current - seen
            if new:
                _sync_and_analyze(drive, new)
                seen = current

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Watch Drive for new checkpoints")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    args = parser.parse_args()
    watch(args.interval)
