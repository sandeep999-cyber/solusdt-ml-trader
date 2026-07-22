"""Analyze a training run — checkpoint metrics + full training trajectory.

Usage:
    python scripts/analyze_run.py                        # latest run
    python scripts/analyze_run.py --run phaseA_20260721_120158
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model.reporting.summary import build_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "model" / "runs"
POINTER_PATH = PROJECT_ROOT / "model" / "checkpoints" / "latest.json"


def _find_run(run_name: str | None) -> Path:
    if run_name:
        p = RUNS_DIR / run_name
        if not p.exists():
            print(f"Run not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p
    if POINTER_PATH.exists():
        pointer = json.loads(POINTER_PATH.read_text())
        p = RUNS_DIR / pointer["run_name"]
        if p.exists():
            return p
    dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir() and "smoketest" not in d.name])
    if dirs:
        return dirs[-1]
    print("No runs found", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a training run")
    parser.add_argument("--run", help="Run name (default: latest)")
    parser.add_argument(
        "--write", action="store_true",
        help="Write summary.md to the run directory as well",
    )
    args = parser.parse_args()

    run_dir = _find_run(args.run)
    summary = build_summary(run_dir)
    print(summary)

    if args.write:
        path = run_dir / "summary.md"
        with open(path, "w") as f:
            f.write(summary)
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
