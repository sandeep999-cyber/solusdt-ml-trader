"""Run configuration — single source of truth for a training run.

Every run is fully described by one config file. No run-specific parameters
live anywhere else (not hardcoded in train.py, not passed as ad hoc CLI flags).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Default feature columns for Phase A (matches model/config/features.md)
PHASE_A_FEATURES = [
    "cvd",
    "cvd_quote",
    "vwap_20",
    "vwap_50",
    "anchored_vwap",
    "realized_vol",
    "log_return",
    "norm_return",
    "return_pct",
    "vol_profile_low_bucket",
]

PROCESSED_DIR = Path("data/processed/v1/SOLUSDT/1m")
RUNS_DIR = Path("model/runs")


@dataclass
class RunConfig:
    phase: str = "A"
    window_length: int = 60
    horizon: int = 12
    horizon_weighting: str = "uniform"  # "uniform" or "decay" — per-step loss weighting
    horizon_decay_rate: float = 0.9    # decay factor when weighting="decay" (closer to 0 = steeper)
    feature_columns: list[str] = field(default_factory=lambda: PHASE_A_FEATURES.copy())
    split_config_ref: str = "model/config/splits.py"
    active_split: str = "train"

    batch_size: int = 256
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    num_epochs: int = 50
    seed: int = 42
    training_stride: int = 1  # Step between training windows. stride=1 = full overlap.
                               # stride=60 = non-overlapping. Affects effective sample size.

    run_name: str = ""
    model_class: str = "model.body.simple_mlp.SimpleMLP"
    notes: str = ""

    # Paths (typically not overridden in YAML)
    processed_dir: str = str(PROCESSED_DIR)
    runs_dir: str = str(RUNS_DIR)

    def __post_init__(self):
        if not self.run_name:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.run_name = f"phase{self.phase}_{ts}"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        logger.info("Loaded run config from %s", path)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.run_name

    def checkpoint_dir(self) -> Path:
        return self.run_dir() / "checkpoints"

    def metrics_path(self) -> Path:
        return self.run_dir() / "metrics.jsonl"
