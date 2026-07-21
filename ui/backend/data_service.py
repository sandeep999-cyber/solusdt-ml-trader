import logging
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "v1"


def load_feature_table(symbol: str = "SOLUSDT", interval: str = "1m") -> pd.DataFrame:
    """Read the partitioned feature parquet dataset for a symbol/interval."""
    src = FEATURE_DIR / symbol / interval
    if not src.exists():
        raise FileNotFoundError(f"No feature data found at {src}")

    dataset = ds.dataset(str(src), format="parquet", partitioning=["year", "month"])
    table = dataset.to_table()
    df = table.to_pandas()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Loaded %d feature rows from %s", len(df), src)
    return df

