"""
src/data/loader.py
==================
DataLoader: responsible for reading raw CSV files from disk.

The December lane has fixed inputs (same pickup, delivery, distance, equipment,
weight every day); only the date changes. The loader injects all fixed lane
fields plus placeholder market/quote signals from config so that the
FeatureEngineer can apply the same transform pipeline to it.
"""

import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Loads raw CSV files for train, validation, December inference, and
    the validation-predictions template.

    Parameters
    ----------
    cfg  : full config dict loaded from config/config.yaml
    root : project root Path (d:/spotter)
    """

    def __init__(self, cfg: dict, root: Path) -> None:
        self.cfg     = cfg
        self.root    = root
        self.raw_dir = root / cfg["paths"]["data_raw"]

    # ── public loaders ────────────────────────────────────────────────────────

    def load_train(self) -> pd.DataFrame:
        """Load d:/spotter/data/raw/train-test.csv (48 000 rows, Jan–Oct 2025)."""
        path = self.raw_dir / self.cfg["data"]["train_file"]
        logger.info(f"Loading train data from {path}")
        df = pd.read_csv(path)
        logger.info(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
        return df

    def load_validation(self) -> pd.DataFrame:
        """Load d:/spotter/data/raw/validation.csv (12 000 rows, Nov–Dec 2025)."""
        path = self.raw_dir / self.cfg["data"]["validation_file"]
        logger.info(f"Loading validation data from {path}")
        df = pd.read_csv(path)
        logger.info(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
        return df

    def load_december(self) -> pd.DataFrame:
        """
        Load d:/spotter/data/raw/december-chart-inputs.csv (31 rows).

        The December sheet only has fixed lane columns and a date column;
        it has no market_index / quote_signal / lat-lon fields.  We inject
        all missing fields here so the cleaner and feature engineer can treat
        it identically to the train/val sets.

        Market signal defaults (market_index=1.0, quote_signal=2.0) are
        intentionally neutral — the feature engineer will override the
        rolling-market features with last-known training values anyway.
        """
        path = self.raw_dir / self.cfg["data"]["december_file"]
        logger.info(f"Loading december data from {path}")
        df = pd.read_csv(path)

        # ── inject fixed lane fields from config ──────────────────────────────
        lane = self.cfg["december_lane"]
        df["pickup"]        = lane["pickup"]           # Lexington
        df["delivery"]      = lane["delivery"]         # Fort Wayne
        df["pickup_lat"]    = lane["pickup_lat"]       # 37.9886
        df["pickup_lon"]    = lane["pickup_lon"]       # -84.4777
        df["delivery_lat"]  = lane["delivery_lat"]     # 41.0793
        df["delivery_lon"]  = lane["delivery_lon"]     # -85.1394
        df["distance"]      = lane["distance"]         # 360.0 miles
        df["equipment"]     = lane["equipment"]        # "Dry Van"
        df["weight"]        = lane["weight"]           # 32000.0

        # ── inject placeholder market signals (will be overridden by engineer) ─
        # Use neutral defaults; engineer replaces these with last-known rolling vals
        df["market_index"]  = 1.0
        df["quote_signal"]  = 2.0

        logger.info(f"  Loaded {len(df)} December rows with injected lane fields")
        return df

    def load_validation_template(self) -> pd.DataFrame:
        """
        Load validation-predictions-template.csv.
        Contains load_id and empty predicted_rate columns — defines the required
        row order and ID set for the final submission file.
        """
        path = self.raw_dir / self.cfg["data"]["validation_template"]
        logger.info(f"Loading validation template from {path}")
        df = pd.read_csv(path)
        logger.info(f"  Loaded template with {len(df):,} rows")
        return df
