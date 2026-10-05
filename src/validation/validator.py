import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

EXPECTED_ROWS = 12_000
EXPECTED_IDS = {f'TE-{i:06d}' for i in range(1, EXPECTED_ROWS + 1)}
DECEMBER_DATES = pd.date_range('2025-12-01', '2025-12-31', freq='D')

class OutputValidator:
    """Validates the output files against official formats and executes the scorer."""
    
    def __init__(self, cfg: dict, root: Path):
        self.cfg = cfg
        self.root = root
        self.output_dir = root / cfg['paths']['outputs']
        self.scorer_script = root / cfg['paths']['scorer_script']
        self.logger = get_logger(__name__)

    def validate_predictions_format(self, df: pd.DataFrame) -> None:
        """
        In-process checks on validation predictions.
        
        Args:
            df (pd.DataFrame): DataFrame to check.
        """
        if len(df) != EXPECTED_ROWS:
            raise ValueError(f"Expected {EXPECTED_ROWS} rows in validation preds, got {len(df)}")
            
        if set(df.columns) != {'load_id', 'predicted_rate'}:
            raise ValueError(f"Invalid columns. Expected {{'load_id', 'predicted_rate'}}, got {set(df.columns)}")
            
        if set(df['load_id']) != EXPECTED_IDS:
            raise ValueError("Load IDs do not match expected TEST load IDs.")
            
        if df['predicted_rate'].isna().any():
            raise ValueError("Found NaN values in validation predictions.")
            
        if (df['predicted_rate'] <= 0).any():
            raise ValueError("Found non-positive values in validation predictions.")

    def validate_december_format(self, df: pd.DataFrame) -> None:
        """
        In-process checks on december predictions.
        
        Args:
            df (pd.DataFrame): DataFrame to check.
        """
        if len(df) != 31:
            raise ValueError(f"Expected 31 rows for December predictions, got {len(df)}")
            
        expected_cols = {'pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate'}
        if set(df.columns) != expected_cols:
            raise ValueError(f"Invalid columns. Expected {expected_cols}, got {set(df.columns)}")
            
        dates_present = pd.to_datetime(df['date']).dt.normalize()
        if not set(dates_present) == set(DECEMBER_DATES):
            raise ValueError("Not all December 2025 dates are present.")
            
        if df['predicted_rate'].isna().any():
            raise ValueError("Found NaN values in December predictions.")
            
        if (df['predicted_rate'] <= 0).any():
            raise ValueError("Found non-positive values in December predictions.")

    def validate_all(self) -> bool:
        """
        Run official score.py via subprocess.
        
        Returns:
            bool: True on success.
            
        Raises:
            RuntimeError: If scorer fails.
        """
        val_file = self.output_dir / 'validation_predictions.csv'
        dec_file = self.output_dir / 'december-chart-inputs.csv'
        scorer_results = self.output_dir / 'scorer_results'
        scorer_results.mkdir(parents=True, exist_ok=True)

        self.logger.info("Running official scorer...")
        result = subprocess.run(
            ['python', str(self.scorer_script),
             '--predictions', str(val_file),
             '--december-predictions', str(dec_file),
             '--output-dir', str(scorer_results)],
            cwd=str(self.root),
            capture_output=True, text=True
        )
        if result.returncode == 0:
            self.logger.info('Scorer PASSED:\n' + result.stdout)
            return True
        else:
            self.logger.error('Scorer FAILED:\n' + result.stderr)
            raise RuntimeError(f'Official scorer failed: {result.stderr}')
