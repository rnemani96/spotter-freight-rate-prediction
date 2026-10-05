import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataCleaner:
    """Handles null imputation, type casting, anomaly clipping."""
    def __init__(self, cfg: dict):
        self.cfg = cfg
        
    def clean(self, df: pd.DataFrame, split: str = 'train') -> pd.DataFrame:
        """Clean dataframe according to rules. split can be 'train', 'val', 'dec'."""
        df = df.copy()
        
        logger.info(f"Starting clean for split={split}. Shape: {df.shape}")
        
        # Log nulls before
        null_counts = df.isnull().sum()
        logger.info(f"Null counts before cleaning:\n{null_counts[null_counts > 0]}")
        
        # Parse 'date' as datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Fill weight nulls -> 30000.0
        if 'weight' in df.columns:
            df['weight'] = df['weight'].fillna(30000.0)
            
        # market_index nulls -> same-day median -> then 1.08
        if 'market_index' in df.columns:
            # same-day median
            daily_median = df.groupby('date')['market_index'].transform('median')
            df['market_index'] = df['market_index'].fillna(daily_median)
            # then 1.08
            df['market_index'] = df['market_index'].fillna(1.08)
            
        # Clip posted_rate to [50, 30000] (train only)
        if split == 'train' and 'posted_rate' in df.columns:
            df['posted_rate'] = df['posted_rate'].clip(50, 30000)
            
        # Add split column
        df['split'] = split
        
        # Log nulls after
        null_counts_after = df.isnull().sum()
        logger.info(f"Null counts after cleaning:\n{null_counts_after[null_counts_after > 0]}")
        
        return df
