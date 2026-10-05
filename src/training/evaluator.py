from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class CVResult:
    """Dataclass to store cross-validation results."""
    model_name: str
    fold_scores: list  # list of dicts with rmse, mae, mape, r2
    mean_rmse: float
    mean_mae: float
    mean_mape: float
    mean_r2: float
    oof_predictions: np.ndarray
    training_time_sec: float

class Evaluator:
    """Evaluator class for computing and comparing model metrics."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute standard regression metrics.
        
        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
            
        Returns:
            dict: Dictionary containing rmse, mae, mape, and r2.
        """
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        
        # Avoid division by zero by clipping y_true
        y_true_clipped = np.clip(y_true, a_min=1e-8, a_max=None)
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true_clipped)) * 100.0)
        
        r2 = float(r2_score(y_true, y_pred))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }

    @staticmethod
    def compare_models(results: list[CVResult]) -> pd.DataFrame:
        """
        Compare models based on CV results and build a DataFrame.
        
        Args:
            results (list[CVResult]): List of cross-validation results.
            
        Returns:
            pd.DataFrame: Comparison dataframe sorted by mean_rmse ascending.
        """
        data = []
        for res in results:
            data.append({
                'model': res.model_name,
                'mean_rmse': res.mean_rmse,
                'mean_mae': res.mean_mae,
                'mean_mape_pct': res.mean_mape,
                'mean_r2': res.mean_r2,
                'training_time_sec': res.training_time_sec
            })
            
        df = pd.DataFrame(data)
        df.sort_values(by='mean_rmse', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Log formatted table — try markdown, fall back to plain string
        try:
            logger.info("\n" + df.to_markdown(index=False))
        except ImportError:
            logger.info("\n" + df.to_string(index=False))
        return df
