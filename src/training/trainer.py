import time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.utils.logger import get_logger
from src.models.base import BaseModel
from src.training.evaluator import Evaluator, CVResult

class Trainer:
    """Trainer class to handle model cross-validation and final training."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.n_folds = cfg['training']['cv_folds']
        self.target = cfg['training']['target']
        self.holdout_frac = cfg['training']['holdout_fraction']
        self.random_state = cfg['project']['random_state']
        self.logger = get_logger(__name__)

    def run_cv(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> CVResult:
        """
        Run TimeSeriesSplit CV.
        For models that support early stopping, pass X_val/y_val.
        Track OOF predictions and log per-fold metrics.
        
        Args:
            model (BaseModel): Model instance to evaluate.
            X (np.ndarray): Feature matrix, assumed to be sorted by date.
            y (np.ndarray): Target array, assumed to be sorted by date.
            
        Returns:
            CVResult: Results of the cross-validation.
        """
        self.logger.info(f"Starting CV for model: {model.name} with {self.n_folds} folds")
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        
        oof_predictions = np.full_like(y, fill_value=np.nan, dtype=float)
        fold_scores = []
        start_time = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Fit model. Assume BaseModel handles passing eval set if it supports early stopping
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            
            preds = model.predict(X_val)
            oof_predictions[val_idx] = preds
            
            metrics = Evaluator.compute_metrics(y_val, preds)
            fold_scores.append(metrics)
            
            self.logger.debug(f"Fold {fold+1} metrics for {model.name}: RMSE={metrics['rmse']:.4f}")
            
        training_time = time.time() - start_time
        
        # Calculate mean metrics across folds
        mean_rmse = float(np.mean([m['rmse'] for m in fold_scores]))
        mean_mae = float(np.mean([m['mae'] for m in fold_scores]))
        mean_mape = float(np.mean([m['mape'] for m in fold_scores]))
        mean_r2 = float(np.mean([m['r2'] for m in fold_scores]))
        
        result = CVResult(
            model_name=model.name,
            fold_scores=fold_scores,
            mean_rmse=mean_rmse,
            mean_mae=mean_mae,
            mean_mape=mean_mape,
            mean_r2=mean_r2,
            oof_predictions=oof_predictions,
            training_time_sec=training_time
        )
        
        self.logger.info(f"CV finished for {model.name}. Mean RMSE: {mean_rmse:.4f}")
        return result

    def compare_all_models(self, X: np.ndarray, y: np.ndarray, models: list[BaseModel]) -> tuple[list[CVResult], pd.DataFrame]:
        """
        Run CV for each model and return comparison.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target array.
            models (list[BaseModel]): List of model instances.
            
        Returns:
            tuple: List of CVResult and a comparison DataFrame.
        """
        self.logger.info("Starting model comparison...")
        results = []
        
        for idx, model in enumerate(models):
            self.logger.info(f"Evaluating model {idx+1}/{len(models)}: {model.name}")
            cv_res = self.run_cv(model, X, y)
            results.append(cv_res)
            
        comparison_df = Evaluator.compare_models(results)
        return results, comparison_df

    def train_final(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> BaseModel:
        """
        Train final model on the entire dataset.
        Uses the last holdout_frac of data for early stopping evaluation.
        
        Args:
            model (BaseModel): Model to train.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target array.
            
        Returns:
            BaseModel: Trained model.
        """
        self.logger.info(f"Training final model: {model.name}")
        n_samples = len(X)
        split_idx = int(n_samples * (1 - self.holdout_frac))
        
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        self.logger.info(f"Final training completed for {model.name}.")
        return model
