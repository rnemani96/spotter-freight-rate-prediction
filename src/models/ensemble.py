import numpy as np
import joblib
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class WeightedEnsembleModel(BaseModel):
    """Blends predictions from multiple BaseModel instances with optimized weights."""
    
    @property
    def name(self) -> str:
        return 'weighted_ensemble'

    def __init__(self, models: list[BaseModel], cfg: dict):
        """Initialize the ensemble with trained constituent models."""
        self._models = models
        self._cfg = cfg
        self._weights = None

    def optimize_weights(self, oof_preds: dict[str, np.ndarray], y_true: np.ndarray) -> np.ndarray:
        """
        Find optimal blend weights via Nelder-Mead minimization of RMSE on OOF predictions.
        Uses scipy.optimize.minimize.
        Weights are constrained to [0,1] and normalized to sum to 1.
        Returns weights array, also stores in self._weights.
        """
        model_names = [model.name for model in self._models]
        preds_matrix = np.column_stack([oof_preds[name] for name in model_names])
        
        def rmse_objective(weights):
            normalized_weights = weights / np.sum(weights)
            blended_preds = np.dot(preds_matrix, normalized_weights)
            return np.sqrt(mean_squared_error(y_true, blended_preds))
            
        initial_weights = np.ones(len(self._models)) / len(self._models)
        bounds = [(0.0, 1.0) for _ in range(len(self._models))]
        
        result = minimize(
            rmse_objective,
            initial_weights,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': self._cfg.get("ensemble", {}).get("n_trials", 300)}
        )
        
        self._weights = result.x / np.sum(result.x)
        logger.info(f"Optimized ensemble weights: {dict(zip(model_names, self._weights))}")
        return self._weights

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> "BaseModel":
        """
        Models are already trained; fit is a no-op.
        Initialises uniform weights if optimize_weights was never called
        (useful when running with --skip-cv / --skip-tuning).
        """
        if self._weights is None:
            n = len(self._models)
            self._weights = np.ones(n) / n
            logger.info(f"Ensemble: no OOF weights — using uniform 1/{n} per model")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted average of constituent model predictions.
        Auto-initialises uniform weights if optimize_weights was never called.
        """
        if self._weights is None:
            n = len(self._models)
            self._weights = np.ones(n) / n
            logger.info(f"Ensemble: auto-setting uniform weights (1/{n} each)")
        preds_matrix = np.column_stack([model.predict(X) for model in self._models])
        blended_preds = np.dot(preds_matrix, self._weights)
        return np.clip(blended_preds, 1.0, None)

    def get_constituent_predictions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Returns {model_name: predictions} for inspection."""
        return {model.name: model.predict(X) for model in self._models}

    def save(self, path: Path) -> None:
        """Save weights array and each constituent model to path/ensemble/"""
        ensemble_dir = path / 'ensemble'
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        # Save weights and config
        joblib.dump({"weights": self._weights, "cfg": self._cfg}, ensemble_dir / 'ensemble_meta.joblib')
        
        # Save each constituent model
        for model in self._models:
            model_dir = ensemble_dir / model.name
            model.save(model_dir)

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from disk."""
        raise NotImplementedError("Dynamic reloading of WeightedEnsembleModel with its constituent models is complex and currently not fully supported. Re-instantiate models and load them individually instead.")
