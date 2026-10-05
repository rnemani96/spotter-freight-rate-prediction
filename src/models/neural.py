import numpy as np
import joblib
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MLPModel(BaseModel):
    """Multi-Layer Perceptron regression model wrapper with internal scaling."""
    
    @property
    def name(self) -> str:
        return 'mlp'

    def __init__(self, params: dict):
        """Initialize MLP model."""
        self._params = params.copy()
        if "hidden_layer_sizes" in self._params and isinstance(self._params["hidden_layer_sizes"], list):
            self._params["hidden_layer_sizes"] = tuple(self._params["hidden_layer_sizes"])
        
        self._model = MLPRegressor(**self._params)
        self._scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> "MLPModel":
        """Fit scaler and MLP. Replaces NaN/Inf with 0 before scaling."""
        X_safe = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self._scaler.fit_transform(X_safe)
        self._model.fit(X_scaled, y_train)
        logger.info("Fitted MLP model.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Scale inputs (NaN-safe), predict, and clip to positive."""
        X_safe = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        preds = self._model.predict(self._scaler.transform(X_safe))
        return np.clip(preds, 1.0, None)

    def save(self, path: Path) -> None:
        """Save model and scaler using joblib."""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump((self._scaler, self._model), path / "mlp.joblib")

    @classmethod
    def load(cls, path: Path) -> "MLPModel":
        """Load scaler and model from disk."""
        instance = cls(params={})
        instance._scaler, instance._model = joblib.load(path / "mlp.joblib")
        return instance
