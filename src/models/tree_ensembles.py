import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RandomForestModel(BaseModel):
    """Random Forest regression model wrapper."""
    
    @property
    def name(self) -> str:
        return 'random_forest'

    def __init__(self, params: dict):
        """Initialize Random Forest model."""
        self._params = params.copy()
        self._model = RandomForestRegressor(**self._params)
        self._feature_names = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None, feature_names: list[str] = None) -> "BaseModel":
        """Fit model."""
        self._feature_names = feature_names if feature_names is not None else [f'f{i}' for i in range(X_train.shape[1])]
        self._model.fit(X_train, y_train)
        logger.info("Fitted RandomForest model.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict and clip."""
        preds = self._model.predict(X)
        return np.clip(preds, 1.0, None)

    def save(self, path: Path) -> None:
        """Save using joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / 'rf.joblib')

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load using joblib."""
        model = joblib.load(path / 'rf.joblib')
        instance = cls(params={})
        instance._model = model
        return instance
        
    def get_feature_importance(self) -> dict | None:
        """Return feature importance."""
        if self._model is not None and self._feature_names is not None:
            importances = self._model.feature_importances_
            return {name: float(imp) for name, imp in zip(self._feature_names, importances)}
        return None

class ExtraTreesModel(BaseModel):
    """Extra Trees regression model wrapper."""
    
    @property
    def name(self) -> str:
        return 'extra_trees'

    def __init__(self, params: dict):
        """Initialize Extra Trees model."""
        self._params = params.copy()
        self._model = ExtraTreesRegressor(**self._params)
        self._feature_names = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None, feature_names: list[str] = None) -> "BaseModel":
        """Fit model."""
        self._feature_names = feature_names if feature_names is not None else [f'f{i}' for i in range(X_train.shape[1])]
        self._model.fit(X_train, y_train)
        logger.info("Fitted ExtraTrees model.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict and clip."""
        preds = self._model.predict(X)
        return np.clip(preds, 1.0, None)

    def save(self, path: Path) -> None:
        """Save using joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / 'et.joblib')

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load using joblib."""
        model = joblib.load(path / 'et.joblib')
        instance = cls(params={})
        instance._model = model
        return instance
        
    def get_feature_importance(self) -> dict | None:
        """Return feature importance."""
        if self._model is not None and self._feature_names is not None:
            importances = self._model.feature_importances_
            return {name: float(imp) for name, imp in zip(self._feature_names, importances)}
        return None
