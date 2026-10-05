"""
src/models/linear.py
=====================
Linear regression model wrappers: Ridge, Lasso, ElasticNet.

All three apply StandardScaler internally so callers don't need to pre-scale.
NaN/Inf values are replaced with 0 before fitting/predicting to ensure
compatibility with sklearn's strict validation (feature engineering can
occasionally produce NaN from log of zero-weight edges, etc.).
"""

import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _safe(X: np.ndarray) -> np.ndarray:
    """Replace NaN / ±Inf with 0 so sklearn validators don't raise."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class RidgeModel(BaseModel):
    """
    Ridge regression with internal StandardScaler.

    Ridge is a strong baseline for regression tasks with many numeric features.
    It adds L2 regularization (alpha controls strength) and handles
    multi-collinear features gracefully.
    """

    @property
    def name(self) -> str:
        return "ridge"

    def __init__(self, params: dict) -> None:
        self._params = params.copy()
        self._model  = Ridge(**self._params)
        self._scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> "RidgeModel":
        """Fit scaler then Ridge. Val data is unused (no early stopping for Ridge)."""
        X_safe = _safe(X_train)
        self._scaler.fit(X_safe)
        self._model.fit(self._scaler.transform(X_safe), y_train)
        logger.info("Fitted Ridge model.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Scale, predict, clip to positive."""
        preds = self._model.predict(self._scaler.transform(_safe(X)))
        return np.clip(preds, 1.0, None)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump((self._scaler, self._model), path / "ridge.joblib")

    @classmethod
    def load(cls, path: Path) -> "RidgeModel":
        instance = cls(params={})
        instance._scaler, instance._model = joblib.load(path / "ridge.joblib")
        return instance


class LassoModel(BaseModel):
    """
    Lasso regression with internal StandardScaler.

    Lasso adds L1 regularization which drives unimportant feature weights to
    exactly zero — useful for implicit feature selection in high-dimensional
    feature spaces.
    """

    @property
    def name(self) -> str:
        return "lasso"

    def __init__(self, params: dict) -> None:
        self._params = params.copy()
        self._model  = Lasso(**self._params)
        self._scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> "LassoModel":
        """Fit scaler then Lasso. Val data unused."""
        X_safe = _safe(X_train)
        self._scaler.fit(X_safe)
        self._model.fit(self._scaler.transform(X_safe), y_train)
        logger.info("Fitted Lasso model.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self._model.predict(self._scaler.transform(_safe(X)))
        return np.clip(preds, 1.0, None)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump((self._scaler, self._model), path / "lasso.joblib")

    @classmethod
    def load(cls, path: Path) -> "LassoModel":
        instance = cls(params={})
        instance._scaler, instance._model = joblib.load(path / "lasso.joblib")
        return instance


class ElasticNetModel(BaseModel):
    """
    ElasticNet regression with internal StandardScaler.

    ElasticNet combines L1 and L2 penalties (controlled by l1_ratio).
    l1_ratio=1.0 → Lasso; l1_ratio=0.0 → Ridge.
    A mixed penalty often outperforms pure L1 or L2 on correlated features.
    """

    @property
    def name(self) -> str:
        return "elastic_net"

    def __init__(self, params: dict) -> None:
        self._params = params.copy()
        self._model  = ElasticNet(**self._params)
        self._scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> "ElasticNetModel":
        """Fit scaler then ElasticNet. Val data unused."""
        X_safe = _safe(X_train)
        self._scaler.fit(X_safe)
        self._model.fit(self._scaler.transform(X_safe), y_train)
        logger.info("Fitted ElasticNet model.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self._model.predict(self._scaler.transform(_safe(X)))
        return np.clip(preds, 1.0, None)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump((self._scaler, self._model), path / "elastic_net.joblib")

    @classmethod
    def load(cls, path: Path) -> "ElasticNetModel":
        instance = cls(params={})
        instance._scaler, instance._model = joblib.load(path / "elastic_net.joblib")
        return instance
