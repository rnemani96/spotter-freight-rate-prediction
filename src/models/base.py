from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pathlib import Path

class BaseModel(ABC):
    """Abstract base class for all freight-rate prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier string."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> "BaseModel":
        """Train the model. Returns self."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions array."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from disk."""

    def get_feature_importance(self) -> dict | None:
        """Return feature importances as {feature_name: importance} or None."""
        return None
