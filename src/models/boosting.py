import numpy as np
import joblib
from pathlib import Path
from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LightGBMModel(BaseModel):
    """LightGBM regression model wrapper."""
    
    @property
    def name(self) -> str:
        return 'lightgbm'

    def __init__(self, params: dict):
        """Initialize the LightGBM model with parameters from config."""
        import lightgbm as lgb
        self._params = params.copy()
        self._early_stopping_rounds = self._params.pop("early_stopping_rounds", None)
        self._model = lgb.LGBMRegressor(**self._params)
        self._feature_names = None
        self._best_iteration = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None, feature_names: list[str] = None) -> "BaseModel":
        """Fit the model with optional validation data for early stopping."""
        import lightgbm as lgb
        
        self._feature_names = feature_names if feature_names is not None else [f'f{i}' for i in range(X_train.shape[1])]
        
        eval_set = None
        callbacks = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            if self._early_stopping_rounds:
                callbacks.append(lgb.early_stopping(stopping_rounds=self._early_stopping_rounds, verbose=False))
        
        self._model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks if callbacks else None)
        
        self._best_iteration = self._model.best_iteration_
        logger.info(f"Fitted LightGBM model. Best iteration: {self._best_iteration}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict and clip output to minimum 1.0."""
        preds = self._model.predict(X)
        return np.clip(preds, 1.0, None)

    def save(self, path: Path) -> None:
        """Save the model using joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / 'lgb.joblib')

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load the model from disk."""
        model = joblib.load(path / 'lgb.joblib')
        instance = cls(params={})
        instance._model = model
        return instance

    def get_feature_importance(self) -> dict | None:
        """Return feature importances if available."""
        if self._model is not None and self._feature_names is not None:
            importances = self._model.feature_importances_
            return {name: float(imp) for name, imp in zip(self._feature_names, importances)}
        return None


class XGBoostModel(BaseModel):
    """XGBoost regression model wrapper."""
    
    @property
    def name(self) -> str:
        return 'xgboost'

    def __init__(self, params: dict):
        """Initialize the XGBoost model."""
        import xgboost as xgb
        self._params = params.copy()
        self._early_stopping_rounds = self._params.pop("early_stopping_rounds", None)
        # XGBRegressor accepts early_stopping_rounds directly in fit in older versions, 
        # but in newer versions it's in init. Based on spec, "use it in constructor (not fit())".
        if self._early_stopping_rounds is not None:
            self._params['early_stopping_rounds'] = self._early_stopping_rounds
        self._model = xgb.XGBRegressor(**self._params)
        self._feature_names = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None, feature_names: list[str] = None) -> "BaseModel":
        """Fit the XGBoost model."""
        self._feature_names = feature_names if feature_names is not None else [f'f{i}' for i in range(X_train.shape[1])]
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            
        self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        logger.info(f"Fitted XGBoost model.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict and clip output."""
        preds = self._model.predict(X)
        return np.clip(preds, 1.0, None)

    def save(self, path: Path) -> None:
        """Save using joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / 'xgb.joblib')

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load using joblib."""
        model = joblib.load(path / 'xgb.joblib')
        instance = cls(params={})
        instance._model = model
        return instance

    def get_feature_importance(self) -> dict | None:
        """Return feature importance."""
        if self._model is not None and self._feature_names is not None:
            importances = self._model.feature_importances_
            return {name: float(imp) for name, imp in zip(self._feature_names, importances)}
        return None


class CatBoostModel(BaseModel):
    """CatBoost regression model wrapper."""
    
    @property
    def name(self) -> str:
        return 'catboost'

    def __init__(self, params: dict = None):
        """Initialize CatBoost model."""
        from catboost import CatBoostRegressor
        if params is None:
            params = {}
        self._params = params.copy()
        self._params["verbose"] = False
        self._early_stopping_rounds = self._params.pop("early_stopping_rounds", None)
        if self._early_stopping_rounds:
            self._params["early_stopping_rounds"] = self._early_stopping_rounds
            
        self._model = CatBoostRegressor(**self._params)
        self._feature_names = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None, feature_names: list[str] = None) -> "BaseModel":
        """Fit the model."""
        self._feature_names = feature_names if feature_names is not None else [f'f{i}' for i in range(X_train.shape[1])]
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            
        self._model.fit(X_train, y_train, eval_set=eval_set)
        logger.info(f"Fitted CatBoost model.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict and clip."""
        preds = self._model.predict(X)
        return np.clip(preds, 1.0, None)

    def save(self, path: Path) -> None:
        """Save using catboost native format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path / 'catboost.cbm'))

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load from catboost native format."""
        from catboost import CatBoostRegressor
        instance = cls(params={})
        instance._model = CatBoostRegressor()
        instance._model.load_model(str(path / 'catboost.cbm'))
        return instance
        
    def get_feature_importance(self) -> dict | None:
        """Return feature importance."""
        if self._model is not None and self._feature_names is not None:
            importances = self._model.get_feature_importance()
            return {name: float(imp) for name, imp in zip(self._feature_names, importances)}
        return None


class HistGradientBoostingModel(BaseModel):
    """HistGradientBoosting regression model wrapper."""
    
    @property
    def name(self) -> str:
        return 'hist_gradient_boosting'

    def __init__(self, params: dict):
        """Initialize HistGradientBoosting model."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        self._params = params.copy()
        self._model = HistGradientBoostingRegressor(**self._params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> "BaseModel":
        """Fit model (validation set is ignored for explicit eval set stopping here)."""
        self._model.fit(X_train, y_train)
        logger.info(f"Fitted HistGradientBoosting model.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict and clip."""
        preds = self._model.predict(X)
        return np.clip(preds, 1.0, None)

    def save(self, path: Path) -> None:
        """Save using joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / 'hgb.joblib')

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load using joblib."""
        model = joblib.load(path / 'hgb.joblib')
        instance = cls(params={})
        instance._model = model
        return instance
