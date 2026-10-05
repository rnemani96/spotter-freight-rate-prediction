import sqlite3
from pathlib import Path
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from src.utils.logger import get_logger
from src.models import get_model
from src.training.evaluator import Evaluator

optuna.logging.set_verbosity(optuna.logging.WARNING)

class OptunaHyperparamTuner:
    """Tuner class utilizing Optuna for hyperparameter optimization."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.n_trials = cfg['tuning']['n_trials']
        self.timeout = cfg['tuning']['timeout']
        self.storage_dir = Path(cfg['paths']['tuning'])
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self.random_state = cfg['project']['random_state']

    def tune(self, model_name: str, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Run Optuna study to find best hyperparameters.
        
        Args:
            model_name (str): Name of the model to tune.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target array.
            
        Returns:
            dict: Best hyperparameters found.
        """
        self.logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        db_path = self.storage_dir / f"{model_name}_study.db"
        storage_url = f"sqlite:///{db_path}"
        
        study_name = f"{model_name}_tuning"
        
        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize",
            load_if_exists=True,
            sampler=TPESampler(seed=self.random_state)
        )
        
        def objective_wrapper(trial):
            return self._objective(trial, model_name, X, y)
            
        study.optimize(
            objective_wrapper, 
            n_trials=self.n_trials, 
            timeout=self.timeout
        )
        
        self.logger.info(f"Tuning finished for {model_name}. Best RMSE: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params

    def _get_search_space(self, trial: optuna.Trial, model_name: str) -> dict:
        """Define search space for each model type."""
        params = {}
        if model_name == 'lightgbm':
            params['num_leaves'] = trial.suggest_int('num_leaves', 31, 255)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
            params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.5, 1.0)
            params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.5, 1.0)
            params['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 50)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True)
            params['n_estimators'] = trial.suggest_int('n_estimators', 500, 3000)
            params['bagging_freq'] = 5
            params['early_stopping_rounds'] = 100
            
        elif model_name == 'catboost':
            params['depth'] = trial.suggest_int('depth', 4, 10)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
            params['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True)
            params['iterations'] = trial.suggest_int('iterations', 500, 3000)
            params['early_stopping_rounds'] = 100
            params['verbose'] = False
            
        elif model_name == 'xgboost':
            params['max_depth'] = trial.suggest_int('max_depth', 4, 10)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True)
            params['n_estimators'] = trial.suggest_int('n_estimators', 500, 3000)
            params['early_stopping_rounds'] = 100
            
        elif model_name == 'hist_gradient_boosting':
            params['max_depth'] = trial.suggest_int('max_depth', 4, 12)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 10, 50)
            params['l2_regularization'] = trial.suggest_float('l2_regularization', 1e-4, 1.0, log=True)
            params['max_iter'] = 1000
            
        elif model_name in ['random_forest', 'extra_trees']:
            params['n_estimators'] = trial.suggest_int('n_estimators', 100, 1000)
            depth_choice = trial.suggest_categorical('max_depth_choice', ['int', 'none'])
            if depth_choice == 'int':
                params['max_depth'] = trial.suggest_int('max_depth_int', 5, 30)
            else:
                params['max_depth'] = None
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 20)
            params['max_features'] = trial.suggest_float('max_features', 0.3, 1.0)
            
        elif model_name in ['ridge', 'lasso', 'elastic_net']:
            params['alpha'] = trial.suggest_float('alpha', 0.01, 1000.0, log=True)
            if model_name == 'elastic_net':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                
        elif model_name == 'mlp':
            params['hidden_layer_sizes'] = trial.suggest_categorical(
                'hidden_layer_sizes',
                [(64,), (128, 64), (256, 128, 64), (256, 128, 64, 32)]
            )
            params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
            params['max_iter'] = 500
            
        return params

    def _objective(self, trial: optuna.Trial, model_name: str, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective function calculating mean RMSE over 3-fold CV."""
        params = self._get_search_space(trial, model_name)
        
        # Instantiate model with trial parameters
        cfg_models = self.cfg.get('models', {}).copy()
        if model_name not in cfg_models:
            cfg_models[model_name] = {}
        cfg_models[model_name]['params'] = params
        
        model = get_model(model_name, cfg_models, self.random_state)
        
        # Run 3-fold CV for speed
        tscv = TimeSeriesSplit(n_splits=3)
        rmses = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            try:
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                preds = model.predict(X_val)
                metrics = Evaluator.compute_metrics(y_val, preds)
                rmses.append(metrics['rmse'])
            except Exception as e:
                # If model fails to train or predict (e.g., instability), penalize heavily
                self.logger.warning(f"Trial failed for {model_name} with params {params}: {e}")
                return float('inf')
                
        return float(np.mean(rmses))
