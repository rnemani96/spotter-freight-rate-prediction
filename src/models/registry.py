from src.models.base import BaseModel
from src.models.boosting import LightGBMModel, XGBoostModel, CatBoostModel, HistGradientBoostingModel
from src.models.tree_ensembles import RandomForestModel, ExtraTreesModel
from src.models.linear import RidgeModel, LassoModel, ElasticNetModel
from src.models.neural import MLPModel

# Maps model name string -> (ModelClass, config_key)
MODEL_REGISTRY = {
    "lightgbm":              (LightGBMModel,            "lightgbm"),
    "catboost":              (CatBoostModel,            "catboost"),
    "xgboost":               (XGBoostModel,             "xgboost"),
    "hist_gradient_boosting":(HistGradientBoostingModel,"hist_gradient_boosting"),
    "random_forest":         (RandomForestModel,        "random_forest"),
    "extra_trees":           (ExtraTreesModel,          "extra_trees"),
    "ridge":                 (RidgeModel,               "ridge"),
    "lasso":                 (LassoModel,               "lasso"),
    "elastic_net":           (ElasticNetModel,          "elastic_net"),
    "mlp":                   (MLPModel,                 "mlp"),
}

def get_model(name: str, models_cfg: dict, random_state: int = 42) -> BaseModel:
    """
    Factory function. Reads params from models_cfg[config_key].
    Injects random_state where supported.
    Returns instantiated (but not yet trained) model.
    Raises ValueError for unknown model names.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
        
    ModelClass, config_key = MODEL_REGISTRY[name]
    
    # Get config for specific model, default to empty dict if missing
    params = models_cfg.get(config_key, {}).copy()
    
    # Inject random state where supported
    if name in ["lightgbm", "xgboost", "random_forest", "extra_trees", "mlp", "hist_gradient_boosting"]:
        params["random_state"] = random_state
    elif name == "catboost":
        params["random_seed"] = random_state
        
    return ModelClass(params=params)
