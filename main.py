"""
main.py — Freight Rate Prediction Pipeline Entry Point
========================================================
Orchestrates the full end-to-end pipeline:

  Raw Load Data → Data Cleaning → Feature Engineering
  → Model CV → Model Selection → Hyperparameter Tuning
  → Final Training → Rate Prediction → Validation → Outputs

Usage:
    python main.py [options]

    python main.py --config config/config.yaml
    python main.py --skip-tuning               # fast run, default params
    python main.py --skip-cv --skip-tuning     # predict only with defaults
    python main.py --models lightgbm catboost  # run specific models only
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from src.data.cleaner import DataCleaner
from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.models.ensemble import WeightedEnsembleModel
from src.models.registry import get_model
from src.prediction.predictor import Predictor
from src.training.trainer import Trainer
from src.tuning.tuner import OptunaHyperparamTuner
from src.utils.io_utils import load_config, save_json
from src.utils.logger import get_logger
from src.validation.validator import OutputValidator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Spotter Freight Rate Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config.yaml relative to project root",
    )
    parser.add_argument(
        "--skip-tuning", action="store_true",
        help="Skip Optuna hyperparameter tuning (use default params)",
    )
    parser.add_argument(
        "--skip-cv", action="store_true",
        help="Skip cross-validation (implies --skip-tuning too)",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Override enabled models list (e.g. --models lightgbm catboost)",
    )
    parser.add_argument(
        "--n-tuning-trials", type=int,
        help="Override config tuning.n_trials",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full freight rate prediction pipeline."""
    t_start = time.time()

    # ── 1. Parse args & load config ──────────────────────────────────────────
    args = parse_args()
    root = Path(__file__).resolve().parent      # project root = d:\spotter
    cfg = load_config(root / args.config)

    # CLI overrides
    if args.models:
        cfg["models"]["enabled"] = args.models
    if args.n_tuning_trials is not None:
        cfg["tuning"]["n_trials"] = args.n_tuning_trials
    if args.skip_cv:
        args.skip_tuning = True                 # CV needed for ensemble OOF

    # ── 2. Setup logger ───────────────────────────────────────────────────────
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("pipeline")

    # ── 3. Banner ─────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(f"  SPOTTER FREIGHT RATE PREDICTION PIPELINE")
    logger.info(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Config  : {args.config}")
    logger.info(f"  Tuning  : {'OFF' if args.skip_tuning else 'ON'}")
    logger.info("=" * 65)

    # ── 4. Load raw data ──────────────────────────────────────────────────────
    logger.info("STAGE 1 — Loading raw data")
    dl = DataLoader(cfg, root)
    train_raw  = dl.load_train()
    val_raw    = dl.load_validation()
    dec_raw    = dl.load_december()
    template   = dl.load_validation_template()
    logger.info(f"  Train : {train_raw.shape}  Val : {val_raw.shape}  Dec : {dec_raw.shape}")

    # ── 5. Clean data ─────────────────────────────────────────────────────────
    logger.info("STAGE 2 — Cleaning data")
    dc = DataCleaner(cfg)
    train_clean = dc.clean(train_raw, split="train")
    val_clean   = dc.clean(val_raw,   split="val")
    dec_clean   = dc.clean(dec_raw,   split="dec")

    # ── 6. Feature engineering ────────────────────────────────────────────────
    logger.info("STAGE 3 — Engineering features")
    fe = FeatureEngineer(cfg)
    train_feat = fe.fit_transform(train_clean)   # fit on train, then transform
    val_feat   = fe.transform(val_clean)
    dec_feat   = fe.transform(dec_clean)

    # Persist the fitted feature engineer for later inference
    artifacts_dir = root / cfg["paths"]["models"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fe.save(artifacts_dir / "feature_engineer.joblib")
    logger.info(f"  Features : {len(fe.get_feature_cols())} columns")

    # ── 7. Prepare sorted training matrix ─────────────────────────────────────
    logger.info("STAGE 4 — Preparing training matrices (sorted by date)")
    train_feat = train_feat.sort_values("date").reset_index(drop=True)
    feature_cols = fe.get_feature_cols()
    target       = cfg["training"]["target"]        # "posted_rate"

    X_train = train_feat[feature_cols].values
    y_train = train_feat[target].values

    # ── 8. Cross-validation model comparison ─────────────────────────────────
    random_state   = cfg["project"]["random_state"]
    enabled_models = cfg["models"]["enabled"]        # list of model name strings

    cv_results = []
    comp_df    = None

    if args.skip_cv:
        logger.info("STAGE 5 — Skipping CV (--skip-cv flag set)")
        top_model_names = enabled_models[:cfg["tuning"].get("top_n_to_tune", 3)]
    else:
        logger.info(f"STAGE 5 — Running {cfg['training']['cv_folds']}-fold CV on {len(enabled_models)} models")
        initial_models = [get_model(n, cfg["models"], random_state) for n in enabled_models]
        trainer = Trainer(cfg)
        cv_results, comp_df = trainer.compare_all_models(X_train, y_train, initial_models)

        # Save comparison report
        reports_dir = root / cfg["paths"]["reports"]
        reports_dir.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(reports_dir / "cv_comparison.csv", index=False)
        logger.info(f"  CV results saved to artifacts/reports/cv_comparison.csv")

        # Select top-N for tuning
        top_n = cfg["tuning"].get("top_n_to_tune", 3)
        top_model_names = comp_df.head(top_n)["model"].tolist()
        logger.info(f"  Top-{top_n} models: {top_model_names}")

    # ── 9. Hyperparameter tuning (Optuna) ─────────────────────────────────────
    tuned_cfg = {}   # model_name -> updated params dict

    if not args.skip_tuning:
        logger.info(f"STAGE 6 — Optuna tuning for: {top_model_names}")
        tuner       = OptunaHyperparamTuner(cfg)
        tuning_dir  = root / cfg["paths"]["tuning"]
        tuning_dir.mkdir(parents=True, exist_ok=True)

        for m_name in top_model_names:
            logger.info(f"  Tuning {m_name} ...")
            best_params = tuner.tune(m_name, X_train, y_train)
            tuned_cfg[m_name] = best_params
            save_json(best_params, tuning_dir / f"{m_name}_best_params.json")
            logger.info(f"  Best params for {m_name}: {best_params}")
    else:
        logger.info("STAGE 6 — Skipping tuning (--skip-tuning flag set)")

    # ── 10. Train final models on full training data ──────────────────────────
    logger.info("STAGE 7 — Training final models on full training data")
    final_models = []
    oof_dict     = {}   # model_name -> np.ndarray of OOF predictions
    trainer      = Trainer(cfg)

    for m_name in top_model_names:
        # Build model config: start from base config, overlay tuned params if any
        models_cfg = dict(cfg["models"])
        if m_name in tuned_cfg:
            # Merge tuned params into the model's config section
            models_cfg[m_name] = {**models_cfg.get(m_name, {}), **tuned_cfg[m_name]}

        model = get_model(m_name, models_cfg, random_state)

        # If we ran CV, re-run it with the (possibly tuned) model to get OOF preds
        if not args.skip_cv:
            logger.info(f"  Getting OOF predictions for {m_name} ...")
            cv_res = trainer.run_cv(model, X_train, y_train)
            oof_dict[m_name] = cv_res.oof_predictions

        # Train final model on all training data
        logger.info(f"  Training final {m_name} ...")
        final_model = trainer.train_final(model, X_train, y_train)

        # Save model artifact
        m_dir = (root / cfg["paths"]["models"]) / m_name
        m_dir.mkdir(parents=True, exist_ok=True)
        final_model.save(m_dir)

        final_models.append(final_model)

    # ── 11. Build weighted ensemble ───────────────────────────────────────────
    logger.info("STAGE 8 — Building weighted ensemble")
    ensemble = WeightedEnsembleModel(models=final_models, cfg=cfg)

    if oof_dict:
        # Use OOF predictions to find the optimal blend weights (Nelder-Mead)
        # Only use rows where ALL models have OOF predictions (non-NaN)
        valid_mask = np.ones(len(y_train), dtype=bool)
        for preds in oof_dict.values():
            valid_mask &= ~np.isnan(preds)

        first_idx = int(np.argmax(valid_mask))   # first row covered by all OOFs
        oof_slice = {n: preds[first_idx:] for n, preds in oof_dict.items()}
        y_slice   = y_train[first_idx:]

        optimal_weights = ensemble.optimize_weights(oof_slice, y_slice)
        logger.info(f"  Ensemble weights: { {m.name: round(float(w), 4) for m, w in zip(final_models, optimal_weights)} }")
    else:
        # No OOF — uniform weights
        logger.info("  No OOF available; using uniform ensemble weights")

    # ── 12. Generate predictions ──────────────────────────────────────────────
    logger.info("STAGE 9 — Generating predictions")
    predictor = Predictor(cfg, ensemble, fe, root)
    val_preds = predictor.predict_validation(val_feat)
    dec_preds = predictor.predict_december(dec_feat, dec_clean)
    logger.info(f"  Val preds: n={len(val_preds)}, mean=${val_preds['predicted_rate'].mean():.2f}")
    logger.info(f"  Dec preds: n={len(dec_preds)}, mean=${dec_preds['predicted_rate'].mean():.2f}")

    # ── 13. Validate output format (in-process, fast) ─────────────────────────
    logger.info("STAGE 10 — Validating output format")
    validator = OutputValidator(cfg, root)
    validator.validate_predictions_format(val_preds)
    validator.validate_december_format(dec_preds)
    logger.info("  Format validation passed")

    # ── 14. Write outputs ─────────────────────────────────────────────────────
    logger.info("STAGE 11 — Writing output files")
    predictor.write_outputs(val_preds, dec_preds, template)

    # ── 15. Run official scorer ────────────────────────────────────────────────
    logger.info("STAGE 12 — Running official scorer (score.py)")
    try:
        passed = validator.validate_all()
        if passed:
            logger.info("  SCORER PASSED")
    except Exception as exc:
        logger.error(f"  SCORER FAILED: {exc}")

    # ── 16. Save final summary ────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    # Safely retrieve weights — ensemble._weights is an ndarray or None
    raw_weights = getattr(ensemble, "_weights", None)
    weights_list = raw_weights.tolist() if raw_weights is not None else []
    summary = {
        "timestamp"        : time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds"  : elapsed,
        "models_used"      : [m.name for m in final_models],
        "ensemble_weights" : {
            m.name: round(w, 4)
            for m, w in zip(final_models, weights_list)
        },
    }
    reports_dir = root / cfg["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, reports_dir / "final_summary.json")

    # ── 17. Final banner ──────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(f"  PIPELINE COMPLETE in {elapsed}s")
    logger.info(f"  Outputs -> {root / cfg['paths']['outputs']}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
