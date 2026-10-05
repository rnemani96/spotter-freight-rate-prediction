import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger
from src.models.base import BaseModel
from src.features.engineer import FeatureEngineer

class Predictor:
    """Handles generating and writing model predictions."""
    
    def __init__(self, cfg: dict, model: BaseModel, feature_engineer: FeatureEngineer, root: Path):
        self.cfg = cfg
        self.model = model
        self.fe = feature_engineer
        self.root = root
        self.output_dir = root / cfg['paths']['outputs']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

    def predict_validation(self, val_feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for the validation set.
        
        Args:
            val_feat_df (pd.DataFrame): Feature-engineered validation data.
            
        Returns:
            pd.DataFrame: DataFrame containing 'load_id' and 'predicted_rate'.
        """
        self.logger.info("Generating predictions for validation set.")
        # Ensure we don't pass 'load_id' or targets to model
        features = self.fe.get_feature_cols()
        X_val = val_feat_df[features].values
        
        preds = self.model.predict(X_val)
        # Clip predictions to minimum 1.0
        preds = np.clip(preds, a_min=1.0, a_max=None)
        
        out_df = pd.DataFrame({
            'load_id': val_feat_df['load_id'],
            'predicted_rate': preds
        })
        return out_df

    def predict_december(self, dec_feat_df: pd.DataFrame, dec_raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for the December set.
        
        Args:
            dec_feat_df (pd.DataFrame): Feature-engineered December data.
            dec_raw_df (pd.DataFrame): Original December data for column structure.
            
        Returns:
            pd.DataFrame: Original dataframe populated with predicted_rate.
        """
        self.logger.info("Generating predictions for December set.")
        features = self.fe.get_feature_cols()
        X_dec = dec_feat_df[features].values
        
        preds = self.model.predict(X_dec)
        preds = np.clip(preds, a_min=1.0, a_max=None)
        
        # Ensure columns are exactly as required: pickup, delivery, distance, equipment, weight, date, predicted_rate
        out_df = dec_raw_df.copy()
        out_df['predicted_rate'] = preds
        
        expected_cols = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']
        out_df = out_df[expected_cols]
        
        return out_df

    def write_outputs(self, val_preds: pd.DataFrame, dec_preds: pd.DataFrame, template_df: pd.DataFrame) -> None:
        """
        Write validation and December predictions to CSV formats.
        
        Args:
            val_preds (pd.DataFrame): Validation predictions.
            dec_preds (pd.DataFrame): December predictions.
            template_df (pd.DataFrame): Template dataframe to dictate output order for validation.
        """
        self.logger.info("Writing predictions to outputs directory.")
        
        # 1. Match val_preds to template by load_id order
        # Set load_id as index to easily align
        val_preds_indexed = val_preds.set_index('load_id')
        ordered_preds = template_df[['load_id']].copy()
        ordered_preds = ordered_preds.join(val_preds_indexed, on='load_id', how='left')
        
        # 4. Assert no nulls, all positive before writing
        assert ordered_preds['predicted_rate'].notna().all(), "Validation predictions contain nulls."
        assert (ordered_preds['predicted_rate'] >= 1.0).all(), "Validation predictions contain values < 1.0."
        
        assert dec_preds['predicted_rate'].notna().all(), "December predictions contain nulls."
        assert (dec_preds['predicted_rate'] >= 1.0).all(), "December predictions contain values < 1.0."
        
        # 2. Write outputs/validation_predictions.csv
        val_path = self.output_dir / 'validation_predictions.csv'
        ordered_preds.to_csv(val_path, index=False)
        self.logger.info(f"Wrote validation predictions to {val_path}")
        
        # 3. Write outputs/december-chart-inputs.csv
        dec_path = self.output_dir / 'december-chart-inputs.csv'
        dec_preds.to_csv(dec_path, index=False)
        self.logger.info(f"Wrote December predictions to {dec_path}")


