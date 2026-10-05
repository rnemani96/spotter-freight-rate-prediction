import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import math

from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    """Stateful transformer: fit on training data, transform any split."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lane_stats = None
        self.pickup_stats = None
        self.delivery_stats = None
        self.daily_market_ctx = None
        self.global_mi_median = None
        
        self.feature_cols = [
            # GEO (14)
            'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
            'haversine_km', 'distance', 'distance_km', 'dist_delta',
            'lat_diff', 'lon_diff', 'lat_midpoint', 'lon_midpoint',
            'is_east', 'is_north',
            # DISTANCE (2)
            'distance_bucket', 'distance_log',
            # EQUIPMENT (2)
            'equipment_code', 'equipment_premium',
            # WEIGHT (3)
            'weight', 'weight_log', 'weight_bucket',
            # TEMPORAL (18)
            'year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter',
            'is_weekend', 'is_monday', 'is_friday', 'month_sin', 'month_cos',
            'dow_sin', 'dow_cos', 'is_q4', 'is_peak_season', 'is_holiday_week', 'days_to_year_end',
            # MARKET (8)
            'market_index', 'market_index_sq', 'quote_signal', 'quote_signal_sq',
            'market_x_quote', 'rolling_7d_rate', 'rolling_30d_rate', 'daily_market_index',
            # LANE (7)
            'lane_rate_mean', 'lane_rate_median', 'lane_rate_std', 'lane_count',
            'lane_seen', 'pickup_city_avg_rate', 'delivery_city_avg_rate',
            # DERIVED (5)
            'expected_base_rate', 'rate_pressure_idx', 'dist_x_market',
            'dist_x_equip', 'lane_mean_x_market'
        ]

    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        """Compute lane stats, city stats, market context from training data."""
        logger.info("Fitting FeatureEngineer...")
        
        # Lane stats
        self.lane_stats = train_df.groupby(['pickup', 'delivery'])['posted_rate'].agg(
            lane_rate_mean='mean',
            lane_rate_median='median',
            lane_rate_std='std',
            lane_count='count'
        ).reset_index()
        
        # City stats
        self.pickup_stats = train_df.groupby('pickup')['posted_rate'].agg(
            pickup_city_avg_rate='mean'
        ).reset_index()
        
        self.delivery_stats = train_df.groupby('delivery')['posted_rate'].agg(
            delivery_city_avg_rate='mean'
        ).reset_index()
        
        # Daily market context
        # groupby('date').agg(daily_avg_rate, daily_quote_signal, daily_market_index)
        daily_ctx = train_df.groupby('date').agg(
            daily_avg_rate=('posted_rate', 'mean'),
            daily_quote_signal=('quote_signal', 'mean'),
            daily_market_index=('market_index', 'mean')
        ).reset_index().sort_values('date')
        
        # rolling 7d and 30d on daily_avg_rate
        daily_ctx.set_index('date', inplace=True)
        # Assuming date is consecutive, rolling window on rows. If gaps, min_periods handles it.
        daily_ctx['rolling_7d_rate'] = daily_ctx['daily_avg_rate'].rolling(window=7, min_periods=1).mean()
        daily_ctx['rolling_30d_rate'] = daily_ctx['daily_avg_rate'].rolling(window=30, min_periods=1).mean()
        self.daily_market_ctx = daily_ctx.reset_index()
        
        # Global market index median
        self.global_mi_median = train_df['market_index'].median()
        
        logger.info("FeatureEngineer fit complete.")
        return self
        
    def _vectorized_haversine(self, lat1, lon1, lat2, lon2):
        """Vectorised haversine distance in km."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all 62 features. Works for train/val/dec splits."""
        logger.info(f"Transforming data shape: {df.shape}")
        
        X = df.copy()
        
        # 1. Merging Lookups
        
        # Lane stats
        X = X.merge(self.lane_stats, on=['pickup', 'delivery'], how='left')
        X['lane_count'] = X['lane_count'].fillna(0)
        X['lane_seen'] = (X['lane_count'] > 0).astype(int)
        
        # City stats
        X = X.merge(self.pickup_stats, on='pickup', how='left')
        X = X.merge(self.delivery_stats, on='delivery', how='left')
        
        # Market ctx
        X = X.sort_values('date')
        ctx = self.daily_market_ctx.copy()
        
        # For future dates, use last known context
        merged = pd.merge_asof(X, ctx, on='date', direction='backward')
        
        # Update X
        X = merged
        
        # 2. Compute Features
        
        # --- DERIVED FIRST for fallback ---
        X['expected_base_rate'] = X['distance'] * 2.1
        
        # Fill missing lane/city stats
        X['lane_rate_mean'] = X['lane_rate_mean'].fillna(X['expected_base_rate'])
        X['lane_rate_median'] = X['lane_rate_median'].fillna(X['expected_base_rate'])
        X['lane_rate_std'] = X['lane_rate_std'].fillna(0)
        X['pickup_city_avg_rate'] = X['pickup_city_avg_rate'].fillna(X['expected_base_rate'])
        X['delivery_city_avg_rate'] = X['delivery_city_avg_rate'].fillna(X['expected_base_rate'])
        
        # --- GEO ---
        X['haversine_km'] = self._vectorized_haversine(X['pickup_lat'], X['pickup_lon'], X['delivery_lat'], X['delivery_lon'])
        X['distance_km'] = X['distance'] * 1.60934
        X['dist_delta'] = X['haversine_km'] - X['distance_km']
        X['lat_diff'] = X['delivery_lat'] - X['pickup_lat']
        X['lon_diff'] = X['delivery_lon'] - X['pickup_lon']
        X['lat_midpoint'] = (X['pickup_lat'] + X['delivery_lat']) / 2
        X['lon_midpoint'] = (X['pickup_lon'] + X['delivery_lon']) / 2
        X['is_east'] = (X['lon_diff'] > 0).astype(int)
        X['is_north'] = (X['lat_diff'] > 0).astype(int)
        
        # --- DISTANCE ---
        X['distance_bucket'] = pd.cut(X['distance'], bins=[0, 300, 600, 1000, 1500, 2500, 99999]).cat.codes.clip(lower=0)
        # Clip to 0 before log1p to prevent NaN from any negative distance values
        X['distance_log'] = np.log1p(X['distance'].clip(lower=0))

        # --- EQUIPMENT ---
        equip_code_map = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
        equip_prem_map = {"Dry Van": 1.000, "Reefer": 1.124, "Flatbed": 1.082}
        X['equipment_code'] = X['equipment'].map(equip_code_map).fillna(-1)
        X['equipment_premium'] = X['equipment'].map(equip_prem_map).fillna(1.0)

        # --- WEIGHT ---
        # Clip to 0 before log1p to prevent NaN from any negative/null weight values
        X['weight_log'] = np.log1p(X['weight'].clip(lower=0))
        X['weight_bucket'] = pd.cut(X['weight'], bins=[0, 10000, 20000, 30000, 40000, 99999]).cat.codes.clip(lower=0)
        
        # --- TEMPORAL ---
        dt = X['date'].dt
        X['year'] = dt.year
        X['month'] = dt.month
        X['day'] = dt.day
        X['dayofweek'] = dt.dayofweek
        X['dayofyear'] = dt.dayofyear
        X['weekofyear'] = dt.isocalendar().week.astype(int)
        X['quarter'] = dt.quarter
        
        X['is_weekend'] = (X['dayofweek'] >= 5).astype(int)
        X['is_monday'] = (X['dayofweek'] == 0).astype(int)
        X['is_friday'] = (X['dayofweek'] == 4).astype(int)
        
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        X['dow_sin'] = np.sin(2 * np.pi * X['dayofweek'] / 7)
        X['dow_cos'] = np.cos(2 * np.pi * X['dayofweek'] / 7)
        
        X['is_q4'] = X['month'].isin([10, 11, 12]).astype(int)
        X['is_peak_season'] = X['month'].isin([5, 6, 7, 8]).astype(int)
        
        # Holiday week logic
        # Dec 22-31, Nov 24-30, Jan 1-5
        is_xmas = (X['month'] == 12) & (X['day'] >= 22) & (X['day'] <= 31)
        is_thanksgiving = (X['month'] == 11) & (X['day'] >= 24) & (X['day'] <= 30)
        is_newyear = (X['month'] == 1) & (X['day'] >= 1) & (X['day'] <= 5)
        X['is_holiday_week'] = (is_xmas | is_thanksgiving | is_newyear).astype(int)
        
        X['days_to_year_end'] = 365 - X['dayofyear']
        
        # --- MARKET ---
        # daily_market_index already added by merge_asof
        # For missing daily_market_index (if any), fallback to global median
        X['daily_market_index'] = X['daily_market_index'].fillna(self.global_mi_median)
        # rolling_7d_rate and rolling_30d_rate are also from merge_asof
        
        # Fill quote_signal if null
        if 'quote_signal' not in X.columns:
            X['quote_signal'] = 0.0 # Just a fallback
        X['quote_signal'] = X['quote_signal'].fillna(0)
            
        X['market_index_sq'] = X['market_index'] ** 2
        X['quote_signal_sq'] = X['quote_signal'] ** 2
        X['market_x_quote'] = X['market_index'] * X['quote_signal']
        
        # --- DERIVED REST ---
        X['rate_pressure_idx'] = X['market_index'] * X['equipment_premium']
        X['dist_x_market'] = X['distance'] * X['market_index']
        X['dist_x_equip'] = X['distance'] * X['equipment_premium']
        X['lane_mean_x_market'] = X['lane_rate_mean'] * X['market_index']
        
        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def get_feature_cols(self) -> list[str]:
        return self.feature_cols

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "FeatureEngineer":
        return joblib.load(path)

