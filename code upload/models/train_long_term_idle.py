import os
import gc
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
MODEL_DIR = Path(__file__).parent
MODEL_DIR.mkdir(exist_ok=True)

def train_long_term_idle():
    logging.info("Loading port data for long-term prediction...")
    files = sorted(DATA_DIR.glob('ports_batch_*.parquet'))
    
    # Filter by a fixed set of ports to maintain continuous time series in memory
    sample_ports = None
    if files:
        tmp = pd.read_parquet(files[0], columns=['port_id', 'port_role'])
        access_ports = tmp[tmp['port_role'] == 'access']['port_id'].unique()
        sample_ports = np.random.choice(access_ports, size=min(100, len(access_ports)), replace=False)
    
    df_list = []
    for f in files:
        tmp = pd.read_parquet(f, columns=[
            'port_id', 'timestamp', 'total_utilization', 'port_role',
            'idle_duration_minutes', 'consecutive_idle_windows',
            'time_since_last_burst_minutes',
            'hour', 'day_of_week', 'is_weekend',
        ])
        tmp = tmp[tmp['port_id'].isin(sample_ports)]
        df_list.append(tmp)
        
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(['port_id', 'timestamp']).reset_index(drop=True)
    
    logging.info("Engineering Macro-Level Trend Features...")
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
    
    # Cyclical daily encoding
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # Cyclical weekly encoding
    df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    logging.info("Calculating EMAs (7-day and 30-day)...")
    df['ema_7d'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x.ewm(span=2016, adjust=False).mean())
    df['ema_30d'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x.ewm(span=8640, adjust=False).mean())
    df['ema_7d_slope'] = df.groupby('port_id')['ema_7d'].transform(lambda x: x - x.shift(288))
    
    logging.info("Engineering idle duration target...")
    df['is_idle'] = df['total_utilization'] < 1.0
    
    def get_remaining_idle_duration(group):
        """Reverse within-group to count remaining consecutive idle intervals."""
        s = group['is_idle'].astype(int).iloc[::-1]
        counts = s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
        return counts.iloc[::-1]
        
    df['idle_duration_hours'] = (
        df.groupby('port_id', group_keys=False)
          .apply(get_remaining_idle_duration) * 5
    ) / 60.0
    
    logging.info("Engineering Rolling Features...")
    df['rolling_mean_1h'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    df['rolling_std_1h'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x.rolling(12, min_periods=1).std().fillna(0))
    df['rolling_slope_1h'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x - x.shift(12)).fillna(0)
    
    df = df.dropna(subset=['ema_7d_slope'])
    df_train = df[df['is_idle'] == True]
    
    features = [
        # Trend
        'ema_7d', 'ema_30d', 'ema_7d_slope',
        'rolling_mean_1h', 'rolling_std_1h', 'rolling_slope_1h',
        # Annual seasonality
        'sin_day', 'cos_day',
        # Daily cyclical
        'hour', 'sin_hour', 'cos_hour',
        # Weekly cyclical
        'day_of_week', 'sin_dow', 'cos_dow',
        'is_weekend',
        # Port-level state
        'idle_duration_minutes', 'consecutive_idle_windows',
        'time_since_last_burst_minutes',
    ]
    X = df_train[features]
    y = df_train['idle_duration_hours']
    y_log = np.log1p(y)  # log-transform to compress heavy right tail
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_log.iloc[:split_idx], y_log.iloc[split_idx:]
    y_test_hours = y.iloc[split_idx:]  # original scale for MAE/R²
    
    try:
        device = 'cuda'
        tmp = xgb.XGBRegressor(n_estimators=10, device=device)
        tmp.fit(X_train[:100], y_train[:100])
    except:
        device = 'cpu'
        
    logging.info(f"Training Long-Term Duration Regressor on log1p target (device={device})...")
    reg = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, tree_method='hist', device=device)
    reg.fit(X_train, y_train)
    
    # Predictions are in log-space — back-transform for interpretable metrics
    preds_log = reg.predict(X_test)
    preds_log = np.clip(preds_log, 0, None)
    preds_hours = np.expm1(preds_log)
    
    # Log-space metric (directly comparable to RMSLE)
    rmse_log = np.sqrt(mean_absolute_error(y_test, preds_log))
    # Original-space metrics
    mae = mean_absolute_error(y_test_hours, preds_hours)
    r2 = r2_score(y_test_hours, preds_hours)
    
    logging.info(f"RMSE (log-space): {rmse_log:.4f}")
    logging.info(f"MAE (Mean Absolute Error, Hours): {mae:.2f}")
    logging.info(f"R2 Score (original scale): {r2:.4f}")
    
    reg.save_model(MODEL_DIR / 'long_term_duration_regressor.json')
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    train_long_term_idle()
