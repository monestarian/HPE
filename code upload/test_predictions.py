import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

DATA_DIR = Path('data/raw')
MODEL_DIR = Path('models')

def test_long_term_model():
    logging.info("\n--- Testing Long-Term Duration Regressor ---")
    
    # Load model
    model = xgb.XGBRegressor(device='cpu')
    model.load_model(MODEL_DIR / 'long_term_duration_regressor.json')
    
    # Load some data
    files = sorted(DATA_DIR.glob('ports_batch_*.parquet'))
    tmp = pd.read_parquet(files[0], columns=['port_id', 'port_role'])
    access_ports = tmp[tmp['port_role'] == 'access']['port_id'].unique()
    sample_ports = np.random.choice(access_ports, size=10, replace=False)
    
    df_list = []
    for f in files:
        tmp = pd.read_parquet(f, columns=['port_id', 'timestamp', 'total_utilization', 'hour', 'day_of_week', 'is_weekend', 'consecutive_idle_windows', 'time_since_last_burst_minutes'])
        tmp = tmp[tmp['port_id'].isin(sample_ports)]
        df_list.append(tmp)
        
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(['port_id', 'timestamp']).reset_index(drop=True)
    
    # Features
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    df['idle_duration_minutes'] = df['consecutive_idle_windows'] * 5
    
    df['ema_7d'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x.ewm(span=2016, adjust=False).mean())
    df['ema_30d'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x.ewm(span=8640, adjust=False).mean())
    df['ema_7d_slope'] = df.groupby('port_id')['ema_7d'].transform(lambda x: x - x.shift(288))
    
    df['rolling_mean_1h'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    df['rolling_std_1h'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x.rolling(12, min_periods=1).std().fillna(0))
    df['rolling_slope_1h'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x - x.shift(12)).fillna(0)
    
    df['is_idle'] = df['total_utilization'] < 1.0
    
    def get_remaining_idle_duration(group):
        s = group['is_idle'].astype(int).iloc[::-1]
        counts = s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
        return counts.iloc[::-1]
        
    df['idle_duration_hours'] = (
        df.groupby('port_id', group_keys=False)
          .apply(get_remaining_idle_duration) * 5
    ) / 60.0
    
    df = df.dropna(subset=['ema_7d_slope'])
    
    # Let's test on idles that are reasonably long (e.g., weekend dips or overnight)
    df_test = df[(df['is_idle'] == True) & (df['idle_duration_hours'] > 12)] 
    
    if len(df_test) < 5:
        logging.info("Not enough >12h idles found in sample.")
        return
        
    features = [
        'ema_7d', 'ema_30d', 'ema_7d_slope',
        'rolling_mean_1h', 'rolling_std_1h', 'rolling_slope_1h',
        'sin_day', 'cos_day',
        'hour', 'sin_hour', 'cos_hour',
        'day_of_week', 'sin_dow', 'cos_dow',
        'is_weekend',
        'idle_duration_minutes', 'consecutive_idle_windows',
        'time_since_last_burst_minutes'
    ]
    
    # Pick 5 random long-duration samples
    sample_test = df_test.sample(5)
    
    X = sample_test[features]
    y_true = sample_test['idle_duration_hours'].values
    
    y_pred_log = model.predict(X)
    y_pred_log = np.clip(y_pred_log, 0, None)
    y_pred = np.expm1(y_pred_log)
    
    for i in range(5):
        logging.info(f"Sample {i+1}:")
        logging.info(f"  Timestamp: {sample_test.iloc[i]['timestamp']}")
        logging.info(f"  Actual Duration: {y_true[i]:.1f} hours")
        logging.info(f"  Predicted Duration: {y_pred[i]:.1f} hours")
        logging.info(f"  Error: {abs(y_true[i] - y_pred[i]):.1f} hours\n")

if __name__ == "__main__":
    test_long_term_model()
