import os
import gc
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
MODEL_DIR = Path(__file__).parent
MODEL_DIR.mkdir(exist_ok=True)

def engineer_labels(df):
    logging.info("Engineering forward-looking idle labels...")
    # Port is idle if total util is very low
    df['is_idle'] = df['total_utilization'] < 1.0
    
    # Reverse within each port group to count remaining consecutive idle intervals
    def get_remaining_idle_duration(group):
        s = group['is_idle'].astype(int).iloc[::-1]
        counts = s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
        return counts.iloc[::-1]
        
    df['idle_duration_future_mins'] = (
        df.groupby('port_id', group_keys=False)
          .apply(get_remaining_idle_duration) * 5
    )
    
    # Binary target: Will it be idle for at least 1 hour (60 mins)?
    df['is_idle_next_1_hour'] = df['idle_duration_future_mins'] >= 60
    return df

def train_port_idle_model():
    logging.info("Loading port data...")
    # Load just access ports from all batches to save memory
    files = sorted(DATA_DIR.glob('ports_batch_*.parquet'))
    
    # We might not have enough RAM for 6 months of all ports. 
    # Let's sample 20% of access ports.
    df_list = []
    sample_ports = None
    
    for f in files:
        tmp = pd.read_parquet(f, columns=['port_id', 'timestamp', 'total_utilization', 'port_role', 
                                          'hour', 'minute', 'day_of_week', 'is_weekend',
                                          'consecutive_idle_windows', 'is_burst'])
        tmp = tmp[tmp['port_role'] == 'access']
        
        if sample_ports is None:
            all_ports = tmp['port_id'].unique()
            sample_ports = np.random.choice(all_ports, size=max(1, len(all_ports)//5), replace=False)
            
        tmp = tmp[tmp['port_id'].isin(sample_ports)]
        df_list.append(tmp)
        
    if not df_list:
        logging.error("No data found! Has the generator finished running?")
        return
        
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(['port_id', 'timestamp']).reset_index(drop=True)
    
    df = engineer_labels(df)
    
    logging.info("Engineering Rolling Features...")
    df['rolling_mean_1h'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    df['rolling_std_1h'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x.rolling(12, min_periods=1).std().fillna(0))
    df['rolling_slope_1h'] = df.groupby('port_id')['total_utilization'].transform(lambda x: x - x.shift(12)).fillna(0)
    
    # Features for the model
    features = ['total_utilization', 'hour', 'minute', 'day_of_week', 'is_weekend', 
                'consecutive_idle_windows', 'is_burst',
                'rolling_mean_1h', 'rolling_std_1h', 'rolling_slope_1h']
    
    X = df[features]
    y_clf = df['is_idle_next_1_hour']
    y_reg = df['idle_duration_future_mins'] / 60.0 # Convert to hours
    y_reg_log = np.log1p(y_reg)
    
    # Split chronologically to avoid data leakage
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_clf_train, y_clf_test = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
    y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_reg_log_train, y_reg_log_test = y_reg_log.iloc[:split_idx], y_reg_log.iloc[split_idx:]
    
    # Try to use GPU if available, else fallback to CPU
    try:
        clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, tree_method='hist', device='cuda')
        clf.fit(X_train[:100], y_clf_train[:100]) # Quick test
    except Exception as e:
        logging.warning(f"CUDA failed ({e}), falling back to CPU")
        clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, tree_method='hist', device='cpu')
    
    logging.info("Training Classification Model (Will it be idle > 1hr?)...")
    clf.fit(X_train, y_clf_train)
    
    preds_clf = clf.predict(X_test)
    logging.info(f"Classifier Accuracy: {accuracy_score(y_clf_test, preds_clf):.4f}")
    logging.info(f"Classifier F1-Score: {f1_score(y_clf_test, preds_clf):.4f}")
    
    logging.info("Training Regression Model (How many hours?)...")
    # Only train regression on instances where it actually IS going to be idle
    mask_train = y_clf_train == True
    mask_test = y_clf_test == True
    
    reg = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        tree_method='hist',
        device=clf.device
    )
    reg.fit(X_train[mask_train], y_reg_log_train[mask_train])
    
    preds_reg_log = reg.predict(X_test[mask_test])
    preds_reg = np.expm1(np.clip(preds_reg_log, 0, None))
    logging.info(f"Regressor MAE (Hours): {mean_absolute_error(y_reg_test[mask_test], preds_reg):.2f}")
    logging.info(f"Regressor RMSE (Hours): {np.sqrt(mean_squared_error(y_reg_test[mask_test], preds_reg)):.2f}")
    
    # Save models
    clf.save_model(MODEL_DIR / 'port_idle_classifier.json')
    reg.save_model(MODEL_DIR / 'port_idle_duration_regressor.json')
    logging.info("Models saved successfully.")

if __name__ == "__main__":
    train_port_idle_model()
