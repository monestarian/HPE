import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
MODEL_DIR = Path(__file__).parent
MODEL_DIR.mkdir(exist_ok=True)

def train_poe():
    logging.info("Loading port data for PoE models (Using 1 month subset for memory)...")
    files = sorted(DATA_DIR.glob('ports_batch_*.parquet'))[:5] 
    
    df_list = []
    for f in files:
        tmp = pd.read_parquet(f, columns=['port_id', 'timestamp', 'rx_utilization', 'tx_utilization',
                                          'poe_consumption_watts', 'packet_rate_pps', 'active_ip_count',
                                          'endpoint_type', 'hour', 'day_of_week'])
        tmp = tmp[tmp['endpoint_type'].isin(['access_point', 'ip_phone', 'ip_camera'])]
        df_list.append(tmp)
        
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(['port_id', 'timestamp']).reset_index(drop=True)
    
    # 1. Device Fingerprinting Model ("what is on that poe")
    logging.info("Training Device Fingerprinting Model...")
    features = ['rx_utilization', 'tx_utilization', 'packet_rate_pps', 'active_ip_count']
    X_dev = df[features]
    
    label_map = {'access_point': 0, 'ip_phone': 1, 'ip_camera': 2}
    y_dev = df['endpoint_type'].map(label_map)
    
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)
    
    try:
        clf_dev = xgb.XGBClassifier(n_estimators=50, tree_method='hist', device='cuda')
        clf_dev.fit(X_train[:100], y_train[:100])
    except:
        clf_dev = xgb.XGBClassifier(n_estimators=50, tree_method='hist', device='cpu')
        
    clf_dev.fit(X_train, y_train)
    preds = clf_dev.predict(X_test)
    logging.info(f"Device Type Accuracy: {accuracy_score(y_test, preds):.4f}")
    
    # 2. PoE Shutdown Duration Model
    logging.info("Engineering shutdown duration labels...")
    df['is_idle'] = (df['rx_utilization'] + df['tx_utilization']) < 1.0
    
    def get_remaining_idle_duration(group):
        s = group['is_idle'].astype(int).iloc[::-1]
        counts = s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
        return counts.iloc[::-1]
        
    df['idle_duration_hours'] = (
        df.groupby('port_id', group_keys=False)
          .apply(get_remaining_idle_duration) * 5
    ) / 60.0
    
    df['can_shutdown'] = df['idle_duration_hours'] >= 1.0
    
    logging.info("Engineering Rolling Features...")
    df['rolling_mean_rx_1h'] = df.groupby('port_id')['rx_utilization'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    df['rolling_std_rx_1h'] = df.groupby('port_id')['rx_utilization'].transform(lambda x: x.rolling(12, min_periods=1).std().fillna(0))
    df['rolling_slope_rx_1h'] = df.groupby('port_id')['rx_utilization'].transform(lambda x: x - x.shift(12)).fillna(0)
    
    df['rolling_mean_tx_1h'] = df.groupby('port_id')['tx_utilization'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    df['rolling_std_tx_1h'] = df.groupby('port_id')['tx_utilization'].transform(lambda x: x.rolling(12, min_periods=1).std().fillna(0))
    df['rolling_slope_tx_1h'] = df.groupby('port_id')['tx_utilization'].transform(lambda x: x - x.shift(12)).fillna(0)
    
    features_poe = ['rx_utilization', 'tx_utilization', 'hour', 'day_of_week',
                    'rolling_mean_rx_1h', 'rolling_std_rx_1h', 'rolling_slope_rx_1h',
                    'rolling_mean_tx_1h', 'rolling_std_tx_1h', 'rolling_slope_tx_1h']
    X_poe = df[features_poe]
    y_clf = df['can_shutdown']
    y_reg = df['idle_duration_hours']
    y_reg_log = np.log1p(y_reg)
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X_poe.iloc[:split_idx], X_poe.iloc[split_idx:]
    y_clf_train, y_clf_test = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
    y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_reg_log_train, y_reg_log_test = y_reg_log.iloc[:split_idx], y_reg_log.iloc[split_idx:]
    
    logging.info("Training PoE Shutdown Classifier...")
    clf_poe = xgb.XGBClassifier(n_estimators=50, tree_method='hist', device=clf_dev.device)
    clf_poe.fit(X_train, y_clf_train)
    logging.info(f"PoE Shutdown Accuracy: {accuracy_score(y_clf_test, clf_poe.predict(X_test)):.4f}")
    
    logging.info("Training PoE Duration Regressor...")
    mask = y_clf_train == True
    reg_poe = xgb.XGBRegressor(n_estimators=50, tree_method='hist', device=clf_dev.device)
    reg_poe.fit(X_train[mask], y_reg_log_train[mask])
    
    mask_test = y_clf_test == True
    preds_poe_log = reg_poe.predict(X_test[mask_test])
    preds_poe = np.expm1(np.clip(preds_poe_log, 0, None))
    logging.info(f"PoE Duration MAE (Hours): {mean_absolute_error(y_reg_test[mask_test], preds_poe):.2f}")

    clf_dev.save_model(MODEL_DIR / 'poe_device_fingerprint.json')
    clf_poe.save_model(MODEL_DIR / 'poe_shutdown_classifier.json')
    reg_poe.save_model(MODEL_DIR / 'poe_duration_regressor.json')
    logging.info("PoE Models saved successfully.")

if __name__ == "__main__":
    train_poe()
