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

def train_lag():
    logging.info("Loading LAG data...")
    files = sorted(DATA_DIR.glob('lags_batch_*.parquet'))
    
    df_list = []
    for f in files:
        df_list.append(pd.read_parquet(f))
        
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(['lag_id', 'timestamp']).reset_index(drop=True)
    
    # 1. Engineer labels
    # Can shut down if total utilization < 20%
    df['can_shutdown_member'] = df['aggregate_utilization'] < 20.0
    
    def get_remaining_shutdown_duration(group):
        s = group['can_shutdown_member'].astype(int).iloc[::-1]
        counts = s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
        return counts.iloc[::-1]
        
    df['shutdown_duration_hours'] = (
        df.groupby('lag_id', group_keys=False)
          .apply(get_remaining_shutdown_duration) * 5
    ) / 60.0
    
    # Filter to cases where it's at least 1 hour
    df['valid_shutdown'] = df['shutdown_duration_hours'] >= 1.0
    
    # Vectorised reason label (Bug 3 fix: replaces slow row-by-row apply)
    ts = pd.to_datetime(df['timestamp'])
    df['hour'] = ts.dt.hour
    df['is_weekend'] = ts.dt.weekday >= 5
    df['reason_code'] = 2  # default: Other
    df.loc[(df['hour'] >= 18) | (df['hour'] < 6), 'reason_code'] = 0   # Nightly
    df.loc[df['is_weekend'], 'reason_code'] = 1                          # Weekend (overrides nightly)
    
    logging.info("Engineering Rolling Features...")
    df['rolling_mean_util_1h'] = df.groupby('lag_id')['aggregate_utilization'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    df['rolling_std_util_1h'] = df.groupby('lag_id')['aggregate_utilization'].transform(lambda x: x.rolling(12, min_periods=1).std().fillna(0))
    df['rolling_slope_util_1h'] = df.groupby('lag_id')['aggregate_utilization'].transform(lambda x: x - x.shift(12)).fillna(0)
    
    features = ['aggregate_utilization', 'member_imbalance_ratio', 'total_members',
                'hour', 'is_weekend',
                'rolling_mean_util_1h', 'rolling_std_util_1h', 'rolling_slope_util_1h']
    X = df[features]
    
    y_clf = df['can_shutdown_member']
    y_reg = df['shutdown_duration_hours']
    y_reg_log = np.log1p(y_reg)
    y_reason = df['reason_code']
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_clf_train, y_clf_test = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
    y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_reg_log_train, y_reg_log_test = y_reg_log.iloc[:split_idx], y_reg_log.iloc[split_idx:]
    y_reason_train, y_reason_test = y_reason.iloc[:split_idx], y_reason.iloc[split_idx:]
    
    try:
        clf_device = 'cuda'
        tmp = xgb.XGBClassifier(n_estimators=10, device=clf_device)
        tmp.fit(X_train[:100], y_clf_train[:100])
    except:
        clf_device = 'cpu'
        
    logging.info("Training LAG Shutdown Classifier...")
    clf = xgb.XGBClassifier(n_estimators=100, tree_method='hist', device=clf_device)
    clf.fit(X_train, y_clf_train)
    logging.info(f"LAG Shutdown Accuracy: {accuracy_score(y_clf_test, clf.predict(X_test)):.4f}")
    
    logging.info("Training LAG Reason Classifier...")
    mask = y_clf_train == True
    clf_reason = xgb.XGBClassifier(n_estimators=50, tree_method='hist', device=clf_device)
    clf_reason.fit(X_train[mask], y_reason_train[mask])
    
    mask_test = y_clf_test == True
    logging.info(f"LAG Reason Accuracy: {accuracy_score(y_reason_test[mask_test], clf_reason.predict(X_test[mask_test])):.4f}")
    
    logging.info("Training LAG Duration Regressor...")
    reg = xgb.XGBRegressor(n_estimators=100, tree_method='hist', device=clf_device)
    reg.fit(X_train[mask], y_reg_log_train[mask])
    
    preds_lag_log = reg.predict(X_test[mask_test])
    preds_lag = np.expm1(np.clip(preds_lag_log, 0, None))
    logging.info(f"LAG Duration MAE (Hours): {mean_absolute_error(y_reg_test[mask_test], preds_lag):.2f}")

    clf.save_model(MODEL_DIR / 'lag_shutdown_classifier.json')
    clf_reason.save_model(MODEL_DIR / 'lag_reason_classifier.json')
    reg.save_model(MODEL_DIR / 'lag_duration_regressor.json')
    logging.info("LAG Models saved successfully.")

if __name__ == "__main__":
    train_lag()
