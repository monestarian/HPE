import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
MODEL_DIR = Path(__file__).parent
MODEL_DIR.mkdir(exist_ok=True)

def train_storm():
    logging.info("Loading port data for Storm Model...")
    files = sorted(DATA_DIR.glob('ports_batch_*.parquet'))[:5] 
    
    df_list = []
    for f in files:
        tmp = pd.read_parquet(f)
        if 'is_storm_affected' not in tmp.columns:
            tmp['is_storm_affected'] = False
        df_list.append(tmp)
        
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(['port_id', 'timestamp']).reset_index(drop=True)
    
    logging.info("Engineering storm prediction labels (Predicting 30 mins in advance)...")
    df['is_storm_affected'] = df['is_storm_affected'].fillna(False).astype(bool)
    
    # Forward-look: is there a storm in the next 6 intervals (30 mins)?
    # shift(-1) then rolling(6).max() looks at future rows within each port group
    def get_storm_future(group):
        return group['is_storm_affected'].shift(-1).rolling(window=6, min_periods=1).max()
        
    df['storm_in_next_30m'] = (
        df.groupby('port_id', group_keys=False)
          .apply(get_storm_future)
          .fillna(0)
          .astype(bool)
    )
    
    features = ['rx_utilization', 'tx_utilization', 'queue_drops', 'queue_depth', 'flow_count']
    X = df[features]
    y = df['storm_in_next_30m']
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    try:
        clf_device = 'cuda'
        tmp = xgb.XGBClassifier(n_estimators=10, device=clf_device)
        tmp.fit(X_train[:100], y_train[:100])
    except:
        clf_device = 'cpu'
        
    logging.info("Training Storm Early Warning Classifier...")
    clf = xgb.XGBClassifier(n_estimators=100, tree_method='hist', device=clf_device)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    # Try to calculate AUC if there are positive cases in the test set
    try:
        auc = roc_auc_score(y_test, probs)
        logging.info(f"Storm Prediction AUC-ROC: {auc:.4f}")
    except:
        logging.info("Not enough storm cases in test split to calculate AUC.")
        
    logging.info(f"Storm Prediction Accuracy: {accuracy_score(y_test, preds):.4f}")
    
    clf.save_model(MODEL_DIR / 'storm_warning_classifier.json')
    logging.info("Storm Model saved successfully.")

if __name__ == "__main__":
    train_storm()
