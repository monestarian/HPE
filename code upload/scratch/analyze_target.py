import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data/raw')
files = sorted(DATA_DIR.glob('ports_batch_*.parquet'))

if files:
    df_list = []
    # Load first batch access ports to sample
    tmp = pd.read_parquet(files[0], columns=['port_id', 'port_role'])
    access_ports = tmp[tmp['port_role'] == 'access']['port_id'].unique()
    sample_ports = np.random.choice(access_ports, size=min(50, len(access_ports)), replace=False)
    
    for f in files[:2]:
        tmp = pd.read_parquet(f, columns=['port_id', 'timestamp', 'total_utilization', 'port_role'])
        tmp = tmp[tmp['port_id'].isin(sample_ports)]
        df_list.append(tmp)
    
    df = pd.concat(df_list, ignore_index=True).sort_values(['port_id', 'timestamp']).reset_index(drop=True)
    df['is_idle'] = df['total_utilization'] < 1.0
    
    def get_remaining_idle_duration(group):
        s = group['is_idle'].astype(int).iloc[::-1]
        counts = s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
        return counts.iloc[::-1]
        
    df['idle_duration_hours'] = (df.groupby('port_id', group_keys=False).apply(get_remaining_idle_duration) * 5) / 60.0
    
    idle_df = df[df['is_idle'] == True]
    print("Total idle samples:", len(idle_df))
    print("Target distribution percentiles:")
    print(idle_df['idle_duration_hours'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
else:
    print("No data found")
