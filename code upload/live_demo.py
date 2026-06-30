import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
from pathlib import Path
import traceback
import sys
import networkx as nx

# Ensure generator module can be imported
sys.path.append(str(Path(__file__).parent))
from generator.topology import CampusTopology

st.set_page_config(layout="wide", page_title="AI Network Live Demo", page_icon="⚡")

DATA_DIR = Path('data/raw')
MODEL_DIR = Path('models')

# -------------------------------------------------------------
# 1. State Management & Model Loading
# -------------------------------------------------------------
@st.cache_resource
def load_models_v2():
    models = {}
    model_names = [
        'port_idle_classifier', 'port_idle_duration_regressor',
        'poe_device_fingerprint', 'poe_shutdown_classifier', 'poe_duration_regressor',
        'lag_shutdown_classifier', 'lag_reason_classifier', 'lag_duration_regressor',
        'storm_warning_classifier', 'long_term_duration_regressor'
    ]
    for m in model_names:
        file_path = MODEL_DIR / f"{m}.json"
        if file_path.exists():
            if 'classifier' in m or 'fingerprint' in m:
                mod = xgb.XGBClassifier()
            else:
                mod = xgb.XGBRegressor()
            mod.load_model(file_path)
            models[m] = mod
    return models

models = load_models_v2()

if "topology" not in st.session_state:
    st.session_state.topology = CampusTopology()
    st.session_state.pos = nx.spring_layout(st.session_state.topology.graph, seed=42)

# Scenario Definitions (Using dynamic lookup for reliability)
def get_valid_id(entity_type, role=None):
    if entity_type == "lag":
        return list(st.session_state.topology.lags.keys())[0]
    else:
        for p in st.session_state.topology.ports.values():
            if role and p.endpoint_type != role: continue
            if p.role == "access": return p.id
        return list(st.session_state.topology.ports.keys())[0]

SCENARIOS = {
    "1A: Overnight Idle": {"type": "port", "id": get_valid_id("port", "student_pc"), "jump": "2024-01-08 15:00:00", "event": "2024-01-08 18:00:00"},
    "1B: Weekend Idle": {"type": "port", "id": get_valid_id("port", "student_pc"), "jump": "2024-01-12 15:00:00", "event": "2024-01-12 18:00:00"},
    "1C: Between-Class Drop": {"type": "port", "id": get_valid_id("port", "student_pc"), "jump": "2024-01-09 06:50:00", "event": "2024-01-09 09:50:00"},
    "2A: AP PoE Overnight": {"type": "port", "id": get_valid_id("port", "access_point"), "jump": "2024-01-08 19:00:00", "event": "2024-01-08 22:00:00"},
    "2B: IP Phone PoE Weekend": {"type": "port", "id": get_valid_id("port", "ip_phone"), "jump": "2024-01-12 15:00:00", "event": "2024-01-12 18:00:00"},
    "3A: Nightly LAG Removal": {"type": "lag", "id": get_valid_id("lag"), "jump": "2024-01-08 15:00:00", "event": "2024-01-08 18:00:00"},
    "3B: Weekend LAG Removal": {"type": "lag", "id": get_valid_id("lag"), "jump": "2024-01-12 15:00:00", "event": "2024-01-12 18:00:00"},
    "4A: Broadcast Storm": {"type": "port", "id": get_valid_id("port"), "jump": "2024-01-03 11:15:00", "event": "2024-01-03 14:15:00"},
    "5A: Out of Session Break": {"type": "port", "id": get_valid_id("port", "student_pc"), "jump": "2024-03-05 05:00:00", "event": "2024-03-05 08:00:00"},
}

if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "speed" not in st.session_state:
    st.session_state.speed = 10
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "target_df" not in st.session_state:
    st.session_state.target_df = None
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None
if "reasoning" not in st.session_state:
    st.session_state.reasoning = []
if "global_is_playing" not in st.session_state:
    st.session_state.global_is_playing = False
if "global_index" not in st.session_state:
    st.session_state.global_index = 0
if "port_state" not in st.session_state:
    st.session_state.port_state = "ACTIVE"
if "wake_events" not in st.session_state:
    st.session_state.wake_events = []

# -------------------------------------------------------------
# 2. Data Buffering & Feature Engineering
# -------------------------------------------------------------
@st.cache_data
def load_scenario_data(scenario_key):
    """Pre-load and compute all features for a specific scenario target."""
    s = SCENARIOS[scenario_key]
    target_id = s["id"]
    is_lag = s["type"] == "lag"
    
    files = sorted(DATA_DIR.glob('lags_batch_*.parquet' if is_lag else 'ports_batch_*.parquet'))
    
    df_list = []
    # Load all files to ensure we have the long-term history for EMA
    for f in files:
        # Optimization: PyArrow engine with columns and filters is faster
        try:
            if is_lag:
                tmp = pd.read_parquet(f, columns=['lag_id', 'timestamp', 'aggregate_utilization', 'member_imbalance_ratio', 'total_members'])
                tmp = tmp[tmp['lag_id'] == target_id]
            else:
                tmp = pd.read_parquet(f, columns=['port_id', 'timestamp', 'total_utilization', 'rx_utilization', 'tx_utilization', 
                                                  'queue_drops', 'queue_depth', 'flow_count', 'is_burst',
                                                  'poe_consumption_watts', 'packet_rate_pps', 'active_ip_count'])
                tmp = tmp[tmp['port_id'] == target_id]
            df_list.append(tmp)
        except Exception:
            continue
            
    if not df_list: return None
    
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Feature Engineering (Vectorized for the entire timeline)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    if not is_lag:
        # EMAs
        df['ema_7d'] = df['total_utilization'].ewm(span=2016, adjust=False).mean()
        df['ema_30d'] = df['total_utilization'].ewm(span=8640, adjust=False).mean()
        df['ema_7d_slope'] = df['ema_7d'] - df['ema_7d'].shift(288)
        df['ema_7d_slope'] = df['ema_7d_slope'].fillna(0)
        
        # Rolling stats for Port Idle / Long Term
        df['rolling_mean_1h'] = df['total_utilization'].rolling(12, min_periods=1).mean()
        df['rolling_std_1h'] = df['total_utilization'].rolling(12, min_periods=1).std().fillna(0)
        df['rolling_slope_1h'] = (df['total_utilization'] - df['total_utilization'].shift(12)).fillna(0)
        
        # Rolling stats for PoE
        df['rolling_mean_rx_1h'] = df['rx_utilization'].rolling(12, min_periods=1).mean()
        df['rolling_std_rx_1h'] = df['rx_utilization'].rolling(12, min_periods=1).std().fillna(0)
        df['rolling_slope_rx_1h'] = (df['rx_utilization'] - df['rx_utilization'].shift(12)).fillna(0)
        
        df['rolling_mean_tx_1h'] = df['tx_utilization'].rolling(12, min_periods=1).mean()
        df['rolling_std_tx_1h'] = df['tx_utilization'].rolling(12, min_periods=1).std().fillna(0)
        df['rolling_slope_tx_1h'] = (df['tx_utilization'] - df['tx_utilization'].shift(12)).fillna(0)
        
        # State trackers
        df['is_idle_frame'] = df['total_utilization'] < 1.0
        # Consecutive idle windows
        df['consecutive_idle_windows'] = df.groupby((df['is_idle_frame'] != df['is_idle_frame'].shift()).cumsum()).cumcount() + 1
        df.loc[~df['is_idle_frame'], 'consecutive_idle_windows'] = 0
        df['idle_duration_minutes'] = df['consecutive_idle_windows'] * 5
        
        # Time since burst
        burst_blocks = df['is_burst'].cumsum()
        df['time_since_last_burst_minutes'] = df.groupby(burst_blocks).cumcount() * 5
        df.loc[df['is_burst'], 'time_since_last_burst_minutes'] = 0
    else:
        # Rolling stats for LAG
        df['rolling_mean_util_1h'] = df['aggregate_utilization'].rolling(12, min_periods=1).mean()
        df['rolling_std_util_1h'] = df['aggregate_utilization'].rolling(12, min_periods=1).std().fillna(0)
        df['rolling_slope_util_1h'] = (df['aggregate_utilization'] - df['aggregate_utilization'].shift(12)).fillna(0)
        
    return df

# -------------------------------------------------------------
@st.cache_data
def load_global_data():
    ports_df = pd.read_parquet(DATA_DIR / "ports_batch_0.parquet")
    devices_df = pd.read_parquet(DATA_DIR / "devices_batch_0.parquet")
    return ports_df, devices_df

# 3. Sidebar & Controls
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚡ AI Predictive Telemetry")
    
    st.header("Simulation Controls")
    cols = st.columns(3)
    if cols[0].button("▶️ Play"): st.session_state.is_playing = True
    if cols[1].button("⏸️ Pause"): st.session_state.is_playing = False
    
    st.session_state.speed = st.slider("Playback Speed (Intervals/sec)", 1, 100, 10)
    
    st.header("Jump to Scenarios")
    
    # Allow user to pick a different target within the scenario
    target_options = []
    
    for group in ["1. Port Idle", "2. PoE Power Down", "3. LAG Removal", "4. Storm Warning", "5. Long Holiday"]:
        with st.expander(group):
            for k, v in SCENARIOS.items():
                if k.startswith(group[0]):
                    if st.button(k):
                        with st.spinner(f"Buffering telemetry for {v['id']}..."):
                            df = load_scenario_data(k)
                            if df is not None and len(df) > 0:
                                st.session_state.target_df = df
                                st.session_state.current_scenario = k
                                st.session_state.reasoning = []
                                st.session_state.port_state = "ACTIVE"
                                st.session_state.wake_events = []
                                jump_dt = pd.to_datetime(v["jump"])
                                # Safely find index
                                future_rows = df[df['timestamp'] >= jump_dt]
                                if not future_rows.empty:
                                    idx = future_rows.index[0]
                                else:
                                    idx = 0
                                st.session_state.current_index = idx
                                st.session_state.is_playing = True
                                st.rerun()
                            else:
                                st.error("No telemetry data found for this entity.")

if "topology" not in st.session_state:
    st.session_state.topology = CampusTopology()
    st.session_state.pos = nx.spring_layout(st.session_state.topology.graph, seed=42)

# -------------------------------------------------------------
# 4. Main Dashboard UI

tab_scenario, tab_global = st.tabs(["Scenario Simulation", "Global Campus View"])

with tab_scenario:
    if st.session_state.target_df is None:
        st.info("Select a scenario from the sidebar to begin the live telemetry simulation.")
    else:

        df = st.session_state.target_df
        idx = st.session_state.current_index
        scenario_meta = SCENARIOS[st.session_state.current_scenario]
        is_lag = scenario_meta["type"] == "lag"

        current_row = df.iloc[idx]

        target_id = scenario_meta["id"]
        if is_lag:
            target_switch_id = st.session_state.topology.lags[target_id].switch_id
        else:
            target_switch_id = st.session_state.topology.ports[target_id].switch_id

        st.markdown(f"### 📍 Target: `{scenario_meta['id']}` | 🕒 Current Time: `{current_row['timestamp']}`")

        # Metrics Row
        if not is_lag:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Utilization", f"{current_row['total_utilization']:.2f}%")
            m2.metric("Queue Depth", f"{current_row['queue_depth']} pkts")
            m3.metric("Queue Drops", f"{current_row['queue_drops']}")
            m4.metric("PoE Draw", f"{current_row['poe_consumption_watts']} W")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Aggregate Utilization", f"{current_row['aggregate_utilization']:.2f}%")
            m2.metric("Active Members", f"{current_row['total_members']}")
            m3.metric("Imbalance Ratio", f"{current_row['member_imbalance_ratio']:.2f}")

        # Inference & Reasoning Engine
        # Max log1p cap: log1p(72 hours) ≈ 4.29. Prevents expm1 overflow to inf.
        MAX_LOG_DURATION = np.log1p(72.0)
        reasoning_msg = None            
        try:
            if st.session_state.current_scenario.startswith("1"):
                # Port Idle — model expects 10 features including rolling stats
                x_class = current_row[['total_utilization', 'hour', 'minute', 'day_of_week', 'is_weekend', 'consecutive_idle_windows', 'is_burst', 'rolling_mean_1h', 'rolling_std_1h', 'rolling_slope_1h']].to_frame().T.astype(float)
                is_idle_prob = models['port_idle_classifier'].predict_proba(x_class)[0][1]
                is_idle = is_idle_prob >= 0.5
                duration = 0
                if is_idle:
                    duration_log = np.clip(models['port_idle_duration_regressor'].predict(x_class)[0], 0, MAX_LOG_DURATION)
                    duration = np.expm1(duration_log) * 60  # hours → minutes
                
                features_str = "Features: total_utilization, hour, minute, day_of_week, is_weekend, consecutive_idle_windows, is_burst, rolling_mean_1h, rolling_std_1h, rolling_slope_1h"
                
                if is_idle and duration >= 60.0:
                    if st.session_state.port_state == "ACTIVE":
                        st.session_state.port_state = "SHUTDOWN"
                        reasoning_msg = f"🔴 **Port Idle Predictor ({target_id} on {target_switch_id})**: Port Shutdown Triggered! Predicted idle duration: {duration:.0f} mins. Recommended: Disable port to save switching power. ({features_str})"
                    else:
                        reasoning_msg = f"💤 **Port Idle Predictor ({target_id} on {target_switch_id})**: Port is in Low Energy State. Predicted duration remaining: {duration:.0f} mins. Action: Port disabled."
                else:
                    if st.session_state.port_state == "SHUTDOWN":
                        st.session_state.port_state = "ACTIVE"
                        st.session_state.wake_events.append(current_row['timestamp'])
                        reasoning_msg = f"🟢 **Port Idle Predictor ({target_id} on {target_switch_id})**: Activity Detected — Port Turned On! Recommended: Re-enable port. ({features_str})"
                    else:
                        reasoning_msg = f"🔵 **Port Idle Predictor ({target_id} on {target_switch_id})**: Port is active. No action needed. ({features_str})"
 
            elif st.session_state.current_scenario.startswith("2"):
                # PoE — model expects 10 features including rolling rx/tx stats
                x_dev = current_row[['rx_utilization', 'tx_utilization', 'packet_rate_pps', 'active_ip_count']].to_frame().T.astype(float)
                dev_class = models['poe_device_fingerprint'].predict(x_dev)[0]
                dev_map = {0: 'Access Point', 1: 'IP Phone', 2: 'IP Camera'}
 
                x_poe = current_row[['rx_utilization', 'tx_utilization', 'hour', 'day_of_week', 'rolling_mean_rx_1h', 'rolling_std_rx_1h', 'rolling_slope_rx_1h', 'rolling_mean_tx_1h', 'rolling_std_tx_1h', 'rolling_slope_tx_1h']].to_frame().T.astype(float)
                poe_shut_prob = models['poe_shutdown_classifier'].predict_proba(x_poe)[0][1]
                poe_shut = poe_shut_prob >= 0.5
                duration = 0
                if poe_shut:
                    dur_log = np.clip(models['poe_duration_regressor'].predict(x_poe)[0], 0, MAX_LOG_DURATION)
                    duration = np.expm1(dur_log) * 60
                
                features_str = "Features: rx_utilization, tx_utilization, hour, day_of_week, rolling_mean_rx_1h, rolling_std_rx_1h, rolling_slope_rx_1h, rolling_mean_tx_1h, rolling_std_tx_1h, rolling_slope_tx_1h"
                poe_watts = current_row['poe_consumption_watts']
                
                if poe_shut and duration >= 60.0:
                    if st.session_state.port_state == "ACTIVE":
                        st.session_state.port_state = "SHUTDOWN"
                        reasoning_msg = f"⚡ **PoE Power Down ({target_id} on {target_switch_id})**: Shutdown Triggered for {dev_map[dev_class]}! Predicted idle: {duration:.0f} mins. Recommended: Cut PoE power, save {poe_watts:.1f}W. ({features_str})"
                    else:
                        reasoning_msg = f"💤 **PoE Power Down ({target_id} on {target_switch_id})**: {dev_map[dev_class]} in Low Energy State. Duration remaining: {duration:.0f} mins. Action: PoE power cut."
                else:
                    if st.session_state.port_state == "SHUTDOWN":
                        st.session_state.port_state = "ACTIVE"
                        st.session_state.wake_events.append(current_row['timestamp'])
                        reasoning_msg = f"🔌 **PoE Status ({target_id} on {target_switch_id})**: Port Turned On! Recommended: Restore PoE power to {dev_map[dev_class]}. ({features_str})"
                    else:
                        reasoning_msg = f"🔌 **PoE Status ({target_id} on {target_switch_id})**: Active. Device: {dev_map[dev_class]}, Draw: {poe_watts:.1f}W. No action needed. ({features_str})"
 
            elif st.session_state.current_scenario.startswith("3"):
                # LAG — model expects 8 features including rolling util stats
                x_lag = current_row[['aggregate_utilization', 'member_imbalance_ratio', 'total_members', 'hour', 'is_weekend', 'rolling_mean_util_1h', 'rolling_std_util_1h', 'rolling_slope_util_1h']].to_frame().T.astype(float)
                lag_shut_prob = models['lag_shutdown_classifier'].predict_proba(x_lag)[0][1]
                lag_shut = lag_shut_prob >= 0.5
                duration = 0
                if lag_shut:
                    dur_log = np.clip(models['lag_duration_regressor'].predict(x_lag)[0], 0, MAX_LOG_DURATION)
                    duration = np.expm1(dur_log) * 60
                    
                features_str = "Features: aggregate_utilization, member_imbalance_ratio, total_members, hour, is_weekend, rolling_mean_util_1h, rolling_std_util_1h, rolling_slope_util_1h"
                total_members = int(current_row['total_members'])
                
                if lag_shut and duration >= 60.0:
                    if st.session_state.port_state == "ACTIVE":
                        st.session_state.port_state = "SHUTDOWN"
                        reason = models['lag_reason_classifier'].predict(x_lag)[0]
                        r_map = {0: 'Nightly', 1: 'Weekend', 2: 'Other'}
                        removable = max(1, total_members - 1)
                        reasoning_msg = f"🔗 **LAG Optimizer ({target_id} on {target_switch_id})**: Member Shutdown Triggered! Duration: {duration:.0f} mins ({r_map[reason]}). Recommended: Remove {removable} of {total_members} members. ({features_str})"
                    else:
                        reasoning_msg = f"💤 **LAG Optimizer ({target_id} on {target_switch_id})**: Members in Low Energy State. Duration remaining: {duration:.0f} mins. Action: {max(1, total_members - 1)} members removed."
                else:
                    if st.session_state.port_state == "SHUTDOWN":
                        st.session_state.port_state = "ACTIVE"
                        st.session_state.wake_events.append(current_row['timestamp'])
                        reasoning_msg = f"✅ **LAG Optimizer ({target_id} on {target_switch_id})**: Activity Detected — Members Turned On! Recommended: Restore all {total_members} LAG members. ({features_str})"
                    else:
                        reasoning_msg = f"✅ **LAG Optimizer ({target_id} on {target_switch_id})**: All {total_members} members active. Util: {current_row['aggregate_utilization']:.1f}%. No action needed. ({features_str})"
 
            elif st.session_state.current_scenario.startswith("4"):
                # Storm
                x_storm = current_row[['rx_utilization', 'tx_utilization', 'queue_drops', 'queue_depth', 'flow_count']].to_frame().T.astype(float)
                storm_prob = models['storm_warning_classifier'].predict_proba(x_storm)[0][1]
                features_str = "Features: rx_utilization, tx_utilization, queue_drops, queue_depth, flow_count"
                if storm_prob > 0.5:
                    reasoning_msg = f"🌪️ **STORM WARNING ({target_id} on {target_switch_id})**: High probability ({storm_prob*100:.1f}%) of broadcast storm in next 30 mins! Queue depth: {current_row['queue_depth']}, drops: {current_row['queue_drops']}. Recommended: Enable storm-control on port. ({features_str})"
                else:
                    reasoning_msg = f"🌤️ **Storm Monitor ({target_id} on {target_switch_id})**: Normal conditions. Storm probability: {storm_prob*100:.1f}%. No action needed. ({features_str})"
 
            elif st.session_state.current_scenario.startswith("5"):
                # Long Term
                x_lt = current_row[['ema_7d', 'ema_30d', 'ema_7d_slope', 'rolling_mean_1h', 'rolling_std_1h', 'rolling_slope_1h', 'sin_day', 'cos_day', 'hour', 'sin_hour', 'cos_hour', 'day_of_week', 'sin_dow', 'cos_dow', 'is_weekend', 'idle_duration_minutes', 'consecutive_idle_windows', 'time_since_last_burst_minutes']].to_frame().T.astype(float)
                dur_log = np.clip(models['long_term_duration_regressor'].predict(x_lt)[0], 0, np.log1p(720.0))  # cap 30 days
                dur_mins = np.expm1(dur_log) * 60
                features_str = "Features: ema_7d, ema_30d, ema_7d_slope, rolling_mean_1h, rolling_std_1h, rolling_slope_1h, sin_day, cos_day, hour, sin_hour, cos_hour, day_of_week, sin_dow, cos_dow, is_weekend, idle_duration_minutes, consecutive_idle_windows, time_since_last_burst_minutes"
                if dur_mins > 1440:
                    reasoning_msg = f"📅 **Macro-Trend Predictor ({target_id} on {target_switch_id})**: Predicted massive long-term idle break! Duration: {dur_mins:.0f} mins ({dur_mins/1440:.1f} days). Recommended: Schedule port for decommission during break. ({features_str})"
                else:
                    reasoning_msg = f"📊 **Macro-Trend Predictor ({target_id} on {target_switch_id})**: Normal operating conditions. Predicted long-term break: {dur_mins:.0f} mins. No action needed. ({features_str})"

        except Exception as e:
            reasoning_msg = f"Model Error: {e}"

        if reasoning_msg:
            prefix = reasoning_msg.split(":")[0]
            if len(st.session_state.reasoning) == 0:
                st.session_state.reasoning.insert(0, reasoning_msg)
            elif st.session_state.reasoning[0].startswith(prefix):
                st.session_state.reasoning[0] = reasoning_msg
            else:
                st.session_state.reasoning.insert(0, reasoning_msg)
                if len(st.session_state.reasoning) > 5:
                    st.session_state.reasoning.pop()

        # UI Blocks
        col_left, col_right = st.columns([1, 1.5])

        with col_left:
            st.subheader("🌐 Campus Network Topology")
            G = st.session_state.topology.graph
            pos = st.session_state.pos

            # Identify target switch (already computed as target_switch_id)
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

            node_x = []
            node_y = []
            node_color = []
            node_size = []
            node_text = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)

                # Color by layer
                if "CORE" in node: color = '#ff0000' # Red Core
                elif "DIST" in node: color = '#ffaa00' # Orange Dist
                else: color = '#00aaff' # Blue Access

                size = 10
                if node == target_switch_id:
                    color = '#ffffff'
                    size = 20
                    # Pulse if storm
                    if "STORM" in str(reasoning_msg).upper():
                        color = '#ff00ff'
                        size = 30

                node_color.append(color)
                node_size.append(size)

            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                                    marker=dict(showscale=False, color=node_color, size=node_size, line_width=2))

            fig_net = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(showlegend=False, hovermode='closest',
                                                 margin=dict(b=0,l=0,r=0,t=0),
                                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                 template="plotly_dark", height=400))
            st.plotly_chart(fig_net, use_container_width=True)

        with col_right:
            st.subheader("Live Telemetry Stream")
            # Show last 60 minutes (12 intervals) up to the current index
            hist = df.iloc[max(0, idx-12):idx+1]

            # Fixed axes ranges for smooth animation
            t_min = current_row['timestamp'] - pd.Timedelta(minutes=60)
            t_max = current_row['timestamp'] + pd.Timedelta(minutes=15)

            if not is_lag:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['total_utilization'], mode='lines+markers', name='Total Util %', line=dict(color='#00ff00', width=3)))
                
                # Add wake events
                wake_pts = [t for t in st.session_state.wake_events if t_min <= t <= t_max]
                if wake_pts:
                    wake_y = [hist[hist['timestamp'] == t]['total_utilization'].values[0] if not hist[hist['timestamp'] == t].empty else 0 for t in wake_pts]
                    fig.add_trace(go.Scatter(x=wake_pts, y=wake_y, mode='markers', name='Port Turned On', marker=dict(color='yellow', size=15, symbol='triangle-up', line=dict(color='black', width=2))))
                
                if st.session_state.current_scenario.startswith("4"):
                     fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['queue_depth'], mode='lines+markers', name='Queue Depth', line=dict(color='#ff0000', width=2)))
                     y_max = max(100, hist['queue_depth'].max() + 20 if not hist.empty else 100)
                else:
                     y_max = max(1.0, hist['total_utilization'].max() * 1.5 if not hist.empty else 1.0)

                fig.update_layout(height=400, template='plotly_dark', margin=dict(l=0,r=0,t=30,b=0),
                                  xaxis_title="Simulation Time", yaxis_title="Metric Value",
                                  xaxis=dict(range=[t_min, t_max], fixedrange=True),
                                  yaxis=dict(range=[-(y_max*0.05), y_max], fixedrange=True))
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['aggregate_utilization'], mode='lines+markers', name='Aggregate Util %', line=dict(color='#00ffff', width=3)))
                
                # Add wake events
                wake_pts = [t for t in st.session_state.wake_events if t_min <= t <= t_max]
                if wake_pts:
                    wake_y = [hist[hist['timestamp'] == t]['aggregate_utilization'].values[0] if not hist[hist['timestamp'] == t].empty else 0 for t in wake_pts]
                    fig.add_trace(go.Scatter(x=wake_pts, y=wake_y, mode='markers', name='Members Turned On', marker=dict(color='yellow', size=15, symbol='triangle-up', line=dict(color='black', width=2))))
                
                y_max = max(10.0, hist['aggregate_utilization'].max() * 1.5 if not hist.empty else 10.0)
                fig.update_layout(height=400, template='plotly_dark', margin=dict(l=0,r=0,t=30,b=0),
                                  xaxis_title="Simulation Time", yaxis_title="Utilization (%)",
                                  xaxis=dict(range=[t_min, t_max], fixedrange=True),
                                  yaxis=dict(range=[-(y_max*0.05), y_max], fixedrange=True))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🤖 Reasoning Engine")
        for r in st.session_state.reasoning:
            st.info(r)

        # Playback Logic
        if st.session_state.is_playing:
            if idx + 1 < len(df):
                st.session_state.current_index += 1
                time.sleep(1.0 / st.session_state.speed)
                st.rerun()
            else:
                st.session_state.is_playing = False
                st.success("End of telemetry stream.")

with tab_global:
    st.header("Global Campus View")
    try:
        ports_df, devices_df = load_global_data()
    except Exception as e:
        st.error(f"Error loading global data: {e}")
        st.stop()
        
    global_times = sorted(ports_df['timestamp'].unique())
    if st.session_state.global_index >= len(global_times):
        st.session_state.global_index = len(global_times) - 1
        
    # Global Controls
    gc1, gc2, gc3 = st.columns([1, 1, 4])
    if gc1.button("▶️ Play", key="gplay"): st.session_state.global_is_playing = True
    if gc2.button("⏸️ Pause", key="gpause"): st.session_state.global_is_playing = False
    
    st.session_state.global_index = st.slider(
        "Timeline Slider (Index)", 
        0, len(global_times) - 1, 
        st.session_state.global_index,
        key="gslider"
    )
    
    current_time = global_times[st.session_state.global_index]
    st.markdown(f"### 🕒 Global Time: `{current_time}`")
    
    gt1, gt2, gt3 = st.tabs(["Network Map", "Device Telemetry", "Port Inference"])
    
    with gt1:
        st.subheader("Campus Network Topology")
        G = st.session_state.topology.graph
        pos = st.session_state.pos
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
        node_x, node_y, node_text, node_color, node_symbol = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            if "Core" in node or "CORE" in node:
                node_color.append("red")
                node_text.append(f"🏢 {node}")
                node_symbol.append("square")
            elif "Dist" in node or "DIST" in node:
                node_color.append("orange")
                node_text.append(f"🔀 {node}")
                node_symbol.append("diamond")
            else:
                node_color.append("lightblue")
                node_text.append(f"🖥️ {node}")
                node_symbol.append("circle")
                
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="bottom center", hoverinfo='text',
            marker=dict(showscale=False, color=node_color, size=14, symbol=node_symbol, line_width=1, line_color="black"))
                
        fig_map = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=600))
        st.plotly_chart(fig_map, use_container_width=True)
        
    with gt2:
        st.subheader("Device Telemetry")
        switch_list = sorted(devices_df['switch_id'].unique())
        selected_switch = st.selectbox("Select Switch", switch_list, key="global_switch")
        
        # FILTER BY CURRENT TIME
        switch_data = devices_df[(devices_df['switch_id'] == selected_switch) & (devices_df['timestamp'] <= current_time)].sort_values('timestamp')
        
        if not switch_data.empty:
            fig_cpu = px.line(switch_data, x='timestamp', y=['cpu_utilization', 'memory_utilization'], title=f"CPU/Memory for {selected_switch}")
            st.plotly_chart(fig_cpu, use_container_width=True)
            
            fig_poe = px.line(switch_data, x='timestamp', y='poe_usage_watts', title=f"PoE Draw for {selected_switch}")
            st.plotly_chart(fig_poe, use_container_width=True)
        else:
            st.info("No data available yet.")
        
    with gt3:
        st.subheader("Port Inference & Telemetry")
        st.markdown(f"Showing ports for **{selected_switch}**")
        switch_ports = ports_df[ports_df['switch_id'] == selected_switch]['port_id'].unique()
        
        if len(switch_ports) > 0:
            selected_port = st.selectbox("Select Port", sorted(switch_ports), key="global_port")
            # FILTER BY CURRENT TIME
            port_data = ports_df[(ports_df['port_id'] == selected_port) & (ports_df['timestamp'] <= current_time)].sort_values('timestamp').copy()
            
            if not port_data.empty:
                latest_row = port_data.iloc[-1]
                
                try:
                    MAX_LOG_DUR = np.log1p(72.0)
                    
                    # 1. Port Idle — compute rolling features from port_data history
                    port_data['rolling_mean_1h'] = port_data['total_utilization'].rolling(12, min_periods=1).mean()
                    port_data['rolling_std_1h'] = port_data['total_utilization'].rolling(12, min_periods=1).std().fillna(0)
                    port_data['rolling_slope_1h'] = (port_data['total_utilization'] - port_data['total_utilization'].shift(12)).fillna(0)
                    port_data['rolling_mean_rx_1h'] = port_data['rx_utilization'].rolling(12, min_periods=1).mean()
                    port_data['rolling_std_rx_1h'] = port_data['rx_utilization'].rolling(12, min_periods=1).std().fillna(0)
                    port_data['rolling_slope_rx_1h'] = (port_data['rx_utilization'] - port_data['rx_utilization'].shift(12)).fillna(0)
                    port_data['rolling_mean_tx_1h'] = port_data['tx_utilization'].rolling(12, min_periods=1).mean()
                    port_data['rolling_std_tx_1h'] = port_data['tx_utilization'].rolling(12, min_periods=1).std().fillna(0)
                    port_data['rolling_slope_tx_1h'] = (port_data['tx_utilization'] - port_data['tx_utilization'].shift(12)).fillna(0)
                    latest_row = port_data.iloc[-1]
                    
                    x_idle = latest_row[['total_utilization', 'hour', 'minute', 'day_of_week', 'is_weekend', 'consecutive_idle_windows', 'is_burst', 'rolling_mean_1h', 'rolling_std_1h', 'rolling_slope_1h']].to_frame().T.astype(float)
                    is_idle_prob = models['port_idle_classifier'].predict_proba(x_idle)[0][1]
                    is_idle = is_idle_prob >= 0.5
                    if is_idle:
                        dur_idle_log = np.clip(models['port_idle_duration_regressor'].predict(x_idle)[0], 0, MAX_LOG_DUR)
                        dur_idle_mins = np.expm1(dur_idle_log) * 60
                        if dur_idle_mins >= 60:
                            st.info(f"🔴 **Port Idle Predictor**: Idle predicted (Confidence: {is_idle_prob*100:.1f}%). Duration: {dur_idle_mins:.0f} mins. Recommended: Disable port.")
                        else:
                            st.success(f"🔵 **Port Idle Predictor**: Short idle detected ({dur_idle_mins:.0f} mins). No action needed.")
                    else:
                        st.success(f"🔵 **Port Idle Predictor**: Port is active (Idle Prob: {is_idle_prob*100:.1f}%). No action needed.")
                    
                    # 2. PoE
                    x_poe_class = latest_row[['rx_utilization', 'tx_utilization', 'packet_rate_pps', 'active_ip_count']].to_frame().T.astype(float)
                    dev_class = models['poe_device_fingerprint'].predict(x_poe_class)[0]
                    dev_map = {0: 'Access Point', 1: 'IP Phone', 2: 'IP Camera'}
                    x_poe = latest_row[['rx_utilization', 'tx_utilization', 'hour', 'day_of_week', 'rolling_mean_rx_1h', 'rolling_std_rx_1h', 'rolling_slope_rx_1h', 'rolling_mean_tx_1h', 'rolling_std_tx_1h', 'rolling_slope_tx_1h']].to_frame().T.astype(float)
                    poe_shut_prob = models['poe_shutdown_classifier'].predict_proba(x_poe)[0][1]
                    poe_shut = poe_shut_prob >= 0.5
                    if poe_shut:
                        dur_poe_log = np.clip(models['poe_duration_regressor'].predict(x_poe)[0], 0, MAX_LOG_DUR)
                        dur_poe_mins = np.expm1(dur_poe_log) * 60
                        if dur_poe_mins >= 60:
                            st.warning(f"⚡ **PoE Power Down**: Shutdown recommended for **{dev_map.get(dev_class, 'Unknown')}** (Confidence: {poe_shut_prob*100:.1f}%). Duration: {dur_poe_mins:.0f} mins. Recommended: Cut PoE power, save {latest_row['poe_consumption_watts']:.1f}W.")
                        else:
                            st.success(f"🔌 **PoE Status**: Short idle ({dur_poe_mins:.0f} mins). Device: **{dev_map.get(dev_class, 'Unknown')}**. No action needed.")
                    else:
                        st.success(f"🔌 **PoE Status**: Active (Shutdown Prob: {poe_shut_prob*100:.1f}%). Device: **{dev_map.get(dev_class, 'Unknown')}**. No action needed.")
                    
                    # 3. Storm
                    x_storm = latest_row[['rx_utilization', 'tx_utilization', 'queue_drops', 'queue_depth', 'flow_count']].to_frame().T.astype(float)
                    storm_prob = models['storm_warning_classifier'].predict_proba(x_storm)[0][1]
                    if storm_prob > 0.5:
                        st.error(f"🌪️ **STORM WARNING**: High probability ({storm_prob*100:.1f}%) of broadcast storm! Queue depth: {latest_row['queue_depth']}. Recommended: Enable storm-control.")
                    else:
                        st.success(f"🌤️ **Storm Monitor**: Normal conditions. Storm probability: {storm_prob*100:.1f}%. No action needed.")
                        
                except Exception as e:
                    st.error(f"Error running inference models: {e}")
                
                st.divider()
                
                # Show trailing window of 60 mins for smooth graph
                t_min = current_time - pd.Timedelta(minutes=60)
                t_max = current_time + pd.Timedelta(minutes=15)
                
                fig_traf = go.Figure()
                fig_traf.add_trace(go.Scatter(x=port_data['timestamp'], y=port_data['rx_utilization'], mode='lines', name='RX Util'))
                fig_traf.add_trace(go.Scatter(x=port_data['timestamp'], y=port_data['tx_utilization'], mode='lines', name='TX Util'))
                fig_traf.update_layout(height=300, template='plotly_dark', margin=dict(l=0,r=0,t=30,b=0), title=f"Traffic Utilization for {selected_port}", xaxis=dict(range=[t_min, t_max], fixedrange=True))
                st.plotly_chart(fig_traf, use_container_width=True)
                
                fig_drops = go.Figure()
                fig_drops.add_trace(go.Scatter(x=port_data['timestamp'], y=port_data['queue_drops'], mode='lines', name='Drops'))
                fig_drops.update_layout(height=300, template='plotly_dark', margin=dict(l=0,r=0,t=30,b=0), title=f"Queue Drops for {selected_port}", xaxis=dict(range=[t_min, t_max], fixedrange=True))
                st.plotly_chart(fig_drops, use_container_width=True)
                
            else:
                st.info("No port data available yet.")
        else:
            st.warning(f"No ports found for {selected_switch}.")
            
    # Playback loop for Global View
    if st.session_state.global_is_playing:
        if st.session_state.global_index + 1 < len(global_times):
            st.session_state.global_index += 1
            time.sleep(1.0 / st.session_state.speed)
            st.rerun()
        else:
            st.session_state.global_is_playing = False
            st.success("End of global telemetry stream.")
