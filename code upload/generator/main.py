import yaml
import random
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

# Add the project root to the python path so it can be run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.topology import CampusTopology
from generator.temporal_engine import TimeContext
from generator.port_generator import PortState, generate_port_row
from generator.device_generator import generate_device_row
from generator.lag_generator import generate_lag_row
from generator.stack_generator import generate_stack_row
from generator.storm_injector import StormState, apply_storm_effects
from generator.correlation_engine import enforce_correlations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_DIR = Path(__file__).parent.parent / 'config'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'

def load_thresholds():
    with open(CONFIG_DIR / 'thresholds.yaml', 'r') as f:
        return yaml.safe_load(f)

def generate_telemetry(start_date: str, end_date: str, interval_mins: int = 5, sleep_seconds: float = 0.0):
    import time
    logging.info(f"Initializing topology...")
    topology = CampusTopology()
    thresholds = load_thresholds()
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    
    # Initialize state
    port_states = {p_id: PortState(p_id) for p_id in topology.ports}
    
    # Schedule some storms
    storms = []
    # 1 storm per week roughly
    storm_start = start_dt + timedelta(days=2, hours=14, minutes=15)
    if storm_start < end_dt:
        storm_port_ids = [p_id for p_id, p in topology.ports.items() if p.role == "access"]
        storms.append(StormState(
            start_timestamp=storm_start,
            storm_type="broadcast",
            port_ids=random.sample(storm_port_ids, k=min(3, len(storm_port_ids))),
            durations=thresholds['storm']
        ))

    current_dt = start_dt
    
    port_data = []
    device_data = []
    lag_data = []
    stack_data = []
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    logging.info(f"Starting generation loop from {start_dt} to {end_dt}")
    
    batch_count = 0
    total_intervals = int((end_dt - start_dt).total_seconds() / (interval_mins * 60))
    current_interval = 0
    
    while current_dt <= end_dt:
        if current_interval % 288 == 0:  # Daily log (12 * 24)
            logging.info(f"Processing day {current_dt.strftime('%Y-%m-%d')}")
            
        context = TimeContext(current_dt)
        
        # Update storms
        active_storms = []
        for s in storms:
            phase = s.update(current_dt)
            if phase is not None and phase != "finished":
                active_storms.append(s)
        
        # 1. Generate port rows
        port_rows_by_switch = {sw_id: [] for sw_id in topology.switches}
        for port_id, port in topology.ports.items():
            state = port_states[port_id]
            row = generate_port_row(port, current_dt, context, state)
            
            # Apply storms
            for s in active_storms:
                apply_storm_effects(row, s)
                
            port_rows_by_switch[port.switch_id].append(row)
        
        # 2. Generate device rows
        device_rows_by_id = {}
        for sw_id, switch in topology.switches.items():
            row = generate_device_row(switch, port_rows_by_switch[sw_id], current_dt)
            device_rows_by_id[sw_id] = row
            
        # 3. Enforce correlations
        enforce_correlations(port_rows_by_switch, device_rows_by_id, {}, {})
        
        # Collect rows
        for p_rows in port_rows_by_switch.values():
            port_data.extend(p_rows)
            
        for d_row in device_rows_by_id.values():
            device_data.append(d_row)
            
        # 4. Generate LAG rows
        for lag_id, lag in topology.lags.items():
            # Estimate aggregate traffic from members
            agg_traffic = 0
            for m in lag.members:
                 # Find the port
                 for r in port_rows_by_switch[lag.switch_id]:
                     if r["port_id"] == m.port_id:
                         agg_traffic += r["tx_bytes"] * 8 / 300 # simple approximation
                         
            l_row = generate_lag_row(lag, context, agg_traffic)
            if l_row:
                lag_data.append(l_row)
                
        # 5. Generate Stack rows
        for stack_id, stack in topology.stacks.items():
            s_row = generate_stack_row(stack, device_rows_by_id, context)
            stack_data.append(s_row)
            
        # Advance time
        current_dt += timedelta(minutes=interval_mins)
        current_interval += 1
        
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        
        # Batch writing every week (288 * 7 intervals)
        if current_interval % 2016 == 0 or current_dt > end_dt:
            logging.info(f"Writing batch up to {current_dt}")
            pd.DataFrame(port_data).to_parquet(DATA_DIR / f'ports_batch_{batch_count}.parquet', index=False)
            pd.DataFrame(device_data).to_parquet(DATA_DIR / f'devices_batch_{batch_count}.parquet', index=False)
            pd.DataFrame(lag_data).to_parquet(DATA_DIR / f'lags_batch_{batch_count}.parquet', index=False)
            pd.DataFrame(stack_data).to_parquet(DATA_DIR / f'stacks_batch_{batch_count}.parquet', index=False)
            
            port_data = []
            device_data = []
            lag_data = []
            stack_data = []
            batch_count += 1

    logging.info("Generation complete.")

if __name__ == "__main__":
    # Generate 6 months of data for model training
    generate_telemetry("2024-01-01 00:00:00", "2024-06-30 23:55:00", interval_mins=5, sleep_seconds=0.0) 
