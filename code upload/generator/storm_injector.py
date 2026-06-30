import random
from typing import List, Dict

class StormState:
    def __init__(self, start_timestamp, storm_type: str, port_ids: List[str], durations: dict):
        self.start_timestamp = start_timestamp
        self.storm_type = storm_type
        self.port_ids = port_ids
        
        self.precursor_mins = random.uniform(*durations['precursor_duration_minutes'])
        self.onset_mins = random.uniform(*durations['onset_duration_minutes'])
        self.peak_mins = random.uniform(*durations['peak_duration_minutes'])
        self.resolution_mins = random.uniform(*durations['resolution_duration_minutes'])
        
        self.phase = None
        self.elapsed_mins = 0.0

    def update(self, current_timestamp) -> str:
        delta = current_timestamp - self.start_timestamp
        self.elapsed_mins = delta.total_seconds() / 60.0
        
        if self.elapsed_mins < 0:
            self.phase = None
        elif self.elapsed_mins < self.precursor_mins:
            self.phase = "precursor"
        elif self.elapsed_mins < self.precursor_mins + self.onset_mins:
            self.phase = "onset"
        elif self.elapsed_mins < self.precursor_mins + self.onset_mins + self.peak_mins:
            self.phase = "peak"
        elif self.elapsed_mins < self.precursor_mins + self.onset_mins + self.peak_mins + self.resolution_mins:
            self.phase = "recovery"
        else:
            self.phase = "finished"
            
        return self.phase


def apply_storm_effects(port_row: dict, storm_state: StormState):
    if storm_state is None or storm_state.phase is None or storm_state.phase == "finished":
        port_row["is_storm_affected"] = False
        port_row["storm_type"] = "none"
        port_row["storm_phase"] = "none"
        return
        
    if port_row["port_id"] not in storm_state.port_ids:
        # Check if same switch for peak spreading
        if storm_state.phase == "peak" and random.random() < 0.1:
             pass
        else:
            port_row["is_storm_affected"] = False
            port_row["storm_type"] = "none"
            port_row["storm_phase"] = "none"
            return
            
    port_row["is_storm_affected"] = True
    port_row["storm_type"] = storm_state.storm_type
    port_row["storm_phase"] = storm_state.phase
    
    phase = storm_state.phase
    
    if phase == "precursor":
        port_row["rx_bytes"] *= random.uniform(1.2, 1.4)
        port_row["rx_utilization"] = (port_row["rx_bytes"] * 8 / (port_row["link_speed_mbps"] * 1e6 * 300)) * 100
        port_row["total_utilization"] = (port_row["rx_utilization"] + port_row["tx_utilization"]) / 2
        port_row["queue_depth"] = min(255, port_row["queue_depth"] + random.randint(10, 30))
        
    elif phase == "onset":
        port_row["rx_utilization"] = random.uniform(80, 100)
        port_row["rx_bytes"] = (port_row["rx_utilization"] / 100.0) * (port_row["link_speed_mbps"] * 1e6 / 8) * 300
        port_row["packet_rate_pps"] *= random.uniform(5, 15)
        port_row["queue_drops"] += random.randint(500, 2000)
        port_row["flow_count"] = int(port_row["flow_count"] * random.uniform(0.1, 0.4))
        port_row["active_mac_count"] = max(0, int(port_row["active_mac_count"] + random.randint(-50, 50)))
        port_row["total_utilization"] = (port_row["rx_utilization"] + port_row["tx_utilization"]) / 2
        
    elif phase == "peak":
        port_row["rx_utilization"] = random.uniform(90, 100)
        port_row["rx_bytes"] = (port_row["rx_utilization"] / 100.0) * (port_row["link_speed_mbps"] * 1e6 / 8) * 300
        port_row["packet_rate_pps"] *= random.uniform(10, 20)
        port_row["queue_drops"] += random.randint(2000, 5000)
        port_row["flow_count"] = int(port_row["flow_count"] * random.uniform(0.1, 0.3))
        port_row["total_utilization"] = (port_row["rx_utilization"] + port_row["tx_utilization"]) / 2
        
    elif phase == "recovery":
        port_row["rx_bytes"] *= 0.1
        port_row["tx_bytes"] *= 0.1
        port_row["rx_utilization"] *= 0.1
        port_row["tx_utilization"] *= 0.1
        port_row["total_utilization"] *= 0.1
        port_row["queue_depth"] = 0
        port_row["queue_drops"] = 0
