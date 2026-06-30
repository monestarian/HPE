import math
import random
from datetime import datetime
from generator.temporal_engine import TimeContext, get_occupancy
from generator.endpoint_profiles import ENDPOINT_PROFILES
from generator.topology import Port

class PortState:
    def __init__(self, port_id: str):
        self.port_id = port_id
        self.last_active_timestamp = None
        self.idle_duration_minutes = 0.0
        self.consecutive_idle_windows = 0
        self.last_burst_timestamp = None
        self.time_since_last_burst_minutes = 9999.0
        self.utilization_buffer = []  # rolling last 12 readings (60 min)
        
    def update(self, timestamp: datetime, total_util: float, is_burst: bool, interval_mins: int = 5):
        self.utilization_buffer.append(total_util)
        if len(self.utilization_buffer) > 12:
            self.utilization_buffer.pop(0)
            
        if total_util > 1.0:
            self.last_active_timestamp = timestamp
            self.idle_duration_minutes = 0.0
            self.consecutive_idle_windows = 0
        else:
            self.idle_duration_minutes += interval_mins
            self.consecutive_idle_windows += 1
            
        if is_burst:
            self.last_burst_timestamp = timestamp
            self.time_since_last_burst_minutes = 0.0
        else:
            self.time_since_last_burst_minutes += interval_mins


def generate_port_row(port: Port, timestamp: datetime, context: TimeContext, port_state: PortState) -> dict:
    # Step 1: Get occupancy fraction
    occupancy = get_occupancy(port.zone, context.hour, context.minute, context)
    
    # Step 2: Apply endpoint profile base rates
    profile = ENDPOINT_PROFILES.get(port.endpoint_type, ENDPOINT_PROFILES["student_pc"])
    
    # Check if uplink always active
    if profile.get("always_active", False):
        occupancy = 1.0
    
    # Step 3: Check if a burst event is occurring
    is_burst = random.random() < profile.get("burst_probability", 0.0)
    burst_mult = profile.get("burst_multiplier", 1.0) if is_burst else 1.0
    
    # Inject precursor patterns for idle duration prediction
    if port.role == "access":
        # 1. Overnight precursor (17:00-17:30 on weekdays): Spike before going idle for ~14 hours
        if not context.is_weekend and context.hour == 17 and context.minute < 30:
            is_burst = True
            burst_mult = profile.get("burst_multiplier", 4.0) * 2.0
        
        # 2. Weekend taper (Friday 14:00-18:00): Slow taper before going idle for ~60 hours
        elif context.weekday == 4 and 14 <= context.hour < 18:
            occupancy = occupancy * max(0.1, 1.0 - ((context.hour - 14) * 60 + context.minute) / 240.0)
            is_burst = False
            burst_mult = 1.0
            
        # 3. Between-class drop: If it's x:50 to x:59, sharp drop (idle for 1-2 hours)
        elif not context.is_weekend and 8 <= context.hour < 16 and context.minute >= 50:
            occupancy *= 0.2
            is_burst = False
            burst_mult = 1.0
    
    # Step 4: Compute raw traffic
    rx_bps = profile.get("base_rx_kbps", 0) * 1000 * occupancy * burst_mult
    tx_bps = profile.get("base_tx_kbps", 0) * 1000 * occupancy * burst_mult
    
    # Step 5: Add realistic gaussian noise (±15%)
    rx_bps *= random.gauss(1.0, 0.15)
    tx_bps *= random.gauss(1.0, 0.15)
    rx_bps = max(0, rx_bps)
    tx_bps = max(0, tx_bps)
    
    # Step 6: Convert to bytes per 5-min interval
    interval_seconds = 300
    rx_bytes = rx_bps / 8 * interval_seconds
    tx_bytes = tx_bps / 8 * interval_seconds
    
    # Step 7: Derive utilization
    link_speed_bps = port.link_speed * 1e6  # port.link_speed in Mbps
    rx_util = (rx_bps / link_speed_bps) * 100
    tx_util = (tx_bps / link_speed_bps) * 100
    total_util = ((rx_bps + tx_bps) / (2 * link_speed_bps)) * 100
    
    # Step 8: Active MAC/IP count
    if port.endpoint_type == "access_point":
        mac_range = profile.get("mac_count_range", (5, 60))
        mac_count = int(mac_range[0] + occupancy * (mac_range[1] - mac_range[0]))
        mac_count = max(0, int(mac_count * random.gauss(1.0, 0.1)))
    elif profile.get("always_active", False):
        mac_range = profile.get("mac_count_range", (100, 3000))
        mac_count = int(mac_range[0] + occupancy * (mac_range[1] - mac_range[0]))
        mac_count = max(0, int(mac_count * random.gauss(1.0, 0.1)))
    elif occupancy > 0:
        mac_count = profile.get("mac_count", 1)
    else:
        mac_count = 0
    
    ip_count = int(mac_count * random.uniform(0.85, 1.0))  # IP <= MAC
    
    # Step 9: Flow count (proportional to active IPs and traffic volume)
    flow_count = int(ip_count * random.gauss(12, 3) * (1 + total_util/100))
    flow_count = max(0, flow_count)
    
    # Step 10: Packet rate
    avg_packet_size_bytes = random.gauss(800, 200)
    packet_rate = (rx_bytes + tx_bytes) / avg_packet_size_bytes / interval_seconds
    if math.isnan(packet_rate) or packet_rate < 0:
        packet_rate = 0.0
        
    # Step 11: Queue depth and drops
    # Queue depth rises non-linearly above 70% utilization
    if total_util < 70:
        queue_depth = int(total_util * 0.5 * random.gauss(1.0, 0.2))
    else:
        queue_depth = int(50 + (total_util - 70) * 8 * random.gauss(1.0, 0.3))
    queue_depth = min(queue_depth, 255)
    queue_depth = max(0, queue_depth)
    
    queue_drops = 0
    if queue_depth > 200 or total_util > 90:
        queue_drops = int(packet_rate * 0.01 * random.gauss(1.0, 0.5))
        queue_drops = max(0, queue_drops)
    
    # Step 12: PoE consumption (0 if port is down or endpoint doesn't use PoE)
    poe_watts = profile.get("poe_watts", 0) if occupancy > 0 else 0
    poe_watts *= random.gauss(1.0, 0.05)  # Slight PoE draw variation
    poe_watts = max(0.0, poe_watts)
    
    # Step 13: Port state flags
    is_active = occupancy > 0.01 or port.endpoint_type in ["always_on", "server", "ip_camera", "uplink"]
    operational_status = "up" if is_active else "down"
    
    # Step 14: Update PortState
    port_state.update(timestamp, total_util, is_burst, interval_mins=5)
    
    return {
        "timestamp": timestamp,
        "port_id": port.id,
        "switch_id": port.switch_id,
        "zone": port.zone,
        "building": port.building,
        "floor": port.floor,
        "endpoint_type": port.endpoint_type,
        "link_speed_mbps": port.link_speed,
        "duplex": "full",
        "mtu": 1500,
        "port_role": port.role,  # access / trunk / uplink
        "vlan": port.vlan,
        "access_trunk_mode": "access" if port.role == "access" else "trunk",
        "admin_status": "up",          # assume all ports are admin up
        "operational_status": operational_status,
        "rx_bytes": rx_bytes,
        "tx_bytes": tx_bytes,
        "rx_utilization": rx_util,
        "tx_utilization": tx_util,
        "total_utilization": total_util,
        "input_rate_bps": rx_bps,
        "output_rate_bps": tx_bps,
        "packet_rate_pps": packet_rate,
        "queue_depth": queue_depth,
        "queue_drops": queue_drops,
        "flow_count": flow_count,
        "active_mac_count": mac_count,
        "active_ip_count": ip_count,
        "poe_consumption_watts": poe_watts,
        "is_burst": is_burst,
        "occupancy": occupancy,          # Keep for label generation; drop before model training
        # Time features
        "hour": context.hour,
        "minute": context.minute,
        "day_of_week": context.weekday,  # 0=Monday
        "is_weekend": context.is_weekend,
        "is_holiday": context.is_holiday,
        "semester_state": context.semester_state,  # orientation/regular/exam_prep/exam
        "is_class_hours": 8 <= context.hour < 18 and not context.is_weekend,
        # Internal state tracking for features
        "idle_duration_minutes": port_state.idle_duration_minutes,
        "consecutive_idle_windows": port_state.consecutive_idle_windows,
        "time_since_last_burst_minutes": port_state.time_since_last_burst_minutes,
    }


