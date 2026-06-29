import math
import random
from generator.temporal_engine import TimeContext
from generator.topology import LAG

def generate_lag_row(lag: LAG, context: TimeContext, aggregate_traffic_bps: float) -> dict:
    active_members = [m for m in lag.members if m.state == "active"]
    standby_members = [m for m in lag.members if m.state == "standby"]
    
    if not active_members:
        return {} # Should not happen in healthy lag
        
    # Distribute traffic across active members with imbalance
    per_member_traffic = []
    for member in active_members:
        share = aggregate_traffic_bps / len(active_members)
        share *= random.gauss(1.0, 0.10)   # ±10% imbalance per member
        per_member_traffic.append(max(0.0, share))
    
    # Normalize to actual aggregate
    actual_aggregate = sum(per_member_traffic)
    if actual_aggregate > 0:
        factor = aggregate_traffic_bps / actual_aggregate
        per_member_traffic = [t * factor for t in per_member_traffic]
        
    actual_aggregate = sum(per_member_traffic)
    
    # Per-member utilization
    member_utils = [
        (t / (m.link_speed_bps)) * 100
        for t, m in zip(per_member_traffic, active_members)
    ]
    
    member_imbalance = (
        (max(member_utils) - min(member_utils)) / (sum(member_utils) / len(member_utils))
        if sum(member_utils) > 0 else 0.0
    )
    
    aggregate_utilization = (
        actual_aggregate / lag.max_bandwidth_bps
    ) * 100
    
    # Minimum required members: ensure single-link failure doesn't exceed 100%
    min_required = math.ceil(aggregate_utilization / 100 * len(active_members) * 1.2)
    min_required = max(1, min_required)
    
    current_bandwidth = actual_aggregate / 1e6  # Mbps
    
    return {
        "timestamp": context.timestamp,
        "lag_id": lag.id,
        "switch_id": lag.switch_id,
        "total_members": len(lag.members),
        "active_member_count": len(active_members),
        "standby_member_count": len(standby_members),
        "aggregate_utilization": aggregate_utilization,
        "per_member_utilization": member_utils,   # Store as list
        "member_imbalance_ratio": member_imbalance,
        "minimum_required_members": min_required,
        "current_bandwidth_mbps": current_bandwidth,
        "max_bandwidth_mbps": lag.max_bandwidth_bps / 1e6,
    }
