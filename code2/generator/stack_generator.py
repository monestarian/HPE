import random
from generator.temporal_engine import TimeContext
from generator.topology import Stack

def generate_stack_row(stack: Stack, device_rows: dict, context: TimeContext) -> dict:
    active_node = stack.active_switch
    standby_node = stack.standby_switch
    
    active_util = 5.0
    if active_node.id in device_rows:
        active_util = device_rows[active_node.id]["cpu_utilization"]
    
    # Standby carries only sync traffic: ~3-8% of active util
    sync_overhead = random.gauss(0.05, 0.01)
    standby_util = active_util * sync_overhead + random.gauss(2, 0.5)
    standby_util = max(0.0, min(standby_util, 100.0))
    
    # Synchronization status: occasionally drifts during high load
    sync_status = "synchronized"
    if active_util > 85 and random.random() < 0.05:
        sync_status = "sync_lag"   # Temporary sync lag under heavy load
    
    # VSF/VSX state
    vsf_state = "active-standby" if "VSF" in stack.id else "active-active"
    
    # VRRP/HSRP state on distribution switches
    vrrp_state = {active_node.id: "master", standby_node.id: "backup"}
    
    # Failover readiness: check if sync is current
    failover_ready = (sync_status == "synchronized")
    
    # Predicted power saving if standby is shut down
    standby_power = 0.0
    if standby_node.id in device_rows:
        standby_power = device_rows[standby_node.id]["total_power_watts"]
    
    return {
        "timestamp": context.timestamp,
        "stack_id": stack.id,
        "zone": stack.zone,
        "stack_enabled": True,
        "stack_member_count": len(stack.members),
        "active_switch_id": active_node.id,
        "standby_switch_id": standby_node.id,
        "active_node_cpu_util": active_util,
        "standby_node_cpu_util": standby_util,
        "vsf_state": vsf_state,
        "vsx_state": "peer-up" if "VSX" in stack.id else "none",
        "vrrp_state": vrrp_state,
        "hsrp_state": "active",
        "sync_status": sync_status,
        "failover_ready": failover_ready,
        "redundant_node_utilization": standby_util,
        "standby_power_watts": standby_power,
        "hours_since_last_failover": stack.hours_since_last_failover,
    }
