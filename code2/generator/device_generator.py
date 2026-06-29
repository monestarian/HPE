import math
import random
from datetime import datetime
from generator.topology import Switch

def generate_device_row(switch: Switch, port_rows: list[dict], timestamp: datetime) -> dict:
    # Aggregate from port rows for this switch at this timestamp
    total_flow_count = sum(r["flow_count"] for r in port_rows)
    total_mac_count = sum(r["active_mac_count"] for r in port_rows)
    active_port_count = sum(1 for r in port_rows if r["operational_status"] == "up")
    poe_usage_watts = sum(r["poe_consumption_watts"] for r in port_rows)
    
    # CPU: driven by flow count, protocol processing, and active ports
    cpu_base = 5.0  # idle baseline
    cpu_from_flows = min(40, total_flow_count * 0.002)
    cpu_from_ports = min(20, active_port_count * 0.3)
    cpu_noise = random.gauss(0, 3)
    cpu_util = min(100.0, max(0.0, cpu_base + cpu_from_flows + cpu_from_ports + cpu_noise))
    
    # Memory: driven by MAC table size, routing table, ARP table
    mem_base = 30.0
    mem_from_macs = min(30, total_mac_count * 0.005)
    mem_noise = random.gauss(0, 2)
    mem_util = min(100.0, max(0.0, mem_base + mem_from_macs + mem_noise))
    
    # Temperature: driven by CPU, total power, and a slow daily thermal cycle
    ambient = 22 + 4 * math.sin(2 * math.pi * (timestamp.hour - 14) / 24)  # Peaks at 2PM
    thermal_from_cpu = cpu_util * 0.15
    thermal_from_poe = poe_usage_watts * 0.01
    temperature = ambient + thermal_from_cpu + thermal_from_poe + random.gauss(0, 0.5)
    
    # PoE budget (fixed per switch model, e.g. 740W for a 48-port PoE+ switch)
    poe_budget_watts = switch.poe_budget
    
    # Total switch power draw (chassis + line cards + PoE)
    chassis_power = switch.chassis_power_watts  # e.g., 60W
    total_power = chassis_power + poe_usage_watts
    
    return {
        "timestamp": timestamp,
        "switch_id": switch.id,
        "zone": switch.zone,
        "cpu_utilization": cpu_util,
        "memory_utilization": mem_util,
        "temperature_celsius": temperature,
        "total_power_watts": total_power,
        "poe_budget_watts": poe_budget_watts,
        "poe_usage_watts": poe_usage_watts,
        "active_port_count": active_port_count,
        "total_mac_count": total_mac_count,
        "total_flow_count": total_flow_count,
    }
