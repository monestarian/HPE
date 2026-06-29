def enforce_correlations(port_rows_by_switch: dict, device_rows: dict, lag_rows: dict, stack_rows: dict):
    """
    Applies logic in-place to ensure telemetry makes physical sense.
    """
    for switch_id, port_rows in port_rows_by_switch.items():
        if not port_rows: continue
        
        access_ports = [r for r in port_rows if r["port_role"] == "access"]
        uplink_ports = [r for r in port_rows if r["port_role"] == "uplink"]
        
        # Rule 1: Uplink traffic >= sum of downstream access port traffic
        total_access_tx_bytes = sum(p["tx_bytes"] for p in access_ports)
        total_access_rx_bytes = sum(p["rx_bytes"] for p in access_ports)
        
        if uplink_ports:
            # Distribute needed uplink traffic across uplinks
            uplink_rx_needed = total_access_tx_bytes * 0.8
            uplink_tx_needed = total_access_rx_bytes * 0.8
            
            for up in uplink_ports:
                if up["rx_bytes"] < uplink_rx_needed / len(uplink_ports):
                    up["rx_bytes"] = uplink_rx_needed / len(uplink_ports)
                if up["tx_bytes"] < uplink_tx_needed / len(uplink_ports):
                    up["tx_bytes"] = uplink_tx_needed / len(uplink_ports)
                    
                up["rx_utilization"] = (up["rx_bytes"] * 8 / (up["link_speed_mbps"] * 1e6 * 300)) * 100
                up["tx_utilization"] = (up["tx_bytes"] * 8 / (up["link_speed_mbps"] * 1e6 * 300)) * 100
                up["total_utilization"] = (up["rx_utilization"] + up["tx_utilization"]) / 2

        # Device level correlations
        dev_row = device_rows.get(switch_id)
        if dev_row:
            # Check for storm peaks causing CPU spikes
            if any(p.get("storm_phase") == "peak" for p in port_rows):
                dev_row["cpu_utilization"] = max(dev_row["cpu_utilization"], 85.0)
                
            # Recalculate based on adjusted port rows
            dev_row["poe_usage_watts"] = sum(p["poe_consumption_watts"] for p in port_rows)
            dev_row["total_power_watts"] = dev_row["poe_usage_watts"] + 60.0 # chassis power proxy
            dev_row["total_flow_count"] = sum(p["flow_count"] for p in port_rows)
            dev_row["total_mac_count"] = sum(p["active_mac_count"] for p in port_rows)
            
    for r in list(port_rows_by_switch.values()):
        for p in r:
            # Rule 2: Utilization bounded by link speed
            if p["rx_utilization"] + p["tx_utilization"] > 200.0:
                p["rx_utilization"] = min(100.0, p["rx_utilization"])
                p["tx_utilization"] = min(100.0, p["tx_utilization"])
                
            # Rule 4: IP count <= MAC count
            if p["active_ip_count"] > p["active_mac_count"]:
                p["active_ip_count"] = p["active_mac_count"]
                
            # Rule 5: PoE is 0 when down
            if p["operational_status"] == 'down':
                p["poe_consumption_watts"] = 0.0
                
            # Ensure no negatives
            p["rx_bytes"] = max(0.0, float(p["rx_bytes"]))
            p["tx_bytes"] = max(0.0, float(p["tx_bytes"]))
