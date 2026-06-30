import yaml
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

CONFIG_DIR = Path(__file__).parent.parent / 'config'

@dataclass
class Port:
    id: str
    switch_id: str
    zone: str
    building: str
    floor: int
    endpoint_type: str
    link_speed: int  # Mbps
    role: str  # access, trunk, uplink
    vlan: int

@dataclass
class Switch:
    id: str
    zone: str
    poe_budget: float  # Watts
    chassis_power_watts: float

@dataclass
class LAGMember:
    port_id: str
    link_speed_bps: float
    state: str  # active, standby

@dataclass
class LAG:
    id: str
    switch_id: str
    members: List[LAGMember]
    max_bandwidth_bps: float

@dataclass
class Stack:
    id: str
    zone: str
    members: List[str]  # switch ids
    active_switch: Switch
    standby_switch: Switch
    hours_since_last_failover: int = 72

class CampusTopology:
    def __init__(self):
        self.graph = nx.Graph()
        self.ports: Dict[str, Port] = {}
        self.switches: Dict[str, Switch] = {}
        self.lags: Dict[str, LAG] = {}
        self.stacks: Dict[str, Stack] = {}
        
        with open(CONFIG_DIR / 'topology.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
            
        self._build_topology()
        
    def _build_topology(self):
        # 1. Core Layer
        core_config = self.config['network']['layer_0_core']
        core_1 = Switch(id=core_config['pair']['active'], zone="infrastructure", poe_budget=0, chassis_power_watts=200)
        core_2 = Switch(id=core_config['pair']['standby'], zone="infrastructure", poe_budget=0, chassis_power_watts=200)
        
        self.switches[core_1.id] = core_1
        self.switches[core_2.id] = core_2
        
        # Core pair stack/VSX
        self.stacks["VSX-CORE"] = Stack(
            id="VSX-CORE", 
            zone="infrastructure", 
            members=[core_1.id, core_2.id],
            active_switch=core_1,
            standby_switch=core_2
        )
        
        # 2. Distribution Layer
        dist_config = self.config['network']['layer_1_distribution']['switches']
        for dist_sw in dist_config:
            zone = dist_sw['zone']
            name = dist_sw['name']
            
            if dist_sw['type'] == 'vsf_stack':
                sw1 = Switch(id=f"{name}-1", zone=zone, poe_budget=0, chassis_power_watts=150)
                sw2 = Switch(id=f"{name}-2", zone=zone, poe_budget=0, chassis_power_watts=150)
                self.switches[sw1.id] = sw1
                self.switches[sw2.id] = sw2
                
                self.stacks[f"VSF-{name}"] = Stack(
                    id=f"VSF-{name}",
                    zone=zone,
                    members=[sw1.id, sw2.id],
                    active_switch=sw1,
                    standby_switch=sw2
                )
                active_dist_id = sw1.id
            else:
                sw = Switch(id=name, zone=zone, poe_budget=0, chassis_power_watts=120)
                self.switches[sw.id] = sw
                active_dist_id = sw.id
                
            # Create uplink port to core
            uplink_port_id = f"{active_dist_id}-UPLINK-1"
            uplink_port = Port(
                id=uplink_port_id, switch_id=active_dist_id, zone=zone, building="Dist", floor=1,
                endpoint_type="uplink", link_speed=dist_sw['uplink_gbps']*1000, role="uplink", vlan=1
            )
            self.ports[uplink_port.id] = uplink_port
            
            # Simplified graph connection
            self.graph.add_edge(core_1.id, active_dist_id)

        # 3. Access Layer & Endpoints
        access_cfg = self.config['network']['layer_2_access']
        ports_per_sw = access_cfg['ports_per_switch']
        
        # Allocate access switches per building in zones
        switch_counter = 1
        for zone_name, zone_data in self.config['zones'].items():
            dist_sw_id = f"DIST-SW-{zone_name.upper()}"
            if f"{dist_sw_id}-1" in self.switches:
                dist_sw_id = f"{dist_sw_id}-1"
            elif dist_sw_id == "DIST-SW-ADMINISTRATIVE":
                dist_sw_id = "DIST-SW-ADMIN"
            elif dist_sw_id == "DIST-SW-LIBRARY":
                dist_sw_id = "DIST-SW-LIB"
            
            if dist_sw_id not in self.switches:
                # fallback just in case
                dist_sw_id = list(self.switches.keys())[0]

            for building in zone_data['buildings']:
                b_name = building['name']
                # Determine how many switches needed for this building based on floors
                floors = building.get('floors', 1)
                
                for floor in range(1, floors + 1):
                    sw_id = f"ACC-SW{switch_counter:02d}-{b_name.replace(' ', '')}-F{floor}"
                    switch_counter += 1
                    
                    sw = Switch(id=sw_id, zone=zone_name, poe_budget=740.0, chassis_power_watts=60.0)
                    self.switches[sw.id] = sw
                    
                    # Connect to dist via LAG
                    lag_id = f"LAG-{sw_id}-TO-{dist_sw_id}"
                    lag_members = []
                    for i in range(access_cfg['lag_to_dist_members']):
                        port_id = f"{sw_id}-UPLINK-{i+1}"
                        p = Port(
                            id=port_id, switch_id=sw_id, zone=zone_name, building=b_name, floor=floor,
                            endpoint_type="uplink", link_speed=access_cfg['lag_speed_gbps']*1000, role="uplink", vlan=1
                        )
                        self.ports[p.id] = p
                        lag_members.append(LAGMember(port_id=p.id, link_speed_bps=p.link_speed * 1e6, state="active"))
                        
                    self.lags[lag_id] = LAG(
                        id=lag_id, switch_id=sw_id, members=lag_members, 
                        max_bandwidth_bps=sum(m.link_speed_bps for m in lag_members)
                    )
                    
                    self.graph.add_edge(dist_sw_id, sw_id)
                    
                    # Generate Access Ports
                    for i in range(1, ports_per_sw - access_cfg['lag_to_dist_members'] + 1):
                        port_id = f"{sw_id}-P{i}"
                        
                        # Determine endpoint type based on zone
                        endpoint_type = "student_pc"
                        if zone_name == "academic":
                            if i % 8 == 0: endpoint_type = "access_point"
                            elif i % 10 == 0: endpoint_type = "faculty_pc"
                        elif zone_name == "residential":
                            if i % 4 == 0: endpoint_type = "access_point"
                            else: endpoint_type = "student_laptop_dorm"
                        elif zone_name == "library":
                            if i % 10 == 0: endpoint_type = "printer"
                            elif i % 5 == 0: endpoint_type = "access_point"
                        elif zone_name == "administrative":
                            if i % 4 == 0: endpoint_type = "ip_phone"
                            elif i % 8 == 0: endpoint_type = "printer"
                            else: endpoint_type = "faculty_pc"
                        elif zone_name == "infrastructure":
                            endpoint_type = "server"

                        # occasional IP camera
                        if i == 40: endpoint_type = "ip_camera"

                        p = Port(
                            id=port_id, switch_id=sw_id, zone=zone_name, building=b_name, floor=floor,
                            endpoint_type=endpoint_type, link_speed=1000, role="access", vlan=10
                        )
                        self.ports[p.id] = p
