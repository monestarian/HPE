# Coding Agent Prompt: Network Power-Saving Telemetry Pipeline
## University Campus — Synthetic Data Generation + ML Model

---

## 1. PROJECT OVERVIEW & OBJECTIVE

Build an end-to-end pipeline for predicting network power-saving decisions on a university campus network. The pipeline has three stages:

1. **Synthetic Telemetry Generator** — Produces 2 months of realistic, interrelated, time-aware network telemetry mimicking a university campus (January–February).
2. **Feature Engineering + Label Generation** — Transforms raw telemetry into ML-ready features with ground-truth labels for four prediction targets.
3. **ML Model Training + Inference** — Trains per-task classifiers to predict power-saving decisions in real time.

### Four Prediction Targets
| ID | Task | Decision |
|----|------|----------|
| T1 | Port Shutdown | Can this port be safely powered off right now? |
| T2 | LAG Member Removal | Can we remove one or more bundle members to save power? |
| T3 | Stack Node Shutdown | Can a standby/redundant switch be powered down safely? |
| T4 | Storm Warning | Is a broadcast storm or loop forming on this segment? |

---

## 2. REPOSITORY STRUCTURE

```
network-power-saver/
├── config/
│   ├── topology.yaml          # Campus network graph: nodes, links, roles
│   ├── schedules.yaml         # Class timetable, semester calendar, holidays
│   └── thresholds.yaml        # Safety thresholds for label generation
│
├── generator/
│   ├── topology.py            # Builds the campus network graph (networkx)
│   ├── temporal_engine.py     # Time context object: hour, weekday, semester state
│   ├── endpoint_profiles.py   # Endpoint type definitions and base traffic rates
│   ├── port_generator.py      # Per-port telemetry row generation
│   ├── lag_generator.py       # LAG bundle + per-member telemetry
│   ├── stack_generator.py     # Switch stack / VSF / VSX / redundancy telemetry
│   ├── device_generator.py    # Switch-level CPU, memory, power, thermal
│   ├── storm_injector.py      # Synthetic storm event injection
│   ├── correlation_engine.py  # Cross-parameter consistency enforcement
│   └── main.py                # Orchestrates full 2-month generation run
│
├── features/
│   ├── engineer.py            # Rolling windows, EWMA, trend, burst features
│   ├── labels.py              # Ground-truth label assignment logic
│   └── dataset.py             # Merges features + labels into final Parquet datasets
│
├── models/
│   ├── port_model.py          # T1: Port shutdown classifier
│   ├── lag_model.py           # T2: LAG member removal classifier
│   ├── stack_model.py         # T3: Stack node shutdown classifier
│   ├── storm_model.py         # T4: Storm warning classifier (sequence-aware)
│   ├── trainer.py             # Shared training loop, cross-val, SHAP explainability
│   └── evaluate.py            # Metrics, confusion matrices, calibration curves
│
├── predict/
│   └── inference.py           # Real-time inference given a telemetry snapshot
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis of generated telemetry
│   └── 02_model_eval.ipynb    # Model evaluation and SHAP visualizations
│
├── tests/
│   ├── test_generator.py
│   └── test_labels.py
│
├── requirements.txt
└── README.md
```

---

## 3. CAMPUS NETWORK TOPOLOGY

### 3.1 Physical Layout

Model a mid-sized university campus with ~8,000 students:

```
ZONES:
  Academic Zone:
    - Engineering Block (3 floors, 6 classrooms + 2 labs/floor)
    - Science Block (3 floors, 5 classrooms + 1 lab/floor)
    - Lecture Hall Complex (2 large halls, 400-seat capacity each)

  Residential Zone:
    - Boys Hostel (5 floors, 40 rooms/floor, 2 students/room)
    - Girls Hostel (5 floors, 40 rooms/floor, 2 students/room)
    - PG Hostel (3 floors, 30 rooms/floor)

  Library & Research:
    - Main Library (2 floors, 80 PC terminals, 4 printers)
    - Research Lab Block (1 floor, 3 specialized labs)

  Administrative:
    - Admin Block (2 floors, offices + conference rooms)
    - Faculty Building (3 floors, faculty cabins)

  Infrastructure:
    - Data Center Room (servers, NAS, UPS)
    - Network Operation Center (NOC)
```

### 3.2 Network Hierarchy

```
Layer 0 — Core (1 VSX Pair):
  CORE-SW-01 (active)
  CORE-SW-02 (standby)
  Connected via 40GbE LAG (4 members)

Layer 1 — Distribution (4 switches, one per zone):
  DIST-SW-ACAD    (VSF stack: 2 members, 10GbE uplinks)
  DIST-SW-RESID   (VSF stack: 2 members, 10GbE uplinks)
  DIST-SW-LIB     (standalone, 10GbE uplink)
  DIST-SW-ADMIN   (standalone, 10GbE uplink)

Layer 2 — Access (20 switches):
  Per zone: 2–6 access switches
  Connected to distribution via LAG (2 members, 1GbE)
  48 ports each, 802.3af/at PoE

Total ports modeled: ~960 access ports + ~40 uplink ports
Total LAG bundles: 12
Total stack nodes: 4 (in the 2 VSF stacks)
```

---

## 4. ENDPOINT PROFILES

Define these in `endpoint_profiles.py`. Each profile controls traffic generation:

```python
ENDPOINT_PROFILES = {
    "student_pc": {
        "base_rx_kbps": 800,       # Receiving (downloads, web)
        "base_tx_kbps": 150,       # Sending (uploads, DNS)
        "poe_watts": 0,
        "mac_count": 1,
        "active_pattern": "class_schedule",   # active during class hours
        "burst_probability": 0.05,
        "burst_multiplier": 8,
    },
    "faculty_pc": {
        "base_rx_kbps": 600,
        "base_tx_kbps": 200,
        "poe_watts": 0,
        "mac_count": 1,
        "active_pattern": "office_hours",
        "burst_probability": 0.03,
        "burst_multiplier": 5,
    },
    "access_point": {
        "base_rx_kbps": 15000,     # Aggregates many wireless clients
        "base_tx_kbps": 25000,
        "poe_watts": 15.4,
        "mac_count_range": (5, 60),  # Varies by occupancy
        "active_pattern": "occupancy",
        "burst_probability": 0.15,
        "burst_multiplier": 3,
    },
    "ip_phone": {
        "base_rx_kbps": 80,
        "base_tx_kbps": 80,
        "poe_watts": 3.84,
        "mac_count": 1,
        "active_pattern": "office_hours",
        "burst_probability": 0.01,
        "burst_multiplier": 2,
    },
    "ip_camera": {
        "base_rx_kbps": 10,
        "base_tx_kbps": 4000,      # Cameras mostly transmit
        "poe_watts": 5.0,
        "mac_count": 1,
        "active_pattern": "always_on",
        "burst_probability": 0.0,
        "burst_multiplier": 1,
    },
    "server": {
        "base_rx_kbps": 50000,
        "base_tx_kbps": 80000,
        "poe_watts": 0,
        "mac_count": 1,
        "active_pattern": "always_on",
        "burst_probability": 0.1,
        "burst_multiplier": 4,
    },
    "printer": {
        "base_rx_kbps": 50,
        "base_tx_kbps": 10,
        "poe_watts": 0,
        "mac_count": 1,
        "active_pattern": "office_hours",
        "burst_probability": 0.08,
        "burst_multiplier": 20,  # Big print jobs
    },
    "student_laptop_dorm": {
        "base_rx_kbps": 2000,
        "base_tx_kbps": 300,
        "poe_watts": 0,
        "mac_count": 1,
        "active_pattern": "dorm_schedule",  # evenings + late nights
        "burst_probability": 0.12,
        "burst_multiplier": 10,
    },
    "uplink": {
        # Uplink ports — aggregated traffic, never shut down
        "always_active": True,
        "mac_count_range": (100, 3000),
    }
}
```

---

## 5. TEMPORAL ENGINE

This is the core of realism. Implement in `temporal_engine.py`.

### 5.1 University Calendar (Jan 1 – Feb 28)

```python
CALENDAR = {
    "semester_start": "2024-01-08",   # First week is orientation
    "orientation_week": ("2024-01-08", "2024-01-12"),
    "regular_weeks": ("2024-01-15", "2024-02-16"),
    "exam_prep_week": ("2024-02-17", "2024-02-23"),
    "exam_week": ("2024-02-24", "2024-02-28"),

    "holidays": [
        "2024-01-14",  # Makar Sankranti
        "2024-01-26",  # Republic Day
    ],

    "events": [
        {"date": "2024-01-20", "name": "Cultural Fest Day 1", "zone_multipliers": {"academic": 0.3, "resid": 1.5}},
        {"date": "2024-01-21", "name": "Cultural Fest Day 2", "zone_multipliers": {"academic": 0.3, "resid": 1.5}},
        {"date": "2024-02-03", "name": "Sports Day", "zone_multipliers": {"academic": 0.5, "resid": 1.2}},
    ]
}
```

### 5.2 Occupancy Multiplier Function

Return a float [0.0, 1.0] representing what fraction of endpoints are active:

```python
def get_occupancy(zone: str, hour: int, minute: int, context: TimeContext) -> float:
    """
    Returns active fraction for a given zone and time context.
    Implement the following rules:
    
    ACADEMIC ZONE:
      - class_hours = 8:00–18:00 on weekdays
      - Periods: 8-10, 10-12, 12-13(lunch), 13-15, 15-17, 17-18
      - Occupancy = 0.85 during class periods (not lunch)
      - Occupancy = 0.30 during lunch
      - Occupancy = 0.05 after 18:00 (late students)
      - Occupancy = 0.0 after 22:00
      - Weekend: 0.02 (security, occasional researchers)
      - Exam week: shift peak to evenings (0.6 at 19:00-22:00), labs stay 0.9
      - Holiday: 0.01
    
    RESIDENTIAL ZONE:
      - Morning (6:00-9:00): 0.6 (students waking up, getting ready)
      - Class hours (9:00-17:00) weekdays: 0.2 (most out)
      - Evening (17:00-23:00): 0.9
      - Late night (23:00-2:00): 0.7 (gamers, Netflix)
      - Dead hours (2:00-6:00): 0.15
      - Weekend morning: 0.5, Weekend afternoon: 0.6, Weekend evening: 0.95
      - Exam week: stays at 0.85 until 2:00 AM
    
    LIBRARY ZONE:
      - Weekday 8:00-20:00: 0.6
      - Weekday 20:00-22:00: 0.3
      - Weekend: 0.4 (11:00-18:00), else 0.0
      - Exam prep week: 0.95 all day until 23:00
      - Exam week: 0.98 (maximum occupancy)
    
    ADMIN ZONE:
      - Weekday 9:00-17:00: 0.85
      - Weekday 8:00-9:00 / 17:00-18:00: 0.3
      - Outside office hours / weekends / holidays: 0.02
    
    DATA CENTER (always_on):
      - 0.95 constant, slight dip at 3:00-4:00 AM (0.90) during backup windows
    
    Apply gaussian noise of ±5% to all values.
    Apply the event multipliers from the calendar if applicable.
    """
```

### 5.3 Class Schedule (for per-port granularity)

Assign each classroom port to a specific class block. Class blocks:
- Block A: Mon/Wed/Fri 8:00–10:00
- Block B: Mon/Wed/Fri 10:00–12:00
- Block C: Tue/Thu 8:00–11:00
- Block D: Tue/Thu 11:00–14:00
- Block E: Mon–Fri 13:00–15:00
- Block F: Mon–Fri 15:00–17:00
- Block G: Mon–Fri 17:00–19:00 (electives/labs)

Each classroom port activates sharply at block start (+5 min ramp) and deactivates at block end (-5 min ramp). Between blocks: ~3 users stay (0.1 occupancy).

---

## 6. PORT TELEMETRY GENERATION

### 6.1 Core Formula

For each port at each 5-minute interval:

```python
def generate_port_row(port: Port, timestamp: datetime, context: TimeContext) -> dict:
    
    # Step 1: Get occupancy fraction
    occupancy = get_occupancy(port.zone, context.hour, context.minute, context)
    
    # Step 2: Apply endpoint profile base rates
    profile = ENDPOINT_PROFILES[port.endpoint_type]
    
    # Step 3: Check if a burst event is occurring
    is_burst = random.random() < profile["burst_probability"]
    burst_mult = profile["burst_multiplier"] if is_burst else 1.0
    
    # Step 4: Compute raw traffic
    rx_bps = profile["base_rx_kbps"] * 1000 * occupancy * burst_mult
    tx_bps = profile["base_tx_kbps"] * 1000 * occupancy * burst_mult
    
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
        mac_count = int(profile["mac_count_range"][0] + 
                       occupancy * (profile["mac_count_range"][1] - profile["mac_count_range"][0]))
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
    
    # Step 11: Queue depth and drops
    # Queue depth rises non-linearly above 70% utilization
    if total_util < 70:
        queue_depth = int(total_util * 0.5 * random.gauss(1.0, 0.2))
    else:
        queue_depth = int(50 + (total_util - 70) * 8 * random.gauss(1.0, 0.3))
    queue_depth = min(queue_depth, 255)
    
    queue_drops = 0
    if queue_depth > 200 or total_util > 90:
        queue_drops = int(packet_rate * 0.01 * random.gauss(1.0, 0.5))
        queue_drops = max(0, queue_drops)
    
    # Step 12: PoE consumption (0 if port is down or endpoint doesn't use PoE)
    poe_watts = profile.get("poe_watts", 0) if occupancy > 0 else 0
    poe_watts *= random.gauss(1.0, 0.05)  # Slight PoE draw variation
    
    # Step 13: Port state flags
    is_active = occupancy > 0.01 or port.endpoint_type in ["always_on", "server", "ip_camera"]
    operational_status = "up" if is_active else "down"
    
    # Step 14: Idle duration tracking
    # (managed externally by the generator's state machine per port)
    
    # Step 15: Derived temporal markers
    # Track consecutive idle 5-min windows (passed in as port state)
    
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
    }
```

### 6.2 Per-Port State Machine

Maintain a `PortState` object per port to track:
- `last_active_timestamp`: last timestamp when traffic > threshold
- `idle_duration_minutes`: minutes since last activity
- `consecutive_idle_windows`: count of 5-min windows with near-zero traffic
- `last_burst_timestamp`
- `time_since_last_burst_minutes`
- Rolling buffer of last 12 utilization readings (60 min)

Update this state at each interval before generating the row.

---

## 7. DEVICE-LEVEL TELEMETRY

Generate in `device_generator.py`. One row per switch per 5-minute interval.

```python
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
    cpu_util = min(100, max(0, cpu_base + cpu_from_flows + cpu_from_ports + cpu_noise))
    
    # Memory: driven by MAC table size, routing table, ARP table
    mem_base = 30.0
    mem_from_macs = min(30, total_mac_count * 0.005)
    mem_noise = random.gauss(0, 2)
    mem_util = min(100, max(0, mem_base + mem_from_macs + mem_noise))
    
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
```

---

## 8. LAG TELEMETRY GENERATION

Generate in `lag_generator.py`.

### Rules:

- Each LAG has 2–4 members
- Member link speeds: 1GbE (access→dist), 10GbE (dist→core)
- Aggregate bandwidth = sum of active member speeds
- Traffic is distributed across members using weighted hash (approximately equal with ±10% imbalance noise)
- One member may be "standby" (LACP passive, carries traffic only when others are overloaded)

```python
def generate_lag_row(lag: LAG, context: TimeContext, aggregate_traffic_bps: float) -> dict:
    
    active_members = [m for m in lag.members if m.state == "active"]
    standby_members = [m for m in lag.members if m.state == "standby"]
    
    # Distribute traffic across active members with imbalance
    per_member_traffic = []
    for member in active_members:
        share = aggregate_traffic_bps / len(active_members)
        share *= random.gauss(1.0, 0.10)   # ±10% imbalance per member
        per_member_traffic.append(max(0, share))
    
    # Normalize to actual aggregate
    actual_aggregate = sum(per_member_traffic)
    
    # Per-member utilization
    member_utils = [
        (t / (m.link_speed_bps)) * 100
        for t, m in zip(per_member_traffic, active_members)
    ]
    
    member_imbalance = (
        (max(member_utils) - min(member_utils)) / (sum(member_utils) / len(member_utils))
        if sum(member_utils) > 0 else 0
    )
    
    aggregate_utilization = (
        actual_aggregate / (lag.max_bandwidth_bps)
    ) * 100
    
    # Minimum required members: ensure single-link failure doesn't exceed 100%
    min_required = math.ceil(aggregate_utilization / 100 * len(active_members) * 1.2)
    min_required = max(1, min_required)
    
    current_bandwidth = actual_aggregate / 1e6  # Mbps
    # Predicted demand: use 15-min trailing average (tracked externally)
    
    return {
        "timestamp": context.timestamp,
        "lag_id": lag.id,
        "switch_id": lag.switch_id,
        "total_members": len(lag.members),
        "active_member_count": len(active_members),
        "standby_member_count": len(standby_members),
        "aggregate_utilization": aggregate_utilization,
        "per_member_utilization": member_utils,   # Store as JSON list
        "member_imbalance_ratio": member_imbalance,
        "minimum_required_members": min_required,
        "current_bandwidth_mbps": current_bandwidth,
        "max_bandwidth_mbps": lag.max_bandwidth_bps / 1e6,
    }
```

---

## 9. SWITCH STACK / REDUNDANCY TELEMETRY

Generate in `stack_generator.py`. Covers VSF stacks and VSX pairs.

```python
def generate_stack_row(stack: Stack, device_rows: dict, context: TimeContext) -> dict:
    
    active_node = stack.active_switch
    standby_node = stack.standby_switch
    
    active_util = device_rows[active_node.id]["cpu_utilization"]
    
    # Standby carries only sync traffic: ~3-8% of active util
    sync_overhead = random.gauss(0.05, 0.01)
    standby_util = active_util * sync_overhead + random.gauss(2, 0.5)
    standby_util = max(0, min(standby_util, 100))
    
    # Synchronization status: occasionally drifts during high load
    sync_status = "synchronized"
    if active_util > 85 and random.random() < 0.05:
        sync_status = "sync_lag"   # Temporary sync lag under heavy load
    
    # VSF/VSX state
    vsf_state = "active-standby"  # Or active-active for VSX
    
    # VRRP/HSRP state on distribution switches
    vrrp_state = {active_node.id: "master", standby_node.id: "backup"}
    
    # Failover readiness: check if sync is current, standby link is up
    failover_ready = (sync_status == "synchronized")
    
    # Predicted power saving if standby is shut down
    standby_power = device_rows[standby_node.id]["total_power_watts"] if standby_node.id in device_rows else 0
    
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
        "vsx_state": "peer-up",
        "vrrp_state": vrrp_state,
        "hsrp_state": "active",
        "sync_status": sync_status,
        "failover_ready": failover_ready,
        "redundant_node_utilization": standby_util,
        "standby_power_watts": standby_power,
        "hours_since_last_failover": stack.hours_since_last_failover,
    }
```

### Stack Failover Injection

Randomly inject 2–3 failover events during the 2-month period:
- Duration: 5–30 seconds (captured across 1–2 intervals)
- Symptoms: active switches swapped, brief sync_status = "re-synchronizing", CPU spike on new active node

---

## 10. STORM EVENT INJECTION

Implement in `storm_injector.py`. Inject 3–5 storm events with realistic precursors.

### Storm Anatomy

```
Phase 1 — PRECURSOR (30–60 min before storm):
  - RX bytes creep up 20-40% above baseline
  - RX/TX ratio begins rising (RX starts outpacing TX)
  - Queue depth rises slightly
  - Flow count increases modestly

Phase 2 — STORM ONSET (5–15 minutes):
  - RX utilization spikes to 80-100% (looping packets multiplying)
  - RX/TX ratio exceeds 4:1 (traffic is looping, not truly bidirectional)
  - Packet rate 5-15x normal
  - Queue drops spike dramatically
  - Flow count paradoxically LOW (most traffic is duplicate frames, not new flows)
  - MAC table instability: active_mac_count oscillates wildly
  - Spreads to multiple ports on same VLAN within 2–5 min

Phase 3 — STORM PEAK (10–30 minutes):
  - Multiple affected ports show >90% utilization simultaneously
  - Device CPU spikes to 80-100% (processing duplicate frames)
  - Temperature rises
  - Surrounding ports may lose connectivity (masked traffic)

Phase 4 — RESOLUTION:
  - Spanning tree reconverges (30–60 seconds of instability)
  - Traffic drops abruptly to near-zero
  - Recovery over 5–10 minutes
```

### Storm Types to Inject

1. **Broadcast storm**: Caused by a loop (student plugged in a hub with uplinks)
   - Affects: 1 VLAN across an access switch
   - Primary indicator: RX ≫ TX
   
2. **Multicast storm**: IGMP snooping misconfigured
   - Affects: Specific multicast groups
   - Moderate RX/TX ratio spike (2:1 to 3:1)
   
3. **Unknown unicast flood**: MAC table exhaustion
   - Affects: All ports on switch flooding to one destination
   - TX spikes across all ports, not just one

Mark each interval in the storm window with `storm_type` and `storm_phase` columns for labeling.

---

## 11. CORRELATION ENGINE

Implement in `correlation_engine.py`. After all generators produce rows, enforce cross-parameter consistency.

### Critical Correlations to Enforce

```python
CORRELATION_RULES = [

    # Rule 1: Uplink traffic >= sum of downstream access port traffic
    # (trunk aggregates everything below it)
    "uplink_rx >= 0.8 * sum(access_port_tx on same switch)",
    
    # Rule 2: Utilization is physically bounded by link speed
    "rx_utilization + tx_utilization <= 200",  # Full duplex
    
    # Rule 3: Queue drops only occur at high utilization
    "queue_drops > 0  implies  total_utilization > 75",
    
    # Rule 4: IP count never exceeds MAC count
    "active_ip_count <= active_mac_count",
    
    # Rule 5: PoE usage is 0 when port is operationally down
    "poe_consumption_watts == 0  when  operational_status == 'down'",
    
    # Rule 6: Flow count correlates with active IPs
    "flow_count approximately_equals  active_ip_count * [8, 20]",
    
    # Rule 7: During storms, flow_count is low despite high packet rate
    "storm_onset  implies  flow_count < 0.5 * normal_flow_count",
    
    # Rule 8: Stack standby CPU is proportional to active CPU
    "standby_cpu approximately_equals  active_cpu * [0.03, 0.08]",
    
    # Rule 9: Temperature follows CPU/power with thermal lag
    # (implement with a rolling exponential smoother, τ = 30 min)
    "temperature = EWMA(f(cpu, poe), alpha=0.1)",
    
    # Rule 10: LAG aggregate utilization <= sum of per-member capacities
    "aggregate_utilization <= 100 * active_member_count / total_member_count",
    
    # Rule 11: Peak utilization is the rolling 24h max
    "peak_utilization = rolling_max(total_utilization, window=288_intervals)",
    
    # Rule 12: MAC count on AP port proportional to occupancy
    "ap_mac_count = int(max_clients * occupancy) ± 10%",
]
```

---

## 12. FEATURE ENGINEERING

Implement in `features/engineer.py`. Transform raw telemetry into ML features.

### Rolling Window Features

For each numeric metric, compute over windows: `[3, 6, 12, 36, 72]` intervals (= 15min, 30min, 1hr, 3hr, 6hr):

```python
ROLLING_FEATURES = [
    ("mean",    ["rx_bytes", "tx_bytes", "total_utilization", "packet_rate_pps",
                 "flow_count", "queue_drops", "active_mac_count"]),
    ("max",     ["total_utilization", "packet_rate_pps", "queue_drops", "queue_depth"]),
    ("min",     ["total_utilization"]),
    ("std",     ["total_utilization", "packet_rate_pps", "rx_bytes"]),
    ("sum",     ["rx_bytes", "tx_bytes", "queue_drops"]),
]
```

### Derived Features

```python
DERIVED_FEATURES = {
    # Traffic asymmetry (storm indicator)
    "rx_tx_ratio": "rx_bytes / (tx_bytes + 1)",
    
    # Utilization trend (slope of last 6 intervals)
    "util_trend": "linregress(total_utilization, last_6).slope",
    
    # Traffic acceleration (change in trend)
    "util_acceleration": "util_trend[t] - util_trend[t-1]",
    
    # EWMA of utilization (smoothed, α=0.3)
    "util_ewma": "EWMA(total_utilization, alpha=0.3)",
    
    # Burstiness (ratio of max to mean over last 12 intervals)
    "burstiness": "rolling_max_12 / (rolling_mean_12 + 1e-6)",
    
    # Packet size anomaly (unusually small packets = storm symptom)
    "avg_packet_size": "(rx_bytes + tx_bytes) / (packet_rate_pps * 300 + 1)",
    
    # Idle probability (fraction of last 12 windows below threshold)
    "idle_probability_1h": "count(util < 1.0, last_12) / 12",
    "idle_probability_6h": "count(util < 1.0, last_72) / 72",
    
    # Time since last burst (minutes)
    "minutes_since_last_burst": "use port state machine output",
    
    # Growth rate
    "traffic_growth_rate": "(rolling_mean_12 - rolling_mean_36) / (rolling_mean_36 + 1e-6)",
    
    # LAG-specific: utilization headroom
    "lag_headroom": "100 - aggregate_utilization",
    "lag_can_lose_member": "(aggregate_utilization / (active_members - 1)) < 70",
    
    # Historical averages (same hour/weekday from prior weeks)
    "hist_avg_util_same_hour_weekday": "lookback into generated data",
    "hist_avg_util_deviation": "current_util - hist_avg_util_same_hour_weekday",
    
    # Time features (cyclic encoding)
    "hour_sin": "sin(2π * hour / 24)",
    "hour_cos": "cos(2π * hour / 24)",
    "weekday_sin": "sin(2π * weekday / 7)",
    "weekday_cos": "cos(2π * weekday / 7)",
    "semester_progress": "(day_of_semester / total_semester_days)",
}
```

---

## 13. LABEL GENERATION

Implement in `features/labels.py`. Assign binary ground-truth labels per row.

### T1: `can_shutdown_port`

```python
def label_port_shutdown(row, port_state) -> bool:
    """
    Returns True if the port CAN be safely powered off.
    ALL conditions must hold:
    """
    return all([
        row["admin_status"] == "up",            # Port is manageable
        row["endpoint_type"] not in ["server", "uplink", "ip_camera"],  # Never shut these
        row["total_utilization"] < 1.0,          # Effectively idle
        port_state.idle_duration_minutes > 30,   # Idle for at least 30 minutes
        row["active_mac_count"] == 0,            # No clients attached
        row["queue_drops"] == 0,                 # No congestion
        not row["is_storm_affected"],            # No storm context
        # At least ONE of these time conditions:
        any([
            row["hour"] < 7 or row["hour"] >= 22,  # Night hours
            row["is_weekend"] and row["hour"] < 9,   # Weekend early morning
            row["is_holiday"],                         # Public holiday
            (row["semester_state"] == "exam" and row["zone"] == "academic"
             and (row["hour"] < 8 or row["hour"] > 22)),
        ])
    ])
```

### T2: `can_remove_lag_member`

```python
def label_lag_member_removal(lag_row, trend_row) -> bool:
    """
    Returns True if at least one LAG member can be safely removed.
    """
    return all([
        lag_row["active_member_count"] > 2,           # Must retain at least 2
        lag_row["aggregate_utilization"] < 40.0,       # Low aggregate load
        min(lag_row["per_member_utilization"]) < 5.0,  # At least one near-idle member
        lag_row["member_imbalance_ratio"] < 0.3,       # Balanced enough to redistribute
        trend_row["util_trend"] <= 0,                  # Traffic not growing
        trend_row["hist_avg_util_deviation"] < 5.0,    # Not abnormally low (watch for lull before surge)
    ])
```

### T3: `can_shutdown_stack_node`

```python
def label_stack_shutdown(stack_row, context) -> bool:
    """
    Returns True if a standby stack node can be safely powered off.
    """
    return all([
        stack_row["stack_member_count"] > 2,            # Stack must remain functional
        stack_row["sync_status"] == "synchronized",     # Don't touch during sync lag
        stack_row["redundant_node_utilization"] < 8.0,  # Standby truly idle
        stack_row["failover_ready"],                     # Can restore if needed
        stack_row["hours_since_last_failover"] > 72,    # Recently stable (no recent failovers)
        # Time conditions: only recommend during very low-risk windows
        any([
            (context.hour >= 2 and context.hour < 5),   # Deep night
            (context.is_weekend and context.hour >= 0 and context.hour < 7),
        ]),
        # Zone-specific: only residential/academic; never core or datacenter
        stack_row["zone"] in ["academic", "residential", "library"],
    ])
```

### T4: `storm_warning` (multi-class)

```python
STORM_CLASSES = {
    0: "normal",
    1: "precursor",     # Early warning — investigate
    2: "storm_active",  # Action required — block port / enable storm control
    3: "recovery",      # Storm ending — monitor
}

def label_storm_warning(row, storm_state) -> int:
    if storm_state is None:
        return 0  # Normal
    
    phase = storm_state.phase  # precursor / onset / peak / recovery
    
    if phase == "precursor":
        return 1
    elif phase in ("onset", "peak"):
        return 2
    elif phase == "recovery":
        return 3
    return 0
```

---

## 14. ML MODEL SPECIFICATIONS

### 14.1 Model Architecture

```
T1 (Port Shutdown):
  Algorithm: LightGBM binary classifier
  Features: Port telemetry + rolling features + time features
  Positive class: can_shutdown_port == True
  Class imbalance: ~5-10% positive (ports can be off for several hours at night)
  Handle with: scale_pos_weight or class_weight='balanced'

T2 (LAG Member Removal):
  Algorithm: LightGBM binary classifier
  Features: LAG telemetry + per-member features (flattened) + trend features
  Positive class: can_remove_lag_member == True
  Class imbalance: ~20-30% positive

T3 (Stack Node Shutdown):
  Algorithm: LightGBM binary classifier
  Features: Stack telemetry + device telemetry for both nodes + time features
  Positive class: can_shutdown_stack_node == True
  Class imbalance: ~15% positive

T4 (Storm Warning):
  Algorithm: 
    Primary: LSTM (sequence length = 12 intervals = 60 min, all port telemetry features)
    Secondary: LightGBM on rolling features (for comparison and ensemble)
  Target: 4-class (0=normal, 1=precursor, 2=storm, 3=recovery)
  Note: Weight class 1 (precursor) heavily — false negatives on precursor are costly
```

### 14.2 Feature Set per Model

**T1 Feature Set** (per port, per interval):
- All raw telemetry columns (excluding labels and internal tracking)
- Rolling means/max/std for 15min, 30min, 1hr, 3hr, 6hr windows
- idle_probability_1h, idle_probability_6h
- hist_avg_util_same_hour_weekday, hist_avg_util_deviation
- minutes_since_last_burst, consecutive_idle_windows
- burstiness, rx_tx_ratio
- Cyclic time encodings (hour_sin, hour_cos, weekday_sin, weekday_cos)
- semester_progress, is_exam_period
- endpoint_type (one-hot), zone (one-hot), port_role (one-hot)

**T4 Feature Set** (port-level, past 12 intervals as sequence):
- rx_bytes_norm, tx_bytes_norm (normalized by link speed)
- rx_tx_ratio
- packet_rate_pps_norm
- queue_drops, queue_depth_norm
- active_mac_count_norm
- flow_count_norm
- util_trend, util_acceleration
- burstiness

### 14.3 Train/Validation Split

- Training: Jan 1 – Feb 14 (~6.5 weeks)
- Validation: Feb 15 – Feb 28 (~2 weeks)
- Do NOT use random split (time series data)
- Use walk-forward validation for storm model (LSTM)

### 14.4 Evaluation Metrics

```
T1, T2, T3: Binary classification
  Primary: F1-Score (balance precision/recall)
  Secondary: AUC-ROC
  Safety metric: False Positive Rate < 1% (never shut down an active port)
  
T4: Multi-class
  Primary: Weighted F1 across all 4 classes
  Critical: Recall on class 1 (precursor) > 85%
  Critical: Recall on class 2 (storm_active) > 95%
```

### 14.5 Explainability

- Compute SHAP values for all LightGBM models
- Generate SHAP summary plots and force plots for top 10 most important features
- Include a human-readable explanation alongside each prediction:
  ```
  "Port ENGR-FL2-P14 can be shut off: idle for 47 min, 
   zero clients, 2:15 AM on Sunday, historically inactive this hour"
  ```

---

## 15. GENERATION PARAMETERS & SCALE

```python
GENERATION_CONFIG = {
    "start_date": "2024-01-01 00:00:00",
    "end_date": "2024-02-29 23:55:00",
    "interval_minutes": 5,
    "total_intervals": 17_568,    # 2 months at 5-min intervals
    "access_ports": 960,
    "uplink_ports": 40,
    "lag_bundles": 12,
    "stack_nodes": 4,
    "switches": 26,               # 1 core pair + 4 dist + 20 access + 1 NOC

    # Output estimates
    "port_rows": 960 * 17_568,    # ~16.9M rows
    "device_rows": 26 * 17_568,   # ~457K rows
    "lag_rows": 12 * 17_568,      # ~211K rows
    "stack_rows": 4 * 17_568,     # ~70K rows

    # Storage
    "output_format": "parquet",   # Partitioned by date
    "output_path": "data/raw/",
    "features_path": "data/features/",
    "models_path": "models/saved/",
}
```

---

## 16. IMPLEMENTATION NOTES

### Performance
- Use `numpy` vectorized operations wherever possible instead of row-by-row loops
- Generate data date-by-date, writing Parquet partitions incrementally
- Use `polars` for feature engineering (faster than pandas for large DataFrames)
- Parallelize port generation across buildings using `concurrent.futures.ProcessPoolExecutor`
- Expected generation time: ~5-10 minutes for full 2-month dataset

### Data Quality Validation

After generation, run automated checks:
```python
VALIDATION_CHECKS = [
    "No null values in any critical column",
    "rx_utilization in [0, 100] for all non-storm rows",
    "ip_count <= mac_count for all rows",
    "poe_consumption == 0 when operational_status == 'down'",
    "queue_drops == 0 when total_utilization < 50",
    "standby_cpu < 15% for all non-storm, non-failover intervals",
    "At least 3 storm events present across full dataset",
    "Storm events have peak rx_tx_ratio > 4.0",
    "Label T1 positive rate between 5% and 15%",
    "Label T4 class distribution: >90% normal, ~2-5% storm_active",
    "All holiday dates show academic zone utilization < 5%",
    "Exam week shows library utilization > 90% on weekday evenings",
]
```

### Configuration Files

`config/thresholds.yaml` should contain all numeric thresholds as configurable values — never hardcode them in the generator or model code. This allows easy tuning without changing logic.

---

## 17. REQUIRED LIBRARIES

```
# requirements.txt
numpy>=1.26
pandas>=2.0
polars>=0.20
pyarrow>=14.0
networkx>=3.2          # Campus topology graph
lightgbm>=4.3
scikit-learn>=1.4
torch>=2.2             # LSTM for storm detection
shap>=0.45
matplotlib>=3.8
seaborn>=0.13
pyyaml>=6.0
tqdm>=4.66
scipy>=1.12            # For linregress in trend features
pytest>=8.0
jupyter>=1.0
```

---

## 18. DELIVERABLES CHECKLIST

- [ ] `config/` — all three YAML files complete
- [ ] `generator/main.py` — runs end-to-end, generates all Parquet files
- [ ] `generator/storm_injector.py` — injects at least 3 typed storm events
- [ ] `features/engineer.py` — all rolling, derived, cyclic, and historical features
- [ ] `features/labels.py` — all four label functions with unit tests
- [ ] `features/dataset.py` — assembles final train/val splits as Parquet
- [ ] `models/trainer.py` — trains all 4 models, saves to `models/saved/`
- [ ] `models/evaluate.py` — prints classification reports + SHAP plots
- [ ] `predict/inference.py` — takes a dict of current telemetry, returns all 4 predictions with human-readable explanations
- [ ] `notebooks/01_eda.ipynb` — at least 8 plots: temporal traffic patterns, storm signature, occupancy heatmap, feature distributions
- [ ] `notebooks/02_model_eval.ipynb` — confusion matrices, AUC curves, SHAP summaries
- [ ] `tests/` — at least 15 unit tests covering label logic and correlation rules
- [ ] `README.md` — setup instructions, architecture diagram, example output

---

## 19. SAMPLE OUTPUT FORMAT

`predict/inference.py` should output JSON like:

```json
{
  "timestamp": "2024-02-18T03:20:00",
  "predictions": {
    "port_shutdown": [
      {
        "port_id": "RESID-HB-FL3-P22",
        "confidence": 0.94,
        "decision": "SHUTDOWN_RECOMMENDED",
        "reason": "Idle 73 minutes. Zero clients. 3:20 AM on Sunday. Dormitory port, historically offline Sat/Sun nights.",
        "estimated_watts_saved": 3.84
      }
    ],
    "lag_member_removal": [
      {
        "lag_id": "LAG-DIST-ACAD-01",
        "removable_member": "Port-Channel1/0/4",
        "confidence": 0.81,
        "decision": "REMOVE_MEMBER",
        "reason": "Aggregate at 28%. Member-4 at 3%. Traffic trending flat. 3 members sufficient.",
        "estimated_watts_saved": 8.5
      }
    ],
    "stack_shutdown": [],
    "storm_warning": [
      {
        "switch_id": "ACCESS-SW-ENGR-B02",
        "storm_class": 1,
        "class_label": "precursor",
        "confidence": 0.73,
        "decision": "INVESTIGATE",
        "reason": "RX/TX ratio rising (2.1x). Packet rate 40% above 1h average. Queue depth increasing."
      }
    ]
  },
  "summary": {
    "total_power_saving_watts": 12.34,
    "ports_recommended_off": 1,
    "lag_members_recommended_off": 1,
    "stack_nodes_recommended_off": 0,
    "active_storm_warnings": 1
  }
}
```

---

## 20. FINAL NOTE TO AGENT

Start with `generator/main.py` and work top-down. Validate generator output before moving to features. Do NOT write feature engineering code until the generator produces statistically sensible output (validated by `notebooks/01_eda.ipynb`). The quality of the synthetic data is the single most important determinant of model quality — invest 50% of your effort here.

Prioritize correctness of storm injection and T4 labels — this is the most complex and most novel part of the system.
