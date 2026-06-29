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
