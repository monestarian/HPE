
from datetime import datetime
from generator.temporal_engine import TimeContext
from generator.topology import Port
from generator.port_generator import PortState, generate_port_row

def test_generate_port_row_basics():
    p = Port("P1", "SW1", "academic", "Eng", 1, "student_pc", 1000, "access", 10)
    dt = datetime(2024, 1, 10, 10, 0) # Wednesday 10 AM, class hours
    context = TimeContext(dt)
    state = PortState("P1")
    
    row = generate_port_row(p, dt, context, state)
    
    assert row["port_id"] == "P1"
    assert row["rx_bytes"] >= 0
    assert row["tx_bytes"] >= 0
    assert row["operational_status"] == "up"
