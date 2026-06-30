import datetime
import math
import random
import yaml
from pathlib import Path

# Load schedules config
CONFIG_DIR = Path(__file__).parent.parent / 'config'
with open(CONFIG_DIR / 'schedules.yaml', 'r') as f:
    SCHEDULES = yaml.safe_load(f)

class TimeContext:
    def __init__(self, timestamp: datetime.datetime):
        self.timestamp = timestamp
        self.hour = timestamp.hour
        self.minute = timestamp.minute
        self.weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        self.is_weekend = self.weekday >= 5
        
        date_str = timestamp.strftime("%Y-%m-%d")
        
        self.is_holiday = date_str in SCHEDULES.get("holidays", [])
        
        self.semester_state = self._get_semester_state(date_str)
        
        # Check for events
        self.active_events = [e for e in SCHEDULES.get("events", []) if e["date"] == date_str]

    def _get_semester_state(self, date_str: str) -> str:
        cal = SCHEDULES.get("calendar", {})
        
        # Simple string comparison works for YYYY-MM-DD
        if cal.get("orientation_week", [""])[0] <= date_str <= cal.get("orientation_week", ["", ""])[1]:
            return "orientation"
        elif cal.get("regular_weeks", [""])[0] <= date_str <= cal.get("regular_weeks", ["", ""])[1]:
            return "regular"
        elif cal.get("exam_prep_week", [""])[0] <= date_str <= cal.get("exam_prep_week", ["", ""])[1]:
            return "exam_prep"
        elif cal.get("exam_week", [""])[0] <= date_str <= cal.get("exam_week", ["", ""])[1]:
            return "exam"
        
        return "out_of_session"

def get_occupancy(zone: str, hour: int, minute: int, context: TimeContext) -> float:
    """
    Returns active fraction for a given zone and time context.
    """
    occupancy = 0.0
    
    # Event multipliers
    event_multiplier = 1.0
    for event in context.active_events:
        mults = event.get("zone_multipliers", {})
        if zone in mults:
            event_multiplier *= mults[zone]
    
    if zone == "academic":
        if context.is_holiday:
            occupancy = 0.01
        elif context.is_weekend:
            occupancy = 0.02
        else:
            # Weekday
            if context.semester_state == "exam":
                if 8 <= hour < 18:
                    occupancy = 0.9 # Lab
                elif 19 <= hour < 22:
                    occupancy = 0.6
                else:
                    occupancy = 0.05
            else:
                if 8 <= hour < 18:
                    if 12 <= hour < 13:
                        occupancy = 0.30
                    else:
                        occupancy = 0.85
                elif hour >= 22:
                    occupancy = 0.0
                else:
                    occupancy = 0.05
                    
    elif zone == "residential":
        if context.semester_state == "exam":
            if hour < 2: # up to 1:59 AM
                occupancy = 0.85
            elif 2 <= hour < 6:
                occupancy = 0.15
            elif 6 <= hour < 9:
                occupancy = 0.6
            elif 9 <= hour < 17:
                occupancy = 0.85 if context.is_weekend else 0.4 # Exam prep during day
            else:
                occupancy = 0.85
        else:
            if context.is_weekend:
                if 6 <= hour < 12:
                    occupancy = 0.5
                elif 12 <= hour < 18:
                    occupancy = 0.6
                elif 18 <= hour < 23:
                    occupancy = 0.95
                elif hour >= 23 or hour < 2:
                    occupancy = 0.7
                else:
                    occupancy = 0.15
            else:
                if 2 <= hour < 6:
                    occupancy = 0.15
                elif 6 <= hour < 9:
                    occupancy = 0.6
                elif 9 <= hour < 17:
                    occupancy = 0.2
                elif 17 <= hour < 23:
                    occupancy = 0.9
                elif hour >= 23 or hour < 2:
                    occupancy = 0.7
                    
    elif zone == "library":
        if context.semester_state == "exam":
            occupancy = 0.98
        elif context.semester_state == "exam_prep":
            if hour < 23:
                occupancy = 0.95
            else:
                occupancy = 0.0
        else:
            if context.is_weekend:
                if 11 <= hour < 18:
                    occupancy = 0.4
                else:
                    occupancy = 0.0
            else:
                if 8 <= hour < 20:
                    occupancy = 0.6
                elif 20 <= hour < 22:
                    occupancy = 0.3
                else:
                    occupancy = 0.0
                    
    elif zone == "administrative":
        if context.is_holiday or context.is_weekend or hour < 8 or hour >= 18:
            occupancy = 0.02
        elif 9 <= hour < 17:
            occupancy = 0.85
        else:
            # 8-9 or 17-18
            occupancy = 0.3
            
    elif zone == "infrastructure": # data center
        if 3 <= hour < 4:
            occupancy = 0.90
        else:
            occupancy = 0.95
            
    # Apply noise (±5%)
    noise = random.gauss(0, 0.05)
    occupancy = max(0.0, min(1.0, occupancy + noise))
    
    # Apply event multiplier
    occupancy = max(0.0, min(1.0, occupancy * event_multiplier))
    
    return occupancy
