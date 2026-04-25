"""
User B Scenario: Emergency Arc (Elderly User)

Simulates an elderly user (age 72) experiencing a potential cardiac event
during the night. This tests the Level 5 emergency detection path.

Arc progression:
  Packets  1-15:  Calm evening baseline (HR ~68, sedentary, 9pm, home, SpO2 98)
  Packet   16:    Emergency onset (HR 134, sedentary, 2:17am, SpO2 93,
                  breathing_rate 22, eda_stress_indicator high)
  Packets 17-18:  Sustained elevated values → triggers Level 5 emergency
"""

from datetime import datetime, timedelta


def generate_packets():
    """Generate 18 telemetry packets for the emergency arc scenario.

    Returns:
        list: 18 packet dicts matching the exact telemetry schema.
    """
    packets = []

    # Evening baseline starts at 9:00 PM on Monday April 20, 2026
    base_time = datetime(2026, 4, 20, 21, 0, 0)

    # Small deterministic variation for baseline packets
    offsets = [0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]

    # ================================================================
    # Packets 1-15: Calm evening baseline
    # Elderly user at home, sedentary, watching TV / preparing for bed
    # ================================================================
    for i in range(15):
        v = offsets[i]
        # Each packet ~3 minutes apart (evening monitoring window)
        timestamp = base_time + timedelta(minutes=i * 3)

        packet = {
            "timestamp": timestamp.isoformat(),
            "user_id": "user_b",
            "vitals": {
                "heart_rate": 68 + v,
                "hrv": 48 + v,
                "spo2": 98,
                "skin_temperature": round(36.4 + v * 0.05, 1),
                "breathing_rate": 16 + (abs(v) % 2),
                "stress_score": 22 + v,
                "recovery_score": 72 + v,
                "eda_stress_indicator": "low"
            },
            "movement": {
                "steps_today": 3200 + v * 100,
                "activity_state": "sedentary",
                "activity_intensity": 0.0,
                "calories_burned": 1400 + v * 20,
                "active_minutes_today": 15 + v
            },
            "sleep_last_night": {
                "total_hours": round(7.0 + v * 0.1, 1),
                "deep_sleep_percentage": 15 + v,
                "rem_percentage": 18 + (abs(v) % 2),
                "light_sleep_percentage": 55 + v,
                "sleep_efficiency": 78 + v,
                "woke_up_times": 2 + (abs(v) % 2),
                "sleep_onset_minutes": 20 + abs(v) * 2
            },
            "context": {
                "time_of_day": "night",
                "day_of_week": "Monday",
                "is_weekend": False,
                "battery_level": 65 - i * 2,
                "location_zone": "home",
                "weather_temp_celsius": 22.0
            },
            "user_reported": {
                "mood": None,
                "stress_level": None,
                "notes": None
            }
        }
        packets.append(packet)

    # ================================================================
    # Packet 16: Emergency onset
    # Sudden HR spike to 134, SpO2 drop to 93, elevated breathing,
    # high EDA — all while sedentary at 2:17 AM
    # ================================================================
    emergency_time = datetime(2026, 4, 21, 2, 17, 0)
    packets.append({
        "timestamp": emergency_time.isoformat(),
        "user_id": "user_b",
        "vitals": {
            "heart_rate": 134,
            "hrv": 22,
            "spo2": 93,
            "skin_temperature": 37.2,
            "breathing_rate": 22,
            "stress_score": 85,
            "recovery_score": 18,
            "eda_stress_indicator": "high"
        },
        "movement": {
            "steps_today": 3200,
            "activity_state": "sedentary",
            "activity_intensity": 0.0,
            "calories_burned": 1400,
            "active_minutes_today": 15
        },
        "sleep_last_night": {
            "total_hours": 7.0,
            "deep_sleep_percentage": 15,
            "rem_percentage": 18,
            "light_sleep_percentage": 55,
            "sleep_efficiency": 78,
            "woke_up_times": 2,
            "sleep_onset_minutes": 20
        },
        "context": {
            "time_of_day": "night",
            "day_of_week": "Tuesday",
            "is_weekend": False,
            "battery_level": 42,
            "location_zone": "home",
            "weather_temp_celsius": 21.0
        },
        "user_reported": {
            "mood": None,
            "stress_level": None,
            "notes": None
        }
    })

    # ================================================================
    # Packets 17-18: Sustained elevated values
    # Confirms this is not a single spurious reading — triggers Level 5
    # ================================================================
    for j in range(2):
        timestamp = emergency_time + timedelta(minutes=(j + 1) * 5)
        packets.append({
            "timestamp": timestamp.isoformat(),
            "user_id": "user_b",
            "vitals": {
                "heart_rate": 131 + j * 2,
                "hrv": 20 - j,
                "spo2": 93,
                "skin_temperature": round(37.3 + j * 0.1, 1),
                "breathing_rate": 23 + j,
                "stress_score": 87 + j * 2,
                "recovery_score": 16 - j,
                "eda_stress_indicator": "high"
            },
            "movement": {
                "steps_today": 3200,
                "activity_state": "sedentary",
                "activity_intensity": 0.0,
                "calories_burned": 1400,
                "active_minutes_today": 15
            },
            "sleep_last_night": {
                "total_hours": 7.0,
                "deep_sleep_percentage": 15,
                "rem_percentage": 18,
                "light_sleep_percentage": 55,
                "sleep_efficiency": 78,
                "woke_up_times": 2,
                "sleep_onset_minutes": 20
            },
            "context": {
                "time_of_day": "night",
                "day_of_week": "Tuesday",
                "is_weekend": False,
                "battery_level": 40 - j,
                "location_zone": "home",
                "weather_temp_celsius": 21.0
            },
            "user_reported": {
                "mood": None,
                "stress_level": None,
                "notes": None
            }
        })

    return packets
