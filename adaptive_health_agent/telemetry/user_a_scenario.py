"""
User A Scenario: Stress Accumulation Arc

Simulates a working professional (~35 years old) experiencing gradual
stress buildup over approximately 10 days.

Arc progression:
  Packets  1-10:  Normal baseline (HR ~68, HRV ~55, stress ~25, sleep 7.5hrs)
  Packets 11-20:  Sleep declining to ~6.0hrs, HRV dropping to ~42, stress rising to ~45
  Packets 21-35:  Stress 60-70, recovery 35-45, sleep_efficiency 65-72, steps decreasing
  Packets 36-40:  User reports "work has been intense", stress_level = "high"
  Packets 41-50:  stress_score >75 sustained for 3+ consecutive → triggers pattern confirmed
"""

from datetime import datetime, timedelta


def generate_packets():
    """Generate 50 telemetry packets for the stress accumulation scenario.

    Returns:
        list: 50 packet dicts matching the exact telemetry schema.
    """
    packets = []
    # Start on Monday April 13, 2026 at 7:00 AM
    base_time = datetime(2026, 4, 13, 7, 0, 0)

    # Deterministic variation offsets — cycles for all 50 packets
    offsets = [0, 1, -1, 2, -2, 1, 0, -1, 2, -1]

    for i in range(50):
        pkt_num = i + 1
        v = offsets[i % 10]

        # Each packet is ~5 hours apart (~4.8 readings/day, ~10.4 days total)
        timestamp = base_time + timedelta(hours=i * 5)

        # Derive time-of-day context from timestamp hour
        hour = timestamp.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        day_of_week = timestamp.strftime("%A")
        is_weekend = day_of_week in ("Saturday", "Sunday")

        # ================================================================
        # Phase 1 (Packets 1-10): Normal healthy baseline
        # ================================================================
        if pkt_num <= 10:
            hr = 68 + v
            hrv = 55 + v
            spo2 = 98 + (abs(v) % 2)
            skin_temp = round(36.5 + v * 0.1, 1)
            breathing_rate = 15 + (abs(v) % 2)
            stress = 25 + v
            recovery = 78 + v
            eda = "low"
            steps = 8000 + v * 200
            activity = ["sedentary", "walking", "sedentary", "active", "walking",
                        "sedentary", "walking", "sedentary", "active", "walking"][i]
            intensity = [0.1, 0.4, 0.1, 0.7, 0.4, 0.1, 0.4, 0.1, 0.7, 0.4][i]
            calories = 1800 + v * 50
            active_mins = 35 + v * 2
            sleep_hrs = round(7.5 + v * 0.1, 1)
            deep_pct = 22 + v
            rem_pct = 23 + (abs(v) % 2)
            light_pct = 48 - v
            sleep_eff = 87 + v
            woke_times = abs(v) % 2
            sleep_onset = 12 + abs(v)
            mood = None
            stress_lvl = None
            notes = None

        # ================================================================
        # Phase 2 (Packets 11-20): Gradual decline
        # Sleep: 7.5 → 6.0, HRV: 55 → 42, Stress: 25 → 45
        # ================================================================
        elif pkt_num <= 20:
            t = (pkt_num - 10) / 10  # 0.1 → 1.0
            hr = round(68 + t * 5 + v * 0.5)
            hrv = round(55 - t * 13 + v * 0.3)
            spo2 = 97 + (abs(v) % 2)
            skin_temp = round(36.5 + t * 0.2 + v * 0.05, 1)
            breathing_rate = round(15 + t * 2 + abs(v) * 0.3)
            stress = round(25 + t * 20 + v)
            recovery = round(78 - t * 20 + v)
            eda = "low" if t < 0.5 else "moderate"
            steps = round(8000 - t * 1500 + v * 100)
            activity = "sedentary" if t > 0.6 else "walking"
            intensity = round(max(0.1, 0.4 - t * 0.2), 1)
            calories = round(1800 - t * 200 + v * 20)
            active_mins = round(35 - t * 10 + v)
            sleep_hrs = round(7.5 - t * 1.5, 1)
            deep_pct = round(22 - t * 5)
            rem_pct = round(23 - t * 3)
            light_pct = round(48 + t * 5)
            sleep_eff = round(87 - t * 9)
            woke_times = 1 + (1 if t > 0.5 else 0)
            sleep_onset = round(12 + t * 10)
            mood = None
            stress_lvl = None
            notes = None

        # ================================================================
        # Phase 3 (Packets 21-35): Sustained high stress
        # Stress: 60-70, Recovery: 35-45, Sleep efficiency: 65-72
        # ================================================================
        elif pkt_num <= 35:
            t = (pkt_num - 20) / 15  # 0.067 → 1.0
            hr = round(74 + t * 6 + v * 0.5)
            hrv = round(42 - t * 7 + v * 0.3)
            spo2 = 96 + (abs(v) % 2)
            skin_temp = round(36.7 + t * 0.3 + v * 0.05, 1)
            breathing_rate = round(17 + t * 2 + abs(v) * 0.2)
            stress = round(60 + t * 10 + v)
            recovery = round(45 - t * 10 + v)
            eda = "moderate" if t < 0.5 else "high"
            steps = round(6500 - t * 2000 + v * 100)
            activity = "sedentary"
            intensity = 0.1
            calories = round(1600 - t * 200 + v * 20)
            active_mins = round(max(5, 25 - t * 10))
            sleep_hrs = round(6.0 - t * 0.5, 1)
            deep_pct = round(17 - t * 3)
            rem_pct = round(20 - t * 2)
            light_pct = round(53 + t * 5)
            sleep_eff = round(72 - t * 7)
            woke_times = 2 + (1 if t > 0.5 else 0)
            sleep_onset = round(22 + t * 8)
            mood = None
            stress_lvl = None
            notes = None

        # ================================================================
        # Phase 4 (Packets 36-40): User self-reports work stress
        # ================================================================
        elif pkt_num <= 40:
            t = (pkt_num - 35) / 5  # 0.2 → 1.0
            hr = round(78 + t * 3 + v * 0.3)
            hrv = round(36 - t * 3 + v * 0.2)
            spo2 = 96
            skin_temp = round(37.0 + v * 0.05, 1)
            breathing_rate = round(19 + abs(v) * 0.3)
            stress = round(68 + t * 7 + v)
            recovery = round(35 - t * 5 + v)
            eda = "high"
            steps = round(4500 - t * 500 + v * 50)
            activity = "sedentary"
            intensity = 0.1
            calories = round(1400 + v * 10)
            active_mins = round(max(5, 15 - t * 3))
            sleep_hrs = round(5.5 - t * 0.3, 1)
            deep_pct = round(14 - t * 2)
            rem_pct = round(18 - t)
            light_pct = round(58 + t * 2)
            sleep_eff = round(65 - t * 5)
            woke_times = 3
            sleep_onset = round(30 + t * 5)
            mood = None
            stress_lvl = "high"
            notes = "work has been intense"

        # ================================================================
        # Phase 5 (Packets 41-50): Sustained stress >75 → pattern confirmed
        # All stress_score values are guaranteed >75 for 10 consecutive packets
        # ================================================================
        else:
            t = (pkt_num - 40) / 10  # 0.1 → 1.0
            hr = round(80 + t * 4 + v * 0.3)
            hrv = round(33 - t * 3 + v * 0.2)
            spo2 = 96
            skin_temp = round(37.1 + t * 0.1 + v * 0.03, 1)
            breathing_rate = round(19 + t + abs(v) * 0.2)
            # stress_score >75 sustained: base 76 + progression + abs(variation)
            stress = round(76 + t * 6 + abs(v))
            recovery = round(30 - t * 5 + v * 0.5)
            eda = "high"
            steps = round(4000 - t * 500 + v * 30)
            activity = "sedentary"
            intensity = 0.1
            calories = round(1350 + v * 10)
            active_mins = round(max(2, 12 - t * 5))
            sleep_hrs = round(5.2 - t * 0.2, 1)
            deep_pct = round(12 - t * 2)
            rem_pct = round(17 - t)
            light_pct = round(62 + t * 2)
            sleep_eff = round(62 - t * 5)
            woke_times = 3 + (1 if t > 0.5 else 0)
            sleep_onset = round(35 + t * 5)
            mood = None
            stress_lvl = "high"
            notes = "work has been intense" if pkt_num <= 43 else None

        # ================================================================
        # Clamp all values to realistic physiological ranges
        # ================================================================
        hr = max(55, min(120, hr))
        hrv = max(15, min(80, hrv))
        spo2 = max(90, min(100, spo2))
        skin_temp = max(35.5, min(38.5, round(skin_temp, 1)))
        breathing_rate = max(12, min(25, breathing_rate))
        stress = max(10, min(100, stress))
        recovery = max(15, min(95, recovery))
        steps = max(1000, min(15000, steps))
        intensity = max(0.0, min(1.0, intensity))
        calories = max(800, min(3000, calories))
        active_mins = max(0, min(120, active_mins))
        sleep_hrs = max(3.0, min(10.0, round(sleep_hrs, 1)))
        deep_pct = max(5, min(30, deep_pct))
        rem_pct = max(10, min(30, rem_pct))
        light_pct = max(35, min(70, light_pct))
        sleep_eff = max(40, min(98, sleep_eff))
        woke_times = max(0, min(8, woke_times))
        sleep_onset = max(5, min(60, sleep_onset))

        # Derive location from time and day context
        if time_of_day == "night" or (time_of_day == "morning" and hour < 9):
            location = "home"
        elif is_weekend:
            location = "home"
        else:
            location = "work"

        # Battery decreases through the day, recharges overnight
        if time_of_day == "morning":
            battery = 95 - abs(v) * 3
        elif time_of_day == "afternoon":
            battery = 75 - abs(v) * 5
        elif time_of_day == "evening":
            battery = 50 - abs(v) * 5
        else:
            battery = 35 - abs(v) * 3
        battery = max(15, min(100, battery))

        # Stable weather with minor variation
        weather = round(24 + v * 0.5, 1)

        packet = {
            "timestamp": timestamp.isoformat(),
            "user_id": "user_a",
            "vitals": {
                "heart_rate": hr,
                "hrv": hrv,
                "spo2": spo2,
                "skin_temperature": skin_temp,
                "breathing_rate": breathing_rate,
                "stress_score": stress,
                "recovery_score": recovery,
                "eda_stress_indicator": eda
            },
            "movement": {
                "steps_today": steps,
                "activity_state": activity,
                "activity_intensity": intensity,
                "calories_burned": calories,
                "active_minutes_today": active_mins
            },
            "sleep_last_night": {
                "total_hours": sleep_hrs,
                "deep_sleep_percentage": deep_pct,
                "rem_percentage": rem_pct,
                "light_sleep_percentage": light_pct,
                "sleep_efficiency": sleep_eff,
                "woke_up_times": woke_times,
                "sleep_onset_minutes": sleep_onset
            },
            "context": {
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "battery_level": battery,
                "location_zone": location,
                "weather_temp_celsius": weather
            },
            "user_reported": {
                "mood": mood,
                "stress_level": stress_lvl,
                "notes": notes
            }
        }

        packets.append(packet)

    return packets
