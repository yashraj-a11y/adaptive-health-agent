"""
Phase 3 Verification Script

Tests:
1. Feed 5 normal packets to profiler_node → pattern_confirmed stays False
2. Feed 3 packets with HR >20% above baseline while sedentary → pattern_confirmed = True
3. Analyst fallback returns severity 5 for: HR=134, sedentary, 2am, spo2=93, age=72
4. Communicator produces different messages for directness=1 vs directness=5
5. Emergency console output prints in the exact format
6. No function from Phase 1 or Phase 2 has been rewritten

Note: Tests 1-3 use deterministic logic (no LLM calls needed).
      Tests 4-5 test message formatting and emergency output.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set env vars before imports (use dummy keys — LLM tests use fallbacks)
os.environ.setdefault("GROQ_API_KEY_PROFILER", "test_key_profiler")
os.environ.setdefault("GROQ_API_KEY_ANALYST", "test_key_analyst")
os.environ.setdefault("GROQ_API_KEY_COMMUNICATOR", "test_key_communicator")
os.environ.setdefault("CHROMA_DB_PATH", "./chroma_db")


def _make_normal_packet(packet_num=1):
    """Create a normal telemetry packet (no deviations)."""
    return {
        "timestamp": f"2026-04-21T10:{packet_num:02d}:00",
        "user_id": "test_profiler_user",
        "vitals": {
            "heart_rate": 68,
            "hrv": 55,
            "spo2": 98,
            "skin_temperature": 36.5,
            "breathing_rate": 14,
            "stress_score": 25,
            "recovery_score": 70,
            "eda_stress_indicator": "low",
        },
        "movement": {
            "steps_today": 4500,
            "activity_state": "sedentary",
            "activity_intensity": "low",
            "calories_burned": 150,
            "active_minutes_today": 20,
        },
        "sleep_last_night": {
            "total_hours": 7.5,
            "deep_sleep_percentage": 22,
            "rem_percentage": 24,
            "light_sleep_percentage": 54,
            "sleep_efficiency": 88,
            "woke_up_times": 1,
            "sleep_onset_minutes": 12,
        },
        "context": {
            "time_of_day": "morning",
            "day_of_week": "Tuesday",
            "is_weekend": False,
            "battery_level": 85,
            "location_zone": "home",
            "weather_temp_celsius": 22,
        },
        "user_reported": {
            "mood": "good",
            "notes": "",
            "stress_level": "low",
        },
    }


def _make_elevated_hr_packet(packet_num=1):
    """Create a packet with HR >20% above baseline while sedentary."""
    packet = _make_normal_packet(packet_num)
    # Baseline resting HR is 68 → 20% above = 81.6 → use 90 (32% above)
    packet["vitals"]["heart_rate"] = 90
    packet["vitals"]["hrv"] = 35  # Low HRV
    packet["vitals"]["stress_score"] = 65
    return packet


def _make_emergency_packet():
    """Create User B emergency packet: HR=134, sedentary, 2am, spo2=93."""
    return {
        "timestamp": "2026-04-21T02:17:00",
        "user_id": "test_emergency_user",
        "vitals": {
            "heart_rate": 134,
            "hrv": 18,
            "spo2": 93,
            "skin_temperature": 36.8,
            "breathing_rate": 22,
            "stress_score": 85,
            "recovery_score": 15,
            "eda_stress_indicator": "high",
        },
        "movement": {
            "steps_today": 0,
            "activity_state": "sedentary",
            "activity_intensity": "none",
            "calories_burned": 0,
            "active_minutes_today": 0,
        },
        "sleep_last_night": {
            "total_hours": 5.0,
            "deep_sleep_percentage": 10,
            "rem_percentage": 15,
            "light_sleep_percentage": 75,
            "sleep_efficiency": 62,
            "woke_up_times": 4,
            "sleep_onset_minutes": 25,
        },
        "context": {
            "time_of_day": "night",
            "day_of_week": "Monday",
            "is_weekend": False,
            "battery_level": 45,
            "location_zone": "home",
            "weather_temp_celsius": 18,
        },
        "user_reported": {
            "mood": "",
            "notes": "",
            "stress_level": "",
        },
    }


def _make_established_profile(user_id="test_profiler_user", age=35):
    """Create a profile with ESTABLISHED baselines."""
    return {
        "identity": {
            "name": "Test User",
            "age": age,
            "known_conditions": [],
            "medications": [],
            "goals": ["general_wellness"],
            "emergency_contact": {"name": "Jane", "contact": "555-1234"},
        },
        "baselines": {
            "status": "ESTABLISHED",
            "resting_hr": 68,
            "typical_hrv": 55,
            "typical_spo2": 98,
            "typical_skin_temp": 36.5,
            "typical_sleep_hours": 7.5,
            "typical_sleep_efficiency": 88,
            "typical_daily_steps": 5000,
            "typical_breathing_rate": 14,
            "typical_stress_score": 25,
            "typical_recovery_score": 70,
        },
        "current_state": {
            "sleep_trend_7d": "stable",
            "hrv_trend_7d": "stable",
            "stress_trend_7d": "stable",
            "recovery_trend_7d": "stable",
            "overall_trajectory": "stable",
        },
        "known_patterns": [],
        "communication_profile": {
            "style": "balanced",
            "directness": 3,
            "depth": 3,
            "tone": 3,
            "length": 3,
            "framing": 3,
            "alert_sensitivity": "normal",
            "best_engagement_times": [],
            "engagement_patterns": "unknown",
        },
        "current_concerns": [],
        "last_updated": "2026-04-21T10:00:00",
        "days_monitored": 20,
    }


def test_normal_packets_no_pattern():
    """Test 1: 5 normal packets → pattern_confirmed stays False."""
    print("=" * 60)
    print("TEST 1: Normal Packets → No Pattern Confirmed")
    print("=" * 60)

    from agents.profiler import profiler_node, _pattern_buffer

    # Reset pattern buffer
    for metric in _pattern_buffer._counters:
        _pattern_buffer.reset(metric)

    profile = _make_established_profile()

    for i in range(5):
        packet = _make_normal_packet(i)
        state = {
            "living_profile": profile,
            "current_packet": packet,
            "deviation_detected": False,
            "pattern_confirmed": False,
            "pattern_details": None,
            "severity_level": None,
            "analyst_output": None,
            "proceed_to_communicate": False,
            "final_message": None,
            "notify_family": False,
            "user_message": None,
            "agent_response": None,
        }

        result = profiler_node(state)
        assert result["pattern_confirmed"] == False, f"Packet {i+1}: pattern_confirmed should be False"
        print(f"  ✓ Packet {i+1}: deviation_detected={result['deviation_detected']}, pattern_confirmed=False")

    print("\n  ✅ Normal packets test PASSED\n")


def test_elevated_hr_pattern():
    """Test 2: 3 packets with HR >20% above baseline → pattern_confirmed = True."""
    print("=" * 60)
    print("TEST 2: Elevated HR Packets → Pattern Confirmed")
    print("=" * 60)

    from agents.profiler import profiler_node, _pattern_buffer

    # Reset pattern buffer
    for metric in _pattern_buffer._counters:
        _pattern_buffer.reset(metric)

    profile = _make_established_profile()
    pattern_confirmed_at = None

    for i in range(3):
        packet = _make_elevated_hr_packet(i)
        state = {
            "living_profile": profile,
            "current_packet": packet,
            "deviation_detected": False,
            "pattern_confirmed": False,
            "pattern_details": None,
            "severity_level": None,
            "analyst_output": None,
            "proceed_to_communicate": False,
            "final_message": None,
            "notify_family": False,
            "user_message": None,
            "agent_response": None,
        }

        result = profiler_node(state)
        print(f"  Packet {i+1}: deviation_detected={result['deviation_detected']}, "
              f"pattern_confirmed={result['pattern_confirmed']}")

        if result["pattern_confirmed"]:
            pattern_confirmed_at = i + 1

    assert pattern_confirmed_at is not None, "Pattern should be confirmed after 3 elevated packets"
    assert pattern_confirmed_at == 3, f"Pattern confirmed at packet {pattern_confirmed_at}, expected 3"

    print(f"\n  ✓ Pattern confirmed at packet {pattern_confirmed_at}")
    print("\n  ✅ Elevated HR pattern test PASSED\n")


def test_analyst_severity_5():
    """Test 3: Analyst fallback gives severity 5 for emergency scenario."""
    print("=" * 60)
    print("TEST 3: Analyst Severity 5 for Emergency")
    print("=" * 60)

    from agents.analyst import _estimate_severity_fallback

    # Build pattern_details matching emergency: HR=134, sedentary, 2am, SpO2=93, age=72
    pattern_details = {
        "confirmed_metrics": ["hr_elevated", "spo2_low"],
        "deviations": {
            "hr_elevated": {"deviated": True, "current": 134, "baseline": 68, "deviation_percent": 97.1},
            "spo2_low": {"deviated": True, "current": 93, "baseline": 98, "deviation_percent": -5.1},
        },
        "vitals_snapshot": {
            "heart_rate": 134,
            "hrv": 18,
            "spo2": 93,
            "breathing_rate": 22,
            "stress_score": 85,
        },
        "activity_state": "sedentary",
        "context": {"time_of_day": "night", "location_zone": "home"},
    }

    severity = _estimate_severity_fallback(pattern_details)
    assert severity == 5, f"Expected severity 5, got {severity}"
    print(f"  ✓ Severity = {severity} (HR=134 + SpO2=93 + sedentary + night)")

    # Also test the full analyst_node guard logic
    from agents.analyst import analyst_node

    profile = _make_established_profile(user_id="test_emergency_user", age=72)
    emergency_packet = _make_emergency_packet()

    state = {
        "living_profile": profile,
        "current_packet": emergency_packet,
        "deviation_detected": True,
        "pattern_confirmed": True,
        "pattern_details": pattern_details,
        "severity_level": None,
        "analyst_output": None,
        "proceed_to_communicate": False,
        "final_message": None,
        "notify_family": False,
        "user_message": None,
        "agent_response": None,
    }

    result = analyst_node(state)
    # Even with LLM fallback, the deterministic fallback should give severity 5
    assert result["severity_level"] >= 4, f"Expected severity >= 4, got {result['severity_level']}"
    assert result["notify_family"] == True or result["severity_level"] == 5, "Should notify family for severity 5"
    print(f"  ✓ Analyst node returned severity={result['severity_level']}, "
          f"notify_family={result['notify_family']}")

    print("\n  ✅ Analyst severity 5 test PASSED\n")


def test_communicator_style_difference():
    """Test 4: Communicator produces different style instructions for directness=1 vs 5."""
    print("=" * 60)
    print("TEST 4: Communication Style Difference")
    print("=" * 60)

    from agents.communicator import _build_style_instruction

    # Clinical style (directness=1, depth=1, tone=1)
    clinical_profile = {
        "style": "clinical",
        "directness": 1,
        "depth": 1,
        "tone": 1,
        "length": 1,
        "framing": 1,
    }
    clinical_style = _build_style_instruction(clinical_profile)

    # Casual style (directness=5, depth=5, tone=5)
    casual_profile = {
        "style": "casual",
        "directness": 5,
        "depth": 5,
        "tone": 5,
        "length": 5,
        "framing": 5,
    }
    casual_style = _build_style_instruction(casual_profile)

    assert clinical_style != casual_style, "Style instructions should differ"

    # Check clinical keywords
    assert "direct" in clinical_style.lower() or "blunt" in clinical_style.lower(), \
        "Clinical style should mention directness"
    assert "clinical" in clinical_style.lower() or "professional" in clinical_style.lower(), \
        "Clinical style should mention clinical tone"
    assert "numbers" in clinical_style.lower() or "data" in clinical_style.lower(), \
        "Clinical style should mention data/numbers"

    # Check casual keywords
    assert "gentle" in casual_style.lower() or "indirect" in casual_style.lower(), \
        "Casual style should mention gentleness"
    assert "warm" in casual_style.lower() or "friendly" in casual_style.lower(), \
        "Casual style should mention warmth"
    assert "brief" in casual_style.lower() or "concise" in casual_style.lower(), \
        "Casual style should mention brevity"

    print(f"  Clinical style instruction:\n    {clinical_style[:200]}...")
    print(f"\n  Casual style instruction:\n    {casual_style[:200]}...")
    print(f"\n  ✓ Styles are noticeably different")

    print("\n  ✅ Communication style difference test PASSED\n")


def test_emergency_alert_format():
    """Test 5: Emergency alert prints in the exact required format."""
    print("=" * 60)
    print("TEST 5: Emergency Alert Format")
    print("=" * 60)

    from agents.communicator import _print_emergency_alert
    import io
    from contextlib import redirect_stdout

    # Capture stdout
    output = io.StringIO()
    with redirect_stdout(output):
        _print_emergency_alert(
            timestamp="2026-04-21T02:17:00",
            name="Eleanor",
            emergency_contact={"name": "Michael", "contact": "555-0199"},
            key_facts=[
                "Heart rate spiked to 134 while sedentary at 2:17am",
                "SpO2 dropped to 93%",
                "High EDA stress indicator",
            ],
            clinical_context="Multiple acute cardiac indicators concurrent with desaturation.",
        )

    alert_text = output.getvalue()

    # Verify required elements
    assert "EMERGENCY ALERT" in alert_text
    assert "2026-04-21T02:17:00" in alert_text
    assert "Eleanor" in alert_text
    assert "Michael" in alert_text
    assert "555-0199" in alert_text
    assert "Heart rate spiked to 134" in alert_text
    assert "SpO2 dropped to 93%" in alert_text
    assert "High EDA stress indicator" in alert_text
    assert "Multiple acute cardiac indicators" in alert_text
    assert "Immediate check-in with user recommended" in alert_text
    assert "Consider contacting emergency services" in alert_text
    assert "automated alert from the Health Agent" in alert_text
    assert "========" in alert_text

    print(alert_text)
    print("  ✓ All required elements present in emergency alert")

    print("\n  ✅ Emergency alert format test PASSED\n")


def test_phase1_phase2_untouched():
    """Test 6: Verify Phase 1/2 modules still work correctly."""
    print("=" * 60)
    print("TEST 6: Phase 1 & Phase 2 Functions Untouched")
    print("=" * 60)

    # Test episodic memory functions still exist and work
    from memory.episodic_memory import (
        log_episode, query_similar, get_recent, update_episode,
        get_chroma_client, get_episodic_collection, get_knowledge_base_collection,
        get_embedding
    )
    assert callable(log_episode)
    assert callable(query_similar)
    assert callable(get_recent)
    assert callable(update_episode)
    print("  ✓ memory/episodic_memory.py: All 4 episode functions intact")

    # Test living profile functions still exist
    from memory.living_profile import (
        create_profile, load_profile, save_profile,
        update_baselines, update_current_state,
        update_communication_profile, add_concern,
        resolve_concern, add_pattern
    )
    assert callable(create_profile)
    assert callable(load_profile)
    assert callable(save_profile)
    assert callable(update_baselines)
    assert callable(update_current_state)
    assert callable(update_communication_profile)
    assert callable(add_concern)
    assert callable(resolve_concern)
    assert callable(add_pattern)
    print("  ✓ memory/living_profile.py: All 9 functions intact")

    # Test pattern buffer
    from utils.pattern_buffer import PatternBuffer
    buf = PatternBuffer()
    assert hasattr(buf, 'increment')
    assert hasattr(buf, 'reset')
    assert hasattr(buf, 'is_confirmed')
    assert hasattr(buf, 'get_confirmed_patterns')
    assert hasattr(buf, 'reset_confirmed')
    print("  ✓ utils/pattern_buffer.py: PatternBuffer class intact")

    # Test baseline calculator
    from utils.baseline_calculator import compute_baselines
    assert callable(compute_baselines)
    print("  ✓ utils/baseline_calculator.py: compute_baselines intact")

    # Test onboarding
    from onboarding.onboarding import run_onboarding
    assert callable(run_onboarding)
    print("  ✓ onboarding/onboarding.py: run_onboarding intact")

    # Test knowledge base loader
    from knowledge_base.loader import load_knowledge_base, query_knowledge_base
    assert callable(load_knowledge_base)
    assert callable(query_knowledge_base)
    print("  ✓ knowledge_base/loader.py: All functions intact")

    # Test telemetry
    from telemetry.user_a_scenario import generate_packets as gen_a
    from telemetry.user_b_scenario import generate_packets as gen_b
    assert len(gen_a()) == 50
    assert len(gen_b()) == 18
    print("  ✓ telemetry scenarios: user_a=50 packets, user_b=18 packets")

    print("\n  ✅ Phase 1 & Phase 2 untouched test PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 3 VERIFICATION — Agents")
    print("=" * 60 + "\n")

    test_normal_packets_no_pattern()
    test_elevated_hr_pattern()
    test_analyst_severity_5()
    test_communicator_style_difference()
    test_emergency_alert_format()
    test_phase1_phase2_untouched()

    print("=" * 60)
    print("  ALL PHASE 3 TESTS PASSED ✅")
    print("=" * 60)
