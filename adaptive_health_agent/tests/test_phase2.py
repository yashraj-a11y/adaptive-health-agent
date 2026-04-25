"""
Phase 2 Verification Script

Tests:
1. All 6 knowledge base JSON files load into ChromaDB without error
2. Query "elevated heart rate sedentary night" returns relevant KB entries
3. Living Profile creates, saves, and loads correctly
4. Pattern buffer works correctly
5. Episodic memory log/query/update works
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clean up any previous test data
import shutil
chroma_test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
profiles_test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles")

# Set env vars before importing modules
os.environ["CHROMA_DB_PATH"] = chroma_test_path


def test_knowledge_base_loading():
    """Verify all 6 KB JSON files load into ChromaDB."""
    print("=" * 60)
    print("TEST 1: Knowledge Base Loading")
    print("=" * 60)

    # Verify all JSON files exist
    kb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "knowledge_base", "documents")
    expected_files = [
        "heart_rate_hrv.json",
        "sleep.json",
        "stress_recovery.json",
        "spo2_breathing.json",
        "temperature.json",
        "combined_patterns.json"
    ]

    for fname in expected_files:
        fpath = os.path.join(kb_dir, fname)
        assert os.path.exists(fpath), f"Missing KB file: {fname}"
        with open(fpath) as f:
            data = json.load(f)
        print(f"  ✓ {fname}: {len(data)} entries")

    # Verify minimum counts
    with open(os.path.join(kb_dir, "heart_rate_hrv.json")) as f:
        assert len(json.load(f)) >= 10, "heart_rate_hrv needs 10+ entries"
    with open(os.path.join(kb_dir, "sleep.json")) as f:
        assert len(json.load(f)) >= 10, "sleep needs 10+ entries"
    with open(os.path.join(kb_dir, "stress_recovery.json")) as f:
        assert len(json.load(f)) >= 10, "stress_recovery needs 10+ entries"
    with open(os.path.join(kb_dir, "spo2_breathing.json")) as f:
        assert len(json.load(f)) >= 8, "spo2_breathing needs 8+ entries"
    with open(os.path.join(kb_dir, "temperature.json")) as f:
        assert len(json.load(f)) >= 6, "temperature needs 6+ entries"
    with open(os.path.join(kb_dir, "combined_patterns.json")) as f:
        combined = json.load(f)
        assert len(combined) >= 20, "combined_patterns needs 20+ entries"

    # Verify required named patterns in combined_patterns.json
    required_patterns = [
        "Burnout Trajectory",
        "Stress-Driven Insomnia",
        "Overtraining",
        "Illness Onset Signature",
        "Cardiac Concern Elderly User",
        "Sleep Apnea Indicators",
        "Accumulated Stress Load"
    ]
    titles = [entry["title"] for entry in combined]
    for pattern in required_patterns:
        assert pattern in titles, f"Missing required pattern: {pattern}"
        print(f"  ✓ Required pattern found: {pattern}")

    # Load into ChromaDB
    from memory.episodic_memory import get_knowledge_base_collection
    kb_col = get_knowledge_base_collection()

    # Reset for clean test
    if kb_col.count() > 0:
        all_ids = kb_col.get()["ids"]
        if all_ids:
            kb_col.delete(ids=all_ids)

    from knowledge_base.loader import load_knowledge_base
    load_knowledge_base()

    count = kb_col.count()
    assert count > 0, "KB collection is empty after loading"
    print(f"\n  ✓ Loaded {count} total documents into ChromaDB")

    print("\n  ✅ Knowledge base loading PASSED\n")


def test_kb_query():
    """Verify RAG query returns relevant results."""
    print("=" * 60)
    print("TEST 2: Knowledge Base Query")
    print("=" * 60)

    from knowledge_base.loader import query_knowledge_base

    results = query_knowledge_base("elevated heart rate sedentary night", n_results=3)
    assert len(results) > 0, "No results returned for KB query"

    print(f"  Query: 'elevated heart rate sedentary night'")
    print(f"  Results returned: {len(results)}")
    for i, r in enumerate(results):
        title = r["metadata"].get("title", "unknown")
        severity = r["metadata"].get("severity_suggestion", "?")
        distance = r.get("distance", "?")
        print(f"    {i+1}. {title} (severity: {severity}, distance: {distance:.4f})")

    # The top result should be related to HR/cardiac/nocturnal
    top_title = results[0]["metadata"].get("title", "").lower()
    assert any(kw in top_title for kw in ["heart", "cardiac", "nocturnal", "hr", "tachycardia", "night"]), \
        f"Top result doesn't seem relevant: {top_title}"
    print(f"\n  ✓ Top result is relevant: {results[0]['metadata']['title']}")

    print("\n  ✅ KB query PASSED\n")


def test_living_profile():
    """Verify Living Profile create/save/load cycle."""
    print("=" * 60)
    print("TEST 3: Living Profile CRUD")
    print("=" * 60)

    from memory.living_profile import (
        create_profile, load_profile, save_profile,
        update_baselines, update_current_state,
        update_communication_profile, add_concern,
        resolve_concern, add_pattern
    )

    test_user_data = {
        "user_id": "test_user_phase2",
        "name": "Test User",
        "age": 35,
        "known_conditions": ["mild asthma"],
        "medications": [],
        "goals": ["manage_stress"],
        "communication_style": "balanced",
        "directness": 3,
        "depth": 3,
        "tone": 3,
        "length": 3,
        "framing": 3,
        "alert_sensitivity": "normal",
        "emergency_contact": {"name": "Jane", "contact": "555-1234"},
        "best_engagement_times": [],
        "engagement_patterns": "unknown"
    }

    # Create
    profile = create_profile(test_user_data)
    assert profile is not None
    assert profile["identity"]["name"] == "Test User"
    assert profile["identity"]["age"] == 35
    assert profile["baselines"]["status"] == "LEARNING"
    assert profile["identity"]["emergency_contact"]["name"] == "Jane"
    print("  ✓ create_profile() works")

    # Load
    loaded = load_profile("test_user_phase2")
    assert loaded is not None
    assert loaded["identity"]["name"] == "Test User"
    print("  ✓ load_profile() works")

    # Update baselines
    update_baselines("test_user_phase2", {"resting_hr": 68.0, "typical_hrv": 55.0})
    loaded = load_profile("test_user_phase2")
    assert loaded["baselines"]["resting_hr"] == 68.0
    assert loaded["baselines"]["typical_hrv"] == 55.0
    print("  ✓ update_baselines() works")

    # Update current state
    update_current_state("test_user_phase2", {"sleep_trend_7d": "declining", "stress_trend_7d": "rising"})
    loaded = load_profile("test_user_phase2")
    assert loaded["current_state"]["sleep_trend_7d"] == "declining"
    print("  ✓ update_current_state() works")

    # Update communication profile
    update_communication_profile("test_user_phase2", {"directness": 4, "tone": 5})
    loaded = load_profile("test_user_phase2")
    assert loaded["communication_profile"]["directness"] == 4
    assert loaded["communication_profile"]["tone"] == 5
    print("  ✓ update_communication_profile() works")

    # Test clamping
    update_communication_profile("test_user_phase2", {"directness": 10})
    loaded = load_profile("test_user_phase2")
    assert loaded["communication_profile"]["directness"] == 5, "Should be clamped to 5"
    print("  ✓ Communication profile clamping works (10 → 5)")

    # Add concern
    add_concern("test_user_phase2", "elevated stress pattern")
    loaded = load_profile("test_user_phase2")
    assert "elevated stress pattern" in loaded["current_concerns"]
    print("  ✓ add_concern() works")

    # Resolve concern
    resolve_concern("test_user_phase2", "elevated stress pattern")
    loaded = load_profile("test_user_phase2")
    assert "elevated stress pattern" not in loaded["current_concerns"]
    print("  ✓ resolve_concern() works")

    # Add pattern
    add_pattern("test_user_phase2", "stress-sleep feedback loop")
    loaded = load_profile("test_user_phase2")
    assert "stress-sleep feedback loop" in loaded["known_patterns"]
    print("  ✓ add_pattern() works")

    # Verify full structure
    required_keys = ["identity", "baselines", "current_state", "known_patterns",
                     "communication_profile", "current_concerns", "last_updated", "days_monitored"]
    for key in required_keys:
        assert key in loaded, f"Missing key: {key}"
    print("  ✓ Full Living Profile structure verified")

    # Cleanup
    test_profile_path = os.path.join(profiles_test_path, "test_user_phase2.json")
    if os.path.exists(test_profile_path):
        os.remove(test_profile_path)

    print("\n  ✅ Living Profile PASSED\n")


def test_pattern_buffer():
    """Verify PatternBuffer operations."""
    print("=" * 60)
    print("TEST 4: Pattern Buffer")
    print("=" * 60)

    from utils.pattern_buffer import PatternBuffer

    buf = PatternBuffer()

    # Initial state
    assert buf.get_count("hr_elevated") == 0
    assert not buf.is_confirmed("hr_elevated")
    assert buf.get_confirmed_patterns() == []
    print("  ✓ Initial state: all counters at 0, no confirmed patterns")

    # Increment below threshold
    buf.increment("hr_elevated")
    buf.increment("hr_elevated")
    assert buf.get_count("hr_elevated") == 2
    assert not buf.is_confirmed("hr_elevated")
    print("  ✓ After 2 increments: count=2, not confirmed")

    # Increment to threshold
    buf.increment("hr_elevated")
    assert buf.get_count("hr_elevated") == 3
    assert buf.is_confirmed("hr_elevated")
    assert "hr_elevated" in buf.get_confirmed_patterns()
    print("  ✓ After 3 increments: count=3, CONFIRMED")

    # Multiple confirmed patterns
    for _ in range(3):
        buf.increment("stress_elevated")
    assert "stress_elevated" in buf.get_confirmed_patterns()
    assert len(buf.get_confirmed_patterns()) == 2
    print("  ✓ Multiple confirmed patterns: hr_elevated + stress_elevated")

    # Reset confirmed
    buf.reset_confirmed("hr_elevated")
    assert buf.get_count("hr_elevated") == 0
    assert not buf.is_confirmed("hr_elevated")
    print("  ✓ reset_confirmed() resets counter to 0")

    # Reset single metric
    buf.reset("stress_elevated")
    assert buf.get_count("stress_elevated") == 0
    print("  ✓ reset() works")

    # Unknown metric
    buf.increment("unknown_metric")  # Should print warning, not crash
    print("  ✓ Unknown metric handled gracefully")

    print("\n  ✅ Pattern buffer PASSED\n")


def test_episodic_memory_functions():
    """Verify episodic memory log/query/update cycle."""
    print("=" * 60)
    print("TEST 5: Episodic Memory Functions")
    print("=" * 60)

    from memory.episodic_memory import (
        log_episode, query_similar, get_recent,
        update_episode, episodic_collection
    )
    from datetime import datetime

    # Clean up previous test episodes
    try:
        existing = episodic_collection.get(where={"user_id": "test_ep_user"})
        if existing["ids"]:
            episodic_collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Log a test episode
    test_episode = {
        "id": f"episode_{datetime.now().strftime('%Y%m%d%H%M%S')}_test_ep_user",
        "timestamp": datetime.now().isoformat(),
        "user_id": "test_ep_user",
        "event_type": "anomaly",
        "metrics_snapshot": {
            "heart_rate": 95,
            "hrv": 32,
            "stress_score": 72,
            "spo2": 96
        },
        "context_snapshot": {
            "time_of_day": "night",
            "activity_state": "sedentary",
            "location_zone": "home"
        },
        "deviation_from_baseline": {
            "heart_rate": "+39.7%",
            "hrv": "-41.8%",
            "stress_score": "+188%"
        },
        "significance": "single_occurrence",
        "agent_action_taken": "logged anomaly for monitoring",
        "user_response": None,
        "outcome": None,
        "tags": ["hr_elevated", "hrv_low", "stress_elevated", "nocturnal"]
    }

    log_episode(test_episode)
    print("  ✓ log_episode() completed")

    # Query similar
    results = query_similar("elevated heart rate low HRV stress at night", n_results=1,
                            where={"user_id": "test_ep_user"})
    assert len(results) > 0, "query_similar returned no results"
    assert results[0]["metadata"]["user_id"] == "test_ep_user"
    print(f"  ✓ query_similar() found match (distance: {results[0]['distance']:.4f})")

    # Get recent
    recent = get_recent("test_ep_user", days=1)
    assert len(recent) > 0, "get_recent returned no results"
    print(f"  ✓ get_recent() returned {len(recent)} episode(s)")

    # Update episode
    update_episode(test_episode["id"], {
        "significance": "pattern_confirmed",
        "outcome": "user notified"
    })
    updated = episodic_collection.get(ids=[test_episode["id"]], include=["metadatas"])
    assert updated["metadatas"][0]["significance"] == "pattern_confirmed"
    assert updated["metadatas"][0]["outcome"] == "user notified"
    print("  ✓ update_episode() modified metadata correctly")

    # Cleanup
    episodic_collection.delete(ids=[test_episode["id"]])
    print("  ✓ Test episode cleaned up")

    print("\n  ✅ Episodic memory functions PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 2 VERIFICATION — Memory Layer")
    print("=" * 60 + "\n")

    test_knowledge_base_loading()
    test_kb_query()
    test_living_profile()
    test_pattern_buffer()
    test_episodic_memory_functions()

    print("=" * 60)
    print("  ALL PHASE 2 TESTS PASSED ✅")
    print("=" * 60)
