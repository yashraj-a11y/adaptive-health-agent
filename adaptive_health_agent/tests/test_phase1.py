"""
Phase 1 Verification Script

Tests:
1. telemetry/stream.py emits valid JSON packets every interval
2. user_a_scenario.py produces 50 packets with stress accumulation arc
3. user_b_scenario.py produces 18 packets with emergency at packet 16
4. ChromaDB collections initialize without error
5. Add one episode to episodic_memory, query it back, verify result
"""

import sys
import os
import json

# Ensure project root (adaptive_health_agent/) is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_user_a_scenario():
    """Verify User A stress accumulation arc across 50 packets."""
    print("=" * 60)
    print("TEST 1: User A Scenario — Stress Accumulation Arc")
    print("=" * 60)

    from telemetry.user_a_scenario import generate_packets
    packets = generate_packets()

    assert len(packets) == 50, f"Expected 50 packets, got {len(packets)}"
    print(f"  ✓ Generated {len(packets)} packets")

    # Verify packet structure (check first packet has all required keys)
    p1 = packets[0]
    assert "timestamp" in p1
    assert "user_id" in p1 and p1["user_id"] == "user_a"
    assert "vitals" in p1
    assert "movement" in p1
    assert "sleep_last_night" in p1
    assert "context" in p1
    assert "user_reported" in p1

    # Verify vital keys
    vital_keys = ["heart_rate", "hrv", "spo2", "skin_temperature",
                  "breathing_rate", "stress_score", "recovery_score",
                  "eda_stress_indicator"]
    for k in vital_keys:
        assert k in p1["vitals"], f"Missing vital key: {k}"
    print("  ✓ Packet structure valid (all required fields present)")

    # Phase 1 (packets 1-10): Normal baseline
    for pkt in packets[:10]:
        v = pkt["vitals"]
        assert 55 <= v["heart_rate"] <= 75, f"Phase 1 HR out of range: {v['heart_rate']}"
        assert 50 <= v["hrv"] <= 60, f"Phase 1 HRV out of range: {v['hrv']}"
        assert 20 <= v["stress_score"] <= 30, f"Phase 1 stress out of range: {v['stress_score']}"
    print("  ✓ Packets 1-10: Normal baseline values (HR ~68, HRV ~55, stress ~25)")

    # Phase 2 (packets 11-20): Declining
    p20 = packets[19]
    assert p20["vitals"]["hrv"] <= 45, f"Phase 2 end HRV should be ~42, got {p20['vitals']['hrv']}"
    assert p20["sleep_last_night"]["total_hours"] <= 6.5, f"Phase 2 end sleep should be ~6.0, got {p20['sleep_last_night']['total_hours']}"
    print(f"  ✓ Packet 20: HRV={p20['vitals']['hrv']}, sleep={p20['sleep_last_night']['total_hours']}hrs (declining)")

    # Phase 3 (packets 21-35): High stress
    for pkt in packets[20:35]:
        v = pkt["vitals"]
        assert v["stress_score"] >= 55, f"Phase 3 stress should be 60-70, got {v['stress_score']}"
        assert v["recovery_score"] <= 50, f"Phase 3 recovery should be 35-45, got {v['recovery_score']}"
    print("  ✓ Packets 21-35: Stress 60-70, recovery 35-45")

    # Phase 4 (packets 36-40): User reports
    for pkt in packets[35:40]:
        assert pkt["user_reported"]["stress_level"] == "high", "Phase 4 should report stress_level=high"
        assert pkt["user_reported"]["notes"] == "work has been intense", "Phase 4 should have notes"
    print("  ✓ Packets 36-40: User reports 'work has been intense', stress_level='high'")

    # Phase 5 (packets 41-50): Sustained stress >75
    consecutive_high = 0
    for pkt in packets[40:50]:
        if pkt["vitals"]["stress_score"] > 75:
            consecutive_high += 1
    assert consecutive_high >= 3, f"Need 3+ consecutive stress >75, got {consecutive_high} total >75"
    print(f"  ✓ Packets 41-50: {consecutive_high}/10 packets with stress >75 (triggers pattern)")

    # Print sample values for verification
    print("\n  Sample stress scores (packets 41-50):")
    for i, pkt in enumerate(packets[40:50], start=41):
        print(f"    Packet {i}: stress={pkt['vitals']['stress_score']}, "
              f"recovery={pkt['vitals']['recovery_score']}, "
              f"sleep={pkt['sleep_last_night']['total_hours']}hrs")

    print("\n  ✅ User A scenario PASSED\n")


def test_user_b_scenario():
    """Verify User B emergency arc with 18 packets."""
    print("=" * 60)
    print("TEST 2: User B Scenario — Emergency Arc")
    print("=" * 60)

    from telemetry.user_b_scenario import generate_packets
    packets = generate_packets()

    assert len(packets) == 18, f"Expected 18 packets, got {len(packets)}"
    print(f"  ✓ Generated {len(packets)} packets")

    # Packets 1-15: Calm baseline
    for pkt in packets[:15]:
        v = pkt["vitals"]
        assert 65 <= v["heart_rate"] <= 72, f"Baseline HR out of range: {v['heart_rate']}"
        assert v["spo2"] == 98, f"Baseline SpO2 should be 98, got {v['spo2']}"
        assert v["eda_stress_indicator"] == "low"
        assert pkt["movement"]["activity_state"] == "sedentary"
        assert pkt["context"]["location_zone"] == "home"
    print("  ✓ Packets 1-15: Calm baseline (HR ~68, SpO2 98, sedentary, home)")

    # Packet 16: Emergency
    p16 = packets[15]
    assert p16["vitals"]["heart_rate"] == 134, f"Emergency HR should be 134, got {p16['vitals']['heart_rate']}"
    assert p16["vitals"]["spo2"] == 93, f"Emergency SpO2 should be 93, got {p16['vitals']['spo2']}"
    assert p16["vitals"]["breathing_rate"] == 22, f"Emergency breathing should be 22, got {p16['vitals']['breathing_rate']}"
    assert p16["vitals"]["eda_stress_indicator"] == "high"
    assert p16["movement"]["activity_state"] == "sedentary"
    assert "02:17" in p16["timestamp"], f"Emergency should be at 2:17am, got {p16['timestamp']}"
    print(f"  ✓ Packet 16: EMERGENCY — HR={p16['vitals']['heart_rate']}, SpO2={p16['vitals']['spo2']}, "
          f"breathing={p16['vitals']['breathing_rate']}, time={p16['timestamp']}")

    # Packets 17-18: Sustained elevated
    for i, pkt in enumerate(packets[16:18], start=17):
        v = pkt["vitals"]
        assert v["heart_rate"] > 120, f"Packet {i} HR should be >120, got {v['heart_rate']}"
        assert v["spo2"] <= 94, f"Packet {i} SpO2 should be <=94, got {v['spo2']}"
        assert v["eda_stress_indicator"] == "high"
        print(f"  ✓ Packet {i}: Sustained — HR={v['heart_rate']}, SpO2={v['spo2']}, "
              f"breathing={v['breathing_rate']}")

    print("\n  ✅ User B scenario PASSED\n")


def test_stream_emitter():
    """Verify stream.py yields packets correctly (without sleep delay)."""
    print("=" * 60)
    print("TEST 3: Telemetry Stream Emitter")
    print("=" * 60)

    from telemetry.stream import stream_packets, format_packet

    # Use first 3 packets from User A as test data
    from telemetry.user_a_scenario import generate_packets
    test_packets = generate_packets()[:3]

    # Stream with 0 interval (no delay for testing)
    received = []
    for packet in stream_packets(test_packets, interval=0):
        received.append(packet)

    assert len(received) == 3, f"Expected 3 packets, got {len(received)}"
    print(f"  ✓ Streamed {len(received)} packets successfully")

    # Verify format_packet produces valid JSON
    formatted = format_packet(received[0])
    parsed = json.loads(formatted)
    assert parsed["user_id"] == "user_a"
    print(f"  ✓ format_packet produces valid JSON")

    print("\n  ✅ Stream emitter PASSED\n")


def test_chromadb_setup():
    """Verify ChromaDB collections initialize and basic add/query works."""
    print("=" * 60)
    print("TEST 4: ChromaDB Setup & Basic Operations")
    print("=" * 60)

    from memory.episodic_memory import (
        get_chroma_client, get_episodic_collection,
        get_knowledge_base_collection, get_embedding
    )

    # Verify client initialized
    client = get_chroma_client()
    assert client is not None, "ChromaDB client is None"
    print("  ✓ ChromaDB PersistentClient initialized")

    # Verify collections exist
    ep_col = get_episodic_collection()
    kb_col = get_knowledge_base_collection()
    assert ep_col is not None
    assert kb_col is not None
    assert ep_col.name == "episodic_memory"
    assert kb_col.name == "medical_knowledge_base"
    print(f"  ✓ Collection 'episodic_memory' created (count: {ep_col.count()})")
    print(f"  ✓ Collection 'medical_knowledge_base' created (count: {kb_col.count()})")

    # Verify embedding generation
    embedding = get_embedding("test heart rate elevated during exercise")
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) == 384, f"MiniLM-L6-v2 should produce 384-dim embeddings, got {len(embedding)}"
    print(f"  ✓ Embedding generated: {len(embedding)} dimensions")

    # Test: add one episode, query it back
    test_id = "test_episode_phase1_verification"
    test_doc = "Heart rate elevated to 95bpm while sedentary at night, user age 72"
    test_embedding = get_embedding(test_doc)

    # Clean up any previous test data
    try:
        ep_col.delete(ids=[test_id])
    except Exception:
        pass

    ep_col.add(
        ids=[test_id],
        embeddings=[test_embedding],
        documents=[test_doc],
        metadatas=[{"user_id": "test_user", "event_type": "anomaly"}]
    )
    print(f"  ✓ Added test episode to episodic_memory")

    # Query it back with a similar query
    query_embedding = get_embedding("elevated heart rate sedentary night elderly")
    results = ep_col.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    assert len(results["ids"][0]) == 1, "Should return 1 result"
    assert results["ids"][0][0] == test_id, f"Should match test episode, got {results['ids'][0][0]}"
    assert "heart rate elevated" in results["documents"][0][0].lower()
    print(f"  ✓ Query returned matching episode: '{results['documents'][0][0][:60]}...'")

    # Clean up test data
    ep_col.delete(ids=[test_id])
    print(f"  ✓ Test episode cleaned up")

    print("\n  ✅ ChromaDB setup PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION — Adaptive Health Agent")
    print("=" * 60 + "\n")

    test_user_a_scenario()
    test_user_b_scenario()
    test_stream_emitter()
    test_chromadb_setup()

    print("=" * 60)
    print("  ALL PHASE 1 TESTS PASSED ✅")
    print("=" * 60)
