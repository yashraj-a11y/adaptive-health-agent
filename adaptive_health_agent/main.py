"""
Main Entry Point — Adaptive Personal Health Agent

Orchestrates the full health monitoring pipeline:
1. Loads/creates user profile via onboarding
2. Loads the medical knowledge base into ChromaDB
3. Builds the LangGraph agent graph
4. Streams telemetry packets through the graph
5. Handles user-initiated chat messages

Usage:
    python main.py                  # Run with User A scenario (stress arc)
    python main.py --scenario b     # Run with User B scenario (emergency arc)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from graph.graph import build_graph, build_user_message_graph
from graph.state import HealthAgentState
from knowledge_base.loader import load_knowledge_base
from memory.living_profile import load_profile, save_profile, create_profile
from memory.summarizer import generate_weekly_summary
from telemetry.stream import stream_packets
from telemetry.user_a_scenario import generate_packets as gen_user_a
from telemetry.user_b_scenario import generate_packets as gen_user_b
from utils.baseline_calculator import compute_baselines


def _create_default_profile(user_id: str, scenario: str) -> dict:
    """Create a default profile for demo scenarios without interactive onboarding.

    Args:
        user_id: The user identifier.
        scenario: "a" or "b" for the demo scenario.

    Returns:
        dict: The created Living Profile.
    """
    if scenario == "a":
        user_data = {
            "user_id": "user_a",
            "name": "Alex",
            "age": 32,
            "known_conditions": [],
            "medications": [],
            "goals": ["manage_stress"],
            "communication_style": "balanced",
            "directness": 3,
            "depth": 3,
            "tone": 3,
            "length": 3,
            "framing": 3,
            "alert_sensitivity": "normal",
            "emergency_contact": {"name": "Partner", "contact": "555-0100"},
            "best_engagement_times": [],
            "engagement_patterns": "unknown",
        }
    else:
        user_data = {
            "user_id": "user_b",
            "name": "Eleanor",
            "age": 72,
            "known_conditions": ["hypertension", "type 2 diabetes"],
            "medications": ["metformin", "lisinopril"],
            "goals": ["monitor_condition"],
            "communication_style": "casual",
            "directness": 4,
            "depth": 4,
            "tone": 5,
            "length": 4,
            "framing": 5,
            "alert_sensitivity": "high",
            "emergency_contact": {"name": "Michael", "contact": "555-0199"},
            "best_engagement_times": [],
            "engagement_patterns": "unknown",
        }

    profile = create_profile(user_data)

    # Set baselines to ESTABLISHED for demo purposes
    profile["baselines"]["status"] = "ESTABLISHED"
    profile["baselines"]["resting_hr"] = 68
    profile["baselines"]["typical_hrv"] = 55
    profile["baselines"]["typical_spo2"] = 98
    profile["baselines"]["typical_skin_temp"] = 36.5
    profile["baselines"]["typical_sleep_hours"] = 7.5
    profile["baselines"]["typical_sleep_efficiency"] = 88
    profile["baselines"]["typical_daily_steps"] = 5000
    profile["baselines"]["typical_breathing_rate"] = 14
    profile["baselines"]["typical_stress_score"] = 25
    profile["baselines"]["typical_recovery_score"] = 70
    profile["days_monitored"] = 20

    save_profile(user_data["user_id"], profile)
    return profile


def _build_initial_state(profile: dict, packet: dict) -> HealthAgentState:
    """Build the initial state for a graph invocation.

    Args:
        profile: The user's Living Profile dict.
        packet: The current telemetry packet.

    Returns:
        HealthAgentState: The initialized state dict.
    """
    return {
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


def run_scenario(scenario: str = "a"):
    """Run a telemetry scenario through the health agent graph.

    Args:
        scenario: "a" for stress accumulation, "b" for emergency arc.
    """
    print("\n" + "=" * 60)
    print("  Adaptive Personal Health Agent")
    print("=" * 60)

    # Step 1: Load knowledge base
    print("\n[Main] Loading medical knowledge base...")
    load_knowledge_base()

    # Step 2: Load or create profile
    user_id = f"user_{scenario}"
    profile = load_profile(user_id)
    if profile is None:
        print(f"[Main] Creating default profile for {user_id}...")
        profile = _create_default_profile(user_id, scenario)
    print(f"[Main] Profile loaded: {profile['identity']['name']} (age {profile['identity']['age']})")

    # Step 3: Build graph
    print("[Main] Building LangGraph agent graph...")
    graph = build_graph()

    # Step 4: Generate packets
    if scenario == "a":
        packets = gen_user_a()
        print(f"[Main] Scenario A: Stress accumulation arc ({len(packets)} packets)")
    else:
        packets = gen_user_b()
        print(f"[Main] Scenario B: Emergency arc ({len(packets)} packets)")

    print("\n" + "-" * 60)
    print("  Starting telemetry stream...")
    print("-" * 60 + "\n")

    # Step 5: Stream packets through the graph
    messages_sent = 0
    for i, packet in enumerate(stream_packets(packets, interval=0), start=1):
        print(f"\n--- Packet {i}/{len(packets)} | {packet['timestamp']} ---")

        # Reload profile to get latest updates
        current_profile = load_profile(user_id) or profile

        # Build initial state
        state = _build_initial_state(current_profile, packet)

        # Invoke the graph
        try:
            result = graph.invoke(state)
        except Exception as e:
            print(f"[Main] Graph error: {e}")
            continue

        # Report results
        if result.get("deviation_detected"):
            print(f"  ⚡ Deviation detected")
        if result.get("pattern_confirmed"):
            confirmed = result.get("pattern_details", {}).get("confirmed_metrics", [])
            print(f"  🔴 Pattern CONFIRMED: {', '.join(confirmed)}")
        if result.get("severity_level"):
            print(f"  ⚠️  Severity: Level {result['severity_level']}")
        if result.get("final_message"):
            messages_sent += 1
            print(f"\n  📨 Message to user:")
            print(f"  {'─' * 40}")
            for line in result["final_message"].split("\n"):
                print(f"  {line}")
            print(f"  {'─' * 40}")
        if result.get("notify_family"):
            print(f"  🚨 FAMILY NOTIFICATION TRIGGERED")

    # Summary
    print("\n" + "=" * 60)
    print(f"  Simulation Complete")
    print(f"  Packets processed: {len(packets)}")
    print(f"  Messages sent: {messages_sent}")
    print("=" * 60)

    # Generate weekly summary if enough data
    print("\n[Main] Generating weekly summary...")
    summary = generate_weekly_summary(user_id)
    if summary:
        print(f"[Main] Summary: {summary.get('outcome', 'N/A')[:200]}")


def handle_user_message(user_id: str, message: str) -> str:
    """Handle a user-initiated chat message.

    Args:
        user_id: The user identifier.
        message: The user's message text.

    Returns:
        str: The agent's response.
    """
    profile = load_profile(user_id)
    if profile is None:
        return "I don't have a profile for you yet. Please complete onboarding first."

    # Build a minimal state for user message handling
    state = {
        "living_profile": profile,
        "current_packet": {"user_id": user_id, "timestamp": datetime.now().isoformat()},
        "deviation_detected": False,
        "pattern_confirmed": False,
        "pattern_details": None,
        "severity_level": None,
        "analyst_output": None,
        "proceed_to_communicate": False,
        "final_message": None,
        "notify_family": False,
        "user_message": message,
        "agent_response": None,
    }

    # Use the simplified user message graph
    msg_graph = build_user_message_graph()
    result = msg_graph.invoke(state)

    return result.get("agent_response", "I'm not sure how to respond to that. Can you rephrase?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Personal Health Agent")
    parser.add_argument("--scenario", type=str, default="a", choices=["a", "b"],
                        help="Scenario to run: 'a' (stress arc) or 'b' (emergency arc)")
    args = parser.parse_args()

    run_scenario(args.scenario)
