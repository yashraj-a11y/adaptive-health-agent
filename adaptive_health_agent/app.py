"""
Streamlit UI — Adaptive Personal Health Agent

Single-page app with:
  - Sidebar: User selection, scenario controls, onboarding
  - Main area: Real-time telemetry display, agent messages, chat
  - Metrics dashboard: Live vitals, trends, pattern history
"""

import os
import sys
import json
import time
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.graph import build_graph, build_user_message_graph
from graph.state import HealthAgentState
from knowledge_base.loader import load_knowledge_base
from memory.living_profile import load_profile, save_profile, create_profile
from memory.episodic_memory import get_recent, query_similar
from memory.summarizer import generate_weekly_summary
from telemetry.user_a_scenario import generate_packets as gen_user_a
from telemetry.user_b_scenario import generate_packets as gen_user_b
from agents.profiler import _pattern_buffer


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Health Agent",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #252b3b 100%);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 16px;
        margin: 4px 0;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #e0e0e0;
    }
    .metric-label {
        font-size: 13px;
        color: #8892a4;
        margin-bottom: 4px;
    }
    .metric-delta-up {
        color: #ff6b6b;
        font-size: 12px;
    }
    .metric-delta-down {
        color: #51cf66;
        font-size: 12px;
    }
    .metric-delta-stable {
        color: #8892a4;
        font-size: 12px;
    }
    .agent-message {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 100%);
        border-left: 4px solid #4dabf7;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .agent-message-emergency {
        background: linear-gradient(135deg, #5f1e1e 0%, #441a1a 100%);
        border-left: 4px solid #ff6b6b;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .user-message {
        background: linear-gradient(135deg, #1e5f3a 0%, #1a4427 100%);
        border-left: 4px solid #51cf66;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .severity-1 { background: #2d3548; color: #8892a4; }
    .severity-2 { background: #1e3a5f; color: #4dabf7; }
    .severity-3 { background: #5f4d1e; color: #ffd43b; }
    .severity-4 { background: #5f3a1e; color: #ff922b; }
    .severity-5 { background: #5f1e1e; color: #ff6b6b; }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 8px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #151922 100%);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "initialized": False,
        "graph": None,
        "msg_graph": None,
        "packets": [],
        "current_packet_idx": 0,
        "is_streaming": False,
        "messages": [],
        "chat_history": [],
        "current_profile": None,
        "current_user_id": None,
        "last_result": None,
        "packet_history": [],
        "deviation_count": 0,
        "pattern_count": 0,
        "messages_sent": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def initialize_system():
    """Load KB and build graphs once."""
    if not st.session_state.initialized:
        with st.spinner("Loading medical knowledge base..."):
            load_knowledge_base()
        st.session_state.graph = build_graph()
        st.session_state.msg_graph = build_user_message_graph()
        st.session_state.initialized = True


def load_or_create_profile(user_id: str, scenario: str) -> dict:
    """Load existing profile or create a demo profile."""
    profile = load_profile(user_id)
    if profile is None:
        if scenario == "a":
            user_data = {
                "user_id": "user_a", "name": "Alex", "age": 32,
                "known_conditions": [], "medications": [],
                "goals": ["manage_stress"], "communication_style": "balanced",
                "directness": 3, "depth": 3, "tone": 3, "length": 3, "framing": 3,
                "alert_sensitivity": "normal",
                "emergency_contact": {"name": "Partner", "contact": "555-0100"},
                "best_engagement_times": [], "engagement_patterns": "unknown",
            }
        else:
            user_data = {
                "user_id": "user_b", "name": "Eleanor", "age": 72,
                "known_conditions": ["hypertension", "type 2 diabetes"],
                "medications": ["metformin", "lisinopril"],
                "goals": ["monitor_condition"], "communication_style": "casual",
                "directness": 4, "depth": 4, "tone": 5, "length": 4, "framing": 5,
                "alert_sensitivity": "high",
                "emergency_contact": {"name": "Michael", "contact": "555-0199"},
                "best_engagement_times": [], "engagement_patterns": "unknown",
            }
        profile = create_profile(user_data)
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


def process_packet(packet: dict) -> dict:
    """Process a single telemetry packet through the graph."""
    profile = st.session_state.current_profile
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

    try:
        result = st.session_state.graph.invoke(state)
    except Exception as e:
        st.error(f"Graph error: {e}")
        result = state

    # Update counters
    if result.get("deviation_detected"):
        st.session_state.deviation_count += 1
    if result.get("pattern_confirmed"):
        st.session_state.pattern_count += 1
    if result.get("final_message"):
        st.session_state.messages_sent += 1
        st.session_state.messages.append({
            "type": "agent",
            "severity": result.get("severity_level", 1),
            "message": result["final_message"],
            "timestamp": packet.get("timestamp", ""),
            "notify_family": result.get("notify_family", False),
        })

    # Reload profile after graph updates
    st.session_state.current_profile = load_profile(st.session_state.current_user_id) or profile

    return result


def send_chat_message(message: str):
    """Send a user chat message through the communicator."""
    profile = st.session_state.current_profile
    state = {
        "living_profile": profile,
        "current_packet": {
            "user_id": st.session_state.current_user_id,
            "timestamp": datetime.now().isoformat(),
        },
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

    try:
        result = st.session_state.msg_graph.invoke(state)
        response = result.get("agent_response", "I couldn't process that. Please try again.")
    except Exception as e:
        response = f"Error: {e}"

    st.session_state.chat_history.append({"role": "user", "content": message})
    st.session_state.chat_history.append({"role": "agent", "content": response})


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Health Agent")
    st.markdown("---")

    # Scenario selection
    scenario = st.radio(
        "Select Scenario",
        options=["a", "b"],
        format_func=lambda x: "User A — Stress Arc (50 packets)" if x == "a"
                              else "User B — Emergency Arc (18 packets)",
        index=0,
        key="scenario_select",
    )

    # Initialize button
    if st.button("🔄 Initialize / Reset", use_container_width=True, key="init_btn"):
        # Reset state
        st.session_state.current_packet_idx = 0
        st.session_state.is_streaming = False
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.session_state.packet_history = []
        st.session_state.deviation_count = 0
        st.session_state.pattern_count = 0
        st.session_state.messages_sent = 0

        # Reset pattern buffer
        for metric in _pattern_buffer._counters:
            _pattern_buffer.reset(metric)

        initialize_system()

        user_id = f"user_{scenario}"
        st.session_state.current_user_id = user_id
        st.session_state.current_profile = load_or_create_profile(user_id, scenario)

        if scenario == "a":
            st.session_state.packets = gen_user_a()
        else:
            st.session_state.packets = gen_user_b()

        st.success(f"Initialized with {len(st.session_state.packets)} packets")

    st.markdown("---")

    # Streaming controls
    st.markdown("### ▶️ Telemetry Controls")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Next", use_container_width=True, key="next_btn"):
            if st.session_state.packets and st.session_state.current_packet_idx < len(st.session_state.packets):
                packet = st.session_state.packets[st.session_state.current_packet_idx]
                result = process_packet(packet)
                st.session_state.last_result = result
                st.session_state.packet_history.append(packet)
                st.session_state.current_packet_idx += 1

    with col2:
        total = len(st.session_state.packets) if st.session_state.packets else 0
        current = st.session_state.current_packet_idx
        st.markdown(f"**{current}/{total}**")

    # Stream all button
    if st.button("⏩ Stream All", use_container_width=True, key="stream_all_btn"):
        if st.session_state.packets:
            progress = st.progress(0)
            total = len(st.session_state.packets)
            start = st.session_state.current_packet_idx

            for i in range(start, total):
                packet = st.session_state.packets[i]
                result = process_packet(packet)
                st.session_state.last_result = result
                st.session_state.packet_history.append(packet)
                st.session_state.current_packet_idx = i + 1
                progress.progress((i + 1 - start) / (total - start) if total > start else 1.0)

            progress.empty()
            st.success(f"Processed {total - start} packets")

    st.markdown("---")

    # Profile info
    if st.session_state.current_profile:
        profile = st.session_state.current_profile
        identity = profile.get("identity", {})
        st.markdown("### 👤 User Profile")
        st.markdown(f"**Name:** {identity.get('name', 'N/A')}")
        st.markdown(f"**Age:** {identity.get('age', 'N/A')}")
        conditions = identity.get("known_conditions", [])
        st.markdown(f"**Conditions:** {', '.join(conditions) if conditions else 'None'}")
        st.markdown(f"**Baseline:** {profile.get('baselines', {}).get('status', 'N/A')}")
        st.markdown(f"**Days Monitored:** {profile.get('days_monitored', 0)}")

    st.markdown("---")

    # Stats
    st.markdown("### 📊 Session Stats")
    st.markdown(f"⚡ Deviations: **{st.session_state.deviation_count}**")
    st.markdown(f"🔴 Patterns: **{st.session_state.pattern_count}**")
    st.markdown(f"📨 Messages: **{st.session_state.messages_sent}**")


# ─────────────────────────────────────────────
# Main Content Area
# ─────────────────────────────────────────────
st.markdown("# 🫀 Adaptive Personal Health Agent")

if not st.session_state.current_profile:
    st.info("👈 Select a scenario and click **Initialize / Reset** in the sidebar to begin.")
    st.stop()

# ─── Top Row: Live Vitals ───
st.markdown("## 📡 Live Telemetry")

if st.session_state.packet_history:
    latest = st.session_state.packet_history[-1]
    vitals = latest.get("vitals", {})
    movement = latest.get("movement", {})
    sleep = latest.get("sleep_last_night", {})
    context = latest.get("context", {})
    baselines = st.session_state.current_profile.get("baselines", {})

    # Vitals row
    cols = st.columns(8)
    metrics = [
        ("❤️ HR", vitals.get("heart_rate"), "bpm", baselines.get("resting_hr")),
        ("💓 HRV", vitals.get("hrv"), "ms", baselines.get("typical_hrv")),
        ("🩸 SpO2", vitals.get("spo2"), "%", baselines.get("typical_spo2")),
        ("🌡️ Temp", vitals.get("skin_temperature"), "°C", baselines.get("typical_skin_temp")),
        ("🫁 Breath", vitals.get("breathing_rate"), "/min", baselines.get("typical_breathing_rate")),
        ("😰 Stress", vitals.get("stress_score"), "", baselines.get("typical_stress_score")),
        ("💪 Recovery", vitals.get("recovery_score"), "", baselines.get("typical_recovery_score")),
        ("⚡ EDA", vitals.get("eda_stress_indicator"), "", None),
    ]

    for col, (label, value, unit, baseline) in zip(cols, metrics):
        with col:
            delta = None
            delta_color = "off"
            if baseline and value and isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
                diff = round(value - baseline, 1)
                if diff != 0:
                    delta = f"{diff:+} {unit}"
                    # For HR, stress: increase is bad. For HRV, recovery: decrease is bad
                    if label in ("❤️ HR", "😰 Stress", "🫁 Breath", "🌡️ Temp"):
                        delta_color = "inverse"
                    else:
                        delta_color = "normal"

            display_value = f"{value} {unit}" if isinstance(value, (int, float)) else str(value)
            st.metric(label=label, value=display_value, delta=delta, delta_color=delta_color)

    # Context row
    st.markdown(f"**Timestamp:** {latest.get('timestamp', 'N/A')} | "
                f"**Activity:** {movement.get('activity_state', 'N/A')} | "
                f"**Location:** {context.get('location_zone', 'N/A')} | "
                f"**Steps:** {movement.get('steps_today', 0)} | "
                f"**Sleep:** {sleep.get('total_hours', 'N/A')}hrs "
                f"({sleep.get('sleep_efficiency', 'N/A')}% efficiency)")
else:
    st.markdown("*No telemetry data yet. Click ▶ Next or ⏩ Stream All to begin.*")

st.markdown("---")

# ─── Two Column Layout: Messages + Chat ───
msg_col, chat_col = st.columns([3, 2])

with msg_col:
    st.markdown("## 🤖 Agent Messages")

    if st.session_state.messages:
        for msg in reversed(st.session_state.messages[-10:]):
            severity = msg.get("severity", 1)
            css_class = "agent-message-emergency" if severity >= 4 else "agent-message"
            severity_badge = f'<span class="status-badge severity-{severity}">Level {severity}</span>'

            st.markdown(
                f'<div class="{css_class}">'
                f'{severity_badge} &nbsp; <small>{msg["timestamp"]}</small>'
                f'<p>{msg["message"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if msg.get("notify_family"):
                st.error("🚨 EMERGENCY: Family notification triggered!")
    else:
        st.markdown("*No agent messages yet. Process telemetry packets to see insights.*")

with chat_col:
    st.markdown("## 💬 Chat")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for entry in st.session_state.chat_history[-10:]:
            if entry["role"] == "user":
                st.markdown(
                    f'<div class="user-message">🧑 {entry["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="agent-message">🤖 {entry["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # Chat input
    user_input = st.text_input("Ask about your health data...", key="chat_input",
                                placeholder="e.g., How has my sleep been?")
    if st.button("Send", key="send_chat_btn") and user_input:
        send_chat_message(user_input)
        st.rerun()

st.markdown("---")

# ─── Bottom: Trends & History ───
st.markdown("## 📈 Trends & Pattern History")

if st.session_state.packet_history and len(st.session_state.packet_history) > 1:
    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:
        # HR trend chart
        hr_data = [p["vitals"]["heart_rate"] for p in st.session_state.packet_history]
        stress_data = [p["vitals"]["stress_score"] for p in st.session_state.packet_history]
        st.markdown("### Heart Rate & Stress")
        chart_data = {"Heart Rate": hr_data, "Stress Score": stress_data}
        st.line_chart(chart_data)

    with trend_col2:
        # HRV and recovery trend
        hrv_data = [p["vitals"]["hrv"] for p in st.session_state.packet_history]
        recovery_data = [p["vitals"]["recovery_score"] for p in st.session_state.packet_history]
        st.markdown("### HRV & Recovery")
        chart_data2 = {"HRV": hrv_data, "Recovery": recovery_data}
        st.line_chart(chart_data2)

    # Current state summary
    if st.session_state.current_profile:
        current_state = st.session_state.current_profile.get("current_state", {})
        concerns = st.session_state.current_profile.get("current_concerns", [])
        patterns = st.session_state.current_profile.get("known_patterns", [])

        state_col1, state_col2, state_col3 = st.columns(3)
        with state_col1:
            st.markdown("### Current State")
            for key, value in current_state.items():
                icon = "🔴" if value in ("declining", "rising") else "🟢" if value == "stable" else "⚪"
                st.markdown(f"{icon} **{key.replace('_', ' ').title()}:** {value}")

        with state_col2:
            st.markdown("### Active Concerns")
            if concerns:
                for c in concerns:
                    st.markdown(f"⚠️ {c}")
            else:
                st.markdown("*No active concerns*")

        with state_col3:
            st.markdown("### Known Patterns")
            if patterns:
                for p in patterns:
                    st.markdown(f"📋 {p}")
            else:
                st.markdown("*No patterns identified yet*")
else:
    st.markdown("*Process multiple packets to see trend charts.*")

# ─── Footer ───
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #8892a4; font-size: 12px;">'
    '🫀 Adaptive Personal Health Agent — Powered by Groq (llama3-70b-8192) + LangGraph + ChromaDB'
    '</div>',
    unsafe_allow_html=True,
)
