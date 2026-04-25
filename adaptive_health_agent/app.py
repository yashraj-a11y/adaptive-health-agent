"""
Streamlit UI — AVA (Adaptive Virtual Assistant)
Cool blue theme, separate scrollable containers, natural adaptation.
"""

import os, sys, json, time
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", category=UserWarning)

from graph.graph import build_graph, build_user_message_graph
from knowledge_base.loader import load_knowledge_base
from memory.living_profile import load_profile, save_profile, create_profile
from telemetry.user_a_scenario import generate_packets as gen_user_a
from telemetry.user_b_scenario import generate_packets as gen_user_b
from agents.profiler import _pattern_buffer

st.set_page_config(page_title="Health Advisor", page_icon="🫀", layout="wide")

# ═══ COOL BLUE THEME CSS ═══
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(160deg, #0f172a 0%, #020617 50%, #0f172a 100%);
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #020617 100%);
    border-right: 1px solid #1e293b;
}

/* Advisor message */
.advisor-bubble {
    background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
    border-left: 3px solid #3b82f6;
    border-radius: 0 14px 14px 14px;
    padding: 12px 16px; margin: 8px 0;
    color: #f8fafc; font-size: 14px; line-height: 1.55;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.advisor-label {
    font-size: 10px; font-weight: 600; color: #60a5fa;
    margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px;
}
/* User message */
.user-bubble {
    background: linear-gradient(135deg, #0369a1 0%, #082f49 100%);
    border-right: 3px solid #38bdf8;
    border-radius: 14px 0 14px 14px;
    padding: 12px 16px; margin: 8px 0 8px 50px;
    color: #f8fafc; font-size: 14px; text-align: right;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
/* Emergency */
.emergency-bubble {
    background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
    border-left: 3px solid #ef4444;
    border-radius: 0 14px 14px 14px;
    padding: 14px 16px; margin: 10px 0;
    color: #fecaca; font-size: 14px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.emergency-bubble .advisor-label { color: #f87171; font-weight: 700; }
/* Concern */
.concern-bubble {
    background: linear-gradient(135deg, #7c2d12 0%, #431407 100%);
    border-left: 3px solid #f97316;
    border-radius: 0 14px 14px 14px;
    padding: 12px 16px; margin: 8px 0;
    color: #fed7aa; font-size: 14px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.concern-bubble .advisor-label { color: #fb923c; }

/* Vitals panel */
.vital-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 10px; padding: 12px 8px; margin: 4px 0; text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.vital-label { font-size: 9px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.vital-value { font-size: 22px; font-weight: 700; color: #f8fafc; }
.vital-unit { font-size: 10px; color: #94a3b8; }

.section-hdr {
    font-size: 10px; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.5px; margin: 16px 0 6px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 4px;
}
.summary-para { font-size: 12px; color: #cbd5e1; line-height: 1.5; margin: 4px 0; }
.feed-line { font-size: 11px; color: #94a3b8; padding: 4px 0; border-bottom: 1px solid #1e293b; }

/* Custom scrollbar for containers (WebKit) */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent; 
}
::-webkit-scrollbar-thumb {
    background: #334155; 
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569; 
}
</style>
""", unsafe_allow_html=True)


# ═══ SESSION STATE ═══
def init_state():
    defaults = {
        "initialized": False, "graph": None, "msg_graph": None,
        "packets": [], "current_packet_idx": 0, "auto_streaming": False,
        "conversation": [],
        "current_profile": None, "current_user_id": None,
        "last_packet_data": None, "packet_history": [],
        "deviation_count": 0, "pattern_count": 0, "messages_sent": 0,
        "last_msg_idx": -10,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═══ HELPERS ═══
def initialize_system():
    if not st.session_state.initialized:
        with st.spinner("Loading knowledge base..."):
            load_knowledge_base()
        st.session_state.graph = build_graph()
        st.session_state.msg_graph = build_user_message_graph()
        st.session_state.initialized = True

def load_or_create_profile(user_id, scenario):
    profile = load_profile(user_id)
    if profile is None:
        if scenario == "a":
            data = {"user_id": "user_a", "name": "Alex", "age": 32,
                    "known_conditions": [], "medications": [],
                    "goals": ["manage_stress"], "communication_style": "balanced",
                    "directness": 3, "depth": 3, "tone": 3, "length": 3, "framing": 3,
                    "alert_sensitivity": "normal",
                    "emergency_contact": {"name": "Partner", "contact": "555-0100"},
                    "best_engagement_times": [], "engagement_patterns": "unknown"}
        else:
            data = {"user_id": "user_b", "name": "Eleanor", "age": 72,
                    "known_conditions": ["hypertension", "type 2 diabetes"],
                    "medications": ["metformin", "lisinopril"],
                    "goals": ["monitor_condition"], "communication_style": "casual",
                    "directness": 4, "depth": 4, "tone": 5, "length": 4, "framing": 5,
                    "alert_sensitivity": "high",
                    "emergency_contact": {"name": "Michael", "contact": "555-0199"},
                    "best_engagement_times": [], "engagement_patterns": "unknown"}
        profile = create_profile(data)
        profile["baselines"]["status"] = "ESTABLISHED"
        profile["baselines"].update({
            "resting_hr": 68, "typical_hrv": 55, "typical_spo2": 98,
            "typical_skin_temp": 36.5, "typical_sleep_hours": 7.5,
            "typical_sleep_efficiency": 88, "typical_daily_steps": 5000,
            "typical_breathing_rate": 14, "typical_stress_score": 25,
            "typical_recovery_score": 70,
        })
        profile["days_monitored"] = 20
        save_profile(data["user_id"], profile)
    return profile

def process_packet(packet):
    profile = st.session_state.current_profile
    state = {
        "living_profile": profile, "current_packet": packet,
        "deviation_detected": False, "pattern_confirmed": False,
        "pattern_details": None, "severity_level": None,
        "analyst_output": None, "proceed_to_communicate": False,
        "final_message": None, "notify_family": False,
        "user_message": None, "agent_response": None,
    }
    try:
        result = st.session_state.graph.invoke(state)
    except Exception:
        result = state

    if result.get("deviation_detected"):
        st.session_state.deviation_count += 1
    if result.get("pattern_confirmed"):
        st.session_state.pattern_count += 1
    st.session_state.last_packet_data = packet

    if result.get("final_message"):
        # Cooldown: don't send messages too frequently
        idx = st.session_state.current_packet_idx
        if idx - st.session_state.last_msg_idx >= 3:
            st.session_state.messages_sent += 1
            st.session_state.last_msg_idx = idx
            severity = result.get("severity_level", 1)
            btype = "emergency" if severity >= 5 else "concern" if severity >= 3 else "advisor"
            st.session_state.conversation.append({
                "role": "advisor", "type": btype,
                "content": result["final_message"], "severity": severity,
                "timestamp": packet.get("timestamp", ""),
                "notify_family": result.get("notify_family", False),
            })
    st.session_state.current_profile = load_profile(st.session_state.current_user_id) or profile
    return result

def send_chat(message):
    profile = st.session_state.current_profile

    # Build recent conversation context under the hood
    recent_conv = ""
    for entry in st.session_state.conversation[-6:]:
        role = "You" if entry["role"] == "user" else "AVA"
        recent_conv += f"{role}: {entry['content'][:150]}\n"

    # Prepend history so the LLM has short-term memory to adhere to instructions
    contextualized_message = f"RECENT CONVERSATION HISTORY:\n{recent_conv}\n\nCURRENT USER MESSAGE:\n{message}"

    state = {
        "living_profile": profile,
        "current_packet": {"user_id": st.session_state.current_user_id,
                           "timestamp": datetime.now().isoformat()},
        "deviation_detected": False, "pattern_confirmed": False,
        "pattern_details": None, "severity_level": None,
        "analyst_output": None, "proceed_to_communicate": False,
        "final_message": None, "notify_family": False,
        "user_message": contextualized_message,
        "agent_response": None,
    }
    try:
        result = st.session_state.msg_graph.invoke(state)
        response = result.get("agent_response") or "I couldn't process that right now."
    except Exception as e:
        response = f"Sorry, trouble right now: {str(e)[:80]}"

    st.session_state.conversation.append({"role": "user", "content": message})
    st.session_state.conversation.append({"role": "advisor", "type": "advisor", "content": response})

    # The magic of adapting the Living Profile based on keywords:
    try:
        from agents.communicator import apply_communication_feedback
        uid = st.session_state.current_user_id
        ml = message.lower().strip()
        
        # Original triggers
        if len(ml.split()) <= 3 and ml in ("ok","okay","sure","fine","k","thanks","got it"):
            apply_communication_feedback(uid, "dismissive")
        if any(kw in ml for kw in ("number","data","metric","stat","bpm","hrv")):
            apply_communication_feedback(uid, "engages_data")
            
        # New explicit tone request triggers
        if any(kw in ml for kw in ("brief", "short", "direct", "quick", "cut to the chase", "less words")):
            apply_communication_feedback(uid, "requests_brief")
        if any(kw in ml for kw in ("warm", "friendly", "nice", "kind")):
            apply_communication_feedback(uid, "requests_warmth")
        if any(kw in ml for kw in ("detailed", "explain", "more info", "elaborate")):
            apply_communication_feedback(uid, "requests_detail")
        if "pirate" in ml:
            apply_communication_feedback(uid, "requests_pirate")
            
    except Exception:
        pass


# ═══════════════════════════════════════
# LEFT SIDEBAR
# ═══════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫀 Health Agent")
    st.markdown("---")

    scenario = st.radio("Scenario", ["a", "b"],
        format_func=lambda x: "👤 Alex (32) — Stress" if x == "a" else "👵 Eleanor (72) — Emergency",
        key="scenario_select")

    if st.button("🔄 Initialize / Reset", use_container_width=True):
        st.session_state.current_packet_idx = 0
        st.session_state.auto_streaming = False
        st.session_state.conversation = []
        st.session_state.last_packet_data = None
        st.session_state.packet_history = []
        st.session_state.deviation_count = 0
        st.session_state.pattern_count = 0
        st.session_state.messages_sent = 0
        st.session_state.last_msg_idx = -10
        for m in _pattern_buffer._counters:
            _pattern_buffer.reset(m)
        initialize_system()
        uid = f"user_{scenario}"
        st.session_state.current_user_id = uid
        st.session_state.current_profile = load_or_create_profile(uid, scenario)
        st.session_state.packets = gen_user_a() if scenario == "a" else gen_user_b()
        name = st.session_state.current_profile.get("identity", {}).get("name", "User")
        st.session_state.conversation.append({
            "role": "advisor", "type": "advisor",
            "content": f"Hi {name}! 👋 I'm your personal health advisor. I'll be monitoring your vitals and speaking up if anything looks unusual. Feel free to ask me anything about your health.",
        })
        st.rerun()

    if st.session_state.current_profile:
        st.markdown("---")
        total = len(st.session_state.packets) if st.session_state.packets else 0
        current = st.session_state.current_packet_idx
        remaining = total - current

        st.markdown(f"**📡 {current}/{total}** packets")

        if remaining > 0:
            c1, c2 = st.columns(2)
            with c1:
                if not st.session_state.auto_streaming:
                    if st.button("▶ Start", use_container_width=True, key="start_stream"):
                        st.session_state.auto_streaming = True
                        st.rerun()
                else:
                    if st.button("⏸ Stop", use_container_width=True, key="stop_stream"):
                        st.session_state.auto_streaming = False
                        st.rerun()
            with c2:
                if st.button("⏩ All", use_container_width=True, key="stream_all"):
                    st.session_state.auto_streaming = False
                    for i in range(current, total):
                        process_packet(st.session_state.packets[i])
                        st.session_state.packet_history.append(st.session_state.packets[i])
                        st.session_state.current_packet_idx = i + 1
                    st.rerun()
        else:
            st.success(f"✅ All {total} packets done")

        # Profile
        st.markdown("---")
        profile = st.session_state.current_profile
        identity = profile.get("identity", {})
        with st.expander(f"👤 {identity.get('name', 'User')}, age {identity.get('age', '?')}", expanded=False):
            conditions = identity.get("known_conditions", [])
            meds = identity.get("medications", [])
            ec = identity.get("emergency_contact", {})
            if conditions:
                st.caption(f"🏥 {', '.join(conditions)}")
            if meds:
                st.caption(f"💊 {', '.join(meds)}")
            if ec:
                st.caption(f"📞 {ec.get('name', '')} — {ec.get('contact', '')}")

        # Active Concerns (paragraph)
        st.markdown("---")
        concerns = profile.get("current_concerns", [])
        patterns = profile.get("known_patterns", [])

        st.markdown("**⚠️ Active Concerns**")
        if concerns:
            para = ". ".join(c.rstrip(". ") for c in concerns[-3:]) + ". Being monitored closely."
            st.caption(para)
        else:
            st.caption("No reading")

        st.markdown("**📋 Patterns Found**")
        if patterns:
            para = ". ".join(p.rstrip(". ") for p in patterns[-3:]) + "."
            st.caption(para)
        else:
            st.caption("No reading")

        # Self-report form (enter key works!)
        st.markdown("---")
        with st.form("self_report_form", clear_on_submit=True):
            note = st.text_input("📝 Tell me about yourself...",
                                 placeholder="e.g., Feeling off, started new meds...")
            submitted = st.form_submit_button("Submit", use_container_width=True)
            if submitted and note:
                send_chat(f"[Self-report] {note}")
                st.rerun()


# ═══════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════
if not st.session_state.current_profile:
    st.markdown("# 🫀 Your Health Advisor")
    st.info("👈 Select a scenario and click **Initialize / Reset** to begin.")
    st.stop()

chat_col, vitals_col = st.columns([6, 4])

# ─── LEFT: Chat ───
with chat_col:
    st.markdown("### 🫀 Conversation")
    
    chat_container = st.container(height=650)
    with chat_container:
        for entry in st.session_state.conversation:
            if entry["role"] == "user":
                content = entry["content"]
                if content.startswith("[Self-report]"):
                    content = content.replace("[Self-report] ", "📝 ")
                st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
            else:
                btype = entry.get("type", "advisor")
                severity = entry.get("severity", 0)
                ts = entry.get("timestamp", "")
                ts_str = f" · {ts[11:16]}" if ts and len(ts) > 11 else ""
                notify = entry.get("notify_family", False)

                if btype == "emergency":
                    extra = "<br>🚨 <b>Family contact has been notified.</b>" if notify else ""
                    st.markdown(f'<div class="emergency-bubble"><div class="advisor-label">🚨 EMERGENCY{ts_str}</div>{entry["content"]}{extra}</div>', unsafe_allow_html=True)
                elif btype == "concern":
                    st.markdown(f'<div class="concern-bubble"><div class="advisor-label">⚠️ Level {severity}{ts_str}</div>{entry["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="advisor-bubble"><div class="advisor-label">AVA{ts_str}</div>{entry["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask about your health...")
    if user_input:
        send_chat(user_input)
        st.rerun()

    # JS hack to auto-scroll the chat container (stVerticalBlock) to the bottom
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        // Find all Streamlit vertical blocks (containers with height)
        var blocks = window.parent.document.querySelectorAll('div[data-testid="stVerticalBlock"]');
        if (blocks.length > 0) {
            // The first one is the chat container
            var chatContainer = blocks[0];
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        </script>
        """,
        height=0
    )


# ─── RIGHT: Live Vitals Container ───
with vitals_col:
    st.markdown("### 📡 Live Vitals")
    
    vitals_container = st.container(height=650)
    with vitals_container:
        packet = st.session_state.last_packet_data
        if packet:
            vitals = packet.get("vitals", {})
            movement = packet.get("movement", {})
            sleep = packet.get("sleep_last_night", {})
            context = packet.get("context", {})
            ts = packet.get("timestamp", "")

            st.caption(f"📅 {ts[:16] if ts else '—'}")

            v1, v2 = st.columns(2)
            with v1:
                for label, key, unit in [("Heart Rate", "heart_rate", "bpm"), ("SpO₂", "spo2", "%"), ("Stress", "stress_score", "/100")]:
                    val = vitals.get(key, "—")
                    st.markdown(f'<div class="vital-card"><div class="vital-label">{label}</div><div class="vital-value">{val}</div><div class="vital-unit">{unit}</div></div>', unsafe_allow_html=True)
            with v2:
                for label, key, unit in [("HRV", "hrv", "ms"), ("Breathing", "breathing_rate", "/min"), ("Recovery", "recovery_score", "/100")]:
                    val = vitals.get(key, "—")
                    st.markdown(f'<div class="vital-card"><div class="vital-label">{label}</div><div class="vital-value">{val}</div><div class="vital-unit">{unit}</div></div>', unsafe_allow_html=True)

            # Context line
            activity = movement.get("activity_state", "—")
            location = context.get("location_zone", "—")
            tod = context.get("time_of_day", "—")
            st.caption(f"🏃 {activity} · 📍 {location} · 🕐 {tod}")
            st.caption(f"👣 {movement.get('steps_today', 0)} steps · 🔥 {movement.get('calories_burned', 0)} cal")

            # Sleep
            sleep_hrs = sleep.get("total_hours")
            if sleep_hrs:
                st.caption(f"😴 Sleep: {sleep_hrs}h · Eff: {sleep.get('sleep_efficiency', '?')}%")

            # Trends
            profile = st.session_state.current_profile
            current_state = profile.get("current_state", {})
            trends = {k: v for k, v in current_state.items() if v and v != "unknown"}
            if trends:
                st.markdown(f'<div class="section-hdr">Trends</div>', unsafe_allow_html=True)
                for k, v in trends.items():
                    label = k.replace("_7d", "").replace("_", " ").title()
                    icon = "🔴" if v in ("declining","deteriorating","rising") else "🟢" if v == "improving" else "🔵"
                    st.caption(f"{icon} {label}: {v}")

            # Recent feed
            if st.session_state.packet_history:
                st.markdown(f'<div class="section-hdr">Feed</div>', unsafe_allow_html=True)
                for pkt in reversed(st.session_state.packet_history[-5:]):
                    pts = pkt.get("timestamp", "")[:16]
                    phr = pkt.get("vitals", {}).get("heart_rate", "?")
                    pstress = pkt.get("vitals", {}).get("stress_score", "?")
                    st.markdown(f'<div class="feed-line">⏱ {pts} · HR {phr} · Stress {pstress}</div>', unsafe_allow_html=True)
        else:
            st.markdown("*Waiting for telemetry...*")
            st.caption("Press ▶ Start in the sidebar.")


# ═══ AUTO-STREAM LOOP ═══
if st.session_state.auto_streaming:
    idx = st.session_state.current_packet_idx
    total = len(st.session_state.packets)
    if idx < total:
        process_packet(st.session_state.packets[idx])
        st.session_state.packet_history.append(st.session_state.packets[idx])
        st.session_state.current_packet_idx = idx + 1
        time.sleep(2)
        st.rerun()
    else:
        st.session_state.auto_streaming = False
        st.rerun()
