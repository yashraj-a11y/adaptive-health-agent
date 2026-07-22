"""
Streamlit UI — AVA (Adaptive Virtual Assistant)
Dark clinical theme: header with live status, chat + vitals panes,
baseline-aware vital cards and trend charts.
"""

import os, sys, json, time, html, re, inspect
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Bridge Streamlit secrets to environment variables for Cloud deployment
for key in ["GROQ_API_KEY_PROFILER", "GROQ_API_KEY_ANALYST", "GROQ_API_KEY_COMMUNICATOR"]:
    try:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
    except FileNotFoundError:
        pass # No secrets.toml locally; keys come from .env


import warnings
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", category=UserWarning)

import altair as alt
import pandas as pd

from graph.graph import build_graph, build_user_message_graph
from knowledge_base.loader import load_knowledge_base
from memory.living_profile import load_profile, save_profile, create_profile
from telemetry.user_a_scenario import generate_packets as gen_user_a
from telemetry.user_b_scenario import generate_packets as gen_user_b
from agents.profiler import _pattern_buffer

st.set_page_config(page_title="AVA · Health Advisor", page_icon="🫀", layout="wide")

# ═══ THEME CSS ═══
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0b1120; --panel: #111a2e; --panel-2: #16213a; --line: #1f2b45;
    --ink: #e8edf6; --ink-2: #a9b4c8; --ink-3: #6b7890;
    --accent: #3987e5; --accent-soft: rgba(57,135,229,0.14);
    --good: #0ca30c; --warn: #fab219; --crit: #d03b3b; --serious: #ec835a;
}
html, body, .stApp, [class*="st-"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #13284d 0%, var(--bg) 55%) fixed; }
.block-container { padding-top: 2.2rem; padding-bottom: 1rem; max-width: 1500px; }
section[data-testid="stSidebar"] { background: #08101f; border-right: 1px solid var(--line); }
header[data-testid="stHeader"] { background: transparent; }

/* ── Brand ── */
.brand { display:flex; align-items:center; gap:10px; margin: 0 0 4px; }
.brand-mark { width:34px; height:34px; border-radius:10px; display:grid; place-items:center;
    background: linear-gradient(135deg, #3987e5, #1d4ed8); font-size:18px; box-shadow: 0 4px 14px rgba(57,135,229,.35); }
.brand-name { font-weight:700; font-size:18px; color:var(--ink); line-height:1.1; }
.brand-sub { font-size:11px; color:var(--ink-3); }

/* ── Generic card ── */
.card { background: var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
.eyebrow { font-size:10.5px; font-weight:600; color:var(--ink-3); text-transform:uppercase; letter-spacing:.08em; margin:0 0 8px; }
.side-text { font-size:12.5px; color:var(--ink-2); line-height:1.55; }
.empty { font-size:12.5px; color:var(--ink-3); font-style:italic; }

/* ── Chips / pills ── */
.chip { display:inline-flex; align-items:center; gap:5px; font-size:11.5px; color:var(--ink-2);
    background: var(--panel-2); border:1px solid var(--line); border-radius:999px; padding:3px 10px; margin:0 6px 6px 0; }
.pill { display:inline-flex; align-items:center; gap:6px; font-size:11px; font-weight:600; letter-spacing:.04em;
    text-transform:uppercase; border-radius:999px; padding:4px 10px; }
.pill .dot { width:7px; height:7px; border-radius:50%; background: currentColor; }
.pill.live { color:#5eead4; background: rgba(45,212,191,.12); }
.pill.live .dot { animation: pulse 1.4s infinite; }
.pill.idle { color:var(--ink-2); background: var(--panel-2); }
.pill.done { color:#86efac; background: rgba(12,163,12,.14); }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(94,234,212,.6)} 70%{box-shadow:0 0 0 7px rgba(94,234,212,0)} 100%{box-shadow:0 0 0 0 rgba(94,234,212,0)} }

/* ── Header bar ── */
.topbar { display:flex; align-items:center; justify-content:space-between; gap:18px; flex-wrap:wrap;
    background: var(--panel); border:1px solid var(--line); border-radius:16px; padding:14px 18px; margin-bottom:14px; }
.who { display:flex; align-items:center; gap:12px; }
.avatar-lg { width:44px; height:44px; border-radius:50%; display:grid; place-items:center; font-weight:700; font-size:17px;
    color:#fff; background: linear-gradient(135deg,#3987e5,#7c3aed); }
.who-name { font-size:17px; font-weight:700; color:var(--ink); }
.who-meta { font-size:12px; color:var(--ink-3); }
.stats { display:flex; gap:22px; align-items:center; flex-wrap:wrap; }
.stat-v { font-size:20px; font-weight:700; color:var(--ink); line-height:1.1; }
.stat-l { font-size:10.5px; color:var(--ink-3); text-transform:uppercase; letter-spacing:.06em; }
.progress { width:170px; }
.progress-track { height:6px; background: var(--panel-2); border-radius:999px; overflow:hidden; margin-top:6px; }
.progress-fill { height:100%; background: linear-gradient(90deg,#3987e5,#5eead4); border-radius:999px; transition: width .4s; }

/* ── Pane titles ── */
.pane-title { display:flex; align-items:center; justify-content:space-between; font-size:15px; font-weight:600; color:var(--ink); margin: 2px 2px 8px; }
.pane-title small { font-size:11.5px; color:var(--ink-3); font-weight:500; }

/* ── Chat ── */
.msg { display:flex; gap:10px; margin: 10px 0; align-items:flex-start; }
.msg.user { flex-direction: row-reverse; }
.avatar { flex:0 0 30px; height:30px; border-radius:50%; display:grid; place-items:center; font-size:12px; font-weight:700; color:#fff; }
.msg.advisor .avatar { background: linear-gradient(135deg,#3987e5,#1d4ed8); }
.msg.user .avatar { background: #334155; }
.bubble { max-width: 82%; padding: 10px 14px; border-radius: 4px 14px 14px 14px; font-size:14px; line-height:1.6; color:var(--ink);
    background: var(--panel-2); border:1px solid var(--line); }
.msg.user .bubble { border-radius: 14px 4px 14px 14px; background: #1d4ed8; border-color:#2563eb; color:#fff; }
.bubble .meta { display:flex; align-items:center; gap:6px; font-size:10.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-3); margin-bottom:4px; }
.bubble.concern { background: rgba(236,131,90,.10); border-color: rgba(236,131,90,.45); border-left: 3px solid var(--serious); }
.bubble.concern .meta { color: var(--serious); }
.bubble.emergency { background: rgba(208,59,59,.14); border-color: rgba(208,59,59,.55); border-left: 3px solid var(--crit); }
.bubble.emergency .meta { color: #f87171; }
.notify { margin-top:8px; padding:6px 10px; border-radius:8px; background: rgba(208,59,59,.22); color:#fecaca; font-size:12.5px; font-weight:600; }

/* ── Vitals ── */
.vgrid { display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:10px; }
.vcard { background: var(--panel-2); border:1px solid var(--line); border-radius:12px; padding:11px 12px; position:relative; overflow:hidden; }
.vcard::before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--good); }
.vcard.warn::before { background: var(--warn); }
.vcard.crit::before { background: var(--crit); }
.vcard.na::before { background: var(--line); }
.vhead { display:flex; justify-content:space-between; align-items:center; font-size:11px; color:var(--ink-2); font-weight:500; }
.vval { font-size:25px; font-weight:700; color:var(--ink); margin-top:4px; line-height:1.1; }
.vval span { font-size:11.5px; color:var(--ink-3); font-weight:500; margin-left:3px; }
.vdelta { font-size:11px; color:var(--ink-3); margin-top:3px; }
.vstatus { font-size:10px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; }
.vstatus.ok { color: var(--good); } .vstatus.warn { color: var(--warn); } .vstatus.crit { color: #f26464; } .vstatus.na { color: var(--ink-3); }

.section-hdr { font-size:10.5px; font-weight:600; color:var(--ink-3); text-transform:uppercase; letter-spacing:.08em; margin: 16px 0 8px; }
.trend-row { display:flex; justify-content:space-between; font-size:12.5px; padding:6px 0; border-bottom:1px solid var(--line); color:var(--ink-2); }
.trend-row b { font-weight:600; }
.trend-row .bad { color:#f59e8b; } .trend-row .good { color:#4ade80; } .trend-row .neutral { color:var(--ink-2); }
.feed { width:100%; border-collapse:collapse; font-size:12px; }
.feed th { text-align:left; color:var(--ink-3); font-weight:600; font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; padding:4px 6px; border-bottom:1px solid var(--line); }
.feed td { padding:5px 6px; color:var(--ink-2); border-bottom:1px solid var(--line); font-variant-numeric: tabular-nums; }

/* ── Landing ── */
.hero { padding: 28px 4px 10px; }
.hero h1 { font-size: 40px; font-weight: 700; letter-spacing:-.02em; margin:0; color:var(--ink); }
.hero h1 em { font-style:normal; background: linear-gradient(90deg,#60a5fa,#5eead4); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { font-size:16px; color:var(--ink-2); max-width:680px; margin-top:10px; line-height:1.6; }
.feature { height:100%; }
.feature .ic { font-size:22px; }
.feature h4 { margin:8px 0 4px; font-size:15px; color:var(--ink); }
.feature p { margin:0; font-size:13px; color:var(--ink-2); line-height:1.55; }
.scenario h3 { margin:0 0 2px; font-size:18px; color:var(--ink); }
.scenario .sub { font-size:12.5px; color:var(--ink-3); margin-bottom:10px; }
.scenario p { font-size:13.5px; color:var(--ink-2); line-height:1.55; margin:0 0 10px; }

/* Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a3754; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3b4a6b; }
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



# ═══ UI HELPERS ═══
SCENARIOS = {
    "a": {"name": "Alex", "age": 32, "icon": "👤", "tag": "Chronic stress",
          "blurb": "A 32-year-old under mounting work pressure. Watch AVA spot sleep, HRV and "
                   "recovery slipping over several days and step in gently."},
    "b": {"name": "Eleanor", "age": 72, "icon": "👵", "tag": "Acute emergency",
          "blurb": "A 72-year-old with hypertension and type 2 diabetes. Watch AVA detect an acute "
                   "event, escalate clearly and notify her emergency contact."},
}

# key, label, icon, unit, baseline key, adverse direction, warn / alert thresholds, threshold mode
VITALS = [
    ("heart_rate", "Heart rate", "❤️", "bpm", "resting_hr", "high", 0.15, 0.35, "rel"),
    ("hrv", "HRV", "〰️", "ms", "typical_hrv", "low", 0.20, 0.40, "rel"),
    ("spo2", "SpO₂", "🫁", "%", "typical_spo2", "low", 2, 4, "abs"),
    ("breathing_rate", "Breathing", "🌬️", "/min", "typical_breathing_rate", "high", 0.20, 0.40, "rel"),
    ("stress_score", "Stress", "⚡", "/100", "typical_stress_score", "high", 20, 40, "abs"),
    ("recovery_score", "Recovery", "🔋", "/100", "typical_recovery_score", "low", 20, 40, "abs"),
]
STATUS_TEXT = {"ok": "✓ Normal", "warn": "! Watch", "crit": "▲ Alert", "na": "—"}

# Streamlit ≥1.57 can pin a scroll container to its newest content
AUTOSCROLL = {"autoscroll": True} if "autoscroll" in inspect.signature(st.container).parameters else {}


def esc(value):
    return html.escape(str(value))


def render_text(text):
    """Escape model/user text, then allow **bold** and line breaks."""
    out = esc(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return out.replace("\n", "<br>")


def fmt_num(x):
    return f"{x:.1f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def vital_status(value, baseline, direction, warn, crit, mode):
    if not isinstance(value, (int, float)) or not baseline:
        return "na", ""
    diff = value - baseline
    adverse = diff if direction == "high" else -diff
    magnitude = adverse / baseline if mode == "rel" else adverse
    status = "crit" if magnitude >= crit else "warn" if magnitude >= warn else "ok"
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "＝"
    return status, f"{arrow} {fmt_num(abs(diff))} vs {fmt_num(baseline)} baseline"


def start_session(scenario):
    st.session_state.scenario_select = scenario
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


def trend_chart(history, key, baseline, unit):
    rows = [{"time": pd.to_datetime(p.get("timestamp")), "value": p.get("vitals", {}).get(key)}
            for p in history if p.get("timestamp")]
    df = pd.DataFrame(rows).dropna()
    line = alt.Chart(df).mark_line(color="#3987e5", strokeWidth=2, point=alt.OverlayMarkDef(size=55, filled=True, color="#3987e5")).encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%b %d %H:%M", labelAngle=0, tickCount=4)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("time:T", title="Time", format="%b %d, %H:%M"),
                 alt.Tooltip("value:Q", title=f"Value ({unit})")],
    )
    layers = [line]
    if baseline:
        layers.append(alt.Chart(pd.DataFrame({"baseline": [baseline]})).mark_rule(
            color="#6b7890", strokeDash=[4, 4], strokeWidth=1.5).encode(
            y="baseline:Q", tooltip=[alt.Tooltip("baseline:Q", title="Your baseline")]))
    return (alt.layer(*layers).properties(height=180, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#1f2b45", domainColor="#2a3754", tickColor="#2a3754",
                            labelColor="#8a96ab", labelFont="Inter", labelFontSize=10.5))


def message_html(entry):
    if entry["role"] == "user":
        content = entry["content"]
        label = "You"
        if content.startswith("[Self-report]"):
            content = content.replace("[Self-report] ", "", 1)
            label = "📝 Self-report"
        return (f'<div class="msg user"><div class="avatar">You</div><div class="bubble">'
                f'<div class="meta" style="color:#bfdbfe">{label}</div>{render_text(content)}</div></div>')

    btype = entry.get("type", "advisor")
    ts = entry.get("timestamp", "")
    ts_str = f" · {esc(ts[11:16])}" if ts and len(ts) > 11 else ""
    if btype == "emergency":
        label = f"🚨 Emergency{ts_str}"
    elif btype == "concern":
        label = f"⚠️ Concern · Level {esc(entry.get('severity', ''))}{ts_str}"
    else:
        label = f"AVA{ts_str}"
    notify = ('<div class="notify">📞 Emergency contact has been notified.</div>'
              if btype == "emergency" and entry.get("notify_family") else "")
    bubble_cls = btype if btype in ("emergency", "concern") else ""
    return (f'<div class="msg advisor"><div class="avatar">A</div><div class="bubble {bubble_cls}">'
            f'<div class="meta">{label}</div>{render_text(entry["content"])}{notify}</div></div>')


# ═══════════════════════════════════════
# LEFT SIDEBAR
# ═══════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">🫀</div><div><div class="brand-name">AVA</div>'
                '<div class="brand-sub">Adaptive health advisor</div></div></div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="eyebrow">Scenario</div>', unsafe_allow_html=True)
    st.radio("Scenario", ["a", "b"], label_visibility="collapsed",
             format_func=lambda x: f"{SCENARIOS[x]['icon']} {SCENARIOS[x]['name']} ({SCENARIOS[x]['age']}) · {SCENARIOS[x]['tag']}",
             key="scenario_select")
    has_session = st.session_state.current_profile is not None
    st.button("↺ Restart session" if has_session else "▶ Start session", width="stretch",
              type="secondary" if has_session else "primary",
              on_click=lambda: start_session(st.session_state.scenario_select))

    if has_session:
        total = len(st.session_state.packets) if st.session_state.packets else 0
        current = st.session_state.current_packet_idx
        remaining = total - current

        st.divider()
        st.markdown('<div class="eyebrow">Telemetry stream</div>', unsafe_allow_html=True)
        if remaining > 0:
            c1, c2 = st.columns(2)
            with c1:
                if not st.session_state.auto_streaming:
                    if st.button("▶ Play", width="stretch", type="primary", key="start_stream"):
                        st.session_state.auto_streaming = True
                        st.rerun()
                else:
                    if st.button("⏸ Pause", width="stretch", key="stop_stream"):
                        st.session_state.auto_streaming = False
                        st.rerun()
            with c2:
                if st.button("⏩ Skip to end", width="stretch", key="stream_all"):
                    st.session_state.auto_streaming = False
                    with st.spinner(f"Processing {remaining} packets…"):
                        for i in range(current, total):
                            process_packet(st.session_state.packets[i])
                            st.session_state.packet_history.append(st.session_state.packets[i])
                            st.session_state.current_packet_idx = i + 1
                    st.rerun()
            st.caption(f"{current} of {total} packets processed")
        else:
            st.markdown(f'<span class="pill done"><span class="dot"></span>All {total} packets processed</span>',
                        unsafe_allow_html=True)

        profile = st.session_state.current_profile
        identity = profile.get("identity", {})

        st.divider()
        st.markdown('<div class="eyebrow">Health profile</div>', unsafe_allow_html=True)
        chips = [f"🏥 {esc(c)}" for c in identity.get("known_conditions", [])]
        chips += [f"💊 {esc(m)}" for m in identity.get("medications", [])]
        ec = identity.get("emergency_contact", {})
        if ec:
            chips.append(f"📞 {esc(ec.get('name', ''))} · {esc(ec.get('contact', ''))}")
        if not identity.get("known_conditions"):
            chips.insert(0, "✓ No known conditions")
        st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)

        st.markdown('<div class="eyebrow" style="margin-top:14px">Active concerns</div>', unsafe_allow_html=True)
        concerns = profile.get("current_concerns", [])
        if concerns:
            st.markdown("".join(f'<div class="side-text">• {esc(c)}</div>' for c in concerns[-3:]), unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">Nothing flagged yet</div>', unsafe_allow_html=True)

        st.markdown('<div class="eyebrow" style="margin-top:14px">Learned patterns</div>', unsafe_allow_html=True)
        patterns = profile.get("known_patterns", [])
        if patterns:
            st.markdown("".join(f'<span class="chip">{esc(p)}</span>' for p in patterns[-4:]), unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty">No patterns yet</div>', unsafe_allow_html=True)

        st.divider()
        with st.form("self_report_form", clear_on_submit=True, border=False):
            note = st.text_area("📝 Self-report", height=80,
                                placeholder="e.g. Feeling off today, started new meds…")
            submitted = st.form_submit_button("Send to AVA", width="stretch")
            if submitted and note:
                send_chat(f"[Self-report] {note}")
                st.rerun()


# ═══════════════════════════════════════
# LANDING
# ═══════════════════════════════════════
if not st.session_state.current_profile:
    st.markdown('<div class="hero"><h1>Meet <em>AVA</em>, your adaptive health advisor</h1>'
                '<p>AVA watches a stream of wearable telemetry, compares it against your personal baseline, '
                'and only speaks up when something matters — in a voice that adapts to how you like to be spoken to.</p></div>',
                unsafe_allow_html=True)

    features = [
        ("📡", "Baseline-aware monitoring", "Every reading is judged against your own normal, not a population average."),
        ("🧠", "Pattern detection", "A multi-agent pipeline confirms sustained deviations before raising a concern."),
        ("💬", "Adaptive tone", "Ask for brief, detailed or warmer replies and AVA's living profile learns it."),
    ]
    for col, (icon, title, body) in zip(st.columns(3), features):
        col.markdown(f'<div class="card feature"><div class="ic">{icon}</div><h4>{title}</h4><p>{body}</p></div>',
                     unsafe_allow_html=True)

    st.markdown('<div class="section-hdr" style="margin-top:28px">Choose a scenario</div>', unsafe_allow_html=True)
    for col, (sid, sc) in zip(st.columns(2), SCENARIOS.items()):
        with col.container(border=True):
            st.markdown(f'<div class="scenario"><h3>{sc["icon"]} {sc["name"]}, {sc["age"]}</h3>'
                        f'<div class="sub">{sc["tag"]}</div><p>{sc["blurb"]}</p></div>', unsafe_allow_html=True)
            st.button(f"Start with {sc['name']}", key=f"land_{sid}", type="primary", width="stretch",
                      on_click=start_session, args=(sid,))
    st.stop()


# ═══════════════════════════════════════
# HEADER BAR
# ═══════════════════════════════════════
profile = st.session_state.current_profile
identity = profile.get("identity", {})
baselines = profile.get("baselines", {})
name = identity.get("name", "User")
total = len(st.session_state.packets) or 1
current = st.session_state.current_packet_idx
pct = int(100 * current / total)

if st.session_state.auto_streaming:
    pill = '<span class="pill live"><span class="dot"></span>Live</span>'
elif current >= len(st.session_state.packets):
    pill = '<span class="pill done"><span class="dot"></span>Complete</span>'
else:
    pill = f'<span class="pill idle"><span class="dot"></span>{"Ready" if current == 0 else "Paused"}</span>'

peak = max((e.get("severity") or 0 for e in st.session_state.conversation if e["role"] == "advisor"), default=0)
scenario_tag = SCENARIOS.get((st.session_state.current_user_id or "_a")[-1], {}).get("tag", "")
stats = [("Deviations", st.session_state.deviation_count), ("Patterns", st.session_state.pattern_count),
         ("Alerts", st.session_state.messages_sent), ("Peak level", peak or "—")]

st.markdown(
    '<div class="topbar">'
    f'<div class="who"><div class="avatar-lg">{esc(name[:1])}</div><div>'
    f'<div class="who-name">{esc(name)}</div>'
    f'<div class="who-meta">Age {esc(identity.get("age", "?"))} · {esc(scenario_tag)} · {esc(profile.get("days_monitored", 0))} days monitored</div>'
    '</div></div>'
    '<div class="stats">'
    + "".join(f'<div><div class="stat-v">{v}</div><div class="stat-l">{l}</div></div>' for l, v in stats)
    + f'<div class="progress"><div class="stat-l" style="display:flex;justify-content:space-between">'
      f'<span>Telemetry</span><span>{current}/{len(st.session_state.packets)}</span></div>'
      f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div></div>'
    f'{pill}</div></div>',
    unsafe_allow_html=True,
)

chat_col, vitals_col = st.columns([11, 9], gap="medium")

# ─── LEFT: Chat ───
with chat_col:
    n_msgs = len(st.session_state.conversation)
    st.markdown(f'<div class="pane-title">Conversation<small>{n_msgs} message{"" if n_msgs == 1 else "s"}</small></div>',
                unsafe_allow_html=True)
    with st.container(height=600, border=True, **AUTOSCROLL):
        st.markdown("".join(message_html(e) for e in st.session_state.conversation), unsafe_allow_html=True)

    user_input = st.chat_input(f"Ask AVA anything, {name}…")
    if user_input:
        send_chat(user_input)
        st.rerun()


# ─── RIGHT: Live Vitals ───
with vitals_col:
    packet = st.session_state.last_packet_data
    ts = packet.get("timestamp", "") if packet else ""
    when = datetime.fromisoformat(ts).strftime("%a %d %b · %H:%M") if ts else "No data yet"
    st.markdown(f'<div class="pane-title">Live vitals<small>{esc(when)}</small></div>', unsafe_allow_html=True)

    with st.container(height=600, border=True):
        if not packet:
            st.markdown('<div style="text-align:center;padding:120px 20px"><div style="font-size:42px">📡</div>'
                        '<div style="font-size:15px;font-weight:600;margin-top:8px">Waiting for telemetry</div>'
                        '<div class="empty" style="margin-top:6px">Press ▶ Play in the sidebar to start streaming.</div></div>',
                        unsafe_allow_html=True)
        else:
            vitals = packet.get("vitals", {})
            movement = packet.get("movement", {})
            sleep = packet.get("sleep_last_night", {})
            context = packet.get("context", {})

            chips = [f"🏃 {esc(movement.get('activity_state', '—'))}",
                     f"📍 {esc(context.get('location_zone', '—'))}",
                     f"🕐 {esc(context.get('time_of_day', '—'))}",
                     f"👣 {movement.get('steps_today', 0):,} steps"]
            if sleep.get("total_hours"):
                chips.append(f"😴 {esc(sleep['total_hours'])}h · {esc(sleep.get('sleep_efficiency', '?'))}% eff.")
            st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)

            cards = []
            for key, label, icon, unit, bkey, direction, warn, crit, mode in VITALS:
                val = vitals.get(key)
                status, delta = vital_status(val, baselines.get(bkey), direction, warn, crit, mode)
                cards.append(
                    f'<div class="vcard {status}"><div class="vhead"><span>{icon} {label}</span>'
                    f'<span class="vstatus {status}">{STATUS_TEXT[status]}</span></div>'
                    f'<div class="vval">{esc(fmt_num(val) if val is not None else "—")}<span>{unit}</span></div>'
                    f'<div class="vdelta">{delta or "&nbsp;"}</div></div>')
            st.markdown(f'<div class="vgrid">{"".join(cards)}</div>', unsafe_allow_html=True)

            history = st.session_state.packet_history
            if len(history) >= 2:
                st.markdown('<div class="section-hdr">History · dashed line is your baseline</div>', unsafe_allow_html=True)
                chart_vitals = [v for v in VITALS if v[0] in ("heart_rate", "hrv", "stress_score", "spo2")]
                for tab, (key, label, _, unit, bkey, *_rest) in zip(st.tabs([v[1] for v in chart_vitals]), chart_vitals):
                    with tab:
                        st.altair_chart(trend_chart(history, key, baselines.get(bkey), unit), theme=None, width="stretch")

            trends = {k: v for k, v in profile.get("current_state", {}).items() if v and v != "unknown"}
            if trends:
                st.markdown('<div class="section-hdr">7-day trends</div>', unsafe_allow_html=True)
                rows = []
                for k, v in trends.items():
                    tlabel = k.replace("_7d", "").replace("_trend", "").replace("_", " ").title()
                    tone = "bad" if v in ("declining", "deteriorating", "rising") else "good" if v == "improving" else "neutral"
                    arrow = {"bad": "↘" if v != "rising" else "↗", "good": "↗", "neutral": "→"}[tone]
                    rows.append(f'<div class="trend-row"><span>{esc(tlabel)}</span><b class="{tone}">{arrow} {esc(v)}</b></div>')
                st.markdown("".join(rows), unsafe_allow_html=True)

            if history:
                st.markdown('<div class="section-hdr">Recent packets</div>', unsafe_allow_html=True)
                body = "".join(
                    f'<tr><td>{esc(datetime.fromisoformat(p["timestamp"]).strftime("%d %b %H:%M")) if p.get("timestamp") else "—"}</td>'
                    f'<td>{esc(p.get("vitals", {}).get("heart_rate", "—"))}</td>'
                    f'<td>{esc(p.get("vitals", {}).get("hrv", "—"))}</td>'
                    f'<td>{esc(p.get("vitals", {}).get("spo2", "—"))}</td>'
                    f'<td>{esc(p.get("vitals", {}).get("stress_score", "—"))}</td></tr>'
                    for p in reversed(history[-6:]))
                st.markdown(f'<table class="feed"><thead><tr><th>Time</th><th>HR</th><th>HRV</th><th>SpO₂</th>'
                            f'<th>Stress</th></tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)


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
