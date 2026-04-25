"""
Analyst Agent

Receives confirmed patterns from the Profiler, retrieves historical and
clinical context via RAG (episodic memory + knowledge base), classifies
severity 1-5, applies timing checks, and produces a structured assessment.

Uses Groq LLM for reasoning about severity and recommended action.
"""

import os
import json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

from graph.state import HealthAgentState
from memory.episodic_memory import query_similar, get_recent
from knowledge_base.loader import query_knowledge_base
from memory.living_profile import load_profile, add_concern

load_dotenv()

# Groq client for the Analyst agent
analyst_client = Groq(api_key=os.getenv("GROQ_API_KEY_ANALYST", "missing_key_please_add_to_secrets"))

MESSAGE_COOLDOWN_MINUTES = int(os.getenv("MESSAGE_COOLDOWN_MINUTES", 120))

# Track last message time per user (in-memory, resets on restart)
_last_message_times = {}


def analyst_node(state: HealthAgentState) -> dict:
    """Analyst node for the LangGraph health agent.

    Evaluates confirmed patterns, retrieves RAG context, classifies severity,
    applies timing constraints, and builds a structured analyst output.

    Args:
        state: The current HealthAgentState.

    Returns:
        dict: State updates with severity_level, analyst_output,
              proceed_to_communicate, and notify_family.
    """
    # Step 1: Guard — only run if pattern is confirmed or deviation detected
    if not state.get("pattern_confirmed", False) and not state.get("deviation_detected", False):
        return {
            "severity_level": None,
            "analyst_output": None,
            "proceed_to_communicate": False,
            "notify_family": False,
        }

    # If only deviation detected (not confirmed pattern), skip deep analysis
    if not state.get("pattern_confirmed", False):
        return {
            "severity_level": 1,
            "analyst_output": {
                "severity_level": 1,
                "trigger": "single_deviation",
                "key_facts": ["Single deviation detected, monitoring for pattern"],
                "historical_context": "No confirmed pattern yet",
                "clinical_context": "Single readings may be transient",
                "recommended_action": "Continue monitoring",
                "question_to_ask": None,
                "notify_family": False,
                "add_to_current_concerns": False,
            },
            "proceed_to_communicate": False,
            "notify_family": False,
        }

    profile = state["living_profile"]
    pattern_details = state.get("pattern_details", {})
    packet = state.get("current_packet", {})
    user_id = packet.get("user_id", "unknown")
    context = packet.get("context", {})

    # Step 2: RAG — retrieve similar past episodes from episodic memory
    confirmed_metrics = pattern_details.get("confirmed_metrics", [])
    deviations = pattern_details.get("deviations", {})

    pattern_description = _build_pattern_description(confirmed_metrics, deviations, pattern_details)
    historical_episodes = query_similar(pattern_description, n_results=3,
                                         where={"user_id": user_id})

    # Step 3: RAG — retrieve relevant clinical knowledge
    signal_description = _build_signal_description(confirmed_metrics, pattern_details)
    clinical_docs = query_knowledge_base(signal_description, n_results=2)

    # Format RAG results for the LLM
    historical_context = _format_historical_context(historical_episodes)
    clinical_context = _format_clinical_context(clinical_docs)

    # Step 4: Build context for severity classification via LLM
    profile_summary = _build_profile_summary(profile)

    # Call Groq LLM for analysis
    analyst_output = _call_analyst_llm(
        profile_summary=profile_summary,
        pattern_details=pattern_details,
        historical_context=historical_context,
        clinical_context=clinical_context,
    )

    severity_level = analyst_output.get("severity_level", 2)
    notify_family = analyst_output.get("notify_family", False)

    # Step 5: Timing checks
    proceed_to_communicate = True

    # Check if user is sleeping and severity <= 3
    time_of_day = context.get("time_of_day", "")
    is_sleeping = time_of_day in ("night", "late_night") or (
        "02:" in packet.get("timestamp", "") or
        "03:" in packet.get("timestamp", "") or
        "04:" in packet.get("timestamp", "")
    )
    if is_sleeping and severity_level <= 3:
        proceed_to_communicate = False

    # Check battery and location constraints
    battery = context.get("battery_level", 100)
    location = context.get("location_zone", "known")
    if battery < 10 and location == "unknown" and severity_level <= 2:
        proceed_to_communicate = False

    # Check message cooldown
    now = datetime.now()
    last_msg_time = _last_message_times.get(user_id)
    if last_msg_time and severity_level <= 2:
        minutes_since = (now - last_msg_time).total_seconds() / 60
        if minutes_since < MESSAGE_COOLDOWN_MINUTES:
            proceed_to_communicate = False

    # Severity 4+ always communicates
    if severity_level >= 4:
        proceed_to_communicate = True

    # Severity 5 always notifies family
    if severity_level >= 5:
        notify_family = True

    # Update last message time if we're going to communicate
    if proceed_to_communicate:
        _last_message_times[user_id] = now

    # Add to current concerns if analyst recommends it
    if analyst_output.get("add_to_current_concerns") and severity_level >= 3:
        trigger = analyst_output.get("trigger", "Unknown pattern")
        concern_text = f"Level {severity_level}: {trigger} (detected {datetime.now().strftime('%b %d %H:%M')})"
        add_concern(user_id, concern_text)

    return {
        "severity_level": severity_level,
        "analyst_output": analyst_output,
        "proceed_to_communicate": proceed_to_communicate,
        "notify_family": notify_family,
    }


def _build_pattern_description(confirmed_metrics: list, deviations: dict, pattern_details: dict) -> str:
    """Build a natural language description of the confirmed pattern.

    Args:
        confirmed_metrics: List of confirmed metric names.
        deviations: Dict of deviation results.
        pattern_details: Full pattern details dict.

    Returns:
        str: Pattern description for RAG query.
    """
    parts = []
    for metric in confirmed_metrics:
        dev = deviations.get(metric, {})
        current = dev.get("current", "unknown")
        baseline = dev.get("baseline", "unknown")
        parts.append(f"{metric}: current={current}, baseline={baseline}")

    vitals = pattern_details.get("vitals_snapshot", {})
    activity = pattern_details.get("activity_state", "unknown")
    ctx = pattern_details.get("context", {})
    time_of_day = ctx.get("time_of_day", "unknown")

    return (
        f"Confirmed pattern: {', '.join(confirmed_metrics)}. "
        f"Activity: {activity}. Time: {time_of_day}. "
        f"Details: {'; '.join(parts)}"
    )


def _build_signal_description(confirmed_metrics: list, pattern_details: dict) -> str:
    """Build a signal description for knowledge base query.

    Args:
        confirmed_metrics: List of confirmed metric names.
        pattern_details: Full pattern details dict.

    Returns:
        str: Signal description for KB query.
    """
    vitals = pattern_details.get("vitals_snapshot", {})
    activity = pattern_details.get("activity_state", "unknown")
    ctx = pattern_details.get("context", {})

    signal_parts = []
    for metric in confirmed_metrics:
        if "hr" in metric:
            signal_parts.append(f"heart rate {vitals.get('heart_rate', 'unknown')}")
        if "hrv" in metric:
            signal_parts.append(f"HRV {vitals.get('hrv', 'unknown')}")
        if "spo2" in metric:
            signal_parts.append(f"SpO2 {vitals.get('spo2', 'unknown')}")
        if "stress" in metric:
            signal_parts.append(f"stress score {vitals.get('stress_score', 'unknown')}")
        if "recovery" in metric:
            signal_parts.append(f"recovery score {vitals.get('recovery_score', 'unknown')}")
        if "temp" in metric:
            signal_parts.append(f"skin temperature {vitals.get('skin_temperature', 'unknown')}")
        if "breathing" in metric:
            signal_parts.append(f"breathing rate {vitals.get('breathing_rate', 'unknown')}")

    return (
        f"{' '.join(signal_parts)} while {activity} "
        f"at {ctx.get('time_of_day', 'unknown')} "
        f"location {ctx.get('location_zone', 'unknown')}"
    )


def _format_historical_context(episodes: list) -> str:
    """Format historical episodes into a readable context string.

    Args:
        episodes: List of episode dicts from query_similar.

    Returns:
        str: Formatted historical context.
    """
    if not episodes:
        return "No relevant historical episodes found."

    parts = []
    for i, ep in enumerate(episodes, 1):
        meta = ep.get("metadata", {})
        parts.append(
            f"Episode {i}: {meta.get('event_type', 'unknown')} "
            f"({meta.get('significance', 'unknown')}) at {meta.get('timestamp', 'unknown')} — "
            f"{ep.get('document', '')[:200]}"
        )

    return "\n".join(parts)


def _format_clinical_context(docs: list) -> str:
    """Format knowledge base documents into a readable clinical context string.

    Args:
        docs: List of document dicts from query_knowledge_base.

    Returns:
        str: Formatted clinical context.
    """
    if not docs:
        return "No relevant clinical knowledge found."

    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        parts.append(
            f"KB {i}: {meta.get('title', 'unknown')} "
            f"(severity suggestion: {meta.get('severity_suggestion', '?')}) — "
            f"{meta.get('recommended_action', 'N/A')}"
        )

    return "\n".join(parts)


def _build_profile_summary(profile: dict) -> str:
    """Build a concise profile summary for the LLM.

    Args:
        profile: The user's Living Profile dict.

    Returns:
        str: Concise profile summary.
    """
    identity = profile.get("identity", {})
    baselines = profile.get("baselines", {})
    current_state = profile.get("current_state", {})
    concerns = profile.get("current_concerns", [])

    return (
        f"Name: {identity.get('name', 'Unknown')}. "
        f"Age: {identity.get('age', 'Unknown')}. "
        f"Conditions: {', '.join(identity.get('known_conditions', [])) or 'None'}. "
        f"Medications: {', '.join(identity.get('medications', [])) or 'None'}. "
        f"Baseline HR: {baselines.get('resting_hr', 'N/A')}. "
        f"Baseline HRV: {baselines.get('typical_hrv', 'N/A')}. "
        f"Baseline SpO2: {baselines.get('typical_spo2', 'N/A')}. "
        f"Sleep trend: {current_state.get('sleep_trend_7d', 'unknown')}. "
        f"Stress trend: {current_state.get('stress_trend_7d', 'unknown')}. "
        f"Current concerns: {', '.join(concerns) or 'None'}."
    )


def _call_analyst_llm(profile_summary: str, pattern_details: dict,
                       historical_context: str, clinical_context: str) -> dict:
    """Call Groq LLM for structured analyst assessment.

    Args:
        profile_summary: Concise profile summary string.
        pattern_details: Full pattern details dict.
        historical_context: Formatted historical episode context.
        clinical_context: Formatted clinical knowledge context.

    Returns:
        dict: Parsed JSON with analyst output keys.
    """
    system_prompt = (
        "You are the Analyst agent. Assess the health pattern and determine severity and action. "
        "Return JSON with the exact keys: severity_level, trigger, key_facts, historical_context, "
        "clinical_context, recommended_action, question_to_ask, notify_family, add_to_current_concerns. "
        "Do not diagnose. Do not add extra keys.\n\n"
        "Severity levels:\n"
        "1 = interesting pattern, not urgent, queue for later\n"
        "2 = actionable now, not alarming, send within hour\n"
        "3 = needs user context, ask a question now\n"
        "4 = abnormal, alert user immediately\n"
        "5 = potential medical event, emergency protocol\n"
    )

    confirmed = pattern_details.get("confirmed_metrics", [])
    vitals = pattern_details.get("vitals_snapshot", {})
    context = pattern_details.get("context", {})
    activity = pattern_details.get("activity_state", "unknown")

    user_prompt = (
        f"USER PROFILE:\n{profile_summary}\n\n"
        f"CONFIRMED PATTERN:\n"
        f"Metrics: {', '.join(confirmed)}\n"
        f"Vitals: {json.dumps(vitals)}\n"
        f"Activity: {activity}\n"
        f"Context: {json.dumps(context)}\n\n"
        f"HISTORICAL EPISODES:\n{historical_context}\n\n"
        f"CLINICAL KNOWLEDGE:\n{clinical_context}\n\n"
        f"Return your assessment as valid JSON only. No markdown, no explanation."
    )

    try:
        response = analyst_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )

        content = response.choices[0].message.content.strip()

        # Robust JSON parsing — try multiple strategies
        parsed = None

        # Strategy 1: direct parse
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip markdown fences
        if parsed is None and "```" in content:
            try:
                lines = content.split("\n")
                inner = "\n".join(l for l in lines if not l.strip().startswith("```"))
                parsed = json.loads(inner)
            except json.JSONDecodeError:
                pass

        # Strategy 3: regex extract first JSON object
        if parsed is None:
            import re
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if parsed is not None:
            return parsed

        # All parsing failed — use rule-based fallback silently
        severity = _estimate_severity_fallback(pattern_details)
        return {
            "severity_level": severity,
            "trigger": ", ".join(confirmed),
            "key_facts": [f"Pattern confirmed for {', '.join(confirmed)}"],
            "historical_context": historical_context[:200],
            "clinical_context": clinical_context[:200],
            "recommended_action": "Monitor closely and alert if pattern persists",
            "question_to_ask": None,
            "notify_family": severity >= 5,
            "add_to_current_concerns": severity >= 3,
        }

    except Exception as e:
        print(f"[Analyst] API error: {e}")
        # Return a reasonable fallback based on pattern details
        severity = _estimate_severity_fallback(pattern_details)
        return {
            "severity_level": severity,
            "trigger": ", ".join(confirmed),
            "key_facts": [f"Pattern confirmed for {', '.join(confirmed)}"],
            "historical_context": historical_context[:200],
            "clinical_context": clinical_context[:200],
            "recommended_action": "Monitor closely and alert if pattern persists",
            "question_to_ask": None,
            "notify_family": severity >= 5,
            "add_to_current_concerns": severity >= 3,
        }


def _estimate_severity_fallback(pattern_details: dict) -> int:
    """Estimate severity when LLM call fails, based on confirmed metrics.

    Args:
        pattern_details: Full pattern details dict.

    Returns:
        int: Estimated severity level 1-5.
    """
    confirmed = pattern_details.get("confirmed_metrics", [])
    vitals = pattern_details.get("vitals_snapshot", {})
    activity = pattern_details.get("activity_state", "unknown")
    context = pattern_details.get("context", {})

    severity = 2  # Default for confirmed pattern

    # Critical: SpO2 below 93 while sedentary
    if "spo2_low" in confirmed and vitals.get("spo2", 99) < 93:
        severity = max(severity, 4)

    # Critical: HR > 120 while sedentary at night
    if "hr_elevated" in confirmed and vitals.get("heart_rate", 0) > 120:
        if activity == "sedentary":
            time_of_day = context.get("time_of_day", "")
            if time_of_day in ("night", "late_night"):
                severity = max(severity, 5)
            else:
                severity = max(severity, 4)

    # Multiple concurrent pattern confirmations
    if len(confirmed) >= 3:
        severity = max(severity, 3)

    # HR + SpO2 combo while sedentary at night
    if "hr_elevated" in confirmed and "spo2_low" in confirmed:
        if activity == "sedentary":
            severity = max(severity, 5)

    return severity
