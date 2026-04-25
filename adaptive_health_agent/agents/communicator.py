"""
Communicator Agent

Formats health insights for the user based on their communication profile.
Adapts tone, directness, depth, and length to personal preferences.
Handles emergency alerts (Level 5) with structured console output.

Uses Groq LLM to generate personalized health messages.
Includes RAG retrieval from episodic memory and medical KB for user-initiated chats.
"""

import os
import json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

from graph.state import HealthAgentState
from memory.episodic_memory import log_episode, query_similar, get_recent
from knowledge_base.loader import query_knowledge_base
from memory.living_profile import update_communication_profile

load_dotenv()

# Groq client for the Communicator agent
communicator_client = Groq(api_key=os.getenv("GROQ_API_KEY_COMMUNICATOR"))


def communicator_node(state: HealthAgentState) -> dict:
    """Communicator node for the LangGraph health agent.

    Formats the analyst's assessment into a user-facing message adapted
    to the user's communication preferences. Handles emergency alerts.

    Args:
        state: The current HealthAgentState.

    Returns:
        dict: State updates with final_message, notify_family, agent_response.
    """
    profile = state["living_profile"]
    analyst_output = state.get("analyst_output")
    severity_level = state.get("severity_level", 1)
    user_message = state.get("user_message")
    packet = state.get("current_packet", {})
    user_id = packet.get("user_id", profile.get("identity", {}).get("name", "unknown").lower().replace(" ", "_"))
    timestamp = packet.get("timestamp", datetime.now().isoformat())

    # Step 1: Read communication profile
    comm_profile = profile.get("communication_profile", {})

    # Step 2: Build style instruction from communication profile scores
    style_instruction = _build_style_instruction(comm_profile)

    # Handle user-initiated messages (bypass analyst)
    if user_message:
        agent_response = _handle_user_message(user_message, profile, style_instruction, user_id, timestamp)
        return {
            "final_message": None,
            "notify_family": False,
            "agent_response": agent_response,
        }

    # Guard: only communicate if analyst says to
    if not state.get("proceed_to_communicate", False) or not analyst_output:
        return {
            "final_message": None,
            "notify_family": False,
            "agent_response": None,
        }

    # Step 3: Call Groq LLM to format the message
    final_message = _call_communicator_llm(analyst_output, style_instruction, severity_level)

    # Step 3b: Append proactive question from profiler if available
    pattern_details = state.get("pattern_details") or {}
    proactive_question = pattern_details.get("proactive_question")
    if proactive_question and severity_level <= 3:
        final_message = final_message + "\n\n💭 " + proactive_question

    # Step 4: Append disclaimer for severity >= 3
    if severity_level >= 3:
        disclaimer = _get_disclaimer(comm_profile)
        final_message = final_message + "\n\n" + disclaimer

    # Step 5: Handle severity 5 — emergency alert
    notify_family = False
    if severity_level >= 5:
        notify_family = True
        _print_emergency_alert(
            timestamp=timestamp,
            name=profile.get("identity", {}).get("name", "Unknown"),
            emergency_contact=profile.get("identity", {}).get("emergency_contact"),
            key_facts=analyst_output.get("key_facts", []),
            clinical_context=analyst_output.get("clinical_context", ""),
        )

    # Step 6: Log interaction to ChromaDB
    episode = {
        "id": f"episode_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{user_id}",
        "timestamp": timestamp,
        "user_id": user_id,
        "event_type": "conversation",
        "metrics_snapshot": state.get("current_packet", {}).get("vitals", {}),
        "context_snapshot": {
            "severity_level": severity_level,
            "trigger": analyst_output.get("trigger", ""),
        },
        "deviation_from_baseline": {},
        "significance": "pattern_confirmed" if severity_level >= 3 else "single_occurrence",
        "agent_action_taken": f"Sent Level {severity_level} message to user",
        "user_response": None,
        "outcome": None,
        "tags": ["communication", f"severity_{severity_level}"],
    }
    log_episode(episode)

    # Step 7: Set state values
    return {
        "final_message": final_message,
        "notify_family": notify_family,
        "agent_response": None,
    }


def _build_style_instruction(comm_profile: dict) -> str:
    """Build a natural language style instruction from communication profile scores.

    Each dimension maps from 1-5 to specific style characteristics.

    Args:
        comm_profile: The user's communication_profile dict.

    Returns:
        str: Style instruction string for the LLM.
    """
    style = comm_profile.get("style", "balanced")
    directness = comm_profile.get("directness", 3)
    depth = comm_profile.get("depth", 3)
    tone = comm_profile.get("tone", 3)
    length = comm_profile.get("length", 3)
    framing = comm_profile.get("framing", 3)

    parts = [f"Communication style: {style}."]

    # Directness: 1 = blunt/clinical, 5 = gentle/indirect
    if directness <= 2:
        parts.append("Be direct and blunt. State facts plainly without softening.")
    elif directness >= 4:
        parts.append("Be gentle and indirect. Ease into the information with empathy.")
    else:
        parts.append("Balance directness with sensitivity.")

    # Depth: 1 = show all data/numbers, 5 = high-level summary only
    if depth <= 2:
        parts.append("Include specific numbers, percentages, and data points.")
    elif depth >= 4:
        parts.append("Give a high-level summary. Avoid raw numbers unless critical.")
    else:
        parts.append("Include key numbers but explain them in context.")

    # Tone: 1 = clinical/professional, 5 = warm/casual/friendly
    if tone <= 2:
        parts.append("Use a clinical, professional tone. Avoid casual language.")
    elif tone >= 4:
        parts.append("Use a warm, friendly, conversational tone. Be approachable.")
    else:
        parts.append("Use a neutral, approachable tone.")

    # Length: 1 = detailed/comprehensive, 5 = brief/concise
    if length <= 2:
        parts.append("Provide a detailed, comprehensive response.")
    elif length >= 4:
        parts.append("Keep it brief. Use 2-3 sentences maximum.")
    else:
        parts.append("Keep it moderate length — a few sentences with key points.")

    # Framing: 1 = problem-focused, 5 = solution/positive-focused
    if framing <= 2:
        parts.append("Focus on the problem and what's happening with their health.")
    elif framing >= 4:
        parts.append("Focus on solutions and positive actions they can take.")
    else:
        parts.append("Balance problem identification with actionable suggestions.")

    return " ".join(parts)


def _get_disclaimer(comm_profile: dict) -> str:
    """Get a disclaimer adapted to the user's communication style.

    Args:
        comm_profile: The user's communication_profile dict.

    Returns:
        str: Adapted disclaimer string.
    """
    tone = comm_profile.get("tone", 3)

    if tone <= 2:
        return ("Note: If this pattern persists, medical consultation is recommended. "
                "I can provide tracking data to share with your healthcare provider.")
    elif tone >= 4:
        return ("If this keeps up, it might be worth chatting with your doctor about it. "
                "I can help you track it so you have something concrete to share! 😊")
    else:
        return ("If this pattern continues, it's worth mentioning to your doctor. "
                "I can help you track it.")


def _call_communicator_llm(analyst_output: dict, style_instruction: str, severity_level: int) -> str:
    """Call Groq LLM to format the health message in the user's preferred style.

    Args:
        analyst_output: The analyst's structured assessment dict.
        style_instruction: Natural language style instruction string.
        severity_level: The severity classification (1-5).

    Returns:
        str: The formatted message for the user.
    """
    system_prompt = (
        "You are AVA (Adaptive Virtual Assistant), a highly advanced personal health advisor living on a smartwatch. "
        "Speak in the first person as AVA. Keep responses SHORT — 1 to 2 sentences max. "
        "No preamble, no lists. Adapt tone to the style instructions. Be warm but concise."
    )

    user_prompt = (
        f"STYLE INSTRUCTIONS:\n{style_instruction}\n\n"
        f"SEVERITY LEVEL: {severity_level}\n\n"
        f"ANALYST ASSESSMENT:\n"
        f"Trigger: {analyst_output.get('trigger', 'unknown')}\n"
        f"Key facts: {json.dumps(analyst_output.get('key_facts', []))}\n"
        f"Clinical context: {analyst_output.get('clinical_context', 'N/A')}\n"
        f"Recommended action: {analyst_output.get('recommended_action', 'N/A')}\n"
        f"Question to ask: {analyst_output.get('question_to_ask', 'None')}\n\n"
        f"Write 1-2 sentences ONLY. No JSON, no markdown."
    )

    try:
        response = communicator_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=150,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Communicator] LLM call error: {e}")
        # Fallback message
        key_facts = analyst_output.get("key_facts", ["Health pattern detected"])
        action = analyst_output.get("recommended_action", "Please monitor your health closely")
        return f"Health update: {'. '.join(key_facts)}. {action}."


def _handle_user_message(user_message: str, profile: dict,
                          style_instruction: str, user_id: str, timestamp: str) -> str:
    """Handle a user-initiated message by responding in their preferred style.

    Args:
        user_message: The user's message text.
        profile: The user's Living Profile dict.
        style_instruction: Style instruction for the LLM.
        user_id: The user identifier.
        timestamp: Current timestamp.

    Returns:
        str: The agent's response to the user.
    """
    identity = profile.get("identity", {})
    baselines = profile.get("baselines", {})
    current_state = profile.get("current_state", {})
    concerns = profile.get("current_concerns", [])
    patterns = profile.get("known_patterns", [])

    # RAG Step 1: Query episodic memory for relevant personal history
    try:
        similar_episodes = query_similar(user_message, n_results=3,
                                          where={"user_id": user_id})
        recent_episodes = get_recent(user_id, days=7)
    except Exception as e:
        print(f"[Communicator] RAG episodic query error: {e}")
        similar_episodes = []
        recent_episodes = []

    # RAG Step 2: Query medical knowledge base for clinical context
    try:
        clinical_docs = query_knowledge_base(user_message, n_results=2)
    except Exception as e:
        print(f"[Communicator] RAG KB query error: {e}")
        clinical_docs = []

    # Format RAG context
    episodic_context = ""
    if similar_episodes:
        ep_parts = []
        for i, ep in enumerate(similar_episodes[:3], 1):
            meta = ep.get("metadata", {})
            ep_parts.append(
                f"  {i}. [{meta.get('event_type', '?')}] {meta.get('significance', '?')} "
                f"at {meta.get('timestamp', '?')} — {ep.get('document', '')[:150]}"
            )
        episodic_context = "\n".join(ep_parts)

    recent_context = ""
    if recent_episodes:
        rc_parts = []
        for i, ep in enumerate(recent_episodes[:5], 1):
            meta = ep.get("metadata", {})
            rc_parts.append(
                f"  {i}. [{meta.get('event_type', '?')}] {meta.get('timestamp', '?')} — "
                f"{meta.get('agent_action_taken', 'N/A')}"
            )
        recent_context = "\n".join(rc_parts)

    clinical_context = ""
    if clinical_docs:
        cl_parts = []
        for i, doc in enumerate(clinical_docs[:2], 1):
            meta = doc.get("metadata", {})
            cl_parts.append(
                f"  {i}. {meta.get('title', '?')} — {meta.get('recommended_action', 'N/A')}"
            )
        clinical_context = "\n".join(cl_parts)

    system_prompt = (
        "You are AVA (Adaptive Virtual Assistant), a highly advanced personal health advisor living on a smartwatch. "
        "The user is asking about their health or conversing with you. Respond as AVA in the first person. "
        "Keep responses SHORT — 2 to 3 sentences max. Be helpful, warm, and reference their personal data. "
        "Do not diagnose. No lists, no bullet points. Just a concise, caring response."
    )

    context_parts = [
        f"STYLE INSTRUCTIONS:\n{style_instruction}",
        f"\nUSER PROFILE:",
        f"Name: {identity.get('name', 'Unknown')}",
        f"Age: {identity.get('age', 'Unknown')}",
        f"Conditions: {', '.join(identity.get('known_conditions', [])) or 'None'}",
        f"Medications: {', '.join(identity.get('medications', [])) or 'None'}",
        f"\nCURRENT STATE:",
        f"Sleep trend: {current_state.get('sleep_trend_7d', 'unknown')}",
        f"Stress trend: {current_state.get('stress_trend_7d', 'unknown')}",
        f"HRV trend: {current_state.get('hrv_trend_7d', 'unknown')}",
        f"Recovery trend: {current_state.get('recovery_trend_7d', 'unknown')}",
        f"\nCurrent concerns: {', '.join(concerns) or 'None'}",
        f"Known patterns: {', '.join(patterns) or 'None'}",
        f"\nBASELINES:",
        f"Resting HR: {baselines.get('resting_hr', 'Learning...')} bpm",
        f"Typical HRV: {baselines.get('typical_hrv', 'Learning...')} ms",
        f"Typical SpO2: {baselines.get('typical_spo2', 'Learning...')}%",
        f"Typical sleep: {baselines.get('typical_sleep_hours', 'Learning...')} hrs",
        f"Typical sleep efficiency: {baselines.get('typical_sleep_efficiency', 'Learning...')}%",
        f"Typical stress score: {baselines.get('typical_stress_score', 'Learning...')}",
        f"Typical recovery score: {baselines.get('typical_recovery_score', 'Learning...')}",
    ]

    # Add RAG context if available
    if episodic_context:
        context_parts.append(f"\nRELEVANT PERSONAL HISTORY (from episodic memory):")
        context_parts.append(episodic_context)
    if recent_context:
        context_parts.append(f"\nRECENT HEALTH EVENTS (last 7 days):")
        context_parts.append(recent_context)
    if clinical_context:
        context_parts.append(f"\nCLINICAL KNOWLEDGE (evidence-based):")
        context_parts.append(clinical_context)

    user_prompt = "\n".join(context_parts) + f"\n\nUSER MESSAGE: {user_message}"

    try:
        response = communicator_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        agent_response = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Communicator] LLM call error: {e}")
        agent_response = "I'm having trouble processing your request right now. Please try again in a moment."

    # Log the conversation to episodic memory
    episode = {
        "id": f"episode_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{user_id}",
        "timestamp": timestamp,
        "user_id": user_id,
        "event_type": "conversation",
        "metrics_snapshot": {},
        "context_snapshot": {"user_initiated": True},
        "deviation_from_baseline": {},
        "significance": "single_occurrence",
        "agent_action_taken": f"Responded to user message: {user_message[:100]}",
        "user_response": user_message,
        "outcome": None,
        "tags": ["user_message", "conversation"],
    }
    log_episode(episode)

    return agent_response


def _print_emergency_alert(timestamp: str, name: str, emergency_contact,
                            key_facts: list, clinical_context: str) -> None:
    """Print a Level 5 emergency alert to the console in the exact required format.

    Args:
        timestamp: The event timestamp.
        name: The user's name.
        emergency_contact: The emergency contact dict (name, contact).
        key_facts: List of key fact strings.
        clinical_context: Clinical context string.
    """
    if emergency_contact and isinstance(emergency_contact, dict):
        contact_str = f"{emergency_contact.get('name', 'N/A')} — {emergency_contact.get('contact', 'N/A')}"
    else:
        contact_str = "Not configured"

    facts_str = "\n".join(f"  • {fact}" for fact in key_facts)

    print(
        f"\n========================================"
        f"\n⚠️  EMERGENCY ALERT"
        f"\n========================================"
        f"\nTime: {timestamp}"
        f"\nUser: {name}"
        f"\nContact: {contact_str}"
        f"\nSITUATION:"
        f"\n{facts_str}"
        f"\nAGENT ASSESSMENT:"
        f"\n{clinical_context}"
        f"\nACTION RECOMMENDED:"
        f"\nImmediate check-in with user recommended."
        f"\nConsider contacting emergency services if user is unresponsive."
        f"\n[This is an automated alert from the Health Agent]"
        f"\n========================================"
    )


def apply_communication_feedback(user_id: str, feedback_type: str) -> None:
    """Adjust communication profile based on observed user behavior.

    Step 8 from spec: Small ±0.5 increments, clamped 1-5.

    Args:
        user_id: The user identifier.
        feedback_type: One of:
            "dismissive" → directness toward 5
            "engages_data" → depth toward 1
            "ignores_long" → length toward 5
            "anxious" → framing toward 5
    """
    adjustments = {
        "dismissive": {"directness": 0.5},
        "engages_data": {"depth": -0.5},
        "ignores_long": {"length": 0.5},
        "anxious": {"framing": 0.5},
        "requests_warmth": {"tone": 2.0},       # Force tone warmer (up to 5)
        "requests_brief": {"directness": 2.0},  # Force directness (up to 5)
        "requests_detail": {"depth": -2.0},     # Force depth (down to 1)
        "requests_pirate": {"tone": 1.0, "style": "pirate"},
    }

    if feedback_type not in adjustments:
        return

    updates = {}
    for key, delta in adjustments[feedback_type].items():
        if key == "style":
            updates["style"] = delta
            continue
        # The update_communication_profile function handles clamping to 1-5
        # We need to get the current value and add delta
        from memory.living_profile import load_profile
        profile = load_profile(user_id)
        if profile:
            current = profile["communication_profile"].get(key, 3)
            new_value = current + delta
            updates[key] = new_value

    if updates:
        update_communication_profile(user_id, updates)
