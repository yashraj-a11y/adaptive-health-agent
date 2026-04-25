"""
Profiler Agent

Compares incoming telemetry packets against the user's personal baselines.
Detects deviations, tracks them via PatternBuffer, and logs anomalies
to ChromaDB episodic memory.

Uses Groq LLM to produce a structured summary of detected anomalies.
"""

import os
import json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

from graph.state import HealthAgentState
from memory.episodic_memory import log_episode
from memory.living_profile import update_current_state
from utils.pattern_buffer import PatternBuffer

load_dotenv()

# Groq client for the Profiler agent
profiler_client = Groq(api_key=os.getenv("GROQ_API_KEY_PROFILER"))

# Module-level PatternBuffer instance (persists across packets in a session)
_pattern_buffer = PatternBuffer()

# Thresholds for ESTABLISHED baselines
THRESHOLDS = {
    "hr_elevated": {"metric": "heart_rate", "direction": "above", "percent": 20, "context_required": "sedentary"},
    "hrv_low": {"metric": "hrv", "direction": "below", "percent": 25, "context_required": None},
    "spo2_low": {"metric": "spo2", "direction": "absolute_below", "value": 94, "context_required": None},
    "stress_elevated": {"metric": "stress_score", "direction": "absolute_above", "value": 75, "context_required": "sedentary"},
    "recovery_low": {"metric": "recovery_score", "direction": "absolute_below", "value": 30, "context_required": None},
    "temp_elevated": {"metric": "skin_temperature", "direction": "absolute_above_delta", "delta": 0.8, "context_required": None},
    "breathing_elevated": {"metric": "breathing_rate", "direction": "above", "percent": 25, "context_required": "sedentary"},
}

# Thresholds for LEARNING baselines (only flag extreme deviations >=40%)
LEARNING_THRESHOLDS = {
    "hr_elevated": {"metric": "heart_rate", "direction": "absolute_above", "value": 100, "context_required": "sedentary"},
    "hrv_low": {"metric": "hrv", "direction": "absolute_below", "value": 30, "context_required": None},
    "spo2_low": {"metric": "spo2", "direction": "absolute_below", "value": 94, "context_required": None},
    "stress_elevated": {"metric": "stress_score", "direction": "absolute_above", "value": 80, "context_required": "sedentary"},
    "recovery_low": {"metric": "recovery_score", "direction": "absolute_below", "value": 25, "context_required": None},
    "temp_elevated": {"metric": "skin_temperature", "direction": "absolute_above", "value": 37.5, "context_required": None},
    "breathing_elevated": {"metric": "breathing_rate", "direction": "absolute_above", "value": 22, "context_required": "sedentary"},
}


def _check_deviation(metric_name: str, current_value, baseline_value, threshold_config: dict, activity_state: str) -> dict:
    """Check if a single metric deviates from its baseline/threshold.

    Args:
        metric_name: The pattern buffer metric key (e.g., "hr_elevated").
        current_value: The current reading from the telemetry packet.
        baseline_value: The established baseline value (may be None during LEARNING).
        threshold_config: The threshold configuration dict for this metric.
        activity_state: Current activity state from the packet.

    Returns:
        dict with keys: deviated (bool), deviation_percent (float or None),
              current (float), baseline (float or None), metric (str).
    """
    result = {
        "deviated": False,
        "deviation_percent": None,
        "current": current_value,
        "baseline": baseline_value,
        "metric": metric_name,
    }

    if current_value is None:
        return result

    # Check if context requirement is met (e.g., must be sedentary)
    context_req = threshold_config.get("context_required")
    if context_req and activity_state != context_req:
        return result

    direction = threshold_config["direction"]

    if direction == "above" and baseline_value is not None and baseline_value > 0:
        # Percentage above baseline
        percent = threshold_config["percent"]
        deviation = ((current_value - baseline_value) / baseline_value) * 100
        if deviation > percent:
            result["deviated"] = True
            result["deviation_percent"] = round(deviation, 1)

    elif direction == "below" and baseline_value is not None and baseline_value > 0:
        # Percentage below baseline
        percent = threshold_config["percent"]
        deviation = ((baseline_value - current_value) / baseline_value) * 100
        if deviation > percent:
            result["deviated"] = True
            result["deviation_percent"] = round(-deviation, 1)

    elif direction == "absolute_below":
        # Absolute value check (below)
        if current_value < threshold_config["value"]:
            result["deviated"] = True
            if baseline_value and baseline_value > 0:
                result["deviation_percent"] = round(
                    ((current_value - baseline_value) / baseline_value) * 100, 1
                )

    elif direction == "absolute_above":
        # Absolute value check (above)
        if current_value > threshold_config["value"]:
            result["deviated"] = True
            if baseline_value and baseline_value > 0:
                result["deviation_percent"] = round(
                    ((current_value - baseline_value) / baseline_value) * 100, 1
                )

    elif direction == "absolute_above_delta":
        # Delta above baseline
        delta = threshold_config["delta"]
        if baseline_value is not None and (current_value - baseline_value) > delta:
            result["deviated"] = True
            result["deviation_percent"] = round(current_value - baseline_value, 2)

    return result


def profiler_node(state: HealthAgentState) -> dict:
    """Profiler node for the LangGraph health agent.

    Compares current telemetry packet against user baselines, tracks
    deviations via PatternBuffer, and calls Groq for structured summary.

    Args:
        state: The current HealthAgentState.

    Returns:
        dict: State updates with deviation_detected, pattern_confirmed,
              pattern_details, and updated living_profile.
    """
    packet = state["current_packet"]
    profile = state["living_profile"]
    baselines = profile.get("baselines", {})
    baseline_status = baselines.get("status", "LEARNING")

    # Extract current values from packet
    vitals = packet.get("vitals", {})
    movement = packet.get("movement", {})
    sleep = packet.get("sleep_last_night", {})
    context = packet.get("context", {})
    activity_state = movement.get("activity_state", "unknown")
    user_id = packet.get("user_id", "unknown")
    timestamp = packet.get("timestamp", datetime.now().isoformat())

    # Select threshold set based on baseline status
    if baseline_status == "ESTABLISHED":
        threshold_set = THRESHOLDS
    else:
        threshold_set = LEARNING_THRESHOLDS

    # Map metric keys to their current and baseline values
    metric_values = {
        "hr_elevated": (vitals.get("heart_rate"), baselines.get("resting_hr")),
        "hrv_low": (vitals.get("hrv"), baselines.get("typical_hrv")),
        "spo2_low": (vitals.get("spo2"), baselines.get("typical_spo2")),
        "stress_elevated": (vitals.get("stress_score"), baselines.get("typical_stress_score")),
        "recovery_low": (vitals.get("recovery_score"), baselines.get("typical_recovery_score")),
        "temp_elevated": (vitals.get("skin_temperature"), baselines.get("typical_skin_temp")),
        "breathing_elevated": (vitals.get("breathing_rate"), baselines.get("typical_breathing_rate")),
    }

    # Check each metric for deviations
    deviations = {}
    any_deviation = False

    for metric_key, (current_val, baseline_val) in metric_values.items():
        if metric_key not in threshold_set:
            continue
        result = _check_deviation(metric_key, current_val, baseline_val,
                                  threshold_set[metric_key], activity_state)
        deviations[metric_key] = result

        if result["deviated"]:
            any_deviation = True
            _pattern_buffer.increment(metric_key)
        else:
            _pattern_buffer.reset(metric_key)

    # Check for confirmed patterns
    confirmed_patterns = _pattern_buffer.get_confirmed_patterns()
    pattern_confirmed = len(confirmed_patterns) > 0

    # Build pattern details
    pattern_details = None
    if pattern_confirmed:
        pattern_details = {
            "confirmed_metrics": confirmed_patterns,
            "deviations": {k: v for k, v in deviations.items() if v["deviated"]},
            "packet_timestamp": timestamp,
            "user_id": user_id,
            "activity_state": activity_state,
            "context": context,
            "vitals_snapshot": vitals,
            "sleep_snapshot": sleep,
        }

        # Log confirmed pattern to ChromaDB
        episode = {
            "id": f"episode_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{user_id}",
            "timestamp": timestamp,
            "user_id": user_id,
            "event_type": "anomaly",
            "metrics_snapshot": vitals,
            "context_snapshot": {
                "time_of_day": context.get("time_of_day", "unknown"),
                "activity_state": activity_state,
                "location_zone": context.get("location_zone", "unknown"),
            },
            "deviation_from_baseline": {
                k: f"{v['deviation_percent']}%"
                for k, v in deviations.items() if v["deviated"] and v["deviation_percent"] is not None
            },
            "significance": "pattern_confirmed",
            "agent_action_taken": f"Pattern confirmed for: {', '.join(confirmed_patterns)}",
            "user_response": None,
            "outcome": None,
            "tags": confirmed_patterns,
        }
        log_episode(episode)

        # Reset confirmed pattern counters after logging
        for metric in confirmed_patterns:
            _pattern_buffer.reset_confirmed(metric)

    elif any_deviation:
        # Log single occurrence to ChromaDB
        deviated_metrics = [k for k, v in deviations.items() if v["deviated"]]
        episode = {
            "id": f"episode_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{user_id}",
            "timestamp": timestamp,
            "user_id": user_id,
            "event_type": "anomaly",
            "metrics_snapshot": vitals,
            "context_snapshot": {
                "time_of_day": context.get("time_of_day", "unknown"),
                "activity_state": activity_state,
                "location_zone": context.get("location_zone", "unknown"),
            },
            "deviation_from_baseline": {
                k: f"{v['deviation_percent']}%"
                for k, v in deviations.items() if v["deviated"] and v["deviation_percent"] is not None
            },
            "significance": "single_occurrence",
            "agent_action_taken": f"Logged deviation for monitoring: {', '.join(deviated_metrics)}",
            "user_response": None,
            "outcome": None,
            "tags": deviated_metrics,
        }
        log_episode(episode)

    # Update living_profile current_state trends
    trend_updates = _compute_trend_updates(vitals, sleep, baselines)
    if trend_updates:
        update_current_state(user_id, trend_updates)
        # Also update the in-memory profile
        for k, v in trend_updates.items():
            profile["current_state"][k] = v

    # Build deviation summary for Groq call
    deviation_summary = []
    for metric_key, result in deviations.items():
        if result["deviated"]:
            deviation_summary.append(
                f"{metric_key}: current={result['current']}, "
                f"baseline={result['baseline']}, "
                f"deviation={result['deviation_percent']}%"
            )

    # Groq LLM call for structured profiler summary
    profiler_summary = _call_profiler_llm(packet, deviation_summary, baseline_status)

    # Merge profiler LLM output into pattern_details if available
    if pattern_details and profiler_summary:
        pattern_details["profiler_assessment"] = profiler_summary

    return {
        "deviation_detected": any_deviation,
        "pattern_confirmed": pattern_confirmed,
        "pattern_details": pattern_details,
        "living_profile": profile,
    }


def _compute_trend_updates(vitals: dict, sleep: dict, baselines: dict) -> dict:
    """Compute simple trend direction updates for the living profile.

    Compares current values against baselines and returns trend strings.

    Args:
        vitals: Current vitals from the telemetry packet.
        sleep: Current sleep data from the telemetry packet.
        baselines: The user's established baselines.

    Returns:
        dict: Trend update key-value pairs for current_state.
    """
    updates = {}

    # Sleep trend
    typical_sleep = baselines.get("typical_sleep_hours")
    current_sleep = sleep.get("total_hours")
    if typical_sleep and current_sleep:
        if current_sleep < typical_sleep * 0.85:
            updates["sleep_trend_7d"] = "declining"
        elif current_sleep > typical_sleep * 1.1:
            updates["sleep_trend_7d"] = "improving"
        else:
            updates["sleep_trend_7d"] = "stable"

    # HRV trend
    typical_hrv = baselines.get("typical_hrv")
    current_hrv = vitals.get("hrv")
    if typical_hrv and current_hrv:
        if current_hrv < typical_hrv * 0.8:
            updates["hrv_trend_7d"] = "declining"
        elif current_hrv > typical_hrv * 1.1:
            updates["hrv_trend_7d"] = "improving"
        else:
            updates["hrv_trend_7d"] = "stable"

    # Stress trend
    typical_stress = baselines.get("typical_stress_score")
    current_stress = vitals.get("stress_score")
    if typical_stress and current_stress:
        if current_stress > typical_stress * 1.3:
            updates["stress_trend_7d"] = "rising"
        elif current_stress < typical_stress * 0.85:
            updates["stress_trend_7d"] = "declining"
        else:
            updates["stress_trend_7d"] = "stable"

    # Recovery trend
    typical_recovery = baselines.get("typical_recovery_score")
    current_recovery = vitals.get("recovery_score")
    if typical_recovery and current_recovery:
        if current_recovery < typical_recovery * 0.75:
            updates["recovery_trend_7d"] = "declining"
        elif current_recovery > typical_recovery * 1.1:
            updates["recovery_trend_7d"] = "improving"
        else:
            updates["recovery_trend_7d"] = "stable"

    return updates


def _call_profiler_llm(packet: dict, deviation_summary: list, baseline_status: str) -> dict:
    """Call Groq LLM to produce a structured profiler assessment.

    Args:
        packet: The full telemetry packet.
        deviation_summary: List of deviation description strings.
        baseline_status: "LEARNING" or "ESTABLISHED".

    Returns:
        dict: Parsed JSON with keys: anomalies, deviation_summary, significance.
              Returns empty dict on error.
    """
    deviation_text = "\n".join(deviation_summary) if deviation_summary else "No deviations detected."

    system_prompt = (
        "You are the Profiler agent. Analyze the telemetry against the user's personal baselines. "
        "Return JSON with keys: anomalies (list), deviation_summary (str), "
        "significance (str: normal|elevated|critical). Do not diagnose."
    )

    user_prompt = (
        f"Baseline status: {baseline_status}\n\n"
        f"Telemetry packet:\n{json.dumps(packet, indent=2)}\n\n"
        f"Detected deviations:\n{deviation_text}\n\n"
        f"Return your analysis as valid JSON only. No markdown, no explanation."
    )

    try:
        response = profiler_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON from response (handle potential markdown wrapping)
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        return json.loads(content)

    except Exception as e:
        print(f"[Profiler] LLM call error: {e}")
        # Return a fallback assessment
        significance = "normal"
        if deviation_summary:
            significance = "elevated"
            if len(deviation_summary) >= 3:
                significance = "critical"
        return {
            "anomalies": [d.split(":")[0] for d in deviation_summary] if deviation_summary else [],
            "deviation_summary": deviation_text,
            "significance": significance,
        }
