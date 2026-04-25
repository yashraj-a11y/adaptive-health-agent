"""
Baseline Calculator Module

Computes personalized health baselines from episodic memory data.
Baselines transition from LEARNING to ESTABLISHED after BASELINE_LEARNING_DAYS.

Computation methods:
  - Resting HR: median of sedentary morning readings
  - HRV: 7-day rolling average of morning readings
  - Sleep metrics: 14-day rolling averages
  - Stress/Recovery: 7-day rolling averages
  - Steps: 14-day rolling average
"""

import os
import statistics
from memory.episodic_memory import get_recent
from memory.living_profile import load_profile, update_baselines
from dotenv import load_dotenv

load_dotenv()

BASELINE_LEARNING_DAYS = int(os.getenv("BASELINE_LEARNING_DAYS", 14))


def compute_baselines(user_id: str, episodic_memory=None) -> dict:
    """Compute personalized baselines from episodic memory data.

    Pulls recent episodes, extracts metric values, and computes
    rolling averages/medians for each baseline metric.

    Args:
        user_id: The user identifier.
        episodic_memory: Unused parameter kept for interface compatibility.
                         Data is pulled directly from the episodic_memory module.

    Returns:
        dict: Computed baseline values matching the baselines structure
              in the Living Profile.
    """
    profile = load_profile(user_id)
    if profile is None:
        print(f"[Baseline Calculator] No profile found for user: {user_id}")
        return {}

    days_monitored = profile.get("days_monitored", 0)

    # Pull episodes from the relevant time windows
    episodes_14d = get_recent(user_id, days=14)
    episodes_7d = get_recent(user_id, days=7)

    if not episodes_14d:
        print(f"[Baseline Calculator] No episodes found for user: {user_id}")
        return {}

    # Extract metric values from episode documents
    # Episodes store metrics as searchable text; we parse from metadata and documents
    hr_sedentary_morning = []
    hrv_morning = []
    spo2_values = []
    skin_temp_values = []
    sleep_hours_14d = []
    sleep_efficiency_14d = []
    breathing_rate_values = []
    steps_14d = []
    stress_7d = []
    recovery_7d = []

    for episode in episodes_14d:
        doc = episode.get("document", "")
        metadata = episode.get("metadata", {})

        # Parse metrics from the searchable document text
        metrics = _parse_metrics_from_document(doc)

        if not metrics:
            continue

        # Context parsing for filtering
        is_morning = "morning" in doc.lower()
        is_sedentary = "sedentary" in doc.lower()

        # Resting HR: median of sedentary morning readings
        if is_sedentary and is_morning and "heart_rate" in metrics:
            hr_sedentary_morning.append(metrics["heart_rate"])

        # HRV: morning readings (used for 7-day average below)
        if is_morning and "hrv" in metrics:
            hrv_morning.append(metrics["hrv"])

        # SpO2 from all readings
        if "spo2" in metrics:
            spo2_values.append(metrics["spo2"])

        # Skin temperature from all readings
        if "skin_temperature" in metrics:
            skin_temp_values.append(metrics["skin_temperature"])

        # Sleep metrics (14-day window)
        if "total_hours" in metrics:
            sleep_hours_14d.append(metrics["total_hours"])
        if "sleep_efficiency" in metrics:
            sleep_efficiency_14d.append(metrics["sleep_efficiency"])

        # Breathing rate from all readings
        if "breathing_rate" in metrics:
            breathing_rate_values.append(metrics["breathing_rate"])

        # Steps (14-day window)
        if "steps_today" in metrics:
            steps_14d.append(metrics["steps_today"])

    # Stress and recovery use 7-day window only
    for episode in episodes_7d:
        doc = episode.get("document", "")
        metrics = _parse_metrics_from_document(doc)
        if not metrics:
            continue
        if "stress_score" in metrics:
            stress_7d.append(metrics["stress_score"])
        if "recovery_score" in metrics:
            recovery_7d.append(metrics["recovery_score"])

    # Compute baselines
    baselines = {}

    # Resting HR: median of sedentary morning readings
    if hr_sedentary_morning:
        baselines["resting_hr"] = round(statistics.median(hr_sedentary_morning), 1)

    # HRV: 7-day rolling average of morning readings
    # Use only the last 7 days worth of morning HRV values
    hrv_7d_morning = hrv_morning[-7:] if len(hrv_morning) > 7 else hrv_morning
    if hrv_7d_morning:
        baselines["typical_hrv"] = round(statistics.mean(hrv_7d_morning), 1)

    # SpO2: overall average
    if spo2_values:
        baselines["typical_spo2"] = round(statistics.mean(spo2_values), 1)

    # Skin temperature: overall average
    if skin_temp_values:
        baselines["typical_skin_temp"] = round(statistics.mean(skin_temp_values), 1)

    # Sleep hours: 14-day rolling average
    if sleep_hours_14d:
        baselines["typical_sleep_hours"] = round(statistics.mean(sleep_hours_14d), 1)

    # Sleep efficiency: 14-day rolling average
    if sleep_efficiency_14d:
        baselines["typical_sleep_efficiency"] = round(statistics.mean(sleep_efficiency_14d), 1)

    # Steps: 14-day rolling average
    if steps_14d:
        baselines["typical_daily_steps"] = round(statistics.mean(steps_14d), 1)

    # Breathing rate: overall average
    if breathing_rate_values:
        baselines["typical_breathing_rate"] = round(statistics.mean(breathing_rate_values), 1)

    # Stress: 7-day rolling average
    if stress_7d:
        baselines["typical_stress_score"] = round(statistics.mean(stress_7d), 1)

    # Recovery: 7-day rolling average
    if recovery_7d:
        baselines["typical_recovery_score"] = round(statistics.mean(recovery_7d), 1)

    # Determine baseline status
    if days_monitored >= BASELINE_LEARNING_DAYS:
        baselines["status"] = "ESTABLISHED"
    else:
        baselines["status"] = "LEARNING"

    # Update the Living Profile with computed baselines
    if baselines:
        update_baselines(user_id, baselines)

    return baselines


def _parse_metrics_from_document(doc: str) -> dict:
    """Extract numeric metric values from an episode document string.

    The episodic memory stores metrics in a searchable text format like:
    "Metrics: heart_rate: 68, hrv: 55, spo2: 98, ..."

    Args:
        doc: The episode document text.

    Returns:
        dict: Extracted metric name-value pairs.
    """
    metrics = {}

    # Look for the "Metrics:" section in the document
    if "Metrics:" not in doc:
        return metrics

    # Extract the metrics portion
    try:
        metrics_start = doc.index("Metrics:") + len("Metrics:")
        # Find the end of the metrics section (next period or end of string)
        metrics_end = doc.find(".", metrics_start)
        if metrics_end == -1:
            metrics_end = len(doc)

        metrics_str = doc[metrics_start:metrics_end].strip()

        # Parse key-value pairs
        for pair in metrics_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                key, value = pair.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    # Try to convert to float
                    metrics[key] = float(value)
                except ValueError:
                    # Keep as string if not numeric
                    metrics[key] = value
    except (ValueError, IndexError):
        pass

    return metrics
