"""
Living Profile Module

Manages persistent user health profiles stored as JSON files.
Each user has a Living Profile at ./profiles/{user_id}.json containing
identity, baselines, current state, patterns, communication preferences,
and ongoing concerns.
"""

import os
import json
from datetime import datetime


PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles")


def _ensure_profiles_dir():
    """Create the profiles directory if it does not exist."""
    os.makedirs(PROFILES_DIR, exist_ok=True)


def _profile_path(user_id: str) -> str:
    """Return the file path for a user's profile JSON."""
    return os.path.join(PROFILES_DIR, f"{user_id}.json")


def create_profile(user_data: dict) -> dict:
    """Create a new Living Profile from onboarding data.

    Args:
        user_data: Dict with keys from onboarding (name, age, conditions,
                   medications, goals, communication_style, emergency_contact).

    Returns:
        dict: The complete Living Profile structure.
    """
    profile = {
        "identity": {
            "name": user_data.get("name", ""),
            "age": user_data.get("age", 0),
            "known_conditions": user_data.get("known_conditions", []),
            "medications": user_data.get("medications", []),
            "goals": user_data.get("goals", []),
            "emergency_contact": user_data.get("emergency_contact", None)
        },
        "baselines": {
            "status": "LEARNING",
            "resting_hr": None,
            "typical_hrv": None,
            "typical_spo2": None,
            "typical_skin_temp": None,
            "typical_sleep_hours": None,
            "typical_sleep_efficiency": None,
            "typical_daily_steps": None,
            "typical_breathing_rate": None,
            "typical_stress_score": None,
            "typical_recovery_score": None
        },
        "current_state": {
            "sleep_trend_7d": "unknown",
            "hrv_trend_7d": "unknown",
            "stress_trend_7d": "unknown",
            "recovery_trend_7d": "unknown",
            "overall_trajectory": "unknown"
        },
        "known_patterns": [],
        "communication_profile": {
            "style": user_data.get("communication_style", "balanced"),
            "directness": user_data.get("directness", 3),
            "depth": user_data.get("depth", 3),
            "tone": user_data.get("tone", 3),
            "length": user_data.get("length", 3),
            "framing": user_data.get("framing", 3),
            "alert_sensitivity": user_data.get("alert_sensitivity", "normal"),
            "best_engagement_times": user_data.get("best_engagement_times", []),
            "engagement_patterns": user_data.get("engagement_patterns", "unknown")
        },
        "current_concerns": [],
        "last_updated": datetime.now().isoformat(),
        "days_monitored": 0
    }

    # Save the newly created profile
    user_id = user_data.get("user_id", user_data.get("name", "unknown").lower().replace(" ", "_"))
    save_profile(user_id, profile)

    return profile


def load_profile(user_id: str) -> dict:
    """Load a user's Living Profile from disk.

    Args:
        user_id: The user identifier.

    Returns:
        dict: The Living Profile, or None if not found.
    """
    path = _profile_path(user_id)
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def save_profile(user_id: str, profile: dict) -> None:
    """Save a user's Living Profile to disk.

    Args:
        user_id: The user identifier.
        profile: The complete Living Profile dict to save.
    """
    _ensure_profiles_dir()
    path = _profile_path(user_id)

    # Update the last_updated timestamp
    profile["last_updated"] = datetime.now().isoformat()

    with open(path, "w") as f:
        json.dump(profile, f, indent=2, default=str)


def update_baselines(user_id: str, new_baselines: dict) -> None:
    """Update baseline values in a user's Living Profile.

    Args:
        user_id: The user identifier.
        new_baselines: Dict of baseline key-value pairs to update.
    """
    profile = load_profile(user_id)
    if profile is None:
        print(f"[Living Profile] No profile found for user: {user_id}")
        return

    for key, value in new_baselines.items():
        if key in profile["baselines"]:
            profile["baselines"][key] = value

    save_profile(user_id, profile)


def update_current_state(user_id: str, state_updates: dict) -> None:
    """Update current state fields in a user's Living Profile.

    Args:
        user_id: The user identifier.
        state_updates: Dict of current_state key-value pairs to update.
    """
    profile = load_profile(user_id)
    if profile is None:
        print(f"[Living Profile] No profile found for user: {user_id}")
        return

    for key, value in state_updates.items():
        if key in profile["current_state"]:
            profile["current_state"][key] = value

    save_profile(user_id, profile)


def update_communication_profile(user_id: str, updates: dict) -> None:
    """Update communication profile fields.

    Args:
        user_id: The user identifier.
        updates: Dict of communication_profile key-value pairs to update.
    """
    profile = load_profile(user_id)
    if profile is None:
        print(f"[Living Profile] No profile found for user: {user_id}")
        return

    for key, value in updates.items():
        if key in profile["communication_profile"]:
            # Clamp numeric fields to 1-5 range
            if key in ("directness", "depth", "tone", "length", "framing"):
                value = max(1, min(5, value))
            profile["communication_profile"][key] = value

    save_profile(user_id, profile)


def add_concern(user_id: str, concern: str) -> None:
    """Add a new concern to the user's current concerns list.

    Args:
        user_id: The user identifier.
        concern: Description of the concern to add.
    """
    profile = load_profile(user_id)
    if profile is None:
        print(f"[Living Profile] No profile found for user: {user_id}")
        return

    if concern not in profile["current_concerns"]:
        profile["current_concerns"].append(concern)
        save_profile(user_id, profile)


def resolve_concern(user_id: str, concern: str) -> None:
    """Remove a resolved concern from the user's current concerns list.

    Args:
        user_id: The user identifier.
        concern: Description of the concern to resolve/remove.
    """
    profile = load_profile(user_id)
    if profile is None:
        print(f"[Living Profile] No profile found for user: {user_id}")
        return

    if concern in profile["current_concerns"]:
        profile["current_concerns"].remove(concern)
        save_profile(user_id, profile)


def add_pattern(user_id: str, pattern: str) -> None:
    """Add a newly identified pattern to the user's known patterns list.

    Args:
        user_id: The user identifier.
        pattern: Description of the identified pattern.
    """
    profile = load_profile(user_id)
    if profile is None:
        print(f"[Living Profile] No profile found for user: {user_id}")
        return

    if pattern not in profile["known_patterns"]:
        profile["known_patterns"].append(pattern)
        save_profile(user_id, profile)
