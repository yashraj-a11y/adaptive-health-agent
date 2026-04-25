"""
Memory Summarizer Module

Produces weekly summaries of health episodes and stores them in
episodic memory. Runs on a configurable interval (SUMMARIZATION_INTERVAL_DAYS).
"""

import os
import json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

from memory.episodic_memory import get_recent, log_episode
from memory.living_profile import load_profile

load_dotenv()

SUMMARIZATION_INTERVAL_DAYS = int(os.getenv("SUMMARIZATION_INTERVAL_DAYS", 7))

# Use the analyst client for summarization
_summarizer_client = Groq(api_key=os.getenv("GROQ_API_KEY_ANALYST"))


def generate_weekly_summary(user_id: str) -> dict:
    """Generate a weekly health summary for a user.

    Retrieves all episodes from the past SUMMARIZATION_INTERVAL_DAYS,
    uses Groq LLM to produce a narrative summary, and logs the
    summary as a new episode in episodic memory.

    Args:
        user_id: The user identifier.

    Returns:
        dict: The summary episode dict, or None if no episodes to summarize.
    """
    # Retrieve recent episodes
    episodes = get_recent(user_id, days=SUMMARIZATION_INTERVAL_DAYS)

    if not episodes:
        print(f"[Summarizer] No episodes found for user {user_id} in the last {SUMMARIZATION_INTERVAL_DAYS} days.")
        return None

    profile = load_profile(user_id)
    if profile is None:
        print(f"[Summarizer] No profile found for user: {user_id}")
        return None

    # Build episode digest for the LLM
    episode_digest = _build_episode_digest(episodes)
    profile_summary = _build_profile_context(profile)

    # Generate summary via Groq LLM
    summary_text = _call_summarizer_llm(episode_digest, profile_summary)

    # Build and log the summary episode
    summary_episode = {
        "id": f"episode_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{user_id}",
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "event_type": "weekly_summary",
        "metrics_snapshot": {},
        "context_snapshot": {
            "summary_period_days": SUMMARIZATION_INTERVAL_DAYS,
            "episodes_summarized": len(episodes),
        },
        "deviation_from_baseline": {},
        "significance": "single_occurrence",
        "agent_action_taken": f"Generated weekly summary covering {len(episodes)} episodes",
        "user_response": None,
        "outcome": summary_text,
        "tags": ["weekly_summary", "automated"],
    }

    log_episode(summary_episode)

    print(f"[Summarizer] Weekly summary generated for {user_id}: {len(episodes)} episodes summarized.")

    return summary_episode


def _build_episode_digest(episodes: list) -> str:
    """Build a concise digest of episodes for the LLM.

    Args:
        episodes: List of episode dicts from get_recent.

    Returns:
        str: Formatted episode digest.
    """
    parts = []

    # Count event types
    event_counts = {}
    for ep in episodes:
        event_type = ep.get("metadata", {}).get("event_type", "unknown")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    parts.append(f"Total episodes: {len(episodes)}")
    parts.append(f"Event types: {json.dumps(event_counts)}")
    parts.append("")

    # Include up to 15 most recent episodes (to stay within token limits)
    for i, ep in enumerate(episodes[:15]):
        meta = ep.get("metadata", {})
        parts.append(
            f"Episode {i+1}: [{meta.get('event_type', '?')}] "
            f"{meta.get('significance', '?')} at {meta.get('timestamp', '?')} — "
            f"Action: {meta.get('agent_action_taken', 'N/A')}"
        )

    if len(episodes) > 15:
        parts.append(f"... and {len(episodes) - 15} more episodes")

    return "\n".join(parts)


def _build_profile_context(profile: dict) -> str:
    """Build profile context for the summary LLM call.

    Args:
        profile: The user's Living Profile dict.

    Returns:
        str: Profile context string.
    """
    identity = profile.get("identity", {})
    baselines = profile.get("baselines", {})
    current_state = profile.get("current_state", {})
    concerns = profile.get("current_concerns", [])
    patterns = profile.get("known_patterns", [])

    return (
        f"User: {identity.get('name', 'Unknown')}, age {identity.get('age', '?')}. "
        f"Baseline status: {baselines.get('status', 'LEARNING')}. "
        f"Sleep trend: {current_state.get('sleep_trend_7d', 'unknown')}. "
        f"Stress trend: {current_state.get('stress_trend_7d', 'unknown')}. "
        f"HRV trend: {current_state.get('hrv_trend_7d', 'unknown')}. "
        f"Recovery trend: {current_state.get('recovery_trend_7d', 'unknown')}. "
        f"Active concerns: {', '.join(concerns) or 'None'}. "
        f"Known patterns: {', '.join(patterns) or 'None'}."
    )


def _call_summarizer_llm(episode_digest: str, profile_context: str) -> str:
    """Call Groq LLM to generate a weekly health summary.

    Args:
        episode_digest: Formatted episode digest string.
        profile_context: Profile context string.

    Returns:
        str: The generated weekly summary text.
    """
    system_prompt = (
        "You are a health data summarizer. Given a week's worth of health monitoring episodes, "
        "produce a concise weekly health summary. Include: key patterns observed, trend directions, "
        "notable events, and overall health trajectory. Be factual, do not diagnose. "
        "Write in plain language, 3-5 sentences."
    )

    user_prompt = (
        f"USER CONTEXT:\n{profile_context}\n\n"
        f"EPISODE DIGEST:\n{episode_digest}\n\n"
        f"Write a concise weekly health summary."
    )

    try:
        response = _summarizer_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Summarizer] LLM call error: {e}")
        # Fallback summary
        return (
            f"Weekly summary: {len(episode_digest.splitlines())} health events recorded this period. "
            f"Review your health trends in the dashboard for details."
        )
