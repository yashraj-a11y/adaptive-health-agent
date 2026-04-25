"""
Graph State Module

Defines the HealthAgentState TypedDict used as the shared state
object passed between all LangGraph nodes (profiler → analyst → communicator).
"""

from typing import TypedDict, Optional


class HealthAgentState(TypedDict):
    """Shared state for the health agent graph.

    Keys:
        living_profile: The user's current Living Profile dict.
        current_packet: The latest telemetry packet being processed.
        deviation_detected: Whether the profiler detected any baseline deviation.
        pattern_confirmed: Whether the pattern buffer confirmed a sustained pattern.
        pattern_details: Details about the confirmed pattern (metrics, deviations, etc.).
        severity_level: Severity classification (1-5) from the analyst.
        analyst_output: Full analyst assessment dict.
        proceed_to_communicate: Whether timing/context allows communication to user.
        final_message: The composed message for the user from the communicator.
        notify_family: Whether to trigger emergency family notification.
        user_message: User-initiated message (bypasses profiler when set).
        agent_response: Agent's response to a user-initiated message.
    """
    living_profile: dict
    current_packet: dict
    deviation_detected: bool
    pattern_confirmed: bool
    pattern_details: Optional[dict]
    severity_level: Optional[int]
    analyst_output: Optional[dict]
    proceed_to_communicate: bool
    final_message: Optional[str]
    notify_family: bool
    user_message: Optional[str]
    agent_response: Optional[str]
