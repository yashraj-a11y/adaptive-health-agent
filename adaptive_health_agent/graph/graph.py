"""
LangGraph Health Agent Graph

Defines the StateGraph connecting: profiler → analyst → communicator.
Conditional edges control flow based on state values:
  - Profiler always runs
  - Analyst runs only if deviation_detected or pattern_confirmed
  - Communicator runs only if proceed_to_communicate or user_message exists
"""

from langgraph.graph import StateGraph, END

from graph.state import HealthAgentState
from agents.profiler import profiler_node
from agents.analyst import analyst_node
from agents.communicator import communicator_node


def _should_analyze(state: HealthAgentState) -> str:
    """Determine whether to route to analyst or skip to end.

    Args:
        state: Current graph state.

    Returns:
        str: "analyst" if analysis is needed, "end" otherwise.
    """
    if state.get("pattern_confirmed", False):
        return "analyst"
    if state.get("deviation_detected", False):
        return "analyst"
    return "end"


def _should_communicate(state: HealthAgentState) -> str:
    """Determine whether to route to communicator or skip to end.

    Args:
        state: Current graph state.

    Returns:
        str: "communicator" if communication is needed, "end" otherwise.
    """
    if state.get("proceed_to_communicate", False):
        return "communicator"
    if state.get("user_message"):
        return "communicator"
    return "end"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph health agent graph.

    Graph flow:
        profiler → (conditional) → analyst → (conditional) → communicator → END
        profiler → (conditional) → END (if no deviation)
        analyst → (conditional) → END (if timing hold)

    Returns:
        CompiledGraph: The compiled LangGraph ready for invocation.
    """
    # Create the state graph
    graph = StateGraph(HealthAgentState)

    # Add nodes
    graph.add_node("profiler", profiler_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("communicator", communicator_node)

    # Set entry point
    graph.set_entry_point("profiler")

    # Add conditional edges
    # After profiler: go to analyst if deviation/pattern detected, else end
    graph.add_conditional_edges(
        "profiler",
        _should_analyze,
        {
            "analyst": "analyst",
            "end": END,
        }
    )

    # After analyst: go to communicator if proceed_to_communicate, else end
    graph.add_conditional_edges(
        "analyst",
        _should_communicate,
        {
            "communicator": "communicator",
            "end": END,
        }
    )

    # After communicator: always end
    graph.add_edge("communicator", END)

    # Compile and return
    return graph.compile()


def build_user_message_graph() -> StateGraph:
    """Build a simplified graph for user-initiated messages.

    This graph skips the profiler and analyst, going directly to
    the communicator for user message handling.

    Graph flow:
        communicator → END

    Returns:
        CompiledGraph: The compiled LangGraph for user messages.
    """
    graph = StateGraph(HealthAgentState)

    graph.add_node("communicator", communicator_node)

    graph.set_entry_point("communicator")
    graph.add_edge("communicator", END)

    return graph.compile()
