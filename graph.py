"""Build and compile the LangGraph travel-planning workflow."""

import psycopg

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from agents import (
    MAX_REVISIONS,
    budget_agent,
    destination_agent,
    final_response_agent,
    flight_agent,
    hotel_agent,
    human_approval_agent,
    itinerary_agent,
    supervisor_agent,
    weather_agent,
)
from config import DATABASE_URL
from state import TravelState


AGENT_ORDER = [
    "destination_agent",
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

ROUTE_MAP = {name: name for name in AGENT_ORDER}
ROUTE_MAP["__end__"] = END


def _selected_agents(state: TravelState) -> list[str]:
    """Return selected agents in the fixed workflow order."""
    selected = state.get("selected_agents", [])
    if not isinstance(selected, list):
        return []
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    """Choose the first agent after supervisor planning."""
    if state.get("guardrail_blocked", False):
        return "__end__"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    """Create a router that finds the next selected agent."""

    def route(state: TravelState) -> str:
        selected = _selected_agents(state)

        try:
            current_index = AGENT_ORDER.index(current_agent)
        except ValueError:
            return "itinerary_agent"

        for next_agent in AGENT_ORDER[current_index + 1:]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


def route_after_approval(state: TravelState) -> str:
    """Continue to the final response or request another itinerary revision."""
    if state.get("approved", False):
        return "final_response"

    if state.get("revision_count", 0) < MAX_REVISIONS:
        return "itinerary_agent"

    return "final_response"


def _build_checkpointer():
    """Create a PostgreSQL checkpointer when configured, otherwise use memory."""
    if not DATABASE_URL:
        print("DATABASE_URL not set. Using in-memory checkpointer.")
        return MemorySaver()

    try:
        from psycopg.rows import dict_row

        conn_kwargs = {
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": 0,
        }

        try:
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                conninfo=DATABASE_URL,
                max_size=10,
                kwargs=conn_kwargs,
                open=True,
            )
            checkpointer = PostgresSaver(pool)
        except ImportError:
            conn = psycopg.connect(DATABASE_URL, **conn_kwargs)
            checkpointer = PostgresSaver(conn)

        checkpointer.setup()
        return checkpointer

    except Exception as exc:
        print("Postgres checkpointer failed:", repr(exc))
        print("Falling back to in-memory checkpointer.")
        return MemorySaver()


def build_graph():
    """Build the travel workflow and attach its checkpointer."""
    graph = StateGraph(TravelState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("destination_agent", destination_agent)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("weather_agent", weather_agent)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("final_response", final_response_agent)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

    for name in (
        "destination_agent",
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
    ):
        graph.add_conditional_edges(
            name,
            route_after_agent(name),
            ROUTE_MAP,
        )

    graph.add_edge("itinerary_agent", "human_approval")

    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "final_response": "final_response",
            "itinerary_agent": "itinerary_agent",
        },
    )

    graph.add_edge("final_response", END)

    return graph.compile(checkpointer=_build_checkpointer())


app = build_graph()
