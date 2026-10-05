"""Shared state passed between the travel-planning agents."""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage


class TravelState(TypedDict, total=False):
    """State schema used by the LangGraph travel planner."""

    messages: Annotated[list[AnyMessage], operator.add]
    user_id: str
    user_query: str

    trip_constraints: dict[str, Any]
    selected_agents: list[str]
    supervisor_reasoning: str

    guardrail_allowed: bool
    guardrail_category: str
    guardrail_risk_level: str
    guardrail_reason: str
    guardrail_blocked: bool
    output_guardrail_passed: bool
    output_guardrail_reason: str

    destination_info: dict[str, Any]
    budget_numbers: dict[str, Any]
    data_sources: Annotated[list[str], operator.add]

    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str

    approval_request: str
    human_feedback: str
    approved: bool
    revision_count: int

    final_response: str
    llm_calls: int
