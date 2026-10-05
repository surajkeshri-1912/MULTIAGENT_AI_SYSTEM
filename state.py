import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage


class TravelState(TypedDict, total=False):

    # ---------------- Conversation ----------------
    messages: Annotated[list[AnyMessage], operator.add]
    user_id: str
    user_query: str

    # ---------------- Supervisor ----------------
    trip_constraints: dict[str, Any]
    selected_agents: list[str]
    supervisor_reasoning: str

    # ---------------- Guardrails ----------------
    guardrail_allowed: bool
    guardrail_category: str
    guardrail_risk_level: str
    guardrail_reason: str
    guardrail_blocked: bool
    output_guardrail_passed: bool
    output_guardrail_reason: str

    # ---------------- Real-world data ----------------
    # resolved places, country info, holidays, attractions
    destination_info: dict[str, Any]
    # numeric budget breakdown (USD + converted currency)
    budget_numbers: dict[str, Any]
    # which free data sources were used (live / failed / estimate)
    data_sources: Annotated[list[str], operator.add]

    # ---------------- Agent results ----------------
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str

    # ---------------- Human approval loop ----------------
    approval_request: str
    human_feedback: str
    approved: bool
    revision_count: int

    # ---------------- Final response ----------------
    final_response: str

    # ---------------- Observability ----------------
    llm_calls: int