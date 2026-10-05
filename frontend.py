"""Streamlit interface for the multi-agent travel planner."""

import re
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app


def _clean_md(text) -> str:
    """Make agent-generated Markdown safe for Streamlit."""
    text = str(text or "")
    text = re.sub(r"\s*<br\s*/?>\s*", " • ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\\)\$", r"\\$", text)
    return text


def _md(text):
    """Render cleaned Markdown output."""
    st.markdown(_clean_md(text))


def _new_thread_id(user_id: str) -> str:
    """Create a short unique thread identifier."""
    return f"{user_id}_{uuid.uuid4().hex[:8]}"


st.set_page_config(
    page_title="Real-World Multi-Agent Travel Planner",
    layout="wide",
)

st.title("Real-World Multi-Agent Travel Planner")
st.caption(
    "Live free data: weather forecast, exchange rates, public holidays, real hotels "
    "and attractions (OpenStreetMap), airports. Airfares and hotel prices are clearly "
    "labelled estimates with links to live prices."
)

with st.sidebar:
    st.subheader("Session")

    user_id = st.text_input("User ID", value="demo_user")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = _new_thread_id(user_id)

    if st.button("New Thread"):
        st.session_state.thread_id = _new_thread_id(user_id)
        st.session_state.pop("waiting_for_approval", None)
        st.session_state.pop("latest_result", None)
        st.session_state.pop("approval_round", None)

    st.caption(f"Thread: {st.session_state.thread_id}")


query = st.text_area(
    "Travel request",
    placeholder=(
        "Plan a 7-day trip from Varanasi to the Netherlands for 2 people, "
        "October 15 to October 21, 2026, budget Rs. 3 lakh, mid-range hotels."
    ),
    height=110,
)


if st.button("Create Draft Plan", type="primary"):
    if not query.strip():
        st.warning("Enter a travel request first.")
    else:
        st.session_state.thread_id = _new_thread_id(user_id)
        st.session_state.approval_round = 0

        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        try:
            with st.spinner("Agents are researching (live data + planning)..."):
                result = app.invoke(
                    {
                        "messages": [HumanMessage(content=query)],
                        "user_id": user_id,
                        "user_query": query,
                        "trip_constraints": {},
                        "selected_agents": [],
                        "supervisor_reasoning": "",
                        "guardrail_blocked": False,
                        "destination_info": {},
                        "budget_numbers": {},
                        "data_sources": [],
                        "flight_results": "",
                        "hotel_results": "",
                        "weather_results": "",
                        "budget_results": "",
                        "itinerary": "",
                        "approval_request": "",
                        "human_feedback": "",
                        "approved": False,
                        "revision_count": 0,
                        "final_response": "",
                        "llm_calls": 0,
                    },
                    config=config,
                )

            st.session_state.latest_result = result
            st.session_state.waiting_for_approval = "__interrupt__" in result

        except Exception as exc:
            st.session_state.latest_result = None
            st.session_state.waiting_for_approval = False
            st.error(f"Planning failed: {exc}")


result = st.session_state.get("latest_result")

if result and result.get("guardrail_blocked"):
    st.warning(
        result.get("guardrail_reason")
        or result.get("final_response")
        or "This request was blocked by the input guardrail."
    )

elif result:
    st.subheader("Supervisor Plan")
    _md(result.get("supervisor_reasoning", ""))

    constraints = result.get("trip_constraints", {}) or {}

    if constraints.get("dates_assumed"):
        st.info(
            f"Dates used: {constraints.get('departure_date')} to "
            f"{constraints.get('return_date')} (assumed by the system). "
            "Include exact dates in your request to change this."
        )

    info = result.get("destination_info", {}) or {}
    dest = info.get("destination") or {}

    if dest and not dest.get("exact", True):
        st.warning(
            f'I interpreted "{dest.get("query", "")}" as **{dest.get("label", "")}**. '
            "If that is not the place you meant, write it with the country, "
            'for example "Goa, India".'
        )

    if dest:
        country = dest.get("country") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Destination", dest.get("label", ""))
        c2.metric(
            "Currency",
            f'{country.get("currency_code", "")} {country.get("currency_symbol", "")}'.strip(),
        )
        c3.metric(
            "Languages",
            ", ".join(country.get("languages", [])[:2]) or "n/a",
        )

    if info.get("holidays"):
        _md(
            "**Public holidays during your trip:** "
            + "; ".join(f"{h['date']} {h['name']}" for h in info["holidays"])
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Flights")
        _md(result.get("flight_results", "") or "_Not requested._")

        st.subheader("Weather")
        _md(result.get("weather_results", "") or "_Not available._")

    with col2:
        st.subheader("Hotels")
        _md(result.get("hotel_results", "") or "_Not available._")

        st.subheader("Budget")
        _md(result.get("budget_results", "") or "_Not available._")

    st.subheader("Draft Itinerary")

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        draft = payload.get("draft_itinerary", "")
        rev = payload.get("revision_count", 0)
        if rev:
            st.caption(f"Revision {rev}")
    else:
        draft = result.get("itinerary", "")

    _md(draft)

    sources = result.get("data_sources", []) or []

    if sources:
        with st.expander("Data sources used (free APIs) and freshness"):
            for source in dict.fromkeys(sources):
                _md(f"- {source}")


if st.session_state.get("waiting_for_approval"):
    st.divider()
    st.subheader("Human Approval")

    rnd = st.session_state.get("approval_round", 0)

    approved = st.radio(
        "Approve this draft?",
        ["Yes", "No, revise it"],
        horizontal=True,
        key=f"approve_{rnd}",
    )

    feedback = st.text_area(
        "What should change?",
        disabled=approved == "Yes",
        key=f"feedback_{rnd}",
    )

    if st.button("Submit", key=f"submit_{rnd}"):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        try:
            with st.spinner(
                "Creating final plan..."
                if approved == "Yes"
                else "Revising the itinerary..."
            ):
                final_result = app.invoke(
                    Command(
                        resume={
                            "approved": approved == "Yes",
                            "feedback": feedback if approved != "Yes" else "",
                        }
                    ),
                    config=config,
                )

            st.session_state.latest_result = final_result
            st.session_state.waiting_for_approval = "__interrupt__" in final_result
            st.session_state.approval_round = rnd + 1
            st.rerun()

        except Exception as exc:
            st.error(f"Could not continue: {exc}")


final_result = st.session_state.get("latest_result")

if (
    final_result
    and final_result.get("final_response")
    and not final_result.get("guardrail_blocked")
    and not st.session_state.get("waiting_for_approval")
):
    st.divider()
    st.header("Final Travel Plan")
    _md(final_result["final_response"])

    st.download_button(
        "Download plan (Markdown)",
        data=final_result["final_response"],
        file_name="travel_plan.md",
        mime="text/markdown",
    )
