"""Application configuration and LLM factory.

Environment variables are loaded once here so the rest of the project can
import configuration without duplicating setup code.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")


@lru_cache(maxsize=None)
def get_llm(max_tokens: int = 1200):
    """Create and cache a Groq chat model for a specific token budget.

    gpt-oss reasoning tokens count toward max_tokens, so callers choose the
    limit according to the amount of output they need.
    """
    kwargs = {
        "model": GROQ_MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "max_retries": 2,
    }

    if "gpt-oss" in GROQ_MODEL and GROQ_REASONING_EFFORT:
        try:
            return ChatGroq(
                **kwargs,
                reasoning_effort=GROQ_REASONING_EFFORT,
            )
        except Exception as exc:
            print(
                "reasoning_effort not supported, continuing without it:",
                repr(exc),
            )

    return ChatGroq(**kwargs)
