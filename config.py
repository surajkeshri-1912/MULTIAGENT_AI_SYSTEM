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

# Optional. Only used for gpt-oss models. Values: low / medium / high.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")


@lru_cache(maxsize=None)
def get_llm(max_tokens: int = 1200):
    """
    Central LLM factory.

    max_tokens is chosen PER CALL (see agents.py). gpt-oss is a reasoning
    model: reasoning tokens count against max_tokens, so a small cap
    (e.g. 800) truncates JSON / itineraries or even returns empty content.

    Models are cached per max_tokens value.
    """

    kwargs = dict(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=max_tokens,
        max_retries=2,
    )

    if "gpt-oss" in GROQ_MODEL and GROQ_REASONING_EFFORT:
        try:
            return ChatGroq(
                **kwargs,
                reasoning_effort=GROQ_REASONING_EFFORT,
            )
        except Exception as exc:
            # Older langchain-groq versions do not know this argument.
            print("reasoning_effort not supported, continuing without it:", repr(exc))

    return ChatGroq(**kwargs)