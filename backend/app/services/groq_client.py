import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

# GROQ_API_KEY_2 is optional. If it's not set, only the primary key is used
# and behavior is unchanged from before this file existed.
_PRIMARY_KEY = os.getenv("GROQ_API_KEY")
_FALLBACK_KEY = os.getenv("GROQ_API_KEY_2")


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return "rate_limit_exceeded" in text or bool(re.search(r"\b429\b", text))


class StructuredLLMWithFallback:
    """
    Wraps one or more Groq-backed structured-output clients bound to the
    same Pydantic schema, and tries them in order. Only falls through to
    the next client on a rate-limit error - each key has its own separate
    free-tier daily quota, so a second key can pick up where the first
    one's quota ran out. Any other kind of failure (bad prompt, real
    outage) raises immediately from the first client, since a different
    key wouldn't fix it.
    """

    def __init__(self, clients: list):
        if not clients:
            raise ValueError("At least one Groq client is required")
        self._clients = clients

    def invoke(self, messages):
        last_error = None
        for index, client in enumerate(self._clients):
            try:
                return client.invoke(messages)
            except Exception as e:
                is_last_client = index == len(self._clients) - 1
                if _is_rate_limit_error(e) and not is_last_client:
                    last_error = e
                    continue
                raise
        raise last_error


def build_structured_llm(pydantic_model, temperature: float = 0) -> StructuredLLMWithFallback:
    """
    Builds a structured-output client for the given Pydantic schema, backed
    by GROQ_API_KEY, with GROQ_API_KEY_2 (if set) as an automatic fallback
    when the primary key's daily/per-minute quota is exhausted.
    """
    api_keys = [key for key in (_PRIMARY_KEY, _FALLBACK_KEY) if key]
    clients = [
        ChatGroq(model=MODEL_NAME, temperature=temperature, groq_api_key=key).with_structured_output(
            pydantic_model, include_raw=True
        )
        for key in api_keys
    ]
    return StructuredLLMWithFallback(clients)
