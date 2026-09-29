import pytest
from app.services.groq_client import StructuredLLMWithFallback


class FakeClient:
    """Stands in for one ChatGroq(...).with_structured_output(...) runnable."""

    def __init__(self, behavior):
        self._behavior = behavior
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return self._behavior(messages)


def make_client(behavior):
    """behavior: a function(messages) -> result, or one that raises."""
    return FakeClient(behavior)


def test_uses_first_client_when_it_succeeds():
    primary = make_client(lambda messages: {"parsed": "ok"})
    fallback = make_client(lambda messages: {"parsed": "should not be used"})

    llm = StructuredLLMWithFallback([primary, fallback])
    result = llm.invoke(["some message"])

    assert result == {"parsed": "ok"}
    assert primary.call_count == 1
    assert fallback.call_count == 0


def test_falls_back_to_second_client_on_rate_limit_error():
    def primary_behavior(messages):
        raise Exception("Error code: 429 - rate_limit_exceeded: tokens per day limit reached")

    primary = make_client(primary_behavior)
    fallback = make_client(lambda messages: {"parsed": "from fallback key"})

    llm = StructuredLLMWithFallback([primary, fallback])
    result = llm.invoke(["some message"])

    assert result == {"parsed": "from fallback key"}
    assert primary.call_count == 1
    assert fallback.call_count == 1


def test_does_not_fall_back_on_non_rate_limit_error():
    def primary_behavior(messages):
        raise ValueError("Could not parse response into schema")

    primary = make_client(primary_behavior)
    fallback = make_client(lambda messages: {"parsed": "should not be reached"})

    llm = StructuredLLMWithFallback([primary, fallback])

    with pytest.raises(ValueError, match="Could not parse response"):
        llm.invoke(["some message"])

    assert fallback.call_count == 0


def test_raises_original_error_when_all_clients_rate_limited():
    def always_rate_limited(messages):
        raise Exception("rate_limit_exceeded")

    primary = make_client(always_rate_limited)
    fallback = make_client(always_rate_limited)

    llm = StructuredLLMWithFallback([primary, fallback])

    with pytest.raises(Exception, match="rate_limit_exceeded"):
        llm.invoke(["some message"])

    assert primary.call_count == 1
    assert fallback.call_count == 1


def test_single_client_behaves_like_before_no_fallback_configured():
    only_client = make_client(lambda messages: {"parsed": "result"})
    llm = StructuredLLMWithFallback([only_client])
    assert llm.invoke(["msg"]) == {"parsed": "result"}


def test_requires_at_least_one_client():
    with pytest.raises(ValueError):
        StructuredLLMWithFallback([])
