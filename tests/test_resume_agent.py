import pytest
from app.agents import resume_agent
from app.models.profile import CandidateProfile


class FakeAIMessage:
    def __init__(self, usage_metadata=None):
        self.usage_metadata = usage_metadata or {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


class FakeStructuredLLM:
    """Stands in for the real ChatGroq structured_llm, which is a pydantic-based
    Runnable that rejects monkeypatching its 'invoke' attribute directly."""

    def __init__(self, invoke_fn):
        self.invoke = invoke_fn


@pytest.fixture
def spy_log_usage(monkeypatch):
    calls = []
    monkeypatch.setattr(resume_agent, "log_usage", lambda agent, raw: calls.append((agent, raw)))
    return calls


def test_analyze_resume_returns_parsed_profile(monkeypatch, spy_log_usage):
    fake_profile = CandidateProfile(name="Jane Doe", skills=["Python"])
    fake_raw = FakeAIMessage()
    monkeypatch.setattr(
        resume_agent, "structured_llm",
        FakeStructuredLLM(lambda messages: {"raw": fake_raw, "parsed": fake_profile, "parsing_error": None}),
    )

    result = resume_agent.analyze_resume("Jane Doe\nSkills: Python")

    assert result is fake_profile
    assert spy_log_usage == [("Resume Analyzer Agent", fake_raw)]


def test_analyze_resume_sends_resume_text_in_messages(monkeypatch, spy_log_usage):
    captured = {}

    def fake_invoke(messages):
        captured["messages"] = messages
        return {"raw": FakeAIMessage(), "parsed": CandidateProfile(), "parsing_error": None}

    monkeypatch.setattr(resume_agent, "structured_llm", FakeStructuredLLM(fake_invoke))

    resume_agent.analyze_resume("My resume content here")

    system_role, system_text = captured["messages"][0]
    human_role, human_text = captured["messages"][1]
    assert system_role == "system"
    assert human_role == "human"
    assert "My resume content here" in human_text


def test_analyze_resume_raises_when_parsing_fails(monkeypatch, spy_log_usage):
    monkeypatch.setattr(
        resume_agent, "structured_llm",
        FakeStructuredLLM(lambda messages: {"raw": FakeAIMessage(), "parsed": None, "parsing_error": "malformed output"}),
    )

    with pytest.raises(ValueError, match="malformed output"):
        resume_agent.analyze_resume("some resume text")

    # Token usage should still be logged even though parsing failed -
    # the LLM call itself succeeded and cost tokens.
    assert len(spy_log_usage) == 1


def test_analyze_resume_propagates_llm_exceptions(monkeypatch, spy_log_usage):
    def boom(messages):
        raise RuntimeError("Groq API unavailable")

    monkeypatch.setattr(resume_agent, "structured_llm", FakeStructuredLLM(boom))

    with pytest.raises(RuntimeError, match="Groq API unavailable"):
        resume_agent.analyze_resume("some resume text")

    assert spy_log_usage == []
