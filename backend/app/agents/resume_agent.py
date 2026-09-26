from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.models.profile import CandidateProfile
from app.services.token_tracker import log_usage

# Load GROQ_API_KEY from the .env file
load_dotenv()

# The LLM we use for resume analysis
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# This makes the LLM return data matching our CandidateProfile shape exactly.
# include_raw=True also gives us the raw AIMessage (with token usage) alongside
# the parsed object, so we can track how many tokens this call actually cost.
structured_llm = llm.with_structured_output(CandidateProfile, include_raw=True)

# Instructions given to the LLM every time we analyze a resume
SYSTEM_PROMPT = """You are a resume analysis expert.
Read the resume text and extract structured information about the candidate.

Rules:
- If information is not present in the resume, leave it empty or use the field's default. Never invent details.
- total_experience_years should be a number (e.g. 0 for freshers, 1.5, 3).
- career_level should be one of: Fresher, Entry Level, Mid Level, Senior, Unknown.
- target_roles should be 2-4 job titles the candidate is realistically suited for, based on their skills and experience.
- For projects, extract a short "name" (the project's title, e.g. "HirePath AI" or "Resume Parser Tool")
  separately from its "description" (1-2 sentences about what it does). If the resume only gives a
  description with no clear title, create a short, accurate name from the description itself - never
  leave the name empty or generic like "Project 1".
"""


def analyze_resume(resume_text: str) -> CandidateProfile:
    """
    Sends resume text to the Groq LLM and returns a structured CandidateProfile.
    """
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Resume text:\n\n{resume_text}"),
    ]
    result = structured_llm.invoke(messages)
    log_usage("Resume Analyzer Agent", result["raw"])

    if result["parsed"] is None:
        raise ValueError(f"Could not parse resume into a profile: {result.get('parsing_error')}")

    return result["parsed"]
