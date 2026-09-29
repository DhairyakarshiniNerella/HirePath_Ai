from datetime import date

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
# {today} is filled in at call time so "Present"/"Current" entries resolve
# against the real date instead of the model's training cutoff.
SYSTEM_PROMPT_TEMPLATE = """You are a resume analysis expert.
Read the resume text and extract structured information about the candidate.
Today's date is {today}.

Rules:
- If information is not present in the resume, leave it empty or use the field's default. Never invent details.
- total_experience_years must be computed from every dated role in the resume - internships,
  full-time roles, and any other paid positions all count. Do not stop at the first role you find.
  Steps:
    1. Find every entry with a start date and an end date (or "Present"/"Current"/"Till date",
       which means today's date, {today}).
    2. Convert each entry's span to years (round to the nearest 0.5 e.g. 3 months ~= 0.25 years).
    3. If two entries' dates overlap (e.g. a promotion from intern to full-time at the same
       company with continuous dates), count the overlapping span once, not twice.
    4. Sum the spans to get total_experience_years. A role still marked "Present" must be counted
       all the way up to {today}, not just from its start date to itself.
- career_level should be one of: Fresher, Entry Level, Mid Level, Senior, Unknown - based on the
  computed total_experience_years (0 = Fresher, <2 = Entry Level, 2-5 = Mid Level, >5 = Senior).
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
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())
    messages = [
        ("system", system_prompt),
        ("human", f"Resume text:\n\n{resume_text}"),
    ]
    result = structured_llm.invoke(messages)
    log_usage("Resume Analyzer Agent", result["raw"])

    if result["parsed"] is None:
        raise ValueError(f"Could not parse resume into a profile: {result.get('parsing_error')}")

    return result["parsed"]
