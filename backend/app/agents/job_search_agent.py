from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from app.models.profile import CandidateProfile
from app.tools.adzuna_tool import search_adzuna_jobs
from app.tools.jooble_tool import search_jooble_jobs
from app.tools.arbeitnow_tool import search_arbeitnow_jobs
from app.services.token_tracker import log_usage

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


class SearchQueries(BaseModel):
    queries: List[str] = Field(
        description="3 to 5 short, realistic job search terms (e.g. 'Python Developer', "
        "'Machine Learning Engineer') based on the candidate's skills and target roles"
    )


structured_llm = llm.with_structured_output(SearchQueries, include_raw=True)

QUERY_PROMPT = """You are a job search expert.
Based on the candidate's skills, technologies, and target roles, generate 3 to 5
short, realistic job search terms someone would type into a job board.

Rules:
- Keep each query short (2-4 words), like a real job title.
- Avoid duplicates.
- Base queries only on the candidate's actual skills/target roles. Do not invent unrelated roles.
"""


def generate_search_queries(candidate_profile: CandidateProfile) -> List[str]:
    """Uses the LLM to turn a candidate profile into a short list of job search terms."""
    profile_summary = f"""
Skills: {', '.join(candidate_profile.skills)}
Technologies: {', '.join(candidate_profile.technologies)}
Target Roles: {', '.join(candidate_profile.target_roles)}
Career Level: {candidate_profile.career_level}
"""
    messages = [
        ("system", QUERY_PROMPT),
        ("human", profile_summary),
    ]
    result = structured_llm.invoke(messages)
    log_usage("Job Search Agent (query generation)", result["raw"])

    if result["parsed"] is None:
        raise ValueError(f"Could not parse search queries: {result.get('parsing_error')}")

    return result["parsed"].queries


def search_all_sources(query: str, location: str = "") -> List[dict]:
    """
    Calls all three job APIs for one search query and combines the results.
    Each job dict is tagged with which source it came from.
    """
    all_jobs = []

    adzuna_result = search_adzuna_jobs(query, location=location)
    if adzuna_result["success"]:
        for job in adzuna_result["jobs"]:
            job["_source"] = "adzuna"
            all_jobs.append(job)

    jooble_result = search_jooble_jobs(query, location=location)
    if jooble_result["success"]:
        for job in jooble_result["jobs"]:
            job["_source"] = "jooble"
            all_jobs.append(job)

    arbeitnow_result = search_arbeitnow_jobs(query)
    if arbeitnow_result["success"]:
        for job in arbeitnow_result["jobs"]:
            job["_source"] = "arbeitnow"
            all_jobs.append(job)

    return all_jobs


def search_jobs_for_candidate(candidate_profile: CandidateProfile, location: str = "") -> dict:
    """
    Full Job Search Agent flow:
    1. Generate smart search queries from the candidate's profile
    2. Search all 3 job sources for each query
    3. Return the combined raw job list + the queries used
    """
    queries = generate_search_queries(candidate_profile)

    all_jobs = []
    for query in queries:
        all_jobs.extend(search_all_sources(query, location=location))

    return {"search_queries": queries, "jobs": all_jobs}
