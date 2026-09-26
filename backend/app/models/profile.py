from typing import List, Optional
from pydantic import BaseModel, Field


class ProjectEntry(BaseModel):
    """
    One project from the candidate's resume, split into a short name and a
    separate description - so the frontend can highlight the name distinctly
    instead of guessing where it ends inside a paragraph of text.
    """

    name: str = Field(description="Short project title, e.g. 'HirePath AI' or 'Resume Parser Tool'")
    description: str = Field(
        default="",
        description="1-2 sentence description of what the project does or did"
    )


class CandidateProfile(BaseModel):
    """
    Structured information extracted from a candidate's resume.
    The Resume Analyzer Agent will fill this in using the Groq LLM.
    """

    name: Optional[str] = Field(default=None, description="Candidate's full name")
    email: Optional[str] = Field(default=None, description="Candidate's email address")

    education: List[str] = Field(
        default_factory=list,
        description="Degrees, institutions, and graduation years"
    )

    skills: List[str] = Field(
        default_factory=list,
        description="Technical and soft skills, e.g. Python, SQL, Communication"
    )

    projects: List[ProjectEntry] = Field(
        default_factory=list,
        description="Notable projects, each with a short name and description"
    )

    internships: List[str] = Field(
        default_factory=list,
        description="Internship roles and companies"
    )

    experience: List[str] = Field(
        default_factory=list,
        description="Work experience entries (role, company, duration, responsibilities)"
    )

    total_experience_years: float = Field(
        default=0.0,
        description="Total years of professional experience. 0 for freshers."
    )

    job_titles: List[str] = Field(
        default_factory=list,
        description="Previous job titles held"
    )

    companies: List[str] = Field(
        default_factory=list,
        description="Companies the candidate has worked at"
    )

    technologies: List[str] = Field(
        default_factory=list,
        description="Specific tools/technologies used, e.g. Docker, AWS, React"
    )

    career_level: str = Field(
        default="Unknown",
        description="e.g. Fresher, Entry Level, Mid Level, Senior"
    )

    target_roles: List[str] = Field(
        default_factory=list,
        description="Roles the candidate seems suited for or is targeting"
    )
