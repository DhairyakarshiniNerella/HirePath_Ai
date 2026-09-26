from typing import List
from app.services.matcher import calculate_match_score


def rank_jobs_for_candidate(candidate_profile, analyzed_jobs: List[dict]) -> List[dict]:
    """
    Scores every analyzed job against the candidate profile using our
    deterministic matcher, merges the scores into each job, and returns
    the list sorted best-match-first.
    """
    ranked_jobs = []

    for job in analyzed_jobs:
        score_result = calculate_match_score(candidate_profile, job)
        job.update(score_result)
        ranked_jobs.append(job)

    ranked_jobs.sort(key=lambda j: j["match_score"], reverse=True)
    return ranked_jobs
