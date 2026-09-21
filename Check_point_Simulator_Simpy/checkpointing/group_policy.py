"""Group-scoped donor policy helpers."""
from typing import Mapping

def normalize_group_policy(policy: Mapping) -> dict[str, frozenset[str]]:
    jobs = policy.get("jobs", policy)
    groups = {}
    for job_id, spec in jobs.items():
        groups.setdefault(int(spec["group_id"]), set()).add(job_id)

    result = {}
    for job_id, spec in jobs.items():
        group_id = int(spec["group_id"])
        allowed = frozenset(spec.get("allowed_donor_job_ids", []))
        invalid = sorted(
            donor for donor in allowed
            if donor == job_id or donor not in groups[group_id]
        )
        if invalid:
            raise ValueError(f"Invalid donors for {job_id}: {invalid}")
        result[job_id] = allowed
    return result
