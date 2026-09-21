"""Create a deliberately small peer-group contract for integration testing.

This does not claim runtime integration. It rewrites an existing contract so
all selected jobs share one group, keeps the first job on direct storage, and
routes the remaining jobs through that job as their allowed donor.
"""
import argparse
import json
from pathlib import Path


def make_peer_contract(contract, jobs):
    out = json.loads(json.dumps(contract))
    missing = [j for j in jobs if j not in out["jobs"]]
    if missing:
        raise ValueError(f"Unknown jobs: {missing}")
    if len(jobs) < 2:
        raise ValueError("Peer smoke contract needs at least two jobs")

    group_id = min(out["jobs"][j]["group_id"] for j in jobs)
    donor = jobs[0]
    selected = set(jobs)
    for jid in jobs:
        p = out["jobs"][jid]
        p["group_id"] = group_id
        if jid == donor:
            p["route"] = "direct"
            p["allowed_donor_job_ids"] = []
            p["donor_gb_per_version"] = {}
        else:
            p["route"] = "peer"
            p["allowed_donor_job_ids"] = [donor]
            allocation = p.get("donor_gb_per_version", {})
            amount = sum(allocation.values()) or 1.0
            p["donor_gb_per_version"] = {donor: amount}

    out["integration_ready"] = False
    out["warning"] = (
        "Peer smoke contract only. Runtime must enforce group-scoped donor "
        "reservation before this file is used by the simulator."
    )
    out["peer_smoke_jobs"] = sorted(selected)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--jobs", nargs="+", required=True)
    args = ap.parse_args()
    contract = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = make_peer_contract(contract, args.jobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote peer smoke contract: {args.output}")


if __name__ == "__main__":
    main()
