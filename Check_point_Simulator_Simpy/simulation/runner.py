from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import simpy

from jobs import SimulatorConfig, load_config, run_configured_jobs
from nodes.node import EventLogger

EVENT_LOG_NAME = "simulation_log.jsonl"
CONFIG_NAME = "simulation_config.json"


@dataclass(frozen=True, slots=True)
class SimulationResult:
    config: SimulatorConfig
    config_path: Path
    run_id: int
    simulated_time: float
    placements: list[dict[str, Any]]
    event_log_path: Path | None
    config_output_path: Path | None
    logger: EventLogger = field(repr=False)

    @property
    def event_count(self) -> int:
        return len(self.logger.events)

    @property
    def logical_ranks(self) -> int:
        return sum(job.rank_count for job in self.config.jobs)

    @property
    def physical_nodes(self) -> int:
        return self.config.required_nodes


def run_simulation(
    config_path: str | Path,
    *,
    results_dir: str | Path | None = "results",
    run_id: int = 0,
    quiet: bool = True,
    write_artifacts: bool = True,
) -> SimulationResult:
    config_file = Path(config_path)
    config = load_config(config_file)

    env = simpy.Environment()
    logger = EventLogger()
    completion, placements = run_configured_jobs(
        env,
        run_id=run_id,
        config=config,
        logger=logger,
        verbose=not quiet,
    )
    env.run(until=completion)

    event_log_path: Path | None = None
    config_output_path: Path | None = None
    if write_artifacts:
        if results_dir is None:
            raise ValueError("results_dir is required when write_artifacts=True")
        output_dir = Path(results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        event_log_path = output_dir / EVENT_LOG_NAME
        config_output_path = output_dir / CONFIG_NAME

        logger.write_jsonl(event_log_path)
        resolved_config = config.to_dict()
        resolved_config["config_path"] = str(config_file)
        resolved_config["placements"] = placements
        config_output_path.write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return SimulationResult(
        config=config,
        config_path=config_file,
        run_id=run_id,
        simulated_time=env.now,
        placements=placements,
        event_log_path=event_log_path,
        config_output_path=config_output_path,
        logger=logger,
    )


simulator = run_simulation
run = run_simulation

__all__ = ["SimulationResult", "run", "run_simulation", "simulator"]
