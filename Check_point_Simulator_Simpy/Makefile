ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
SYSTEM_PYTHON ?= python
else
VENV_PYTHON := .venv/bin/python
SYSTEM_PYTHON ?= python3
endif
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(SYSTEM_PYTHON))
INSTALL_PYTHON ?= $(VENV_PYTHON)

LOG_FILE ?= results/simulation_log.jsonl
HTML_FILE ?= results/simulation_visualization.html
AGGREGATE_LOG ?= $(LOG_FILE)
AGGREGATE_HTML ?= results/simulation_aggregate.html
AGGREGATE_BUCKETS ?= 240
AGGREGATE_TOP_SPANS ?= 500
CONFIG ?= configs/jobs.toml
SCENARIO ?= scenarios/realnode26.yaml
ARM ?= ours_full
SEED ?= 7
SCENARIO_WORKERS ?= 1
PROGRESS_INTERVAL ?= 10
EVENT_BATCH_SIZE ?= 50000
TRACE_DIR ?= results/traces
SCENARIO_OUT ?= results/scenario_$(basename $(notdir $(SCENARIO)))_$(ARM)_s$(SEED).json
SCENARIO_TRACE ?= $(TRACE_DIR)/$(basename $(notdir $(SCENARIO)))_$(ARM)_s$(SEED).jsonl.gz
SCENARIO_AGGREGATE_HTML ?= results/$(basename $(notdir $(SCENARIO)))_$(ARM)_s$(SEED)_aggregate.html
WORLD_SIZE ?=
MODE ?= auto
NODES ?= 4
VALIDATION_ARGS ?=
VALIDATION_PREFLIGHT_ARGS ?=
VALIDATE_SCRIPT ?= validation/four_node/validate.sh
VALIDATE_STARTUP_SCRIPT ?= validation/four_node/validate_startup.sh

RUN_ARGS := --config $(CONFIG)

.PHONY: venv install run run-legacy run-scenario visualise visualize visualise-aggregate visualize-aggregate visualise-context visualize-context visualise-scenario-aggregate visualize-scenario-aggregate run-and-visualise-scenario-aggregate run-and-visualize-scenario-aggregate validate validate-startup validate-preflight test lint check clean

venv:
	$(SYSTEM_PYTHON) -c "from pathlib import Path; import subprocess, sys; venv = Path('.venv'); python = venv / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python'); subprocess.check_call([sys.executable, '-m', 'venv', str(venv)]) if not python.exists() else None"

install: venv
	$(INSTALL_PYTHON) -m pip install --upgrade pip
	$(INSTALL_PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) main.py $(RUN_ARGS)

run-legacy:
	$(PYTHON) main.py --world-size $(if $(WORLD_SIZE),$(WORLD_SIZE),4) --mode $(MODE)

run-scenario:
	$(PYTHON) run_scenario.py --scenario $(SCENARIO) --arm $(ARM) --seed $(SEED) --workers $(SCENARIO_WORKERS) --progress-interval $(PROGRESS_INTERVAL) --event-batch-size $(EVENT_BATCH_SIZE) --out $(SCENARIO_OUT) --trace-dir $(TRACE_DIR)

visualise:
	$(PYTHON) visualise.py --log $(LOG_FILE) --output $(HTML_FILE)

visualize: visualise

visualise-aggregate:
	$(PYTHON) visualise_aggregate.py --log $(AGGREGATE_LOG) --output $(AGGREGATE_HTML) --buckets $(AGGREGATE_BUCKETS) --top-spans $(AGGREGATE_TOP_SPANS)

visualize-aggregate visualise-context visualize-context: visualise-aggregate

visualise-scenario-aggregate:
	$(PYTHON) visualise_aggregate.py --log $(SCENARIO_TRACE) --output $(SCENARIO_AGGREGATE_HTML) --buckets $(AGGREGATE_BUCKETS) --top-spans $(AGGREGATE_TOP_SPANS) --scenario $(SCENARIO) --result $(SCENARIO_OUT) --arm $(ARM) --seed $(SEED)

visualize-scenario-aggregate: visualise-scenario-aggregate

run-and-visualise-scenario-aggregate: run-scenario visualise-scenario-aggregate

run-and-visualize-scenario-aggregate: run-and-visualise-scenario-aggregate

validate:
	bash $(VALIDATE_SCRIPT) --nodes $(NODES) $(VALIDATION_ARGS)

validate-startup:
	bash $(VALIDATE_STARTUP_SCRIPT) --nodes $(NODES) $(VALIDATION_ARGS)

validate-preflight:
	$(PYTHON) validation/four_node/preflight.py --nodes $(NODES) $(VALIDATION_PREFLIGHT_ARGS)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

check: lint test

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; shutil.rmtree(Path('results'), ignore_errors=True)"
