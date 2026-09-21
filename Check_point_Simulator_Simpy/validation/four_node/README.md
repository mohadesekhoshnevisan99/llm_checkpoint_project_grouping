# Four-node GCE T4 validation

This folder runs a real PyTorch validation pass on Google Compute Engine VMs
with one Tesla T4 GPU per node, then calibrates this repository's simulator
from the collected timings and writes side-by-side HTML traces.

[Back to validation overview](../README.md) ·
[Back to repository overview](../../README.md)

The normal entry point is:

```bash
make validate NODES=4 VALIDATION_ARGS="--project YOUR_PROJECT --zone us-central1-a"
```

Before launching cloud resources, run the local no-cost preflight:

```bash
make validate-preflight NODES=4
```

The simpler cloud path avoids SSH setup entirely by running each rank from the
VM startup script and collecting logs through GCS:

```bash
make validate-startup NODES=4 VALIDATION_ARGS="--project YOUR_PROJECT --zone us-central1-a --iterations 3 --tensor-mb 4 --checkpoint-mb 4"
```

## Folder Structure

| File | Responsibility |
| --- | --- |
| `validate.sh` | Creates and coordinates the SSH-based GCE run, collects results, invokes postprocessing, and cleans up resources created for that run. |
| `validate_startup.sh` | Runs ranks through VM startup scripts and exchanges logs through GCS, avoiding direct SSH setup. |
| `real_training_benchmark.py` | Executes the PyTorch data/pipeline workloads, measures resources and transfers, and writes per-rank JSONL plus summaries. |
| `postprocess_validation.py` | Combines rank traces, infers simulator parameters, runs the calibrated simulation, and builds comparison artifacts. |
| `preflight.py` | Generates synthetic remote logs locally and verifies the complete postprocess/simulate/report pipeline without cloud calls. |
| `README.md` | Operational prerequisites, safety behavior, commands, and artifact reference for this validation target. |

The shell launchers own external-resource lifecycle. The Python benchmark owns
measurement. Postprocessing is the only layer that translates those
measurements into simulator configuration.

Run it from Google Cloud Shell in the repository root. The script creates only
resources tagged and named for the current run, records those names in
`results/validation_four_node/<run-id>/created_instances.txt`, and deletes only
those instances during cleanup.

## What It Creates

- `NODES` Compute Engine instances, default `4`
- one T4 GPU per instance
- a per-run firewall rule allowing traffic only among VMs tagged for that run
- a GCS bucket if you do not pass `--bucket`
- bucket-level object access for the VM service account used by the run
- detailed real PyTorch JSONL traces for four validation jobs: data parallel
  synchronous checkpointing, pipeline parallel synchronous checkpointing, data
  parallel asynchronous checkpointing, and pipeline parallel asynchronous
  checkpointing
- measured per-node GPU memory, CPU memory, CPU core count, GPU-to-CPU
  checkpoint bandwidth, inter-node communication bandwidth, and object-store
  upload bandwidth folded into `simulator_from_real.toml`
- a larger synthetic dense model by default (`hidden-size=8192`, `layers=6`)
  plus a persistent GPU memory reserve targeting 80% of each T4, with a
  1024 MB safety margin
- simulator traces generated from the measured timings
- HTML reports under `results/validation_four_node/<run-id>/`

Auto-created VMs and firewall rules are deleted at the end. An auto-created
bucket is also deleted after results are copied back locally unless you pass
`--keep-bucket`. Existing buckets passed with `--bucket` are never deleted.

The launcher grants `roles/storage.objectAdmin` on the validation bucket to the
VM service account so startup scripts, checkpoint uploads, and log uploads can
work. By default this is the Compute Engine default service account. Override it
with `--service-account EMAIL`, or pass `--skip-bucket-iam` if your
administrator has already granted bucket access.

## Prerequisites

1. Enable Compute Engine and Cloud Storage APIs in the selected GCP project.
2. Make sure the project has enough T4 GPU quota in the target region.
3. Run from Cloud Shell or another machine authenticated with `gcloud`.
4. Use a zone where `nvidia-tesla-t4` is available.
5. Make sure your account can update IAM on the validation bucket, or pass a
   service account that already has object access to the bucket.

Check accelerator availability with:

```bash
gcloud compute accelerator-types list --filter="name=nvidia-tesla-t4"
```

## Common Commands

Default four-node run:

```bash
make validate NODES=4 VALIDATION_ARGS="--project YOUR_PROJECT --zone us-central1-a"
```

Recommended four-node cloud smoke run with VM startup scripts:

```bash
make validate-startup NODES=4 VALIDATION_ARGS="--project YOUR_PROJECT --zone us-central1-a --iterations 3 --tensor-mb 4 --checkpoint-mb 4"
```

Three-iteration four-node validation with VM startup scripts:

```bash
make validate-startup NODES=4 VALIDATION_ARGS="--project YOUR_PROJECT --zone us-central1-a --iterations 3 --warmup-iterations 1 --tensor-mb 16 --checkpoint-mb 256 --checkpoint-every 1 --hidden-size 8192 --layers 6 --gpu-memory-target-percent 80 --startup-timeout 1800"
```

To reduce runtime while debugging launcher behavior, lower `--hidden-size` and
set `--gpu-memory-target-percent 0`. To stress memory harder, increase
`--gpu-memory-target-percent` only if there is enough free headroom; the
benchmark rejects values above 95%.

Use an existing bucket and keep it:

```bash
make validate NODES=4 VALIDATION_ARGS="--project YOUR_PROJECT --zone us-central1-a --bucket gs://YOUR_BUCKET --keep-bucket"
```

Run a smaller, faster smoke validation:

```bash
make validate NODES=4 VALIDATION_ARGS="--project YOUR_PROJECT --zone us-central1-a --iterations 2 --tensor-mb 4 --checkpoint-mb 4"
```

Use IAP tunneling when direct SSH to port 22 is blocked by the project network:

```bash
make validate NODES=4 VALIDATION_ARGS="--project YOUR_PROJECT --zone us-central1-a --iterations 2 --tensor-mb 4 --checkpoint-mb 4 --tunnel-through-iap"
```

With `--tunnel-through-iap`, the launcher creates a per-run TCP 22 firewall
rule from `35.235.240.0/20` to only the VMs tagged for that validation run and
removes the rule during cleanup.

Override the VM image family if your project uses a different GPU PyTorch
image:

```bash
make validate NODES=4 VALIDATION_ARGS="--project YOUR_PROJECT --zone us-central1-a --image-family PYTORCH_GPU_IMAGE_FAMILY --image-project IMAGE_PROJECT"
```

The launcher default is currently
`pytorch-2-9-cu129-ubuntu-2204-nvidia-580` from
`deeplearning-platform-release`.

Keep instances for debugging:

```bash
make validate NODES=4 VALIDATION_ARGS="--project YOUR_PROJECT --zone us-central1-a --keep-instances"
```

If you keep instances, delete them manually after debugging:

```bash
gcloud compute instances delete simval-t4-<run-id>-0 simval-t4-<run-id>-1 simval-t4-<run-id>-2 simval-t4-<run-id>-3 --zone us-central1-a
```

## Outputs

The main output folder is:

```text
results/validation_four_node/<run-id>/
```

Important files:

- `trace_comparison.html`: side-by-side real PyTorch and simulator dashboards
- `trace_comparison_detailed.html`: side-by-side detailed real PyTorch and
  simulator dashboards
- `real_trace.jsonl`: combined real PyTorch trace
- `real_trace_detailed.jsonl`: full combined real PyTorch trace with setup,
  reserve, wait, microbatch, checkpoint, and synchronization events
- `simulated_trace.jsonl`: simulator trace generated from measured timings
- `simulator_from_real.toml`: generated simulator config
- `validation_metrics.json`: wall-time and mean operation summaries
- `idle_analysis.json`: inferred idle/wait gaps between traced events
- `real_pytorch_visualization.html`: standalone real trace dashboard
- `real_pytorch_detailed_visualization.html`: standalone detailed real trace
  dashboard
- `simulator_visualization.html`: standalone simulator dashboard
- `remote/logs/<instance>/startup.log`: stdout/stderr from each VM startup
  job, including `[simval]` JSON progress lines from the PyTorch benchmark
- `remote/logs/<instance>/logs/*_rank_*.jsonl`: per-rank detailed real
  PyTorch event logs before they are combined into `real_trace.jsonl`

## Preflight

`make validate-preflight` does not call `gcloud` and does not create cloud
resources. It writes synthetic rank logs, runs the same postprocessing path,
runs the simulator from generated timings, and verifies the expected HTML and
JSONL outputs exist.

The preflight output is:

```text
results/validation_four_node/preflight/trace_comparison.html
```

## Direct Script Usage

The Make target calls this script:

```bash
validation/four_node/validate.sh --nodes 4 --project YOUR_PROJECT --zone us-central1-a
```

The script accepts `--nodes`, so the same harness can launch a different node
count when quota allows, even though this folder is the four-node validation
baseline.
