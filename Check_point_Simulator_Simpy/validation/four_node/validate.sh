#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

NODES="${NODES:-4}"
PROJECT="${PROJECT:-}"
ZONE="${ZONE:-us-central1-a}"
BUCKET="${BUCKET:-}"
BUCKET_LOCATION="${BUCKET_LOCATION:-US}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-nvidia-tesla-t4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
NETWORK="${NETWORK:-default}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
SKIP_BUCKET_IAM="${SKIP_BUCKET_IAM:-0}"
SSH_TUNNEL_THROUGH_IAP="${SSH_TUNNEL_THROUGH_IAP:-0}"
SSH_SOURCE_RANGE="${SSH_SOURCE_RANGE:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
PREFIX="${PREFIX:-simval-t4}"
REMOTE_DIR="${REMOTE_DIR:-/home/$USER/validation-four-node}"
ITERATIONS="${ITERATIONS:-3}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-1}"
MICROBATCHES="${MICROBATCHES:-4}"
TENSOR_MB="${TENSOR_MB:-16}"
CHECKPOINT_MB="${CHECKPOINT_MB:-256}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
HIDDEN_SIZE="${HIDDEN_SIZE:-8192}"
LAYERS="${LAYERS:-6}"
GPU_MEMORY_TARGET_PERCENT="${GPU_MEMORY_TARGET_PERCENT:-80}"
GPU_MEMORY_RESERVE_SAFETY_MB="${GPU_MEMORY_RESERVE_SAFETY_MB:-1024}"
KEEP_INSTANCES=0
KEEP_BUCKET=0
SKIP_PROVISION=0
PROGRESS_TOTAL=15
PROGRESS_STEP=0

usage() {
  cat <<EOF
Usage: $0 [options]

Launch GCE Tesla T4 VMs, run real PyTorch data/pipeline parallel benchmarks,
collect traces, run the simulator from measured timings, and build comparison
HTML in results/validation_four_node/<run-id>.

Options:
  --nodes N                 VM/GPU count (default: ${NODES})
  --project PROJECT         GCP project (default: active gcloud project)
  --zone ZONE               Compute zone (default: ${ZONE})
  --bucket gs://NAME        Existing bucket or bucket to create
  --bucket-location LOC     Location for auto-created bucket (default: US)
  --machine-type TYPE       GCE machine type (default: ${MACHINE_TYPE})
  --accelerator-type TYPE   GPU accelerator type (default: ${ACCELERATOR_TYPE})
  --image-family FAMILY     VM image family (default: ${IMAGE_FAMILY})
  --image-project PROJECT   VM image project (default: ${IMAGE_PROJECT})
  --network NETWORK         VPC network (default: ${NETWORK})
  --service-account EMAIL   VM service account (default: Compute Engine default)
  --skip-bucket-iam         Do not grant VM service account access to the bucket
  --tunnel-through-iap      Use IAP tunneling for SSH/SCP
  --ssh-source-range CIDR   Create a per-run SSH firewall rule for this CIDR
  --iterations N            Measured iterations (default: ${ITERATIONS})
  --warmup-iterations N     Warmup iterations before tracing
  --microbatches N          Pipeline microbatches
  --tensor-mb MB            Synthetic activation tensor size per microbatch
  --checkpoint-mb MB        Synthetic checkpoint shard size
  --checkpoint-every N      0 disables checkpoint upload timing
  --hidden-size N           Width of synthetic dense layers
  --layers N                Number of hidden Linear/ReLU blocks
  --gpu-memory-target-percent P
                            Reserve GPU memory up to this percent after setup
  --gpu-memory-reserve-safety-mb MB
                            Free-memory safety margin for GPU reserve
  --run-id ID               Stable run identifier
  --prefix NAME             Instance name prefix
  --keep-instances          Do not delete VMs at the end
  --keep-bucket             Do not delete an auto-created bucket
  --skip-provision          Use already-created instances named by this run
  -h, --help                Show this help

Environment variables with the same names can also be used.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) NODES="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --bucket-location) BUCKET_LOCATION="$2"; shift 2 ;;
    --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
    --accelerator-type) ACCELERATOR_TYPE="$2"; shift 2 ;;
    --image-family) IMAGE_FAMILY="$2"; shift 2 ;;
    --image-project) IMAGE_PROJECT="$2"; shift 2 ;;
    --network) NETWORK="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --skip-bucket-iam) SKIP_BUCKET_IAM=1; shift ;;
    --tunnel-through-iap) SSH_TUNNEL_THROUGH_IAP=1; shift ;;
    --ssh-source-range) SSH_SOURCE_RANGE="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --warmup-iterations) WARMUP_ITERATIONS="$2"; shift 2 ;;
    --microbatches) MICROBATCHES="$2"; shift 2 ;;
    --tensor-mb) TENSOR_MB="$2"; shift 2 ;;
    --checkpoint-mb) CHECKPOINT_MB="$2"; shift 2 ;;
    --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    --hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --gpu-memory-target-percent) GPU_MEMORY_TARGET_PERCENT="$2"; shift 2 ;;
    --gpu-memory-reserve-safety-mb) GPU_MEMORY_RESERVE_SAFETY_MB="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --keep-instances) KEEP_INSTANCES=1; shift ;;
    --keep-bucket) KEEP_BUCKET=1; shift ;;
    --skip-provision) SKIP_PROVISION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

progress_bar() {
  local message="$1"
  local width=30
  local filled=$((PROGRESS_STEP * width / PROGRESS_TOTAL))
  local empty=$((width - filled))
  local filled_bar
  local empty_bar
  filled_bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')"
  empty_bar="$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  printf '[%s%s] %2d/%2d %s\n' \
    "${filled_bar}" "${empty_bar}" "${PROGRESS_STEP}" "${PROGRESS_TOTAL}" "${message}"
}

progress_next() {
  PROGRESS_STEP=$((PROGRESS_STEP + 1))
  progress_bar "$1"
}

progress_item() {
  local current="$1"
  local total="$2"
  local message="$3"
  printf '  (%s/%s) %s\n' "${current}" "${total}" "${message}"
}

progress_next "Checking arguments and project"
if [[ "${NODES}" -lt 2 ]]; then
  echo "--nodes must be at least 2" >&2
  exit 2
fi

if [[ -z "${PROJECT}" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "${PROJECT}" ]]; then
  echo "No GCP project is configured. Pass --project or run gcloud config set project." >&2
  exit 2
fi

RESULT_DIR="${REPO_ROOT}/results/validation_four_node/${RUN_ID}"
REMOTE_LOG_ROOT="${RESULT_DIR}/remote"
CREATED_INSTANCES_FILE="${RESULT_DIR}/created_instances.txt"
FIREWALL_RULE="${PREFIX}-${RUN_ID}-dist"
SSH_FIREWALL_RULE="${PREFIX}-${RUN_ID}-ssh"
TAG="${PREFIX}-${RUN_ID}"

progress_next "Preparing local result directories"
mkdir -p "${RESULT_DIR}" "${REMOTE_LOG_ROOT}"
: > "${CREATED_INSTANCES_FILE}"

CREATED_BUCKET=0
CREATED_FIREWALL=0
CREATED_SSH_FIREWALL=0
INSTANCE_NAMES=()
for rank in $(seq 0 "$((NODES - 1))"); do
  INSTANCE_NAMES+=("${PREFIX}-${RUN_ID}-${rank}")
done

cleanup() {
  local exit_code=$?
  set +e
  if [[ "${KEEP_INSTANCES}" -eq 0 && -s "${CREATED_INSTANCES_FILE}" ]]; then
    echo "Cleaning up instances created by this run..."
    local delete_pids=()
    while IFS= read -r instance; do
      [[ -z "${instance}" ]] && continue
      gcloud compute instances delete "${instance}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --quiet >/dev/null 2>&1 &
      delete_pids+=("$!")
    done < "${CREATED_INSTANCES_FILE}"
    for pid in "${delete_pids[@]}"; do
      wait "${pid}" || true
    done
  fi
  if [[ "${CREATED_FIREWALL}" -eq 1 ]]; then
    gcloud compute firewall-rules delete "${FIREWALL_RULE}" \
      --project="${PROJECT}" \
      --quiet >/dev/null 2>&1 || true
  fi
  if [[ "${CREATED_SSH_FIREWALL}" -eq 1 ]]; then
    gcloud compute firewall-rules delete "${SSH_FIREWALL_RULE}" \
      --project="${PROJECT}" \
      --quiet >/dev/null 2>&1 || true
  fi
  if [[ "${CREATED_BUCKET}" -eq 1 && "${KEEP_BUCKET}" -eq 0 ]]; then
    echo "Deleting auto-created bucket ${BUCKET_URI}..."
    if command -v gsutil >/dev/null 2>&1; then
      gsutil -m rm -r "${BUCKET_URI}" >/dev/null 2>&1 || true
    else
      gcloud storage rm --recursive "${BUCKET_URI}/**" >/dev/null 2>&1 || true
      gcloud storage buckets delete "${BUCKET_URI}" --quiet >/dev/null 2>&1 || true
    fi
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 2
  fi
}

require_python_module() {
  local module="$1"
  if ! "${PYTHON_BIN}" -c "import ${module}" >/dev/null 2>&1; then
    echo "Python dependency missing for ${PYTHON_BIN}: ${module}" >&2
    echo "Run 'make install' in the repository root before launching validation." >&2
    exit 2
  fi
}

storage_cp_recursive() {
  local source="$1"
  local destination="$2"
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp --recursive "${source}" "${destination}"
  else
    gsutil -m cp -r "${source}" "${destination}"
  fi
}

gcloud_ssh_transport_args() {
  if [[ "${SSH_TUNNEL_THROUGH_IAP}" -eq 1 ]]; then
    printf '%s\n' "--tunnel-through-iap"
  fi
}

ssh_cmd() {
  local instance="$1"
  local command="$2"
  local transport_args=()
  while IFS= read -r arg; do
    [[ -n "${arg}" ]] && transport_args+=("${arg}")
  done < <(gcloud_ssh_transport_args)
  gcloud compute ssh "${instance}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --quiet \
    "${transport_args[@]}" \
    --ssh-flag="-o BatchMode=yes" \
    --ssh-flag="-o ConnectTimeout=10" \
    --ssh-flag="-o ServerAliveInterval=60" \
    --command="${command}"
}

scp_to_instance() {
  local source="$1"
  local instance="$2"
  local destination="$3"
  local transport_args=()
  while IFS= read -r arg; do
    [[ -n "${arg}" ]] && transport_args+=("${arg}")
  done < <(gcloud_ssh_transport_args)
  gcloud compute scp "${source}" "${instance}:${destination}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    "${transport_args[@]}" \
    --quiet
}

scp_from_instance() {
  local instance="$1"
  local source="$2"
  local destination="$3"
  local transport_args=()
  while IFS= read -r arg; do
    [[ -n "${arg}" ]] && transport_args+=("${arg}")
  done < <(gcloud_ssh_transport_args)
  mkdir -p "${destination}"
  gcloud compute scp --recurse "${instance}:${source}" "${destination}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    "${transport_args[@]}" \
    --quiet
}

require_command gcloud
require_python_module plotly
require_python_module simpy
gcloud config set project "${PROJECT}" >/dev/null
if [[ -z "${SERVICE_ACCOUNT}" ]]; then
  PROJECT_NUMBER="$(gcloud projects describe "${PROJECT}" --format="value(projectNumber)" 2>/dev/null || true)"
  if [[ -z "${PROJECT_NUMBER}" ]]; then
    echo "Could not determine the project number. Pass --service-account EMAIL." >&2
    exit 2
  fi
  SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi

progress_next "Preparing GCS bucket"
if [[ -z "${BUCKET}" ]]; then
  SAFE_PROJECT="$(printf '%s' "${PROJECT}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-.' '-')"
  BUCKET="gs://${SAFE_PROJECT}-${PREFIX}-${RUN_ID}"
fi
if [[ "${BUCKET}" != gs://* ]]; then
  BUCKET="gs://${BUCKET}"
fi
BUCKET_URI="${BUCKET%/}"
RUN_URI="${BUCKET_URI}/validation-four-node/${RUN_ID}"

if ! gcloud storage buckets describe "${BUCKET_URI}" >/dev/null 2>&1; then
  gcloud storage buckets create "${BUCKET_URI}" \
    --project="${PROJECT}" \
    --location="${BUCKET_LOCATION}" \
    --uniform-bucket-level-access
  CREATED_BUCKET=1
fi
if [[ "${SKIP_BUCKET_IAM}" -eq 0 ]]; then
  echo "Granting ${SERVICE_ACCOUNT} access to ${BUCKET_URI}..."
  if ! gcloud storage buckets add-iam-policy-binding "${BUCKET_URI}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/storage.objectAdmin" >/dev/null; then
    echo "Could not grant bucket access to ${SERVICE_ACCOUNT}." >&2
    echo "Grant it manually or pass a service account that can access ${BUCKET_URI}." >&2
    exit 2
  fi
fi

progress_next "Checking accelerator and VM image"
if ! gcloud compute accelerator-types describe "${ACCELERATOR_TYPE}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" >/dev/null 2>&1; then
  echo "${ACCELERATOR_TYPE} is not advertised in ${ZONE}; choose another zone." >&2
  exit 2
fi

if ! gcloud compute images describe-from-family "${IMAGE_FAMILY}" \
  --project="${IMAGE_PROJECT}" >/dev/null 2>&1; then
  echo "Image family ${IMAGE_FAMILY} was not found in ${IMAGE_PROJECT}." >&2
  echo "Pass --image-family and --image-project for a GPU PyTorch image." >&2
  exit 2
fi

progress_next "Configuring per-run firewall"
if ! gcloud compute firewall-rules describe "${FIREWALL_RULE}" \
  --project="${PROJECT}" >/dev/null 2>&1; then
  gcloud compute firewall-rules create "${FIREWALL_RULE}" \
    --project="${PROJECT}" \
    --network="${NETWORK}" \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=all \
    --source-tags="${TAG}" \
    --target-tags="${TAG}"
  CREATED_FIREWALL=1
fi

if [[ "${SSH_TUNNEL_THROUGH_IAP}" -eq 1 && -z "${SSH_SOURCE_RANGE}" ]]; then
  SSH_SOURCE_RANGE="35.235.240.0/20"
fi
if [[ -n "${SSH_SOURCE_RANGE}" ]]; then
  if ! gcloud compute firewall-rules describe "${SSH_FIREWALL_RULE}" \
    --project="${PROJECT}" >/dev/null 2>&1; then
    gcloud compute firewall-rules create "${SSH_FIREWALL_RULE}" \
      --project="${PROJECT}" \
      --network="${NETWORK}" \
      --direction=INGRESS \
      --action=ALLOW \
      --rules=tcp:22 \
      --source-ranges="${SSH_SOURCE_RANGE}" \
      --target-tags="${TAG}"
    CREATED_SSH_FIREWALL=1
  fi
fi

echo "Validation run ${RUN_ID}"
echo "  project: ${PROJECT}"
echo "  zone:    ${ZONE}"
echo "  nodes:   ${NODES}"
echo "  bucket:  ${BUCKET_URI}"
echo "  service: ${SERVICE_ACCOUNT}"
echo "  model:   hidden=${HIDDEN_SIZE} layers=${LAYERS} gpu_target=${GPU_MEMORY_TARGET_PERCENT}%"
echo "  results: ${RESULT_DIR}"
echo "  ssh:     $([[ "${SSH_TUNNEL_THROUGH_IAP}" -eq 1 ]] && echo IAP || echo external)"

if [[ "${SKIP_PROVISION}" -eq 0 ]]; then
  progress_next "Creating GPU instances"
  created_count=0
  create_pids=()
  create_instances=()
  for instance in "${INSTANCE_NAMES[@]}"; do
    created_count=$((created_count + 1))
    progress_item "${created_count}" "${NODES}" "Launching create for ${instance}"
    echo "${instance}" >> "${CREATED_INSTANCES_FILE}"
    gcloud compute instances create "${instance}" \
      --project="${PROJECT}" \
      --zone="${ZONE}" \
      --machine-type="${MACHINE_TYPE}" \
      --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
      --maintenance-policy=TERMINATE \
      --restart-on-failure \
      --image-family="${IMAGE_FAMILY}" \
      --image-project="${IMAGE_PROJECT}" \
      --boot-disk-size="${BOOT_DISK_SIZE}" \
      --service-account="${SERVICE_ACCOUNT}" \
      --scopes=cloud-platform \
      --network="${NETWORK}" \
      --tags="${TAG}" \
      --metadata=install-nvidia-driver=True,validation-run-id="${RUN_ID}" \
      > "${RESULT_DIR}/create-$((created_count - 1)).log" 2>&1 &
    create_pids+=("$!")
    create_instances+=("${instance}")
  done
  create_failed=0
  for index in "${!create_pids[@]}"; do
    instance="${create_instances[$index]}"
    if wait "${create_pids[$index]}"; then
      progress_item "$((index + 1))" "${NODES}" "Created ${instance}"
    else
      create_failed=1
      echo "Failed to create ${instance}; see ${RESULT_DIR}/create-${index}.log" >&2
      tail -80 "${RESULT_DIR}/create-${index}.log" >&2 || true
    fi
  done
  if [[ "${create_failed}" -ne 0 ]]; then
    exit 1
  fi
else
  progress_next "Using existing instances"
  echo "Using existing instances; they will not be deleted by cleanup."
fi

progress_next "Waiting for SSH"
ssh_count=0
for instance in "${INSTANCE_NAMES[@]}"; do
  ssh_count=$((ssh_count + 1))
  progress_item "${ssh_count}" "${NODES}" "Waiting for ${instance}"
  for attempt in $(seq 1 30); do
    if ssh_cmd "${instance}" "echo ready" >/dev/null 2>&1; then
      break
    fi
    if [[ "${attempt}" -eq 30 ]]; then
      echo "SSH did not become ready on ${instance}" >&2
      exit 1
    fi
    sleep 10
  done
done

progress_next "Bootstrapping PyTorch benchmark"
bootstrap_count=0
for instance in "${INSTANCE_NAMES[@]}"; do
  bootstrap_count=$((bootstrap_count + 1))
  progress_item "${bootstrap_count}" "${NODES}" "Preparing ${instance}"
  ssh_cmd "${instance}" "mkdir -p '${REMOTE_DIR}/logs'"
  scp_to_instance "${SCRIPT_DIR}/real_training_benchmark.py" "${instance}" "${REMOTE_DIR}/real_training_benchmark.py"
  ssh_cmd "${instance}" "cd '${REMOTE_DIR}' && python3 - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec('torch') is None:
    subprocess.check_call([
        sys.executable,
        '-m',
        'pip',
        'install',
        '--user',
        '--upgrade',
        'torch',
        '--index-url',
        'https://download.pytorch.org/whl/cu121',
    ])
PY"
done

MASTER_INSTANCE="${INSTANCE_NAMES[0]}"
MASTER_IP="$(gcloud compute instances describe "${MASTER_INSTANCE}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --format='get(networkInterfaces[0].networkIP)')"

run_mode() {
  local mode="$1"
  local checkpoint_mode="$2"
  local port="$3"
  local label="${mode}-${checkpoint_mode}"
  local pids=()
  progress_next "Running ${label}"
  for rank in $(seq 0 "$((NODES - 1))"); do
    local instance="${INSTANCE_NAMES[$rank]}"
    progress_item "$((rank + 1))" "${NODES}" "Starting rank ${rank} on ${instance}"
    local ssh_log="${RESULT_DIR}/${mode}_${checkpoint_mode}_rank_${rank}.ssh.log"
    local remote_command
    remote_command="cd '${REMOTE_DIR}' && NCCL_SOCKET_IFNAME='ens5' GLOO_SOCKET_IFNAME='ens5' NCCL_IB_DISABLE='1' NCCL_DEBUG='INFO' MASTER_ADDR='${MASTER_IP}' MASTER_PORT='${port}' WORLD_SIZE='${NODES}' RANK='${rank}' LOCAL_RANK='0' python3 real_training_benchmark.py --mode '${mode}' --checkpoint-mode '${checkpoint_mode}' --iterations '${ITERATIONS}' --warmup-iterations '${WARMUP_ITERATIONS}' --microbatches '${MICROBATCHES}' --tensor-mb '${TENSOR_MB}' --checkpoint-mb '${CHECKPOINT_MB}' --checkpoint-every '${CHECKPOINT_EVERY}' --hidden-size '${HIDDEN_SIZE}' --layers '${LAYERS}' --gpu-memory-target-percent '${GPU_MEMORY_TARGET_PERCENT}' --gpu-memory-reserve-safety-mb '${GPU_MEMORY_RESERVE_SAFETY_MB}' --bucket-uri '${RUN_URI}' --run-id '${RUN_ID}' --node-name '${instance}' --output 'logs/${mode}_${checkpoint_mode}_rank_${rank}.jsonl' --summary-output 'logs/${mode}_${checkpoint_mode}_rank_${rank}.summary.json'"
    ssh_cmd "${instance}" "${remote_command}" > "${ssh_log}" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "${label} failed. See ${RESULT_DIR}/${mode}_${checkpoint_mode}_rank_*.ssh.log" >&2
    exit 1
  fi
}

run_mode "data_parallel" "synchronous" 29500
run_mode "pipeline_parallel" "synchronous" 29501
run_mode "data_parallel" "asynchronous" 29502
run_mode "pipeline_parallel" "asynchronous" 29503

progress_next "Collecting remote logs"
collect_count=0
for instance in "${INSTANCE_NAMES[@]}"; do
  collect_count=$((collect_count + 1))
  progress_item "${collect_count}" "${NODES}" "Copying logs from ${instance}"
  ssh_cmd "${instance}" "rm -rf '${REMOTE_DIR}/log-export' && mkdir -p '${REMOTE_DIR}/log-export/logs' && find '${REMOTE_DIR}/logs' -maxdepth 1 -type f \\( -name '*.jsonl' -o -name '*.summary.json' -o -name 'progress.txt' -o -name 'status.txt' \\) -exec cp {} '${REMOTE_DIR}/log-export/logs/' \\;"
  scp_from_instance "${instance}" "${REMOTE_DIR}/log-export/logs" "${REMOTE_LOG_ROOT}/${instance}"
done

progress_next "Building simulator and HTML reports"
"${PYTHON_BIN}" "${SCRIPT_DIR}/postprocess_validation.py" \
  --real-root "${REMOTE_LOG_ROOT}" \
  --output-dir "${RESULT_DIR}" \
  --repo-root "${REPO_ROOT}" \
  --nodes "${NODES}" \
  --iterations "${ITERATIONS}" \
  --microbatches "${MICROBATCHES}" \
  --run-simulator

progress_next "Uploading result bundle"
storage_cp_recursive "${RESULT_DIR}" "${RUN_URI}/collected-results"

echo "Validation complete."
echo "  comparison HTML: ${RESULT_DIR}/trace_comparison.html"
echo "  detailed HTML:   ${RESULT_DIR}/trace_comparison_detailed.html"
echo "  real trace:      ${RESULT_DIR}/real_trace.jsonl"
echo "  detailed trace:  ${RESULT_DIR}/real_trace_detailed.jsonl"
echo "  simulated trace: ${RESULT_DIR}/simulated_trace.jsonl"
