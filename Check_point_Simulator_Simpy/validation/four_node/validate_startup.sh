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
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
PREFIX="${PREFIX:-simval-t4}"
REMOTE_DIR="${REMOTE_DIR:-/var/tmp/validation-four-node}"
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
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-3600}"
KEEP_INSTANCES=0
KEEP_BUCKET=0
PROGRESS_TOTAL=10
PROGRESS_STEP=0

usage() {
  cat <<EOF
Usage: $0 [options]

Launch GCE T4 VMs that run the PyTorch validation from startup scripts.
This path does not use SSH/SCP for setup or log collection.

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
  --startup-timeout N       Seconds to wait for VM startup jobs
  --run-id ID               Stable run identifier
  --prefix NAME             Instance name prefix
  --keep-instances          Do not delete VMs at the end
  --keep-bucket             Do not delete an auto-created bucket
  -h, --help                Show this help
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
    --startup-timeout) STARTUP_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --keep-instances) KEEP_INSTANCES=1; shift ;;
    --keep-bucket) KEEP_BUCKET=1; shift ;;
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
  gcloud storage cp --recursive "${source}" "${destination}"
}

count_gcs_matches() {
  local pattern="$1"
  { gcloud storage ls "${pattern}" 2>/dev/null || true; } | wc -l | tr -d ' '
}

progress_next "Checking arguments and project"
if [[ "${NODES}" -lt 2 ]]; then
  echo "--nodes must be at least 2" >&2
  exit 2
fi
require_command gcloud
require_python_module plotly
require_python_module simpy
if [[ -z "${PROJECT}" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "${PROJECT}" ]]; then
  echo "No GCP project is configured. Pass --project or run gcloud config set project." >&2
  exit 2
fi
gcloud config set project "${PROJECT}" >/dev/null
if [[ -z "${SERVICE_ACCOUNT}" ]]; then
  PROJECT_NUMBER="$(gcloud projects describe "${PROJECT}" --format="value(projectNumber)" 2>/dev/null || true)"
  if [[ -z "${PROJECT_NUMBER}" ]]; then
    echo "Could not determine the project number. Pass --service-account EMAIL." >&2
    exit 2
  fi
  SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi

RESULT_DIR="${REPO_ROOT}/results/validation_four_node/${RUN_ID}"
REMOTE_LOG_ROOT="${RESULT_DIR}/remote"
CREATED_INSTANCES_FILE="${RESULT_DIR}/created_instances.txt"
FIREWALL_RULE="${PREFIX}-${RUN_ID}-dist"
TAG="${PREFIX}-${RUN_ID}"
CREATED_BUCKET=0
CREATED_FIREWALL=0

progress_next "Preparing local result directories"
mkdir -p "${RESULT_DIR}" "${REMOTE_LOG_ROOT}" "${RESULT_DIR}/startup-scripts"
: > "${CREATED_INSTANCES_FILE}"

INSTANCE_NAMES=()
for rank in $(seq 0 "$((NODES - 1))"); do
  INSTANCE_NAMES+=("${PREFIX}-${RUN_ID}-${rank}")
done
MASTER_INSTANCE="${INSTANCE_NAMES[0]}"

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
  if [[ "${CREATED_BUCKET}" -eq 1 && "${KEEP_BUCKET}" -eq 0 ]]; then
    echo "Deleting auto-created bucket ${BUCKET_URI}..."
    gcloud storage rm --recursive "${BUCKET_URI}/**" >/dev/null 2>&1 || true
    gcloud storage buckets delete "${BUCKET_URI}" --quiet >/dev/null 2>&1 || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

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

progress_next "Uploading benchmark script"
gcloud storage cp "${SCRIPT_DIR}/real_training_benchmark.py" \
  "${RUN_URI}/scripts/real_training_benchmark.py"

progress_next "Checking accelerator, image, and network firewall"
if ! gcloud compute accelerator-types describe "${ACCELERATOR_TYPE}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" >/dev/null 2>&1; then
  echo "${ACCELERATOR_TYPE} is not advertised in ${ZONE}; choose another zone." >&2
  exit 2
fi
if ! gcloud compute images describe-from-family "${IMAGE_FAMILY}" \
  --project="${IMAGE_PROJECT}" >/dev/null 2>&1; then
  echo "Image family ${IMAGE_FAMILY} was not found in ${IMAGE_PROJECT}." >&2
  exit 2
fi
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

write_startup_script() {
  local rank="$1"
  local instance="$2"
  local path="${RESULT_DIR}/startup-scripts/rank-${rank}.sh"
  cat > "${path}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
exec > >(tee -a /var/log/simval-startup.log | logger -t simval-startup) 2>&1

INSTANCE_NAME="${instance}"
RANK="${rank}"
WORLD_SIZE="${NODES}"
MASTER_ADDR="${MASTER_INSTANCE}.${ZONE}.c.${PROJECT}.internal"
RUN_URI="${RUN_URI}"
REMOTE_DIR="${REMOTE_DIR}"

upload_logs() {
  local status="\$1"
  mkdir -p "\${REMOTE_DIR}/logs"
  printf '%s\n' "\${status}" > "\${REMOTE_DIR}/logs/status.txt" || true
  printf '%s\n' "\${status}" > "/tmp/\${INSTANCE_NAME}.\${status}" || true
  gcloud storage cp /var/log/simval-startup.log "\${RUN_URI}/logs/\${INSTANCE_NAME}/startup.log" || true
  for artifact in "\${REMOTE_DIR}"/logs/*.jsonl "\${REMOTE_DIR}"/logs/*.summary.json "\${REMOTE_DIR}"/logs/progress.txt "\${REMOTE_DIR}"/logs/status.txt; do
    [[ -e "\${artifact}" ]] || continue
    gcloud storage cp "\${artifact}" "\${RUN_URI}/logs/\${INSTANCE_NAME}/logs/\$(basename "\${artifact}")" || true
  done
  gcloud storage cp "/tmp/\${INSTANCE_NAME}.\${status}" "\${RUN_URI}/status/\${INSTANCE_NAME}.\${status}" || true
}

stage() {
  local name="\$1"
  mkdir -p "\${REMOTE_DIR}/logs"
  printf '%s %s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\${name}" >> "\${REMOTE_DIR}/logs/progress.txt"
  gcloud storage cp "\${REMOTE_DIR}/logs/progress.txt" "\${RUN_URI}/logs/\${INSTANCE_NAME}/progress.txt" || true
  gcloud storage cp /var/log/simval-startup.log "\${RUN_URI}/logs/\${INSTANCE_NAME}/startup.log" || true
}

trap 'upload_logs failed' ERR

stage startup-entered
mkdir -p "\${REMOTE_DIR}/logs"
cd "\${REMOTE_DIR}"
gcloud storage cp "\${RUN_URI}/scripts/real_training_benchmark.py" real_training_benchmark.py
stage benchmark-downloaded

stage torch-check-started
python3 - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("torch") is None:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--user",
        "--upgrade",
        "torch",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ])
PY
stage torch-ready

run_mode() {
  local mode="\$1"
  local checkpoint_mode="\$2"
  local port="\$3"
  local label="\${mode}-\${checkpoint_mode}"
  stage "starting-\${label}"
  MASTER_ADDR="\${MASTER_ADDR}" MASTER_PORT="\${port}" WORLD_SIZE="\${WORLD_SIZE}" RANK="\${RANK}" LOCAL_RANK=0 \\
    NCCL_SOCKET_IFNAME=ens5 GLOO_SOCKET_IFNAME=ens5 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO \\
    python3 real_training_benchmark.py \\
      --mode "\${mode}" \\
      --checkpoint-mode "\${checkpoint_mode}" \\
      --iterations "${ITERATIONS}" \\
      --warmup-iterations "${WARMUP_ITERATIONS}" \\
      --microbatches "${MICROBATCHES}" \\
      --tensor-mb "${TENSOR_MB}" \\
      --checkpoint-mb "${CHECKPOINT_MB}" \\
      --checkpoint-every "${CHECKPOINT_EVERY}" \\
      --hidden-size "${HIDDEN_SIZE}" \\
      --layers "${LAYERS}" \\
      --gpu-memory-target-percent "${GPU_MEMORY_TARGET_PERCENT}" \\
      --gpu-memory-reserve-safety-mb "${GPU_MEMORY_RESERVE_SAFETY_MB}" \\
      --bucket-uri "\${RUN_URI}" \\
      --run-id "${RUN_ID}" \\
      --node-name "\${INSTANCE_NAME}" \\
      --output "logs/\${mode}_\${checkpoint_mode}_rank_\${RANK}.jsonl" \\
      --summary-output "logs/\${mode}_\${checkpoint_mode}_rank_\${RANK}.summary.json"
  stage "finished-\${label}"
}

run_mode data_parallel synchronous 29500
run_mode pipeline_parallel synchronous 29501
run_mode data_parallel asynchronous 29502
run_mode pipeline_parallel asynchronous 29503

upload_logs done
EOF
}

progress_next "Creating GPU instances with startup jobs"
echo "Validation run ${RUN_ID}"
echo "  project: ${PROJECT}"
echo "  zone:    ${ZONE}"
echo "  nodes:   ${NODES}"
echo "  bucket:  ${BUCKET_URI}"
echo "  service: ${SERVICE_ACCOUNT}"
echo "  model:   hidden=${HIDDEN_SIZE} layers=${LAYERS} gpu_target=${GPU_MEMORY_TARGET_PERCENT}%"
echo "  results: ${RESULT_DIR}"
create_count=0
create_pids=()
create_instances=()
for rank in $(seq 0 "$((NODES - 1))"); do
  instance="${INSTANCE_NAMES[$rank]}"
  startup_script="${RESULT_DIR}/startup-scripts/rank-${rank}.sh"
  write_startup_script "${rank}" "${instance}"
  create_count=$((create_count + 1))
  progress_item "${create_count}" "${NODES}" "Launching create for ${instance}"
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
    --metadata-from-file=startup-script="${startup_script}" \
    > "${RESULT_DIR}/create-${rank}.log" 2>&1 &
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

progress_next "Waiting for startup jobs to finish"
deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while true; do
  done_count="$(count_gcs_matches "${RUN_URI}/status/*.done")"
  failed_count="$(count_gcs_matches "${RUN_URI}/status/*.failed")"
  printf '  done=%s/%s failed=%s\n' "${done_count}" "${NODES}" "${failed_count}"
  if [[ "${failed_count}" -gt 0 ]]; then
    echo "At least one VM startup job failed; collecting available logs." >&2
    break
  fi
  if [[ "${done_count}" -ge "${NODES}" ]]; then
    break
  fi
  if [[ "${SECONDS}" -ge "${deadline}" ]]; then
    echo "Timed out waiting for startup jobs after ${STARTUP_TIMEOUT_SECONDS}s." >&2
    break
  fi
  sleep 20
done

progress_next "Collecting logs from GCS"
storage_cp_recursive "${RUN_URI}/logs" "${REMOTE_LOG_ROOT}"
if [[ "$(count_gcs_matches "${RUN_URI}/status/*.done")" -lt "${NODES}" ]]; then
  echo "Startup validation did not complete on all nodes. Logs: ${REMOTE_LOG_ROOT}" >&2
  exit 1
fi

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
