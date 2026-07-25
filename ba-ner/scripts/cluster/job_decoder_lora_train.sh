#!/usr/bin/env bash
#SBATCH --job-name=ner-lora-train
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 <lora_config>" >&2
    exit 2
fi

CONFIG="$1"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/cluster"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_ba_ner_env
print_runtime_info
require_path "${CONFIG}"

if [ -n "${BA_NER_COMPLETION_FILE:-}" ] && [ -f "${BA_NER_COMPLETION_FILE}" ]; then
    echo "Training result already exists: ${BA_NER_COMPLETION_FILE}"
    exit 0
fi

requeue_job() {
    echo "Requeue signal received; training will resume from the latest checkpoint." >&2
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}

if [ "${BA_NER_REQUEUE_ON_SIGNAL:-0}" = "1" ]; then
    trap requeue_job USR1
fi

python -m src.decoder.train "${CONFIG}" &
train_pid=$!
wait "${train_pid}"
