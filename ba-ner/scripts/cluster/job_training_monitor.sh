#!/usr/bin/env bash
#SBATCH --job-name=ner-training-monitor
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/cluster"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_ba_ner_env
print_runtime_info

OUTPUT_FILE="${PROJECT_DIR}/results/training_monitor.md"

requeue_monitor() {
    echo "Monitor time limit approaching; requeueing job ${SLURM_JOB_ID}." >&2
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}

trap requeue_monitor USR1

python -m src.evaluate.monitor_training --output "${OUTPUT_FILE}" &
monitor_pid=$!
wait "${monitor_pid}"
