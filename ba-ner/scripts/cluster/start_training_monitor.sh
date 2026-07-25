#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${BA_NER_SCRATCH:-}" ] && [ -d "/netscratch/${USER}/ba-ner" ]; then
    export BA_NER_SCRATCH="/netscratch/${USER}/ba-ner"
fi

# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

shared_venv="${HOME}/bachelor-thesis-llm-vs-encoder-ner/ba-ner/.venv"
if [ -z "${BA_NER_VENV:-}" ] && [ -x "${shared_venv}/bin/python" ]; then
    export BA_NER_VENV="${shared_venv}"
fi

JOB_FILE="${PROJECT_DIR}/results/training_monitor.jobid"
OUTPUT_FILE="${PROJECT_DIR}/results/training_monitor.md"

if [ -s "${JOB_FILE}" ]; then
    old_job="$(cat "${JOB_FILE}")"
    if [ -n "$(squeue -h -j "${old_job}" 2>/dev/null)" ]; then
        echo "Training monitor is already running (Slurm job ${old_job})."
        echo "View: ${OUTPUT_FILE}"
        exit 0
    fi
fi

mkdir -p "${PROJECT_DIR}/results" "${PROJECT_DIR}/logs"
monitor_job="$(
    sbatch --parsable \
        --export="ALL,BA_NER_VENV=${BA_NER_VENV},BA_NER_SCRATCH=${BA_NER_SCRATCH:-}" \
        "${SCRIPT_DIR}/job_training_monitor.sh"
)"
echo "${monitor_job}" >"${JOB_FILE}"

echo "Training monitor submitted (Slurm job ${monitor_job})."
echo "View: ${OUTPUT_FILE}"
echo "Log: ${PROJECT_DIR}/logs/ner-training-monitor_${monitor_job}.out"
