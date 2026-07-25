#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JOB_FILE="${PROJECT_DIR}/results/training_monitor.jobid"

if [ ! -s "${JOB_FILE}" ]; then
    echo "Training monitor is not running."
    exit 0
fi

monitor_job="$(cat "${JOB_FILE}")"
if [ -n "$(squeue -h -j "${monitor_job}" 2>/dev/null)" ]; then
    scancel "${monitor_job}"
    echo "Training monitor stopped (Slurm job ${monitor_job})."
else
    echo "Training monitor job ${monitor_job} is no longer running."
fi
rm -f "${JOB_FILE}"
