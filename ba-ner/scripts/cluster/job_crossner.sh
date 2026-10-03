#!/usr/bin/env bash
#SBATCH --output=logs/crossner-%j.out
#SBATCH --error=logs/crossner-%j.err

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/cluster"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/common.sh"
activate_ba_ner_env
print_runtime_info

if [ "${1:-}" = "--aggregate" ]; then
    shift
    python -m src.evaluate.crossner_aggregate "$@"
else
    python -m src.evaluate.crossner "$@"
fi
