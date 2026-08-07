#!/usr/bin/env bash
#SBATCH --job-name=ner-seed-eval
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 <seed_config>" >&2
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
python -m src.evaluate.validate_seed_run --config "${CONFIG}"
