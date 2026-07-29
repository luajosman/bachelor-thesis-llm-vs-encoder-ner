#!/usr/bin/env bash
#SBATCH --job-name=ner-infer-bench
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: sbatch $0 <config> <output.json> [extra benchmark args...]" >&2
    exit 2
fi

CONFIG="$1"
OUTPUT="$2"
shift 2

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

python -m src.decoder.benchmark_inference \
    --config "${CONFIG}" \
    --output "${OUTPUT}" \
    "$@"
