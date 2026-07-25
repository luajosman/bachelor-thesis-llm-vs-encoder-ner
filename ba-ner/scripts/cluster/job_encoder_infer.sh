#!/usr/bin/env bash
#SBATCH --job-name=ner-enc-infer
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 <encoder_config> <best_model_dir>" >&2
    exit 2
fi

CONFIG="$1"
MODEL_DIR="$2"
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
require_path "${MODEL_DIR}"

python -m src.encoder.inference \
    --model "${MODEL_DIR}" \
    --config "${CONFIG}"
