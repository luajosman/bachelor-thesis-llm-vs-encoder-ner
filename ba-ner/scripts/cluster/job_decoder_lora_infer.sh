#!/usr/bin/env bash
#SBATCH --job-name=ner-lora-infer
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 <lora_config> <adapter_dir>" >&2
    exit 2
fi

CONFIG="$1"
ADAPTER_DIR="$2"
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
require_path "${ADAPTER_DIR}"

python -m src.decoder.inference \
    --adapter "${ADAPTER_DIR}" \
    --config "${CONFIG}"
