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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_ba_ner_env
print_runtime_info
require_path "${CONFIG}"

python -m src.decoder.train "${CONFIG}"
