#!/usr/bin/env bash
#SBATCH --job-name=ner-prepare-data
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

python - <<'PY'
from src.data.dataset_loader import load_ner_dataset

dataset, info = load_ner_dataset()
print(f"Dataset ready: {info.name} ({info.language})")
for split, samples in dataset.items():
    print(f"{split}: {len(samples)}")
PY
