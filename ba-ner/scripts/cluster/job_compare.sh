#!/usr/bin/env bash
#SBATCH --job-name=ner-compare
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

RESULTS_DIR="${1:-results}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_ba_ner_env
print_runtime_info

python -m src.evaluate.compare_all --results-dir "${RESULTS_DIR}"
